#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm
from models.linear_emb_aug import LinearEmbAug
from collections import defaultdict
from torchvision import transforms
import fire
from torchinfo import summary
from datasets import load_dataset, Dataset
import bitsandbytes as bnb
import wandb
from utils import get_model_gradient_norm
from torch.amp.autocast_mode import autocast

size = 512
to_tensor = transforms.ToTensor()
image_transforms = transforms.Compose(
    [
        transforms.RandomCrop(size, pad_if_needed=True, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def collate_fn(batch):
    processed_batch = {}
    for label in ("img_better", "img_worse"):
        imgs = [sample[label] for sample in batch]
        imgs = [image_transforms(img) for img in imgs]
        img_tensors = torch.stack(imgs)
        processed_batch[label] = img_tensors

    return processed_batch


def calculate_loss(
    model: LinearEmbAug,
    batch,
):
    prompt_input = batch["Prompt"]
    prompt_target = batch["Upsampled"]
    loss = F.mse_loss(resized, latent_target)
    logs = {"mse_latent": mse_latent.cpu().item()}
    return loss, logs


def train(
    vae_path: str = "runwayml/stable-diffusion-v1-5",
    objective: str = "upscale",
    test_steps: int = 1000,
    test_batches: int = 10,
    output_filename: str = "sdxl_resizer.pt",
    steps: float = 1e4,
    save_steps: int = 5000,
    batch_size: int = 2,
    num_dataloader_workers: int = 0,
    lr: float = 2e-4,
    dropout: float = 0.0,
    clip_grad_val: float = 50.0,
    use_bnb: bool = True,
    use_wandb: bool = False,
):
    device = torch.device("cuda")
    steps = int(steps)

    dataset = load_dataset(
        "roborovski/upsampled-prompts-parti", "train", verification_mode="no_checks"
    )
    dataset = dataset.train_test_split(test_size=0.05)  # type: ignore
    train_dataset: Dataset = dataset["train"]
    test_dataset: Dataset = dataset["test"]

    train_dataloader = DataLoader(
        train_dataset, # type: ignore
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_dataloader_workers,
    )
    test_dataloader = DataLoader(
        test_dataset, # type: ignore
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_dataloader_workers,
    )


    model = LinearEmbAug().to(device)

    model.train()
    model.requires_grad_(True)

    if use_wandb:
        wandb.init(project="superprompt-latent-aug")
        wandb.watch(model)

    print(summary(model))

    if use_bnb:
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True) # type: ignore
    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, total_iters=200
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[20]
    )

    model.train()
    epoch = 0
    step = 0
    progress_bar = tqdm(range(steps))
    progress_bar.set_description("Steps")

    while step < steps:
        epoch += 1
        for batch in train_dataloader:
            step += 1

            # get loss
            with autocast(device_type="cuda", dtype=torch.float16):
                loss, logs = calculate_loss(
                    model,
                    batch,
                )
                loss_rounded = round(loss.cpu().item(), 2)

            scaler.scale(loss).backward() # type: ignore
            scaler.unscale_(optimizer)

            if clip_grad_val > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_val) # type: ignore

            norm_text, lr_text = (
                round(get_model_gradient_norm(model), 3),
                scheduler.get_last_lr()[0],
            )
            progress_bar.set_postfix(loss=loss_rounded, lr=lr_text, norm=norm_text)

            if use_wandb:
                wandb.log({**logs, "lr": lr_text, "norm": norm_text})

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            progress_bar.update(1)
            scheduler.step()

            if step >= steps:
                break
            if (step % save_steps) == 0:
                base, ext = os.path.splitext(output_filename)
                save_filename = f"{base}-{step}{ext}"
                torch.save(model.state_dict(), save_filename)
            if step % test_steps == 0:
                test_batches = 0
                test_logs = defaultdict(float)
                model.eval()
                for batch in test_dataloader:
                    with torch.inference_mode():
                        _, logs = calculate_loss(
                            model,
                            batch,
                        )
                    test_batches += 1
                    for k in logs.keys():
                        test_logs[k] += logs[k]
                    if test_batches >= test_batches:
                        break
                model.train()

    wandb.finish()
    torch.save(model.state_dict(), output_filename)
    print("Model saved")


if __name__ == "__main__":
    fire.Fire(train)
