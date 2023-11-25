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
from transformers import CLIPTextModel, AutoTokenizer, CLIPTokenizer

torch.manual_seed(42)

size = 512
to_tensor = transforms.ToTensor()
image_transforms = transforms.Compose(
    [
        transforms.RandomCrop(size, pad_if_needed=True, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def calculate_loss(
    model: LinearEmbAug,
    batch,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    device,
):
    input_tokenized = tokenizer(
        batch["Prompt"], truncation=True, padding=True, return_tensors="pt"
    ).to(device)
    input_encoded = text_encoder(**input_tokenized).pooler_output

    label_tokenized = tokenizer(
        batch["Upsampled"], truncation=True, padding=True, return_tensors="pt"
    ).to(device)
    label_encoded = text_encoder(**label_tokenized).pooler_output

    out = model(input_encoded)
    loss = F.mse_loss(out, label_encoded)
    return loss


def train(
    test_steps: int = 1000,
    test_batches: int = 10,
    output_filename: str = "linear_emb_aug.pt",
    save_steps: int = 5000,
    batch_size: int = 8,
    num_dataloader_workers: int = 0,
    n_epochs: int = 10,
    lr: float = 1e-4,
    clip_grad_val: float = 0,
    use_bnb: bool = True,
    use_wandb: bool = False,
):
    device = torch.device("cuda")

    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")  # type: ignore
    text_encoder.to(device)  # type: ignore
    tokenizer: CLIPTokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")  # type: ignore

    dataset = load_dataset(
        "roborovski/upsampled-prompts-parti", verification_mode="no_checks"
    )
    dataset = dataset["train"].train_test_split(test_size=0.05)  # type: ignore
    train_dataset: Dataset = dataset["train"]
    test_dataset: Dataset = dataset["test"]

    train_dataloader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=batch_size,
        num_workers=num_dataloader_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,  # type: ignore
        batch_size=batch_size,
        num_workers=num_dataloader_workers,
    )
    n_steps = len(train_dataloader)

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
    scaler = torch.cuda.amp.GradScaler(enabled=True)  # type: ignore
    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, total_iters=200
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[20]
    )

    model.train()
    epoch = 0
    global_step = 0

    for epoch in range(n_epochs):

        progress_bar = tqdm(range(n_steps))
        progress_bar.set_description("Steps")

        for batch in train_dataloader:
            global_step += 1

            # get loss
            with autocast(device_type="cuda", dtype=torch.float16):
                loss = calculate_loss(model, batch, text_encoder, tokenizer, device)
                loss_rounded = round(loss.cpu().item(), 2)

            scaler.scale(loss).backward()  # type: ignore
            scaler.unscale_(optimizer)

            if clip_grad_val > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_val)  # type: ignore

            norm_text, lr_text = (
                round(get_model_gradient_norm(model), 3),
                scheduler.get_last_lr()[0],
            )
            progress_bar.set_postfix(
                loss=loss_rounded, lr=lr_text, norm=norm_text, epoch=epoch
            )

            if use_wandb:
                wandb.log({"lr": lr_text, "norm": norm_text, "loss": loss})

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            progress_bar.update(1)
            scheduler.step()

            if (global_step % save_steps) == 0:
                base, ext = os.path.splitext(output_filename)
                save_filename = f"{base}-{global_step}{ext}"
                torch.save(model.state_dict(), save_filename)
            if global_step % test_steps == 0:
                test_batches = 0
                model.eval()
                for batch in test_dataloader:
                    with torch.inference_mode():
                        loss = calculate_loss(
                            model, batch, text_encoder, tokenizer, device
                        )
                    test_batches += 1
                    if test_batches >= test_batches:
                        break
                model.train()

    wandb.finish()
    torch.save(model.state_dict(), output_filename)
    print("Model saved")


if __name__ == "__main__":
    fire.Fire(train)
