#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from diffusers import AutoencoderKL
import argparse
import os
import random
import webdataset as wds
import io
from tqdm.auto import tqdm
from models.latent_aug import LatentAugmenter
import lpips
from collections import defaultdict
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import fire
from typing import List
from torchinfo import summary
from enum import Enum
from datasets import load_dataset
import bitsandbytes as bnb
import wandb

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
    model: LatentAugmenter,
    batch,
    device: torch.device,
    vae: AutoencoderKL,
    lpips_fn,
    dtype,
    decoded_weight: float,
    lpips_weight: float,
    mse_latent_weight: float,
):
    img_input = batch["img_worse"].to(device, dtype=dtype)
    img_target = batch["img_better"].to(device, dtype=dtype)
    latent_input = (
        vae.config.scaling_factor * vae.encode(img_input).latent_dist.sample()
    )
    latent_target = (
        vae.config.scaling_factor * vae.encode(img_target).latent_dist.sample()
    )
    size = latent_target.shape[-2:]
    resized = model(latent_input, size=size)
    mse_latent = F.mse_loss(resized, latent_target)
    logs = {"mse_latent": mse_latent.cpu().item()}
    loss = mse_latent_weight * mse_latent
    if decoded_weight > 0:
        decoded = vae.decode(resized / vae.config.scaling_factor)[0]
        decoded_loss = F.mse_loss(decoded, img_target)
        logs["mse"] = decoded_loss
        loss = loss + decoded_weight * decoded_loss
    if lpips_weight > 0:
        lpips_loss = lpips_fn(decoded, img_target).mean()
        logs["lpips"] = lpips_loss.cpu().item()
        loss = loss + lpips_weight * lpips_loss
    logs["loss"] = loss.cpu().item()
    return loss, logs


class Objective(Enum):
    Upscale = "upscale"
    augment = "augment"


def get_model_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None and p.grad.data is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def train(
    vae_path: str = "runwayml/stable-diffusion-v1-5",
    objective: str = "upscale",
    test_path: List[str] = None,
    test_steps: int = 1000,
    test_batches: int = 10,
    output_filename: str = "sdxl_resizer.pt",
    steps: int = 1e4,
    save_steps: int = 5000,
    batch_size: int = 2,
    num_dataloader_workers: int = 0,
    lr: float = 2e-4,
    dropout: float = 0.0,
    clip_grad_val: float = 50.0,
    device: str = "cuda",
    init_weights: str = None,
    fp16: bool = True,
    use_bnb: bool = True,
    use_wandb: bool = False,
):
    device = torch.device(device)
    objective = Objective(objective)
    steps = int(steps)

    dataset = load_dataset(
        "THUDM/ImageRewardDB", "2k_pair", verification_mode="no_checks"
    )
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_dataloader_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_dataloader_workers,
    )

    vae_dtype = torch.float32
    if fp16:
        vae_dtype = torch.float16

    vae = (
        AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
        .to(device)
        .to(dtype=vae_dtype)
    )
    # Use this scale even with SD 1.5
    vae.config.scaling_factor = 0.13025

    lpips_fn = lpips.LPIPS(net="vgg").to(device=device, dtype=vae_dtype)

    if init_weights:
        model = LatentAugmenter.load_model(
            init_weights,
            device=device,
            dropout=dropout,
            dtype=torch.float32,
        )
    else:
        model = LatentAugmenter(dropout=dropout).to(device)

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
    scaler = torch.cuda.amp.GradScaler(enabled=True)
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
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, logs = calculate_loss(
                    model,
                    batch,
                    device,
                    vae,
                    lpips_fn,
                    vae_dtype,
                    lpips_weight=0,
                    decoded_weight=0,
                    mse_latent_weight=1,
                )
                loss_rounded = round(loss.cpu().item(), 2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if clip_grad_val > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_val)

            norm_text, lr_text = round(get_model_grad_norm(model), 3), scheduler.get_last_lr()[0]
            progress_bar.set_postfix(
                loss=loss_rounded, lr=lr_text, norm=norm_text
            )

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
            if test_path and (step % test_steps) == 0:
                test_batches = 0
                test_logs = defaultdict(float)
                model.eval()
                for batch in test_dataloader:
                    with torch.inference_mode():
                        _, logs = calculate_loss(
                            model,
                            batch,
                            device,
                            vae,
                            lpips_fn,
                            vae_dtype,
                            lpips_weight=1,
                            decoded_weight=1,
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
