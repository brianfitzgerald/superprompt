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
from PIL import Image
from torchvision import transforms
import fire
from typing import List
from torchinfo import summary


def init_dataset(dataset_path, size=512):
    shards = []
    if type(dataset_path) not in (list, tuple):
        dataset_path = [dataset_path]
    for path in dataset_path:
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if full_path.endswith(".tar"):
                shards.append(full_path)
    print(f"{len(shards)} shards")

    def preprocess(sample):
        k = [k for k in sample.keys() if k in ["jpg", "png"]]
        if len(k) == 0:
            raise ValueError("Dataset images should be in jpg or png format")
        k = k[0]
        img = sample[k]

        img = Image.open(io.BytesIO(img))
        if not img.mode == "RGB":
            img = img.convert("RGB")

        pil_image = img

        image_transforms = transforms.Compose(
            [
                transforms.RandomCrop(size, pad_if_needed=True, padding_mode="reflect"),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        img = image_transforms(img)
        examples = {}
        examples["img"] = img
        return examples

    dataset = (
        wds.WebDataset(shards, handler=wds.warn_and_continue, shardshuffle=True)
        .repeat()
        .shuffle(256)
        .map(preprocess)
    )
    return dataset


def collate_fn(examples):
    imgs = [example["img"] for example in examples]
    imgs = torch.stack(imgs)

    scale = random.uniform(1.0, 2.1)
    size = [int(round(imgs.shape[-2] * scale)), int(round(imgs.shape[-1] * scale))]
    size[0] -= size[0] % 8
    size[1] -= size[1] % 8
    imgs_scaled = F.interpolate(imgs, size=size, mode="bilinear")

    if scale < 1:
        batch = {
            "img_input": imgs_scaled,
            "img_target": imgs,
        }
    else:
        batch = {
            "img_input": imgs,
            "img_target": imgs_scaled,
        }

    return batch


def calculate_loss(
    model,
    batch,
    device,
    vae,
    lpips,
    dtype=torch.float16,
    mse_weight=1,
    lpips_weight=0.1,
    mse_latent_weight=0.01,
):
    img_input = batch["img_input"].to(device, dtype=dtype)
    img_target = batch["img_target"].to(device, dtype=dtype)
    latent_input = (
        vae.config.scaling_factor * vae.encode(img_input).latent_dist.sample()
    )
    latent_target = (
        vae.config.scaling_factor * vae.encode(img_target).latent_dist.sample()
    )
    size = latent_target.shape[-2:]
    with torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32):
        resized = model(latent_input, size=size)
    mse_latent = F.mse_loss(resized, latent_target)
    logs = {"mse_latent": mse_latent.detach().cpu().item()}
    decoded = vae.decode(resized / vae.config.scaling_factor)[0]
    mse = F.mse_loss(decoded, img_target)
    logs["mse"] = mse
    loss = mse_weight * mse + mse_latent_weight * mse_latent
    if lpips_weight > 0:
        ploss = lpips(decoded, img_target).mean()
        logs["lpips"] = ploss.detach().cpu().item()
        loss = loss + lpips_weight * ploss
    logs["loss"] = loss.detach().cpu().item()
    return loss, logs


def train(
    train_path: List[str],
    vae_path: str,
    test_path: List[str] = None,
    test_steps: int = 1000,
    test_batches: int = 10,
    output_filename: str = "sdxl_resizer.pt",
    steps: int = 100000,
    save_steps: int = 5000,
    batch_size: int = 4,
    num_workers: int = 4,
    lr: float = 2e-4,
    dropout: float = 0.0,
    grad_clip: float = 5.0,
    device: str = "cuda",
    resolution: int = 256,
    init_weights: str = None,
    fp16: bool = False,
    gradient_checkpointing: bool = False,
    display_norm: bool = True,
):
    device = torch.device(device)

    vae_dtype = torch.float32
    if fp16:
        vae_dtype = torch.float16

    vae = AutoencoderKL.from_single_file(vae_path).to(device, dtype=vae_dtype)
    # Use this scale even with SD 1.5
    vae.config.scaling_factor = 0.13025

    vae.train()
    if gradient_checkpointing:
        vae.enable_gradient_checkpointing()
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

    print(summary(model))

    train_dataset = init_dataset(train_path, size=resolution)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    if test_path:
        test_dataset = init_dataset(test_path, size=resolution)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, total_iters=200
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[20]
    )
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(params, "Parameters")
    model.train()
    epoch = 0
    step = 0
    progress_bar = tqdm(range(steps))
    progress_bar.set_description("Steps")
    train_fn = lambda batch: calculate_loss(
        model, batch, device, vae, lpips_fn, vae_dtype
    )

    while step < steps:
        epoch += 1
        for batch in train_dataloader:
            if batch["img_input"].shape == batch["img_target"].shape:
                continue
            step += 1
            loss, logs = train_fn(batch)
            l = loss.detach().cpu().item()
            print(logs)
            progress_bar.set_postfix(loss=round(l, 2), lr=scheduler.get_last_lr()[0])
            scaler.scale(loss).backward()
            if display_norm:
                total_norm = 0
                for p in model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1.0 / 2)
                print("norm", total_norm)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
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
                        _, logs = loss, logs = train_fn(batch)
                    test_batches += 1
                    for k in logs.keys():
                        test_logs[k] += logs[k]
                    if test_batches >= test_batches:
                        break
                model.train()
                print(test_logs)

    torch.save(model.state_dict(), output_filename)
    print("Model saved")
