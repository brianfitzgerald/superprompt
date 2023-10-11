from argparse import Namespace
from typing import Dict
from datasets import load_dataset
import spacy
import wandb
from models.clip_emb_aug import CLIPEmbeddingAugmenter
from models.cross_encoder import CrossEncoder
import torch.nn as nn
import fire
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers.utils.import_utils import is_xformers_available
import gc
from torchinfo import summary
import os
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from diffusers import (
    StableDiffusionPipeline,
)
from transformers import AutoTokenizer, CLIPTextModel
from utils import get_available_device
from enum import Enum
from typing import Dict
import torch.nn.functional as F
from pathlib import Path
import shutil

if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")

torch.manual_seed(0)


def loss_fn_emb_aug(
    batch: Dict,
    device: torch.device,
    model: CLIPEmbeddingAugmenter,
):
    mask_emb, unmask_emb = (
        batch["masked_embeddings"].to(device),
        batch["unmasked_embeddings"].to(device),
    )

    model_out = model(mask_emb)
    loss = F.mse_loss(model_out, unmask_emb)

    return loss, model_out


def main(use_wandb: bool = False, eval_every: int = 25):
    device = get_available_device()

    print("Loading dataset..")
    dataset = load_dataset("roborovski/diffusiondb-seq2seq")

    print("Loading model..")
    model = CrossEncoder(device)
    print(summary(model))

    # Hyperparameters
    num_epochs: int = 200
    learning_rate: float = 1e-5
    batch_size: int = 64

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs // 4, eta_min=learning_rate / 10
    )

    if use_wandb:
        wandb.init(project="superprompt-aug")
        wandb.watch(model)

    train_dataset, val_dataset = dataset["train"], dataset["validation"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=4)

    for epoch in range(num_epochs):
        train_iter = tqdm(train_loader, total=len(train_loader))
        for batch in train_iter:
            loss, model_out = loss_fn_emb_aug(batch, device, model)

            log_dict = {
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }

            train_iter.set_postfix(log=log_dict)
            if use_wandb:
                wandb.log(log_dict)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

        if i % eval_every == 0:
            for batch in val_loader:
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    safety_checker=None,
                )
                pipe = pipe.to("cuda")
                loss, model_out = loss_fn_emb_aug(batch, device, model)

                if is_xformers_available():
                    pipe.unet.enable_xformers_memory_efficient_attention()

                log_dict = {
                    "loss": loss.item(),
                    "unmasked": [],
                    "encoded": [],
                }

                unmask_emb = batch["unmasked_embeddings"].to(device)

                Path("out").mkdir(exist_ok=True)
                shutil.rmtree("out")
                Path("out").mkdir(exist_ok=True)

                for key in ("unmasked", "encoded"):
                    print(f"Generating {key} images...")
                    embeds = unmask_emb if key == "unmasked" else model_out
                    generations = pipe(prompt_embeds=embeds).images
                    os.makedirs("out", exist_ok=True)
                    for i, generation in enumerate(generations):
                        generation.save(f"out/{key}_{i}.png")
                        if use_wandb:
                            log_dict[key].append(wandb.Image(generation))

                if use_wandb:
                    wandb.log(log_dict)

                del pipe
                gc.collect()
                torch.cuda.empty_cache()

                break

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
