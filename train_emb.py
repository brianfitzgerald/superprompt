from argparse import Namespace
from typing import Dict
from datasets import load_dataset
import spacy
import wandb
from models.clip_emb_aug import CLIPEmbeddingAugmenter
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

if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")

torch.manual_seed(0)


def process_batch(
    batch, model: CLIPEmbeddingAugmenter, optimizer, device, epoch
):
    mask_emb, unmask_emb = (
        batch["masked_embeddings"].to(device),
        batch["unmasked_embeddings"].to(device),
    )

    mask_emb_enc = model(mask_emb)
    loss = model.loss_fn(mask_emb_enc, unmask_emb)

    loss_formatted = float(round(loss.item(), 4))
    log_dict = {
        "loss": loss_formatted,
        "epoch": epoch,
        "lr": optimizer.param_groups[0]["lr"],
    }
    return loss, log_dict, mask_emb, unmask_emb, mask_emb_enc


def main(use_wandb: bool = False, eval_every: int = 25):
    device = get_available_device()

    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    max_clip_length = clip_model.config.max_position_embeddings

    model = CLIPEmbeddingAugmenter(clip_model)
    print(summary(model))

    nlp = spacy.load("en_core_web_sm")

    def mask_non_nouns(prompt):
        doc = nlp(prompt)
        masked_tokens = []
        pos = set({"NOUN", "PROPN"})
        for token in doc:
            if token.pos_ in pos or (token.dep_ == "nsubj" and token.pos_ == "VERB"):
                masked_tokens.append(token.text)
        return " ".join(masked_tokens)

    def preprocess_dataset(batch):
        unmasked_prompts = batch["prompt"]
        unmasked_inputs = tokenizer(
            text=unmasked_prompts,
            return_tensors="pt",
            max_length=max_clip_length,
            padding="max_length",
            truncation=True,
        ).to(device)
        unmasked_clip_out = clip_model(**unmasked_inputs)
        unmasked_embeddings = unmasked_clip_out.last_hidden_state


        masked_prompts = [mask_non_nouns(prompt) for prompt in unmasked_prompts]
        masked_inputs = tokenizer(
            text=masked_prompts,
            return_tensors="pt",
            max_length=max_clip_length,
            padding="max_length",
            truncation=True,
        ).to(device)
        masked_clip_out = clip_model(**masked_inputs)
        masked_embeddings = masked_clip_out.last_hidden_state
        batch_dict = {
            "unmasked_embeddings": unmasked_embeddings,
            "masked_embeddings": masked_embeddings,
            "masked_prompts": masked_prompts,
            "unmasked_prompts": unmasked_prompts,
        }
        return batch_dict

    # TODO filter for only high image text alignment scores
    remove_cols = [
        "image",
        "prompt_id",
        "prompt",
        "classification",
        "image_amount_in_total",
        "rank",
        "overall_rating",
        "image_text_alignment_rating",
        "fidelity_rating",
    ]
    dataset = load_dataset("THUDM/ImageRewardDB", "4k", verification_mode="no_checks")
    dataset.set_format("torch")
    dataset = dataset.map(
        preprocess_dataset,
        cache_file_names={"train": "train_cache", "validation": "val_cache", "test": "test_cache"},
        batched=True,
        num_proc=1,
        drop_last_batch=True,
        batch_size=96,
        remove_columns=remove_cols,
    )
    # dataset.save_to_disk("image_reward_processed")

    print("Loading model..")

    model.train()

    # Hyperparameters
    num_epochs: int = 200
    learning_rate: float = 1e-5
    batch_size: int = 64

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs // 4, eta_min=learning_rate / 10
    )

    epoch = 0

    if use_wandb:
        wandb.init(project="superprompt-aug")
        wandb.watch(model)

    train_dataset, val_dataset = dataset["train"], dataset["validation"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=4)

    for i, epoch in enumerate(range(num_epochs)):
        train_iter = tqdm(train_loader, total=len(train_loader))
        for batch in train_iter:
            loss, log_dict, _, _, _ = process_batch(
                batch, model, optimizer, device, epoch
            )

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
                loss, log_dict, mask_emb, unmask_emb, mask_emb_enc = process_batch(
                    batch, model, optimizer, device, epoch
                )

                if is_xformers_available():
                    pipe.unet.enable_xformers_memory_efficient_attention()

                log_dict["unmasked"] = []
                log_dict["encoded"] = []
                for key in ("unmasked", "encoded"):
                    print(f"Generating {key} images...")
                    embeds = unmask_emb if key == "unmasked" else mask_emb_enc
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
