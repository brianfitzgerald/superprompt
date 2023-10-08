from argparse import Namespace
from typing import Dict
from datasets import load_dataset
import spacy
import wandb
from models.emb_aug_linear import EmbAugLinear
from models.emb_retrieval import SiameseEmbRetriever
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


def process_batch(batch, model, criterion, optimizer, device, epoch):
    mask_emb, unmask_emb = (
        batch["masked_embeddings"].to(device),
        batch["unmasked_embeddings"].to(device),
    )

    outputs = model(mask_emb)

    loss = criterion(outputs, unmask_emb)
    loss_formatted = float(round(loss.item(), 4))
    log_dict = {
        "loss": loss_formatted,
        "epoch": epoch,
        "lr": optimizer.param_groups[0]["lr"],
    }
    return loss, log_dict, mask_emb, unmask_emb, outputs

class ModelChoice(Enum):
    LINEAR = "aug_linear"
    RETRIEVAL = "retrieval"
    
def main(use_wandb: bool = False, use_model: str = "retrieval", eval_every: int = 10):

    model_choice = ModelChoice(use_model)
    if model_choice == ModelChoice.LINEAR:
        model = EmbAugLinear(device=device)
    elif model_choice == ModelChoice.RETRIEVAL:
        model = SiameseEmbRetriever(device=device)
    print(summary(model))

    device = get_available_device()

    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    max_clip_length = clip_model.config.max_position_embeddings

    nlp = spacy.load("en_core_web_sm")

    def mask_non_nouns(prompt):
        doc = nlp(prompt)
        masked_tokens = []
        pos = set({"NOUN", "PROPN"})
        for token in doc:
            if token.pos_ in pos or (token.dep_ == "nsubj" and token.pos_ == "VERB"):
                masked_tokens.append(token.text)
        return " ".join(masked_tokens)

    def preprocess_dataset(batch: Dict):
        unmasked_prompts = batch["prompt"]
        unmasked_inputs = tokenizer(
            text=unmasked_prompts,
            return_tensors="pt",
            max_length=max_clip_length,
            padding="max_length",
            truncation=True,
        ).to(device)
        unmasked_embeddings = clip_model(**unmasked_inputs).last_hidden_state

        masked_prompts = [mask_non_nouns(prompt) for prompt in unmasked_prompts]
        masked_inputs = tokenizer(
            text=masked_prompts,
            return_tensors="pt",
            max_length=max_clip_length,
            padding="max_length",
            truncation=True,
        ).to(device)
        masked_embeddings = clip_model(**masked_inputs).last_hidden_state
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
    dataset = load_dataset("THUDM/ImageRewardDB", "4k")
    dataset = dataset.map(
        preprocess_dataset,
        batched=True,
        num_proc=1,
        batch_size=96,
        remove_columns=remove_cols,
    )
    dataset.set_format("torch")

    print("Loading model..")

    model.train()

    # Hyperparameters
    num_epochs: int = 200
    learning_rate: float = 1e-4
    batch_size: int = 256

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs // 4, eta_min=learning_rate / 10
    )
    criterion = nn.MSELoss()

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
                batch, model, criterion, optimizer, device, epoch
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
                loss, log_dict, mask_emb, _, outputs = process_batch(
                    batch, model, criterion, optimizer, device, epoch
                )

                if is_xformers_available():
                    pipe.unet.enable_xformers_memory_efficient_attention()

                generations = pipe(prompt_embeds=outputs).images
                os.makedirs("out", exist_ok=True)
                for i, generation in enumerate(generations):
                    log_dict[f"generation_{i}"] = generation
                    generation.save(f"out/generation_{i}.png")

                del pipe
                gc.collect()
                torch.cuda.empty_cache()

                break

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
