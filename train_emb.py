from argparse import Namespace
from typing import Dict
from datasets import load_dataset
import spacy
import wandb
from models.emb_aug_model import EmbeddingAugModel
import torch.nn as nn
import fire
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader


from transformers import AutoTokenizer, CLIPTextModel
from utils import get_available_device

if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")

def main(use_wandb: bool = False):
    device = get_available_device()

    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
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
    dataset = load_dataset("THUDM/ImageRewardDB", "1k")
    dataset = dataset.map(
        preprocess_dataset,
        batched=True,
        num_proc=1,
        batch_size=48,
        remove_columns=remove_cols,
    )

    model = EmbeddingAugModel(device=device)

    # display(model)
    model.train()
    num_epochs = 100
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    epoch = 0

    if use_wandb:
        wandb.init(project="superprompt-aug")
        wandb.watch(model)

    train_dataset, val_dataset = dataset["train"], dataset["validation"]
    train_loader = DataLoader(train_dataset, batch_size=48)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

    # Training loop
    for epoch in range(num_epochs):
        train_iter = tqdm(train_loader, total=len(train_loader))
        for batch in train_iter:
            mask_emb, unmask_emb = (
                batch["masked_embeddings"],
                batch["unmasked_embeddings"],
            )

            optimizer.zero_grad()

            outputs = model(mask_emb)

            loss = criterion(outputs, unmask_emb)
            loss_formatted = "{:.4f}".format(loss.item())
            log_dict = {
                "loss": loss_formatted,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
            }
            train_iter.set_postfix(log=log_dict)
            if use_wandb:
                wandb.log(log_dict)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)