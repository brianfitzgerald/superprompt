from argparse import Namespace
from typing import Dict
from datasets import load_dataset
import spacy

from transformers import AutoTokenizer, CLIPTextModel
from utils import should_use_wandb, get_available_device

device = get_available_device()

model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
max_clip_length = model.config.max_position_embeddings

if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


class Args(Namespace):
    use_wandb = should_use_wandb()


def mask_non_nouns(prompt):
    doc = nlp(prompt)
    masked_tokens = []
    pos = set({"NOUN", "PROPN"})
    for token in doc:
        if token.pos_ in pos or (token.dep_ == "nsubj" and token.pos_ == "VERB"):
            masked_tokens.append(token.text)
    return " ".join(masked_tokens)


def process_batch(batch):
    batch["masked_prompts"] = [mask_non_nouns(prompt) for prompt in batch["prompt"]]
    return batch


def preprocess_dataset(batch: Dict):
    unmasked_prompts = batch["prompt"]
    unmasked_inputs = tokenizer(
        text=unmasked_prompts,
        return_tensors="pt",
        max_length=max_clip_length,
        padding="max_length",
        truncation=True,
    ).to(device)
    unmasked_embeddings = model(**unmasked_inputs).last_hidden_state

    masked_prompts = [mask_non_nouns(prompt) for prompt in unmasked_prompts]
    masked_inputs = tokenizer(
        text=masked_prompts,
        return_tensors="pt",
        max_length=max_clip_length,
        padding="max_length",
        truncation=True,
    ).to(device)
    masked_embeddings = model(**masked_inputs).last_hidden_state
    batch_dict = {
        "unmasked_embeddings": unmasked_embeddings,
        "masked_embeddings": masked_embeddings,
        "masked_prompts": masked_prompts,
        "unmasked_prompts": unmasked_prompts,
    }
    return batch_dict


# TODO filter for only high image text alignment scores
remove_cols = ['image', 'prompt_id', 'prompt', 'classification', 'image_amount_in_total', 'rank', 'overall_rating', 'image_text_alignment_rating', 'fidelity_rating']
dataset = load_dataset("THUDM/ImageRewardDB", "1k")
dataset = dataset.map(preprocess_dataset, batched=True, num_proc=1, batch_size=48, remove_columns=remove_cols)
