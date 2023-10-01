from argparse import Namespace
import gc
from typing import Dict
from datasets import load_dataset
from PIL import Image
import requests
import datasets

from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, CLIPTextModel
from utils import should_use_wandb, get_available_device

device = get_available_device()

model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
max_clip_length = model.config.max_position_embeddings

class Args(Namespace):
    use_wandb = should_use_wandb()


def preprocess_dataset(batch: Dict):
    prompts = batch["prompt"]
    inputs = tokenizer(
        text=prompts,
        return_tensors="pt",
        max_length=max_clip_length,
        padding="max_length",
        truncation=True,
    ).to(device)
    clip_output = model(**inputs)
    return {
        "embeddings": clip_output.last_hidden_state,
    }


dataset = load_dataset("THUDM/ImageRewardDB", "1k")
dataset = dataset.map(preprocess_dataset, batched=True, num_proc=1, batch_size=12)
