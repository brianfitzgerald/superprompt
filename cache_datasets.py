from datasets import load_dataset
from diffusers import (
    StableDiffusionPipeline,
)
import torch

dataset = load_dataset("roborovski/diffusiondb-seq2seq")
dataset = load_dataset("THUDM/ImageRewardDB", "4k", verification_mode="no_checks")
dataset = load_dataset("bentrevett/multi30k")
dataset = load_dataset(
    "huggan/CelebA-HQ",
    data_files={"train": "data/train-00000-of-00068.parquet"},
    verification_mode="no_checks",
)
dataset = load_dataset(
    "roborovski/celeba-faces-captioned",
    data_files={
        "train": [
            "data/train-00000-of-00036-416615b669d11cd3.parquet",
            "data/train-00001-of-00036-411c3786c0f93eac.parquet",
        ]
    },
    verification_mode="no_checks",
)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
)
