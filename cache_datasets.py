from datasets import load_dataset
from diffusers import (
    StableDiffusionPipeline,
)
import torch

dataset = load_dataset("roborovski/diffusiondb-seq2seq")
dataset = load_dataset("THUDM/ImageRewardDB", "4k", verification_mode="no_checks")
dataset = load_dataset("bentrevett/multi30k")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
)