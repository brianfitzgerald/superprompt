import torch
import os
import platform


def get_available_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def should_use_wandb():
    if os.environ["NO_WANDB"]:
        return False
    return os.environ.get("USER") == "ubuntu" and platform.system().lower() == "linux"

sample_prompt_pairs = [
    ("portait witch hyper background", "portait of mystical witch, hyper detailed, flowing background, intricate and detailed, trippy, 8 k "),
    ("painting ghost riders sky sunrise wlop tooth wu charlie russell", "a beautiful painting of ghost riders in the sky, sunrise, by wlop, tooth wu and charlie russell")
]
