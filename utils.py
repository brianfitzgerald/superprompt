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
    return os.environ.get("USER") == "ubuntu" or platform.system() != "Darwin"

sample_prompts = [
    "human sculpture of lanky tall alien on a romantic date at italian restaurant with smiling woman, nice restaurant, photography, bokeh",
    "portrait of barbaric spanish conquistador, symmetrical, by yoichi hatakenaka, studio ghibli and dan mumford",
    "a small liquid sculpture, corvette, viscous, reflective, digital art",
    "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render.",
    "cyber moai on easter island, digital painting, highly detailed, concept art, trending on artstation, epic composition, sharp focus, flawless render, masterpiece, volumetric lighting",
]
