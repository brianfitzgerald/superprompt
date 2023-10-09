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
    if os.environ.get("NO_WANDB", False):
        return False
    return os.environ.get("USER") == "ubuntu" and platform.system().lower() == "linux"


sample_prompt_pairs = [
    (
        "portait witch hyper background",
        "portait of mystical witch, hyper detailed, flowing background, intricate and detailed, trippy, 8 k ",
    ),
    (
        "painting ghost riders sky sunrise wlop tooth wu charlie russell",
        "a beautiful painting of ghost riders in the sky, sunrise, by wlop, tooth wu and charlie russell",
    ),
    (
        "princess dance fairy gold dress sky stanley artgerm lau greg rutkowski victo ngai alphonse loish norman",
        "chinese princess, dance, fairy, beautiful, stunning, red and gold dress, spinning in the sky, by stanley artgerm lau, greg rutkowski, victo ngai, alphonse mucha, loish, norman rockwell",
    ),
    (
        "scene painting girl balustrade dress pattern seaside resort buildings background dusk clouds seagulls artstation krenz cushart alphonse maria mucha point composition k resolution hand illustration style",
        "a beautiful scene painting of a young girl, with a maiden balustrade in a white dress with a beautiful pattern, a beautiful deserted seaside resort with many wooden buildings in the background, romantic dusk, beautiful clouds, seagulls, trending on artstation, by krenz cushart and alphonse maria mucha, three - point composition, 8 k resolution, hand - painted, illustration style",
    ),
    (
        "portrait samurai goth punk colors style alexander mcqueen hyper art bill sienkiewicz artstation background",
        "close up portrait of old samurai, goth punk, vibrant yellow colors, surreal, french baroque style by alexander mcqueen, hyper detailed, cinematic, art by bill sienkiewicz trending artstation, remove red background",
    ),
]

sample_translate_pairs = [
    ("Ich bin ein Mann mit einem Pferd", "I am a man with a horse"),
    ("Ich möchte den Gipfel des Berges sehen", "I wish to see the top of the mountain"),
]

def get_model_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None and p.grad.data is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def weights_biases_sum(model):
    total_weight_sum = 0.0
    for param in model.parameters():
        total_weight_sum += param.data.sum().item()
    return total_weight_sum