import torch
import os
import platform
from typing import Dict, List
import numpy as np


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


def get_model_gradient_norm(model):
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


def split_subject_descriptors(batch: Dict, nlp):
    """
    Splits a batch of prompts into subjects and descriptors.
    """
    out = {
        "subject": [],
        "descriptor": [],
    }
    for prompt in batch["prompt"]:
        doc = nlp(prompt)
        subject_tokens, descriptor_tokens = [], []

        # find the first chunk with either an entity or a proper noun.
        subject_found = False
        for chunk in doc.noun_chunks:
            if subject_found:
                descriptor_tokens.append(chunk.text)
            else:
                proper_nouns = [token for token in chunk if token.pos_ == "PROPN"]
                proper_ents, non_proper_ents = [], []
                for ent in chunk.ents:
                    if ent.label_ == "PERSON" or ent.label_ == "ORG":
                        proper_ents.append(ent)
                    else:
                        non_proper_ents.append(ent)
                subject_tokens.append(chunk.root.text)
                if len(non_proper_ents) > 0:
                    subject_tokens.append(chunk.text)
                    subject_found = True
                elif len(proper_nouns) > 0 and len(proper_ents) == 0:
                    subject_tokens.append(chunk.text)
                    subject_found = True

        # print("token deps")
        subject_tokens = [
            tok for i, tok in enumerate(subject_tokens) if tok not in subject_tokens[:i]
        ]
        out["subject"].append(" ".join(subject_tokens))
        out["descriptor"].append(" ".join(descriptor_tokens))
    return out

def compute_dcg(relevance: List[int], k):
    dcg = 0.0
    for i in range(k):
        dcg += (2 ** relevance[i] - 1) / np.log2(i + 2)
    return dcg

def compute_ndcg(true_rankings, pred_rankings, k):
    true_relevance = [1 if i in true_rankings else 0 for i in range(k)]
    true_dcg = compute_dcg(true_relevance, k)
    pred_relevance = [1 if i in pred_rankings else 0 for i in range(k)]
    pred_dcg = compute_dcg(pred_relevance, k)
    return pred_dcg / true_dcg