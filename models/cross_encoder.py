#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Union
from torch import Tensor

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def pooling(
    model_output: Tensor, attention_mask: Tensor
) -> Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def multiple_negatives_ranking_loss(
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    scale: float = 20.0,
) -> Tensor:
    """
    Cross entropy between a[i] and b[i] where b is the batch of embeddings.
    """

    # [bsz, bsz]
    scores = cos_sim(embeddings_a, embeddings_b) * scale
    # label here is the index of the positive example for a given example
    labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
    return F.cross_entropy(scores, labels)

class CrossEncoder(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        self.language_model = AutoModel.from_pretrained(
            "distilroberta-base"
        )
        self.language_model.to(device)
        self.language_model.train()
        self.embedding_dimension: int = self.language_model.config.hidden_size
        self.device = device

    def forward(self, input_text: List[str],) -> Tensor:
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 128,
        }
        tokenized = self.tokenizer(input_text, **tokenizer_kwargs).to(self.device)
        out = self.language_model(**tokenized)
        out = pooling(out, tokenized["attention_mask"])

        return out