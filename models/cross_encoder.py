#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Union
from torch import Tensor


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
        out = self.pooling(out, tokenized["attention_mask"])

        return out

    def pooling(
        self, model_output: torch.tensor, attention_mask: torch.tensor
    ) -> torch.tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
