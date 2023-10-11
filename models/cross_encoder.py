#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple, Union
from torch import Tensor

class CrossEncoder(nn.Module):
    
    def __init__(self, device: torch.device) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        self.language_model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base")
        self.embedding_dimension = self.language_model.config.hidden_size
        self.device = device
        self.pooling = Pooling(device, self.embedding_dimension)

    def forward(self, subject_text: str):
        subject_tokenized = self.tokenizer(subject_text, return_tensors="pt").to(self.device)
        x = self.language_model(**subject_tokenized)
        x = self.pooling(x)

        
class Pooling(nn.Module):
    def __init__(self, device: torch.device, embedding_dimension) -> None:
        self.device = device
        self.pooling_output_dimension = embedding_dimension
 
    def forward(self, features: Dict[str, Tensor]):   
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
