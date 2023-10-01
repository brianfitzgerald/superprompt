import torch
import torch.nn as nn
import os
from typing import List
import torch.nn.functional as F


class EmbeddingAugModel(nn.Module):
    def __init__(
        self,
        context_dim: int = 512,
        layers: List[int] = [768, 1024, 1024, 768],
        device="cpu",
    ):
        super().__init__()

        self.dropout = nn.Dropout(0.1)
        fc_layers = []
        fc_layers.append(torch.nn.Linear(context_dim, layers[0]))
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            fc_layers.append(torch.nn.Linear(in_size, out_size))
            fc_layers.append(torch.nn.ReLU())
        fc_layers.append(torch.nn.Linear(layers[-1], context_dim))
        self.embed_fc = torch.nn.Sequential(*fc_layers).to(device)
        self.bn = nn.BatchNorm1d(context_dim).to(device)

    def forward(self, x):
        if self.training:
            x = self.dropout(x)
        x = self.embed_fc(x)
        return x

    def grad_norm_sum(self):
        grads = [
            param.grad.detach().flatten()
            for param in self.parameters()
            if param.grad is not None and param.requires_grad
        ]
        total_norm = torch.cat(grads).norm()
        return total_norm.item()

    def weights_biases_sum(self):
        total_weight_sum = 0.0
        for param in self.parameters():
            total_weight_sum += param.data.sum().item()
        return total_weight_sum
