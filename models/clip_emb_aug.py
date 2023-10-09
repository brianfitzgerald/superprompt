import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPEncoder
import copy

class CLIPEmbeddingAugmenter(nn.Module):
    def __init__(self, clip_model: CLIPTextModel):
        super().__init__()
        self.unfrozen_clip_encoder: CLIPEncoder = copy.deepcopy(clip_model.text_model.encoder)
        self.unfrozen_clip_encoder.train()
        self.unfrozen_clip_encoder.requires_grad_(True)

    def forward(self, masked_emb: torch.Tensor):
        # x is the last hidden layer of clip text encoder
        emb_enc = self.unfrozen_clip_encoder(masked_emb).last_hidden_state
        return emb_enc

    def loss_fn(self, emb_enc: torch.Tensor, unmasked_emb: torch.Tensor):
        loss = F.mse_loss(emb_enc, unmasked_emb)
        return loss
