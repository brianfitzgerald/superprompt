import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPEncoder, CLIPEncoderLayer
import copy

class CLIPEmbeddingAugmenter(nn.Module):
    def __init__(self, clip_model: CLIPTextModel):
        super().__init__()
        self.unfrozen_encoder_layer: CLIPEncoder = CLIPEncoderLayer(copy.deepcopy(clip_model.config))
        self.unfrozen_encoder_layer.to(clip_model.device)
        self.unfrozen_encoder_layer.train()
        self.unfrozen_encoder_layer.requires_grad_(True)

    def forward(self, masked_emb: torch.Tensor):
        # x is the last hidden layer of clip text encoder
        emb_enc = self.unfrozen_encoder_layer(masked_emb, None, None)[0]
        return emb_enc
