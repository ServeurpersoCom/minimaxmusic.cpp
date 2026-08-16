#!/usr/bin/env python3
# RVQ encoder model: maps VAE latents to MiniMax Music 3 code frames.
# The codebooks belong to the frozen model, so this is plain supervised
# classification with 8 independent heads per frame: one semantic head
# over 16384 classes and seven acoustic heads over 1024 classes each.
#
# Shape flow: padded latents [B, L, 128] at 86.1328 Hz -> dilated conv
# stack -> per-sample mean pooling to [B, 128, d] at 25 Hz (the pooling
# matrix comes from the dataset, built on the stitched timeline) ->
# bidirectional transformer -> logits per head.

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_CHANNELS  = 128
FRAMES_PER_WIN   = 128
SEMANTIC_CLASSES = 16384
ACOUSTIC_CLASSES = 1024
ACOUSTIC_HEADS   = 7
DILATIONS        = (1, 3, 9)


class ResBlock(nn.Module):
    def __init__(self, d: int, dilation: int):
        super().__init__()
        self.norm  = nn.GroupNorm(1, d)
        self.conv1 = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(d, d, 1)

    def forward(self, x):
        h = self.conv1(F.gelu(self.norm(x)))
        return x + self.conv2(F.gelu(h))


class RVQEncoder(nn.Module):
    def __init__(self, d_model: int = 512, n_layers: int = 8, n_heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.conv_in = nn.Conv1d(LATENT_CHANNELS, d_model, 7, padding=3)
        self.blocks  = nn.ModuleList(ResBlock(d_model, dil) for dil in DILATIONS)
        self.pos     = nn.Parameter(torch.zeros(1, FRAMES_PER_WIN, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * ff_mult, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm_out    = nn.LayerNorm(d_model)
        self.head_sem    = nn.Linear(d_model, SEMANTIC_CLASSES)
        self.heads_ac    = nn.ModuleList(nn.Linear(d_model, ACOUSTIC_CLASSES) for _ in range(ACOUSTIC_HEADS))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, latents, pool):
        # latents [B, L, 128] padded, pool [B, 128, L] -> logits:
        # semantic [B, 128, 16384], acoustic list of 7 x [B, 128, 1024]
        x = self.conv_in(latents.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        x = torch.bmm(pool, x.transpose(1, 2)) + self.pos
        x = self.norm_out(self.transformer(x))
        return self.head_sem(x), [head(x) for head in self.heads_ac]
