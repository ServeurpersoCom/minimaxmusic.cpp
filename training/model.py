#!/usr/bin/env python3
# RVQ encoder model, hiddens route: maps VAE latents to the LM
# per-frame hidden states h [4096]. The published lm_head and depth
# decoder audio_heads then turn h into the 8 codes per frame in closed
# form (semantic argmax, greedy acoustic chain), so this network never
# classifies: it regresses the state the language model would have
# carried while generating the audio.
#
# Shape flow: padded latents [B, L, 128] at 86.1328 Hz -> dilated conv
# stack -> per-sample mean pooling to [B, 128, d] at 25 Hz (the pooling
# matrix comes from the dataset, built on the stitched timeline) ->
# bidirectional transformer -> h [B, 128, 4096].

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_CHANNELS = 128
FRAMES_PER_WIN  = 128
H_DIM           = 4096
DILATIONS       = (1, 3, 9)


class ResBlock(nn.Module):
    def __init__(self, d: int, dilation: int):
        super().__init__()
        self.norm  = nn.GroupNorm(1, d)
        self.conv1 = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(d, d, 1)

    def forward(self, x):
        h = self.conv1(F.gelu(self.norm(x)))
        return x + self.conv2(F.gelu(h))


class HiddenEncoder(nn.Module):
    def __init__(self, d_model: int = 512, n_layers: int = 8, n_heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.conv_in = nn.Conv1d(LATENT_CHANNELS, d_model, 7, padding=3)
        self.blocks  = nn.ModuleList(ResBlock(d_model, dil) for dil in DILATIONS)
        self.pos     = nn.Parameter(torch.zeros(1, FRAMES_PER_WIN, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * ff_mult, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm_out    = nn.LayerNorm(d_model)
        self.head        = nn.Linear(d_model, H_DIM)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, latents, pool):
        # latents [B, L, 128] padded, pool [B, 128, L] -> h [B, 128, 4096]
        x = self.conv_in(latents.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        x = torch.bmm(pool, x.transpose(1, 2)) + self.pos
        return self.head(self.norm_out(self.transformer(x)))
