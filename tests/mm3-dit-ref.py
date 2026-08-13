#!/usr/bin/env python3
"""Torch reference of the Music3 flow matching DiT.

Imports MiniMaxMusic3Transformer1DModel from the reference project ../../diffusers.
Dumps seeded random inputs and the predicted velocity for the GGML parity.
Run from tests/ directory. All paths relative to CWD.

Usage:
    ./mm3-dit-ref.py <checkpoint_dir> [t_latent] [timestep]
"""
import sys

sys.path.insert(0, "../../diffusers/src")

import torch
from diffusers.models.transformers.transformer_minimax_music3 import (
    MiniMaxMusic3Transformer1DModel,
)
from diffusers.utils import logging as diffusers_logging

diffusers_logging.disable_progress_bar()

model = MiniMaxMusic3Transformer1DModel.from_pretrained(sys.argv[1], torch_dtype=torch.float32)
model.eval()

T = int(sys.argv[2]) if len(sys.argv) > 2 else 24
t_val = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
torch.manual_seed(777)
xt = torch.randn(1, 128, T)
cond = torch.randn(1, T, 2048)
with torch.no_grad():
    vel = model(xt, torch.tensor([t_val]), cond, return_dict=False)[0]

# dumps are time-major to match the GGML input layout
xt.squeeze(0).T.contiguous().numpy().astype("float32").tofile("parity/dit_xt.bin")
cond.squeeze(0).contiguous().numpy().astype("float32").tofile("parity/dit_cond.bin")
vel.squeeze(0).T.contiguous().numpy().astype("float32").tofile("parity/dit_ref.bin")
print("xt", tuple(xt.shape), "t", t_val, "-> vel", tuple(vel.shape), "rms %.6f" % vel.pow(2).mean().sqrt().item())
