#!/usr/bin/env python3
"""Torch reference of the Music3 RVQ depth decoder.

Imports MiniMaxMusic3RVQDepthDecoder from the reference project ../../diffusers.
Dumps a seeded random projected sequence, the hidden states, and the logits
of the matching codebook head (sequence of S predicts codebook S-1).
Run from tests/ directory. All paths relative to CWD.

Usage:
    ./mm3-depth-ref.py <checkpoint_dir> [seq_len]
"""
import sys

sys.path.insert(0, "../../diffusers/src")

import torch
from diffusers.models.transformers.minimax_music3_rvq_depth_decoder import (
    MiniMaxMusic3RVQDepthDecoder,
)
from diffusers.utils import logging as diffusers_logging

diffusers_logging.disable_progress_bar()

model = MiniMaxMusic3RVQDepthDecoder.from_pretrained(sys.argv[1], torch_dtype=torch.float32)
model.eval()

S = int(sys.argv[2]) if len(sys.argv) > 2 else 8
torch.manual_seed(555)
seq = torch.randn(1, S, 4096)
with torch.no_grad():
    hidden = model(model.projection(seq))
    logits = model.audio_heads[S - 2](hidden[:, -1])
seq.squeeze(0).contiguous().numpy().astype("float32").tofile("parity/depth_seq.bin")
hidden.squeeze(0).contiguous().numpy().astype("float32").tofile("parity/depth_hidden_ref.bin")
logits.squeeze(0).contiguous().numpy().astype("float32").tofile("parity/depth_logits_ref.bin")
print("seq", tuple(seq.shape), "-> hidden", tuple(hidden.shape), "logits head", S - 2)
