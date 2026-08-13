#!/usr/bin/env python3
"""Torch reference of the Music3 condition encoder.

Imports MiniMaxMusic3ConditionEncoder from the reference project ../../diffusers.
Dumps a seeded random hidden state block and the conditioning output.
Run from tests/ directory. All paths relative to CWD.

Usage:
    ./mm3-cond-ref.py <checkpoint_dir> [n_frames]
"""
import sys

sys.path.insert(0, "../../diffusers/src")

import torch
from diffusers.models.condition_embedders.condition_embedder_minimax_music3 import (
    MiniMaxMusic3ConditionEncoder,
)
from diffusers.utils import logging as diffusers_logging

diffusers_logging.disable_progress_bar()

model = MiniMaxMusic3ConditionEncoder.from_pretrained(sys.argv[1], torch_dtype=torch.float32)
model.eval()

T = int(sys.argv[2]) if len(sys.argv) > 2 else 7
torch.manual_seed(123)
hidden = torch.randn(1, T, 8 * 4096)
with torch.no_grad():
    cond = model(hidden)
hidden.numpy().astype("float32").tofile("parity/cond_hidden.bin")
cond.numpy().astype("float32").tofile("parity/cond_ref.bin")
print("hidden", tuple(hidden.shape), "-> cond", tuple(cond.shape))
