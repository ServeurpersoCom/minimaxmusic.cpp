#!/usr/bin/env python3
"""Parity of the flow matching DiT, one velocity evaluation.

Owns both sides: builds the torch reference from the diffusers implementation
of the sibling clone ../../diffusers on a seeded latent and condition track,
runs the GGML harness on the same inputs, compares.
Run from the tests/ directory.

Usage:
    ./test-dit.py
    GGML_BACKEND=CPU ./test-dit.py
"""
import os
import subprocess
import sys

sys.path.insert(0, "../../diffusers/src")

import numpy as np
import torch
from diffusers.models.transformers.transformer_minimax_music3 import (
    MiniMaxMusic3Transformer1DModel,
)
from diffusers.utils import logging as diffusers_logging

BIN = "../build/test-dit"
CKPT = "../checkpoints/transformer"
GGUF = "../models/MiniMax-Music3-transformer-F32.gguf"
TMP = "tmp"
T_LATENT = 24
TIMESTEP = 0.35
MAX_REL = 5e-2


def report(label, ref, got, max_rel):
    if ref.size != got.size:
        print("[Parity] %s FAIL size %d vs %d" % (label, ref.size, got.size))
        return False
    rel = float(np.sqrt(((ref - got) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-20))
    mx = float(np.abs(ref - got).max())
    ok = rel <= max_rel
    print("[Parity] %s: rel rms %.3e max abs %.3e %s" % (label, rel, mx, "OK" if ok else "FAIL"))
    return ok


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TMP, exist_ok=True)
    diffusers_logging.disable_progress_bar()

    model = MiniMaxMusic3Transformer1DModel.from_pretrained(CKPT, torch_dtype=torch.float32)
    model.eval()

    torch.manual_seed(777)
    xt = torch.randn(1, 128, T_LATENT)
    cond = torch.randn(1, T_LATENT, 2048)
    with torch.no_grad():
        vel = model(xt, torch.tensor([TIMESTEP]), cond, return_dict=False)[0]

    # dumps are time-major to match the GGML input layout
    xt.squeeze(0).T.contiguous().numpy().astype("float32").tofile(TMP + "/dit_xt.bin")
    cond.squeeze(0).contiguous().numpy().astype("float32").tofile(TMP + "/dit_cond.bin")
    ref = vel.squeeze(0).T.contiguous().numpy().astype("float32").ravel()

    subprocess.run([BIN, GGUF, TMP + "/dit_xt.bin", TMP + "/dit_cond.bin", str(T_LATENT), str(TIMESTEP),
                    TMP + "/dit.bin"], check=True)

    got = np.fromfile(TMP + "/dit.bin", dtype="float32")
    sys.exit(0 if report("dit", ref, got, MAX_REL) else 1)


main()
