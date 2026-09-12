#!/usr/bin/env python3
"""Parity of the condition encoder.

Owns both sides: builds the torch reference from the diffusers implementation
of the sibling clone ../../diffusers on a seeded block of fused hidden states,
runs the GGML harness on the same block, compares.
Run from the tests/ directory.

Usage:
    ./test-cond.py
    GGML_BACKEND=CPU ./test-cond.py
"""
import os
import subprocess
import sys

sys.path.insert(0, "../../diffusers/src")

import numpy as np
import torch
from diffusers.models.condition_embedders.condition_embedder_minimax_music3 import (
    MiniMaxMusic3ConditionEncoder,
)
from diffusers.utils import logging as diffusers_logging

BIN = "../build/test-cond"
CKPT = "../checkpoints/condition_encoder"
GGUF = "../models/MiniMax-Music3-condition_encoder-F32.gguf"
TMP = "tmp"
FRAMES = 7
MAX_REL = 1e-2


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

    model = MiniMaxMusic3ConditionEncoder.from_pretrained(CKPT, torch_dtype=torch.float32)
    model.eval()

    torch.manual_seed(123)
    hidden = torch.randn(1, FRAMES, 8 * 4096)
    with torch.no_grad():
        cond = model(hidden)

    hidden.numpy().astype("float32").tofile(TMP + "/cond_hidden.bin")
    ref = cond.numpy().astype("float32").ravel()

    subprocess.run([BIN, GGUF, TMP + "/cond_hidden.bin", str(FRAMES), TMP + "/cond.bin"], check=True)

    got = np.fromfile(TMP + "/cond.bin", dtype="float32")
    sys.exit(0 if report("cond", ref, got, MAX_REL) else 1)


main()
