#!/usr/bin/env python3
"""Parity of the flow VAE decoder.

Owns both sides: builds the torch reference from the diffusers implementation
of the sibling clone ../../diffusers, runs the GGML harness on the same
latent, compares, exits non zero on failure.
Run from the tests/ directory.

Usage:
    ./test-vae.py
    GGML_BACKEND=CPU ./test-vae.py
"""
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, "../../diffusers/src")

import numpy as np
import torch
from diffusers.models.autoencoders.minimax_music3_vocoder import MiniMaxMusic3Vocoder
from diffusers.utils import logging as diffusers_logging

BIN = "../build/test-vae"
CKPT = "../checkpoints/vocoder"
GGUF = "../models/MiniMax-Music3-vocoder-F32.gguf"
TMP = "tmp"
T_LATENT = 64
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

    model = MiniMaxMusic3Vocoder.from_pretrained(CKPT, torch_dtype=torch.float32)
    model.eval()

    torch.manual_seed(42)
    latent = torch.randn(1, 128, T_LATENT)
    with torch.no_grad():
        audio = model(latent)

    # audio interleaved stereo [T, 2] to match the harness output
    latent.numpy().astype("float32").tofile(TMP + "/latent.bin")
    ref = audio.squeeze(0).T.contiguous().numpy().astype("float32").ravel()

    subprocess.run([BIN, GGUF, TMP + "/latent.bin", str(T_LATENT), TMP + "/audio.bin"], check=True)

    got = np.fromfile(TMP + "/audio.bin", dtype="float32")
    sys.exit(0 if report("vae", ref, got, MAX_REL) else 1)


main()
