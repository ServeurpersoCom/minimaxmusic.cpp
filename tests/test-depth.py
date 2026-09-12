#!/usr/bin/env python3
"""Parity of the RVQ depth decoder, hidden states and codebook logits.

Owns both sides: builds the torch reference from the diffusers implementation
of the sibling clone ../../diffusers on a seeded projected sequence, runs the
GGML harness on the same sequence, compares. A sequence of S predicts
codebook S-1, so the logits come from the matching head.
Run from the tests/ directory.

Usage:
    ./test-depth.py
    GGML_BACKEND=CPU ./test-depth.py
"""
import os
import subprocess
import sys

sys.path.insert(0, "../../diffusers/src")

import numpy as np
import torch
from diffusers.models.transformers.minimax_music3_rvq_depth_decoder import (
    MiniMaxMusic3RVQDepthDecoder,
)
from diffusers.utils import logging as diffusers_logging

BIN = "../build/test-depth"
CKPT = "../checkpoints/rvq_depth_decoder"
GGUF = "../models/MiniMax-Music3-rvq_depth_decoder-BF16.gguf"
TMP = "tmp"
SEQ_LEN = 8
MAX_REL = 2e-2


def report(label, ref, got, max_rel, argmax=False):
    if ref.size != got.size:
        print("[Parity] %s FAIL size %d vs %d" % (label, ref.size, got.size))
        return False
    rel = float(np.sqrt(((ref - got) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-20))
    mx = float(np.abs(ref - got).max())
    line = "[Parity] %s: rel rms %.3e max abs %.3e" % (label, rel, mx)
    ok = rel <= max_rel
    if argmax:
        a, b = int(ref.argmax()), int(got.argmax())
        line += " argmax %d vs %d" % (a, b)
        ok = ok and a == b
    print(line + (" OK" if ok else " FAIL"))
    return ok


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TMP, exist_ok=True)
    diffusers_logging.disable_progress_bar()

    model = MiniMaxMusic3RVQDepthDecoder.from_pretrained(CKPT, torch_dtype=torch.float32)
    model.eval()

    torch.manual_seed(555)
    seq = torch.randn(1, SEQ_LEN, 4096)
    with torch.no_grad():
        hidden = model(model.projection(seq))
        logits = model.audio_heads[SEQ_LEN - 2](hidden[:, -1])

    seq.squeeze(0).contiguous().numpy().astype("float32").tofile(TMP + "/depth_seq.bin")
    ref_hidden = hidden.squeeze(0).contiguous().numpy().astype("float32").ravel()
    ref_logits = logits.squeeze(0).contiguous().numpy().astype("float32").ravel()

    subprocess.run([BIN, GGUF, TMP + "/depth_seq.bin", str(SEQ_LEN), TMP + "/depth_h.bin", TMP + "/depth_l.bin"],
                   check=True)

    ok = report("depth-hiddens", ref_hidden, np.fromfile(TMP + "/depth_h.bin", dtype="float32"), MAX_REL)
    ok = report("depth-logits", ref_logits, np.fromfile(TMP + "/depth_l.bin", dtype="float32"), MAX_REL,
                argmax=True) and ok
    sys.exit(0 if ok else 1)


main()
