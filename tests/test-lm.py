#!/usr/bin/env python3
"""Parity of the global LM, prefill and decode.

Owns both sides: builds the torch reference with Qwen3ForCausalLM from the
sibling clone ../../transformers, runs the GGML harness on the same prompt,
compares the last token logits and the argmax.
Run from the tests/ directory.

Usage:
    ./test-lm.py
    GGML_BACKEND=CPU ./test-lm.py
"""
import os
import subprocess
import sys

sys.path.insert(0, "../../transformers/src")

import numpy as np
import torch
from transformers import Qwen3ForCausalLM
from transformers.utils import logging as hf_logging

BIN = "../build/test-lm"
CKPT = "../checkpoints/language_model"
GGUF = "../models/MiniMax-Music3-language_model-BF16.gguf"
TMP = "tmp"
MAX_REL = 2e-2

# A short prompt ending on the caption start token
IDS = [8948, 3837, 374, 264, 6543, 11, 279, 151671, 3703, 92]


def report(label, ref, got, max_rel):
    if ref.size != got.size:
        print("[Parity] %s FAIL size %d vs %d" % (label, ref.size, got.size))
        return False
    rel = float(np.sqrt(((ref - got) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-20))
    mx = float(np.abs(ref - got).max())
    a, b = int(ref.argmax()), int(got.argmax())
    ok = rel <= max_rel and a == b
    print("[Parity] %s: rel rms %.3e max abs %.3e argmax %d vs %d %s" % (label, rel, mx, a, b, "OK" if ok else "FAIL"))
    return ok


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TMP, exist_ok=True)
    hf_logging.disable_progress_bar()

    model = Qwen3ForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        # The harness prefills every id but the last, then decodes that one
        ref = {"lm-prefill": model(input_ids=torch.tensor([IDS[:-1]])).logits[0, -1].numpy().astype("float32"),
               "lm-decode": model(input_ids=torch.tensor([IDS])).logits[0, -1].numpy().astype("float32")}

    subprocess.run([BIN, GGUF, TMP + "/lm"] + [str(i) for i in IDS], check=True)

    ok = True
    for label, path in (("lm-prefill", "/lm_prefill_logits.bin"), ("lm-decode", "/lm_decode_logits.bin")):
        got = np.fromfile(TMP + path, dtype="float32")
        ok = report(label, ref[label], got, MAX_REL) and ok

    sys.exit(0 if ok else 1)


main()
