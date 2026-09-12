#!/usr/bin/env python3
"""Parity of the Philox noise against torch.randn on CUDA in bfloat16.

Owns both sides: runs the C++ dumper on a set of fixed seeds, draws the same
stream on a CUDA generator, and compares. The DiT takes its initial noise from
this stream, so a seed only reproduces a song if it matches.

The draw is sized on one DiT window. Past a few hundred thousand elements the
CUDA kernel of torch stops mapping one subsequence per element, and the two
streams diverge by construction; the bound depends on the device.
Run from the tests/ directory.

Usage:
    ./test-philox.py
"""
import os
import subprocess
import sys

import numpy as np
import torch

BIN = "../build/test-philox"
TMP = "tmp"
COUNT = 128 * 689  # 128 latent channels over one window of 689 frames
SEEDS = [42, 1234, 831001, 20260912]


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TMP, exist_ok=True)

    if not torch.cuda.is_available():
        print("[Parity] philox FAIL: CUDA required for the reference generator")
        sys.exit(1)

    ok = True
    for seed in SEEDS:
        subprocess.run([BIN, str(seed), str(COUNT), TMP + "/philox.f32"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        got = np.fromfile(TMP + "/philox.f32", dtype="float32")

        generator = torch.Generator(device="cuda").manual_seed(seed)
        ref = torch.randn([COUNT], generator=generator, device="cuda", dtype=torch.bfloat16)
        ref = ref.float().cpu().numpy()

        exact = int((ref == got).sum())
        cos = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got)))
        verdict = "PERFECT" if exact == got.size else "OK" if cos > 0.9999 else "FAIL"
        print("[Parity] philox seed %d: %d/%d exact, cos %.8f %s" % (seed, exact, got.size, cos, verdict))
        ok = verdict != "FAIL" and ok

    sys.exit(0 if ok else 1)


main()
