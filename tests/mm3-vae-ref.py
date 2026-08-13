#!/usr/bin/env python3
"""Torch reference of the Music3 flow VAE decoder.

Imports MiniMaxMusic3Vocoder from the reference project ../../diffusers.
Dumps a seeded random latent and the decoded audio for the GGML parity.
Run from tests/ directory. All paths relative to CWD.

Usage:
    ./mm3-vae-ref.py <checkpoint_dir> [t_latent]
"""
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, "../../diffusers/src")

import torch
from diffusers.models.autoencoders.minimax_music3_vocoder import MiniMaxMusic3Vocoder
from diffusers.utils import logging as diffusers_logging

diffusers_logging.disable_progress_bar()


def main():
    ckpt = sys.argv[1]
    t_latent = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    model = MiniMaxMusic3Vocoder.from_pretrained(ckpt, torch_dtype=torch.float32)
    model.eval()

    torch.manual_seed(42)
    latent = torch.randn(1, 128, t_latent)

    with torch.no_grad():
        audio = model(latent)

    # audio dumped interleaved stereo [T, 2] to match the harness output
    latent.numpy().astype("float32").tofile("parity/latent.bin")
    audio.squeeze(0).T.contiguous().numpy().astype("float32").tofile("parity/audio_ref.bin")
    print("latent", tuple(latent.shape), "-> audio", tuple(audio.shape))
    print("audio stats: min %.6f max %.6f rms %.6f" % (
        audio.min().item(), audio.max().item(), audio.pow(2).mean().sqrt().item()))


if __name__ == "__main__":
    main()
