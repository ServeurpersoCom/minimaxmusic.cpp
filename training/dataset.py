#!/usr/bin/env python3
# Dataset for the RVQ encoder training: pairs latent windows with code
# targets from a generated corpus directory. Each sample is one window
# of 128 LM frames (25 Hz) with the VAE latents (86.1328 Hz) it covers
# and a per-sample mean-pooling matrix.
#
# Two timeline facts drive the indexing:
# - The rendered WAV lives on the stitched timeline: the pipeline
#   denoises 200-frame windows hopped by 100 frames and stitches them
#   with an integer hop of 345 latents, so the global frame-to-latent
#   mapping is piecewise per DiT window, not t * 441 / 128.
# - The code stream carries one frame more than the audio: the first
#   code frame only warms up the LM feedback and renders no sound, so
#   audio frame j pairs with codes[j + 1].
#
# Corpus layout (per track): <base>.vae flat [T, 128] f32 latents and
# <base>.json replay request whose audio_codes field is a flat CSV of
# 8 * n values, frame = 8 consecutive [semantic, ac1..ac7].

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

LATENT_CHANNELS   = 128
FRAMES_PER_WIN    = 128
LATENT_WINDOW_MAX = 448  # 441 latents per 128 frames plus stitch seam drift, padded
CODES_PER_FRAME   = 8
LATENT_BYTES      = LATENT_CHANNELS * 4
RATIO_NUM         = 441  # latents per RATIO_DEN frames, (44100/512) / 25 exactly
RATIO_DEN         = 128
CHUNK_FRAMES      = 200  # DiT denoising window
CHUNK_HOP         = 100
HOP_LATENTS       = 345  # integer latent hop of the stitched timeline
OWNED_FROM        = 25   # first frame owned by a non-first window, ceil(86 * 128 / 441)
VAE_EXT           = ".vae"
JSON_EXT          = ".json"


def n_dit_windows(n_frames: int) -> int:
    # Chunk starts run 0, 100, ... while start < n_frames - CHUNK_HOP
    return max(1, (n_frames - 1) // CHUNK_HOP)


def frame_latent_starts(n_frames: int) -> np.ndarray:
    # Latent start of every frame boundary on the stitched timeline:
    # frame t is owned by DiT window k, its start is the window's
    # integer stitch offset plus the ceil boundary of the local frame.
    # The condition encoder nearest-interpolates each chunk of F frames
    # to L = floor(F * 441 / 128) latents, so the exact inverse of its
    # frame assignment is ceil(tau * L / F) with the chunk's own L and
    # F; the distinction only bites on the partial last chunk. Returns
    # n_frames + 1 boundaries.
    t   = np.arange(n_frames + 1, dtype=np.int64)
    k   = np.clip((t - OWNED_FROM) // CHUNK_HOP, 0, n_dit_windows(n_frames) - 1)
    tau = t - k * CHUNK_HOP
    F   = np.minimum(CHUNK_FRAMES, n_frames - k * CHUNK_HOP)
    L   = F * RATIO_NUM // RATIO_DEN
    return k * HOP_LATENTS + (tau * L + F - 1) // F


def pool_matrix(bounds: np.ndarray) -> np.ndarray:
    # Mean over the latents each frame covers, [128, LATENT_WINDOW_MAX],
    # bounds = 129 window-relative latent boundaries
    pool = np.zeros((FRAMES_PER_WIN, LATENT_WINDOW_MAX), dtype=np.float32)
    for j in range(FRAMES_PER_WIN):
        a, b = int(bounds[j]), int(bounds[j + 1])
        pool[j, a:b] = 1.0 / (b - a)
    return pool


def load_codes(path: str) -> np.ndarray:
    with open(path) as f:
        vals = np.array(json.load(f)["audio_codes"].split(","), dtype=np.int64)
    assert vals.size % CODES_PER_FRAME == 0
    return vals.reshape(-1, CODES_PER_FRAME)


class CorpusDataset(Dataset):
    def __init__(self, corpus_dir: str, bases: list[str] | None = None):
        if bases is None:
            bases = sorted(f[: -len(VAE_EXT)] for f in os.listdir(corpus_dir) if f.endswith(VAE_EXT))
        self.windows = []  # (vae_path, codes [n, 8], starts [n_frames + 1], start_frame)
        for base in bases:
            vae_path = os.path.join(corpus_dir, base + VAE_EXT)
            codes    = load_codes(os.path.join(corpus_dir, base + JSON_EXT))
            n_lat    = os.path.getsize(vae_path) // LATENT_BYTES
            n_frames = codes.shape[0] - 1
            starts   = frame_latent_starts(n_frames)
            while n_frames > 0 and starts[n_frames] > n_lat:
                n_frames -= 1
            for t0 in range(0, n_frames - FRAMES_PER_WIN + 1, FRAMES_PER_WIN):
                self.windows.append((vae_path, codes, starts, t0))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        vae_path, codes, starts, t0 = self.windows[i]
        bounds = starts[t0 : t0 + FRAMES_PER_WIN + 1] - starts[t0]
        n_lat  = int(bounds[-1])
        assert n_lat <= LATENT_WINDOW_MAX
        latents = np.zeros((LATENT_WINDOW_MAX, LATENT_CHANNELS), dtype=np.float32)
        latents[:n_lat] = np.fromfile(vae_path, dtype=np.float32, count=n_lat * LATENT_CHANNELS,
                                      offset=int(starts[t0]) * LATENT_BYTES).reshape(n_lat, LATENT_CHANNELS)
        target = codes[t0 + 1 : t0 + 1 + FRAMES_PER_WIN]
        return (torch.from_numpy(latents), torch.from_numpy(pool_matrix(bounds)),
                torch.from_numpy(target.copy()))
