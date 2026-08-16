#!/usr/bin/env python3
# End-to-end evaluation of an RVQ encoder checkpoint: predicts codes
# from a track's latents, replays them through mm-synth (teacher-forced
# LM bypass), and scores the render against the original audio with
# STFT magnitude cosine similarity and waveform correlation.
#
# The replay request keeps the track's true first code frame as the
# warm-up prefix (it renders no audio, the encoder never predicts it),
# followed by the predicted frames: N + 1 codes render N audio frames
# on the same timeline as the original.
#
# Reference points from the repaint sessions: exact codes render at
# 0.998 STFT, resampled codes of the same passage at 0.6-0.88.
#
# Usage: ./eval-e2e.py <dataset> <base> [--ckpt checkpoints/v1/best.pt]

import argparse
import json
import os
import subprocess
import wave

import numpy as np
import torch

from dataset import (CODES_PER_FRAME, FRAMES_PER_WIN, LATENT_CHANNELS, LATENT_WINDOW_MAX,
                     frame_latent_starts, load_codes, pool_matrix)
from model import RVQEncoder

ROOT         = os.path.join(os.path.dirname(__file__), "..")
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
MM_SYNTH     = os.path.join(ROOT, "build/mm-synth")
MODELS_DIR   = os.path.join(ROOT, "models")
EVAL_TMP     = "/tmp/rvq-eval"
STFT_WINDOW  = 2048
STFT_HOP     = 512
S16_SCALE    = 32768.0


def predict_codes(model, latents, starts, n_frames, device):
    # Full-track prediction over non-overlapping 128-frame windows on
    # the stitched timeline, argmax per head, output [n, 8]
    frames = []
    with torch.no_grad():
        for t0 in range(0, n_frames - FRAMES_PER_WIN + 1, FRAMES_PER_WIN):
            bounds = starts[t0 : t0 + FRAMES_PER_WIN + 1] - starts[t0]
            n_lat  = int(bounds[-1])
            lat    = np.zeros((LATENT_WINDOW_MAX, LATENT_CHANNELS), dtype=np.float32)
            lat[:n_lat] = latents[starts[t0] : starts[t0] + n_lat]
            pool = pool_matrix(bounds)
            sem_logits, ac_logits = model(torch.from_numpy(lat).unsqueeze(0).to(device),
                                          torch.from_numpy(pool).unsqueeze(0).to(device))
            cols = [sem_logits.argmax(-1)] + [l.argmax(-1) for l in ac_logits]
            frames.append(torch.stack(cols, dim=-1)[0].cpu().numpy())
    return np.concatenate(frames, axis=0)


def read_wav_left(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return data.reshape(-1, w.getnchannels())[:, 0].astype(np.float32) / S16_SCALE


def stft_mag(x: np.ndarray) -> np.ndarray:
    win    = np.hanning(STFT_WINDOW)
    n      = (len(x) - STFT_WINDOW) // STFT_HOP + 1
    frames = np.stack([x[i * STFT_HOP : i * STFT_HOP + STFT_WINDOW] * win for i in range(n)])
    return np.abs(np.fft.rfft(frames, axis=1))


def cossim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("base")
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(__file__), "checkpoints/v1/best.pt"))
    args = ap.parse_args()

    device = "cuda"
    model  = RVQEncoder().to(device).eval()
    state  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    print(f"[Eval] {args.ckpt} (epoch {state['epoch']}, val loss {state['val_loss']:.4f})")

    corpus_dir = os.path.join(DATASETS_DIR, args.dataset)
    latents = np.fromfile(os.path.join(corpus_dir, args.base + ".vae"),
                          dtype=np.float32).reshape(-1, LATENT_CHANNELS)
    truth    = load_codes(os.path.join(corpus_dir, args.base + ".json"))
    n_frames = truth.shape[0] - 1
    starts   = frame_latent_starts(n_frames)
    while n_frames > 0 and starts[n_frames] > latents.shape[0]:
        n_frames -= 1

    codes = predict_codes(model, latents, starts, n_frames, device)
    print(f"[Eval] {args.base}: {latents.shape[0]} latents -> {codes.shape[0]} predicted frames")

    n = codes.shape[0]
    sem_acc = float((codes[:, 0] == truth[1 : n + 1, 0]).mean())
    ac_acc  = float((codes[:, 1:] == truth[1 : n + 1, 1:]).mean())
    print(f"[Eval] Token accuracy (proxy only): sem {sem_acc:.3f}, ac {ac_acc:.3f}")

    with open(os.path.join(corpus_dir, args.base + ".json")) as f:
        request = json.load(f)
    replay_codes = np.concatenate([truth[:1], codes], axis=0)
    request["audio_codes"] = ",".join(str(v) for v in replay_codes.flatten())

    os.makedirs(EVAL_TMP, exist_ok=True)
    pred_json = os.path.join(EVAL_TMP, args.base + "-pred.json")
    pred_wav  = os.path.join(EVAL_TMP, args.base + "-pred.wav")
    with open(pred_json, "w") as f:
        json.dump(request, f)
    subprocess.run([MM_SYNTH, "--models", MODELS_DIR, "--request", pred_json, "--out", pred_wav], check=True)

    ref  = read_wav_left(os.path.join(corpus_dir, args.base + ".wav"))
    pred = read_wav_left(pred_wav)
    n    = min(len(ref), len(pred))
    ref, pred = ref[:n], pred[:n]
    stft = cossim(stft_mag(ref), stft_mag(pred))
    corr = float(np.corrcoef(ref, pred)[0, 1])
    print(f"[Eval] STFT cossim {stft:.4f}, waveform corr {corr:.4f} "
          f"(baselines: exact codes 0.998, resampled 0.6-0.88)")


if __name__ == "__main__":
    main()
