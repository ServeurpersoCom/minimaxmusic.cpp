#!/usr/bin/env python3
# End-to-end evaluation of the hiddens-route encoder: predicts h per
# frame from a track's latents, decodes the 8 codes per frame in
# closed form with the published heads (lm_head argmax for the
# semantic, depth decoder greedy chain for the acoustics), replays the
# codes through mm-synth (teacher-forced LM bypass), and scores the
# render against the original audio with STFT magnitude cosine
# similarity. The replay keeps the track's true first code frame as the
# warm-up prefix: it renders no audio and the encoder never predicts it.
#
# Runs on the training venv (torch + transformers + diffusers).
#
# Usage: ./eval-e2e.py <dataset> <base> [--ckpt checkpoints/v3/best.pt]

import argparse
import json
import os
import subprocess
import wave

import numpy as np
import torch

from dataset import (FRAMES_PER_WIN, H_DIM, HID_EXT, JSON_EXT, VAE_EXT, LATENT_CHANNELS,
                     build_window, frame_latent_starts, load_codes)
from model import HiddenEncoder

ROOT         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
CKPT         = os.path.join(ROOT, "checkpoints")
MM_SYNTH     = os.path.join(ROOT, "build/mm-synth")
MODELS_DIR   = os.path.join(ROOT, "models")
EVAL_TMP     = "/tmp/rvq-eval"
DEVICE       = "cuda"
DTYPE        = torch.bfloat16
SEM_OFF      = 151675
SEM_N        = 16384
AC_N         = 1024
STFT_WINDOW  = 2048
STFT_HOP     = 512
S16_SCALE    = 32768.0


def predict_h(model, vae_path, starts, n_frames):
    # Full-track h prediction over non-overlapping 128-frame windows on
    # the stitched timeline
    outs = []
    with torch.no_grad():
        for t0 in range(0, n_frames - FRAMES_PER_WIN + 1, FRAMES_PER_WIN):
            latents, pool = build_window(vae_path, starts, t0)
            outs.append(model(latents.unsqueeze(0).to(DEVICE), pool.unsqueeze(0).to(DEVICE))[0].float())
    return torch.cat(outs, dim=0)  # [n, 4096]


def decode_codes(lm, depth, h):
    # Closed-form readout of the published heads: semantic by lm_head
    # argmax over the audio code range, acoustics by the depth decoder
    # greedy chain within each frame, frames batched
    sem_table = lm.get_input_embeddings().weight
    ac_table  = depth.audio_embeddings.weight.view(7, AC_N, -1)
    with torch.no_grad():
        sem = lm.lm_head(h.to(DTYPE)).float()[:, SEM_OFF : SEM_OFF + SEM_N].argmax(-1)
        seq = [depth.projection(h.to(DTYPE)).unsqueeze(1),
               depth.projection(sem_table[sem + SEM_OFF].to(DTYPE)).unsqueeze(1)]
        ac = []
        for k in range(7):
            d = depth(torch.cat(seq, dim=1))[:, -1]
            code = depth.audio_heads[k](d).float().argmax(-1)
            ac.append(code)
            if k < 6:
                seq.append(depth.projection(ac_table[k][code].to(DTYPE)).unsqueeze(1))
    return sem, torch.stack(ac, dim=1)


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
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(__file__), "checkpoints/v3/best.pt"))
    args = ap.parse_args()

    corpus   = os.path.join(DATASETS_DIR, args.dataset)
    vae_path = os.path.join(corpus, args.base + VAE_EXT)
    codes    = torch.from_numpy(load_codes(os.path.join(corpus, args.base + JSON_EXT))).to(DEVICE)
    with open(os.path.join(corpus, args.base + JSON_EXT)) as f:
        request = json.load(f)
    n_frames = codes.shape[0] - 1
    starts   = frame_latent_starts(n_frames)
    n_lat    = os.path.getsize(vae_path) // (LATENT_CHANNELS * 4)
    while n_frames > 0 and starts[n_frames] > n_lat:
        n_frames -= 1

    model = HiddenEncoder().to(DEVICE).eval()
    state = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(state["model"])
    print(f"[Eval] {args.ckpt} (epoch {state['epoch']}, val loss {state['val_loss']:.4f})", flush=True)

    h_hat = predict_h(model, vae_path, starts, n_frames)
    n = h_hat.shape[0]

    hid_path = os.path.join(corpus, args.base + HID_EXT)
    if os.path.exists(hid_path):
        true_h = np.fromfile(hid_path, dtype=np.float16, count=n * H_DIM).reshape(n, H_DIM)
        hcos = torch.nn.functional.cosine_similarity(
            h_hat, torch.from_numpy(true_h.astype(np.float32)).to(DEVICE), dim=-1).mean().item()
        print(f"[Eval] h cossim vs true hiddens: {hcos:.4f}", flush=True)

    from transformers import Qwen3ForCausalLM
    from diffusers.models.transformers.minimax_music3_rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder
    lm = Qwen3ForCausalLM.from_pretrained(CKPT + "/language_model", torch_dtype=DTYPE).to(DEVICE)
    depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(CKPT + "/rvq_depth_decoder",
                                                         torch_dtype=DTYPE).to(DEVICE)
    lm.requires_grad_(False).eval()
    depth.requires_grad_(False).eval()

    sem, ac = decode_codes(lm, depth, h_hat)
    decoded = codes.clone()
    decoded[1 : n + 1, 0]  = sem
    decoded[1 : n + 1, 1:] = ac

    os.makedirs(EVAL_TMP, exist_ok=True)
    request["audio_codes"] = ",".join(str(int(v)) for v in decoded.cpu().numpy().flatten())
    pred_json = os.path.join(EVAL_TMP, args.base + "-pred.json")
    pred_wav  = os.path.join(EVAL_TMP, args.base + "-pred.wav")
    with open(pred_json, "w") as f:
        json.dump(request, f)
    subprocess.run([MM_SYNTH, "--models", MODELS_DIR, "--request", pred_json, "--out", pred_wav], check=True)

    ref, pred = read_wav_left(os.path.join(corpus, args.base + ".wav")), read_wav_left(pred_wav)
    m = min(len(ref), len(pred))
    print(f"[Eval] STFT cossim {cossim(stft_mag(ref[:m]), stft_mag(pred[:m])):.4f} "
          f"(exact codes 0.998, true-hidden decode 0.896)", flush=True)


if __name__ == "__main__":
    main()
