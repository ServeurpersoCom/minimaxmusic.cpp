#!/usr/bin/env python3
"""GGML vs Python cosine similarity comparison for the MiniMax Music 3 DiT.

Run from tests/ directory. All paths relative to CWD.

The GGML side runs the full mm-synth pipeline with --dump. The Python side
reloads the dumped window-0 condition track and initial noise, then runs the
reference diffusers transformer and vocoder (../../diffusers, CUDA float32)
through the same Euler CFG schedule, with hooks mirroring the GGML probe
names. Sharing the condition and noise isolates the denoising stack from the
stochastic AR stage, whose logits are covered by parity.sh.

Usage:
    cd tests/
    ./debug-dit-cossim.py                   # DiT BF16
    ./debug-dit-cossim.py --quant Q6_K      # DiT Q6_K

Backend follows GGML_BACKEND; the archive loop lives in debug-dit-cossim.sh.
"""
import argparse
import json
import warnings
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
KEY_LAYERS = [0, 6, 12, 18, 35]


def save_dump(path, data):
    import torch

    if isinstance(data, torch.Tensor):
        data = data.detach().float().cpu().numpy()
    data = np.ascontiguousarray(data.astype(np.float32))
    shape = data.shape
    header = struct.pack("i", len(shape))
    for s in shape:
        header += struct.pack("i", s)
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


def load_dump(path):
    raw = np.fromfile(path, dtype=np.float32)
    ndim = int(struct.unpack("i", struct.pack("f", raw[0]))[0])
    shape = [int(struct.unpack("i", struct.pack("f", raw[1 + i]))[0]) for i in range(ndim)]
    data = raw[1 + ndim:]
    return data, shape


def _cos_flat(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0


def cos(a, b, shape_a=None, shape_b=None):
    if shape_a and shape_b and len(shape_a) == 2 and len(shape_b) == 2:
        if shape_a[0] == shape_b[1] and shape_a[1] == shape_b[0]:
            ra = a.reshape(shape_a)
            rb = b.reshape(shape_b)
            c_normal = _cos_flat(ra.flatten(), rb.flatten())
            c_transposed = _cos_flat(ra.T.flatten(), rb.flatten())
            if c_transposed > c_normal:
                return c_transposed
            return c_normal
    return _cos_flat(a, b)


def stft_cos(a, b, win=2048, hop=512):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    window = np.hanning(win)
    frames = (n - win) // hop + 1
    sa = np.zeros((frames, win // 2 + 1))
    sb = np.zeros((frames, win // 2 + 1))
    for i in range(frames):
        s = i * hop
        sa[i] = np.abs(np.fft.rfft(a[s:s + win] * window))
        sb[i] = np.abs(np.fft.rfft(b[s:s + win] * window))
    return _cos_flat(sa.flatten(), sb.flatten())


# GGML runner

def run_ggml(dump_dir, req, quant):
    ggml_bin = "../build/mm-synth"
    if not os.path.isfile(ggml_bin):
        print(f"[GGML] binary not found: {ggml_bin}")
        return False
    os.makedirs(dump_dir, exist_ok=True)

    merged = dict(req)
    merged["seed"] = SEED
    merged["lm_seed"] = SEED
    merged["output_format"] = "wav16"
    merged["dit_model"] = f"MiniMax-Music3-transformer-{quant}.gguf"
    merged["lm_model"] = "MiniMax-Music3-language_model-BF16.gguf"
    merged["depth_model"] = "MiniMax-Music3-rvq_depth_decoder-BF16.gguf"
    merged["cond_model"] = "MiniMax-Music3-condition_encoder-F32.gguf"
    merged["vae_model"] = "MiniMax-Music3-vocoder-F32.gguf"

    request_json = os.path.join(dump_dir, "request0.json")
    with open(request_json, "w") as f:
        json.dump(merged, f, indent=4)

    cmd = [ggml_bin, "--models", "../models", "--request", request_json,
           "--dump", dump_dir, "--no-fa", "--out", os.path.join(dump_dir, "output.wav")]
    print(f"[GGML] Running MiniMax-Music3-transformer-{quant}.gguf...")
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    n = len([f for f in os.listdir(dump_dir) if f.endswith(".bin")])
    if r.returncode != 0:
        print(f"[GGML] FAILED (exit {r.returncode})")
        return False
    print(f"[GGML] Done, {n} dump files")
    return True


# Python runner

def run_python(dump_dir, ggml_dir, req):
    sys.path.insert(0, "../../diffusers/src")
    import torch
    from diffusers.models.autoencoders.minimax_music3_vocoder import MiniMaxMusic3Vocoder
    from diffusers.models.transformers.transformer_minimax_music3 import MiniMaxMusic3Transformer1DModel
    from diffusers.utils import logging as diffusers_logging

    diffusers_logging.disable_progress_bar()
    os.makedirs(dump_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cond_raw, cond_shape = load_dump(os.path.join(ggml_dir, "window0_cond.bin"))
    noise_raw, noise_shape = load_dump(os.path.join(ggml_dir, "noise.bin"))
    T = cond_shape[0]
    cond = torch.from_numpy(cond_raw.reshape(cond_shape)).unsqueeze(0).to(device)  # [1, T, 2048]
    xt = torch.from_numpy(noise_raw.reshape(noise_shape)).T.unsqueeze(0).contiguous().to(device)  # [1, 128, T]
    zeros_cond = torch.zeros_like(cond)

    steps = int(req.get("steps", 30))
    cfg = float(req.get("dit_cfg", 1.7))

    print(f"[Python] Initializing transformer on {device} (T={T}, {steps} steps, CFG={cfg})...")
    model = MiniMaxMusic3Transformer1DModel.from_pretrained("../checkpoints/transformer", torch_dtype=torch.float32)
    model.eval().to(device)

    _dumps = {}
    _call = [0]
    _hooks = []

    def make_hook(name, transpose_ct=False, add_input=False):
        def hook(module, input, output):
            if _call[0] != 0:
                return
            out = output[0] if isinstance(output, tuple) else output
            if add_input:
                out = out + input[0]
            t = out[0].clone().float()
            if transpose_ct:
                t = t.T
            _dumps[name] = t
        return hook

    _hooks.append(model.time_embed.register_forward_hook(make_hook("temb_t")))
    _hooks.append(model.preprocess_conv.register_forward_hook(
        make_hook("hidden_after_preprocess", transpose_ct=True, add_input=True)))
    _hooks.append(model.proj_in.register_forward_hook(make_hook("hidden_after_proj_in")))
    _hooks.append(model.transformer_blocks[0].attn.register_forward_hook(make_hook("layer0_sa_output")))
    for li in KEY_LAYERS:
        _hooks.append(model.transformer_blocks[li].register_forward_hook(make_hook(f"hidden_after_layer{li}")))

    # Euler CFG schedule identical to pipeline.cpp: ascending sigmas + final 1.0
    sig = list(1.0 - np.linspace(1.0, 1.0 / steps, steps)) + [1.0]

    print("[Python] Denoising...")
    with torch.no_grad():
        for i in range(steps):
            t = torch.tensor([sig[i]], dtype=torch.float32, device=device)
            vt_cond = model(xt, t, cond, return_dict=False)[0]
            _call[0] += 1
            vt_uncond = model(xt, t, zeros_cond, return_dict=False)[0]
            _call[0] += 1
            vt = vt_uncond + (vt_cond - vt_uncond) * cfg
            xt = xt + (sig[i + 1] - sig[i]) * vt
            _dumps[f"dit_step{i}_vt_cond"] = vt_cond[0].T
            if i < 2:
                _dumps[f"dit_step{i}_vt_uncond"] = vt_uncond[0].T
            _dumps[f"dit_step{i}_vt"] = vt[0].T
            _dumps[f"dit_step{i}_xt"] = xt[0].T
    _dumps["dit_x0"] = xt[0].T
    _dumps["noise"] = torch.from_numpy(noise_raw.reshape(noise_shape))

    for h in _hooks:
        h.remove()

    print("[Python] Decoding audio...")
    vocoder = MiniMaxMusic3Vocoder.from_pretrained("../checkpoints/vocoder", torch_dtype=torch.float32)
    vocoder.eval().to(device)
    with torch.no_grad():
        audio = vocoder(xt)  # [1, 2, N]
    _dumps["vae_audio"] = audio.squeeze(0).T  # [N, 2] interleaved like the GGML dump

    for name, tensor in sorted(_dumps.items()):
        save_dump(os.path.join(dump_dir, f"{name}.bin"), tensor)
    print(f"[Python] Done, {len(_dumps)} dump files")
    return True


# comparison

def build_stages(steps):
    stages = ["noise", "temb_t", "hidden_after_preprocess", "hidden_after_proj_in", "layer0_sa_output"]
    stages += [f"hidden_after_layer{li}" for li in KEY_LAYERS]
    if steps <= 8:
        step_indices = list(range(steps))
    else:
        step_indices = list(range(0, steps, 5))
        if (steps - 1) not in step_indices:
            step_indices.append(steps - 1)
    for si in step_indices:
        stages.append(f"dit_step{si}_vt_cond")
        if si < 2:
            stages.append(f"dit_step{si}_vt_uncond")
        stages.append(f"dit_step{si}_vt")
        if si < steps - 1:
            stages.append(f"dit_step{si}_xt")
    stages += ["dit_x0", "vae_audio"]
    return stages


def compare(dirs, stages, tag):
    labels = sorted(dirs.keys())
    pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]

    print(f"[{tag}] Cosine similarities GGML vs Python")
    print(f"  {'stage':30s}", end="")
    for a, b in pairs:
        print(f" {a + ' vs ' + b:>14s}", end="")
    print()

    for stage in stages:
        data = {}
        for label, d in dirs.items():
            f = os.path.join(d, stage + ".bin")
            if os.path.isfile(f):
                data[label] = load_dump(f)
        if not data:
            continue
        print(f"  {stage:30s}", end="")
        for a, b in pairs:
            if a in data and b in data:
                da, sa = data[a]
                db, sb = data[b]
                c = cos(da, db, sa, sb)
                print(f" {c:>14.6f}", end="")
            else:
                print(f" {'N/A':>14s}", end="")
        print()

    vae_data = {}
    for label, d in dirs.items():
        f = os.path.join(d, "vae_audio.bin")
        if os.path.isfile(f):
            vae_data[label] = load_dump(f)
    if len(vae_data) >= 2:
        print(f"  {'vae_audio (STFT cosine)':30s}", end="")
        for a, b in pairs:
            if a in vae_data and b in vae_data:
                left_a = vae_data[a][0].reshape(-1, 2)[:, 0]
                left_b = vae_data[b][0].reshape(-1, 2)[:, 0]
                c = stft_cos(left_a, left_b)
                print(f" {c:>14.6f}", end="")
            else:
                print(f" {'N/A':>14s}", end="")
        print()

    if len(pairs) > 0:
        a_label, b_label = pairs[0]
        a_dir, b_dir = dirs[a_label], dirs[b_label]
        xt_stages = [s for s in stages if "_xt" in s]
        if xt_stages:
            print(f"[{tag}] Error growth GGML vs Python")
            print(f"  {'stage':22s} {'cos':>10s} {'max_err':>10s} {'mean_err':>10s}"
                  f" {'mean_A':>10s} {'std_A':>10s} {'mean_B':>10s} {'std_B':>10s}")
            for stage in xt_stages:
                fa = os.path.join(a_dir, stage + ".bin")
                fb = os.path.join(b_dir, stage + ".bin")
                if os.path.isfile(fa) and os.path.isfile(fb):
                    da, sa = load_dump(fa)
                    db, sb = load_dump(fb)
                    n = min(len(da), len(db))
                    da, db = da[:n], db[:n]
                    c = _cos_flat(da, db)
                    diff = np.abs(da - db)
                    print(f"  {stage:22s} {c:10.6f} {diff.max():10.6f} {diff.mean():10.6f}"
                          f" {da.mean():10.6f} {da.std():10.6f} {db.mean():10.6f} {db.std():10.6f}")
                else:
                    missing = []
                    if not os.path.isfile(fa):
                        missing.append(a_label)
                    if not os.path.isfile(fb):
                        missing.append(b_label)
                    print(f"  {stage:22s} missing: {', '.join(missing)}")


# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="F32", help="DiT GGUF quant suffix (F32, Q8_0, Q6_K, Q5_K_M, Q4_K_M)")
    args = ap.parse_args()

    if not os.path.isfile("request0.json"):
        print("[Error] request0.json not found in CWD")
        sys.exit(1)
    with open("request0.json") as f:
        req = json.load(f)
    print("[Request] Loaded request0.json")

    dump_ggml = "ggml-dit"
    dump_python = "python-dit"
    steps = int(req.get("steps", 30))

    print(f"[DiT] steps={steps}, CFG={req.get('dit_cfg', 1.7)} | mm3-dit-{args.quant}.gguf")

    if os.path.isdir(dump_ggml):
        shutil.rmtree(dump_ggml)
    if not run_ggml(dump_ggml, req, args.quant):
        print("[DiT] GGML failed")
        sys.exit(1)

    if os.path.isdir(dump_python):
        shutil.rmtree(dump_python)
    if not run_python(dump_python, dump_ggml, req):
        print("[DiT] Python failed")
        sys.exit(1)

    compare({"ggml": dump_ggml, "python": dump_python}, build_stages(steps), "DiT")


if __name__ == "__main__":
    main()
