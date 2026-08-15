#!/usr/bin/env python3
# convert.py: safetensors to GGUF for MiniMax Music 3
# Reads the HF diffusers layout from checkpoints/, writes GGUF to models/
# Each GGUF is self-contained: weights + hparams, the LM GGUF also carries
# the Qwen2 BPE tokenizer.
#
# Components (HF subfolder -> GGUF):
#   language_model/    -> MiniMax-Music3-language_model-BF16.gguf    (Qwen3 8B, semantic codebook)
#   rvq_depth_decoder/ -> MiniMax-Music3-rvq_depth_decoder-BF16.gguf (0.6B intra frame transformer)
#   condition_encoder/ -> MiniMax-Music3-condition_encoder-F32.gguf (hidden state fusion)
#   transformer/       -> MiniMax-Music3-transformer-F32.gguf       (2.4B flow matching DiT)
#   vocoder/ + dav.pth -> MiniMax-Music3-vocoder-F32.gguf           (flow VAE decoder + the
#                                                                    encoder of the dav.pth
#                                                                    training checkpoint, weight
#                                                                    norm folded: w = g*v/||v||)

import os
import sys
import json
import struct
import numpy as np
import gguf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models")

COMPONENTS = {
    "lm":    "language_model",
    "depth": "rvq_depth_decoder",
    "cond":  "condition_encoder",
    "dit":   "transformer",
    "vae":   "vocoder",
}

def log(tag, msg):
    print("[%s] %s" % (tag, msg), file=sys.stderr, flush=True)

# Safetensors reader
def read_sf_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    meta.pop("__metadata__", None)
    return meta, 8 + n

def find_sf_files(model_dir):
    """Return the list of safetensors paths (single or sharded)."""
    for name in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
        single = os.path.join(model_dir, name)
        if os.path.exists(single):
            return [single]
    for name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        index = os.path.join(model_dir, name)
        if os.path.exists(index):
            with open(index, "r", encoding="utf-8") as f:
                idx = json.load(f)
            shards = sorted(set(idx["weight_map"].values()))
            return [os.path.join(model_dir, s) for s in shards]
    raise FileNotFoundError("no safetensors in %s" % model_dir)

DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}

def load_sf_tensors(model_dir):
    """Load every tensor of a component as float32 numpy arrays."""
    tensors = {}
    for path in find_sf_files(model_dir):
        meta, data_start = read_sf_header(path)
        with open(path, "rb") as f:
            for name, t in meta.items():
                f.seek(data_start + t["data_offsets"][0])
                raw = f.read(t["data_offsets"][1] - t["data_offsets"][0])
                a = np.frombuffer(raw, dtype=DTYPES[t["dtype"]])
                if t["dtype"] == "BF16":
                    a = (a.astype(np.uint32) << 16).view(np.float32)
                tensors[name] = a.astype(np.float32).reshape(t["shape"])
    return tensors

def fold_weight_norm(tensors):
    """Fold every weight_g/weight_v pair: w = g * v / ||v|| along dim 0."""
    out = {}
    for name, a in tensors.items():
        if name.endswith(".weight_g"):
            continue
        if name.endswith(".weight_v"):
            base = name[: -len(".weight_v")]
            g = tensors[base + ".weight_g"]
            d0 = a.shape[0]
            norm = np.sqrt((a.reshape(d0, -1) ** 2).sum(axis=1))
            scale = (g.reshape(d0) / norm).reshape((d0,) + (1,) * (a.ndim - 1))
            out[base + ".weight"] = (a * scale).astype(np.float32)
        else:
            out[name] = a
    return out

# PyTorch checkpoint reader (pure python: zip + pickle with stub classes)
PTH_DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64}

def load_pth_tensors(path):
    """Load a torch .pth state dict as float32 numpy arrays, no torch needed."""
    import zipfile, pickle, io, importlib

    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    root = pkl.rsplit("/", 1)[0]

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.split(".")[0] in ("collections", "builtins"):
                return getattr(importlib.import_module(mod), name)
            if name.endswith("Storage"):
                return name
            if name == "_rebuild_tensor_v2":
                return lambda st, off, size, stride, *a, **k: ("T", st, off, tuple(size))
            return lambda *a, **k: None

        def persistent_load(self, pid):
            return (pid[1], pid[2])  # (storage type name, zip data key)

    obj = Unpickler(io.BytesIO(z.read(pkl))).load()

    def walk(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from walk(v, prefix + str(k) + ".")
            elif isinstance(v, tuple) and v and v[0] == "T":
                yield prefix + str(k), v

    tensors = {}
    for name, (_, (tname, key), off, shape) in walk(obj):
        raw = z.read("%s/data/%s" % (root, key))
        a = np.frombuffer(raw, dtype=PTH_DTYPES[tname])
        n = int(np.prod(shape)) if shape else 1
        tensors[name] = a[off : off + n].astype(np.float32).reshape(shape)
    return tensors

def load_dav_encoder(path):
    """DAC-VAE encoder of the dav.pth training checkpoint, renamed to the
    decoder naming vocabulary (its decoder half is bit-identical to the
    published vocoder). Kept: encoder.* and the posterior mean projection."""
    raw = load_pth_tensors(path)
    sub = {"0": "snake1.alpha", "1": "conv1", "2": "snake2.alpha", "3": "conv2"}
    out = {}
    for name, a in raw.items():
        p = name.split(".")
        if name.startswith("mean_proj."):
            out["encoder.mean_proj." + p[1]] = a
        elif name.startswith("encoder.block.0."):
            out["encoder.conv_in." + p[3]] = a
        elif name.startswith("encoder.block.5."):
            out["encoder.snake_out.alpha"] = a
        elif name.startswith("encoder.block.6."):
            out["encoder.conv_out." + p[3]] = a
        elif name.startswith("encoder.block."):
            blk = "encoder.blocks.%d." % (int(p[2]) - 1)
            if p[4] in ("0", "1", "2"):
                tail = sub[p[6]]
                out[blk + "res_unit%d." % (int(p[4]) + 1) + (tail if p[6] in ("0", "2") else tail + "." + p[7])] = a
            elif p[4] == "3":
                out[blk + "snake1.alpha"] = a
            else:
                out[blk + "conv1." + p[5]] = a
    return out

def convert_vae():
    """vocoder/ + dav.pth encoder -> MiniMax-Music3-vocoder-F32.gguf,
    weight norm folded, native F32. Decoder tensors mirror the published
    subfolder byte-perfect; encoder tensors come from the dav.pth training
    checkpoint whose decoder half is bit-identical to the published one."""
    tensors = fold_weight_norm(load_sf_tensors(os.path.join(CHECKPOINT_DIR, COMPONENTS["vae"])))
    dav_path = os.path.join(CHECKPOINT_DIR, "dav.pth")
    if not os.path.exists(dav_path):
        log("vae", "FATAL: %s missing (run ./checkpoints.sh)" % dav_path)
        sys.exit(1)
    tensors.update(fold_weight_norm(load_dav_encoder(dav_path)))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "MiniMax-Music3-vocoder-F32.gguf")

    w = gguf.GGUFWriter(out_path, arch="mm3-vae")
    w.add_name("MiniMax-Music3 Flow-VAE")
    w.add_uint32("mm3-vae.latent_channels", 128)
    w.add_uint32("mm3-vae.sampling_rate", 44100)
    w.add_array("mm3-vae.upsampling_ratios", [8, 8, 4, 2])

    for name in sorted(tensors):
        a = tensors[name]
        if name.endswith(".alpha"):
            a = a.reshape(-1)
        w.add_tensor(name, np.ascontiguousarray(a.astype(np.float32)))
        log("vae", "%s %s" % (name, list(a.shape)))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log("vae", "wrote %s (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1e6))

def convert_cond():
    """condition_encoder/ -> MiniMax-Music3-condition_encoder-F32.gguf, native F32."""
    tensors = load_sf_tensors(os.path.join(CHECKPOINT_DIR, COMPONENTS["cond"]))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "MiniMax-Music3-condition_encoder-F32.gguf")

    w = gguf.GGUFWriter(out_path, arch="mm3-cond")
    w.add_name("MiniMax-Music3 condition encoder")
    w.add_uint32("mm3-cond.condition_hidden_dim", 4096)
    w.add_uint32("mm3-cond.num_condition_layers", 8)
    w.add_uint32("mm3-cond.out_dim", 2048)
    w.add_uint32("mm3-cond.input_sampling_rate", 24000)
    w.add_uint32("mm3-cond.input_hop_length", 960)
    w.add_uint32("mm3-cond.output_sampling_rate", 44100)
    w.add_uint32("mm3-cond.output_hop_length", 512)

    for name in sorted(tensors):
        w.add_tensor(name, np.ascontiguousarray(tensors[name].astype(np.float32)))
        log("cond", "%s %s" % (name, list(tensors[name].shape)))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log("cond", "wrote %s (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1e6))

def stream_native_tensors(w, model_dir, tag):
    """Stream every safetensors tensor into the writer byte perfect in its native dtype."""
    BF16 = gguf.GGMLQuantizationType.BF16
    F32 = gguf.GGMLQuantizationType.F32
    count = 0
    for path in find_sf_files(model_dir):
        meta, data_start = read_sf_header(path)
        with open(path, "rb") as f:
            for name in sorted(meta):
                t = meta[name]
                f.seek(data_start + t["data_offsets"][0])
                raw = f.read(t["data_offsets"][1] - t["data_offsets"][0])
                if t["dtype"] == "BF16":
                    arr = np.frombuffer(raw, dtype=np.uint16).reshape(t["shape"])
                    w.add_tensor(name, arr, raw_dtype=BF16)
                elif t["dtype"] == "F32":
                    arr = np.frombuffer(raw, dtype=np.float32).reshape(t["shape"])
                    w.add_tensor(name, arr, raw_dtype=F32)
                else:
                    raise SystemExit("unexpected dtype %s for %s" % (t["dtype"], name))
                count += 1
    log(tag, "%d tensors" % count)

def convert_dit():
    """transformer/ -> MiniMax-Music3-transformer-F32.gguf, native F32, streamed shard by shard."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "MiniMax-Music3-transformer-F32.gguf")

    w = gguf.GGUFWriter(out_path, arch="mm3-dit")
    w.add_name("MiniMax-Music3 flow matching DiT")
    w.add_uint32("mm3-dit.in_channels", 128)
    w.add_uint32("mm3-dit.condition_dim", 2048)
    w.add_uint32("mm3-dit.num_layers", 36)
    w.add_uint32("mm3-dit.num_attention_heads", 32)
    w.add_uint32("mm3-dit.attention_head_dim", 64)
    w.add_uint32("mm3-dit.ff_inner_dim", 8192)
    w.add_uint32("mm3-dit.rotary_dim", 32)
    w.add_uint32("mm3-dit.fourier_embedding_dim", 256)

    stream_native_tensors(w, os.path.join(CHECKPOINT_DIR, COMPONENTS["dit"]), "dit")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log("dit", "wrote %s (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1e6))

def convert_depth():
    """rvq_depth_decoder/ -> MiniMax-Music3-rvq_depth_decoder-BF16.gguf, native BF16."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "MiniMax-Music3-rvq_depth_decoder-BF16.gguf")

    w = gguf.GGUFWriter(out_path, arch="mm3-depth")
    w.add_name("MiniMax-Music3 RVQ depth decoder")
    w.add_uint32("mm3-depth.hidden_size", 4096)
    w.add_uint32("mm3-depth.num_layers", 4)
    w.add_uint32("mm3-depth.num_attention_heads", 16)
    w.add_uint32("mm3-depth.intermediate_size", 6144)
    w.add_uint32("mm3-depth.audio_vocab_size", 1024)
    w.add_uint32("mm3-depth.num_codebooks", 8)
    w.add_uint32("mm3-depth.max_position_embeddings", 16)

    stream_native_tensors(w, os.path.join(CHECKPOINT_DIR, COMPONENTS["depth"]), "depth")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log("depth", "wrote %s (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1e6))

def convert_lm():
    """language_model/ + tokenizer/ -> MiniMax-Music3-language_model-BF16.gguf, BF16 weights, config json and BPE tokenizer embedded."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "MiniMax-Music3-language_model-BF16.gguf")

    with open(os.path.join(CHECKPOINT_DIR, "language_model", "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    w = gguf.GGUFWriter(out_path, arch="mm3-lm")
    w.add_name("MiniMax-Music3 global LM (Qwen3 8B)")
    w.add_string("mm3.config_json", json.dumps(cfg, separators=(",", ":")))

    with open(os.path.join(CHECKPOINT_DIR, "tokenizer", "tokenizer.json"), "r", encoding="utf-8") as f:
        tj = json.load(f)
    vocab = tj["model"]["vocab"]
    added = {t["id"]: t["content"] for t in tj.get("added_tokens", [])}
    n_tokens = max(max(vocab.values()), max(added)) + 1
    tokens = [""] * n_tokens
    for tok_str, tok_id in vocab.items():
        tokens[tok_id] = tok_str
    for tok_id, tok_str in added.items():
        tokens[tok_id] = tok_str
    merges = [" ".join(m) if isinstance(m, list) else m for m in tj["model"]["merges"]]
    w.add_tokenizer_model("gpt2")
    w.add_token_list(tokens)
    w.add_token_merges(merges)
    log("lm", "tokenizer: %d tokens, %d merges" % (n_tokens, len(merges)))

    stream_native_tensors(w, os.path.join(CHECKPOINT_DIR, COMPONENTS["lm"]), "lm")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log("lm", "wrote %s (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1e6))

def convert(component):
    if component == "vae":
        convert_vae()
        return
    if component == "cond":
        convert_cond()
        return
    if component == "dit":
        convert_dit()
        return
    if component == "depth":
        convert_depth()
        return
    if component == "lm":
        convert_lm()

def main():
    if not os.path.isdir(CHECKPOINT_DIR):
        log("GGUF", "checkpoints/ not found")
        return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    native = {"lm": "BF16", "depth": "BF16", "cond": "F32", "dit": "F32", "vae": "F32"}
    converted = 0
    for comp in COMPONENTS:
        output_path = os.path.join(OUTPUT_DIR, "MiniMax-Music3-%s-%s.gguf" % (COMPONENTS[comp], native[comp]))
        if os.path.exists(output_path):
            log("GGUF", "skip %s: %s exists" % (comp, os.path.basename(output_path)))
            converted += 1
            continue
        convert(comp)
        converted += 1

    log("GGUF", "done: %d model(s) in %s" % (converted, OUTPUT_DIR))
    return 0

if __name__ == "__main__":
    sys.exit(main())
