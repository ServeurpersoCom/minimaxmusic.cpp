#!/usr/bin/env python3
# Dumps the regression targets of the hiddens route: for every corpus
# track, teacher-forces the true codes through the language model
# (torch checkpoint, conditional branch, full sequence) and saves the
# per-frame last hidden states h [n, 4096] as <base>.hid float16 next
# to the .vae. h[j] is the hidden that predicted stream frame j + 1,
# aligned to audio frame j. Idempotent: tracks whose .hid exists are
# skipped.
#
# Runs on the training venv (torch + transformers + diffusers editable
# from the local checkout). The prompt assembly mirrors the reference
# tokenize block: even whitespace changes alter the hiddens.
#
# Usage: dump-hiddens.py <dataset>

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import HID_EXT, JSON_EXT, VAE_EXT, load_codes

ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CKPT     = os.path.join(ROOT, "checkpoints")
DATASETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
DEVICE   = "cuda"
DTYPE    = torch.bfloat16
SEM_OFF  = 151675
AC_N     = 1024


def prompt_ids(tokenizer, caption: str, lyrics: str):
    from diffusers.modular_pipelines.minimax_music3 import encoders as enc

    text = (f"{enc._IM_START}{enc._CAPTION_START}{enc._clean_caption(caption)}{enc._CAPTION_END}"
            f"{enc._LYRICS_START}{enc._normalize_lyrics(lyrics)}{enc._LYRICS_END}{enc._IM_END}{enc._AUDIO_START}")
    return tokenizer(text, return_tensors="pt")["input_ids"].to(DEVICE)


def main():
    corpus = os.path.join(DATASETS, sys.argv[1])
    bases  = sorted(f[: -len(VAE_EXT)] for f in os.listdir(corpus) if f.endswith(VAE_EXT))
    todo   = [b for b in bases if not os.path.exists(os.path.join(corpus, b + HID_EXT))]
    print(f"[Dump] {len(todo)} tracks to dump, {len(bases) - len(todo)} skipped", flush=True)
    if not todo:
        return

    from transformers import Qwen2Tokenizer, Qwen3ForCausalLM
    from diffusers.models.transformers.minimax_music3_rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder

    tokenizer = Qwen2Tokenizer.from_pretrained(CKPT + "/tokenizer")
    lm = Qwen3ForCausalLM.from_pretrained(CKPT + "/language_model", torch_dtype=DTYPE).to(DEVICE)
    depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(CKPT + "/rvq_depth_decoder",
                                                         torch_dtype=DTYPE).to(DEVICE)
    lm.requires_grad_(False).eval()
    sem_table = lm.get_input_embeddings().weight
    ac_table  = depth.audio_embeddings.weight.view(7, AC_N, -1)

    for i, base in enumerate(todo):
        codes = torch.from_numpy(load_codes(os.path.join(corpus, base + JSON_EXT))).to(DEVICE)
        n = codes.shape[0] - 1
        with open(os.path.join(corpus, base + JSON_EXT)) as f:
            req = json.load(f)
        sem_emb  = sem_table[codes[:, 0] + SEM_OFF]
        ac_emb   = torch.stack([ac_table[k][codes[:, k + 1]] for k in range(7)])
        feedback = (sem_emb + ac_emb.sum(0)) / (8.0 ** 0.5)
        ids = prompt_ids(tokenizer, req["caption"], req["lyrics"])
        P   = ids.shape[1]
        embeds = torch.cat([lm.get_input_embeddings()(ids), feedback[:n].unsqueeze(0).to(DTYPE)], dim=1)
        with torch.no_grad():
            h = lm.model(inputs_embeds=embeds).last_hidden_state[0, P : P + n]
        h = h.float().cpu().numpy().astype(np.float16)
        assert np.isfinite(h).all()
        h.tofile(os.path.join(corpus, base + HID_EXT))
        if (i + 1) % 50 == 0:
            print(f"[Dump] {i + 1}/{len(todo)}", flush=True)
    print("[Done]", flush=True)


if __name__ == "__main__":
    main()
