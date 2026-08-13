#!/usr/bin/env python3
"""Reference forward of the Music3 global LM.

Imports Qwen3ForCausalLM from the reference project ../../transformers.
Dumps last-token logits and last hidden state for a fixed prompt: once for
the full prefill and once for prefill minus one plus one decode (same math).
Run from tests/ directory. All paths relative to CWD.

Usage:
    ./mm3-lm-ref.py <checkpoint_dir> <token_ids...>
"""
import sys

sys.path.insert(0, "../../transformers/src")

import torch
from transformers import Qwen3ForCausalLM
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

ids = [int(x) for x in sys.argv[2:]]
model = Qwen3ForCausalLM.from_pretrained(sys.argv[1], torch_dtype=torch.float32)
model.eval()

with torch.no_grad():
    # Reference for the prefill dump: all ids but the last
    out = model(input_ids=torch.tensor([ids[:-1]]), output_hidden_states=True)
    logits_prefill = out.logits[0, -1]
    hidden_prefill = out.hidden_states[-1][0, -1]
    # Reference for the decode dump: the full sequence, last position
    out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    logits_decode = out.logits[0, -1]
    hidden_decode = out.hidden_states[-1][0, -1]

logits_prefill.numpy().astype("float32").tofile("parity/lm_ref_prefill_logits.bin")
hidden_prefill.numpy().astype("float32").tofile("parity/lm_ref_prefill_hidden.bin")
logits_decode.numpy().astype("float32").tofile("parity/lm_ref_decode_logits.bin")
hidden_decode.numpy().astype("float32").tofile("parity/lm_ref_decode_hidden.bin")
print("prefill argmax", int(logits_prefill.argmax()), "| decode argmax", int(logits_decode.argmax()))
