#!/bin/bash
# parity.sh: parity suite, torch references vs the GGML component harnesses.
# Run from tests/ directory. All paths relative to CWD.
# References import the model code from the sibling clones ../../diffusers
# and ../../transformers.
#
# usage:
#     ./parity.sh [vae|cond|dit|depth|lm|all]
#
# Backend follows GGML_BACKEND (empty = best available), for instance:
#     env GGML_BACKEND=CUDA0 ./parity.sh 2>&1 | tee parity-CUDA0-BF16.log
#     env GGML_BACKEND=CPU ./parity.sh 2>&1 | tee parity-CPU-BF16.log

set -eu

COMP="${1:-all}"
CKPT=../checkpoints
MODELS=../models

mkdir -p parity

run_vae() {
    echo "[Parity] VAE reference"
    ./mm3-vae-ref.py "$CKPT/vocoder" 64
    ../build/test-vae "$MODELS/MiniMax-Music3-vocoder-F32.gguf" parity/latent.bin 64 parity/audio_ggml.bin
    ./parity-compare.py vae parity/audio_ref.bin parity/audio_ggml.bin --max-rel 1e-2
}

run_cond() {
    echo "[Parity] Cond reference"
    ./mm3-cond-ref.py "$CKPT/condition_encoder" 7
    ../build/test-cond "$MODELS/MiniMax-Music3-condition_encoder-F32.gguf" parity/cond_hidden.bin 7 parity/cond_ggml.bin
    ./parity-compare.py cond parity/cond_ref.bin parity/cond_ggml.bin --max-rel 1e-2
}

run_depth() {
    echo "[Parity] Depth reference"
    ./mm3-depth-ref.py "$CKPT/rvq_depth_decoder" 8
    ../build/test-depth "$MODELS/MiniMax-Music3-rvq_depth_decoder-BF16.gguf" parity/depth_seq.bin 8 parity/depth_h.bin parity/depth_l.bin
    ./parity-compare.py depth-hiddens parity/depth_hidden_ref.bin parity/depth_h.bin --max-rel 2e-2
    ./parity-compare.py depth-logits parity/depth_logits_ref.bin parity/depth_l.bin --max-rel 2e-2 --argmax
}

run_dit() {
    echo "[Parity] DiT reference"
    ./mm3-dit-ref.py "$CKPT/transformer" 24 0.35
    ../build/test-dit "$MODELS/MiniMax-Music3-transformer-F32.gguf" parity/dit_xt.bin parity/dit_cond.bin 24 0.35 parity/dit_ggml.bin
    ./parity-compare.py dit parity/dit_ref.bin parity/dit_ggml.bin --max-rel 5e-2
}

run_lm() {
    echo "[Parity] LM reference"
    ./mm3-lm-ref.py "$CKPT/language_model" 8948 3837 374 264 6543 11 279 151671 3703 92
    ../build/test-lm "$MODELS/MiniMax-Music3-language_model-BF16.gguf" parity/lm_ggml 8948 3837 374 264 6543 11 279 151671 3703 92
    ./parity-compare.py lm-prefill parity/lm_ref_prefill_logits.bin parity/lm_ggml_prefill_logits.bin --max-rel 2e-2 --argmax
    ./parity-compare.py lm-decode parity/lm_ref_decode_logits.bin parity/lm_ggml_decode_logits.bin --max-rel 2e-2 --argmax
}

case "$COMP" in
    vae) run_vae ;;
    cond) run_cond ;;
    depth) run_depth ;;
    dit) run_dit ;;
    lm) run_lm ;;
    all) run_vae; run_cond; run_depth; run_dit; run_lm ;;
    *) echo "[Parity] Unknown component $COMP"; exit 1 ;;
esac

echo "[Parity] Suite passed"
