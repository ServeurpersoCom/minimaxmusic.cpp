#!/bin/bash
# Download pre-quantized MiniMax Music 3 GGUF models from HuggingFace
#
# Usage: ./models.sh [options]
#   default:    Q8_0 set (LM Q8 + depth Q8 + DiT Q8 + cond F32 + VAE F32)
#   --all:      all models, all quants
#   --quant X:  use quant X (Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, F32)

set -eu

REPO="Serveurperso/MiniMax-Music3-GGUF"
DIR="models"
QUANT="Q8_0"
ALL=0

while [ $# -gt 0 ]; do
    case "$1" in
        --all)   ALL=1 ;;
        --quant) QUANT="$2"; shift ;;
        *)       echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$DIR"

dl() {
    local file="$1"
    if [ -f "$DIR/$file" ]; then
        echo "[OK] $file"
        return
    fi
    echo "[Download] $file"
    hf download --quiet "$REPO" "$file" --local-dir "$DIR"
}

if [ "$ALL" = 1 ]; then
    dl "MiniMax-Music3-condition_encoder-F32.gguf"
    dl "MiniMax-Music3-language_model-BF16.gguf"
    dl "MiniMax-Music3-language_model-Q5_K_M.gguf"
    dl "MiniMax-Music3-language_model-Q6_K.gguf"
    dl "MiniMax-Music3-language_model-Q8_0.gguf"
    dl "MiniMax-Music3-rvq_depth_decoder-BF16.gguf"
    dl "MiniMax-Music3-rvq_depth_decoder-Q8_0.gguf"
    dl "MiniMax-Music3-transformer-F32.gguf"
    dl "MiniMax-Music3-transformer-Q4_K_M.gguf"
    dl "MiniMax-Music3-transformer-Q5_K_M.gguf"
    dl "MiniMax-Music3-transformer-Q6_K.gguf"
    dl "MiniMax-Music3-transformer-Q8_0.gguf"
    dl "MiniMax-Music3-vocoder-F32.gguf"
    exit 0
fi

# Resolve quant to best available for each component.
# Matches the quantize.sh matrix exactly:
#   LM:    BF16, Q5_K_M, Q6_K, Q8_0  (no Q4)
#   Depth: BF16, Q8_0
#   DiT:   F32, Q4_K_M, Q5_K_M, Q6_K, Q8_0
#   Cond and VAE ship as F32 only (small, quality-critical).
# If requested quant unavailable, picks the next larger available.
resolve_quant() {
    local requested="$1" component="$2"
    case "$component" in
        lm)
            case "$requested" in
                BF16|F32)      echo "BF16" ;;
                Q6_K)          echo "Q6_K" ;;
                Q5_K_M|Q4_K_M) echo "Q5_K_M" ;;
                *)             echo "Q8_0" ;;
            esac ;;
        depth)
            case "$requested" in
                BF16|F32) echo "BF16" ;;
                *)        echo "Q8_0" ;;
            esac ;;
        dit)
            case "$requested" in
                BF16|F32) echo "F32" ;;
                *)        echo "$requested" ;;
            esac ;;
    esac
}

dl "MiniMax-Music3-condition_encoder-F32.gguf"
dl "MiniMax-Music3-vocoder-F32.gguf"
dl "MiniMax-Music3-language_model-$(resolve_quant "$QUANT" lm).gguf"
dl "MiniMax-Music3-rvq_depth_decoder-$(resolve_quant "$QUANT" depth).gguf"
dl "MiniMax-Music3-transformer-$(resolve_quant "$QUANT" dit).gguf"
