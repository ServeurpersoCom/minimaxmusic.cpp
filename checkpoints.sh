#!/bin/bash
# Download MiniMax Music 3 checkpoints from HuggingFace
# Usage: ./checkpoints.sh
#   Fetches the five component subfolders, scheduler, tokenizer, and
#   dav.pth (the VAE encoder source).
#   The rest of the repository (qwen_7B, training checkpoints) is skipped.

set -eu

DIR="checkpoints"
mkdir -p "$DIR"

HF="hf download --quiet"
REPO="MiniMaxAI/MiniMax-Music3"

dl_weights() {
    local name="$1"
    local target="$DIR/$name"
    if [ -d "$target" ] && [ "$(ls "$target"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[OK] $name"
        return
    fi
    echo "[Download] $name <- $REPO"
    $HF "$REPO" --include "$name/*" --local-dir "$DIR"
}

dl_config() {
    local name="$1"
    local target="$DIR/$name"
    if [ -d "$target" ] && [ "$(ls -A "$target" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[OK] $name"
        return
    fi
    echo "[Download] $name <- $REPO"
    $HF "$REPO" --include "$name/*" --local-dir "$DIR"
}

dl_file() {
    local name="$1"
    if [ -f "$DIR/$name" ]; then
        echo "[OK] $name"
        return
    fi
    echo "[Download] $name <- $REPO"
    $HF "$REPO" --include "$name" --local-dir "$DIR"
}

dl_weights "language_model"
dl_weights "rvq_depth_decoder"
dl_weights "condition_encoder"
dl_weights "transformer"
dl_weights "vocoder"
dl_config "scheduler"
dl_config "tokenizer"
dl_file "dav.pth"
dl_file "config.json"
dl_file "modular_model_index.json"

find "$DIR" -name '.cache' -type d -exec rm -rf {} + 2>/dev/null
echo "[Done] Checkpoints ready in $DIR"
echo "[Done] Run: ./convert.py"
