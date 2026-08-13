#!/bin/bash

set -eu

Q="./build/quantize"

quantize() {
    local native="$1" type="$2"
    local out="${native%-*.gguf}-${type}.gguf"
    if [ -f "$out" ]; then
        echo "[Skip] $out"
    else
        $Q "$native" "$out" "$type"
    fi
}

# Depth decoder 0.6B (native BF16): Q8_0 only (too small to survive aggressive quant)
quantize models/MiniMax-Music3-rvq_depth_decoder-BF16.gguf Q8_0

# LM 8B (native BF16): no Q4_K_M (audio code LMs break below Q5, same rule as the acestep 4B)
for type in Q5_K_M Q6_K Q8_0; do
    quantize models/MiniMax-Music3-language_model-BF16.gguf "$type"
done

# DiT (native F32): full range, quantized straight from the F32 source
for type in Q4_K_M Q5_K_M Q6_K Q8_0; do
    quantize models/MiniMax-Music3-transformer-F32.gguf "$type"
done

# Vocoder and condition encoder: never quantized (stay native F32)
