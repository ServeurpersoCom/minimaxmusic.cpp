#!/bin/bash

set -eu

# Multi-GPU: set GGML_BACKEND to pick a device (CUDA0, CUDA1, Vulkan0...)
#export GGML_BACKEND=CUDA0

./build/mm-server \
    --host 0.0.0.0 \
    --port 8086 \
    --models ./models
