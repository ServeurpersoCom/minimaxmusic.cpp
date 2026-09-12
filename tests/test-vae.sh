#!/bin/bash

set -eo pipefail

for backend in CUDA0 CPU; do
    env GGML_BACKEND=$backend ./test-vae.py 2>&1 | tee ${backend}-vae.log
done
