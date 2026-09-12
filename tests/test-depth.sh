#!/bin/bash

set -eo pipefail

for backend in CUDA0 CPU; do
    env GGML_BACKEND=$backend ./test-depth.py 2>&1 | tee ${backend}-depth.log
done
