# minimaxmusic.cpp

Local MiniMax Music 3 song generation server with browser UI, powered by GGML.
Lyrics and a structured caption in, complete stereo 44.1kHz songs out.
Runs on CPU, CUDA, Vulkan.

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/minimaxmusic.cpp.git
cd minimaxmusic.cpp
```

### Windows

To build from source, install
[Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
(select "Desktop development with C++" workload) and optionally the
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and/or the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home).

```cmd
buildcuda.cmd     # NVIDIA GPU
buildvulkan.cmd   # AMD/Intel GPU (Vulkan)
buildall.cmd      # all backends (CUDA + Vulkan + CPU, runtime loading)
```

### Linux / macOS

```bash
./buildcuda.sh    # NVIDIA GPU
./buildvulkan.sh  # AMD/Intel GPU (Vulkan)
./buildcpu.sh     # CPU only (with BLAS)
./buildall.sh     # all backends (CUDA + Vulkan + CPU, runtime loading)
```

macOS auto-enables Metal and Accelerate BLAS with any of the above.

## Convert

No pre-quantized GGUF repository is published yet: the GGUFs are built
locally from the official checkpoints. Download
[MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
into `checkpoints/`. Only the five component subfolders and the tokenizer
are used; the rest of the repository (`qwen_7B/`, training checkpoints)
can be skipped.

```bash
pip install hf gguf numpy
hf download MiniMaxAI/MiniMax-Music3 --local-dir checkpoints
./convert.py      # native GGUF, byte-exact dtypes from the source, skips existing
./quantize.sh     # every quant from the natives, idempotent
```

| GGUF | Component | Size |
|------|-----------|------|
| MiniMax-Music3-language_model-BF16.gguf | global LM 8B (Qwen3) | 17.2 GB |
| MiniMax-Music3-rvq_depth_decoder-BF16.gguf | RVQ depth decoder 0.6B | 1.3 GB |
| MiniMax-Music3-condition_encoder-F32.gguf | condition encoder | 101 MB |
| MiniMax-Music3-transformer-F32.gguf | flow matching DiT 2.4B | 9.7 GB |
| MiniMax-Music3-vocoder-F32.gguf | flow VAE decoder | 217 MB |

Quantized variants: LM in Q5_K_M / Q6_K / Q8_0, DiT in Q4_K_M / Q5_K_M /
Q6_K / Q8_0, depth decoder in Q8_0. The full quantized combo runs in
about 9 GB of VRAM, the full native set in about 29 GB.

## Run

```bash
./server.sh       # Linux / macOS
server.cmd        # Windows
```

Open http://localhost:8086 in your browser. The WebUI handles everything:
write a structured caption, set lyrics and duration, generate, play, and
download tracks.

Models are loaded on the first job (zero GPU at startup) and hot-swapped
per component when you pick a different one in the UI: switching one
quant reloads only that model, the other four stay resident.

## Server options

```
Usage: ./mm-server --models <dir> [options]

Required:
  --models <dir>         Directory of GGUF model files

Server:
  --host <addr>          Listen address (default: 127.0.0.1)
  --port <N>             Listen port (default: 8086)
  --max-seq <N>          LM KV cache size (default: model context)

Debug:
  --no-fa                Disable flash attention
  --no-batch-cfg         Split CFG into two separate forwards
  --clamp-fp16           Clamp hidden states to FP16 range
  --dump <dir>           Dump intermediate tensors
```

<details>
<summary>API endpoints</summary>

The server exposes one compute endpoint and a job system:

**POST /synth** - Submit a generation job (JSON MM3Request), returns a job
ID immediately. The single worker thread processes jobs in FIFO order.

**GET /job?id=N** - Poll job status. **GET /job?id=N&result=1** fetches the
result (MP3 or WAV, selected by `output_format` in the request).
**POST /job?id=N&cancel=1** cancels a running job.

**GET /health** - Returns `{"status":"ok"}`.

**GET /props** - Available models per component, server version, default
request parameters.

**GET /logs** - SSE stream of server stderr.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full API reference
and MM3Request JSON specification.

</details>

<details>
<summary>CLI tools (advanced)</summary>

For scripting without the server, `mm-synth` runs the full pipeline:

```bash
# quick one-shot
./build/mm-synth \
    --models models \
    --caption "Melancholic synthwave, slow tempo, analog pads" \
    --lyrics "[verse]..." \
    --out song.mp3

# same request schema as the server
./build/mm-synth \
    --models models \
    --request /tmp/request.json
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full JSON
reference and the `quantize` tool.

</details>

## Technical documentation

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers the complete MM3Request
JSON reference, the five model components, the autoregressive and flow
matching inference recipe, quantization strategy, VRAM and model routing,
the parity and cosine similarity test suites, and architecture internals.

## Acknowledgements

Independent C++ implementation based on
[MiniMax Music 3](https://github.com/MiniMax-AI/MiniMax-Music3) by MiniMax.
All model weights are theirs, this is just a native backend.
Structural template: [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp).

```bibtex
@misc{minimax2026music3,
	title={MiniMax Music 3},
	author={MiniMax},
	howpublished={\url{https://github.com/MiniMax-AI/MiniMax-Music3}},
	year={2026},
	note={GitHub repository}
}
```
