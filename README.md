# minimaxmusic.cpp

Local MiniMax Music 3 song generation server with browser UI, powered by GGML.
Lyrics and a structured caption in, complete stereo 44.1kHz songs out.
Runs on CPU, CUDA, Vulkan.

## Download models

Grab one GGUF of each type from Hugging Face and drop them in the
`models/` folder:

https://huggingface.co/Serveurperso/MiniMax-Music3-GGUF/tree/main

| Type | Pick one | Size |
|------|----------|------|
| LM | MiniMax-Music3-language_model-Q8_0.gguf | 9.1 GB |
| Depth decoder | MiniMax-Music3-rvq_depth_decoder-Q8_0.gguf | 690 MB |
| DiT | MiniMax-Music3-transformer-Q8_0.gguf | 2.6 GB |
| Condition encoder | MiniMax-Music3-condition_encoder-F32.gguf | 101 MB |
| VAE | MiniMax-Music3-vocoder-F32.gguf | 306 MB |

The LM also ships in BF16 / Q6_K / Q5_K_M, the DiT in F32 / Q6_K /
Q5_K_M / Q4_K_M, the depth decoder in BF16. The full quantized combo
runs in about 9 GB of VRAM, the full native set in about 29 GB.

Alternative: `./models.sh` downloads the default set automatically
(needs `pip install hf`), `./models.sh --all` everything.

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

`-DGGML_SOURCE_DIR=<path>` swaps the ggml submodule for another tree (upstream ggml, llama.cpp/ggml).

macOS auto-enables Metal and Accelerate BLAS with any of the above.

## Convert

To build the GGUFs locally from the official checkpoints instead,
download
[MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
into `checkpoints/`. Only the five component subfolders and the tokenizer
are used; the rest of the repository (`qwen_7B/`, training checkpoints)
can be skipped.

```bash
pip install hf gguf numpy
./checkpoints.sh  # downloads the needed subfolders of MiniMaxAI/MiniMax-Music3
./convert.py      # native GGUF, byte-exact dtypes from the source, skips existing
./quantize.sh     # every quant from the natives, idempotent
```

| GGUF | Component | Size |
|------|-----------|------|
| MiniMax-Music3-language_model-BF16.gguf | global LM 8B (Qwen3) | 17.2 GB |
| MiniMax-Music3-rvq_depth_decoder-BF16.gguf | RVQ depth decoder 0.6B | 1.3 GB |
| MiniMax-Music3-condition_encoder-F32.gguf | condition encoder | 101 MB |
| MiniMax-Music3-transformer-F32.gguf | flow matching DiT 2.4B | 9.7 GB |
| MiniMax-Music3-vocoder-F32.gguf | flow VAE encoder + decoder | 306 MB |

## Run

```bash
./server.sh       # Linux / macOS
server.cmd        # Windows
```

Open http://localhost:8086 in your browser. The WebUI handles everything:
write a structured caption, set lyrics and duration, generate, play, and
download tracks.

Models are loaded on the first job (zero GPU at startup). By default the
server runs the strict VRAM policy: the LM stage and the synthesis stage
swap in and out per job, so the LM and the DiT never coexist. Pass
`--keep-loaded` to keep every model resident instead; switching one quant
in the UI then loads only that model, the others stay warm.

## Adapters

Community LoRAs for the planner LM and for the flow DiT load at request
time. Point the server at a directory with `--adapters <dir>` and name its
entries in the request:

```json
"adapters": [
    { "name": "reggae.safetensors", "scale": 1.0 },
    { "name": "my-artist", "lm_scale": 1.0, "dit_scale": 0.5 }
]
```

An entry is a `.safetensors` file or a directory of them (a PEFT
`adapter_model.safetensors` with its `adapter_config.json`). The LM plans
the song, so its adapters change the composition, the vocal line and where
the song ends; the DiT renders the sound, so its adapters change the timbre.
`scale` applies to both, `lm_scale` and `dit_scale` override it for one, and
a model an adapter does not touch is not reloaded when that adapter changes.

The factors are merged into the weights while the model loads, before the
LM projections are fused, every contribution to a tensor summed in one
backend graph and encoded back to the GGUF type once. Read are SimpleTuner's
diffusers keys (`transformer.transformer_blocks.N.attn.to_q`) and ComfyUI
keys (`diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_qkv`,
split back onto q, k and v by rows), PEFT LM keys
(`language_model.model.layers.N.self_attn.q_proj`, ComfyUI's
`text_encoders.` form too), LyCORIS LoKr (`lycoris_layers_N_mlp_up_proj.lokr_w1`
with `lokr_w2`, or with `lokr_w2_a` and `lokr_w2_b`) and ComfyUI `.diff`
weights. Per tensor `.alpha`, `__metadata__` alpha and `adapter_config.json`
alpha are honoured in that order, and a diffusers checkpoint marked
`swiglu_gate_first` has the halves of its `ff_in` rows swapped to the
value-then-gate order of the GGUF. DoRA, LoHa and PiSSA delta files are
refused by name: their update is not `B @ A`, and merging one as if it were
renders plausible, wrong audio. `GET /props` lists the directory with the
models each entry touches.

## Short schedules

`flow_shift` warps the DiT's noise levels, `t' = shift t / (1 + (shift - 1) t)`.
Left at 0 it follows the step count: `(30 - 1) / (steps - 1)` below the 30
reference steps, 1 (the native schedule) from 30 up. At 10 steps without it
the stereo image collapses (left/right correlation -0.09 on a test song
against +0.77 at 30 steps); with the automatic shift of 3.22 it is +0.79.

## Server options

```
Usage: ./mm-server --models <dir> [options]

Required:
  --models <dir>         Directory of GGUF model files

Server:
  --host <addr>          Listen address (default: 127.0.0.1)
  --port <N>             Listen port (default: 8086)
  --max-batch <N>        LM batch limit (default: 1)
  --max-seq <N>          LM KV cache size (default: model context)
  --keep-loaded          Keep every model resident in VRAM (default: evict between stages)
  --adapters <dir>       Directory of LoRA / LoKr adapters requests may name

Debug:
  --no-fa                Disable flash attention
  --no-batch-cfg         Split CFG into two separate forwards (LM + DiT)
  --clamp-fp16           Clamp hidden states to FP16 range
  --dump <dir>           Dump intermediate tensors
```

<details>
<summary>API endpoints</summary>

The server exposes one compute endpoint and a job system:

**POST /synth** - Submit a generation job (JSON MM3Request), returns a job
ID immediately. The single worker thread processes jobs in FIFO order.
The body must be sent with `Content-Type: application/json` (the HTTP
server caps urlencoded bodies at 8 KB).

**GET /job?id=N** - Poll job status. **GET /job?id=N&result=1** fetches the
result as multipart/mixed: one JSON replay request part (the request with
`audio_codes` and the exact seed of the track) then one audio part per
track (MP3 or WAV, selected by `output_format` in the request).
**POST /job?id=N&cancel=1** cancels a running job.

**GET /health** - Returns `{"status":"ok"}`.

**GET /props** - Available models per component, server version, default
request parameters, and the adapters of `--adapters`.

**GET /logs** - SSE stream of server stderr.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full API reference
and MM3Request JSON specification.

</details>

<details>
<summary>CLI tools (advanced)</summary>

For scripting without the server, `mm-synth` runs the full pipeline.
Every rendered track gets its replay request written next to it
(`song.mp3` + `song.json`): the sampled codes travel in `audio_codes`,
and feeding that JSON back re-renders the same song with any synthesis
settings (models, steps, seed, CFG).

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

The `mm-lm` tool runs the autoregressive stage alone and writes the
same replayable request JSONs without synthesizing anything.

```bash
./build/mm-lm --models models --request song.json --out plan.json
./build/mm-synth --models models --request plan.json --out song.mp3
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full JSON
reference and the `neural-codec` (flow VAE audio codec, encode and decode, f32 and
reduced-bitrate Q8/Q4 latent formats), `mp3-codec` and `quantize` tools.

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
