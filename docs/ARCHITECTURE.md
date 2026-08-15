# Architecture

> Full technical reference for minimaxmusic.cpp. For a quick start guide, see [README.md](../README.md).

# minimaxmusic.cpp

Portable C++17 implementation of MiniMax Music 3 song generation using GGML.
Structured caption + lyrics in, stereo 44.1kHz MP3 or WAV out. Runs on CPU,
CUDA, Vulkan.

Source of truth: the HF diffusers layout of `MiniMaxAI/MiniMax-Music3`
(`modular_model_index.json`) and the reference implementation merged in
diffusers (PR 14456), expected as a sibling clone at `../diffusers` for the
parity suite. The `qwen_7B/` subfolder of the checkpoint is the fused SGLang
serving model and `dav.pth` / `flowmatching_vae.pth` are training
checkpoints: none of them is used by this port.

## Build

```bash
git submodule update --init

mkdir build && cd build

# macOS (Metal + Accelerate BLAS auto-enabled)
cmake ..

# Linux with NVIDIA GPU
cmake .. -DGGML_CUDA=ON

# Linux with Vulkan
cmake .. -DGGML_VULKAN=ON

cmake --build . --config Release -j$(nproc)
```

### Windows

Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
(select "Desktop development with C++" workload) and optionally the
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and/or the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home).

```cmd
git submodule update --init

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

mkdir build
cd build

rem NVIDIA GPU
cmake .. -DGGML_CUDA=ON

rem AMD/Intel GPU (Vulkan)
cmake .. -DGGML_VULKAN=ON

rem all backends (CUDA + Vulkan + CPU, runtime loading)
cmake .. -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON

cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
```

Builds six binaries: `mm-lm` (autoregressive stage CLI), `mm-synth`
(full pipeline CLI), `mm-server` (HTTP server with embedded WebUI),
`neural-codec` (flow VAE latent decoder), `mp3-codec` (MP3
encoder/decoder) and `quantize` (GGUF requantizer).

A single `build/` directory serves every backend combination. With
`buildall`, the backend is picked at runtime: the `GGML_BACKEND`
environment variable forces a specific device (`CUDA0`, `Vulkan0`, `CPU`),
unset picks the best available one.

## Models

Five GGUF files, one per pipeline component, named after the official
checkpoint subfolders:

| GGUF | Component | Native dtype | Size |
|------|-----------|--------------|------|
| MiniMax-Music3-language_model-BF16.gguf | global LM 8B (Qwen3) | BF16 | 17.2 GB |
| MiniMax-Music3-rvq_depth_decoder-BF16.gguf | RVQ depth decoder 0.6B | BF16 | 1.3 GB |
| MiniMax-Music3-condition_encoder-F32.gguf | condition encoder | F32 | 101 MB |
| MiniMax-Music3-transformer-F32.gguf | flow matching DiT 2.4B | F32 | 9.7 GB |
| MiniMax-Music3-vocoder-F32.gguf | flow VAE encoder + decoder | F32 | 306 MB |

The converter always produces the native dtype of the safetensors source,
byte for byte: BF16 tensors pass through untouched, F32 tensors stay F32.
No dtype exists in a GGUF that does not exist in the checkpoint. The only
transformation is the VAE weight norm folding (`w = g * v / ||v||`,
axis 0), which is the inference form of the same weights.

The vocoder GGUF carries a second source: the `encoder.*` tensors come
from the `dav.pth` training checkpoint at the repository root, whose
decoder half is bit-identical to the published `vocoder/` subfolder, so
its encoder is the exact companion of the decoder. Only the encoder and
the posterior mean projection are extracted; the training-side flow and
variance projection stay in `dav.pth`.

Quantized variants are generated from the natives:

| Component | Quants |
|-----------|--------|
| language_model (BF16) | Q5_K_M 6.3 GB, Q6_K 7.0 GB, Q8_0 9.1 GB |
| transformer (F32) | Q4_K_M 1.4 GB, Q5_K_M 1.7 GB, Q6_K 2.0 GB, Q8_0 2.6 GB |
| rvq_depth_decoder (BF16) | Q8_0 0.7 GB |
| condition_encoder, vocoder | never quantized |

The mapping follows the audio model quantization experience of acestep.cpp:
no Q4 on the code-generating LM (audio code LMs degrade below Q5), small
models only get Q8_0, and the bandwidth-bound frontends (condition encoder,
vocoder) stay native. Norm tensors are kept in F32 inside the quants.

The LM GGUF is self-contained: it embeds the Qwen2 BPE tokenizer (151643
base vocab + 32 added tokens) and the model `config.json`, so no external
file is needed at runtime.

<details>
<summary>Building GGUFs from source (checkpoints + convert)</summary>

```bash
pip install hf gguf numpy
hf download MiniMaxAI/MiniMax-Music3 --local-dir checkpoints
./convert.py     # convert each missing native GGUF from checkpoints/
./quantize.sh    # generate each missing quant from the natives
```

Both scripts take zero arguments and are idempotent: existing outputs are
skipped (`skip X: Y exists` / `[Skip]`), so a partial download or an
interrupted run resumes where it left off. `convert.py` reads the sharded
or single safetensors of each component subfolder and writes to `models/`.

</details>

## VRAM and model routing

Module loads go through a `ModelStore` (`src/model-store.h`), the single
owner of the GGML module instances. The pipeline borrows modules through
refcounted RAII handles (`ModelHandle`), stage by stage, and the store
decides what stays in VRAM following its eviction policy:

- `EVICT_STRICT` (default, hardcoded in the CLIs): at most one
  coexistence group resident at a time. The AR group `{ LM, depth }` is
  interleaved per frame, the synthesis group `{ cond, DiT, VAE }` per
  window and per song, so eviction operates on groups: when the
  synthesis group is required, the AR group has been released and is
  unloaded. The LM weights and the DiT weights never coexist, peak VRAM
  is the AR group.
- `EVICT_NEVER` (`mm-server --keep-loaded`): nothing is ever evicted,
  modules accumulate. Swapping a quant under this policy keeps both
  instances resident; the user opting in declares the budget for it.

A require of an already resident key is a cache hit on the same
instance. A conflicting require while a module of another group is still
held aborts: the strict invariant is enforced, not documented.

| Combo | LM | Depth | DiT | STRICT peak | Resident (`--keep-loaded`) |
|-------|----|-------|-----|-------------|----------------------------|
| Full native | BF16 | BF16 | F32 | ~19 GB | ~29 GB |
| Q8_0 | Q8_0 | Q8_0 | Q8_0 | ~10 GB | ~13 GB |
| Light | Q5_K_M | Q8_0 | Q4_K_M | ~7 GB | ~9 GB |

`pipeline_configure()` records the resolved GGUF path of each component;
the loads happen inside `pipeline_generate()` at stage boundaries,
through the store. Under `--keep-loaded`, switching the DiT quant in the
UI loads the new DiT and hits the cache for the other four modules; under
STRICT every job reloads each stage as it reaches it. A failed load
surfaces as a failed job and the next job retries it.

Model routing is a property of the request, not of the command line. The
`MM3Request` carries one optional GGUF filename per component (`lm_model`,
`depth_model`, `cond_model`, `dit_model`, `vae_model`), resolved against
the registry scanned from `--models <dir>` at startup. The registry
classifies each GGUF by its embedded `general.architecture` metadata into
the five buckets, so file names are free. An empty field keeps the
previously requested model, or falls to the first bucket entry
(alphabetical: the native variant with the official names) on the first
job, so a request without model fields never forces a swap. Unknown
names get a 400 from the server and a FATAL from the CLI.

## Pipeline

```
caption + lyrics (ChatML, Qwen2 BPE, 200k music vocab)
        v
global LM 8B (Qwen3 dense)          semantic codebook 16384, 25 Hz
        v hidden states
RVQ depth decoder 0.6B              7 acoustic codebooks x 1024 per frame
        v fused hidden states
condition encoder                   8 state mix, 4096 -> 2048, 25 -> 86.13 Hz
        v condition track
flow matching DiT 2.4B              latent 128 ch at 86.13 Hz
        v
flow VAE decoder 123M               2 x 64 ch tracks -> 44.1 kHz stereo
```

Frame rates: LM 24000 / 960 = 25 Hz, VAE latent 44100 / 512 = 86.13 Hz.

## Components

### Global LM (`language_model/`, Qwen3ForCausalLM)

Stock Qwen3 dense: 36 layers, hidden 4096, GQA 32/8, head_dim 128, SwiGLU
12288, RMSNorm eps 1e-6, q/k norm, RoPE theta 1e6, ctx 10240, untied
lm_head, vocab 200000 (text + semantic audio tokens). Predicts the first
RVQ codebook frame by frame at 25 Hz.

### RVQ depth decoder (`rvq_depth_decoder/`)

Intra frame causal transformer over codebook positions, run once per
25 Hz frame:

```
projection            [4096, 4096]     global LM hidden -> position 0
pos_embedding         [16, 4096]       learned, added after projection
audio_embeddings      [7168, 4096]     7 codebooks x 1024 entries
layers.{0..3}         to_{q,k,v,out} 4096, 16 heads x 256,
                      SwiGLU gate/up/down 6144, RMSNorm pair eps 1e-6
norm                  [4096]
audio_heads.{0..6}    [1024, 4096]
```

Causal attention without RoPE, scale 1/16. A sequence of length S predicts
codebook S-1 through `audio_heads[S-2]`.

### Condition encoder (`condition_encoder/`)

```
layer_weight_logits   [8]              softmax mix over 8 hidden states
layer_scale           [1]
proj                  conv1d 4096 -> 2048, k=3, pad 1
```

The 8 fused states per frame are the global LM last hidden state plus the
conditional hidden state of each of the 7 depth decoder steps, concatenated
layer-major. `softmax(layer_weight_logits) * layer_scale` is constant at
inference and folds into 8 fixed mix weights at load time. After the
projection, the track is resampled 25 Hz -> 86.13 Hz by nearest neighbor
(torch convention: `src = floor(i * T_in / T_out)`), with
`latent_length = int(n_frames * 44100 / 24000 * 960 / 512)`.

### Flow matching DiT (`transformer/`, MiniMaxMusic3Transformer1DModel)

36 self attention blocks, model dim 2048 (32 heads x 64), no cross
attention, no adaLN. Conditioning enters by channel concat: the input is
`[latent 128, zeros 128, condition 2048]` = 2304 channels through
`preprocess_conv` (k=1, residual: `conv(x) + x`), and `postprocess_conv`
(also residual) maps back to 128 latent channels.

The timestep is a prefix token: Fourier features `2*pi*t*w` ->
`cat(cos, sin)` 256 -> linear -> SiLU -> linear -> 2048, prepended to the
sequence and removed after the blocks. RoPE is NeoX-style partial
(32 of 64 dims per head, theta 10000) with positions that include the
timestep token. Attention is bidirectional, scale 1/8. The feed forward is
a chunked SwiGLU: `ff_in` 2048 -> 16384 splits into (value, gate) and
`out = ff_out(value * silu(gate))`. LayerNorm with bias, eps 1e-5.

The zeros channel slot is structurally a context latent input (the
architecture reserves `2 * in_channels + cond_dim` concat channels), but
the published model hardcodes it to zeros: no audio input, repaint or
continuation is exposed.

```
preprocess_conv, proj_in, time_proj, time_embed.linear_{1,2}
transformer_blocks.N.{norm1, attn.to_{q,k,v,out}, norm2, ff_in, ff_out}
proj_out, postprocess_conv
```

### Flow VAE (`vocoder/` + `dav.pth`, MiniMaxMusic3Vocoder)

DAC style stack. All convs are weight normalized in the checkpoint
(`weight_g` / `weight_v`), folded at conversion. The snake activation
uses the alpha parameter directly:
`y = x + sin^2(a * x) / (a + 1e-9)`.

The encoder (`src/vae-enc.h`, used by `neural-codec --encode`) is the
mirror: per side, conv_in (1 -> 64, k=7), 4 blocks of 3 res_units
(dilations 1, 3, 9) then snake and a strided conv (strides 2/4/8/8,
k = 2s, pad s/2), snake_out, conv_out (1024 -> 1024, k=3), and the
posterior mean projection (1024 -> 64, k=1). Downsample 512x, the exact
inverse of the decoder hop.

```
dec_in_proj           conv1d 64 -> 1024, k=1 (not weight normalized)
conv_in               conv1d 1024 -> 1536, k=7
blocks.{0..3}         snake1 -> conv_t1 (stride s, k = 2s, pad s/2)
                      -> 3 x res_unit (dilations 1, 3, 9)
                      strides [8, 8, 4, 2], dims 1536 -> 768 -> 384 -> 192 -> 96
res_unit              skip -> snake1 -> conv1 k=7 -> snake2 -> conv2 k=1 -> + skip
snake_out, conv_out   conv1d 96 -> 1, k=7 (with bias) -> tanh
```

The 128 latent channels are two independent 64 channel tracks, channels
0..63 for the left side. The decoder runs once per side and emits mono at
x512, 44.1 kHz.

GGML lowering: transposed convs as GEMM + `col2im_1d`, snake activations
fused by graph pattern recognition (same custom ops as acestep.cpp and
pocket-tts, see [Patched GGML fork](#patched-ggml-fork)). Conv weights are
stored F16 on device with F32 activations.

## Inference recipe

Pinned against the diffusers modular pipeline (`../diffusers`,
`src/diffusers/{models,modular_pipelines,schedulers}/*minimax_music3*`).

### Prompt assembly

```
ids = [151644 im_start, 151671 caption_start]
    + bpe(clean_caption)
    + [151672 caption_end, 151673 lyrics_start]
    + bpe(normalize_lyrics)
    + [151674 lyrics_end, 151645 im_end, 151669 audio_start]
```

The unconditional CFG stream is the same sequence with indices [1, n-3]
replaced by the audio CFG token 151654: only `im_start` and the trailing
`[lyrics_end, im_end, audio_start]` survive.

`_clean_caption` strips markdown and rewrites `<|x y|>` tags as "x is y";
any `Global Metadata` header text is part of the trained prompt format and
passes through. `_normalize_lyrics` keeps leading `[tag]` markers alone on
their line, lowercases the tag interior, and prepends `[start]\n`; lines in
parentheses pass through verbatim (the official demo prompts mix both
conventions freely). Exact ports live in `src/prompt.h`. BPE encoding adds
no EOS.

### Autoregressive stage

Semantic tokens occupy LM vocab ids [151675, 151675 + 16384), the end of
audio token is 151670. The LM runs the conditional and unconditional
streams as one batch of 2 over two KV cache sets. The first decode step
only advances past `audio_start` and emits no frame. Budget: 9000 frames
(6 minutes), also the cap applied to `duration`.

Semantic sampling per frame: logits masked to the semantic range plus the
end token, CFG on logits `guided = uncond + (cond - uncond) * lm_cfg`
restricted to the conditional branch's top `lm_top_k`, then softmax
sampled (mt19937 seeded by `lm_seed`).

Depth decoding per frame is a single fused graph: the 7 acoustic codebook
steps run as batch 2 (cond, uncond) with sampling inside the graph
(top-k by `argsort`, guided softmax, cumsum, CDF crossing against host
pre-drawn uniforms, `get_rows` on the device embedding table feeds the
next step). The sequence grows as
`[proj(LM last hidden), proj(LM_embed(semantic + 151675)),
proj(audio_emb(c_i + (i-1) * 1024)), ...]` with the learned positional
embedding added after projection. The conditional hidden state of each
step is collected before sampling; those 7 states plus the LM last hidden
are the 8 states fused by the condition encoder.

LM frame feedback: the next LM input embedding is
`(LM_embed(semantic + 151675) + sum of the 7 audio embeddings) * 8^-0.5`,
injected through the input_embeds path of the batched forward.

`--no-batch-cfg` splits both stages back into two separate forwards
(reference sequential path; outputs differ from the batched path at
epsilon level, which the sampling then amplifies, as with acestep).

### Denoising stage

The condition track is processed in windows of 200 LM frames with hop 100.
Per window, the DiT runs `steps` Euler iterations at T = latent window
length (689 for a full window):

- Sigmas: `1 - linspace(1, 1/N, N)`, ascending, with a final 1.0 appended
  (scheduler config: shift 1.0, `invert_sigmas`, num_train_timesteps 1).
  The DiT consumes the current sigma directly as flow time (0 = noise).
  Euler update: `x += (sigma_next - sigma) * v`.
- CFG on the velocity: `v = v_uncond + (v_cond - v_uncond) * dit_cfg`,
  the unconditional branch conditions on zeros. Both branches run as one
  batch of 2 under batch CFG.
- Overlap blending, applied at every step on the first 172 latent frames:
  `lat[:172] = (1 - (1 - 1e-6) * t) * noise_prompt + t * prev_latent`,
  restored after the step. The carry for the next window is latent range
  [L - 344, L - 172).
- Initial noise: `philox_normal4` seeded by `seed`, drawn as one
  continuous stream across windows.

### Decode and stitch

The VAE decodes each window; the first window keeps its left edge, later
windows crop 86 latents (x512 samples) on the left, and all but the last
crop 258 on the right. Segments are stitched and clamped.

### Post-processing

The pipeline outputs planar stereo float `[L: T][R: T]` at 44100 Hz, full
range. Normalization and encoding belong to the output stage (server
worker or CLI): percentile peak normalization targeting the
`1 - peak_clip / 1e6` percentile (`peak_clip = 0` is plain peak
normalization; WAV32 skips normalization entirely and writes raw IEEE
float), then MP3 (bitrate `mp3_bitrate`) or WAV 16/24/32 encoding in
memory.

## Request JSON reference

Every field has a default. Omitting a field is strictly equivalent to
sending it with its default value. Only `caption` and `lyrics` are
required: the server rejects requests missing either.

```json
{
    "caption":       "",
    "lyrics":        "",
    "duration":      60.0,
    "steps":         30,
    "seed":          -1,
    "lm_seed":       -1,
    "lm_cfg":        1.5,
    "lm_top_k":      50,
    "lm_batch_size": 1,
    "synth_batch_size": 1,
    "dit_cfg":       1.7,
    "peak_clip":     10,
    "output_format": "mp3",
    "mp3_bitrate":   128,
    "lm_model":      "",
    "depth_model":   "",
    "cond_model":    "",
    "dit_model":     "",
    "vae_model":     ""
}
```

The defaults mirror the diffusers pipeline: `lm_cfg`, `lm_top_k` and
`dit_cfg` are hardcoded constants in the reference and exposed as
parameters here with the reference values as defaults.

**`caption`** (string, required)
Music description fed to the global LM. The trained prompt format is the
Structured Caption of the official demos (`Global Metadata` header, bpm,
key, scale, per-section emotional progression); a plain natural language
description also works. See `tools/webui/example/` for the 61 official
demo prompts.

**`lyrics`** (string, required)
Song lyrics with structural tags. `[tag]` lines are normalized (kept alone
on their line, lowercased); parenthesized lines pass through verbatim
(section titles or backing vocals, interpreted in context by the model).
Instrumental tracks still carry lyrics in the official demos.

**`duration`** (float seconds, default `60.0`)
Target audio duration, capped by the 9000 frame LM budget (6 minutes).
The LM can end the song earlier with the end of audio token.

**`steps`** (int, default `30`)
Euler steps per DiT window. Minimum 2.

**`seed`** (int64, default `-1` = random)
DiT noise seed (Philox, low 32 bits consumed). Same seed, same noise.

**`lm_seed`** (int64, default `-1` = random)
Autoregressive sampling seed (mt19937, low 32 bits consumed). The song
structure, melody and length come from this one.

**`lm_cfg`** (float, default `1.5`)
CFG scale applied on LM and depth decoder logits.

**`lm_top_k`** (int, default `50`)
Top-K restriction, ranked on the conditional branch before guidance.

**`lm_batch_size`** (int, default `1`)
Number of songs generated from the prompt in one batched autoregressive
pass. Song i samples with its own stream seeded `lm_seed + i`
(consecutive internal seeds), so a song is bit-identical whether
generated alone or in a batch. Limited by the server's `--max-batch`.

**`synth_batch_size`** (int, default `1`)
Number of flow matching variations per song, batched in the DiT on the
shared condition track with consecutive noise seeds (`seed + j`).
Between 1 and 9. The job returns `lm_batch_size * synth_batch_size`
tracks in song-major order.

**`audio_codes`** (string, default `""`)
Explicit code stream, flat comma separated, 8 values per frame (the
semantic code then the 7 acoustic codebooks). Non-empty replaces the
autoregressive sampling: the hidden states are re-derived teacher-forced
from the codes (about an order of magnitude faster than sampling) and
the song renders deterministically, so the synthesis side (models,
steps, seed, CFG) can be iterated without re-rolling the LM. Produced by
`mm-lm`, written by `mm-synth` next to every rendered track, and
returned by the server as the JSON part paired with each audio track;
`lm_batch_size` is ignored when codes are present.

**`dit_cfg`** (float, default `1.7`)
CFG scale on the DiT velocity field.

**`peak_clip`** (int, default `10`)
Output normalization percentile control: the normalization peak is the
`1 - peak_clip / 1e6` percentile of the absolute signal. `0` normalizes to
the true peak with no clipping. Ignored by `wav32`.

**`output_format`** (string, default `"mp3"`)
Audio encoder: `"mp3"`, `"wav16"`, `"wav24"`, `"wav32"`. `wav32` writes
raw IEEE float without normalization.

**`mp3_bitrate`** (int, default `128`)
MP3 encoder bitrate in kbps. WAV outputs ignore it.

**`lm_model`**, **`depth_model`**, **`cond_model`**, **`dit_model`**,
**`vae_model`** (string, default `""`)
GGUF filename per component, resolved against the `--models` registry.
Empty keeps the previously requested model, or falls to the first
registry entry. Unknown names get a 400 from the server, a FATAL from the CLI. See
[VRAM and model routing](#vram-and-model-routing).

## mm-lm reference

Runs the autoregressive stage alone (global LM + depth decoder) and
writes one replayable request JSON per song, the sampled code stream in
`audio_codes`. The expensive stochastic stage runs once; feeding the
output back to `mm-synth --request` or `POST /synth` re-renders the same
song with any synthesis parameters.

```
Usage: ./mm-lm --models <dir> --request <json> [options]
       ./mm-lm --models <dir> --caption <text> --lyrics <text> [options]

Required:
  --models <dir>         Directory of GGUF model files
  --request <json>       Input request JSON (carries model routing)

Optional:
  --caption <text>       Caption (instead of --request)
  --lyrics <text>        Lyrics (instead of --request)
  --out <path>           Output request JSON (default: request.json)
  --duration <s>         Target duration in seconds
  --lm-seed <N>          Autoregressive sampling seed

Output is numbered for batches: request.json -> request0.json ...

Debug:
  --max-seq <N>          LM KV cache size (default: model context)
  --no-fa                Disable flash attention
  --no-batch-cfg         Split CFG into two separate forwards (LM + DiT)
  --clamp-fp16           Clamp hidden states to FP16 range
  --dump-tokens <path>   Dump prompt token IDs (CSV)
```

Only the LM and the depth decoder load (about 18.5 GB native, 7 GB
quantized). The written requests carry the input request fields plus the
song's codes and its traceable seed (`lm_seed + i`).

The replay re-derives the hiddens without sampling: the LM runs the
whole feedback sequence as one forward (conditional stream only) and the
depth decoder runs one causal S=8 forward per frame, so the CFG batch
shape differs from generation and the replayed hiddens sit at the usual
activation epsilon from the sampled run (measured rel rms ~2e-3, final
audio cosine ~0.99999: the same song).

## mm-synth reference

```
Usage: ./mm-synth --models <dir> --request <json> [options]
       ./mm-synth --models <dir> --caption <text> --lyrics <text> [options]

Required:
  --models <dir>         Directory of GGUF model files
  --request <json>       Input request JSON (carries model routing)

Optional:
  --caption <text>       Caption (instead of --request)
  --lyrics <text>        Lyrics (instead of --request)
  --out <path>           Output audio path (default: out.mp3)
  --duration <s>         Target duration in seconds
  --steps <N>            Euler steps per DiT window
  --seed <N>             DiT noise seed
  --lm-seed <N>          Autoregressive sampling seed

Debug:
  --max-seq <N>          LM KV cache size (default: model context)
  --no-fa                Disable flash attention
  --no-batch-cfg         Split CFG into two separate forwards
  --clamp-fp16           Clamp hidden states to FP16 range
  --dump <dir>           Dump intermediate tensors
```

`--request` takes the same MM3Request JSON as the server; CLI flags
override nothing, they are the lightweight alternative for one-shot runs.
The output container follows `output_format` in the request (`--out` names
the file). A default run and the server produce bit-identical audio for
the same resolved request.

Every rendered track gets its replay request written next to it, the
audio extension swapped to `.json`: the base request with `audio_codes`
and the exact seed of the track. Feeding it back re-renders the track
deterministically. Batches number the base with song then variation
index (`song.mp3` -> `song00.mp3` + `song00.json` ...).

`--dump <dir>` writes the intermediate tensors consumed by the cosine
similarity harness: per-frame fused hidden states, per-window condition
and latent tracks, per-step DiT velocities and states of the first window,
and the decoded audio, as flat binary f32 with a ndims + shape header.

The `quantize` tool regenerates any GGUF at another type:

```
./build/quantize <in.gguf> <out.gguf> <type>
```

`quantize.sh` drives it with the validated component mapping.

## mm-server reference

HTTP server exposing the pipeline behind an asynchronous job queue, with
the WebUI embedded (gzipped single page app, served at `/`).

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

Debug:
  --no-fa                Disable flash attention
  --no-batch-cfg         Split CFG into two separate forwards (LM + DiT)
  --clamp-fp16           Clamp hidden states to FP16 range
  --dump <dir>           Dump intermediate tensors
```

`--max-batch` sizes the LM KV cache at load time (2N sets of about
1.5 GB each) and bounds `lm_batch_size`; requests above the limit get
a 400.

The debug flags are global to the process and applied to the components as
they load (the graph caches bake them in), so they are boot options, not
request fields.

Models are loaded lazily on the first job: startup touches no GPU. A load
failure is treated as permanent and fails subsequent jobs fast.

### Endpoints

```
POST /synth                     Submit a generation job, returns job ID
  body: application/json MM3Request (the Content-Type header is
  required, urlencoded bodies are capped at 8 KB by the HTTP server)
  response: {"id":"1a2b..."}
  400 on malformed JSON, missing caption or lyrics, duration <= 0,
  steps < 2, prompt over the 5000 token budget, invalid output_format,
  unknown model name

GET  /job?id=N                  Poll job status
  response: {"status":"queued|running|done|failed|cancelled"}

GET  /job?id=N&result=1         Fetch job result
  multipart/mixed, boundary mm3-batch-boundary: one application/json
  replay request part (the request with audio_codes and the exact seed
  of the track) then one audio/mpeg or audio/wav part per track
  (output_format of the request), song-major order
  404 while the result is not ready

POST /job?id=N&cancel=1         Cancel a specific job
  response: {"status":"cancelled"}

GET  /health                    Server health check
  response: {"status":"ok"}

GET  /props                     Version, models, default request
  response: application/json

GET  /logs                      SSE stream of server stderr
  response: text/event-stream

GET  /                          Embedded WebUI (gzipped HTML)
```

Request bodies are limited to 8 MB (caption + lyrics JSON only). Error
responses are JSON: `{"error":"message"}`.

**GET /props** returns the model buckets and the full default request
(source of truth for the WebUI selects and placeholders):

```json
{
  "version": "...",
  "models": {
    "lm": ["MiniMax-Music3-language_model-BF16.gguf", "..."],
    "depth": ["..."],
    "cond": ["..."],
    "dit": ["..."],
    "vae": ["..."]
  },
  "defaults": { "caption": "", "duration": 60.0, ... }
}
```

### Concurrency

One worker thread consumes jobs from a FIFO queue and runs them serially
through the shared pipeline. The HTTP handler enqueues and returns a job
id immediately; the client polls. Completed jobs sit in memory and are
evicted FIFO past 32 entries, so a disconnected client can still fetch its
result after reconnecting.

Each job has a cancel flag polled by the pipeline between AR frames,
between DiT steps and between VAE windows, and passed down to the MP3
encoder. Shutdown (SIGINT/SIGTERM) cancels the active job through the same
flag: Ctrl+C lands in about 100 ms even mid-generation.

## neural-codec reference

GGML-native codec for the flow VAE latent space. The decoder weights come
from the published `vocoder/`; the encoder comes from the `dav.pth`
training checkpoint, whose decoder half is bit-identical to the published
vocoder, so the encoder is the exact companion of the decoder. Both live
in the vocoder GGUF (`encoder.*` tensors). The encode is deterministic:
the posterior mean, no sampling and no flow.

```
Usage: ./neural-codec --vae <gguf> --encode|--decode -i <input> [-o <output>] [--q8|--q4]

Required:
  --vae <path>            VAE GGUF file
  --encode | --decode     Encode audio to latent, or decode latent to audio
  -i <path>               Input (WAV/MP3 for encode, latent for decode)

Output:
  -o <path>               Output file (auto-named if omitted)
  --q8                    Quantize latent to int8 (~89.6 kbit/s)
  --q4                    Quantize latent to int4 (~45.5 kbit/s)
  --format <fmt>          mp3, wav16, wav24, wav32 (default: wav16)

Output naming: song.wav -> song.vae (f32) or song.nac8 (Q8) or song.nac4 (Q4)
               song.vae -> song.wav

Memory control:
  --vae-chunk <N>         Latent frames per tile (default: 689)
  --vae-overlap <N>       Overlap frames per side (default: 86)

Latent formats (decode auto-detects):
  .vae:  flat [T, 128] f32, no header. ~353 kbit/s.
  .nac8: header + per-frame Q8. ~89.6 kbit/s.
  .nac4: header + per-frame Q4. ~45.5 kbit/s.
```

The `.vae` file is the raw latent, frame-major, one 128 channel f32 frame
per latent step (512 bytes, x512 audio samples). The pipeline `--dump`
latents carry the same payload behind a 12 byte debug header (i32 ndims +
shape); stripping the header yields a valid `.vae`.

The `.nac8` and `.nac4` files are the reduced-bitrate codec formats:
an 8 byte header (4 byte magic `NAC8`/`NAC4` + uint32 frame count),
then one record per frame. Quantization is symmetric per frame: the
scale is the frame absmax over 127 (Q8) or 7 (Q4), stored as f16, the
values as int8 (130 byte frame) or signed nibbles packed two per byte,
low channel in the low nibble (66 byte frame). Decode auto-detects the
format from the magic, so the extension is a convention, not a contract.

Output is never normalized: the codec reproduces the decoder output
exactly. A single-tile decode is bit-identical to the audio the pipeline
emits for the same latent window. Long signals run in tiles with
symmetric overlap cropped on both sides; the overlap (86 latents = 44032
samples) sits far beyond the receptive field, so tiling only introduces
backend-level GEMM epsilon (max abs error ~4e-4 measured against a
single-tile decode on CUDA).

The encode input resamples to 44.1 kHz when needed and right-pads with
silence to whole latent frames. Roundtrip fidelity (CUDA, F32):
encode(decode(z)) recovers a pipeline latent at 0.993 cosine with the
variance preserved, and audio -> latent -> audio holds 0.994 STFT
magnitude cosine on an official demo track; the residual is the
intrinsic VAE reconstruction loss.

```bash
# encode to the Q8 reduced-bitrate format, decode auto-detects it
./neural-codec --vae models/MiniMax-Music3-vocoder-F32.gguf --encode --q8 -i song.wav
./neural-codec --vae models/MiniMax-Music3-vocoder-F32.gguf --decode -i song.nac8

# decode a dumped window latent
./neural-codec --vae models/MiniMax-Music3-vocoder-F32.gguf --decode -i song.vae -o song.wav

# raw float output, no 16 bit quantization
./neural-codec --vae models/MiniMax-Music3-vocoder-F32.gguf --decode -i song.vae --format wav32
```

## mp3-codec reference

Standalone MIT-licensed MPEG1 Layer III encoder and decoder. No external
dependencies, no GGML. The encoder is the one `mm-synth` and `mm-server`
use for MP3 output; the decoder uses minimp3 (CC0). Reads WAV or MP3,
writes WAV or MP3 (auto-detected from the output extension).

```
Usage: ./mp3-codec -i <input> -o <output> [options]

  -i <path>     Input file (WAV or MP3)
  -o <path>     Output file (WAV or MP3)
  -b <kbps>     Bitrate for MP3 encoding (default: 128)
  --format <fmt>  WAV format: wav16, wav24, wav32 (default: wav16)

Mode is auto-detected from output extension.

Examples:
  ./mp3-codec -i song.wav -o song.mp3
  ./mp3-codec -i song.wav -o song.mp3 -b 192
  ./mp3-codec -i song.mp3 -o song.wav
  ./mp3-codec -i song.mp3 -o song.wav --format wav32
```

## Accuracy

Two harnesses, both under `tests/`, both expecting the sibling clones
`../diffusers` (reference models) and `../transformers` (Qwen3 LM), run
from `tests/` with the ready build in `../build`.

### Parity suite

`./parity.sh [component|all]` compares each GGML module against the
`from_pretrained` reference on fixed seeds: `mm3-{vae,cond,dit,depth,lm}-ref.py`
dump torch outputs, the `test-*` harnesses dump GGML outputs, and
`parity-compare.py` checks relative RMS + max abs error (+ argmax token
match for the samplers), exit 1 on FAIL. Thresholds: vae/cond 1e-2, depth
2e-2, dit 5e-2, lm 2e-2 + argmax. `GGML_BACKEND` selects the device (empty
= CUDA0).

Audit method behind the thresholds: with a temporary F32 GGUF, GGML
matches torch F32 at 2.3e-6 relative RMS (exact semantics). The residual
at BF16 is activation quantization inside the GGML mul_mat (torch computes
F32 x F32), the same behavior as acestep.cpp.

### Cosine similarity harness

`debug-dit-cossim.py` isolates the denoising stage from the stochastic AR:
the GGML side runs the full pipeline with `--dump` on the shared
`tests/request0.json`, then the python side reloads the dumped condition
track and noise and replays transformer + vocoder in diffusers (CUDA
float32) on the same Euler CFG schedule. It reports per-probe cosines
(named DiT layers, per-step velocities and states, final x0, decoded
audio + STFT cosine) and the error growth across steps.

`./debug-dit-cossim.sh` archives the campaign as
`{backend}-[NOFUSION-]{quant}.log` over CUDA0 / Vulkan0 / CPU, fusion on
and off, quants F32 / Q8_0 / Q6_K / Q5_K_M / Q4_K_M (DiT quant varies, LM
and depth stay native so every log shares the same AR codes per backend:
the delta is the pure DiT quant effect).

Headline `dit_x0` / STFT cosines on CUDA0: F32 1.000000 / 0.999998,
Q8_0 0.999894 / 0.999793, Q6_K 0.999197 / 0.998838, Q5_K_M 0.996980 /
0.996034, Q4_K_M 0.992033 / 0.988668. Monotone degradation, fusion and
no-fusion identical.

## Performance

Measured on an RTX PRO 6000 Blackwell, 8 s of music, defaults (30 steps):

| Combo | AR ms/frame | Total |
|-------|-------------|-------|
| Full native (batch CFG) | 19.9 | 5.6 s |
| LM Q6_K | 13.9 | |
| Light (LM Q5_K_M, depth Q8_0, DiT Q4_K_M) | 10.7 | 3.9 s |

The AR stage is bandwidth-bound on weight rereads (LM + 7 depth decoder
steps per frame), so LM and depth quants convert directly into speed. The
DiT at window length T = 689 is compute-bound: DiT quants trade a slight
slowdown (dequant cost) for VRAM. The VAE decodes 30 s of audio in about
0.35 s. Without batch CFG the AR runs at 35.9 ms/frame (two forwards per
stream); fusing the cond and uncond streams into one batch of 2 (LM and
depth) brings it to 19.9.

## Patched GGML fork

Uses the same patched GGML fork as acestep.cpp (two custom ops, no
upstream kernel modified). The LM, depth decoder, condition encoder and
DiT use only standard GGML ops.

### `GGML_OP_SNAKE` (fused Snake activation)

Computes `y = x + sin^2(a * x) * inv_b` in a single kernel. The vocoder
graph recognizes the 5-op snake pattern (mul, sin, sqr, mul, add) and
fuses it, reading x once and writing y once instead of 5x the memory
traffic.

### `GGML_OP_COL2IM_1D` (scatter-add for GEMM-based conv_transpose_1d)

The vocoder decomposes each transposed convolution as `mul_mat +
col2im_1d`, routing the heavy GEMM through the backend tensor cores
instead of the naive upstream `ggml_conv_transpose_1d` kernel. The
col2im_1d gather is pure bandwidth with fused padding crop.
