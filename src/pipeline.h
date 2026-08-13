#pragma once
// pipeline.h: MiniMax Music 3 generation pipeline
//
// Owns the five GGML modules plus the tokenizer, all resident together
// (~24 GB at BF16). Load once, generate many times. The full working
// set is interleaved per frame (LM + depth) then per window (cond + DiT
// + VAE), so nothing is evictable mid-request.
//
// pipeline_ensure() reloads only the components whose path changed since
// the previous job, so switching one quantization in the UI does not
// reload the other four modules.
//
// generate() is synchronous and cancellable: the cancel flag is polled
// between AR frames, between DiT steps, and between VAE windows.

#include "bpe.h"
#include "cond-enc.h"
#include "debug.h"
#include "depth-decoder.h"
#include "dit.h"
#include "qwen3-lm.h"
#include "request.h"
#include "vae.h"

#include <atomic>
#include <string>
#include <vector>

// Runtime knobs mirroring the ace-synth debug surface. Applied to the
// components as they load, so they must be set before the first ensure
// and stay fixed for the process lifetime (graph caches bake them in).
struct MM3PipelineParams {
    bool         use_fa        = true;     // flash attention on GPU backends
    bool         use_batch_cfg = true;     // fuse the cond and uncond CFG streams in one LM decode
    bool         clamp_fp16    = false;    // clamp hidden states to FP16 range
    int          max_seq       = 0;        // LM KV cache size, 0 = model context
    const char * dump_dir      = nullptr;  // dump intermediate tensors
};

// Resolved GGUF paths for the five modules
struct MM3ModelPaths {
    std::string lm;
    std::string depth;
    std::string cond;
    std::string dit;
    std::string vae;
};

struct MM3Pipeline {
    BPETokenizer      tok;
    Qwen3LM           lm;
    DepthDecoder      depth;
    CondEnc           cond_enc;
    DiT               dit;
    FlowVAE           vae;
    MM3ModelPaths     loaded;  // empty strings until the first ensure
    MM3PipelineParams params;
    DebugDumper       dumper = {};
};

enum PipelineStatus {
    PIPELINE_OK        = 0,
    PIPELINE_FAILED    = 1,
    PIPELINE_CANCELLED = 2,
};

// Load or reload the modules whose path differs from the loaded set,
// applying params to each component as it loads.
// Returns false on any load failure; the failed component path stays
// empty so the next call retries it.
bool pipeline_ensure(MM3Pipeline * p, const MM3ModelPaths & paths, const MM3PipelineParams & params);

// Full text to audio generation. Seeds must be resolved by the caller
// (request_resolve_seed / request_resolve_lm_seed).
// audio_out: planar stereo float [L:T][R:T] at 44100 Hz, full range
// (normalization and clipping belong to the output encoding stage).
// cancel: optional, polled at stage boundaries. NULL disables cancellation.
PipelineStatus pipeline_generate(MM3Pipeline *        p,
                                 const MM3Request &   req,
                                 std::atomic<bool> *  cancel,
                                 std::vector<float> & audio_out);
