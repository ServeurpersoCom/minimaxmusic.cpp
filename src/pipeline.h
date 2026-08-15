#pragma once
// pipeline.h: MiniMax Music 3 generation pipeline
//
// The pipeline owns no GGML module: it borrows them from a ModelStore
// through RAII handles, stage by stage. The AR stage holds { LM, depth },
// the synthesis stage holds { cond, DiT, VAE }; under the default STRICT
// policy the store unloads a stage's modules when its handles go out of
// scope, so the LM weights and the DiT weights never coexist. See the
// VRAM policy doctrine in model-store.h.
//
// pipeline_configure() records the resolved paths and parameters; loads
// happen inside generate at stage boundaries, through the store. A path
// unchanged since the previous job is a store cache hit under
// --keep-loaded, a reload under STRICT.
//
// generate() is synchronous and cancellable: the cancel flag is polled
// between AR frames, between DiT steps, and between VAE windows.

#include "debug.h"
#include "model-store.h"
#include "request.h"

#include <atomic>
#include <string>
#include <vector>

// Runtime knobs mirroring the ace-synth debug surface. Applied to the
// components as they load, so they must be set before the first
// configure and stay fixed for the process lifetime (graph caches and
// store keys bake them in).
struct MM3PipelineParams {
    bool         use_fa        = true;     // flash attention on GPU backends
    bool         use_batch_cfg = true;     // fuse the cond and uncond CFG streams in one LM decode
    bool         clamp_fp16    = false;    // clamp hidden states to FP16 range
    int          max_seq       = 0;        // LM KV cache size, 0 = model context
    int          max_batch     = 1;        // song batch limit, sizes the 2N LM KV sets at load
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
    ModelStore *      store = nullptr;  // borrowed, owned by the tool
    MM3ModelPaths     wanted;           // empty strings until the first configure
    MM3PipelineParams params;
    DebugDumper       dumper = {};
};

enum PipelineStatus {
    PIPELINE_OK        = 0,
    PIPELINE_FAILED    = 1,
    PIPELINE_CANCELLED = 2,
};

// Record the resolved paths and parameters for the next generate. No
// module is loaded here; a load failure surfaces as PIPELINE_FAILED
// from the generate that first requires the failing component.
void pipeline_configure(MM3Pipeline * p, const MM3ModelPaths & paths, const MM3PipelineParams & params);

// Full text to audio generation. Seeds must be resolved by the caller
// (request_resolve_seed / request_resolve_lm_seed).
// tracks_out: lm_batch_size tracks, each planar stereo float [L:T][R:T]
// at 44100 Hz, full range (normalization and clipping belong to the
// output encoding stage). Song i samples with lm_seed + i.
// cancel: optional, polled at stage boundaries. NULL disables cancellation.
// codes_out: optional, the audio_codes stream of each song, identical
// to the input codes under replay. Feeds request_replay.
PipelineStatus pipeline_generate(MM3Pipeline *                     p,
                                 const MM3Request &                req,
                                 std::atomic<bool> *               cancel,
                                 std::vector<std::vector<float>> & tracks_out,
                                 std::vector<std::string> *        codes_out);

// Autoregressive stage only: lm_batch_size code streams out, no
// synthesis. codes_out[i] is the audio_codes string of song i (8 comma
// separated codes per frame). Only requires the AR group from the store.
PipelineStatus pipeline_lm_generate(MM3Pipeline *              p,
                                    const MM3Request &         req,
                                    std::atomic<bool> *        cancel,
                                    std::vector<std::string> & codes_out);
