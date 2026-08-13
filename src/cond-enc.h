// cond-enc.h: condition encoder, LLM hidden state fusion for the DiT (GGML)
//
// Each 25 Hz frame carries 8 hidden states of 4096: one from the global LM
// and one per RVQ depth decoder step. They are mixed with fixed weights
// (softmax over layer_weight_logits, scaled by layer_scale, both folded at
// load), projected 4096 -> 2048 with a conv1d k=3 pad=1, then resampled to
// the 86.13 Hz VAE latent track by nearest neighbor interpolation
// (latent_length = int(n_frames * 44100 / 24000 * 960 / 512)).
//
// The mix and the resample run on CPU (8 multiply-adds and column copies),
// the projection runs as a GGML graph on the compute backend.
//
// Tensors (condition_encoder/): layer_weight_logits [8], layer_scale [1],
// proj.weight [2048, 4096, 3], proj.bias [2048].
#pragma once

#include "backend.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct CondEnc {
    float mix[8];                            // softmax(layer_weight_logits) * layer_scale, folded at load

    struct ggml_tensor *pw, *pb;             // proj [3, 4096, 2048] F16, bias [2048]
    bool                clamp_fp16 = false;  // clamp proj output on sub-Ampere CUDA (FP16 accumulation overflow)

    ggml_backend_t        backend     = nullptr;
    ggml_backend_t        cpu_backend = nullptr;
    ggml_backend_sched_t  sched       = nullptr;
    ggml_backend_buffer_t buf         = nullptr;
    struct ggml_context * weight_ctx  = nullptr;

    // Graph cache: rebuilt only when T changes
    struct ggml_context * graph_ctx    = nullptr;
    uint8_t *             graph_buf    = nullptr;
    struct ggml_cgraph *  graph        = nullptr;
    struct ggml_tensor *  graph_input  = nullptr;
    struct ggml_tensor *  graph_output = nullptr;
    int                   graph_T      = 0;

    bool load(const char * gguf_path);

    // hidden_states: [n_frames, 8, 4096] frame-major, layer-major inside a
    // frame (the reference concat layout). condition: [n_latents, 2048]
    // frame-major on the latent track.
    bool encode(const std::vector<float> & hidden_states,
                int                        n_frames,
                std::vector<float> &       condition,
                int &                      n_latents);

    void free();
};

inline bool CondEnc::load(const char * gguf_path) {
    GGUFModel gf = {};
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[Cond] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    std::vector<float> logits, scale, pw_f32v, pb_f32v;
    if (!gf_host_f32(gf, "layer_weight_logits", logits) || !gf_host_f32(gf, "layer_scale", scale) ||
        !gf_host_f32(gf, "proj.weight", pw_f32v) || !gf_host_f32(gf, "proj.bias", pb_f32v)) {
        fprintf(stderr, "[Cond] FATAL: missing tensors in %s\n", gguf_path);
        return false;
    }
    const float * pw_f32 = pw_f32v.data();
    const float * pb_f32 = pb_f32v.data();

    // Fold softmax(logits) * layer_scale into the fixed mix weights
    float mx = logits[0];
    for (int i = 1; i < 8; i++) {
        mx = logits[i] > mx ? logits[i] : mx;
    }
    float sum = 0;
    for (int i = 0; i < 8; i++) {
        mix[i] = expf(logits[i] - mx);
        sum += mix[i];
    }
    for (int i = 0; i < 8; i++) {
        mix[i] = mix[i] / sum * scale[0];
    }

    size_t                  ctx_size = ggml_tensor_overhead() * 4;
    struct ggml_init_params p        = { ctx_size, NULL, true };
    weight_ctx                       = ggml_init(p);

    pw = ggml_new_tensor_3d(weight_ctx, GGML_TYPE_F16, 3, 4096, 2048);
    pb = ggml_new_tensor_1d(weight_ctx, GGML_TYPE_F32, 2048);

    BackendPair bp = backend_init("Cond");
    backend        = bp.backend;
    cpu_backend    = bp.cpu_backend;
    sched          = backend_sched_new(bp, 64);
    buf            = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);
    if (!buf) {
        fprintf(stderr, "[Cond] FATAL: failed to allocate weight buffer\n");
        return false;
    }

    size_t                   n = (size_t) 2048 * 4096 * 3;
    std::vector<ggml_fp16_t> w16(n);
    ggml_fp32_to_fp16_row(pw_f32, w16.data(), (int64_t) n);
    ggml_backend_tensor_set(pw, w16.data(), 0, n * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(pb, pb_f32, 0, 2048 * sizeof(float));

    fprintf(stderr, "[Cond] Loaded: mix over 8 states, proj 4096 -> 2048\n");
    gf_close(&gf);
    return true;
}

inline bool CondEnc::encode(const std::vector<float> & hidden_states,
                            int                        n_frames,
                            std::vector<float> &       condition,
                            int &                      n_latents) {
    if (hidden_states.size() != (size_t) n_frames * 8 * 4096) {
        fprintf(stderr, "[Cond] hidden size %zu does not match T=%d\n", hidden_states.size(), n_frames);
        return false;
    }

    // Mix: [T, 8, 4096] -> ggml input [T, 4096] channel-major
    std::vector<float> mixed((size_t) 4096 * n_frames);
    for (int t = 0; t < n_frames; t++) {
        const float * frame = hidden_states.data() + (size_t) t * 8 * 4096;
        for (int h = 0; h < 4096; h++) {
            float acc = 0;
            for (int l = 0; l < 8; l++) {
                acc += mix[l] * frame[(size_t) l * 4096 + h];
            }
            mixed[(size_t) h * n_frames + t] = acc;
        }
    }

    // Projection graph: conv1d k=3 pad=1, [T, 4096] -> [T, 2048]
    if (graph_T != n_frames) {
        if (graph_ctx) {
            ggml_backend_sched_reset(sched);
            ggml_free(graph_ctx);
            std::free(graph_buf);
        }

        size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false);
        graph_buf       = (uint8_t *) malloc(ctx_size);
        if (!graph_buf) {
            fprintf(stderr, "[Cond] FATAL: OOM allocating graph context for T=%d\n", n_frames);
            graph_T = 0;
            return false;
        }
        struct ggml_init_params p   = { ctx_size, graph_buf, true };
        struct ggml_context *   ctx = ggml_init(p);

        graph_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_frames, 4096);
        ggml_set_name(graph_input, "cond_input");
        ggml_set_input(graph_input);

        struct ggml_tensor * y   = ggml_conv_1d(ctx, pw, graph_input, 1, 1, 1);
        y                        = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, pb, 1, 2048);
        graph_output             = ggml_add(ctx, y, b2d);
        if (clamp_fp16) {
            graph_output = ggml_clamp(ctx, graph_output, -65504.0f, 65504.0f);
        }
        ggml_set_name(graph_output, "cond_output");
        ggml_set_output(graph_output);

        graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, graph_output);

        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            fprintf(stderr, "[Cond] FATAL: graph alloc failed for T=%d\n", n_frames);
            ggml_free(ctx);
            std::free(graph_buf);
            graph_ctx = NULL;
            graph_buf = NULL;
            graph_T   = 0;
            return false;
        }

        graph_ctx = ctx;
        graph_T   = n_frames;
    }

    ggml_backend_tensor_set(graph_input, mixed.data(), 0, mixed.size() * sizeof(float));
    ggml_backend_sched_graph_compute(sched, graph);

    std::vector<float> proj((size_t) 2048 * n_frames);
    ggml_backend_tensor_get(graph_output, proj.data(), 0, proj.size() * sizeof(float));

    // Nearest resample to the latent track, output frame-major [n_latents, 2048]
    n_latents = (int) ((double) n_frames * 44100.0 / 24000.0 * 960.0 / 512.0);
    n_latents = n_latents < 1 ? 1 : n_latents;
    condition.resize((size_t) n_latents * 2048);
    for (int i = 0; i < n_latents; i++) {
        int src = (int) ((double) i * n_frames / n_latents);
        for (int c = 0; c < 2048; c++) {
            condition[(size_t) i * 2048 + c] = proj[(size_t) c * n_frames + src];
        }
    }

    return true;
}

inline void CondEnc::free() {
    if (graph_ctx) {
        ggml_free(graph_ctx);
        std::free(graph_buf);
        graph_ctx = nullptr;
        graph_buf = nullptr;
        graph_T   = 0;
    }
    if (sched) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (buf) {
        ggml_backend_buffer_free(buf);
        buf = nullptr;
    }
    if (weight_ctx) {
        ggml_free(weight_ctx);
        weight_ctx = nullptr;
    }
    // backends are refcounted and shared across all modules
    backend_release(backend, cpu_backend);
    backend     = nullptr;
    cpu_backend = nullptr;
}
