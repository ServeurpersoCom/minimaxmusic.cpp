// vae.h: flow VAE decoder (123M), DAC style upsampling stack (GGML)
//
// Decodes the 128 channel flow VAE latent at 86.13 Hz into 44.1 kHz audio.
// The 128 latent channels are two independent 64 channel tracks, one per
// stereo side: the decoder runs once per side and emits a mono waveform.
//
// Topology (vocoder/):
//   dec_in_proj: conv1d 64 -> 1024 k=1
//   conv_in:     conv1d 1024 -> 1536 k=7
//   blocks.{0..3}: snake1 -> conv_t1 (stride s, k = 2s, pad s/2) -> 3 x res_unit
//     strides [8, 8, 4, 2], dims 1536 -> 768 -> 384 -> 192 -> 96
//   ResUnit(dil 1/3/9): skip -> snake1 -> conv1 k=7 dil -> snake2 -> conv2 k=1 -> + skip
//   snake_out -> conv_out: conv1d 96 -> 1 k=7 -> tanh
//
// Snake: y = x + sin(alpha * x)^2 / (alpha + 1e-9), emitted as the 5-op
// decomposition (mul -> sin -> sqr -> mul -> add) that backends pattern-match
// into their fused snake kernel.
// ConvTranspose1d: mul_mat on the pre-permuted [IC, K*OC] weight + col2im_1d.
// Weight norm is folded at conversion (convert.py), the GGUF holds plain
// F32 weights, conv weights are cast to F16 at load.
#pragma once

#include "backend.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct VAEResUnit {
    struct ggml_tensor *s1a, *s1i;  // snake1 alpha, 1/(alpha + 1e-9) [1, C]
    struct ggml_tensor *c1w, *c1b;  // conv1 [7, C, C], bias [C]
    struct ggml_tensor *s2a, *s2i;  // snake2
    struct ggml_tensor *c2w, *c2b;  // conv2 [1, C, C], bias [C]
    int                 dilation;
};

struct VAEBlock {
    struct ggml_tensor *sa, *si;    // snake1 alpha, inv [1, in_ch]
    struct ggml_tensor *ctw, *ctb;  // conv_t1 F16 [IC, K*OC] pre-permuted, bias [out_ch]
    int                 in_ch, out_ch, stride, kernel;
    VAEResUnit          ru[3];
};

struct FlowVAE {
    struct ggml_tensor *piw, *pib;  // dec_in_proj [1, 64, 1024], bias [1024]
    struct ggml_tensor *c1w, *c1b;  // conv_in [7, 1024, 1536], bias [1536]
    VAEBlock            blk[4];
    struct ggml_tensor *sa, *si;    // snake_out alpha, inv [1, 96]
    struct ggml_tensor *c2w, *c2b;  // conv_out [7, 96, 1], bias [1]

    ggml_backend_t        backend     = nullptr;
    ggml_backend_t        cpu_backend = nullptr;
    ggml_backend_sched_t  sched       = nullptr;
    ggml_backend_buffer_t buf         = nullptr;
    struct ggml_context * weight_ctx  = nullptr;

    // Graph cache: rebuilt only when T_latent changes
    struct ggml_context * graph_ctx    = nullptr;
    uint8_t *             graph_buf    = nullptr;
    struct ggml_cgraph *  graph        = nullptr;
    struct ggml_tensor *  graph_input  = nullptr;
    struct ggml_tensor *  graph_output = nullptr;
    int                   graph_T      = 0;

    bool load(const char * gguf_path);

    // Decodes the latent [128, T] channel-major (torch memory order) into
    // interleaved stereo samples at 44.1 kHz (T * 512 frames per side).
    bool decode(const std::vector<float> & latent, int n_latents, std::vector<float> & audio);

    void free();
};

// Load helpers (GGUF holds folded F32 weights)

// Conv weight -> F16 tensor, same layout
static void vae_load_conv(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name) {
    std::vector<float> w;
    if (!gf_host_f32(gf, name.c_str(), w)) {
        fprintf(stderr, "[VAE] FATAL: missing tensor %s\n", name.c_str());
        exit(1);
    }
    std::vector<ggml_fp16_t> w16(w.size());
    ggml_fp32_to_fp16_row(w.data(), w16.data(), (int64_t) w.size());
    ggml_backend_tensor_set(dst, w16.data(), 0, w16.size() * sizeof(ggml_fp16_t));
}

// ConvTranspose1d weight [K, OC, IC] -> F16 [IC, K*OC] pre-permuted for
// mul_mat, col rows ordered oc-major k-minor to match col2im_1d
static void vae_load_conv_t(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * mt = ggml_get_tensor(gf.meta, name.c_str());
    std::vector<float>   wv;
    if (!mt || !gf_host_f32(gf, name.c_str(), wv)) {
        fprintf(stderr, "[VAE] FATAL: missing tensor %s\n", name.c_str());
        exit(1);
    }
    const float *            w  = wv.data();
    int                      K  = (int) mt->ne[0];
    int                      OC = (int) mt->ne[1];
    int                      IC = (int) mt->ne[2];
    std::vector<ggml_fp16_t> p((size_t) IC * K * OC);
    for (int ic = 0; ic < IC; ic++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int k = 0; k < K; k++) {
                p[(size_t) (oc * K + k) * IC + ic] = ggml_fp32_to_fp16(w[((size_t) ic * OC + oc) * K + k]);
            }
        }
    }
    ggml_backend_tensor_set(dst, p.data(), 0, p.size() * sizeof(ggml_fp16_t));
}

// 1D tensor -> F32 tensor
static void vae_load_f32(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name) {
    std::vector<float> w;
    if (!gf_host_f32(gf, name.c_str(), w)) {
        fprintf(stderr, "[VAE] FATAL: missing tensor %s\n", name.c_str());
        exit(1);
    }
    ggml_backend_tensor_set(dst, w.data(), 0, w.size() * sizeof(float));
}

// Snake alpha [C] -> two F32 [1, C]: alpha and 1/(alpha + 1e-9)
static void vae_load_snake(struct ggml_tensor * dst_a,
                           struct ggml_tensor * dst_i,
                           const GGUFModel &    gf,
                           const std::string &  name) {
    std::vector<float> a;
    if (!gf_host_f32(gf, name.c_str(), a)) {
        fprintf(stderr, "[VAE] FATAL: missing tensor %s\n", name.c_str());
        exit(1);
    }
    std::vector<float> inv(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        inv[i] = 1.0f / (a[i] + 1e-9f);
    }
    ggml_backend_tensor_set(dst_a, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_set(dst_i, inv.data(), 0, inv.size() * sizeof(float));
}

inline bool FlowVAE::load(const char * gguf_path) {
    GGUFModel gf = {};
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[VAE] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    static const int strides[]   = { 8, 8, 4, 2 };
    static const int in_ch[]     = { 1536, 768, 384, 192 };
    static const int out_ch[]    = { 768, 384, 192, 96 };
    static const int dilations[] = { 1, 3, 9 };

    // Phase 1: tensor metadata (no_alloc context)
    size_t                  ctx_size = ggml_tensor_overhead() * 128;
    struct ggml_init_params p        = { ctx_size, NULL, true };
    weight_ctx                       = ggml_init(p);
    struct ggml_context * ctx        = weight_ctx;

    piw = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, 64, 1024);
    pib = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 1024, 1536);
    c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1536);

    for (int i = 0; i < 4; i++) {
        VAEBlock & b = blk[i];
        b.in_ch      = in_ch[i];
        b.out_ch     = out_ch[i];
        b.stride     = strides[i];
        b.kernel     = strides[i] * 2;
        int C        = out_ch[i];
        b.sa         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, in_ch[i]);
        b.si         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, in_ch[i]);
        b.ctw        = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, in_ch[i], b.kernel * out_ch[i]);
        b.ctb        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch[i]);
        for (int r = 0; r < 3; r++) {
            VAEResUnit & ru = b.ru[r];
            ru.dilation     = dilations[r];
            ru.s1a          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s1i          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c1w          = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, C, C);
            ru.c1b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            ru.s2a          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s2i          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c2w          = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, C, C);
            ru.c2b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        }
    }
    sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 96);
    si  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 96);
    c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 96, 1);
    c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    // Phase 2: backend buffer
    BackendPair bp = backend_init("VAE");
    backend        = bp.backend;
    cpu_backend    = bp.cpu_backend;
    sched          = backend_sched_new(bp, 8192);
    buf            = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "[VAE] FATAL: failed to allocate weight buffer\n");
        return false;
    }
    fprintf(stderr, "[VAE] Backend: %s, Weight buffer: %.1f MB\n", ggml_backend_name(backend),
            (float) ggml_backend_buffer_get_size(buf) / (1024 * 1024));

    // Phase 3: load weights
    vae_load_conv(piw, gf, "dec_in_proj.weight");
    vae_load_f32(pib, gf, "dec_in_proj.bias");
    vae_load_conv(c1w, gf, "conv_in.weight");
    vae_load_f32(c1b, gf, "conv_in.bias");

    for (int i = 0; i < 4; i++) {
        VAEBlock &  b       = blk[i];
        std::string blk_pfx = "blocks." + std::to_string(i);
        vae_load_snake(b.sa, b.si, gf, blk_pfx + ".snake1.alpha");
        vae_load_conv_t(b.ctw, gf, blk_pfx + ".conv_t1.weight");
        vae_load_f32(b.ctb, gf, blk_pfx + ".conv_t1.bias");
        for (int r = 0; r < 3; r++) {
            VAEResUnit & ru = b.ru[r];
            std::string  rp = blk_pfx + ".res_unit" + std::to_string(r + 1);
            vae_load_snake(ru.s1a, ru.s1i, gf, rp + ".snake1.alpha");
            vae_load_conv(ru.c1w, gf, rp + ".conv1.weight");
            vae_load_f32(ru.c1b, gf, rp + ".conv1.bias");
            vae_load_snake(ru.s2a, ru.s2i, gf, rp + ".snake2.alpha");
            vae_load_conv(ru.c2w, gf, rp + ".conv2.weight");
            vae_load_f32(ru.c2b, gf, rp + ".conv2.bias");
        }
    }
    vae_load_snake(sa, si, gf, "snake_out.alpha");
    vae_load_conv(c2w, gf, "conv_out.weight");
    vae_load_f32(c2b, gf, "conv_out.bias");

    fprintf(stderr, "[VAE] Loaded: 4 blocks, upsample=512x, F32 activations\n");
    gf_close(&gf);
    return true;
}

// Graph building

// Snake activation, 5-op decomposition for backend pattern fusion
// y = x + sin(alpha * x)^2 * inv, x: [T, C], alpha/inv: [1, C]
static struct ggml_tensor * vae_snake(struct ggml_context * ctx,
                                      struct ggml_tensor *  x,
                                      struct ggml_tensor *  alpha,
                                      struct ggml_tensor *  inv) {
    struct ggml_tensor * ax = ggml_mul(ctx, x, alpha);
    struct ggml_tensor * s  = ggml_sin(ctx, ax);
    struct ggml_tensor * s2 = ggml_sqr(ctx, s);
    struct ggml_tensor * d  = ggml_mul(ctx, s2, inv);
    return ggml_add(ctx, x, d);
}

// Conv1d + bias: data [T, IC] -> [T_out, OC]
static struct ggml_tensor * vae_conv1d(struct ggml_context * ctx,
                                       struct ggml_tensor *  w,  // [K, IC, OC] F16
                                       struct ggml_tensor *  b,  // [OC] or NULL
                                       struct ggml_tensor *  x,  // [T, IC]
                                       int                   stride,
                                       int                   padding,
                                       int                   dilation) {
    struct ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, padding, dilation);
    y                      = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (b) {
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y                        = ggml_add(ctx, y, b2d);
    }
    return y;
}

// ConvTranspose1d via GEMM + col2im_1d
// w: [IC, K*OC] pre-permuted at load, x: [T_in, IC] -> [T_out, OC]
static struct ggml_tensor * vae_conv_t1d(struct ggml_context * ctx,
                                         struct ggml_tensor *  w,
                                         struct ggml_tensor *  b,
                                         struct ggml_tensor *  x,
                                         int                   stride,
                                         int                   padding,
                                         int                   oc) {
    struct ggml_tensor * xt  = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor * col = ggml_mul_mat(ctx, w, xt);
    struct ggml_tensor * y   = ggml_col2im_1d(ctx, col, stride, oc, padding);
    if (b) {
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y                        = ggml_add(ctx, y, b2d);
    }
    return y;
}

static struct ggml_tensor * vae_res_unit(struct ggml_context * ctx, VAEResUnit * ru, struct ggml_tensor * x) {
    struct ggml_tensor * skip = x;
    int                  pad  = 3 * ru->dilation;
    x                         = vae_snake(ctx, x, ru->s1a, ru->s1i);
    x                         = vae_conv1d(ctx, ru->c1w, ru->c1b, x, 1, pad, ru->dilation);
    x                         = vae_snake(ctx, x, ru->s2a, ru->s2i);
    x                         = vae_conv1d(ctx, ru->c2w, ru->c2b, x, 1, 0, 1);
    return ggml_add(ctx, skip, x);
}

// One stereo side: latent [T, 64] -> waveform [T * 512, 1]
static struct ggml_tensor * vae_build_graph(struct ggml_context * ctx, FlowVAE * m, struct ggml_tensor * latent) {
    struct ggml_tensor * x = vae_conv1d(ctx, m->piw, m->pib, latent, 1, 0, 1);
    x                      = vae_conv1d(ctx, m->c1w, m->c1b, x, 1, 3, 1);

    for (int i = 0; i < 4; i++) {
        VAEBlock & b = m->blk[i];
        x            = vae_snake(ctx, x, b.sa, b.si);
        int pad      = (b.kernel - b.stride) / 2;
        x            = vae_conv_t1d(ctx, b.ctw, b.ctb, x, b.stride, pad, b.out_ch);
        for (int r = 0; r < 3; r++) {
            x = vae_res_unit(ctx, &b.ru[r], x);
        }
    }

    x = vae_snake(ctx, x, m->sa, m->si);
    x = vae_conv1d(ctx, m->c2w, m->c2b, x, 1, 3, 1);
    return ggml_tanh(ctx, x);
}

// Ensures the graph is cached for T_latent, runs one side, returns T_audio or -1
static int vae_compute(FlowVAE * m, const float * latent_side, int T_latent) {
    if (m->graph_T != T_latent) {
        if (m->graph_ctx) {
            ggml_backend_sched_reset(m->sched);
            ggml_free(m->graph_ctx);
            std::free(m->graph_buf);
        }

        size_t ctx_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false);
        m->graph_buf    = (uint8_t *) malloc(ctx_size);
        if (!m->graph_buf) {
            fprintf(stderr, "[VAE] FATAL: OOM allocating graph context for T=%d\n", T_latent);
            m->graph_T = 0;
            return -1;
        }
        struct ggml_init_params p   = { ctx_size, m->graph_buf, true };
        struct ggml_context *   ctx = ggml_init(p);

        m->graph_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_latent, 64);
        ggml_set_name(m->graph_input, "vae_input");
        ggml_set_input(m->graph_input);

        m->graph_output = vae_build_graph(ctx, m, m->graph_input);
        ggml_set_name(m->graph_output, "vae_output");
        ggml_set_output(m->graph_output);

        m->graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(m->graph, m->graph_output);

        if (!ggml_backend_sched_alloc_graph(m->sched, m->graph)) {
            fprintf(stderr, "[VAE] FATAL: graph alloc failed for T=%d\n", T_latent);
            ggml_free(ctx);
            std::free(m->graph_buf);
            m->graph_ctx = NULL;
            m->graph_buf = NULL;
            m->graph_T   = 0;
            return -1;
        }

        m->graph_ctx = ctx;
        m->graph_T   = T_latent;
        fprintf(stderr, "[VAE] Graph: %d nodes, T_latent=%d\n", ggml_graph_n_nodes(m->graph), T_latent);
    }

    ggml_backend_tensor_set(m->graph_input, latent_side, 0, (size_t) 64 * T_latent * sizeof(float));
    ggml_backend_sched_graph_compute(m->sched, m->graph);

    return (int) m->graph_output->ne[0];
}

inline bool FlowVAE::decode(const std::vector<float> & latent, int n_latents, std::vector<float> & audio) {
    if ((int) latent.size() != 128 * n_latents) {
        fprintf(stderr, "[VAE] latent size %zu does not match T=%d\n", latent.size(), n_latents);
        return false;
    }

    int T_audio = n_latents * 512;
    audio.resize((size_t) T_audio * 2);
    std::vector<float> side(T_audio);

    for (int ch = 0; ch < 2; ch++) {
        int T_out = vae_compute(this, latent.data() + (size_t) ch * 64 * n_latents, n_latents);
        if (T_out != T_audio) {
            fprintf(stderr, "[VAE] decode failed on side %d (T_out=%d)\n", ch, T_out);
            return false;
        }
        ggml_backend_tensor_get(graph_output, side.data(), 0, (size_t) T_audio * sizeof(float));
        for (int t = 0; t < T_audio; t++) {
            audio[(size_t) t * 2 + ch] = side[t];
        }
    }

    fprintf(stderr, "[VAE] Decoded: T_latent=%d -> T_audio=%d (%.2fs @ 44.1kHz)\n", n_latents, T_audio,
            (float) T_audio / 44100.0f);
    return true;
}

inline void FlowVAE::free() {
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
