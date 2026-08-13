// dit.h: flow matching transformer (2.4B), 1D DiT over VAE latents (GGML)
//
// 36 self attention blocks, model dim 2048 (32 heads x 64), fused SwiGLU FF
// (ff_in emits value then gate, output = ff_out(value * silu(gate))),
// LayerNorm with bias, partial NeoX RoPE on the first 32 dims per head,
// theta 10000, bidirectional attention.
//
// The input concatenates [latent(128), zeros(128), condition(2048)] along
// channels (the zeros slot carries context latents during training),
// preprocess_conv and postprocess_conv are residual k=1 convs
// (conv(x) + x). The timestep enters as one prefix token: trained Fourier
// features -> linear -> SiLU -> linear, prepended before the blocks and
// dropped after them. The timestep embedding runs on CPU (4.7M MACs), the
// transformer runs as a cached GGML graph on the compute backend.
//
// Tensors (transformer/): time_proj.weight [128, 1],
// time_embed.linear_{1,2}.{weight, bias}, preprocess_conv.weight,
// proj_in.weight, transformer_blocks.N.{norm1.{weight, bias},
// attn.to_{q,k,v}.weight, attn.to_out.0.weight, norm2.{weight, bias},
// ff_in.{weight, bias}, ff_out.{weight, bias}}, proj_out.weight,
// postprocess_conv.weight.
#pragma once

#include "backend.h"
#include "debug.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct DiTBlock {
    struct ggml_tensor *norm1_w, *norm1_b;
    struct ggml_tensor *wq, *wk, *wv, *wo;
    struct ggml_tensor *norm2_w, *norm2_b;
    struct ggml_tensor *ff_in_w, *ff_in_b;
    struct ggml_tensor *ff_out_w, *ff_out_b;
};

struct DiT {
    static const int DIM        = 2048;
    static const int IN_CH      = 128;
    static const int COND_DIM   = 2048;
    static const int CONCAT_CH  = 2 * IN_CH + COND_DIM;
    static const int N_LAYERS   = 36;
    static const int N_HEADS    = 32;
    static const int HEAD_DIM   = 64;
    static const int FF_INNER   = 8192;
    static const int ROTARY_DIM = 32;
    static const int FOURIER    = 256;

    struct ggml_tensor * pre_w;   // preprocess_conv [2304, 2304] reshaped at load
    struct ggml_tensor * in_w;    // proj_in [2304, 2048]
    DiTBlock             blocks[N_LAYERS];
    struct ggml_tensor * out_w;   // proj_out [2048, 128]
    struct ggml_tensor * post_w;  // postprocess_conv [128, 128] reshaped at load

    // Timestep embedding, host side F32 copies
    std::vector<float> fourier_w;     // [128]
    std::vector<float> te1_w, te1_b;  // [2048, 256], [2048]
    std::vector<float> te2_w, te2_b;  // [2048, 2048], [2048]

    ggml_backend_t       backend        = nullptr;
    ggml_backend_t       cpu_backend    = nullptr;
    ggml_backend_sched_t sched          = nullptr;
    WeightCtx            wctx           = {};
    bool                 use_flash_attn = false;
    bool                 clamp_fp16     = false;  // clamp hidden states on sub-Ampere CUDA (FP16 accumulation overflow)

    // Graph cache: rebuilt only when T changes
    struct ggml_context * graph_ctx    = nullptr;
    uint8_t *             graph_buf    = nullptr;
    struct ggml_cgraph *  graph        = nullptr;
    struct ggml_tensor *  in_xt        = nullptr;
    struct ggml_tensor *  in_cond      = nullptr;
    struct ggml_tensor *  in_temb      = nullptr;
    struct ggml_tensor *  in_pos       = nullptr;
    struct ggml_tensor *  graph_output = nullptr;
    int                   graph_T      = 0;

    bool load(const char * gguf_path);

    // One denoising evaluation: xt [T, 128] time-major, cond [T, 2048]
    // time-major (zeros for the unconditional CFG branch), t in [0, 1]
    // (0 = noise, 1 = data). Writes the velocity [T, 128] time-major.
    bool forward(const float * xt, const float * cond, int T, float t, float * velocity);

    // Dump the named probe tensors of the last computed graph (cossim harness)
    void dump_named(const DebugDumper * dbg);

    void free();
};

inline bool DiT::load(const char * gguf_path) {
    GGUFModel gf = {};
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[DiT] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    BackendPair bp = backend_init("DiT");
    backend        = bp.backend;
    cpu_backend    = bp.cpu_backend;
    sched          = backend_sched_new(bp, 4096);
    use_flash_attn = bp.has_gpu;

    wctx_init(&wctx, 512);

    // k=1 convs [1, C, C] flattened to 2D [C_in, C_out] for mul_mat
    static const int64_t pre_shape[2]  = { CONCAT_CH, CONCAT_CH };
    static const int64_t post_shape[2] = { IN_CH, IN_CH };
    pre_w                              = gf_load_tensor(&wctx, gf, "preprocess_conv.weight", pre_shape, 2);
    in_w                               = gf_load_tensor(&wctx, gf, "proj_in.weight");
    out_w                              = gf_load_tensor(&wctx, gf, "proj_out.weight");
    post_w                             = gf_load_tensor(&wctx, gf, "postprocess_conv.weight", post_shape, 2);

    for (int i = 0; i < N_LAYERS; i++) {
        DiTBlock &  b   = blocks[i];
        std::string pfx = "transformer_blocks." + std::to_string(i) + ".";
        b.norm1_w       = gf_load_tensor_f32(&wctx, gf, pfx + "norm1.weight");
        b.norm1_b       = gf_load_tensor_f32(&wctx, gf, pfx + "norm1.bias");
        b.wq            = gf_load_tensor(&wctx, gf, pfx + "attn.to_q.weight");
        b.wk            = gf_load_tensor(&wctx, gf, pfx + "attn.to_k.weight");
        b.wv            = gf_load_tensor(&wctx, gf, pfx + "attn.to_v.weight");
        b.wo            = gf_load_tensor(&wctx, gf, pfx + "attn.to_out.0.weight");
        b.norm2_w       = gf_load_tensor_f32(&wctx, gf, pfx + "norm2.weight");
        b.norm2_b       = gf_load_tensor_f32(&wctx, gf, pfx + "norm2.bias");
        b.ff_in_w       = gf_load_tensor(&wctx, gf, pfx + "ff_in.weight");
        b.ff_in_b       = gf_load_tensor_f32(&wctx, gf, pfx + "ff_in.bias");
        b.ff_out_w      = gf_load_tensor(&wctx, gf, pfx + "ff_out.weight");
        b.ff_out_b      = gf_load_tensor_f32(&wctx, gf, pfx + "ff_out.bias");
    }

    if (!wctx_alloc(&wctx, backend)) {
        return false;
    }

    gf_host_f32(gf, "time_proj.weight", fourier_w);
    gf_host_f32(gf, "time_embed.linear_1.weight", te1_w);
    gf_host_f32(gf, "time_embed.linear_1.bias", te1_b);
    gf_host_f32(gf, "time_embed.linear_2.weight", te2_w);
    gf_host_f32(gf, "time_embed.linear_2.bias", te2_b);

    fprintf(stderr, "[DiT] Loaded: %d layers, dim %d, flash_attn=%d\n", N_LAYERS, DIM, use_flash_attn);
    gf_close(&gf);
    return true;
}

// LayerNorm with weight and bias, eps 1e-5
static struct ggml_tensor * dit_layer_norm(struct ggml_context * ctx,
                                           struct ggml_tensor *  x,
                                           struct ggml_tensor *  w,
                                           struct ggml_tensor *  b) {
    struct ggml_tensor * n = ggml_norm(ctx, x, 1e-5f);
    n                      = ggml_mul(ctx, n, w);
    return ggml_add(ctx, n, b);
}

// F32 manual attention (CPU fallback), bidirectional, no mask
static struct ggml_tensor * dit_attn_f32(struct ggml_context * ctx,
                                         struct ggml_tensor *  q,
                                         struct ggml_tensor *  k,
                                         struct ggml_tensor *  v,
                                         float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_soft_max_ext(ctx, scores, NULL, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

static struct ggml_tensor * dit_block(struct ggml_context * ctx,
                                      DiT *                 m,
                                      DiTBlock *            b,
                                      struct ggml_tensor *  x,  // [2048, S]
                                      struct ggml_tensor *  pos,
                                      int                   S,
                                      const char *          sa_name) {
    const int D  = DiT::HEAD_DIM;
    const int Nh = DiT::N_HEADS;

    // Attention
    struct ggml_tensor * h = dit_layer_norm(ctx, x, b->norm1_w, b->norm1_b);

    struct ggml_tensor * q = ggml_mul_mat(ctx, b->wq, h);
    struct ggml_tensor * k = ggml_mul_mat(ctx, b->wk, h);
    struct ggml_tensor * v = ggml_mul_mat(ctx, b->wv, h);

    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nh, S);
    v = ggml_reshape_3d(ctx, v, D, Nh, S);

    // Partial NeoX RoPE: only the first 32 dims of each head rotate
    q = ggml_rope_ext(ctx, q, pos, NULL, DiT::ROTARY_DIM, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    k = ggml_rope_ext(ctx, k, pos, NULL, DiT::ROTARY_DIM, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    // [D, Nh, S] -> [D, S, Nh]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    float                scale = 1.0f / sqrtf((float) D);
    struct ggml_tensor * attn;
    if (m->use_flash_attn) {
        k    = ggml_cast(ctx, k, GGML_TYPE_F16);
        v    = ggml_cast(ctx, v, GGML_TYPE_F16);
        attn = ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = dit_attn_f32(ctx, q, k, v, scale);
    }
    attn = ggml_reshape_2d(ctx, attn, Nh * D, S);

    struct ggml_tensor * sa_out = ggml_mul_mat(ctx, b->wo, attn);
    if (sa_name) {
        ggml_set_name(sa_out, sa_name);
        ggml_set_output(sa_out);
    }
    x = ggml_add(ctx, x, sa_out);

    // FF: ff_in emits [value(8192), gate(8192)], out = ff_out(value * silu(gate))
    h                          = dit_layer_norm(ctx, x, b->norm2_w, b->norm2_b);
    struct ggml_tensor * ff    = ggml_mul_mat(ctx, b->ff_in_w, h);
    ff                         = ggml_add(ctx, ff, b->ff_in_b);
    struct ggml_tensor * value = ggml_cont(ctx, ggml_view_2d(ctx, ff, DiT::FF_INNER, S, ff->nb[1], 0));
    struct ggml_tensor * gate =
        ggml_cont(ctx, ggml_view_2d(ctx, ff, DiT::FF_INNER, S, ff->nb[1], (size_t) DiT::FF_INNER * ff->nb[0]));
    struct ggml_tensor * act = ggml_swiglu_split(ctx, gate, value);
    struct ggml_tensor * out = ggml_mul_mat(ctx, b->ff_out_w, act);
    out                      = ggml_add(ctx, out, b->ff_out_b);

    return ggml_add(ctx, x, out);
}

static struct ggml_tensor * dit_build_graph(struct ggml_context * ctx, DiT * m, int T) {
    int S = T + 1;

    // Channel concat [latent, zeros, condition] -> [2304, T]
    struct ggml_tensor * zeros = ggml_scale(ctx, m->in_xt, 0.0f);
    struct ggml_tensor * x     = ggml_concat(ctx, m->in_xt, zeros, 0);
    x                          = ggml_concat(ctx, x, m->in_cond, 0);

    // Residual k=1 convs as mul_mat
    x = ggml_add(ctx, ggml_mul_mat(ctx, m->pre_w, x), x);
    ggml_set_name(x, "hidden_after_preprocess");
    ggml_set_output(x);

    // proj_in then prepend the timestep token
    x = ggml_mul_mat(ctx, m->in_w, x);
    ggml_set_name(x, "hidden_after_proj_in");
    ggml_set_output(x);
    x = ggml_concat(ctx, m->in_temb, x, 1);

    for (int i = 0; i < DiT::N_LAYERS; i++) {
        x = dit_block(ctx, m, &m->blocks[i], x, m->in_pos, S, i == 0 ? "layer0_sa_output" : NULL);
        if (m->clamp_fp16) {
            x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
        }
        // Named probes at key depths for the cossim harness
        if (i == 0 || i == 6 || i == 12 || i == 18 || i == DiT::N_LAYERS - 1) {
            char lname[64];
            snprintf(lname, sizeof(lname), "hidden_after_layer%d", i);
            ggml_set_name(x, lname);
            ggml_set_output(x);
        }
    }

    // Drop the timestep token, project back to latent channels
    struct ggml_tensor * tokens = ggml_view_2d(ctx, x, DiT::DIM, T, x->nb[1], x->nb[1]);
    struct ggml_tensor * y      = ggml_mul_mat(ctx, m->out_w, ggml_cont(ctx, tokens));
    y                           = ggml_add(ctx, ggml_mul_mat(ctx, m->post_w, y), y);
    return y;  // [128, T]
}

// CPU timestep embedding: Fourier features -> linear -> SiLU -> linear
static void dit_time_embed(DiT * m, float t, std::vector<float> & temb) {
    float emb[DiT::FOURIER];
    for (int i = 0; i < 128; i++) {
        float angle  = 2.0f * (float) M_PI * t * m->fourier_w[i];
        emb[i]       = cosf(angle);
        emb[128 + i] = sinf(angle);
    }
    std::vector<float> h(DiT::DIM);
    for (int j = 0; j < DiT::DIM; j++) {
        float acc = m->te1_b[j];
        for (int i = 0; i < DiT::FOURIER; i++) {
            acc += m->te1_w[(size_t) j * DiT::FOURIER + i] * emb[i];
        }
        h[j] = acc / (1.0f + expf(-acc)) * 1.0f;  // SiLU
    }
    temb.resize(DiT::DIM);
    for (int j = 0; j < DiT::DIM; j++) {
        float acc = m->te2_b[j];
        for (int i = 0; i < DiT::DIM; i++) {
            acc += m->te2_w[(size_t) j * DiT::DIM + i] * h[i];
        }
        temb[j] = acc;
    }
}

inline bool DiT::forward(const float * xt, const float * cond, int T, float t, float * velocity) {
    if (graph_T != T) {
        if (graph_ctx) {
            ggml_backend_sched_reset(sched);
            ggml_free(graph_ctx);
            std::free(graph_buf);
        }

        size_t ctx_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false);
        graph_buf       = (uint8_t *) malloc(ctx_size);
        if (!graph_buf) {
            fprintf(stderr, "[DiT] FATAL: OOM allocating graph context for T=%d\n", T);
            graph_T = 0;
            return false;
        }
        struct ggml_init_params p   = { ctx_size, graph_buf, true };
        struct ggml_context *   ctx = ggml_init(p);

        in_xt   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IN_CH, T);
        in_cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, COND_DIM, T);
        in_temb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIM, 1);
        ggml_set_name(in_temb, "temb_t");
        in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T + 1);
        ggml_set_input(in_xt);
        ggml_set_input(in_cond);
        ggml_set_input(in_temb);
        ggml_set_input(in_pos);

        graph_output = dit_build_graph(ctx, this, T);
        ggml_set_name(graph_output, "dit_output");
        ggml_set_output(graph_output);

        graph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(graph, graph_output);

        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            fprintf(stderr, "[DiT] FATAL: graph alloc failed for T=%d\n", T);
            ggml_free(ctx);
            std::free(graph_buf);
            graph_ctx = NULL;
            graph_buf = NULL;
            graph_T   = 0;
            return false;
        }

        graph_ctx = ctx;
        graph_T   = T;
        fprintf(stderr, "[DiT] Graph: %d nodes, T=%d\n", ggml_graph_n_nodes(graph), T);
    }

    std::vector<float> temb;
    dit_time_embed(this, t, temb);

    std::vector<int32_t> pos(T + 1);
    for (int i = 0; i <= T; i++) {
        pos[i] = i;
    }

    ggml_backend_tensor_set(in_xt, xt, 0, (size_t) IN_CH * T * sizeof(float));
    ggml_backend_tensor_set(in_cond, cond, 0, (size_t) COND_DIM * T * sizeof(float));
    ggml_backend_tensor_set(in_temb, temb.data(), 0, DIM * sizeof(float));
    ggml_backend_tensor_set(in_pos, pos.data(), 0, (T + 1) * sizeof(int32_t));

    ggml_backend_sched_graph_compute(sched, graph);

    ggml_backend_tensor_get(graph_output, velocity, 0, (size_t) IN_CH * T * sizeof(float));
    return true;
}

inline void DiT::free() {
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
    wctx_free(&wctx);
    // backends are refcounted and shared across all modules
    backend_release(backend, cpu_backend);
    backend     = nullptr;
    cpu_backend = nullptr;
}

// Reads each named probe tensor from the cached graph and dumps it
// time-major [ne1, ne0], matching the torch reference hook layouts.
inline void DiT::dump_named(const DebugDumper * dbg) {
    if (!dbg || !dbg->enabled || !graph) {
        return;
    }
    const char * names[] = { "temb_t",
                             "hidden_after_preprocess",
                             "hidden_after_proj_in",
                             "layer0_sa_output",
                             "hidden_after_layer0",
                             "hidden_after_layer6",
                             "hidden_after_layer12",
                             "hidden_after_layer18",
                             "hidden_after_layer35" };
    for (const char * name : names) {
        struct ggml_tensor * t = ggml_graph_get_tensor(graph, name);
        if (!t) {
            continue;
        }
        int64_t            n0 = t->ne[0];
        int64_t            n1 = t->ne[1];
        std::vector<float> buf((size_t) n0 * n1);
        ggml_backend_tensor_get(t, buf.data(), 0, (size_t) n0 * n1 * sizeof(float));
        if (n1 <= 1) {
            debug_dump_1d(dbg, name, buf.data(), (int) n0);
        } else {
            debug_dump_2d(dbg, name, buf.data(), (int) n1, (int) n0);
        }
    }
}
