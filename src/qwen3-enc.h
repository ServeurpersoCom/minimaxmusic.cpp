// qwen3-enc.h: Qwen3 transformer encoder via ggml
//
// Generic Qwen3 backbone used by:
//   Global LM (MiniMax Music 3): 36L, H=4096, causal, GQA 32/8, vocab lookup
//
// Architecture per layer:
//   RMSNorm -> Q/K/V proj -> QK-Norm -> RoPE -> GQA -> O proj -> +residual
//   RMSNorm -> gate/up proj -> SwiGLU -> down proj -> +residual
// Final: RMSNorm

#pragma once
#include "backend.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define QWEN3_MAX_LAYERS 32

// Config
struct Qwen3Config {
    int   hidden_size;        // H
    int   intermediate_size;  // FFN inner dim
    int   n_heads;            // Nh (query heads)
    int   n_kv_heads;         // Nkv (key/value heads, for GQA)
    int   head_dim;           // D = H / Nh
    int   n_layers;
    float rope_theta;
    float rms_norm_eps;
    bool  is_causal;  // true for text encoder, false for lyric/timbre
};

// Per-layer weights
struct Qwen3Layer {
    struct ggml_tensor * input_layernorm;      // [H]
    struct ggml_tensor * post_attn_layernorm;  // [H]

    // Attention (fused or separate, same pattern as DiT)
    struct ggml_tensor * qkv;     // [H, (Nh+2*Nkv)*D] full fused (or NULL)
    struct ggml_tensor * qk;      // [H, (Nh+Nkv)*D] Q+K fused (or NULL)
    struct ggml_tensor * q_proj;  // [H, Nh*D]  (NULL when fused)
    struct ggml_tensor * k_proj;  // [H, Nkv*D] (NULL when fused)
    struct ggml_tensor * v_proj;  // [H, Nkv*D] (NULL when QKV fused)
    struct ggml_tensor * o_proj;  // [Nh*D, H]
    struct ggml_tensor * q_norm;  // [D]
    struct ggml_tensor * k_norm;  // [D]

    // MLP (fused or separate)
    struct ggml_tensor * gate_up;    // [H, 2*FFN] fused (or NULL)
    struct ggml_tensor * gate_proj;  // [H, FFN] (NULL when fused)
    struct ggml_tensor * up_proj;    // [H, FFN] (NULL when fused)
    struct ggml_tensor * down_proj;  // [FFN, H]
};

// Helpers (pure graph ops, no side effects)
static struct ggml_tensor * qwen3_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) {
        return t;
    }
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

static struct ggml_tensor * qwen3_linear(struct ggml_context * ctx, struct ggml_tensor * w, struct ggml_tensor * x) {
    return ggml_mul_mat(ctx, w, x);
}

// F32 manual attention (fallback when flash_attn_ext is disabled).
// Works for 3D [D, S, X] and 4D [D, S, X, N] inputs.
// Returns same layout as flash_attn_ext: dims 1 and 2 swapped vs input.
static struct ggml_tensor * qwen3_attn_f32(struct ggml_context * ctx,
                                           struct ggml_tensor *  q,
                                           struct ggml_tensor *  k,
                                           struct ggml_tensor *  v,
                                           struct ggml_tensor *  mask,
                                           float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

static struct ggml_tensor * qwen3_rms_norm(struct ggml_context * ctx,
                                           struct ggml_tensor *  x,
                                           struct ggml_tensor *  w,
                                           float                 eps) {
    struct ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, qwen3_f32(ctx, w));
}

// Graph builders
// These build sub-graphs and return output tensors.
// They operate on ggml layout: [H, S] for hidden states.

// MLP: SwiGLU (fused gate+up or separate)
static struct ggml_tensor * qwen3_build_mlp(struct ggml_context * ctx,
                                            Qwen3Layer *          ly,
                                            struct ggml_tensor *  x,  // [H, S]
                                            int                   S) {
    (void) S;
    struct ggml_tensor * ff;
    if (ly->gate_up) {
        struct ggml_tensor * gu = qwen3_linear(ctx, ly->gate_up, x);
        ff                      = ggml_swiglu(ctx, gu);
    } else {
        struct ggml_tensor * gate = qwen3_linear(ctx, ly->gate_proj, x);
        struct ggml_tensor * up   = qwen3_linear(ctx, ly->up_proj, x);
        ff                        = ggml_swiglu_split(ctx, gate, up);
    }
    return qwen3_linear(ctx, ly->down_proj, ff);
}

// Loading
static void qwen3_load_layer(WeightCtx *         wctx,
                             const GGUFModel &   gf,
                             Qwen3Layer *        ly,
                             const std::string & prefix,
                             int                 layer_idx = -1) {
    ly->input_layernorm     = gf_load_tensor_f32(wctx, gf, prefix + ".input_layernorm.weight");
    ly->post_attn_layernorm = gf_load_tensor_f32(wctx, gf, prefix + ".post_attention_layernorm.weight");

    // Attention: try Q+K+V fused, then Q+K partial, then separate
    ly->qkv = gf_load_qkv_fused(wctx, gf, prefix + ".self_attn.q_proj.weight", prefix + ".self_attn.k_proj.weight",
                                prefix + ".self_attn.v_proj.weight");
    if (!ly->qkv) {
        ly->qk = gf_load_pair_fused(wctx, gf, prefix + ".self_attn.q_proj.weight", prefix + ".self_attn.k_proj.weight");
        if (ly->qk) {
            ly->v_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.v_proj.weight");
            if (layer_idx == 0) {
                fprintf(stderr, "[Qwen3] Attn: Q+K fused, V separate\n");
            }
        } else {
            ly->q_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.q_proj.weight");
            ly->k_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.k_proj.weight");
            ly->v_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.v_proj.weight");
            if (layer_idx == 0) {
                fprintf(stderr, "[Qwen3] Attn: all separate\n");
            }
        }
    } else {
        if (layer_idx == 0) {
            fprintf(stderr, "[Qwen3] Attn: Q+K+V fused\n");
        }
    }

    ly->o_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.o_proj.weight");
    ly->q_norm = gf_load_tensor_f32(wctx, gf, prefix + ".self_attn.q_norm.weight");
    ly->k_norm = gf_load_tensor_f32(wctx, gf, prefix + ".self_attn.k_norm.weight");

    // MLP: try gate+up fused, then separate
    ly->gate_up = gf_load_pair_fused(wctx, gf, prefix + ".mlp.gate_proj.weight", prefix + ".mlp.up_proj.weight");
    if (ly->gate_up) {
        if (layer_idx == 0) {
            fprintf(stderr, "[Qwen3] MLP: gate+up fused\n");
        }
    } else {
        ly->gate_proj = gf_load_tensor(wctx, gf, prefix + ".mlp.gate_proj.weight");
        ly->up_proj   = gf_load_tensor(wctx, gf, prefix + ".mlp.up_proj.weight");
        if (layer_idx == 0) {
            fprintf(stderr, "[Qwen3] MLP: gate+up separate\n");
        }
    }
    ly->down_proj = gf_load_tensor(wctx, gf, prefix + ".mlp.down_proj.weight");
}
