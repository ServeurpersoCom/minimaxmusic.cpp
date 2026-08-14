// depth-decoder.h: RVQ depth decoder (0.6B), intra frame transformer (GGML)
//
// For each 25 Hz frame it runs a short causal sequence over codebook
// positions and autoregressively predicts the 7 acoustic codebooks. The
// caller (pipeline) assembles the raw step sequence: the global LM last
// hidden state, then the LM embedding of the semantic code, then the audio
// embedding of each sampled acoustic code. Every step goes through
// `projection`, gets the learned positional embedding, runs 4 causal
// attention layers (16 heads x 256, no RoPE, RMSNorm eps 1e-6, SwiGLU
// 6144), and the final RMSNorm. The last hidden state feeds
// audio_heads[S - 2]: a sequence of length S predicts codebook S - 1.
//
// The normalized per-step hidden states are what the condition encoder
// fuses (one global LM state + 7 depth states per frame).
//
// Two execution paths share the same model body:
// - forward(): one cached graph per sequence length (S in 2..8), single
//   stream, host-side sampling between steps. This is the parity path.
// - forward_frame(): one cached graph for the whole frame under CFG. The
//   7 autoregressive steps run batch-2 (cond, uncond) with the CFG
//   sampling chain in graph (argsort top-k of the conditional logits,
//   guided softmax, cumsum CDF crossing against a host-drawn uniform),
//   and each sampled code feeds the next step through the device audio
//   embedding table.
//
// The audio embedding table is kept both as a device tensor for the fused
// in-graph code embedding steps and as a host F32 copy for the single
// path and the global LM frame feedback sum.
//
// Tensors (rvq_depth_decoder/): projection.weight [4096, 4096],
// pos_embedding.weight [16, 4096], audio_embeddings.weight [7168, 4096],
// layers.N.{input_layernorm.weight, attn.to_{q,k,v,out}.weight,
// post_attention_layernorm.weight, gate_proj, up_proj, down_proj},
// norm.weight, audio_heads.{0..6}.weight [1024, 4096].
#pragma once

#include "backend.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct DepthLayer {
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * wq, *wk, *wv, *wo;
    struct ggml_tensor * ffn_norm;
    struct ggml_tensor * w_gate, *w_up, *w_down;
};

struct DepthDecoder {
    static const int DIM       = 4096;
    static const int N_LAYERS  = 4;
    static const int N_HEADS   = 16;
    static const int HEAD_DIM  = 256;
    static const int FF_INNER  = 6144;
    static const int VOCAB     = 1024;
    static const int CODEBOOKS = 8;
    static const int MAX_POS   = 16;
    static const int MAX_SEQ   = CODEBOOKS;  // last predicting sequence: 8 steps

    struct ggml_tensor * projection;
    struct ggml_tensor * pos_embedding;
    DepthLayer           layers[N_LAYERS];
    struct ggml_tensor * out_norm;
    struct ggml_tensor * audio_heads[CODEBOOKS - 1];
    struct ggml_tensor * audio_embeddings;  // device [7168, 4096] for the fused code embedding steps

    std::vector<float> audio_emb;           // host F32 [7168, 4096] for the single path and LM feedback

    ggml_backend_t backend     = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    BackendPair    bp          = {};
    WeightCtx      wctx        = {};

    // One cached graph per sequence length S in 2..MAX_SEQ.
    // Each slot owns its scheduler: a shared one would drop the previous
    // slot's compute buffer allocation on every ggml_backend_sched_alloc_graph.
    struct GraphSlot {
        struct ggml_context * ctx     = nullptr;
        uint8_t *             buf     = nullptr;
        ggml_backend_sched_t  sched   = nullptr;
        struct ggml_cgraph *  graph   = nullptr;
        struct ggml_tensor *  input   = nullptr;
        struct ggml_tensor *  hiddens = nullptr;
        struct ggml_tensor *  logits  = nullptr;
    };

    // One cached fused frame graph, keyed on the baked sampling params.
    struct FusedSlot {
        struct ggml_context * ctx                  = nullptr;
        uint8_t *             buf                  = nullptr;
        ggml_backend_sched_t  sched                = nullptr;
        struct ggml_cgraph *  graph                = nullptr;
        struct ggml_tensor *  seq_init             = nullptr;  // [4096, 2, 2]
        struct ggml_tensor *  rand                 = nullptr;  // [7]
        struct ggml_tensor *  cond_hiddens         = nullptr;  // [4096, 7]
        struct ggml_tensor *  codes[CODEBOOKS - 1] = {};
        float                 cfg                  = 0.0f;
        int                   top_k                = 0;
    };

    GraphSlot slots[MAX_SEQ + 1];  // single sequence graphs
    FusedSlot fused;               // whole frame CFG graph

    bool load(const char * gguf_path);

    // seq: [S, 4096] time-major raw step embeddings (projection applied in
    // graph). Writes the normalized hidden states [S, 4096] time-major and
    // the logits [1024] of audio_heads[S - 2] on the last step.
    bool forward(const float * seq, int S, float * hiddens, float * logits);

    // Fused CFG frame: seq2 is the initial two-step sequences of both
    // streams concatenated [2, 2, 4096] (cond first: global LM hidden then
    // the semantic code embedding). rand7 holds one uniform(0, 1) draw per
    // acoustic codebook. Samples the 7 codes in graph and writes them to
    // codes7, plus the 7 conditional last-step hidden states [7, 4096]
    // time-major to cond_hiddens.
    bool forward_frame(const float * seq2,
                       const float * rand7,
                       float         cfg,
                       int           top_k,
                       int *         codes7,
                       float *       cond_hiddens);

    // Row of the audio embedding table for acoustic codebook cb (1..7)
    // entry code (0..1023).
    const float * audio_embedding_row(int cb, int code) const {
        return audio_emb.data() + ((size_t) (cb - 1) * VOCAB + code) * DIM;
    }

    void free();
};

inline bool DepthDecoder::load(const char * gguf_path) {
    GGUFModel gf = {};
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[Depth] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    bp          = backend_init("Depth");
    backend     = bp.backend;
    cpu_backend = bp.cpu_backend;

    wctx_init(&wctx, 64);

    projection    = gf_load_tensor(&wctx, gf, "projection.weight");
    pos_embedding = gf_load_tensor_f32(&wctx, gf, "pos_embedding.weight");
    out_norm      = gf_load_tensor_f32(&wctx, gf, "norm.weight");
    for (int i = 0; i < N_LAYERS; i++) {
        DepthLayer & l  = layers[i];
        std::string  pf = "layers." + std::to_string(i) + ".";
        l.attn_norm     = gf_load_tensor_f32(&wctx, gf, pf + "input_layernorm.weight");
        l.wq            = gf_load_tensor(&wctx, gf, pf + "attn.to_q.weight");
        l.wk            = gf_load_tensor(&wctx, gf, pf + "attn.to_k.weight");
        l.wv            = gf_load_tensor(&wctx, gf, pf + "attn.to_v.weight");
        l.wo            = gf_load_tensor(&wctx, gf, pf + "attn.to_out.weight");
        l.ffn_norm      = gf_load_tensor_f32(&wctx, gf, pf + "post_attention_layernorm.weight");
        l.w_gate        = gf_load_tensor(&wctx, gf, pf + "gate_proj.weight");
        l.w_up          = gf_load_tensor(&wctx, gf, pf + "up_proj.weight");
        l.w_down        = gf_load_tensor(&wctx, gf, pf + "down_proj.weight");
    }
    for (int i = 0; i < CODEBOOKS - 1; i++) {
        audio_heads[i] = gf_load_tensor(&wctx, gf, "audio_heads." + std::to_string(i) + ".weight");
    }
    audio_embeddings = gf_load_tensor(&wctx, gf, "audio_embeddings.weight");

    if (!wctx_alloc(&wctx, backend)) {
        return false;
    }

    // Host F32 copy of the audio embedding table
    if (!gf_host_f32(gf, "audio_embeddings.weight", audio_emb)) {
        fprintf(stderr, "[Depth] FATAL: missing audio_embeddings.weight\n");
        return false;
    }

    fprintf(stderr, "[Depth] Loaded: %d layers, dim %d, %d heads\n", N_LAYERS, DIM, CODEBOOKS - 1);
    gf_close(&gf);
    return true;
}

static struct ggml_tensor * depth_rms_norm(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * w) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w);
}

static struct ggml_tensor * depth_layer(struct ggml_context * ctx,
                                        DepthLayer *          l,
                                        struct ggml_tensor *  x,
                                        int                   S,
                                        int                   B) {
    const int D  = DepthDecoder::HEAD_DIM;
    const int Nh = DepthDecoder::N_HEADS;

    // Causal attention, no RoPE. Batch rides dim 3, broadcast everywhere.
    struct ggml_tensor * h = depth_rms_norm(ctx, x, l->attn_norm);
    struct ggml_tensor * q = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, l->wq, h), D, Nh, S, B);
    struct ggml_tensor * k = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, l->wk, h), D, Nh, S, B);
    struct ggml_tensor * v = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, l->wv, h), D, Nh, S, B);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_scale(ctx, scores, 1.0f / sqrtf((float) D));
    scores                      = ggml_diag_mask_inf(ctx, scores, 0);
    scores                      = ggml_soft_max(ctx, scores);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * attn   = ggml_mul_mat(ctx, vt, scores);
    attn                        = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn                        = ggml_reshape_3d(ctx, attn, Nh * D, S, B);

    x = ggml_add(ctx, x, ggml_mul_mat(ctx, l->wo, attn));

    // SwiGLU
    h                         = depth_rms_norm(ctx, x, l->ffn_norm);
    struct ggml_tensor * gate = ggml_mul_mat(ctx, l->w_gate, h);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, l->w_up, h);
    struct ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
    return ggml_add(ctx, x, ggml_mul_mat(ctx, l->w_down, act));
}

// Full model body over an in-graph input [DIM, S, B]: projection,
// positional add over the first S table columns, layers, final norm, and
// the head of the last step. The batch rides dim 2; the last-step view
// keeps the batch stride so logits come out [VOCAB, B].
struct DepthStep {
    struct ggml_tensor * hiddens;
    struct ggml_tensor * logits;
};

static DepthStep depth_model(struct ggml_context * ctx, DepthDecoder * m, struct ggml_tensor * input, int S, int B) {
    const int DIM = DepthDecoder::DIM;

    struct ggml_tensor * x   = ggml_mul_mat(ctx, m->projection, input);
    struct ggml_tensor * pos = ggml_view_2d(ctx, m->pos_embedding, DIM, S, m->pos_embedding->nb[1], 0);
    x                        = ggml_add(ctx, x, pos);

    for (int i = 0; i < DepthDecoder::N_LAYERS; i++) {
        x = depth_layer(ctx, &m->layers[i], x, S, B);
    }
    struct ggml_tensor * hiddens = depth_rms_norm(ctx, x, m->out_norm);

    struct ggml_tensor * last =
        ggml_view_3d(ctx, hiddens, DIM, 1, B, hiddens->nb[1], hiddens->nb[2], (size_t) (S - 1) * hiddens->nb[1]);
    struct ggml_tensor * last2d = ggml_reshape_2d(ctx, ggml_cont(ctx, last), DIM, B);
    struct ggml_tensor * logits = ggml_mul_mat(ctx, m->audio_heads[S - 2], last2d);
    return { hiddens, logits };
}

// Build one cached single-stream graph for sequence length S.
inline bool depth_build_slot(DepthDecoder * m, DepthDecoder::GraphSlot & slot, int S) {
    const int DIM      = DepthDecoder::DIM;
    size_t    ctx_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false);
    slot.buf           = (uint8_t *) malloc(ctx_size);
    if (!slot.buf) {
        fprintf(stderr, "[Depth] FATAL: OOM allocating graph context for S=%d\n", S);
        return false;
    }
    struct ggml_init_params p   = { ctx_size, slot.buf, true };
    struct ggml_context *   ctx = ggml_init(p);

    slot.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DIM, S, 1);
    ggml_set_input(slot.input);

    DepthStep out = depth_model(ctx, m, slot.input, S, 1);
    slot.hiddens  = out.hiddens;
    slot.logits   = out.logits;
    ggml_set_output(slot.hiddens);
    ggml_set_output(slot.logits);

    slot.graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(slot.graph, slot.hiddens);
    ggml_build_forward_expand(slot.graph, slot.logits);

    slot.sched = backend_sched_new(m->bp, 512);
    if (!ggml_backend_sched_alloc_graph(slot.sched, slot.graph)) {
        fprintf(stderr, "[Depth] FATAL: graph alloc failed for S=%d\n", S);
        ggml_backend_sched_free(slot.sched);
        ggml_free(ctx);
        std::free(slot.buf);
        slot = {};
        return false;
    }
    slot.ctx = ctx;
    return true;
}

// Build the fused CFG frame graph with the sampling params baked in.
// Per step: batch-2 model forward, argsort top-k of the conditional
// logits (contractual descending order on every backend), guided logits
// on those k candidates, softmax, cumsum CDF crossed against the step's
// uniform draw (clamped: the draw can land past the CDF tail), gather of
// the vocab id, then the code's device embedding row appended to both
// streams for the next step.
inline bool depth_build_fused(DepthDecoder * m, DepthDecoder::FusedSlot & slot, float cfg, int top_k) {
    const int DIM = DepthDecoder::DIM;
    const int V   = DepthDecoder::VOCAB;
    const int NC  = DepthDecoder::CODEBOOKS - 1;
    const int k   = top_k < 1 ? 1 : (top_k > V ? V : top_k);

    size_t ctx_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false);
    slot.buf        = (uint8_t *) malloc(ctx_size);
    if (!slot.buf) {
        fprintf(stderr, "[Depth] FATAL: OOM allocating fused graph context\n");
        return false;
    }
    struct ggml_init_params p   = { ctx_size, slot.buf, true };
    struct ggml_context *   ctx = ggml_init(p);

    slot.seq_init = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DIM, 2, 2);
    ggml_set_input(slot.seq_init);
    slot.rand = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NC);
    ggml_set_input(slot.rand);

    // Scalar gather over a row vector: values reshaped to one column per row
    auto gather = [ctx](struct ggml_tensor * vals, struct ggml_tensor * idx) {
        struct ggml_tensor * cols = ggml_reshape_2d(ctx, vals, 1, vals->ne[0]);
        return ggml_reshape_1d(ctx, ggml_get_rows(ctx, cols, idx), idx->ne[0]);
    };

    struct ggml_tensor * seq = slot.seq_init;
    struct ggml_tensor * hid_parts[DepthDecoder::CODEBOOKS - 1];

    for (int cb = 1; cb <= NC; cb++) {
        int       S   = cb + 1;
        DepthStep out = depth_model(ctx, m, seq, S, 2);

        hid_parts[cb - 1] =
            ggml_reshape_2d(ctx, ggml_view_1d(ctx, out.hiddens, DIM, (size_t) (S - 1) * out.hiddens->nb[1]), DIM, 1);

        struct ggml_tensor * cond   = ggml_view_1d(ctx, out.logits, V, 0);
        struct ggml_tensor * uncond = ggml_view_1d(ctx, out.logits, V, out.logits->nb[1]);

        struct ggml_tensor * idxk     = ggml_cont(ctx, ggml_argsort_top_k(ctx, cond, k));
        struct ggml_tensor * cond_k   = gather(cond, idxk);
        struct ggml_tensor * uncond_k = gather(uncond, idxk);
        struct ggml_tensor * guided   = ggml_add(ctx, uncond_k, ggml_scale(ctx, ggml_sub(ctx, cond_k, uncond_k), cfg));

        struct ggml_tensor * cdf = ggml_cumsum(ctx, ggml_soft_max(ctx, guided));
        struct ggml_tensor * u   = ggml_view_1d(ctx, slot.rand, 1, (size_t) (cb - 1) * sizeof(float));
        struct ggml_tensor * idxf =
            ggml_scale_bias(ctx, ggml_sum(ctx, ggml_step(ctx, ggml_sub(ctx, cdf, u))), -1.0f, (float) k);
        struct ggml_tensor * pick = ggml_cast(ctx, ggml_clamp(ctx, idxf, 0.0f, (float) (k - 1)), GGML_TYPE_I32);

        struct ggml_tensor * code = ggml_get_rows(ctx, ggml_reshape_2d(ctx, idxk, 1, k), pick);
        slot.codes[cb - 1]        = code;
        ggml_set_output(code);

        if (cb < NC) {
            struct ggml_tensor * table = ggml_view_2d(ctx, m->audio_embeddings, DIM, V, m->audio_embeddings->nb[1],
                                                      (size_t) (cb - 1) * V * m->audio_embeddings->nb[1]);
            struct ggml_tensor * row   = ggml_get_rows(ctx, table, ggml_reshape_1d(ctx, code, 1));
            struct ggml_tensor * row3  = ggml_reshape_3d(ctx, row, DIM, 1, 1);
            seq                        = ggml_concat(ctx, seq, ggml_concat(ctx, row3, row3, 2), 1);
        }
    }

    slot.cond_hiddens = hid_parts[0];
    for (int i = 1; i < NC; i++) {
        slot.cond_hiddens = ggml_concat(ctx, slot.cond_hiddens, hid_parts[i], 1);
    }
    ggml_set_output(slot.cond_hiddens);

    slot.graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(slot.graph, slot.cond_hiddens);
    for (int i = 0; i < NC; i++) {
        ggml_build_forward_expand(slot.graph, slot.codes[i]);
    }

    slot.sched = backend_sched_new(m->bp, 4096);
    if (!ggml_backend_sched_alloc_graph(slot.sched, slot.graph)) {
        fprintf(stderr, "[Depth] FATAL: fused graph alloc failed\n");
        ggml_backend_sched_free(slot.sched);
        ggml_free(ctx);
        std::free(slot.buf);
        slot = {};
        return false;
    }
    slot.ctx   = ctx;
    slot.cfg   = cfg;
    slot.top_k = top_k;
    return true;
}

inline bool DepthDecoder::forward(const float * seq, int S, float * hiddens, float * logits) {
    if (S < 2 || S > MAX_SEQ) {
        fprintf(stderr, "[Depth] invalid sequence length %d\n", S);
        return false;
    }

    GraphSlot & slot = slots[S];
    if (!slot.ctx && !depth_build_slot(this, slot, S)) {
        return false;
    }

    ggml_backend_tensor_set(slot.input, seq, 0, (size_t) DIM * S * sizeof(float));
    ggml_backend_sched_graph_compute(slot.sched, slot.graph);

    ggml_backend_tensor_get(slot.hiddens, hiddens, 0, (size_t) DIM * S * sizeof(float));
    ggml_backend_tensor_get(slot.logits, logits, 0, VOCAB * sizeof(float));
    return true;
}

inline bool DepthDecoder::forward_frame(const float * seq2,
                                        const float * rand7,
                                        float         cfg,
                                        int           top_k,
                                        int *         codes7,
                                        float *       cond_hiddens) {
    // The graph bakes the sampling params: rebuild when they change
    if (fused.ctx && (fused.cfg != cfg || fused.top_k != top_k)) {
        ggml_backend_sched_free(fused.sched);
        ggml_free(fused.ctx);
        std::free(fused.buf);
        fused = {};
    }
    if (!fused.ctx && !depth_build_fused(this, fused, cfg, top_k)) {
        return false;
    }

    ggml_backend_tensor_set(fused.seq_init, seq2, 0, (size_t) 2 * 2 * DIM * sizeof(float));
    ggml_backend_tensor_set(fused.rand, rand7, 0, (CODEBOOKS - 1) * sizeof(float));
    ggml_backend_sched_graph_compute(fused.sched, fused.graph);

    for (int i = 0; i < CODEBOOKS - 1; i++) {
        int32_t code = 0;
        ggml_backend_tensor_get(fused.codes[i], &code, 0, sizeof(int32_t));
        codes7[i] = code;
    }
    ggml_backend_tensor_get(fused.cond_hiddens, cond_hiddens, 0, (size_t) (CODEBOOKS - 1) * DIM * sizeof(float));
    return true;
}

inline void DepthDecoder::free() {
    for (int s = 0; s <= MAX_SEQ; s++) {
        if (slots[s].ctx) {
            ggml_backend_sched_free(slots[s].sched);
            ggml_free(slots[s].ctx);
            std::free(slots[s].buf);
            slots[s] = {};
        }
    }
    if (fused.ctx) {
        ggml_backend_sched_free(fused.sched);
        ggml_free(fused.ctx);
        std::free(fused.buf);
        fused = {};
    }
    wctx_free(&wctx);
    // backends are refcounted and shared across all modules
    backend_release(backend, cpu_backend);
    backend     = nullptr;
    cpu_backend = nullptr;
}
