// vae-enc.h: DAC-VAE encoder (audio -> latent) via ggml
//
// Mirror of the vae.h decoder. Reuses VAEResUnit, the load helpers and the
// graph ops. Architecture per side: conv_in(1->64, k7) -> 4 blocks of
// (3 res units + snake + strided conv, strides 2/4/8/8) -> snake_out ->
// conv_out(1024->1024, k3) -> mean_proj(1024->64, k1). Deterministic encode
// returns the posterior mean. Stereo = two mono passes, downsample 512x
// (mirror of the decoder upsample). Weights come from the encoder.* tensors
// of the vocoder GGUF.

#pragma once

#include "vae.h"

// Encoder block: 3 res units (in_ch) -> snake (in_ch) -> strided conv (in_ch -> out_ch)
// The decoder block is the mirror: snake -> conv transpose -> 3 res units.
struct VAEEncBlock {
    VAEResUnit          ru[3];
    struct ggml_tensor *sa, *si;  // snake alpha, inv [1, in_ch]
    struct ggml_tensor *dw, *db;  // strided conv [K, in_ch, out_ch], bias [out_ch]
    int                 in_ch, out_ch, stride, kernel, padding;
};

struct VAEEncoder {
    struct ggml_tensor *c1w, *c1b;  // conv_in [7, 1, 64], bias [64]
    VAEEncBlock         blk[4];
    struct ggml_tensor *sa, *si;    // snake_out alpha, inv [1, 1024]
    struct ggml_tensor *c2w, *c2b;  // conv_out [3, 1024, 1024], bias [1024]
    struct ggml_tensor *mpw, *mpb;  // mean_proj [1, 1024, 64], bias [64]

    ggml_backend_t        backend     = nullptr;
    ggml_backend_t        cpu_backend = nullptr;
    ggml_backend_sched_t  sched       = nullptr;
    ggml_backend_buffer_t buf         = nullptr;
    struct ggml_context * weight_ctx  = nullptr;

    // Graph cache: rebuilt only when T_audio changes
    struct ggml_context * graph_ctx    = nullptr;
    uint8_t *             graph_buf    = nullptr;
    struct ggml_cgraph *  graph        = nullptr;
    struct ggml_tensor *  graph_input  = nullptr;
    struct ggml_tensor *  graph_output = nullptr;
    int                   graph_T      = 0;

    bool load(const char * gguf_path);

    // audio: interleaved stereo [T_audio, 2], T_audio a multiple of 512.
    // latent out: channel-major [128, T_latent], left side channels 0..63.
    bool encode(const float * audio, int T_audio, std::vector<float> & latent, int & T_latent);

    void free();
};

inline bool VAEEncoder::load(const char * gguf_path) {
    GGUFModel gf = {};
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[VAE-Enc] FATAL: cannot load %s\n", gguf_path);
        return false;
    }
    if (!ggml_get_tensor(gf.meta, "encoder.conv_in.weight")) {
        fprintf(stderr, "[VAE-Enc] FATAL: %s carries no encoder.* tensors (regenerate with ./convert.py)\n", gguf_path);
        gf_close(&gf);
        return false;
    }

    static const int strides[]   = { 2, 4, 8, 8 };
    static const int in_ch[]     = { 64, 128, 256, 512 };
    static const int out_ch[]    = { 128, 256, 512, 1024 };
    static const int dilations[] = { 1, 3, 9 };

    // Phase 1: tensor metadata (no_alloc context)
    size_t                  ctx_size = ggml_tensor_overhead() * 128;
    struct ggml_init_params p        = { ctx_size, NULL, true };
    weight_ctx                       = ggml_init(p);
    struct ggml_context * ctx        = weight_ctx;

    c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 1, 64);
    c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);

    for (int i = 0; i < 4; i++) {
        VAEEncBlock & b = blk[i];
        b.in_ch         = in_ch[i];
        b.out_ch        = out_ch[i];
        b.stride        = strides[i];
        b.kernel        = strides[i] * 2;
        b.padding       = (strides[i] + 1) / 2;
        int C           = in_ch[i];
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
        b.sa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
        b.si = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
        b.dw = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, b.kernel, in_ch[i], out_ch[i]);
        b.db = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch[i]);
    }

    sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1024);
    si  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1024);
    c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, 1024, 1024);
    c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    mpw = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, 1024, 64);
    mpb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);

    // Phase 2: backend buffer
    BackendPair bp = backend_init("VAE-Enc");
    backend        = bp.backend;
    cpu_backend    = bp.cpu_backend;
    sched          = backend_sched_new(bp, 8192);
    buf            = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "[VAE-Enc] FATAL: failed to allocate weight buffer\n");
        return false;
    }
    fprintf(stderr, "[VAE-Enc] Backend: %s, Weight buffer: %.1f MB\n", ggml_backend_name(backend),
            (float) ggml_backend_buffer_get_size(buf) / (1024 * 1024));

    // Phase 3: load weights
    vae_load_conv(c1w, gf, "encoder.conv_in.weight");
    vae_load_f32(c1b, gf, "encoder.conv_in.bias");

    for (int i = 0; i < 4; i++) {
        VAEEncBlock & b       = blk[i];
        std::string   blk_pfx = "encoder.blocks." + std::to_string(i);
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
        vae_load_snake(b.sa, b.si, gf, blk_pfx + ".snake1.alpha");
        vae_load_conv(b.dw, gf, blk_pfx + ".conv1.weight");
        vae_load_f32(b.db, gf, blk_pfx + ".conv1.bias");
    }

    vae_load_snake(sa, si, gf, "encoder.snake_out.alpha");
    vae_load_conv(c2w, gf, "encoder.conv_out.weight");
    vae_load_f32(c2b, gf, "encoder.conv_out.bias");
    vae_load_conv(mpw, gf, "encoder.mean_proj.weight");
    vae_load_f32(mpb, gf, "encoder.mean_proj.bias");

    fprintf(stderr, "[VAE-Enc] Loaded: 4 blocks, downsample 512x, F32 activations\n");
    gf_close(&gf);
    return true;
}

// Build encoder graph for one side: audio [T, 1] -> mean latent [T_latent, 64]
static struct ggml_tensor * vae_enc_build_graph(struct ggml_context * ctx, VAEEncoder * m, struct ggml_tensor * audio) {
    struct ggml_tensor * x = vae_conv1d(ctx, m->c1w, m->c1b, audio, 1, 3, 1);
    for (int i = 0; i < 4; i++) {
        VAEEncBlock & b = m->blk[i];
        for (int r = 0; r < 3; r++) {
            x = vae_res_unit(ctx, &b.ru[r], x);
        }
        x = vae_snake(ctx, x, b.sa, b.si);
        x = vae_conv1d(ctx, b.dw, b.db, x, b.stride, b.padding, 1);
    }
    x = vae_snake(ctx, x, m->sa, m->si);
    x = vae_conv1d(ctx, m->c2w, m->c2b, x, 1, 1, 1);
    return vae_conv1d(ctx, m->mpw, m->mpb, x, 1, 0, 1);
}

// Build or reuse the graph for T_audio, set one side as input, run.
// Returns T_latent, or -1 on failure. Output stays in graph_output.
static int vae_enc_compute(VAEEncoder * m, const float * side, int T_audio) {
    if (m->graph_T != T_audio) {
        if (m->graph_ctx) {
            ggml_backend_sched_reset(m->sched);
            ggml_free(m->graph_ctx);
            std::free(m->graph_buf);
        }
        size_t ctx_size             = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false);
        m->graph_buf                = (uint8_t *) malloc(ctx_size);
        struct ggml_init_params p   = { ctx_size, m->graph_buf, true };
        struct ggml_context *   ctx = ggml_init(p);

        m->graph_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_audio, 1);
        ggml_set_name(m->graph_input, "enc_input");
        ggml_set_input(m->graph_input);

        m->graph_output = vae_enc_build_graph(ctx, m, m->graph_input);
        ggml_set_name(m->graph_output, "enc_output");
        ggml_set_output(m->graph_output);

        m->graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(m->graph, m->graph_output);

        if (!ggml_backend_sched_alloc_graph(m->sched, m->graph)) {
            fprintf(stderr, "[VAE-Enc] FATAL: graph alloc failed for T=%d\n", T_audio);
            ggml_free(ctx);
            std::free(m->graph_buf);
            m->graph_ctx = nullptr;
            m->graph_buf = nullptr;
            m->graph_T   = 0;
            return -1;
        }
        m->graph_ctx = ctx;
        m->graph_T   = T_audio;
    }

    ggml_backend_tensor_set(m->graph_input, side, 0, (size_t) T_audio * sizeof(float));
    ggml_backend_sched_graph_compute(m->sched, m->graph);
    return (int) m->graph_output->ne[0];
}

inline bool VAEEncoder::encode(const float * audio, int T_audio, std::vector<float> & latent, int & T_latent) {
    if (T_audio <= 0 || T_audio % 512 != 0) {
        fprintf(stderr, "[VAE-Enc] T_audio %d is not a multiple of 512\n", T_audio);
        return false;
    }
    T_latent = T_audio / 512;
    latent.resize((size_t) 128 * T_latent);
    std::vector<float> side(T_audio);

    for (int ch = 0; ch < 2; ch++) {
        for (int t = 0; t < T_audio; t++) {
            side[t] = audio[(size_t) t * 2 + ch];
        }
        int T_out = vae_enc_compute(this, side.data(), T_audio);
        if (T_out != T_latent) {
            fprintf(stderr, "[VAE-Enc] encode failed on side %d (T_out=%d)\n", ch, T_out);
            return false;
        }
        // Graph output [T_latent, 64] is channel-contiguous: exactly the
        // 64 rows of this side in the channel-major latent
        ggml_backend_tensor_get(graph_output, latent.data() + (size_t) ch * 64 * T_latent, 0,
                                (size_t) 64 * T_latent * sizeof(float));
    }
    return true;
}

inline void VAEEncoder::free() {
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
