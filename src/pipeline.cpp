// pipeline.cpp: full MiniMax Music 3 pipeline, text to stereo float
//
// Stages, mirroring the diffusers modular pipeline:
//   prompt assembly -> global LM batch 2 [cond, uncond] with logit CFG
//   -> RVQ depth decoder per frame (7 acoustic codebooks, shared CFG)
//   -> condition encoder per 200 frame window
//   -> flow matching DiT (Euler steps, velocity CFG, overlap blending)
//   -> flow VAE decoder -> crop and stitch -> 44.1 kHz stereo.

#include "pipeline.h"

#include "philox.h"
#include "prompt.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

static const int FRAME_RATE     = 25;
static const int MAX_FRAMES     = 9000;
static const int CHUNK_FRAMES   = 200;
static const int CHUNK_HOP      = 100;
static const int OVERLAP_LATENT = 172;
static const int CROP_LEFT      = 86;
static const int CROP_RIGHT     = 344 - 86;
static const int HOP            = 512;
static const int SAMPLE_RATE    = 44100;

// CFG on logits: guided = uncond + (cond - uncond) * scale, restricted to
// the conditional branch's top k among allowed ids, then top-k sampled.
static int mm3_cfg_sample(const float *            cond,
                          const float *            uncond,
                          const std::vector<int> & allowed,
                          float                    cfg,
                          int                      top_k,
                          std::mt19937_64 &        rng) {
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(allowed.size());
    for (int id : allowed) {
        ranked.push_back({ cond[id], id });
    }
    int k = (int) std::min((size_t) top_k, ranked.size());
    std::partial_sort(
        ranked.begin(), ranked.begin() + k, ranked.end(),
        [](const std::pair<float, int> & a, const std::pair<float, int> & b) { return a.first > b.first; });
    ranked.resize(k);

    std::vector<float> guided(k);
    float              mx = -INFINITY;
    for (int i = 0; i < k; i++) {
        int id    = ranked[i].second;
        guided[i] = uncond[id] + (cond[id] - uncond[id]) * cfg;
        mx        = std::max(mx, guided[i]);
    }
    float sum = 0;
    for (int i = 0; i < k; i++) {
        guided[i] = expf(guided[i] - mx);
        sum += guided[i];
    }
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float                                 r   = uni(rng) * sum;
    float                                 acc = 0;
    for (int i = 0; i < k; i++) {
        acc += guided[i];
        if (r <= acc) {
            return ranked[i].second;
        }
    }
    return ranked[k - 1].second;
}

// Reads one embedding table row [H] from the LM GPU tensor as F32,
// dequantizing through the ggml type traits (BF16, Q8_0, K-quants).
static void mm3_lm_embed_row(Qwen3LM * lm, int token_id, float * out) {
    int                  H         = lm->cfg.hidden_size;
    size_t               row_bytes = ggml_row_size(lm->embed_tokens->type, H);
    std::vector<uint8_t> raw(row_bytes);
    ggml_backend_tensor_get(lm->embed_tokens, raw.data(), (size_t) token_id * row_bytes, row_bytes);
    if (lm->embed_tokens->type == GGML_TYPE_F32) {
        memcpy(out, raw.data(), (size_t) H * sizeof(float));
        return;
    }
    ggml_get_type_traits(lm->embed_tokens->type)->to_float(raw.data(), out, H);
}

static bool is_cancelled(std::atomic<bool> * cancel) {
    return cancel && cancel->load();
}

// Reload one component when its resolved path changed.
// unload() must fully release the previous instance before load().
template <typename LoadFn, typename UnloadFn>
static bool ensure_component(const char *        tag,
                             std::string &       loaded_path,
                             const std::string & want,
                             LoadFn              load,
                             UnloadFn            unload) {
    if (loaded_path == want) {
        return true;
    }
    Timer timer;
    if (!loaded_path.empty()) {
        fprintf(stderr, "[Pipeline] %s: unload %s\n", tag, loaded_path.c_str());
        unload();
        loaded_path.clear();
    }
    if (!load()) {
        fprintf(stderr, "[Pipeline] FATAL: %s load failed: %s\n", tag, want.c_str());
        return false;
    }
    loaded_path = want;
    fprintf(stderr, "[Pipeline] %s: %s, %.0f ms\n", tag, want.c_str(), timer.ms());
    return true;
}

bool pipeline_ensure(MM3Pipeline * p, const MM3ModelPaths & paths, const MM3PipelineParams & params) {
    p->params = params;
    debug_init(&p->dumper, params.dump_dir);
    if (params.clamp_fp16) {
        fprintf(stderr, "[Pipeline] FP16 clamp enabled\n");
    }
    if (!params.use_fa) {
        fprintf(stderr, "[Pipeline] Flash attention disabled\n");
    }

    // Tokenizer travels with the LM GGUF
    bool ok = ensure_component(
        "LM", p->loaded.lm, paths.lm,
        [&] {
            p->tok = BPETokenizer();
            if (!load_bpe_from_gguf(&p->tok, paths.lm.c_str()) ||
                !qw3lm_load(&p->lm, paths.lm.c_str(), params.max_seq, 2)) {
                return false;
            }
            p->lm.use_flash_attn = p->lm.use_flash_attn && params.use_fa;
            p->lm.clamp_fp16     = params.clamp_fp16;
            return true;
        },
        [&] { qw3lm_free(&p->lm); });
    ok = ok && ensure_component(
                   "Depth", p->loaded.depth, paths.depth, [&] { return p->depth.load(paths.depth.c_str()); },
                   [&] { p->depth.free(); });
    ok = ok && ensure_component(
                   "Cond", p->loaded.cond, paths.cond,
                   [&] {
                       if (!p->cond_enc.load(paths.cond.c_str())) {
                           return false;
                       }
                       p->cond_enc.clamp_fp16 = params.clamp_fp16;
                       return true;
                   },
                   [&] { p->cond_enc.free(); });
    ok = ok && ensure_component(
                   "DiT", p->loaded.dit, paths.dit,
                   [&] {
                       if (!p->dit.load(paths.dit.c_str())) {
                           return false;
                       }
                       p->dit.use_flash_attn = p->dit.use_flash_attn && params.use_fa;
                       p->dit.clamp_fp16     = params.clamp_fp16;
                       return true;
                   },
                   [&] { p->dit.free(); });
    ok = ok &&
         ensure_component(
             "VAE", p->loaded.vae, paths.vae, [&] { return p->vae.load(paths.vae.c_str()); }, [&] { p->vae.free(); });
    return ok;
}

PipelineStatus pipeline_generate(MM3Pipeline *        p,
                                 const MM3Request &   req,
                                 std::atomic<bool> *  cancel,
                                 std::vector<float> & audio_out) {
    Timer total_timer;

    // Prompt pair
    std::vector<int> cond_ids = mm3_build_prompt_ids(
        [&](const std::string & s) { return bpe_encode(&p->tok, s, false); }, req.caption, req.lyrics);
    std::vector<int> uncond_ids = cond_ids;
    for (size_t i = 1; i + 2 < uncond_ids.size(); i++) {
        uncond_ids[i] = MM3_AUDIO_CFG;
    }
    fprintf(stderr, "[Prompt] %zu tokens\n", cond_ids.size());

    const int H = p->lm.cfg.hidden_size;
    const int V = p->lm.cfg.vocab_size;

    // Allowed semantic sampling ids: the semantic range plus the end token
    std::vector<int> allowed;
    allowed.reserve(MM3_SEMANTIC_VOCAB + 1);
    for (int i = 0; i < MM3_SEMANTIC_VOCAB; i++) {
        allowed.push_back(MM3_AUDIO_CODE_OFFSET + i);
    }
    allowed.push_back(MM3_AUDIO_END);
    std::vector<int> depth_allowed(DepthDecoder::VOCAB);
    for (int i = 0; i < DepthDecoder::VOCAB; i++) {
        depth_allowed[i] = i;
    }

    // Autoregressive stage
    Timer           ar_timer;
    std::mt19937_64 rng((uint64_t) req.lm_seed);

    Timer              prefill_timer;
    std::vector<float> logits0(V), logits1(V), hidden0(H), hidden1(H);
    qw3lm_reset_kv(&p->lm, 0);
    qw3lm_reset_kv(&p->lm, 1);
    qw3lm_forward(&p->lm, cond_ids.data(), (int) cond_ids.size(), 0, logits0.data(), nullptr, hidden0.data());
    qw3lm_forward(&p->lm, uncond_ids.data(), (int) uncond_ids.size(), 1, logits1.data(), nullptr, hidden1.data());
    fprintf(stderr, "[AR] Prefill %.0f ms, %zu tokens, CFG=%.2f, top_k=%d\n", prefill_timer.ms(), cond_ids.size(),
            req.lm_cfg, req.lm_top_k);

    // Frame budget: requested duration, the model cap, and the KV room
    // left after the prompt (one decode per frame).
    int max_frames = std::min((int) (req.duration * FRAME_RATE), MAX_FRAMES);
    int kv_budget  = p->lm.cfg.max_seq_len - (int) cond_ids.size() - 1;
    if (max_frames > kv_budget) {
        fprintf(stderr, "[AR] Frame budget clamped to %d by the KV cache (prompt %zu tokens)\n", kv_budget,
                cond_ids.size());
        max_frames = kv_budget;
    }

    std::vector<float> frame_hiddens;  // [n_frames, 8, 4096]
    std::vector<float> seq0, seq1, depth_hid((size_t) 8 * H), depth_logits0(DepthDecoder::VOCAB),
        depth_logits1(DepthDecoder::VOCAB);
    std::vector<float> emb(H), feedback(H);
    std::vector<float> batch_embeds((size_t) 2 * H), batch_logits((size_t) 2 * V), batch_hidden((size_t) 2 * H);
    std::vector<float> seq2;

    for (int frame_index = 0; frame_index <= max_frames; frame_index++) {
        if (is_cancelled(cancel)) {
            return PIPELINE_CANCELLED;
        }
        int sampled = mm3_cfg_sample(logits0.data(), logits1.data(), allowed, req.lm_cfg, req.lm_top_k, rng);
        if (sampled == MM3_AUDIO_END) {
            fprintf(stderr, "[AR] end of audio token at frame %d\n", frame_index);
            break;
        }
        int semantic = sampled - MM3_AUDIO_CODE_OFFSET;

        mm3_lm_embed_row(&p->lm, sampled, emb.data());

        int                codes[8] = { semantic };
        std::vector<float> collected;  // 7 conditional hiddens
        if (p->params.use_batch_cfg) {
            // Whole frame in one fused graph: both CFG streams batch-2,
            // sampling in graph fed by host-drawn uniforms (same RNG
            // consumption order as the step-by-step path)
            float                                 u7[DepthDecoder::CODEBOOKS - 1];
            std::uniform_real_distribution<float> uni(0.0f, 1.0f);
            for (int i = 0; i < DepthDecoder::CODEBOOKS - 1; i++) {
                u7[i] = uni(rng);
            }
            seq2.assign(hidden0.begin(), hidden0.end());
            seq2.insert(seq2.end(), emb.begin(), emb.end());
            seq2.insert(seq2.end(), hidden1.begin(), hidden1.end());
            seq2.insert(seq2.end(), emb.begin(), emb.end());
            collected.resize((size_t) (DepthDecoder::CODEBOOKS - 1) * H);
            p->depth.forward_frame(seq2.data(), u7, req.lm_cfg, req.lm_top_k, codes + 1, collected.data());
        } else {
            // Depth loop: sequences share every step except the leading hidden
            seq0.assign(hidden0.begin(), hidden0.end());
            seq1.assign(hidden1.begin(), hidden1.end());
            seq0.insert(seq0.end(), emb.begin(), emb.end());
            seq1.insert(seq1.end(), emb.begin(), emb.end());
            for (int cb = 1; cb < DepthDecoder::CODEBOOKS; cb++) {
                int S = cb + 1;
                p->depth.forward(seq0.data(), S, depth_hid.data(), depth_logits0.data());
                collected.insert(collected.end(), depth_hid.begin() + (size_t) (S - 1) * H,
                                 depth_hid.begin() + (size_t) S * H);
                p->depth.forward(seq1.data(), S, depth_hid.data(), depth_logits1.data());
                codes[cb] = mm3_cfg_sample(depth_logits0.data(), depth_logits1.data(), depth_allowed, req.lm_cfg,
                                           req.lm_top_k, rng);
                if (cb < DepthDecoder::CODEBOOKS - 1) {
                    const float * row = p->depth.audio_embedding_row(cb, codes[cb]);
                    seq0.insert(seq0.end(), row, row + H);
                    seq1.insert(seq1.end(), row, row + H);
                }
            }
        }

        if (frame_index > 0) {
            frame_hiddens.insert(frame_hiddens.end(), hidden0.begin(), hidden0.end());
            frame_hiddens.insert(frame_hiddens.end(), collected.begin(), collected.end());
            if ((int) (frame_hiddens.size() / (8 * (size_t) H)) >= max_frames) {
                break;
            }
        }

        // Frame feedback: summed embeddings scaled by 8^-0.5
        mm3_lm_embed_row(&p->lm, sampled, feedback.data());
        for (int cb = 1; cb < DepthDecoder::CODEBOOKS; cb++) {
            const float * row = p->depth.audio_embedding_row(cb, codes[cb]);
            for (int i = 0; i < H; i++) {
                feedback[i] += row[i];
            }
        }
        float scale = 1.0f / sqrtf((float) DepthDecoder::CODEBOOKS);
        for (int i = 0; i < H; i++) {
            feedback[i] *= scale;
        }
        if (p->params.use_batch_cfg) {
            // Both CFG streams share the feedback embedding: one batch-2 decode
            std::copy(feedback.begin(), feedback.end(), batch_embeds.begin());
            std::copy(feedback.begin(), feedback.end(), batch_embeds.begin() + H);
            static const int KV_SETS[2] = { 0, 1 };
            qw3lm_forward_batch(&p->lm, nullptr, KV_SETS, 2, batch_logits.data(), 0, 0, batch_embeds.data(),
                                batch_hidden.data());
            std::copy(batch_logits.begin(), batch_logits.begin() + V, logits0.begin());
            std::copy(batch_logits.begin() + V, batch_logits.end(), logits1.begin());
            std::copy(batch_hidden.begin(), batch_hidden.begin() + H, hidden0.begin());
            std::copy(batch_hidden.begin() + H, batch_hidden.end(), hidden1.begin());
        } else {
            qw3lm_forward(&p->lm, nullptr, 1, 0, logits0.data(), feedback.data(), hidden0.data());
            qw3lm_forward(&p->lm, nullptr, 1, 1, logits1.data(), feedback.data(), hidden1.data());
        }

        if ((frame_index % 100) == 0) {
            fprintf(stderr, "[AR] Frame %d/%d\n", frame_index, max_frames);
        }
    }
    int n_frames = (int) (frame_hiddens.size() / (8 * (size_t) H));
    if (p->dumper.enabled) {
        int shape[3] = { n_frames, 8, H };
        debug_dump(&p->dumper, "frame_hiddens", frame_hiddens.data(), shape, 3);
    }
    fprintf(stderr, "[AR] %d frames (%.1fs of music), %.1f s (%.1f ms/frame)\n", n_frames,
            (float) n_frames / FRAME_RATE, ar_timer.ms() / 1000.0, n_frames > 0 ? ar_timer.ms() / n_frames : 0.0);
    if (n_frames == 0) {
        fprintf(stderr, "[AR] ERROR: generated zero audio frames\n");
        return PIPELINE_FAILED;
    }

    // Windowed flow matching
    Timer            synth_timer;
    std::vector<int> chunk_starts;
    if (n_frames <= CHUNK_FRAMES) {
        chunk_starts.push_back(0);
    } else {
        for (int s = 0; s < n_frames - CHUNK_HOP; s += CHUNK_HOP) {
            chunk_starts.push_back(s);
        }
    }

    // Ascending sigma schedule: linspace(1, 1/steps) inverted, final 1.0
    int                steps = req.steps;
    std::vector<float> sig(steps + 1);
    for (int i = 0; i < steps; i++) {
        float lin = 1.0f + (1.0f / (float) steps - 1.0f) * (float) i / (float) (steps - 1);
        sig[i]    = 1.0f - lin;
    }
    sig[steps] = 1.0f;

    std::vector<std::vector<float>> latent_chunks;  // each [T_lat, 128] time-major
    std::vector<int>                chunk_lat(chunk_starts.size());
    std::vector<float>              prev_latent, prev_condition;
    std::vector<float>              cond_track, zeros_track, noise_prompt;
    std::vector<float>              xt, v_cond, v_uncond;
    int64_t                         noise_index = 0;

    for (size_t k = 0; k < chunk_starts.size(); k++) {
        Timer window_timer;
        int   start = chunk_starts[k];
        int   end   = std::min(start + CHUNK_FRAMES, n_frames);

        std::vector<float> window(frame_hiddens.begin() + (size_t) start * 8 * H,
                                  frame_hiddens.begin() + (size_t) end * 8 * H);
        int                T_lat = 0;
        p->cond_enc.encode(window, end - start, cond_track, T_lat);
        chunk_lat[k] = T_lat;

        int overlap = 0;
        if (!prev_latent.empty()) {
            overlap = std::min((int) (prev_latent.size() / 128), T_lat);
            std::copy(prev_condition.begin(), prev_condition.begin() + (size_t) overlap * 2048, cond_track.begin());
        }

        // Initial noise, Philox stream continued across windows
        xt.resize((size_t) T_lat * 128);
        for (float & x : xt) {
            float vals[4];
            philox_normal4((uint64_t) req.seed, noise_index++, 0, vals);
            x = vals[0];
        }
        noise_prompt.assign(xt.begin(), xt.begin() + (size_t) overlap * 128);

        zeros_track.assign(cond_track.size(), 0.0f);
        v_cond.resize(xt.size());
        v_uncond.resize(xt.size());
        std::vector<float> v_cfg;

        bool dump_win = p->dumper.enabled && k == 0;
        if (dump_win) {
            debug_dump_2d(&p->dumper, "noise", xt.data(), T_lat, 128);
            v_cfg.resize(xt.size());
        }

        for (int i = 0; i < steps; i++) {
            if (is_cancelled(cancel)) {
                return PIPELINE_CANCELLED;
            }
            float t = sig[i];
            for (int j = 0; j < overlap * 128; j++) {
                xt[j] = (1.0f - (1.0f - 1e-6f) * t) * noise_prompt[j] + t * prev_latent[j];
            }
            p->dit.forward(xt.data(), cond_track.data(), T_lat, t, v_cond.data());
            if (dump_win && i == 0) {
                p->dit.dump_named(&p->dumper);
            }
            p->dit.forward(xt.data(), zeros_track.data(), T_lat, t, v_uncond.data());
            float dt = sig[i + 1] - sig[i];
            for (size_t j = 0; j < xt.size(); j++) {
                float v = v_uncond[j] + (v_cond[j] - v_uncond[j]) * req.dit_cfg;
                if (dump_win) {
                    v_cfg[j] = v;
                }
                xt[j] += dt * v;
            }
            if (dump_win) {
                char name[64];
                snprintf(name, sizeof(name), "dit_step%d_vt_cond", i);
                debug_dump_2d(&p->dumper, name, v_cond.data(), T_lat, 128);
                snprintf(name, sizeof(name), "dit_step%d_vt_uncond", i);
                debug_dump_2d(&p->dumper, name, v_uncond.data(), T_lat, 128);
                snprintf(name, sizeof(name), "dit_step%d_vt", i);
                debug_dump_2d(&p->dumper, name, v_cfg.data(), T_lat, 128);
                snprintf(name, sizeof(name), "dit_step%d_xt", i);
                debug_dump_2d(&p->dumper, name, xt.data(), T_lat, 128);
            }
        }
        if (dump_win) {
            debug_dump_2d(&p->dumper, "dit_x0", xt.data(), T_lat, 128);
        }
        for (int j = 0; j < overlap * 128; j++) {
            xt[j] = prev_latent[j];
        }

        int os = std::max(0, T_lat - 2 * OVERLAP_LATENT);
        int oe = std::max(os, T_lat - OVERLAP_LATENT);
        prev_latent.assign(xt.begin() + (size_t) os * 128, xt.begin() + (size_t) oe * 128);
        prev_condition.assign(cond_track.begin() + (size_t) os * 2048, cond_track.begin() + (size_t) oe * 2048);

        if (p->dumper.enabled) {
            char name[64];
            snprintf(name, sizeof(name), "window%zu_cond", k);
            debug_dump_2d(&p->dumper, name, cond_track.data(), T_lat, 2048);
            snprintf(name, sizeof(name), "window%zu_latent", k);
            debug_dump_2d(&p->dumper, name, xt.data(), T_lat, 128);
        }
        latent_chunks.push_back(xt);
        fprintf(stderr, "[DiT] Window %zu/%zu: T=%d, %d steps, %.0f ms (%.1f ms/step)\n", k + 1, chunk_starts.size(),
                T_lat, steps, window_timer.ms(), window_timer.ms() / steps);
    }
    fprintf(stderr, "[DiT] CFG=%.2f, %zu windows, %.1f s\n", req.dit_cfg, chunk_starts.size(),
            synth_timer.ms() / 1000.0);

    // Decode, crop, stitch
    Timer vae_timer;
    audio_out.clear();
    for (size_t k = 0; k < latent_chunks.size(); k++) {
        if (is_cancelled(cancel)) {
            return PIPELINE_CANCELLED;
        }
        int                T_lat = chunk_lat[k];
        std::vector<float> chan((size_t) 128 * T_lat);
        for (int t = 0; t < T_lat; t++) {
            for (int c = 0; c < 128; c++) {
                chan[(size_t) c * T_lat + t] = latent_chunks[k][(size_t) t * 128 + c];
            }
        }
        std::vector<float> wav;
        if (!p->vae.decode(chan, T_lat, wav)) {
            return PIPELINE_FAILED;
        }
        int left  = (k == 0) ? 0 : CROP_LEFT * HOP;
        int right = (k + 1 == latent_chunks.size()) ? 0 : CROP_RIGHT * HOP;
        audio_out.insert(audio_out.end(), wav.begin() + (size_t) left * 2, wav.end() - (size_t) right * 2);
    }
    if (p->dumper.enabled) {
        debug_dump_2d(&p->dumper, "vae_audio", audio_out.data(), (int) (audio_out.size() / 2), 2);
    }

    // Interleaved [T, 2] -> planar [L:T][R:T] for the output encoding stage
    int T = (int) (audio_out.size() / 2);
    {
        std::vector<float> planar((size_t) T * 2);
        for (int t = 0; t < T; t++) {
            planar[t]              = audio_out[(size_t) t * 2];
            planar[(size_t) T + t] = audio_out[(size_t) t * 2 + 1];
        }
        audio_out.swap(planar);
    }
    fprintf(stderr, "[VAE] Decode: %zu windows -> %.1fs of audio, %.0f ms\n", latent_chunks.size(),
            (float) T / (float) SAMPLE_RATE, vae_timer.ms());
    fprintf(stderr, "[Done] %.1f s total\n", total_timer.ms() / 1000.0);
    return PIPELINE_OK;
}
