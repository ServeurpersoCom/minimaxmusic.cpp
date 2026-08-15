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
#include <cstring>
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
    int k = (size_t) top_k < ranked.size() ? top_k : (int) ranked.size();
    std::partial_sort(
        ranked.begin(), ranked.begin() + k, ranked.end(),
        [](const std::pair<float, int> & a, const std::pair<float, int> & b) { return a.first > b.first; });
    ranked.resize(k);

    std::vector<float> guided(k);
    float              mx = -INFINITY;
    for (int i = 0; i < k; i++) {
        int id    = ranked[i].second;
        guided[i] = uncond[id] + (cond[id] - uncond[id]) * cfg;
        mx        = guided[i] > mx ? guided[i] : mx;
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

static bool ensure_ar_components(MM3Pipeline * p, const MM3ModelPaths & paths, const MM3PipelineParams & params);
static bool ensure_synth_components(MM3Pipeline * p, const MM3ModelPaths & paths, const MM3PipelineParams & params);

bool pipeline_ensure(MM3Pipeline * p, const MM3ModelPaths & paths, const MM3PipelineParams & params) {
    p->params = params;
    debug_init(&p->dumper, params.dump_dir);
    if (params.clamp_fp16) {
        fprintf(stderr, "[Pipeline] FP16 clamp enabled\n");
    }
    if (!params.use_fa) {
        fprintf(stderr, "[Pipeline] Flash attention disabled\n");
    }

    return ensure_ar_components(p, paths, params) && ensure_synth_components(p, paths, params);
}

// The autoregressive pair: tokenizer travels with the LM GGUF
static bool ensure_ar_components(MM3Pipeline * p, const MM3ModelPaths & paths, const MM3PipelineParams & params) {
    bool ok = ensure_component(
        "LM", p->loaded.lm, paths.lm,
        [&] {
            p->tok = BPETokenizer();
            if (!load_bpe_from_gguf(&p->tok, paths.lm.c_str()) ||
                !qw3lm_load(&p->lm, paths.lm.c_str(), params.max_seq,
                            2 * (params.max_batch < 1 ? 1 : params.max_batch))) {
                return false;
            }
            p->lm.use_flash_attn = p->lm.use_flash_attn && params.use_fa;
            p->lm.clamp_fp16     = params.clamp_fp16;
            return true;
        },
        [&] { qw3lm_free(&p->lm); });
    return ok && ensure_component(
                     "Depth", p->loaded.depth, paths.depth, [&] { return p->depth.load(paths.depth.c_str()); },
                     [&] { p->depth.free(); });
}

bool pipeline_ensure_lm(MM3Pipeline * p, const MM3ModelPaths & paths, const MM3PipelineParams & params) {
    p->params = params;
    debug_init(&p->dumper, params.dump_dir);
    if (params.clamp_fp16) {
        fprintf(stderr, "[Pipeline] FP16 clamp enabled\n");
    }
    if (!params.use_fa) {
        fprintf(stderr, "[Pipeline] Flash attention disabled\n");
    }
    return ensure_ar_components(p, paths, params);
}

static bool ensure_synth_components(MM3Pipeline * p, const MM3ModelPaths & paths, const MM3PipelineParams & params) {
    bool ok = ensure_component(
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

// Serialize a code stream to the audio_codes wire format: flat comma
// separated, 8 values per frame (semantic then the 7 acoustic codebooks)
static std::string codes_serialize(const std::vector<int> & codes) {
    std::string out;
    out.reserve(codes.size() * 6);
    char buf[16];
    for (size_t i = 0; i < codes.size(); i++) {
        snprintf(buf, sizeof(buf), i ? ",%d" : "%d", codes[i]);
        out += buf;
    }
    return out;
}

// Parse and validate the audio_codes wire format. Empty result = invalid.
static std::vector<int> codes_parse(const std::string & s) {
    std::vector<int> out;
    const char *     c = s.c_str();
    while (*c) {
        char * end  = nullptr;
        long   code = strtol(c, &end, 10);
        if (end == c) {
            fprintf(stderr, "[AR] FATAL: invalid audio_codes near \"%.16s\"\n", c);
            return {};
        }
        out.push_back((int) code);
        c = *end == ',' ? end + 1 : end;
    }
    if (out.size() < 16 || out.size() % 8 != 0) {
        fprintf(stderr, "[AR] FATAL: audio_codes needs 8 codes per frame, at least 2 frames (got %zu values)\n",
                out.size());
        return {};
    }
    for (size_t i = 0; i < out.size(); i++) {
        int hi = (i % 8 == 0) ? MM3_SEMANTIC_VOCAB : DepthDecoder::VOCAB;
        if (out[i] < 0 || out[i] >= hi) {
            fprintf(stderr, "[AR] FATAL: audio_codes value %d out of range at index %zu\n", out[i], i);
            return {};
        }
    }
    return out;
}

// Teacher-forced replay: re-derive the 8 hidden states per frame from an
// explicit code stream, no sampling. The LM runs the whole feedback
// sequence as one full forward (conditional stream only: the hiddens
// never depended on the CFG branch), the depth decoder runs one S=8
// causal forward per frame, positions 1..7 reproducing the step hiddens.
static PipelineStatus replay_stage(MM3Pipeline *        p,
                                   const MM3Request &   req,
                                   std::atomic<bool> *  cancel,
                                   std::vector<float> & frame_hiddens) {
    Timer ar_timer;

    std::vector<int> codes = codes_parse(req.audio_codes);
    if (codes.empty()) {
        return PIPELINE_FAILED;
    }
    int n = (int) (codes.size() / 8) - 1;  // hidden frames: frame 0 only feeds the first LM feedback

    std::vector<int> cond_ids = mm3_build_prompt_ids(
        [&](const std::string & str) { return bpe_encode(&p->tok, str, false); }, req.caption, req.lyrics);
    fprintf(stderr, "[Prompt] %zu tokens\n", cond_ids.size());

    const int H  = p->lm.cfg.hidden_size;
    const int V  = p->lm.cfg.vocab_size;
    const int NC = DepthDecoder::CODEBOOKS - 1;

    if (n > MAX_FRAMES || (int) cond_ids.size() + n >= p->lm.cfg.max_seq_len) {
        fprintf(stderr, "[AR] FATAL: audio_codes carry %d frames, over the budget (prompt %zu tokens)\n", n,
                cond_ids.size());
        return PIPELINE_FAILED;
    }

    qw3lm_reset_kv(&p->lm, 0);
    std::vector<float> logits(V);
    qw3lm_forward(&p->lm, cond_ids.data(), (int) cond_ids.size(), 0, logits.data());

    // Feedback embeddings f_0 .. f_{n-1}, one full-sequence forward
    std::vector<float> embeds((size_t) n * H), emb(H);
    float              scale = 1.0f / sqrtf((float) DepthDecoder::CODEBOOKS);
    for (int t = 0; t < n; t++) {
        const int * f = codes.data() + (size_t) t * 8;
        mm3_lm_embed_row(&p->lm, f[0] + MM3_AUDIO_CODE_OFFSET, emb.data());
        for (int cb = 1; cb <= NC; cb++) {
            const float * row = p->depth.audio_embedding_row(cb, f[cb]);
            for (int j = 0; j < H; j++) {
                emb[j] += row[j];
            }
        }
        for (int j = 0; j < H; j++) {
            emb[j] *= scale;
        }
        memcpy(embeds.data() + (size_t) t * H, emb.data(), (size_t) H * sizeof(float));
    }
    std::vector<float> all_hidden((size_t) n * H);
    qw3lm_forward(&p->lm, nullptr, n, 0, logits.data(), embeds.data(), nullptr, all_hidden.data());

    // Depth hiddens per frame: the whole step sequence is known upfront,
    // one causal S=8 forward reproduces the 7 step hiddens
    frame_hiddens.clear();
    frame_hiddens.reserve((size_t) n * 8 * H);
    std::vector<float> seq8((size_t) 8 * H), hid8((size_t) 8 * H), dlogits(DepthDecoder::VOCAB);
    for (int k = 1; k <= n; k++) {
        if (is_cancelled(cancel)) {
            return PIPELINE_CANCELLED;
        }
        const int *   f  = codes.data() + (size_t) k * 8;
        const float * hk = all_hidden.data() + (size_t) (k - 1) * H;
        memcpy(seq8.data(), hk, (size_t) H * sizeof(float));
        mm3_lm_embed_row(&p->lm, f[0] + MM3_AUDIO_CODE_OFFSET, seq8.data() + H);
        for (int cb = 1; cb < NC; cb++) {
            memcpy(seq8.data() + (size_t) (1 + cb) * H, p->depth.audio_embedding_row(cb, f[cb]),
                   (size_t) H * sizeof(float));
        }
        if (!p->depth.forward(seq8.data(), 8, hid8.data(), dlogits.data())) {
            return PIPELINE_FAILED;
        }
        frame_hiddens.insert(frame_hiddens.end(), hk, hk + H);
        frame_hiddens.insert(frame_hiddens.end(), hid8.begin() + H, hid8.end());
    }
    fprintf(stderr, "[AR] Replay: %d frames from audio_codes (%.1fs of music), %.1f s\n", n, (float) n / FRAME_RATE,
            ar_timer.ms() / 1000.0);
    return PIPELINE_OK;
}

// Autoregressive stage: N songs sampled in one batched pass. Records
// every song's code stream; hiddens_out is optional (null for mm-lm).
static PipelineStatus ar_stage(MM3Pipeline *                     p,
                               const MM3Request &                req,
                               std::atomic<bool> *               cancel,
                               int                               N,
                               std::vector<std::vector<float>> * hiddens_out,
                               std::vector<std::vector<int>> &   codes_out,
                               std::vector<int> &                n_frames) {
    // Prompt pair, shared by every song in the batch
    std::vector<int> cond_ids = mm3_build_prompt_ids(
        [&](const std::string & s) { return bpe_encode(&p->tok, s, false); }, req.caption, req.lyrics);
    std::vector<int> uncond_ids = cond_ids;
    for (size_t i = 1; i + 2 < uncond_ids.size(); i++) {
        uncond_ids[i] = MM3_AUDIO_CFG;
    }
    fprintf(stderr, "[Prompt] %zu tokens\n", cond_ids.size());

    const int H  = p->lm.cfg.hidden_size;
    const int V  = p->lm.cfg.vocab_size;
    const int NC = DepthDecoder::CODEBOOKS - 1;

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

    // Autoregressive stage. KV set blocks: [cond 0..N-1, uncond N..2N-1].
    // Song i samples with its own stream seeded lm_seed + i, so a song's
    // output only depends on its own seed, never on its batch siblings.
    Timer                        ar_timer;
    std::vector<std::mt19937_64> rng;
    for (int i = 0; i < N; i++) {
        rng.emplace_back((uint64_t) (req.lm_seed + i));
    }

    for (int s = 0; s < 2 * N; s++) {
        qw3lm_reset_kv(&p->lm, s);
    }

    // Shared prompt: one prefill per CFG branch, replicated to the other
    // songs by KV copy
    Timer              prefill_timer;
    std::vector<float> logits0(V), logits1(V), hidden0(H), hidden1(H);
    qw3lm_forward(&p->lm, cond_ids.data(), (int) cond_ids.size(), 0, logits0.data(), nullptr, hidden0.data());
    qw3lm_forward(&p->lm, uncond_ids.data(), (int) uncond_ids.size(), N, logits1.data(), nullptr, hidden1.data());
    for (int i = 1; i < N; i++) {
        qw3lm_copy_kv(&p->lm, 0, i);
        qw3lm_copy_kv(&p->lm, N, N + i);
    }
    fprintf(stderr, "[AR] Prefill %.0f ms, %zu tokens, CFG=%.2f, top_k=%d, songs=%d\n", prefill_timer.ms(),
            cond_ids.size(), req.lm_cfg, req.lm_top_k, N);

    // Frame budget: requested duration, the model cap, and the KV room
    // left after the prompt (one decode per frame).
    int max_frames = (int) (req.duration * FRAME_RATE);
    if (max_frames > MAX_FRAMES) {
        max_frames = MAX_FRAMES;
    }
    int kv_budget = p->lm.cfg.max_seq_len - (int) cond_ids.size() - 1;
    if (max_frames > kv_budget) {
        fprintf(stderr, "[AR] Frame budget clamped to %d by the KV cache (prompt %zu tokens)\n", kv_budget,
                cond_ids.size());
        max_frames = kv_budget;
    }

    // Current per-stream state in KV set order: rows [cond 0..N-1,
    // uncond N..2N-1], seeded from the shared prefill
    std::vector<float> cur_logits((size_t) 2 * N * V), cur_hidden((size_t) 2 * N * H);
    for (int i = 0; i < N; i++) {
        memcpy(cur_logits.data() + (size_t) i * V, logits0.data(), (size_t) V * sizeof(float));
        memcpy(cur_logits.data() + (size_t) (N + i) * V, logits1.data(), (size_t) V * sizeof(float));
        memcpy(cur_hidden.data() + (size_t) i * H, hidden0.data(), (size_t) H * sizeof(float));
        memcpy(cur_hidden.data() + (size_t) (N + i) * H, hidden1.data(), (size_t) H * sizeof(float));
    }

    // A song that ends stays in the batch as a passive row (stable graph
    // shapes, hot CUDA graphs); only its accumulation stops.
    std::vector<int>  frames(N, 0);
    std::vector<bool> done(N, false);
    std::vector<int>                sampled(N);
    std::vector<int>                kv_sets(2 * N);
    for (int s = 0; s < 2 * N; s++) {
        kv_sets[s] = s;
    }

    std::vector<float> seqN((size_t) 2 * N * 2 * H), uN((size_t) N * NC);
    std::vector<int>   codesN((size_t) N * NC);
    std::vector<float> collectedN((size_t) N * NC * H);
    std::vector<float> emb(H), feedback(H), depth_hid((size_t) 8 * H);
    std::vector<float> depth_logits0(DepthDecoder::VOCAB), depth_logits1(DepthDecoder::VOCAB);
    std::vector<float> seq0, seq1;
    std::vector<float> batch_embeds((size_t) 2 * N * H);

    for (int frame_index = 0; frame_index <= max_frames; frame_index++) {
        if (is_cancelled(cancel)) {
            return PIPELINE_CANCELLED;
        }

        bool all_done = true;
        for (int i = 0; i < N; i++) {
            sampled[i] = mm3_cfg_sample(cur_logits.data() + (size_t) i * V, cur_logits.data() + (size_t) (N + i) * V,
                                        allowed, req.lm_cfg, req.lm_top_k, rng[i]);
            if (!done[i] && sampled[i] == MM3_AUDIO_END) {
                fprintf(stderr, "[AR] Song %d: end of audio token at frame %d\n", i, frame_index);
                done[i] = true;
            }
            if (!done[i]) {
                all_done = false;
            }
        }
        if (all_done) {
            break;
        }

        if (p->params.use_batch_cfg) {
            // Whole frame in one fused graph across all songs: 7 depth
            // steps batch-2N, sampling in graph fed by host-drawn
            // uniforms (per song, same RNG consumption order as the
            // split path)
            std::uniform_real_distribution<float> uni(0.0f, 1.0f);
            for (int i = 0; i < N; i++) {
                mm3_lm_embed_row(&p->lm, sampled[i], emb.data());
                memcpy(seqN.data() + (size_t) i * 2 * H, cur_hidden.data() + (size_t) i * H,
                       (size_t) H * sizeof(float));
                memcpy(seqN.data() + ((size_t) i * 2 + 1) * H, emb.data(), (size_t) H * sizeof(float));
                memcpy(seqN.data() + (size_t) (N + i) * 2 * H, cur_hidden.data() + (size_t) (N + i) * H,
                       (size_t) H * sizeof(float));
                memcpy(seqN.data() + ((size_t) (N + i) * 2 + 1) * H, emb.data(), (size_t) H * sizeof(float));
                for (int j = 0; j < NC; j++) {
                    uN[(size_t) i * NC + j] = uni(rng[i]);
                }
            }
            if (!p->depth.forward_frame(seqN.data(), uN.data(), req.lm_cfg, req.lm_top_k, N, codesN.data(),
                                        collectedN.data())) {
                return PIPELINE_FAILED;
            }
        } else {
            // Split reference path: per song, per stream, per step
            for (int i = 0; i < N; i++) {
                mm3_lm_embed_row(&p->lm, sampled[i], emb.data());
                seq0.assign(cur_hidden.begin() + (size_t) i * H, cur_hidden.begin() + (size_t) (i + 1) * H);
                seq1.assign(cur_hidden.begin() + (size_t) (N + i) * H, cur_hidden.begin() + (size_t) (N + i + 1) * H);
                seq0.insert(seq0.end(), emb.begin(), emb.end());
                seq1.insert(seq1.end(), emb.begin(), emb.end());
                for (int cb = 1; cb < DepthDecoder::CODEBOOKS; cb++) {
                    int S = cb + 1;
                    p->depth.forward(seq0.data(), S, depth_hid.data(), depth_logits0.data());
                    memcpy(collectedN.data() + ((size_t) i * NC + cb - 1) * H, depth_hid.data() + (size_t) (S - 1) * H,
                           (size_t) H * sizeof(float));
                    p->depth.forward(seq1.data(), S, depth_hid.data(), depth_logits1.data());
                    int code = mm3_cfg_sample(depth_logits0.data(), depth_logits1.data(), depth_allowed, req.lm_cfg,
                                              req.lm_top_k, rng[i]);
                    codesN[(size_t) i * NC + cb - 1] = code;
                    if (cb < DepthDecoder::CODEBOOKS - 1) {
                        const float * row = p->depth.audio_embedding_row(cb, code);
                        seq0.insert(seq0.end(), row, row + H);
                        seq1.insert(seq1.end(), row, row + H);
                    }
                }
            }
        }

        // The recorded code stream covers one more frame than the
        // hiddens: frame 0 only feeds the first LM feedback
        for (int i = 0; i < N; i++) {
            if (done[i]) {
                continue;
            }
            codes_out[i].push_back(sampled[i] - MM3_AUDIO_CODE_OFFSET);
            codes_out[i].insert(codes_out[i].end(), codesN.begin() + (size_t) i * NC,
                                codesN.begin() + (size_t) (i + 1) * NC);
        }

        if (frame_index > 0) {
            for (int i = 0; i < N; i++) {
                if (done[i]) {
                    continue;
                }
                if (hiddens_out) {
                    (*hiddens_out)[i].insert((*hiddens_out)[i].end(), cur_hidden.begin() + (size_t) i * H,
                                             cur_hidden.begin() + (size_t) (i + 1) * H);
                    (*hiddens_out)[i].insert((*hiddens_out)[i].end(), collectedN.begin() + (size_t) i * NC * H,
                                             collectedN.begin() + (size_t) (i + 1) * NC * H);
                }
                frames[i]++;
                if (frames[i] >= max_frames) {
                    done[i] = true;
                }
            }
            bool budget_done = true;
            for (int i = 0; i < N; i++) {
                if (!done[i]) {
                    budget_done = false;
                }
            }
            if (budget_done) {
                break;
            }
        }

        // Frame feedback per song: summed embeddings scaled by 8^-0.5,
        // shared by the song's two CFG streams
        for (int i = 0; i < N; i++) {
            mm3_lm_embed_row(&p->lm, sampled[i], feedback.data());
            for (int cb = 1; cb < DepthDecoder::CODEBOOKS; cb++) {
                const float * row = p->depth.audio_embedding_row(cb, codesN[(size_t) i * NC + cb - 1]);
                for (int j = 0; j < H; j++) {
                    feedback[j] += row[j];
                }
            }
            float scale = 1.0f / sqrtf((float) DepthDecoder::CODEBOOKS);
            for (int j = 0; j < H; j++) {
                feedback[j] *= scale;
            }
            memcpy(batch_embeds.data() + (size_t) i * H, feedback.data(), (size_t) H * sizeof(float));
            memcpy(batch_embeds.data() + (size_t) (N + i) * H, feedback.data(), (size_t) H * sizeof(float));
        }
        if (p->params.use_batch_cfg) {
            qw3lm_forward_batch(&p->lm, nullptr, kv_sets.data(), 2 * N, cur_logits.data(), 0, 0, batch_embeds.data(),
                                cur_hidden.data());
        } else {
            for (int s = 0; s < 2 * N; s++) {
                qw3lm_forward(&p->lm, nullptr, 1, s, cur_logits.data() + (size_t) s * V,
                              batch_embeds.data() + (size_t) s * H, cur_hidden.data() + (size_t) s * H);
            }
        }

        if ((frame_index % 100) == 0) {
            fprintf(stderr, "[AR] Frame %d/%d\n", frame_index, max_frames);
        }
    }

    n_frames.assign(N, 0);
    for (int i = 0; i < N; i++) {
        n_frames[i] = frames[i];
        fprintf(stderr, "[AR] Song %d: %d frames (%.1fs of music)\n", i, n_frames[i], (float) n_frames[i] / FRAME_RATE);
        if (n_frames[i] == 0) {
            fprintf(stderr, "[AR] ERROR: song %d generated zero audio frames\n", i);
            return PIPELINE_FAILED;
        }
    }
    if (p->dumper.enabled && hiddens_out) {
        int shape[3] = { n_frames[0], 8, H };
        debug_dump(&p->dumper, "frame_hiddens", (*hiddens_out)[0].data(), shape, 3);
    }
    {
        int total = 0;
        for (int i = 0; i < N; i++) {
            total += n_frames[i];
        }
        fprintf(stderr, "[AR] %d frames total, %.1f s (%.1f ms/frame)\n", total, ar_timer.ms() / 1000.0,
                total > 0 ? ar_timer.ms() / total : 0.0);
    }
    return PIPELINE_OK;
}

PipelineStatus pipeline_generate(MM3Pipeline *                     p,
                                 const MM3Request &                req,
                                 std::atomic<bool> *               cancel,
                                 std::vector<std::vector<float>> & tracks_out) {
    Timer total_timer;

    int N = req.lm_batch_size < 1 ? 1 : req.lm_batch_size;
    if (N > p->params.max_batch) {
        fprintf(stderr, "[Pipeline] FATAL: lm_batch_size %d exceeds the batch limit %d\n", N, p->params.max_batch);
        return PIPELINE_FAILED;
    }

    const int H = p->lm.cfg.hidden_size;

    std::vector<std::vector<float>> frame_hiddens;
    std::vector<int>                n_frames;
    if (!req.audio_codes.empty()) {
        // Teacher-forced replay: the codes replace the sampling
        if (req.lm_batch_size > 1) {
            fprintf(stderr, "[AR] audio_codes provided: lm_batch_size ignored\n");
        }
        N = 1;
        frame_hiddens.resize(1);
        PipelineStatus st = replay_stage(p, req, cancel, frame_hiddens[0]);
        if (st != PIPELINE_OK) {
            return st;
        }
        n_frames.assign(1, (int) (frame_hiddens[0].size() / (8 * (size_t) H)));
        if (p->dumper.enabled) {
            int shape[3] = { n_frames[0], 8, H };
            debug_dump(&p->dumper, "frame_hiddens", frame_hiddens[0].data(), shape, 3);
        }
    } else {
        frame_hiddens.resize(N);
        std::vector<std::vector<int>> codes(N);
        PipelineStatus                st = ar_stage(p, req, cancel, N, &frame_hiddens, codes, n_frames);
        if (st != PIPELINE_OK) {
            return st;
        }
    }

    // Windowed flow matching and decode: per song, M noise variations
    // batched in the DiT on the shared condition track (seeds seed + j)
    int M = req.synth_batch_size < 1 ? 1 : (req.synth_batch_size > 9 ? 9 : req.synth_batch_size);
    tracks_out.assign((size_t) N * M, {});
    if (M > 1) {
        fprintf(stderr, "[Synth] %d variations per song\n", M);
    }
    for (int song = 0; song < N; song++) {
        if (N > 1) {
            fprintf(stderr, "[Synth] Song %d/%d\n", song + 1, N);
        }
        Timer            synth_timer;
        std::vector<int> chunk_starts;
        if (n_frames[song] <= CHUNK_FRAMES) {
            chunk_starts.push_back(0);
        } else {
            for (int s = 0; s < n_frames[song] - CHUNK_HOP; s += CHUNK_HOP) {
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

        // Per-variation window state; the condition track is shared
        std::vector<std::vector<std::vector<float>>> latent_chunks(M);  // [variation][window][T_lat, 128]
        std::vector<int>                             chunk_lat(chunk_starts.size());
        std::vector<std::vector<float>>              prev_latent(M), noise_prompt(M), xt(M);
        std::vector<int64_t>                         noise_index(M, 0);
        std::vector<float>                           prev_condition, cond_track, zeros_track;
        std::vector<float>                           v_cond, v_uncond;

        for (size_t k = 0; k < chunk_starts.size(); k++) {
            Timer window_timer;
            int   start = chunk_starts[k];
            int   end   = start + CHUNK_FRAMES < n_frames[song] ? start + CHUNK_FRAMES : n_frames[song];

            std::vector<float> window(frame_hiddens[song].begin() + (size_t) start * 8 * H,
                                      frame_hiddens[song].begin() + (size_t) end * 8 * H);
            int                T_lat = 0;
            p->cond_enc.encode(window, end - start, cond_track, T_lat);
            chunk_lat[k] = T_lat;

            int overlap = 0;
            if (!prev_latent[0].empty()) {
                overlap = (int) (prev_latent[0].size() / 128) < T_lat ? (int) (prev_latent[0].size() / 128) : T_lat;
                std::copy(prev_condition.begin(), prev_condition.begin() + (size_t) overlap * 2048, cond_track.begin());
            }

            // Initial noise per variation, Philox stream continued across
            // windows, one seed per variation
            for (int j = 0; j < M; j++) {
                xt[j].resize((size_t) T_lat * 128);
                for (float & x : xt[j]) {
                    float vals[4];
                    philox_normal4((uint64_t) (req.seed + j), noise_index[j]++, 0, vals);
                    x = vals[0];
                }
                noise_prompt[j].assign(xt[j].begin(), xt[j].begin() + (size_t) overlap * 128);
            }

            zeros_track.assign(cond_track.size(), 0.0f);
            size_t             lat_sz = xt[0].size();
            std::vector<float> v_cfg;
            std::vector<float> xt2, cond2, v2;
            v_cond.resize(lat_sz);
            v_uncond.resize(lat_sz);
            if (p->params.use_batch_cfg) {
                // All variations and both CFG branches in one batch-2M
                // forward: [cond track x M, zeros x M]
                cond2.resize(cond_track.size() * 2 * M);
                for (int j = 0; j < M; j++) {
                    memcpy(cond2.data() + (size_t) j * cond_track.size(), cond_track.data(),
                           cond_track.size() * sizeof(float));
                }
                memset(cond2.data() + (size_t) M * cond_track.size(), 0,
                       (size_t) M * cond_track.size() * sizeof(float));
                xt2.resize(lat_sz * 2 * M);
                v2.resize(lat_sz * 2 * M);
            }

            bool dump_win = p->dumper.enabled && song == 0 && k == 0;
            if (dump_win) {
                debug_dump_2d(&p->dumper, "noise", xt[0].data(), T_lat, 128);
                v_cfg.resize(lat_sz);
            }

            for (int i = 0; i < steps; i++) {
                if (is_cancelled(cancel)) {
                    return PIPELINE_CANCELLED;
                }
                float t = sig[i];
                for (int j = 0; j < M; j++) {
                    for (int e = 0; e < overlap * 128; e++) {
                        xt[j][e] = (1.0f - (1.0f - 1e-6f) * t) * noise_prompt[j][e] + t * prev_latent[j][e];
                    }
                }
                float dt = sig[i + 1] - sig[i];
                if (p->params.use_batch_cfg) {
                    for (int j = 0; j < M; j++) {
                        memcpy(xt2.data() + (size_t) j * lat_sz, xt[j].data(), lat_sz * sizeof(float));
                        memcpy(xt2.data() + (size_t) (M + j) * lat_sz, xt[j].data(), lat_sz * sizeof(float));
                    }
                    p->dit.forward(xt2.data(), cond2.data(), T_lat, 2 * M, t, v2.data());
                    if (dump_win && i == 0) {
                        p->dit.dump_named(&p->dumper);
                    }
                    for (int j = 0; j < M; j++) {
                        const float * vc = v2.data() + (size_t) j * lat_sz;
                        const float * vu = v2.data() + (size_t) (M + j) * lat_sz;
                        for (size_t e = 0; e < lat_sz; e++) {
                            float v = vu[e] + (vc[e] - vu[e]) * req.dit_cfg;
                            if (dump_win && j == 0) {
                                v_cfg[e] = v;
                            }
                            xt[j][e] += dt * v;
                        }
                        if (dump_win && j == 0) {
                            memcpy(v_cond.data(), vc, lat_sz * sizeof(float));
                            memcpy(v_uncond.data(), vu, lat_sz * sizeof(float));
                        }
                    }
                } else {
                    for (int j = 0; j < M; j++) {
                        p->dit.forward(xt[j].data(), cond_track.data(), T_lat, 1, t, v_cond.data());
                        if (dump_win && j == 0 && i == 0) {
                            p->dit.dump_named(&p->dumper);
                        }
                        p->dit.forward(xt[j].data(), zeros_track.data(), T_lat, 1, t, v_uncond.data());
                        for (size_t e = 0; e < lat_sz; e++) {
                            float v = v_uncond[e] + (v_cond[e] - v_uncond[e]) * req.dit_cfg;
                            if (dump_win && j == 0) {
                                v_cfg[e] = v;
                            }
                            xt[j][e] += dt * v;
                        }
                    }
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
                    debug_dump_2d(&p->dumper, name, xt[0].data(), T_lat, 128);
                }
            }
            if (dump_win) {
                debug_dump_2d(&p->dumper, "dit_x0", xt[0].data(), T_lat, 128);
            }
            for (int j = 0; j < M; j++) {
                for (int e = 0; e < overlap * 128; e++) {
                    xt[j][e] = prev_latent[j][e];
                }
            }

            int os = T_lat - 2 * OVERLAP_LATENT > 0 ? T_lat - 2 * OVERLAP_LATENT : 0;
            int oe = T_lat - OVERLAP_LATENT > os ? T_lat - OVERLAP_LATENT : os;
            for (int j = 0; j < M; j++) {
                prev_latent[j].assign(xt[j].begin() + (size_t) os * 128, xt[j].begin() + (size_t) oe * 128);
            }
            prev_condition.assign(cond_track.begin() + (size_t) os * 2048, cond_track.begin() + (size_t) oe * 2048);

            if (p->dumper.enabled && song == 0) {
                char name[64];
                snprintf(name, sizeof(name), "window%zu_cond", k);
                debug_dump_2d(&p->dumper, name, cond_track.data(), T_lat, 2048);
                snprintf(name, sizeof(name), "window%zu_latent", k);
                debug_dump_2d(&p->dumper, name, xt[0].data(), T_lat, 128);
            }
            for (int j = 0; j < M; j++) {
                latent_chunks[j].push_back(xt[j]);
            }
            fprintf(stderr, "[DiT] Window %zu/%zu: T=%d, %d steps, %.0f ms (%.1f ms/step)\n", k + 1,
                    chunk_starts.size(), T_lat, steps, window_timer.ms(), window_timer.ms() / steps);
        }
        fprintf(stderr, "[DiT] CFG=%.2f, %zu windows, %.1f s\n", req.dit_cfg, chunk_starts.size(),
                synth_timer.ms() / 1000.0);

        // Decode, crop, stitch each variation
        Timer vae_timer;
        int   T_last = 0;
        for (int j = 0; j < M; j++) {
            std::vector<float> & audio_out = tracks_out[(size_t) song * M + j];
            for (size_t k = 0; k < latent_chunks[j].size(); k++) {
                if (is_cancelled(cancel)) {
                    return PIPELINE_CANCELLED;
                }
                int                T_lat = chunk_lat[k];
                std::vector<float> chan((size_t) 128 * T_lat);
                for (int t = 0; t < T_lat; t++) {
                    for (int c = 0; c < 128; c++) {
                        chan[(size_t) c * T_lat + t] = latent_chunks[j][k][(size_t) t * 128 + c];
                    }
                }
                std::vector<float> wav;
                if (!p->vae.decode(chan, T_lat, wav)) {
                    return PIPELINE_FAILED;
                }
                int left  = (k == 0) ? 0 : CROP_LEFT * HOP;
                int right = (k + 1 == latent_chunks[j].size()) ? 0 : CROP_RIGHT * HOP;
                audio_out.insert(audio_out.end(), wav.begin() + (size_t) left * 2, wav.end() - (size_t) right * 2);
            }
            if (p->dumper.enabled && song == 0 && j == 0) {
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
            T_last = T;
        }
        fprintf(stderr, "[VAE] Decode: %zu windows -> %.1fs of audio, %.0f ms\n", chunk_starts.size(),
                (float) T_last / (float) SAMPLE_RATE, vae_timer.ms());
    }
    fprintf(stderr, "[Done] %.1f s total\n", total_timer.ms() / 1000.0);
    return PIPELINE_OK;
}

PipelineStatus pipeline_lm_generate(MM3Pipeline *              p,
                                    const MM3Request &         req,
                                    std::atomic<bool> *        cancel,
                                    std::vector<std::string> & codes_out) {
    Timer total_timer;

    int N = req.lm_batch_size < 1 ? 1 : req.lm_batch_size;
    if (N > p->params.max_batch) {
        fprintf(stderr, "[Pipeline] FATAL: lm_batch_size %d exceeds the batch limit %d\n", N, p->params.max_batch);
        return PIPELINE_FAILED;
    }

    std::vector<std::vector<int>> codes(N);
    std::vector<int>              n_frames;
    PipelineStatus                st = ar_stage(p, req, cancel, N, nullptr, codes, n_frames);
    if (st != PIPELINE_OK) {
        return st;
    }
    codes_out.resize(N);
    for (int i = 0; i < N; i++) {
        codes_out[i] = codes_serialize(codes[i]);
    }
    fprintf(stderr, "[Done] %.1f s total\n", total_timer.ms() / 1000.0);
    return PIPELINE_OK;
}
