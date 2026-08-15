// mm-lm.cpp: autoregressive stage CLI, prompt to audio codes
//
// Runs the global LM (semantic codebook, 25 Hz) and the RVQ depth
// decoder (7 acoustic codebooks per frame), and writes one request JSON
// per song with the sampled code stream in audio_codes. mm-synth and
// mm-server replay those requests deterministically without resampling:
// the expensive stochastic stage runs once, the synthesis parameters
// (models, steps, seed, CFG) stay free to iterate on.

#include "model-registry.h"
#include "pipeline.h"
#include "prompt.h"
#include "request.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "minimaxmusic.cpp %s\n\n", MM3_VERSION);
    fprintf(stderr,
            "Usage: %s --models <dir> --request <json> [options]\n"
            "       %s --models <dir> --caption <text> --lyrics <text> [options]\n"
            "\n"
            "Required:\n"
            "  --models <dir>         Directory of GGUF model files\n"
            "  --request <json>       Input request JSON (carries model routing)\n"
            "\n"
            "Optional:\n"
            "  --caption <text>       Caption (instead of --request)\n"
            "  --lyrics <text>        Lyrics (instead of --request)\n"
            "  --out <path>           Output request JSON (default: request.json)\n"
            "  --duration <s>         Target duration in seconds\n"
            "  --lm-seed <N>          Autoregressive sampling seed\n"
            "\n"
            "Output is numbered for batches: request.json -> request0.json ...\n"
            "\n"
            "Debug:\n"
            "  --max-seq <N>          LM KV cache size (default: model context)\n"
            "  --no-fa                Disable flash attention\n"
            "  --no-batch-cfg         Split CFG into two separate forwards (LM + DiT)\n"
            "  --clamp-fp16           Clamp hidden states to FP16 range\n"
            "  --dump-tokens <path>   Dump prompt token IDs (CSV)\n",
            prog, prog);
}

// write the prompt token dump
static bool write_file(const char * path, const std::string & data) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[LM] FATAL: cannot write %s\n", path);
        return false;
    }
    size_t wr = fwrite(data.data(), 1, data.size(), f);
    fclose(f);
    return wr == data.size();
}

int main(int argc, char ** argv) {
    std::string       models, out_path = "request.json", request_path, dump_tokens_path;
    MM3Request        req;
    MM3PipelineParams params;
    request_init(&req);

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--models" && i + 1 < argc) {
            models = argv[++i];
        } else if (a == "--request" && i + 1 < argc) {
            request_path = argv[++i];
        } else if (a == "--caption" && i + 1 < argc) {
            req.caption = argv[++i];
        } else if (a == "--lyrics" && i + 1 < argc) {
            req.lyrics = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "--duration" && i + 1 < argc) {
            req.duration = (float) atof(argv[++i]);
        } else if (a == "--lm-seed" && i + 1 < argc) {
            req.lm_seed = atoll(argv[++i]);
        } else if (a == "--max-seq" && i + 1 < argc) {
            params.max_seq = atoi(argv[++i]);
        } else if (a == "--no-fa") {
            params.use_fa = false;
        } else if (a == "--no-batch-cfg") {
            params.use_batch_cfg = false;
        } else if (a == "--clamp-fp16") {
            params.clamp_fp16 = true;
        } else if (a == "--dump-tokens" && i + 1 < argc) {
            dump_tokens_path = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (models.empty()) {
        fprintf(stderr, "[CLI] ERROR: --models required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (!request_path.empty() && !request_parse(&req, request_path.c_str())) {
        return 1;
    }
    if (req.caption.empty() || req.lyrics.empty()) {
        fprintf(stderr, "[LM] FATAL: caption and lyrics are required\n");
        return 1;
    }
    if (!req.audio_codes.empty()) {
        fprintf(stderr, "[LM] FATAL: the request already carries audio_codes\n");
        return 1;
    }
    request_resolve_lm_seed(&req);
    params.max_batch = req.lm_batch_size < 1 ? 1 : req.lm_batch_size;

    ModelRegistry reg;
    if (!registry_scan(&reg, models.c_str())) {
        fprintf(stderr, "[LM] FATAL: cannot scan models directory %s\n", models.c_str());
        return 1;
    }
    MM3ModelPaths paths;
    paths.lm    = registry_resolve(reg.lm, req.lm_model, "lm");
    paths.depth = registry_resolve(reg.depth, req.depth_model, "depth");
    if (paths.lm.empty() || paths.depth.empty()) {
        return 1;
    }

    // Model loads go through the store in STRICT policy; only the AR
    // group { LM, depth } is ever required here.
    ModelStore * store = store_create(EVICT_STRICT);
    MM3Pipeline  pipeline;
    pipeline.store = store;
    pipeline_configure(&pipeline, paths, params);

    if (!dump_tokens_path.empty()) {
        BPETokenizer * tok = store_bpe(store, paths.lm.c_str());
        if (!tok) {
            return 1;
        }
        std::vector<int> ids = mm3_build_prompt_ids([&](const std::string & s) { return bpe_encode(tok, s, false); },
                                                    req.caption, req.lyrics);
        std::string      csv;
        char             buf[16];
        for (size_t i = 0; i < ids.size(); i++) {
            snprintf(buf, sizeof(buf), i ? ",%d" : "%d", ids[i]);
            csv += buf;
        }
        csv += "\n";
        if (!write_file(dump_tokens_path.c_str(), csv)) {
            return 1;
        }
        fprintf(stderr, "[LM] Dumped %zu prompt token IDs to %s\n", ids.size(), dump_tokens_path.c_str());
    }

    std::vector<std::string> codes;
    if (pipeline_lm_generate(&pipeline, req, nullptr, codes) != PIPELINE_OK) {
        return 1;
    }

    // One replayable request per song: the input request with its song's
    // codes and consumed seed. song.json -> song0.json ...
    for (size_t i = 0; i < codes.size(); i++) {
        std::string path = out_path;
        if (codes.size() > 1) {
            size_t dot = out_path.rfind('.');
            path       = dot != std::string::npos ? out_path.substr(0, dot) + std::to_string(i) + out_path.substr(dot) :
                                                    out_path + std::to_string(i);
        }
        MM3Request replay = request_replay(req, codes[i], (int) i, 0);
        if (!request_write(&replay, path.c_str())) {
            return 1;
        }
    }
    store_free(store);
    return 0;
}
