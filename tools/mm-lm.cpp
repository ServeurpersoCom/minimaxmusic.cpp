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
#include <cstring>
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

static std::string read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[LM] FATAL: cannot open %s\n", path);
        return "";
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        fclose(f);
        fprintf(stderr, "[LM] FATAL: empty file %s\n", path);
        return "";
    }
    std::string data((size_t) size, '\0');
    size_t      rd = fread(&data[0], 1, (size_t) size, f);
    fclose(f);
    if (rd != (size_t) size) {
        fprintf(stderr, "[LM] FATAL: short read on %s\n", path);
        return "";
    }
    return data;
}

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

static std::string resolve_model(const std::vector<ModelEntry> & bucket, const std::string & requested,
                                 const char * component) {
    if (bucket.empty()) {
        fprintf(stderr, "[LM] FATAL: no %s model found in --models directory\n", component);
        return "";
    }
    if (requested.empty()) {
        return bucket.front().path;
    }
    const ModelEntry * e = registry_find(bucket, requested.c_str());
    if (!e) {
        fprintf(stderr, "[LM] FATAL: unknown %s model %s\n", component, requested.c_str());
        return "";
    }
    return e->path;
}

int main(int argc, char ** argv) {
    std::string       models = "./models", out_path = "request.json", request_path, dump_tokens_path;
    MM3Request        req;
    MM3PipelineParams params;
    request_init(&req);

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

    if (!request_path.empty()) {
        std::string json = read_file(request_path.c_str());
        if (json.empty() || !request_parse_json(&req, json.c_str())) {
            fprintf(stderr, "[LM] FATAL: invalid request JSON %s\n", request_path.c_str());
            return 1;
        }
    }
    if (req.caption.empty() || req.lyrics.empty()) {
        fprintf(stderr, "[LM] FATAL: caption and lyrics are required\n");
        print_usage(argv[0]);
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
    paths.lm    = resolve_model(reg.lm, req.lm_model, "lm");
    paths.depth = resolve_model(reg.depth, req.depth_model, "depth");
    if (paths.lm.empty() || paths.depth.empty()) {
        return 1;
    }

    MM3Pipeline pipeline;
    if (!pipeline_ensure_lm(&pipeline, paths, params)) {
        return 1;
    }

    if (!dump_tokens_path.empty()) {
        std::vector<int> ids = mm3_build_prompt_ids(
            [&](const std::string & s) { return bpe_encode(&pipeline.tok, s, false); }, req.caption, req.lyrics);
        std::string csv;
        char        buf[16];
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
    // codes and traceable per-song seed. song.json -> song0.json ...
    for (size_t i = 0; i < codes.size(); i++) {
        MM3Request out       = req;
        out.audio_codes      = codes[i];
        out.lm_seed          = req.lm_seed + (int64_t) i;
        out.lm_batch_size    = 1;
        std::string path     = out_path;
        if (codes.size() > 1) {
            size_t dot = out_path.rfind('.');
            path       = dot != std::string::npos ? out_path.substr(0, dot) + std::to_string(i) + out_path.substr(dot) :
                                                    out_path + std::to_string(i);
        }
        if (!write_file(path.c_str(), request_to_json(&out) + "\n")) {
            return 1;
        }
        fprintf(stderr, "[Out] %s\n", path.c_str());
    }
    return 0;
}
