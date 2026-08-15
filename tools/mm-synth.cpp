// mm-synth.cpp: command line MiniMax Music 3 generation, text to WAV file
//
// Thin CLI over src/pipeline: build an MM3Request from a JSON file or
// from flags, resolve models through the registry, generate, write the
// PCM16 WAV. Shares the request contract with mm-server.

#include "audio-io.h"
#include "model-registry.h"
#include "pipeline.h"
#include "request.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
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
            "  --out <path>           Output audio path (default: out.mp3)\n"
            "  --duration <s>         Target duration in seconds\n"
            "  --steps <N>            Euler steps per DiT window\n"
            "  --seed <N>             DiT noise seed\n"
            "  --lm-seed <N>          Autoregressive sampling seed\n"
            "\n"
            "Debug:\n"
            "  --max-seq <N>          LM KV cache size (default: model context)\n"
            "  --no-fa                Disable flash attention\n"
            "  --no-batch-cfg         Split CFG into two separate forwards (LM + DiT)\n"
            "  --clamp-fp16           Clamp hidden states to FP16 range\n"
            "  --dump <dir>           Dump intermediate tensors\n",
            argv0, argv0);
}

int main(int argc, char ** argv) {
    std::string       models, out_path = "out.mp3", request_path;
    MM3Request        req;
    MM3PipelineParams params;
    request_init(&req);
    req.seed    = 42;
    req.lm_seed = 42;

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
        } else if (a == "--steps" && i + 1 < argc) {
            req.steps = atoi(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            req.seed = atoll(argv[++i]);
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
        } else if (a == "--dump" && i + 1 < argc) {
            params.dump_dir = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (!request_path.empty() && !request_parse(&req, request_path.c_str())) {
        return 1;
    }
    if (models.empty()) {
        fprintf(stderr, "[CLI] ERROR: --models required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (req.caption.empty() || req.lyrics.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    request_resolve_seed(&req);
    request_resolve_lm_seed(&req);
    params.max_batch = req.lm_batch_size < 1 ? 1 : req.lm_batch_size;

    ModelRegistry reg;
    registry_scan(&reg, models.c_str());
    MM3ModelPaths paths;
    paths.lm    = registry_resolve(reg.lm, req.lm_model, "lm");
    paths.depth = registry_resolve(reg.depth, req.depth_model, "depth");
    paths.cond  = registry_resolve(reg.cond, req.cond_model, "cond");
    paths.dit   = registry_resolve(reg.dit, req.dit_model, "dit");
    paths.vae   = registry_resolve(reg.vae, req.vae_model, "vae");
    if (paths.lm.empty() || paths.depth.empty() || paths.cond.empty() || paths.dit.empty() || paths.vae.empty()) {
        return 1;
    }

    // Model loads go through the store in STRICT policy: at most one
    // coexistence group resident, the LM and the DiT never overlap.
    ModelStore * store = store_create(EVICT_STRICT);
    MM3Pipeline  pipeline;
    pipeline.store = store;
    pipeline_configure(&pipeline, paths, params);

    bool      is_mp3  = true;
    WavFormat wav_fmt = WAV_S16;
    if (!audio_parse_format(req.output_format.c_str(), is_mp3, wav_fmt)) {
        fprintf(stderr, "[Out] FATAL: invalid output_format (use: mp3, wav16, wav24, wav32)\n");
        return 1;
    }

    std::vector<std::vector<float>> tracks;
    std::vector<std::string>        codes;
    if (pipeline_generate(&pipeline, req, nullptr, tracks, &codes) != PIPELINE_OK) {
        return 1;
    }

    // A single track lands on --out as given; a batch numbers the base
    // with song then variation index: song.mp3 -> song00.mp3 ...
    // Every track gets its replay request next to it (.json), carrying
    // the audio_codes and the exact seed of that track.
    int M = (int) tracks.size() / (req.lm_batch_size < 1 ? 1 : req.lm_batch_size);
    for (size_t i = 0; i < tracks.size(); i++) {
        std::string path = out_path;
        if (tracks.size() > 1) {
            std::string idx = std::to_string(i / M) + std::to_string(i % M);
            size_t      dot = out_path.rfind('.');
            path = dot != std::string::npos ? out_path.substr(0, dot) + idx + out_path.substr(dot) : out_path + idx;
        }
        std::vector<float> & audio   = tracks[i];
        int                  T_audio = (int) (audio.size() / 2);
        if (!audio_write(path.c_str(), audio.data(), T_audio, 44100, is_mp3, wav_fmt, req.mp3_bitrate, req.peak_clip)) {
            return 1;
        }

        size_t      pdot      = path.rfind('.');
        std::string json_path = (pdot != std::string::npos ? path.substr(0, pdot) : path) + ".json";
        MM3Request  replay    = request_replay(req, codes[i / M], (int) (i / M), (int) (i % M));
        if (!request_write(&replay, json_path.c_str())) {
            return 1;
        }
    }
    store_free(store);
    return 0;
}
