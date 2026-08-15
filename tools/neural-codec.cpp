// neural-codec.cpp: neural audio codec (flow VAE encoder + decoder)
//
// encode: 44.1 kHz stereo audio (WAV or MP3) -> latent file (.vae)
// decode: latent file (.vae) -> 44.1 kHz stereo audio (WAV or MP3)
//
// Latent format:
//   .vae: flat [T, 128] f32 frame-major, no header.
//     T = file_size / 512. 86.13 Hz, ~353 kbit/s.
//
// The decoder weights come from the published vocoder; the encoder is its
// bit-identical training companion (dav.pth), both in the vocoder GGUF.
// The encode is deterministic: the posterior mean, no sampling.
//
// Long signals run in tiles with symmetric overlap: each tile is
// processed with context on both sides and the overlap is cropped, so
// the seams fall far beyond the receptive field.
//
// Usage:
//   neural-codec --vae model.gguf --encode -i song.wav -o song.vae
//   neural-codec --vae model.gguf --decode -i song.vae -o song.wav

#include "audio-io.h"
#include "vae-enc.h"
#include "vae.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int LATENT_CH = 128;
static const int HOP       = 512;

static void print_usage(const char * prog) {
    fprintf(stderr, "minimaxmusic.cpp %s\n\n", MM3_VERSION);
    fprintf(stderr,
            "Usage: %s --vae <gguf> --encode|--decode -i <input> [-o <output>] [options]\n"
            "\n"
            "Required:\n"
            "  --vae <path>            VAE GGUF file\n"
            "  --encode | --decode     Encode audio to latent, or decode latent to audio\n"
            "  -i <path>               Input (WAV/MP3 for encode, .vae for decode)\n"
            "\n"
            "Output:\n"
            "  -o <path>               Output file (auto-named if omitted)\n"
            "  --format <fmt>          mp3, wav16, wav24, wav32 (default: wav16)\n"
            "\n"
            "Output naming: song.wav -> song.vae, song.vae -> song.wav\n"
            "\n"
            "Memory control:\n"
            "  --vae-chunk <N>         Latent frames per tile (default: 689)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: 86)\n"
            "\n"
            "Latent format:\n"
            "  .vae: flat [T, 128] f32, no header. 86.13 Hz, ~353 kbit/s.\n",
            prog);
}

static std::string auto_output(const char * input, const char * ext) {
    std::string s   = input;
    size_t      dot = s.rfind('.');
    if (dot != std::string::npos) {
        return s.substr(0, dot) + ext;
    }
    return s + ext;
}

// Read a .vae latent file. Returns frame-major [T, 128]. Caller frees.
static float * read_latent(const char * path, int * T_latent) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[Latent] Cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    long frame_bytes = (long) LATENT_CH * sizeof(float);
    if (fsize <= 0 || fsize % frame_bytes != 0) {
        fprintf(stderr, "[Latent] Invalid .vae size %ld (must be a multiple of %ld)\n", fsize, frame_bytes);
        fclose(f);
        return NULL;
    }
    long t = fsize / frame_bytes;

    float * data = (float *) malloc((size_t) fsize);
    if (!data) {
        fprintf(stderr, "[Latent] OOM allocating %ld frames\n", t);
        fclose(f);
        return NULL;
    }
    if (fread(data, 1, (size_t) fsize, f) != (size_t) fsize) {
        fprintf(stderr, "[Latent] Short read on %s\n", path);
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);

    *T_latent = (int) t;
    fprintf(stderr, "[Latent] Read %s: %d frames (%.2fs)\n", path, (int) t, (float) t * HOP / 44100.0f);
    return data;
}

int main(int argc, char ** argv) {
    const char * vae_path    = NULL;
    const char * input_path  = NULL;
    const char * output_path = NULL;
    const char * format      = "wav16";
    int          mode        = -1;
    int          chunk_size  = 689;
    int          overlap     = 86;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--vae") == 0 && i + 1 < argc) {
            vae_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            format = argv[++i];
        } else if (strcmp(argv[i], "--vae-chunk") == 0 && i + 1 < argc) {
            chunk_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vae-overlap") == 0 && i + 1 < argc) {
            overlap = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--decode") == 0) {
            mode = 1;
        } else if (strcmp(argv[i], "--encode") == 0) {
            mode = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!vae_path || !input_path || mode < 0) {
        print_usage(argv[0]);
        return 1;
    }
    if (chunk_size < 1 || overlap < 0) {
        fprintf(stderr, "[VAE] FATAL: invalid tiling (chunk %d, overlap %d)\n", chunk_size, overlap);
        return 1;
    }

    bool      is_mp3  = false;
    WavFormat wav_fmt = WAV_S16;
    if (!audio_parse_format(format, is_mp3, wav_fmt)) {
        fprintf(stderr, "Unknown format: %s\n", format);
        print_usage(argv[0]);
        return 1;
    }

    std::string out_str;
    if (!output_path) {
        out_str     = auto_output(input_path, mode == 0 ? ".vae" : (is_mp3 ? ".mp3" : ".wav"));
        output_path = out_str.c_str();
    }

    fprintf(stderr, "\n[VAE] Mode: %s\n", mode == 0 ? "encode" : "decode");
    fprintf(stderr, "[VAE] Input: %s\n", input_path);
    fprintf(stderr, "[VAE] Output: %s\n\n", output_path);

    if (mode == 0) {
        // Encode: audio -> .vae. Deterministic (posterior mean).
        int     T_audio = 0, sr = 0;
        float * planar = audio_read(input_path, &T_audio, &sr);
        if (!planar) {
            return 1;
        }
        if (sr != 44100) {
            fprintf(stderr, "[VAE] Resampling %d Hz -> 44100 Hz\n", sr);
            int     T_rs  = 0;
            float * left  = audio_resample(planar, T_audio, sr, 44100, 1, &T_rs);
            float * right = audio_resample(planar + T_audio, T_audio, sr, 44100, 1, &T_rs);
            free(planar);
            if (!left || !right) {
                fprintf(stderr, "[VAE] FATAL: resampling failed\n");
                return 1;
            }
            planar = (float *) malloc((size_t) T_rs * 2 * sizeof(float));
            memcpy(planar, left, (size_t) T_rs * sizeof(float));
            memcpy(planar + T_rs, right, (size_t) T_rs * sizeof(float));
            free(left);
            free(right);
            T_audio = T_rs;
        }

        // Right pad with silence to whole latent frames
        int T_latent = (T_audio + HOP - 1) / HOP;
        int T_padded = T_latent * HOP;

        VAEEncoder enc = {};
        if (!enc.load(vae_path)) {
            free(planar);
            return 1;
        }

        // Tiled encode: [start - overlap, end + overlap] clipped to the
        // signal, the encoded overlap frames cropped from each side.
        std::vector<float> out((size_t) T_latent * LATENT_CH);
        int                n_tiles = (T_latent + chunk_size - 1) / chunk_size;
        fprintf(stderr, "[VAE] Encoding %d samples (%.2fs), %d tile(s)...\n", T_audio, (float) T_audio / 44100.0f,
                n_tiles);
        for (int start = 0; start < T_latent; start += chunk_size) {
            int end = start + chunk_size < T_latent ? start + chunk_size : T_latent;
            int lo  = start - overlap > 0 ? start - overlap : 0;
            int hi  = end + overlap < T_latent ? end + overlap : T_latent;
            int T_t = hi - lo;

            // padded planar window -> interleaved [T_t * HOP, 2]
            std::vector<float> tile((size_t) T_t * HOP * 2, 0.0f);
            for (int t = 0; t < T_t * HOP; t++) {
                int s = lo * HOP + t;
                if (s < T_audio) {
                    tile[(size_t) t * 2 + 0] = planar[s];
                    tile[(size_t) t * 2 + 1] = planar[(size_t) T_audio + s];
                }
            }

            std::vector<float> lat;  // channel-major [128, T_t]
            int                T_out = 0;
            if (!enc.encode(tile.data(), T_t * HOP, lat, T_out)) {
                free(planar);
                enc.free();
                return 1;
            }

            // crop the overlap, transpose to the frame-major file layout
            for (int t = start; t < end; t++) {
                for (int c = 0; c < LATENT_CH; c++) {
                    out[(size_t) t * LATENT_CH + c] = lat[(size_t) c * T_t + (t - lo)];
                }
            }
        }
        free(planar);
        enc.free();

        FILE * f = fopen(output_path, "wb");
        if (!f || fwrite(out.data(), 1, out.size() * sizeof(float), f) != out.size() * sizeof(float)) {
            fprintf(stderr, "[VAE] FATAL: failed to write %s\n", output_path);
            if (f) {
                fclose(f);
            }
            return 1;
        }
        fclose(f);
        fprintf(stderr, "[VAE] Output: %s (%d frames, %.2fs, %.1f KB)\n", output_path, T_latent,
                (float) T_padded / 44100.0f, (float) (out.size() * sizeof(float)) / 1024.0f);
        fprintf(stderr, "[VAE] Done.\n");
        return 0;
    }

    int     T_latent = 0;
    float * latent   = read_latent(input_path, &T_latent);
    if (!latent) {
        return 1;
    }

    FlowVAE vae = {};
    if (!vae.load(vae_path)) {
        free(latent);
        return 1;
    }

    // Tiled decode: [start - overlap, end + overlap] clipped to the
    // latent, the decoded overlap samples cropped from each side.
    std::vector<float> audio;  // interleaved [T, 2]
    audio.reserve((size_t) T_latent * HOP * 2);
    int n_tiles = (T_latent + chunk_size - 1) / chunk_size;
    fprintf(stderr, "[VAE] Decoding %d frames, %d tile(s)...\n", T_latent, n_tiles);
    for (int start = 0; start < T_latent; start += chunk_size) {
        int end = start + chunk_size < T_latent ? start + chunk_size : T_latent;
        int lo  = start - overlap > 0 ? start - overlap : 0;
        int hi  = end + overlap < T_latent ? end + overlap : T_latent;
        int T_t = hi - lo;

        // frame-major file slice -> channel-major [128, T_t] for decode
        std::vector<float> chan((size_t) LATENT_CH * T_t);
        for (int t = 0; t < T_t; t++) {
            for (int c = 0; c < LATENT_CH; c++) {
                chan[(size_t) c * T_t + t] = latent[(size_t) (lo + t) * LATENT_CH + c];
            }
        }

        std::vector<float> wav;
        if (!vae.decode(chan, T_t, wav)) {
            free(latent);
            vae.free();
            return 1;
        }
        int left  = (start - lo) * HOP;
        int right = (hi - end) * HOP;
        audio.insert(audio.end(), wav.begin() + (size_t) left * 2, wav.end() - (size_t) right * 2);
    }
    free(latent);
    vae.free();

    // interleaved [T, 2] -> planar [L: T][R: T] for the audio writers
    int                T_audio = (int) (audio.size() / 2);
    std::vector<float> planar((size_t) T_audio * 2);
    for (int t = 0; t < T_audio; t++) {
        planar[t]           = audio[(size_t) t * 2 + 0];
        planar[T_audio + t] = audio[(size_t) t * 2 + 1];
    }

    bool ok;
    if (is_mp3) {
        ok = audio_write_mp3(output_path, planar.data(), T_audio, 44100, 128);
    } else {
        ok = audio_write_wav(output_path, planar.data(), T_audio, 44100, wav_fmt);
    }
    if (!ok) {
        fprintf(stderr, "[VAE] FATAL: failed to write %s\n", output_path);
        return 1;
    }
    fprintf(stderr, "[VAE] Output: %s (%d samples, %.2fs @ 44.1kHz)\n", output_path, T_audio,
            (float) T_audio / 44100.0f);
    fprintf(stderr, "[VAE] Done.\n");
    return 0;
}
