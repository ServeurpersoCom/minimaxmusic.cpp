// neural-codec.cpp: neural audio codec (flow VAE encoder + decoder)
//
// encode: 44.1 kHz stereo audio (WAV or MP3) -> latent file (.vae, .nac8, or .nac4)
// decode: latent file -> 44.1 kHz stereo audio (WAV or MP3)
//
// Three latent formats, decode auto-detects:
//
//   .vae (default): flat [T, 128] f32 frame-major, no header.
//     T = file_size / 512. 86.13 Hz, ~353 kbit/s.
//
//   .nac8 (--q8): symmetric per-frame int8 quantization.
//     header: "NAC8" magic (4B) + uint32 T_latent (4B)
//     frame:  f16 scale (2B) + int8[128] (128B) = 130B
//     86.13 Hz, ~89.6 kbit/s.
//
//   .nac4 (--q4): symmetric per-frame 4-bit quantization.
//     header: "NAC4" magic (4B) + uint32 T_latent (4B)
//     frame:  f16 scale (2B) + nibbles[64] (64B) = 66B
//     86.13 Hz, ~45.5 kbit/s.
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
//   neural-codec --vae model.gguf --encode --q8 -i song.wav -o song.nac8
//   neural-codec --vae model.gguf --encode --q4 -i song.wav -o song.nac4
//   neural-codec --vae model.gguf --decode -i song.nac4 -o song.wav

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

// Q8 format constants
static const char NAC8_MAGIC[4] = { 'N', 'A', 'C', '8' };
static const int  NAC8_HEADER   = 8;    // 4B magic + 4B T_latent
static const int  NAC8_FRAME    = 130;  // 2B f16 scale + 128B int8

// Q4 format constants
static const char NAC4_MAGIC[4] = { 'N', 'A', 'C', '4' };
static const int  NAC4_HEADER   = 8;   // 4B magic + 4B T_latent
static const int  NAC4_FRAME    = 66;  // 2B f16 scale + 64B packed nibbles

static void log_latent(const char * verb, const char * path, const char * kind, int T_latent, size_t bytes) {
    float duration = (float) T_latent * HOP / 44100.0f;
    fprintf(stderr, "[Latent] %s %s: %s, %d frames (%.2fs, %.1f KB, %.1f kbit/s)\n", verb, path, kind, T_latent,
            duration, (float) bytes / 1024.0f, (float) bytes * 8.0f / (duration * 1000.0f));
}

// Write f32 raw latent (no header)
static bool write_latent_f32(const char * path, const float * data, int T_latent) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    size_t bytes = (size_t) T_latent * LATENT_CH * sizeof(float);
    fwrite(data, 1, bytes, f);
    fclose(f);
    log_latent("Wrote", path, "f32", T_latent, bytes);
    return true;
}

// Write Q8 quantized latent: symmetric per-frame int8, range [-127, 127]
static bool write_latent_q8(const char * path, const float * data, int T_latent) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    fwrite(NAC8_MAGIC, 1, 4, f);
    uint32_t t = (uint32_t) T_latent;
    fwrite(&t, 4, 1, f);

    for (int i = 0; i < T_latent; i++) {
        const float * frame = data + (size_t) i * LATENT_CH;
        float         amax  = 0.0f;
        for (int j = 0; j < LATENT_CH; j++) {
            float a = fabsf(frame[j]);
            if (a > amax) {
                amax = a;
            }
        }
        float       scale     = amax / 127.0f;
        ggml_fp16_t scale_f16 = ggml_fp32_to_fp16(scale);
        fwrite(&scale_f16, 2, 1, f);

        int8_t q[LATENT_CH];
        float  inv = (scale > 0.0f) ? 127.0f / amax : 0.0f;
        for (int j = 0; j < LATENT_CH; j++) {
            int v = (int) roundf(frame[j] * inv);
            q[j]  = (int8_t) (v < -127 ? -127 : (v > 127 ? 127 : v));
        }
        fwrite(q, 1, LATENT_CH, f);
    }
    fclose(f);
    log_latent("Wrote", path, "Q8", T_latent, NAC8_HEADER + (size_t) T_latent * NAC8_FRAME);
    return true;
}

// Write Q4 quantized latent: symmetric per-frame 4-bit, range [-7, 7],
// two signed nibbles per byte: (low & 0x0F) | (high << 4)
static bool write_latent_q4(const char * path, const float * data, int T_latent) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    fwrite(NAC4_MAGIC, 1, 4, f);
    uint32_t t = (uint32_t) T_latent;
    fwrite(&t, 4, 1, f);

    for (int i = 0; i < T_latent; i++) {
        const float * frame = data + (size_t) i * LATENT_CH;
        float         amax  = 0.0f;
        for (int j = 0; j < LATENT_CH; j++) {
            float a = fabsf(frame[j]);
            if (a > amax) {
                amax = a;
            }
        }
        float       scale     = amax / 7.0f;
        ggml_fp16_t scale_f16 = ggml_fp32_to_fp16(scale);
        fwrite(&scale_f16, 2, 1, f);

        float   inv = (scale > 0.0f) ? 7.0f / amax : 0.0f;
        uint8_t packed[LATENT_CH / 2];
        for (int j = 0; j < LATENT_CH / 2; j++) {
            int lo    = (int) roundf(frame[j * 2 + 0] * inv);
            int hi    = (int) roundf(frame[j * 2 + 1] * inv);
            lo        = lo < -7 ? -7 : (lo > 7 ? 7 : lo);
            hi        = hi < -7 ? -7 : (hi > 7 ? 7 : hi);
            packed[j] = (uint8_t) ((lo & 0x0F) | (hi << 4));
        }
        fwrite(packed, 1, LATENT_CH / 2, f);
    }
    fclose(f);
    log_latent("Wrote", path, "Q4", T_latent, NAC4_HEADER + (size_t) T_latent * NAC4_FRAME);
    return true;
}

static void print_usage(const char * prog) {
    fprintf(stderr, "minimaxmusic.cpp %s\n\n", MM3_VERSION);
    fprintf(stderr,
            "Usage: %s --vae <gguf> --encode|--decode -i <input> [-o <output>] [--q8|--q4]\n"
            "\n"
            "Required:\n"
            "  --vae <path>            VAE GGUF file\n"
            "  --encode | --decode     Encode audio to latent, or decode latent to audio\n"
            "  -i <path>               Input (WAV/MP3 for encode, latent for decode)\n"
            "\n"
            "Output:\n"
            "  -o <path>               Output file (auto-named if omitted)\n"
            "  --q8                    Quantize latent to int8 (~89.6 kbit/s)\n"
            "  --q4                    Quantize latent to int4 (~45.5 kbit/s)\n"
            "  --format <fmt>          mp3, wav16, wav24, wav32 (default: wav16)\n"
            "\n"
            "Output naming: song.wav -> song.vae (f32) or song.nac8 (Q8) or song.nac4 (Q4)\n"
            "               song.vae -> song.wav\n"
            "\n"
            "Memory control:\n"
            "  --vae-chunk <N>         Latent frames per tile (default: 689)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: 86)\n"
            "\n"
            "Latent formats (decode auto-detects):\n"
            "  .vae:  flat [T, 128] f32, no header. ~353 kbit/s.\n"
            "  .nac8: header + per-frame Q8. ~89.6 kbit/s.\n"
            "  .nac4: header + per-frame Q4. ~45.5 kbit/s.\n",
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

// Read a latent file, auto-detecting the format from the magic
// (NAC8 -> Q8, NAC4 -> Q4, else raw f32).
// Returns frame-major [T, 128] f32 (dequantized if quantized). Caller frees.
static float * read_latent(const char * path, int * T_latent) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[Latent] Cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char magic[4] = {};
    if (fsize >= 8) {
        if (fread(magic, 1, 4, f) != 4) {
            fclose(f);
            return NULL;
        }
    }

    if (memcmp(magic, NAC8_MAGIC, 4) == 0) {
        uint32_t t;
        if (fread(&t, 4, 1, f) != 1) {
            fclose(f);
            return NULL;
        }
        long expected = NAC8_HEADER + (long) t * NAC8_FRAME;
        if (fsize != expected) {
            fprintf(stderr, "[Latent] Q8 size mismatch: expected %ld, got %ld\n", expected, fsize);
            fclose(f);
            return NULL;
        }
        float * data = (float *) malloc((size_t) t * LATENT_CH * sizeof(float));
        if (!data) {
            fprintf(stderr, "[Latent] OOM allocating Q8 decode buffer for %u frames\n", t);
            fclose(f);
            return NULL;
        }
        for (int i = 0; i < (int) t; i++) {
            ggml_fp16_t scale_f16;
            int8_t      q[LATENT_CH];
            if (fread(&scale_f16, 2, 1, f) != 1 || fread(q, 1, LATENT_CH, f) != LATENT_CH) {
                fprintf(stderr, "[Latent] Short read on %s\n", path);
                free(data);
                fclose(f);
                return NULL;
            }
            float   scale = ggml_fp16_to_fp32(scale_f16);
            float * frame = data + (size_t) i * LATENT_CH;
            for (int j = 0; j < LATENT_CH; j++) {
                frame[j] = (float) q[j] * scale;
            }
        }
        fclose(f);
        *T_latent = (int) t;
        log_latent("Read", path, "Q8", (int) t, (size_t) fsize);
        return data;
    }

    if (memcmp(magic, NAC4_MAGIC, 4) == 0) {
        uint32_t t;
        if (fread(&t, 4, 1, f) != 1) {
            fclose(f);
            return NULL;
        }
        long expected = NAC4_HEADER + (long) t * NAC4_FRAME;
        if (fsize != expected) {
            fprintf(stderr, "[Latent] Q4 size mismatch: expected %ld, got %ld\n", expected, fsize);
            fclose(f);
            return NULL;
        }
        float * data = (float *) malloc((size_t) t * LATENT_CH * sizeof(float));
        if (!data) {
            fprintf(stderr, "[Latent] OOM allocating Q4 decode buffer for %u frames\n", t);
            fclose(f);
            return NULL;
        }
        for (int i = 0; i < (int) t; i++) {
            ggml_fp16_t scale_f16;
            uint8_t     packed[LATENT_CH / 2];
            if (fread(&scale_f16, 2, 1, f) != 1 || fread(packed, 1, LATENT_CH / 2, f) != LATENT_CH / 2) {
                fprintf(stderr, "[Latent] Short read on %s\n", path);
                free(data);
                fclose(f);
                return NULL;
            }
            float   scale = ggml_fp16_to_fp32(scale_f16);
            float * frame = data + (size_t) i * LATENT_CH;
            for (int j = 0; j < LATENT_CH / 2; j++) {
                int lo = (int) (packed[j] & 0x0F);
                int hi = (int) (packed[j] >> 4);
                if (lo >= 8) {
                    lo -= 16;
                }
                if (hi >= 8) {
                    hi -= 16;
                }
                frame[j * 2 + 0] = (float) lo * scale;
                frame[j * 2 + 1] = (float) hi * scale;
            }
        }
        fclose(f);
        *T_latent = (int) t;
        log_latent("Read", path, "Q4", (int) t, (size_t) fsize);
        return data;
    }

    // Raw f32 (no header), rewind past the magic probe
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
    log_latent("Read", path, "f32", (int) t, (size_t) fsize);
    return data;
}

int main(int argc, char ** argv) {
    const char * vae_path    = NULL;
    const char * input_path  = NULL;
    const char * output_path = NULL;
    const char * format      = "wav16";
    int          mode        = -1;  // 0 = encode, 1 = decode
    int          quant       = 0;   // 0 = f32, 8 = q8, 4 = q4
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
        } else if (strcmp(argv[i], "--q8") == 0) {
            quant = 8;
        } else if (strcmp(argv[i], "--q4") == 0) {
            quant = 4;
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

    // Auto output names. f32 dumps land as .vae since they are the raw VAE
    // encoder output with no codec applied; nac8/nac4 keep their codec name.
    std::string out_str;
    if (!output_path) {
        if (mode == 0) {
            const char * ext = ".vae";
            if (quant == 8) {
                ext = ".nac8";
            }
            if (quant == 4) {
                ext = ".nac4";
            }
            out_str = auto_output(input_path, ext);
        } else {
            out_str = auto_output(input_path, is_mp3 ? ".mp3" : ".wav");
        }
        output_path = out_str.c_str();
    }

    const char * quant_str = "";
    if (mode == 0 && quant == 8) {
        quant_str = " (Q8)";
    }
    if (mode == 0 && quant == 4) {
        quant_str = " (Q4)";
    }
    fprintf(stderr, "\n[VAE] Mode: %s%s\n", mode == 0 ? "encode" : "decode", quant_str);
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

        bool ok;
        if (quant == 8) {
            ok = write_latent_q8(output_path, out.data(), T_latent);
        } else if (quant == 4) {
            ok = write_latent_q4(output_path, out.data(), T_latent);
        } else {
            ok = write_latent_f32(output_path, out.data(), T_latent);
        }
        if (!ok) {
            fprintf(stderr, "[VAE] FATAL: failed to write %s\n", output_path);
            return 1;
        }
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
