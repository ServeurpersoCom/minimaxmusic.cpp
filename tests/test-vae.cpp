// test-vae.cpp: flow VAE decoder parity harness
//
// Decodes a raw f32 latent [128, T] channel-major through the GGML decoder
// and writes the interleaved stereo waveform as raw f32, for comparison
// against the torch reference dump.

#include "vae.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s vae.gguf latent.bin T_latent audio_out.bin\n", argv[0]);
        return 1;
    }

    const char * gguf_path   = argv[1];
    const char * latent_path = argv[2];
    int          T           = atoi(argv[3]);
    const char * out_path    = argv[4];

    std::vector<float> latent((size_t) 128 * T);
    FILE *             f = fopen(latent_path, "rb");
    if (!f || fread(latent.data(), sizeof(float), latent.size(), f) != latent.size()) {
        fprintf(stderr, "cannot read latent %s\n", latent_path);
        return 1;
    }
    fclose(f);

    FlowVAE vae;
    if (!vae.load(gguf_path)) {
        return 1;
    }

    std::vector<float> audio;
    if (!vae.decode(latent, T, audio)) {
        return 1;
    }

    FILE * out = fopen(out_path, "wb");
    if (!out || fwrite(audio.data(), sizeof(float), audio.size(), out) != audio.size()) {
        fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }
    fclose(out);

    vae.free();
    fprintf(stderr, "[Test-VAE] Wrote %zu samples to %s\n", audio.size(), out_path);
    return 0;
}
