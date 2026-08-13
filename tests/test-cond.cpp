// test-cond.cpp: condition encoder parity harness
//
// Encodes a raw f32 hidden state block [T, 8, 4096] through the GGML
// condition encoder and writes the conditioning track [n_latents, 2048]
// as raw f32, for comparison against the torch reference dump.

#include "cond-enc.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s cond.gguf hidden.bin T_frames cond_out.bin\n", argv[0]);
        return 1;
    }

    const char * gguf_path   = argv[1];
    const char * hidden_path = argv[2];
    int          T           = atoi(argv[3]);
    const char * out_path    = argv[4];

    std::vector<float> hidden((size_t) T * 8 * 4096);
    FILE *             f = fopen(hidden_path, "rb");
    if (!f || fread(hidden.data(), sizeof(float), hidden.size(), f) != hidden.size()) {
        fprintf(stderr, "cannot read hidden %s\n", hidden_path);
        return 1;
    }
    fclose(f);

    CondEnc enc;
    if (!enc.load(gguf_path)) {
        return 1;
    }

    std::vector<float> cond;
    int                n_latents = 0;
    if (!enc.encode(hidden, T, cond, n_latents)) {
        return 1;
    }

    FILE * out = fopen(out_path, "wb");
    if (!out || fwrite(cond.data(), sizeof(float), cond.size(), out) != cond.size()) {
        fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }
    fclose(out);

    enc.free();
    fprintf(stderr, "[Test-Cond] T=%d -> n_latents=%d, wrote %zu floats to %s\n", T, n_latents, cond.size(), out_path);
    return 0;
}
