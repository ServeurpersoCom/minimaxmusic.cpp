// test-dit.cpp: flow matching DiT parity harness
//
// Runs one denoising evaluation on raw f32 time-major inputs and writes the
// predicted velocity, for comparison against the torch reference dump.

#include "dit.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s dit.gguf xt.bin cond.bin T t out.bin\n", argv[0]);
        return 1;
    }

    int   T = atoi(argv[4]);
    float t = (float) atof(argv[5]);

    std::vector<float> xt((size_t) T * 128), cond((size_t) T * 2048);
    FILE *             f = fopen(argv[2], "rb");
    if (!f || fread(xt.data(), sizeof(float), xt.size(), f) != xt.size()) {
        fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    fclose(f);
    f = fopen(argv[3], "rb");
    if (!f || fread(cond.data(), sizeof(float), cond.size(), f) != cond.size()) {
        fprintf(stderr, "cannot read %s\n", argv[3]);
        return 1;
    }
    fclose(f);

    DiT dit;
    if (!dit.load(argv[1])) {
        return 1;
    }

    std::vector<float> vel((size_t) T * 128);
    if (!dit.forward(xt.data(), cond.data(), T, 1, t, vel.data())) {
        return 1;
    }

    FILE * out = fopen(argv[6], "wb");
    if (!out || fwrite(vel.data(), sizeof(float), vel.size(), out) != vel.size()) {
        fprintf(stderr, "cannot write %s\n", argv[6]);
        return 1;
    }
    fclose(out);

    dit.free();
    fprintf(stderr, "[Test-DiT] T=%d t=%.3f done\n", T, t);
    return 0;
}
