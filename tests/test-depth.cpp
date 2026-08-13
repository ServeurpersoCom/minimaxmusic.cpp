// test-depth.cpp: RVQ depth decoder parity harness
//
// Runs one depth forward on a raw f32 step sequence and writes the
// normalized hiddens and the last-step head logits, for comparison against
// the torch reference dump.

#include "depth-decoder.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s depth.gguf seq.bin S hiddens_out.bin logits_out.bin\n", argv[0]);
        return 1;
    }

    int S = atoi(argv[3]);

    std::vector<float> seq((size_t) S * 4096);
    FILE *             f = fopen(argv[2], "rb");
    if (!f || fread(seq.data(), sizeof(float), seq.size(), f) != seq.size()) {
        fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    fclose(f);

    DepthDecoder depth;
    if (!depth.load(argv[1])) {
        return 1;
    }

    std::vector<float> hiddens((size_t) S * 4096), logits(1024);
    if (!depth.forward(seq.data(), S, hiddens.data(), logits.data())) {
        return 1;
    }

    FILE * out = fopen(argv[4], "wb");
    if (!out || fwrite(hiddens.data(), sizeof(float), hiddens.size(), out) != hiddens.size()) {
        fprintf(stderr, "cannot write %s\n", argv[4]);
        return 1;
    }
    fclose(out);
    out = fopen(argv[5], "wb");
    if (!out || fwrite(logits.data(), sizeof(float), logits.size(), out) != logits.size()) {
        fprintf(stderr, "cannot write %s\n", argv[5]);
        return 1;
    }
    fclose(out);

    depth.free();
    fprintf(stderr, "[Test-Depth] S=%d done\n", S);
    return 0;
}
