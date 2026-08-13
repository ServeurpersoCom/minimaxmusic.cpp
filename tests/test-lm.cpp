// test-lm.cpp: global LM parity harness
//
// Prefills a fixed token id sequence, then runs one decode step, dumping
// the logits and the last hidden state of both forwards for comparison
// against the transformers reference.

#include "qwen3-lm.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s lm.gguf out_prefix id0 [id1 ...]\n", argv[0]);
        return 1;
    }

    std::vector<int> ids;
    for (int i = 3; i < argc; i++) {
        ids.push_back(atoi(argv[i]));
    }

    Qwen3LM lm;
    if (!qw3lm_load(&lm, argv[1], 0, 1)) {
        return 1;
    }

    int                V = lm.cfg.vocab_size;
    int                H = lm.cfg.hidden_size;
    std::vector<float> logits(V), hidden(H);

    // Prefill all ids but the last, then decode the last id alone
    qw3lm_forward(&lm, ids.data(), (int) ids.size() - 1, 0, logits.data(), nullptr, hidden.data());
    std::string p = std::string(argv[2]) + "_prefill";
    FILE *      f = fopen((p + "_logits.bin").c_str(), "wb");
    fwrite(logits.data(), sizeof(float), V, f);
    fclose(f);
    f = fopen((p + "_hidden.bin").c_str(), "wb");
    fwrite(hidden.data(), sizeof(float), H, f);
    fclose(f);

    int last = ids.back();
    qw3lm_forward(&lm, &last, 1, 0, logits.data(), nullptr, hidden.data());
    p = std::string(argv[2]) + "_decode";
    f = fopen((p + "_logits.bin").c_str(), "wb");
    fwrite(logits.data(), sizeof(float), V, f);
    fclose(f);
    f = fopen((p + "_hidden.bin").c_str(), "wb");
    fwrite(hidden.data(), sizeof(float), H, f);
    fclose(f);

    fprintf(stderr, "[Test-LM] Prefill %d tokens + decode 1, V=%d H=%d\n", (int) ids.size() - 1, V, H);
    return 0;
}
