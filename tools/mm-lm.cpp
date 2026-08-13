// mm-lm.cpp: hybrid LM inference, prompt to RVQ codes
//
// Runs the global LM (semantic codebook, 25 Hz) and the RVQ depth decoder
// (7 acoustic codebooks per frame), dumps the code stream and the fused
// hidden states consumed by the synthesis stage.
//
// Skeleton: prints the component summary and exits.

#include "depth-decoder.h"
#include "hparams.h"
#include "prompt.h"
#include "qwen3-lm.h"
#include "version.h"

#include <cstdio>

static void print_usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s --lm LM.gguf --depth DEPTH.gguf --caption TEXT --lyrics TEXT [--max-frames N] [--seed N]\n",
            argv0);
}

int main(int argc, char ** argv) {
    fprintf(stderr, "mm-lm %s\n", MM3_VERSION);

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    GlobalLMHParams     lm;
    DepthDecoderHParams depth;

    fprintf(stderr, "global LM: %d layers, hidden %d, GQA %d/%d, vocab %d, ctx %d, %d Hz\n", lm.n_layers,
            lm.hidden_size, lm.n_heads, lm.n_kv_heads, lm.vocab_size, lm.max_seq_len, lm.frame_rate_hz);
    fprintf(stderr, "depth decoder: %d layers, hidden %d, %d codebooks x %d\n", depth.n_layers, depth.hidden_size,
            depth.num_codebooks, depth.audio_vocab_size);
    fprintf(stderr, "LM inference is not implemented in this skeleton\n");
    return 1;
}
