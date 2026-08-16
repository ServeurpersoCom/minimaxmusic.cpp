#!/usr/bin/env node
// Corpus generator for the RVQ encoder training: runs mm-synth over the
// official example prompts and lands aligned (wav, replay json) pairs
// under datasets/. Every track carries its audio_codes targets in the
// replay json written by mm-synth next to the audio. Idempotent: a round
// whose expected outputs all exist is skipped, so an interrupted run
// resumes where it stopped.
//
// Usage: ./gen-corpus.mjs <dataset> [rounds_per_prompt]
//   dataset           subdirectory of datasets/ (e.g. pilot)
//   rounds_per_prompt LM batched mm-synth calls per prompt (default 2)

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";

const ROOT          = join(dirname(new URL(import.meta.url).pathname), "..");
const EXAMPLE_DIR   = join(ROOT, "tools/webui/example");
const DATASETS_DIR  = join(ROOT, "training/datasets");
const MM_SYNTH      = join(ROOT, "build/mm-synth");
const MODELS_DIR    = join(ROOT, "models");
const REQUEST_TMP   = "/tmp/gen-corpus-request.json";

const LM_BATCH          = 5;
const LM_MODEL          = "MiniMax-Music3-language_model-Q8_0.gguf";
const DEPTH_MODEL       = "MiniMax-Music3-rvq_depth_decoder-Q8_0.gguf";
const DIT_MODEL         = "MiniMax-Music3-transformer-F32.gguf";
const DEFAULT_ROUNDS    = 2;
const DURATIONS_S       = [30, 45, 60, 75, 90];
const LM_SEED_BASE      = 100000;
const OUTPUT_FORMAT     = "wav16";
const JSON_EXT          = ".json";
const WAV_EXT           = ".wav";
const VARIATION_SUFFIX  = "0"; // synth_batch_size 1: mm-synth suffixes song index then variation 0

const args = process.argv.slice(2);
if (args.length < 1) {
    console.error("usage: ./gen-corpus.mjs <dataset> [rounds_per_prompt]");
    process.exit(1);
}
const outDir = join(DATASETS_DIR, args[0]);
const rounds = args.length > 1 ? parseInt(args[1], 10) : DEFAULT_ROUNDS;
mkdirSync(outDir, { recursive: true });

const prompts = readdirSync(EXAMPLE_DIR).filter((f) => f.endsWith(JSON_EXT)).sort();
console.log(`[Corpus] ${prompts.length} prompts x ${rounds} rounds x ${LM_BATCH} songs -> ${outDir}`);

let generatedS = 0;
let wallStart  = Date.now();
for (let r = 0; r < rounds; r++) {
    for (let p = 0; p < prompts.length; p++) {
        const globalRound = r * prompts.length + p;
        const durationS   = DURATIONS_S[globalRound % DURATIONS_S.length];
        const base        = `p${String(p).padStart(2, "0")}-r${String(r).padStart(2, "0")}`;
        const expected    = [];
        for (let s = 0; s < LM_BATCH; s++) {
            expected.push(join(outDir, `${base}${s}${VARIATION_SUFFIX}${WAV_EXT}`));
            expected.push(join(outDir, `${base}${s}${VARIATION_SUFFIX}${JSON_EXT}`));
        }
        if (expected.every(existsSync)) {
            console.log(`[Skip] ${base}: exists`);
            generatedS += durationS * LM_BATCH;
            continue;
        }

        const example = JSON.parse(readFileSync(join(EXAMPLE_DIR, prompts[p]), "utf8"));
        const request = {
            caption:       example.caption,
            lyrics:        example.lyrics,
            duration:      durationS,
            lm_seed:       LM_SEED_BASE + globalRound * LM_BATCH,
            lm_batch_size: LM_BATCH,
            lm_model:      LM_MODEL,
            depth_model:   DEPTH_MODEL,
            dit_model:     DIT_MODEL,
            output_format: OUTPUT_FORMAT,
        };
        writeFileSync(REQUEST_TMP, JSON.stringify(request));

        console.log(`[Gen] ${base}: ${basename(prompts[p])} ${durationS}s x ${LM_BATCH} (lm_seed ${request.lm_seed})`);
        execFileSync(MM_SYNTH, ["--models", MODELS_DIR, "--request", REQUEST_TMP, "--out", join(outDir, base + WAV_EXT)], {
            stdio: ["ignore", "inherit", "inherit"],
        });

        generatedS += durationS * LM_BATCH;
        const wallS = (Date.now() - wallStart) / 1000;
        console.log(`[Corpus] ${(generatedS / 3600).toFixed(2)} h audio, wall ${(wallS / 3600).toFixed(2)} h`);
    }
}
console.log(`[Done] ${(generatedS / 3600).toFixed(2)} h audio in ${outDir}`);
