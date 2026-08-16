#!/usr/bin/env node
// Latent extraction for the RVQ encoder training: runs neural-codec
// --encode over every wav of a dataset and lands a .vae latent file
// (flat [T, 128] f32, frame = 512 bytes, 86.1328 Hz) next to each
// track. Idempotent: a wav whose .vae already exists is skipped.
//
// Usage: ./encode-corpus.mjs <dataset>

import { execFileSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";

const ROOT         = join(dirname(new URL(import.meta.url).pathname), "..");
const DATASETS_DIR = join(ROOT, "training/datasets");
const NEURAL_CODEC = join(ROOT, "build/neural-codec");
const VAE_GGUF     = join(ROOT, "models/MiniMax-Music3-vocoder-F32.gguf");
const WAV_EXT      = ".wav";
const VAE_EXT      = ".vae";

const args = process.argv.slice(2);
if (args.length < 1) {
    console.error("usage: ./encode-corpus.mjs <dataset>");
    process.exit(1);
}
const dir  = join(DATASETS_DIR, args[0]);
const wavs = readdirSync(dir).filter((f) => f.endsWith(WAV_EXT)).sort();
console.log(`[Encode] ${wavs.length} tracks in ${dir}`);

let done = 0;
for (const wav of wavs) {
    const vae = wav.slice(0, -WAV_EXT.length) + VAE_EXT;
    if (existsSync(join(dir, vae))) {
        console.log(`[Skip] ${vae}: exists`);
        continue;
    }
    execFileSync(NEURAL_CODEC, ["--vae", VAE_GGUF, "--encode", "-i", join(dir, wav), "-o", join(dir, vae)], {
        stdio: ["ignore", "inherit", "inherit"],
    });
    done++;
}
console.log(`[Done] ${done} encoded, ${wavs.length - done} skipped`);
