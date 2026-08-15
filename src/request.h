#pragma once
// request.h: MiniMax Music 3 generation request (JSON serialization)
//
// Pure data container + JSON read/write. Zero business logic.
// Only fields the pipeline consumes: the recipe constants (window sizes,
// crops, sigma schedule shape) live in the pipeline, not here.

#include <cstdint>
#include <string>

struct MM3Request {
    // text content
    std::string caption;  // ""
    std::string lyrics;   // ""

    // generation
    float   duration;  // 60.0, seconds, capped by the 9000 frame LM budget
    int     steps;     // 30, Euler steps per DiT window
    int64_t seed;      // -1 = random. DiT Philox noise consumes the low
                       // 32 bits. Stored in int64_t to land positive
                       // after rd(), no -1 collision, no mask needed.
    int64_t lm_seed;   // -1 = random. AR sampling mt19937 consumes the low
                       // 32 bits. Same int64_t storage trick as seed.

    // batching: number of songs generated from this prompt. Each song
    // samples with its own stream (lm_seed + index), acestep convention
    // of consecutive internal seeds.
    int lm_batch_size;  // 1

    // number of flow matching variations per song, consecutive noise
    // seeds (seed + index) on the same condition track. Output order is
    // song-major: song * synth_batch_size + variation.
    int synth_batch_size;  // 1

    // codes (Python-compatible string: "8113,404,...", 8 per frame:
    // semantic then the 7 acoustic codebooks). Non-empty replaces the
    // autoregressive stage: the hiddens are re-derived teacher-forced
    // and the song renders deterministically from these codes.
    std::string audio_codes;  // ""

    // guidance
    float lm_cfg;    // 1.5, CFG scale on LM and depth decoder logits
    int   lm_top_k;  // 50, applied to the conditional branch ranking
    float dit_cfg;   // 1.7, CFG scale on the DiT velocity field

    // audio output: peak clip via percentile normalization.
    // 0 = peak normalization (100.0000th percentile, no clipping).
    // Target percentile is 1.0 - peak_clip/1000000.0. WAV32 skips it.
    int peak_clip;  // 10

    // audio output format: "mp3", "wav16", "wav24", "wav32"
    std::string output_format;

    // MP3 encoder bitrate in kbps, used when output_format is "mp3".
    // WAV outputs ignore this field.
    int mp3_bitrate;  // 128

    // model routing: GGUF filename per component, resolved against the
    // server registry. Empty keeps the previously requested model, or
    // falls to the first registry entry.
    std::string lm_model;
    std::string depth_model;
    std::string cond_model;
    std::string dit_model;
    std::string vae_model;
};

// Initialize all fields to defaults (matches the diffusers pipeline defaults)
void request_init(MM3Request * r);

// Parse JSON string into struct. Missing fields keep their defaults.
// Returns false on malformed JSON.
bool request_parse_json(MM3Request * r, const char * json);

// Serialize struct to JSON string.
// sparse=true: omit fields at their default value (for cards and exports).
// sparse=false: serialize all fields (for /props documentation).
std::string request_to_json(const MM3Request * r, bool sparse = true);

// Resolve seed: if negative, replace with a hardware random value.
void request_resolve_seed(MM3Request * r);

// Resolve LM seed: if negative, replace with a hardware random value.
void request_resolve_lm_seed(MM3Request * r);

// Sparse replay request for one rendered track: the base request with
// audio_codes set, the seeds offset to the ones the track consumed
// (lm_seed by the song index, the DiT seed by the variation index), and
// the batch sizes reset. Running it reproduces that track
// deterministically.
std::string request_replay_json(const MM3Request & base, const std::string & codes, int song, int variation);
