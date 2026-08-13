// hparams.h: MiniMax Music 3 hyperparameters (from HF diffusers layout configs)
//
// Pipeline: tokenizer -> global LM (8B) -> RVQ depth decoder (0.6B)
//        -> condition encoder -> flow matching DiT (2.4B) -> flow VAE decoder
//
// Frame rates: LM 25 Hz (24000/960), VAE latent 86.13 Hz (44100/512)
#pragma once

// Global LM: Qwen3 dense, predicts the semantic codebook frame by frame
// (language_model/config.json, architectures Qwen3ForCausalLM)
struct GlobalLMHParams {
    int   vocab_size        = 200000;
    int   hidden_size       = 4096;
    int   intermediate_size = 12288;
    int   n_layers          = 36;
    int   n_heads           = 32;
    int   n_kv_heads        = 8;
    int   head_dim          = 128;
    int   max_seq_len       = 10240;
    float rope_theta        = 1000000.0f;
    float rms_norm_eps      = 1e-6f;
    int   semantic_codebook = 16384;  // first RVQ codebook size
    int   frame_rate_hz     = 25;
};

// RVQ depth decoder: intra frame transformer over codebook positions,
// predicts the 7 acoustic codebooks from the global LM hidden state
// (rvq_depth_decoder/config.json)
struct DepthDecoderHParams {
    int audio_vocab_size  = 1024;  // per acoustic codebook
    int num_codebooks     = 8;     // 1 semantic + 7 acoustic
    int hidden_size       = 4096;
    int intermediate_size = 6144;
    int n_layers          = 4;
    int n_heads           = 16;
    int head_dim          = 256;
    int max_positions     = 16;  // learned positional embedding table
};

// Condition encoder: learned weighted mix of 8 hidden states, conv1d
// projection 4096 -> 2048 k=3, temporal resample 25 Hz -> 86.13 Hz
// (condition_encoder/config.json)
struct CondEncHParams {
    int condition_hidden_dim = 4096;
    int num_condition_layers = 8;
    int out_dim              = 2048;
    int input_sampling_rate  = 24000;
    int input_hop_length     = 960;
    int output_sampling_rate = 44100;
    int output_hop_length    = 512;
    int proj_kernel          = 3;
};

// Flow matching DiT: 36 self attention blocks, no cross attention, no adaLN,
// conditioning by channel concat through preprocess_conv, partial RoPE
// (transformer/config.json)
struct DiTHParams {
    int in_channels       = 128;  // flow VAE latent channels
    int condition_dim     = 2048;
    int n_layers          = 36;
    int n_heads           = 32;
    int head_dim          = 64;  // model dim 2048
    int ff_inner_dim      = 8192;
    int rotary_dim        = 32;
    int fourier_embed_dim = 256;
};

// Flow VAE decoder: DAC style upsampling stack, snake activations,
// weight normalized convs folded at conversion
// (vocoder/config.json)
// Block: snake -> conv_transpose(k = 2 * stride) -> 3 x res_unit
// ResUnit: skip = x -> snake -> conv(k=7) -> snake -> conv(k=1) -> + skip
// ConvT path: GEMM + col2im_1d, snake fused by graph pattern recognition
struct FlowVAEHParams {
    int latent_channels      = 128;
    int decoder_input_dim    = 1024;
    int decoder_hidden_dim   = 1536;
    int sampling_rate        = 44100;
    int n_blocks             = 4;
    int upsampling_ratios[4] = { 8, 8, 4, 2 };  // total hop 512
    int n_channels           = 2;               // stereo output
};
