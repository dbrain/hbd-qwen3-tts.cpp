#pragma once

// Shared config for the Breeze-TTS-2 port. Every field is read from the GGUF's
// `breeze-tts.*` metadata written by scripts/convert_breeze_to_gguf.py -- the
// converter takes them from the real config.json, so nothing here is guessed
// and a checkpoint with different hyperparameters just works.
//
// Four models live in one file, under four tensor prefixes:
//   text_enc.*   T5Gemma2 encoder, 26 x 1152, GQA 4/1, head_dim 256
//   backbone.*   Qwen3, 28 x 2048, GQA 16/8, QK-norm
//   depth.*      depth decoder, 12 x 1024, GQA 8/2, NO QK-norm, 16 codebooks
//   codec.*      Mimi, 12.5 Hz / 24 kHz

#include "gguf.h"

#include <string>
#include <vector>

namespace breeze {

// RoPE scaling flavours we actually see in this checkpoint.
enum class rope_kind { none, linear, llama3 };

struct text_enc_config {
    int   n_layers   = 26;
    int   n_embd     = 1152;
    int   n_ff       = 6912;
    int   n_head     = 4;
    int   n_kv_head  = 1;
    int   head_dim   = 256;    // NOTE head_dim*n_head = 1024 != n_embd 1152
    int   vocab      = 262158;
    int   sliding_window = 512;
    int   qk_scalar  = 256;    // attention scale is qk_scalar^-0.5, NOT head_dim
    float rms_eps    = 1e-6f;
    int   eoi_token  = 256000;
    std::vector<int> layer_is_full;   // 1 = full_attention, 0 = sliding
    float rope_theta_sliding = 10000.0f;
    float rope_theta_full    = 1000000.0f;
    rope_kind rope_full_kind = rope_kind::linear;
    float rope_factor_full   = 8.0f;
    float embed_scale        = 33.941124f;  // sqrt(n_embd), computed at load
};

struct backbone_config {
    int   n_layers  = 28;
    int   n_embd    = 2048;
    int   n_ff      = 6144;
    int   n_head    = 16;
    int   n_kv_head = 8;
    int   head_dim  = 128;
    float rope_theta = 1000000.0f;
    float rms_eps   = 1e-6f;
    int   max_pos   = 40960;
    int   audio_vocab   = 2051;   // per-codebook code space
    int   lm_head_size  = 2052;   // audio_vocab + 1 (the extra class is EOS)
    int   n_codebooks   = 16;
};

struct depth_config {
    int   n_layers  = 12;
    int   n_embd    = 1024;
    int   n_ff      = 8192;
    int   n_head    = 8;
    int   n_kv_head = 2;
    int   head_dim  = 128;
    float rope_theta = 500000.0f;
    float rms_eps   = 1e-5f;
    int   n_codebooks = 16;
    int   vocab     = 2051;
    int   audio_embd_size = 2048;   // depth.token_embd is [2048, 16*2051]
    rope_kind rope = rope_kind::llama3;
    float rope_factor    = 32.0f;
    float rope_low_freq  = 0.001953125f;
    float rope_high_freq = 0.0078125f;
    int   rope_orig_ctx  = 16;
};

struct codec_config {
    int   sample_rate   = 24000;
    int   hidden        = 512;
    int   codebook_size = 2048;
    int   codebook_dim  = 256;
    int   n_quantizers  = 16;      // 1 semantic + 15 acoustic
    float frame_rate    = 12.5f;
    std::vector<int> upsample_rates{8, 6, 5, 4};
    int   n_tfm_layers  = 8;
    int   n_heads       = 8;
    int   head_dim      = 64;
    int   ffn_dim       = 2048;
    int   sliding_window = 250;
    float norm_eps      = 1e-5f;
    float rope_theta    = 10000.0f;
    int   num_filters   = 64;
    int   kernel_size   = 7;
    int   last_kernel   = 3;
    int   res_kernel    = 3;
    int   dilation_growth = 2;
    int   compress      = 2;
    int   n_res_layers  = 1;
    int   upsample_groups = 512;
    bool  causal        = true;
    // samples of PCM per codec frame: 2 (the x2 upsample) * prod(upsample_rates)
    int samples_per_frame() const {
        int p = 2;
        for (int r : upsample_rates) p *= r;
        return p;
    }
};

// Token ids that the prompt assembler needs. All from config.json.
struct special_tokens {
    int audio        = 262144;   // <|AUDIO|>   -- one per reference codec frame
    int audio_eos    = 262145;   // <|audio_eos|>
    int ins_bos      = 262156;   // <ins_bos>
    int ins_eos      = 262157;   // <ins_eos>
    int speaker_base = 262146;   // [S0]..[S9]
    int n_speakers   = 10;
    int bos          = 2;
    int codebook_eos = 0;
    int codebook_pad = 2050;
    int text_vocab   = 262158;
};

struct breeze_config {
    text_enc_config  te;
    backbone_config  bb;
    depth_config     dd;
    codec_config     cc;
    special_tokens   sp;
};

// Fill `out` from an open gguf context. Returns false (with `err` set) only if
// the file is not a breeze-tts GGUF; individual keys fall back to the struct
// defaults so an older converter output still loads.
bool load_breeze_config(struct gguf_context * ctx, breeze_config & out, std::string & err);

// Build the ggml `freq_factors` vector implementing HF's llama3 RoPE scaling:
// freq_factors[i] = inv_freq_default[i] / inv_freq_llama3[i], which is exactly
// what ggml_rope_ext divides theta by. Returns an empty vector for rope_kind::none.
std::vector<float> build_llama3_freq_factors(int head_dim, float theta,
                                             float factor, float low_freq_factor,
                                             float high_freq_factor, int orig_ctx);

} // namespace breeze
