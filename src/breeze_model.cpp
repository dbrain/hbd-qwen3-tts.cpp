#include "breeze_model.h"

#include <cmath>
#include <cstring>
#include <cstdio>

namespace breeze {

namespace {

const char * ARCH = "breeze-tts";

std::string k(const char * suffix) { return std::string(ARCH) + "." + suffix; }

int32_t gu32(gguf_context * c, const char * key, int32_t def) {
    const int64_t i = gguf_find_key(c, k(key).c_str());
    if (i < 0) return def;
    switch (gguf_get_kv_type(c, i)) {
        case GGUF_TYPE_UINT32:  return (int32_t) gguf_get_val_u32(c, i);
        case GGUF_TYPE_INT32:   return           gguf_get_val_i32(c, i);
        case GGUF_TYPE_UINT64:  return (int32_t) gguf_get_val_u64(c, i);
        case GGUF_TYPE_INT64:   return (int32_t) gguf_get_val_i64(c, i);
        default: return def;
    }
}

float gf32(gguf_context * c, const char * key, float def) {
    const int64_t i = gguf_find_key(c, k(key).c_str());
    if (i < 0) return def;
    if (gguf_get_kv_type(c, i) == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(c, i);
    if (gguf_get_kv_type(c, i) == GGUF_TYPE_FLOAT64) return (float) gguf_get_val_f64(c, i);
    return def;
}

std::string gstr(gguf_context * c, const char * key, const char * def) {
    const int64_t i = gguf_find_key(c, k(key).c_str());
    if (i < 0 || gguf_get_kv_type(c, i) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(c, i);
}

std::vector<int> gi32arr(gguf_context * c, const char * key) {
    std::vector<int> out;
    const int64_t i = gguf_find_key(c, k(key).c_str());
    if (i < 0 || gguf_get_kv_type(c, i) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(c, i);
    const enum gguf_type et = gguf_get_arr_type(c, i);
    const void * d = gguf_get_arr_data(c, i);
    out.resize(n);
    for (size_t j = 0; j < n; ++j) {
        switch (et) {
            case GGUF_TYPE_UINT32: out[j] = (int) ((const uint32_t *) d)[j]; break;
            case GGUF_TYPE_INT32:  out[j] =       ((const int32_t  *) d)[j]; break;
            case GGUF_TYPE_UINT8:  out[j] = (int) ((const uint8_t  *) d)[j]; break;
            case GGUF_TYPE_INT8:   out[j] = (int) ((const int8_t   *) d)[j]; break;
            default: out[j] = 0; break;
        }
    }
    return out;
}

rope_kind parse_rope(const std::string & s) {
    if (s == "linear") return rope_kind::linear;
    if (s == "llama3") return rope_kind::llama3;
    return rope_kind::none;
}

} // namespace

bool load_breeze_config(gguf_context * c, breeze_config & o, std::string & err) {
    const int64_t ai = gguf_find_key(c, "general.architecture");
    if (ai < 0 || std::strcmp(gguf_get_val_str(c, ai), ARCH) != 0) {
        err = "not a breeze-tts GGUF (general.architecture mismatch)";
        return false;
    }

    auto & te = o.te;
    te.n_layers  = gu32(c, "text_enc.block_count",   te.n_layers);
    te.n_embd    = gu32(c, "text_enc.embedding_len", te.n_embd);
    te.n_ff      = gu32(c, "text_enc.ffn_len",       te.n_ff);
    te.n_head    = gu32(c, "text_enc.head_count",    te.n_head);
    te.n_kv_head = gu32(c, "text_enc.head_count_kv", te.n_kv_head);
    te.head_dim  = gu32(c, "text_enc.head_dim",      te.head_dim);
    te.vocab     = gu32(c, "text_enc.vocab_size",    te.vocab);
    te.sliding_window = gu32(c, "text_enc.sliding_window", te.sliding_window);
    te.qk_scalar = gu32(c, "text_enc.query_pre_attn_scalar", te.qk_scalar);
    te.rms_eps   = gf32(c, "text_enc.rms_norm_eps",  te.rms_eps);
    te.eoi_token = gu32(c, "text_enc.eoi_token_index", te.eoi_token);
    te.rope_theta_sliding = gf32(c, "text_enc.rope_theta_sliding", te.rope_theta_sliding);
    te.rope_theta_full    = gf32(c, "text_enc.rope_theta_full",    te.rope_theta_full);
    te.rope_full_kind     = parse_rope(gstr(c, "text_enc.rope_type_full", "linear"));
    te.rope_factor_full   = gf32(c, "text_enc.rope_factor_full", te.rope_factor_full);
    te.layer_is_full      = gi32arr(c, "text_enc.layer_is_full");
    if ((int) te.layer_is_full.size() != te.n_layers) {
        // Fall back to the Gemma default pattern (every 6th layer is full).
        te.layer_is_full.assign(te.n_layers, 0);
        for (int i = 5; i < te.n_layers; i += 6) te.layer_is_full[i] = 1;
    }
    te.embed_scale = std::sqrt((float) te.n_embd);

    auto & bb = o.bb;
    bb.n_layers  = gu32(c, "backbone.block_count",   bb.n_layers);
    bb.n_embd    = gu32(c, "backbone.embedding_len", bb.n_embd);
    bb.n_ff      = gu32(c, "backbone.ffn_len",       bb.n_ff);
    bb.n_head    = gu32(c, "backbone.head_count",    bb.n_head);
    bb.n_kv_head = gu32(c, "backbone.head_count_kv", bb.n_kv_head);
    bb.head_dim  = gu32(c, "backbone.head_dim",      bb.head_dim);
    bb.rope_theta = gf32(c, "backbone.rope_theta",   bb.rope_theta);
    bb.rms_eps   = gf32(c, "backbone.rms_norm_eps",  bb.rms_eps);
    bb.max_pos   = gu32(c, "backbone.max_position",  bb.max_pos);
    bb.audio_vocab  = gu32(c, "backbone.audio_vocab_size", bb.audio_vocab);
    bb.lm_head_size = gu32(c, "backbone.lm_head_size",     bb.lm_head_size);
    bb.n_codebooks  = gu32(c, "backbone.num_codebooks",    bb.n_codebooks);

    auto & dd = o.dd;
    dd.n_layers  = gu32(c, "depth.block_count",   dd.n_layers);
    dd.n_embd    = gu32(c, "depth.embedding_len", dd.n_embd);
    dd.n_ff      = gu32(c, "depth.ffn_len",       dd.n_ff);
    dd.n_head    = gu32(c, "depth.head_count",    dd.n_head);
    dd.n_kv_head = gu32(c, "depth.head_count_kv", dd.n_kv_head);
    dd.head_dim  = gu32(c, "depth.head_dim",      dd.head_dim);
    dd.rope_theta = gf32(c, "depth.rope_theta",   dd.rope_theta);
    dd.rms_eps   = gf32(c, "depth.rms_norm_eps",  dd.rms_eps);
    dd.n_codebooks = gu32(c, "depth.num_codebooks", dd.n_codebooks);
    dd.vocab     = gu32(c, "depth.vocab_size",    dd.vocab);
    dd.rope      = parse_rope(gstr(c, "depth.rope_scaling_type", "llama3"));
    dd.rope_factor    = gf32(c, "depth.rope_scaling_factor",    dd.rope_factor);
    dd.rope_low_freq  = gf32(c, "depth.rope_scaling_low_freq",  dd.rope_low_freq);
    dd.rope_high_freq = gf32(c, "depth.rope_scaling_high_freq", dd.rope_high_freq);
    dd.rope_orig_ctx  = gu32(c, "depth.rope_scaling_orig_ctx",  dd.rope_orig_ctx);
    dd.audio_embd_size = bb.n_embd;

    auto & cc = o.cc;
    cc.sample_rate   = gu32(c, "codec.sample_rate",   cc.sample_rate);
    cc.hidden        = gu32(c, "codec.hidden_size",   cc.hidden);
    cc.codebook_size = gu32(c, "codec.codebook_size", cc.codebook_size);
    cc.codebook_dim  = gu32(c, "codec.codebook_dim",  cc.codebook_dim);
    cc.n_quantizers  = gu32(c, "codec.num_quantizers", cc.n_quantizers);
    cc.frame_rate    = gf32(c, "codec.frame_rate",    cc.frame_rate);
    { auto ur = gi32arr(c, "codec.upsample_rates"); if (!ur.empty()) cc.upsample_rates = ur; }
    cc.n_tfm_layers  = gu32(c, "codec.num_tfm_layers", cc.n_tfm_layers);
    cc.n_heads       = gu32(c, "codec.num_heads",     cc.n_heads);
    cc.head_dim      = gu32(c, "codec.head_dim",      cc.head_dim);
    cc.ffn_dim       = gu32(c, "codec.ffn_dim",       cc.ffn_dim);
    cc.sliding_window = gu32(c, "codec.sliding_window", cc.sliding_window);
    cc.norm_eps      = gf32(c, "codec.norm_eps",      cc.norm_eps);
    cc.rope_theta    = gf32(c, "codec.rope_theta",    cc.rope_theta);
    cc.num_filters   = gu32(c, "codec.num_filters",   cc.num_filters);
    cc.kernel_size   = gu32(c, "codec.kernel_size",   cc.kernel_size);
    cc.last_kernel   = gu32(c, "codec.last_kernel_size", cc.last_kernel);
    cc.res_kernel    = gu32(c, "codec.residual_kernel_size", cc.res_kernel);
    cc.dilation_growth = gu32(c, "codec.dilation_growth_rate", cc.dilation_growth);
    cc.compress      = gu32(c, "codec.compress",      cc.compress);
    cc.n_res_layers  = gu32(c, "codec.num_residual_layers", cc.n_res_layers);
    cc.upsample_groups = gu32(c, "codec.upsample_groups", cc.upsample_groups);
    cc.causal        = gu32(c, "codec.causal", 1) != 0;

    auto & sp = o.sp;
    sp.audio        = gu32(c, "tokens.audio",        sp.audio);
    sp.audio_eos    = gu32(c, "tokens.audio_eos",    sp.audio_eos);
    sp.ins_bos      = gu32(c, "tokens.ins_bos",      sp.ins_bos);
    sp.ins_eos      = gu32(c, "tokens.ins_eos",      sp.ins_eos);
    sp.speaker_base = gu32(c, "tokens.speaker_base", sp.speaker_base);
    sp.n_speakers   = gu32(c, "tokens.n_speakers",   sp.n_speakers);
    sp.bos          = gu32(c, "tokens.bos",          sp.bos);
    sp.codebook_eos = gu32(c, "tokens.codebook_eos", sp.codebook_eos);
    sp.codebook_pad = gu32(c, "tokens.codebook_pad", sp.codebook_pad);
    sp.text_vocab   = gu32(c, "tokens.text_vocab_size", sp.text_vocab);
    return true;
}

// HF _compute_llama3_parameters, expressed as the divisor ggml wants.
std::vector<float> build_llama3_freq_factors(int head_dim, float theta,
                                             float factor, float low_freq_factor,
                                             float high_freq_factor, int orig_ctx) {
    const int n = head_dim / 2;
    std::vector<float> ff(n, 1.0f);
    if (factor <= 0.0f || high_freq_factor == low_freq_factor) return ff;

    const float low_wavelen  = (float) orig_ctx / low_freq_factor;
    const float high_wavelen = (float) orig_ctx / high_freq_factor;
    for (int i = 0; i < n; ++i) {
        const float inv_freq = 1.0f / std::pow(theta, (float) (2 * i) / (float) head_dim);
        const float wavelen  = 2.0f * (float) M_PI / inv_freq;
        float scaled;
        if (wavelen > low_wavelen) {
            scaled = inv_freq / factor;
        } else if (wavelen < high_wavelen) {
            scaled = inv_freq;
        } else {
            const float smooth = ((float) orig_ctx / wavelen - low_freq_factor) /
                                 (high_freq_factor - low_freq_factor);
            scaled = (1.0f - smooth) * inv_freq / factor + smooth * inv_freq;
        }
        ff[i] = inv_freq / scaled;   // ggml divides theta by this
    }
    return ff;
}

} // namespace breeze
