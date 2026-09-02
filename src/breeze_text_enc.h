#pragma once

// T5Gemma2 text encoder + the 1152->2048 projection into the backbone's
// embedding space. Bidirectional (NOT causal), no KV cache: one forward per
// text segment, and each segment is encoded independently so segments never
// attend across each other (matches _batched_text_encoder_forward's
// one-row-per-segment batching).
//
// Three Gemma-isms this file exists to get right:
//   * (1 + w) RMSNorm, and four sandwich norms per layer.
//   * embeddings scaled by sqrt(hidden).
//   * per-layer-type RoPE: `sliding_attention` layers use theta 10000 with a
//     bidirectional +-256 window; `full_attention` layers (5, 11, 17, 23) use
//     theta 1000000 with LINEAR scaling factor 8. Getting the split wrong
//     degrades prosody without ever raising an error.
// Attention scale is query_pre_attn_scalar^-0.5, not head_dim^-0.5 (equal here,
// but they are different knobs).

#include "breeze_weights.h"

#include <string>
#include <vector>

namespace breeze {

class TextEncoder {
public:
    ~TextEncoder() { free_state(); }

    bool init(BreezeWeights * w);

    // ids -> projected embeddings [n_embd_backbone * n], row-major per token.
    bool encode(const std::vector<int32_t> & ids, std::vector<float> & out_proj);

    // Un-projected encoder output (parity fixtures / debugging).
    bool encode_hidden(const std::vector<int32_t> & ids, std::vector<float> & out_hidden);

    size_t sched_bytes() const {
        return (sched_ && w_) ? ggml_backend_sched_get_buffer_size(sched_, w_->backend()) : 0;
    }

    const std::string & get_error() const { return error_msg_; }

private:
    struct layer {
        struct ggml_tensor * attn_norm = nullptr;       // pre_self_attn_layernorm
        struct ggml_tensor * attn_post_norm = nullptr;  // post_self_attn_layernorm
        struct ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
        struct ggml_tensor * q_norm = nullptr, * k_norm = nullptr;
        struct ggml_tensor * ffn_norm = nullptr;        // pre_feedforward_layernorm
        struct ggml_tensor * ffn_post_norm = nullptr;   // post_feedforward_layernorm
        struct ggml_tensor * ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;
        bool full = false;
    };

    bool run(const std::vector<int32_t> & ids, bool project, std::vector<float> & out);
    struct ggml_cgraph * build_graph(struct ggml_context * c, int n, bool project,
                                     bool has_eoi);
    void free_state();

    BreezeWeights * w_ = nullptr;
    text_enc_config cfg_;
    std::vector<layer> layers_;
    struct ggml_tensor * tok_embd_ = nullptr;
    struct ggml_tensor * eoi_embd_ = nullptr;
    struct ggml_tensor * out_norm_ = nullptr;
    struct ggml_tensor * proj_ = nullptr;

    ggml_backend_sched_t sched_ = nullptr;
    std::vector<uint8_t> compute_meta_;
    std::string error_msg_;
};

} // namespace breeze
