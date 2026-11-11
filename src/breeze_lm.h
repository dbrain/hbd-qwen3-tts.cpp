#pragma once

// Breeze backbone: a stock Qwen3 decoder (28 x 2048, GQA 16/8, QK-norm, NeoX
// RoPE theta 1e6) that is NEVER fed token ids. Two input shapes only:
//
//   prefill_embeds(E, n)   E = [n_embd, n] float32 -- the assembled prompt
//                          (text-encoder projections at text positions, summed
//                          audio-codebook embeddings at <|AUDIO|> positions)
//   decode_frame(codes)    one 16-codebook frame; embedding = sum over c of
//                          depth.token_embd[code_c + c*audio_vocab]
//
// Both return the last position's hidden state (2048, the depth decoder's
// conditioning) and its lm_head logits (2052 = 2051 codes + one EOS class).

#include "breeze_weights.h"

#include <string>
#include <vector>

namespace breeze {

class Backbone {
public:
    ~Backbone() { free_state(); }

    // share_sched: run prefills on that backbone's scheduler instead of a new
    // one. The CFG negative branch prefills strictly after the positive one, so
    // a second prefill compute buffer (the biggest one, ~prompt length) would
    // only ever sit idle.
    bool init(BreezeWeights * w, int n_ctx = 2048, Backbone * share_sched = nullptr);
    void reset();

    bool prefill_embeds(const float * embeds, int n,
                        std::vector<float> & hidden_last,
                        std::vector<float> & logits);
    bool decode_frame(const int32_t * codes,
                      std::vector<float> & hidden_last,
                      std::vector<float> & logits);
    // Append one frame to this sequence and one to `other` (separate KV,
    // separate positions) in one batch-2 graph, so the weights are read once
    // for both. Equivalent to decode_frame() on each. CFG passes the same frame
    // twice; two concurrent requests pass their own.
    bool decode_frame_pair(Backbone & other, const int32_t * codes_self,
                           const int32_t * codes_other,
                           std::vector<float> & hidden_self, std::vector<float> & logits_self,
                           std::vector<float> & hidden_other, std::vector<float> & logits_other);
    bool decode_frame_pair(Backbone & other, const int32_t * codes,
                           std::vector<float> & hidden_self, std::vector<float> & logits_self,
                           std::vector<float> & hidden_other, std::vector<float> & logits_other) {
        return decode_frame_pair(other, codes, codes, hidden_self, logits_self,
                                 hidden_other, logits_other);
    }
    // Prefill a run of frames (used to seed generated context; not on the hot path).
    bool prefill_frames(const int32_t * codes, int n_frames,
                        std::vector<float> & hidden_last,
                        std::vector<float> & logits);

    // Summed 16-codebook frame embeddings, [n_embd * n_frames]. Same table and
    // offsets the decode path uses; exposed so the prompt assembler can place
    // reference-audio frames without touching the KV cache.
    bool embed_frames(const int32_t * codes, int n_frames, std::vector<float> & out);

    int  n_past() const { return n_past_; }
    int  n_ctx()  const { return n_ctx_; }
    size_t kv_bytes() const { return kv_buffer_ ? ggml_backend_buffer_get_size(kv_buffer_) : 0; }
    size_t sched_bytes() const;   // excludes a shared scheduler
    const std::string & get_error() const { return error_msg_; }

private:
    struct layer {
        struct ggml_tensor * attn_norm = nullptr;
        struct ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
        // wq||wk||wv as one tensor when the three are adjacent in the weight buffer.
        struct ggml_tensor * wqkv = nullptr;
        struct ggml_tensor * q_norm = nullptr, * k_norm = nullptr;
        struct ggml_tensor * ffn_norm = nullptr;
        struct ggml_tensor * ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;
        // ffn_gate||ffn_up as one tensor when the two are adjacent in the weight
        // buffer; halves the MMVQ launches and the q8_1 activation quantisation.
        struct ggml_tensor * ffn_gate_up = nullptr;
    };

    void free_state();
    bool alloc_kv(int n_alloc, bool copy);
    bool ensure_kv(int need);
    struct ggml_cgraph * build_graph(struct ggml_context * c, int n, bool from_codes);
    struct ggml_cgraph * build_pair_graph(struct ggml_context * c, Backbone & other);
    bool run(int n, const float * embeds, const int32_t * codes,
             std::vector<float> & hidden_last, std::vector<float> & logits);

    BreezeWeights * w_ = nullptr;
    backbone_config cfg_;
    std::vector<layer> layers_;
    struct ggml_tensor * out_norm_ = nullptr;
    struct ggml_tensor * lm_head_ = nullptr;
    struct ggml_tensor * audio_embd_ = nullptr;   // shared with the depth decoder

    struct ggml_context * kv_ctx_ = nullptr;
    ggml_backend_buffer_t kv_buffer_ = nullptr;
    std::vector<struct ggml_tensor *> k_cache_, v_cache_;
    ggml_type kv_type_ = GGML_TYPE_F16;
    int n_ctx_ = 0, kv_alloc_ = 0, kv_initial_ = 0, n_past_ = 0;
    // Bumped on every KV (re)allocation: a cached pair graph holds the OTHER
    // backbone's cache tensors, and a size match alone cannot prove they are
    // the same tensors (reset() can shrink and a later grow land on the same size).
    unsigned kv_gen_ = 0;

    // The prompt prefill is one-shot (rebuild each time), but the per-frame
    // decode graph is identical for a whole 256-frame KV block, so it is built
    // and allocated once and then re-run by re-setting inputs. Same trick as the
    // depth decoder; worth ~0.26 ms of CPU per frame.
    struct decode_slot {
        ggml_backend_sched_t sched = nullptr;
        struct ggml_context * ctx = nullptr;
        struct ggml_cgraph *  gf = nullptr;
        std::vector<uint8_t>  meta;
        int n_kv_eff = -1;
        int kv_alloc = -1;
    } dec_;
    void drop_decode_slot();

    // Batch-2 CFG decode graph, keyed on both sequences' KV shape + generation.
    struct pair_slot {
        ggml_backend_sched_t sched = nullptr;
        struct ggml_context * ctx = nullptr;
        struct ggml_cgraph *  gf = nullptr;
        std::vector<uint8_t>  meta;
        const Backbone * other = nullptr;
        int n_kv_a = -1, n_kv_b = -1;
        unsigned gen_a = 0, gen_b = 0;
    } pair_;
    void drop_pair_slot();

    ggml_backend_sched_t sched_ = nullptr;
    bool sched_owned_ = true;
    std::vector<uint8_t> compute_meta_;
    std::string error_msg_;
};

} // namespace breeze
