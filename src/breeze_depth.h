#pragma once

// Breeze depth decoder: 12 x 1024, GQA 8/2, NO QK-norm (unlike the backbone),
// llama3-scaled RoPE (theta 5e5, original_max_position 16). It turns the
// backbone's hidden state plus codebook 0 into codebooks 1..15, one AR step
// each -- 15 sequential passes per 80 ms frame, i.e. the hot loop.
//
// Position semantics, straight from BreezeDepthDecoderModel:
//   cache_position p  ->  input embedding offset (p-1)*vocab, clamped at 0
//                     ->  logits use codebooks_head[p-1]
// so the 2-token prefill at p=0,1 is [backbone_hidden, embed_cb0(c0)] and
// predicts codebook 1; step p predicts codebook p.

#include "breeze_weights.h"

#include <string>
#include <vector>

namespace breeze {

class DepthDecoder {
public:
    ~DepthDecoder() { free_state(); }

    bool init(BreezeWeights * w);

    // Full frame: given the backbone hidden state and codebook-0 code, fill
    // codes[1..n_codebooks-1]. `codes[0]` must already hold c0.
    // temperature <= 0 is greedy. `logits_out`, when non-null, receives the
    // per-step [n_codebooks-1][vocab] logit block (parity fixtures).
    bool run_frame(const float * backbone_hidden, int32_t * codes,
                   float temperature, int top_k, float top_p, uint32_t & rng_state,
                   std::vector<float> * logits_out = nullptr);

    size_t kv_bytes() const { return kv_buf_ ? ggml_backend_buffer_get_size(kv_buf_) : 0; }
    size_t sched_bytes() const;
    void   log_sched_detail() const;
    const std::string & get_error() const { return error_msg_; }

private:
    struct layer {
        struct ggml_tensor * attn_norm = nullptr;
        struct ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
        // wq||wk||wv as one tensor when the three are adjacent in the weight buffer.
        struct ggml_tensor * wqkv = nullptr;
        struct ggml_tensor * ffn_norm = nullptr;
        struct ggml_tensor * ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;
        // ffn_gate||ffn_up as one tensor when the two are adjacent in the weight
        // buffer; halves the MMVQ launches and the q8_1 activation quantisation.
        struct ggml_tensor * ffn_gate_up = nullptr;
    };

    void free_state();
    struct ggml_cgraph * build_graph(struct ggml_context * c, int n, int n_past, int head_idx);
    bool step(int n, int n_past, int head_idx, const float * hidden_or_null,
              const int32_t * codes_in, std::vector<float> & logits);

    BreezeWeights * w_ = nullptr;
    depth_config cfg_;
    std::vector<layer> layers_;
    struct ggml_tensor * tok_embd_ = nullptr;
    struct ggml_tensor * in_proj_ = nullptr;
    struct ggml_tensor * out_norm_ = nullptr;
    // All 15 heads as ONE [n_embd, vocab, 15] tensor. The graph evaluates every
    // head each step and only the wanted slice is read back: that costs ~0.07 ms
    // of extra bandwidth but keeps the graph topology bit-identical across the
    // 15 AR steps, which is what lets ggml's CUDA-graph cache stay warm.
    struct ggml_tensor * heads_ = nullptr;
    struct ggml_tensor * rope_ff_ = nullptr;    // llama3 freq_factors, in aux_buf_

    struct ggml_context * aux_ctx_ = nullptr;
    ggml_backend_buffer_t aux_buf_ = nullptr;
    struct ggml_context * kv_ctx_ = nullptr;
    ggml_backend_buffer_t kv_buf_ = nullptr;
    std::vector<struct ggml_tensor *> k_cache_, v_cache_;
    int kv_slots_ = 0;

    // One persistent, pre-allocated graph per (shape, codebook head): the
    // 2-token prefill and 15 single-token AR steps. Each is allocated ONCE and
    // re-run by re-setting inputs -- ggml_backend_sched_graph_compute skips
    // split+alloc while is_alloc holds, so the 15 launches per 80 ms frame stop
    // paying graph-build and gallocr costs (~0.09 ms each, ~8% of decode).
    //
    // Baking head_idx into the graph is why there are 15 of them: the
    // alternative (one static graph evaluating all 15 heads and reading back a
    // slice) keeps the topology fixed but re-reads 33 MB of head weights on
    // EVERY step, which measured as ~15% of the depth decoder's bandwidth. With
    // the graphs pre-allocated there is nothing left to gain from a fixed
    // topology -- CUDA graphs were measured at ~1% here.
    struct graph_slot {
        ggml_backend_sched_t sched = nullptr;
        struct ggml_context * ctx = nullptr;
        struct ggml_cgraph *  gf = nullptr;
        std::vector<uint8_t>  meta;
        bool allocated = false;
    };
    graph_slot pre_;
    std::vector<graph_slot> step_;
    bool init_slot(graph_slot & g, int n, int n_past, int head_idx);
    std::string error_msg_;
};

} // namespace breeze
