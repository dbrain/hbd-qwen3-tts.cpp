#pragma once

// Higgs-Audio-v3 LM backbone: a vanilla Qwen3-4B (fine-tuned) decoder with a
// tied 8208-way audio head (8 codebooks × 1026). Loads the stock-Qwen3 Q4_K_M
// backbone GGUF + the audio_embd from the codec aux sidecar.
//
//   prefill(prompt_ids)            -> last-token audio logits [8,1026]
//   decode_step(frame_codes[8])    -> next-token audio logits [8,1026]
//
// Audio input embedding = Σ_c audio_embd[c*1026 + code_c]  (sum of 8 rows of
// the same tied matrix). Audio logits = h @ audio_embdᵀ -> [8208] -> [8,1026].
// See PORT-SPEC.md §2-§4 and higgs-audio-ref (ref_full.py).

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>

namespace higgs {

struct lm_layer {
    struct ggml_tensor * attn_norm = nullptr;
    struct ggml_tensor * wq = nullptr;
    struct ggml_tensor * wk = nullptr;
    struct ggml_tensor * wv = nullptr;
    struct ggml_tensor * wo = nullptr;
    struct ggml_tensor * q_norm = nullptr;
    struct ggml_tensor * k_norm = nullptr;
    struct ggml_tensor * ffn_norm = nullptr;
    struct ggml_tensor * ffn_gate = nullptr;
    struct ggml_tensor * ffn_up = nullptr;
    struct ggml_tensor * ffn_down = nullptr;
};

struct lm_config {
    int n_layers   = 36;
    int n_embd     = 2560;
    int n_head     = 32;
    int n_kv_head  = 8;
    int head_dim   = 128;
    int n_ff       = 9728;
    int vocab      = 151936;
    float rope_theta = 1000000.0f;
    float rms_eps  = 1e-6f;
    // audio head
    int n_codebooks = 8;
    int cb_vocab    = 1026;   // 1024 codes + BOC(1024) + EOC(1025)
    int audio_vocab = 8208;   // n_codebooks * cb_vocab
};

class HiggsLM {
public:
    HiggsLM() = default;
    ~HiggsLM();

    // backbone_gguf: stock-Qwen3 Q4_K_M. aux_gguf: codec sidecar (audio_embd).
    bool load_model(const std::string & backbone_gguf, const std::string & aux_gguf,
                    int n_ctx = 4096);
    void unload_model();

    void reset();  // clear KV / n_past

    // Run prefill over prompt token ids (text + control tokens). Returns audio
    // logits for the LAST token as [n_codebooks * cb_vocab] (row-major:
    // cb*cb_vocab + v). Advances n_past by n.
    bool prefill(const int32_t * ids, int n, std::vector<float> & audio_logits_out);

    // Run one decode step on an audio frame (n_codebooks codes incl. BOC/EOC).
    // Returns audio logits for the produced token. Advances n_past by 1.
    bool decode_step(const int32_t * frame_codes, std::vector<float> & audio_logits_out);

    // Prefill a run of `n_frames` audio frames at once (delayed codes, row-major
    // c-fastest: codes[i*n_codebooks + c], incl. BOC/EOC). Embeds Σ_c
    // audio_embd[code + c*cb_vocab] per frame, writes KV, advances n_past by
    // n_frames. Returns last-frame audio logits (usually ignored when used to
    // prefill reference-audio context). Used by the voice-clone / rolling-context
    // long-form path to inject delayed reference codes in the <|ref_audio|> slot.
    bool prefill_audio(const int32_t * delayed_codes, int n_frames, std::vector<float> & audio_logits_out);

    const lm_config & get_config() const { return cfg_; }
    int n_past() const { return n_past_; }
    const std::string & get_error() const { return error_msg_; }

    void log_vram(const char * label) const;

    // Per-section perf accumulators for the decode/prefill loop (populated only
    // when HIGGS_PROF env is set). dump_prof_reset prints the per-call averages
    // and zeroes the accumulators. build = ggml_init+build_graph, alloc =
    // sched_alloc_graph, upload = pos/codes/mask H2D, compute =
    // sched_graph_compute (blocks until GPU done = GPU-busy proxy), get =
    // logits D2H.
    double prof_build_ms = 0, prof_alloc_ms = 0, prof_upload_ms = 0,
           prof_compute_ms = 0, prof_get_ms = 0;
    long   prof_calls = 0;
    void dump_prof_reset(const char * label);

    // Current KV allocation (rows) and the hard cap (= n_ctx). Idle/short
    // utterances keep a small slab; only a long continuous read grows it.
    int kv_alloc() const { return kv_alloc_; }
    int kv_cap() const { return n_ctx_; }

private:
    bool alloc_kv(int n_ctx, ggml_type kv_type);
    // (Re)allocate the KV slab to `n_alloc` rows. If `copy`, the populated
    // [0..n_past_) rows are preserved (host-bounce save/restore); else discarded.
    bool realloc_kv_slab(int n_alloc, bool copy);
    // Ensure the slab can hold `need_pos` positions (rounded up to the FA stride),
    // growing geometrically (cap = n_ctx). Returns false on cap overflow.
    bool ensure_kv(int need_pos);
    // is_audio: input is n_codebooks codes per token (summed embed) vs token ids.
    struct ggml_cgraph * build_graph(struct ggml_context * ctx0, int n_tokens, bool is_audio);
    struct ggml_tensor * rms_norm(struct ggml_context * c, struct ggml_tensor * x,
                                  struct ggml_tensor * w);
    struct ggml_tensor * apply_layer(struct ggml_context * c, struct ggml_cgraph * gf,
                                     struct ggml_tensor * x, const lm_layer & L, int il,
                                     int n_tokens, struct ggml_tensor * pos,
                                     struct ggml_tensor * mask);
    bool run(int n_tokens, bool is_audio, const int32_t * ids_or_codes,
             std::vector<float> & logits_out);

    lm_config cfg_;
    std::map<std::string, struct ggml_tensor *> tensors_;
    struct ggml_tensor * tok_embd_ = nullptr;
    struct ggml_tensor * out_norm_ = nullptr;
    struct ggml_tensor * audio_embd_ = nullptr;
    std::vector<lm_layer> layers_;

    struct ggml_context * ctx_ = nullptr;        // backbone weights metadata
    ggml_backend_buffer_t buffer_ = nullptr;
    struct ggml_context * aux_ctx_ = nullptr;    // audio_embd metadata
    ggml_backend_buffer_t aux_buffer_ = nullptr;

    // KV cache slabs
    struct ggml_context * kv_ctx_ = nullptr;
    ggml_backend_buffer_t kv_buffer_ = nullptr;
    std::vector<struct ggml_tensor *> k_cache_;  // [head_dim, n_kv_head, n_ctx]
    std::vector<struct ggml_tensor *> v_cache_;
    ggml_type kv_type_ = GGML_TYPE_F16;
    int n_ctx_ = 0;       // hard cap (= n_ctx)
    int kv_alloc_ = 0;    // current slab rows (grows on demand, shrinks on reset)
    int kv_initial_ = 0;  // small slab size kept for short/idle utterances
    int n_past_ = 0;

    ggml_backend_t backend_ = nullptr;
    ggml_backend_t backend_cpu_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    std::vector<uint8_t> compute_meta_;
    std::string error_msg_;
};

} // namespace higgs
