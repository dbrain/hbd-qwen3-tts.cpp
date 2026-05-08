#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "coreml_code_predictor.h"

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#ifdef QWEN3_TTS_TIMING
#include <chrono>
#endif

namespace qwen3_tts {

#ifdef QWEN3_TTS_TIMING
struct tts_timing {
    // Prefill phase
    double t_prefill_build_ms = 0;      // build_prefill_graph (embedding lookups, text projection)
    double t_prefill_forward_ms = 0;    // forward_prefill total
    double t_prefill_graph_build_ms = 0;  // build_prefill_forward_graph
    double t_prefill_graph_alloc_ms = 0;  // sched_alloc_graph
    double t_prefill_compute_ms = 0;      // sched_graph_compute
    double t_prefill_data_ms = 0;         // tensor_set + tensor_get + reset

    // Talker forward_step totals (accumulated across all frames)
    double t_talker_forward_ms = 0;       // total time in forward_step()
    double t_talker_graph_build_ms = 0;   // build_step_graph
    double t_talker_graph_alloc_ms = 0;   // sched_alloc_graph
    double t_talker_compute_ms = 0;       // sched_graph_compute
    double t_talker_data_ms = 0;          // tensor_set + tensor_get + reset

    // Code predictor totals (accumulated across all frames)
    double t_code_pred_ms = 0;            // total predict_codes_autoregressive
    double t_code_pred_init_ms = 0;       // init/clear KV cache + CB0 embed lookup
    double t_code_pred_prefill_ms = 0;    // code pred prefill (2-token, per frame)
    double t_code_pred_steps_ms = 0;      // code pred autoregressive steps (14 steps, per frame)
    double t_code_pred_graph_build_ms = 0;  // graph build (prefill + steps combined)
    double t_code_pred_graph_alloc_ms = 0;  // sched_alloc_graph
    double t_code_pred_compute_ms = 0;      // sched_graph_compute
    double t_code_pred_data_ms = 0;         // tensor_set + tensor_get + reset
    double t_code_pred_coreml_ms = 0;       // CoreML predictor compute + I/O

    // Embed lookups in generate() loop
    double t_embed_lookup_ms = 0;

    int32_t n_frames = 0;
    double t_generate_total_ms = 0;
};
#endif

#define QWEN3_TTS_MAX_NODES 16384

// TTS Transformer configuration (Qwen2-based Talker)
struct tts_transformer_config {
    // Model variant: "base", "custom_voice", or "voice_design"
    std::string model_type = "base";
    std::string model_size;  // "0b6", "1b7"

    // Speaker presets (custom_voice models only)
    std::vector<std::string> speaker_names;
    std::vector<int32_t> speaker_ids;
    std::vector<std::string> speaker_dialects;

    // Language map
    std::vector<std::string> language_names;
    std::vector<int32_t> language_ids;

    bool has_speaker_encoder = false;

    // Text embedding
    int32_t text_vocab_size = 151936;
    int32_t text_embd_dim = 2048;

    // Talker transformer
    int32_t hidden_size = 1024;
    int32_t n_layers = 28;
    int32_t n_attention_heads = 16;
    int32_t n_key_value_heads = 8;
    int32_t intermediate_size = 3072;
    int32_t head_dim = 128;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;

    // M-RoPE sections [time, freq, channel] = [24, 20, 20]
    int32_t mrope_section[3] = {24, 20, 20};

    // Codec vocabulary
    int32_t codec_vocab_size = 3072;  // talker.codec_embd/codec_head
    int32_t n_codebooks = 16;

    // Code predictor
    int32_t code_pred_layers = 5;
    int32_t code_pred_vocab_size = 2048;  // Per-codebook vocab
    int32_t code_pred_hidden_size = 0;    // 0 = same as hidden_size (0.6B), otherwise separate (1.7B)
    int32_t code_pred_intermediate_size = 0; // 0 = same as intermediate_size

    // Special codec tokens
    int32_t codec_pad_id = 2148;
    int32_t codec_bos_id = 2149;
    int32_t codec_eos_id = 2150;

    int32_t tts_bos_token_id = 151672;
    int32_t tts_eos_token_id = 151673;
    int32_t tts_pad_token_id = 151671;

    int32_t codec_think_id = 2154;
    int32_t codec_nothink_id = 2155;
    int32_t codec_think_bos_id = 2156;
    int32_t codec_think_eos_id = 2157;

    int32_t english_language_id = 2050;
};

// Transformer layer weights
struct transformer_layer {
    struct ggml_tensor * attn_norm = nullptr;
    
    struct ggml_tensor * attn_q = nullptr;
    struct ggml_tensor * attn_k = nullptr;
    struct ggml_tensor * attn_v = nullptr;
    struct ggml_tensor * attn_output = nullptr;
    struct ggml_tensor * attn_q_norm = nullptr;
    struct ggml_tensor * attn_k_norm = nullptr;
    
    struct ggml_tensor * ffn_norm = nullptr;
    
    struct ggml_tensor * ffn_gate = nullptr;
    struct ggml_tensor * ffn_up = nullptr;
    struct ggml_tensor * ffn_down = nullptr;
};

// TTS Transformer model weights
struct tts_transformer_model {
    tts_transformer_config config;
    
    // Text embedding and projection
    struct ggml_tensor * text_embd = nullptr;      // [text_embd_dim, text_vocab_size]
    struct ggml_tensor * text_proj_fc1 = nullptr;  // [text_embd_dim, text_embd_dim]
    struct ggml_tensor * text_proj_fc1_bias = nullptr;
    struct ggml_tensor * text_proj_fc2 = nullptr;  // [text_embd_dim, hidden_size]
    struct ggml_tensor * text_proj_fc2_bias = nullptr;
    
    // Codec embedding (for autoregressive input)
    struct ggml_tensor * codec_embd = nullptr;     // [hidden_size, codec_vocab_size]
    
    // Talker transformer layers
    std::vector<transformer_layer> layers;
    
    // Final RMSNorm
    struct ggml_tensor * output_norm = nullptr;    // [hidden_size]
    
    // Codec head (for first codebook prediction)
    struct ggml_tensor * codec_head = nullptr;     // [hidden_size, codec_vocab_size]
    
     // Code predictor layers
     std::vector<transformer_layer> code_pred_layers;
     
     // Code predictor output norm (final RMS norm before lm_head)
     struct ggml_tensor * code_pred_output_norm = nullptr;  // [hidden_size]
     
     // Code predictor per-codebook embeddings and heads (15 codebooks, 0 uses talker output)
     std::vector<struct ggml_tensor *> code_pred_embd;  // [hidden_size, code_pred_vocab_size] x 15
     std::vector<struct ggml_tensor *> code_pred_head;  // [hidden_size, code_pred_vocab_size] x 15

     // MTP projection (optional, 1.7B only: projects talker hidden to code_pred hidden)
     struct ggml_tensor * mtp_proj_weight = nullptr;  // [code_pred_hidden, talker_hidden]
     struct ggml_tensor * mtp_proj_bias = nullptr;    // [code_pred_hidden]
    
    // GGML context for tensor metadata
    struct ggml_context * ctx = nullptr;
    
    // Backend buffer for weights
    ggml_backend_buffer_t buffer = nullptr;
    
    // Tensor name to tensor mapping
    std::map<std::string, struct ggml_tensor *> tensors;
};

// KV cache for autoregressive generation
struct tts_kv_cache {
    std::vector<struct ggml_tensor *> k_cache;
    std::vector<struct ggml_tensor *> v_cache;
    
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    int32_t n_ctx = 0;
    int32_t n_used = 0;
    int32_t head_dim = 128;
    int32_t n_kv_heads = 8;
    int32_t n_layers = 28;
    // Soft cap for lazy growth, set per-synth from worst_case (prefill +
    // max_audio_tokens + 8). Caps overshoot when audio decode hits the
    // requested ceiling — without it, geometric/round-up growth allocates
    // the next 2048-step beyond worst_case (60-120 MiB regression vs eager
    // alloc on max-budget synths). 0 = no cap.
    int32_t max_n_ctx_hint = 0;
};

// TTS Transformer state
struct tts_transformer_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;

    tts_kv_cache cache;           // Talker KV cache (28 layers)
    tts_kv_cache code_pred_cache; // Code predictor KV cache (5 layers)

    // Code-pred full-AR cgraph cache (v9): the full-AR cgraph is identical
    // across frames as long as (temperature_path, gumbel_top_k, n_ctx_kv)
    // matches, and the model weights / KV-cache buffer pointers are stable.
    // Caching the cgraph + ctx avoids the per-frame ~0.2 ms ggml graph build.
    // Must use a SEPARATE compute_meta because the talker's build_step_graph
    // re-ggml_init's over the shared state_.compute_meta between frames,
    // which would otherwise trample the cached tensor structs.
    std::vector<uint8_t>  cp_compute_meta;
    struct ggml_context * cp_ctx_cached = nullptr;
    struct ggml_cgraph  * cp_gf_cached  = nullptr;
    bool                  cp_cache_temp_pos = false; // signature: temperature > 0?
    int32_t               cp_cache_top_k    = 0;
    int32_t               cp_cache_n_ctx_kv = 0;

    // Dedicated gallocr for the cached full-AR cgraph path (v9.4). Bypasses
    // ggml_backend_sched entirely on cache hits — sched_split_graph frees and
    // re-inits sched->ctx every alloc_graph, so any cgraph cached across
    // frames hits the gallocr offset-reuse trap (siglip2 graph_cache_gotcha).
    // The full-AR cgraph runs entirely on the talker backend (no CPU
    // fallback paths), so we can drive it with a one-backend gallocr +
    // direct ggml_backend_graph_compute and skip the split/copy mutation.
    ggml_gallocr_t cp_galloc = nullptr;
};

// TTS Transformer class
class TTSTransformer {
public:
    TTSTransformer();
    ~TTSTransformer();
    
    // Load model from GGUF file
    bool load_model(const std::string & model_path);

    // Release all model/runtime resources
    void unload_model();
    
    // Initialize KV cache
    bool init_kv_cache(int32_t n_ctx);

    // Lazily grow the KV cache to at least `target_n_ctx` slots, preserving
    // the populated [0, n_used) rows. No-op if current capacity is enough.
    // Used to defer the worst-case n_ctx allocation: synth starts with a
    // small budget and only pays for slots audio actually consumes. Mirrors
    // the vocoder's stream-KV slab grow pattern.
    bool grow_kv_cache(int32_t target_n_ctx);

    // Clear KV cache
    void clear_kv_cache();

    // Snapshot of K/V cache contents for the first `n_pos` positions across
    // all 28 layers. Bytes are F16, packed [head_dim, n_kv_heads, n_pos] per
    // layer. Used to seed the talker prefill with cached prefix state on
    // requests that share an instruct + ref_text + speaker prefix.
    struct prefill_kv_snapshot {
        int32_t n_pos = 0;
        std::vector<std::vector<uint8_t>> k_layers; // size = n_layers
        std::vector<std::vector<uint8_t>> v_layers;
    };

    // Capture current KV cache contents for positions [0, n_pos) into the
    // snapshot. Cheap — one ggml_backend_tensor_get per layer per K/V (so
    // 56 small downloads; for 28 layers x 8 heads x 128 head_dim x 60 pos
    // x 2 (F16) ≈ 7 MB total).
    bool capture_kv_state(int32_t n_pos, prefill_kv_snapshot & out);

    // Restore KV cache: write `snap.k_layers[il]` and `snap.v_layers[il]`
    // into positions [0, snap.n_pos) of each layer, then zero positions
    // [snap.n_pos, n_ctx) so subsequent forward_prefill cannot read stale
    // bytes through full-n_ctx views. Sets state_.cache.n_used = snap.n_pos.
    bool restore_kv_state(const prefill_kv_snapshot & snap);
    
    // Initialize code predictor KV cache (5 layers, max 16 context)
    bool init_code_pred_kv_cache(int32_t n_ctx);
    
    // Clear code predictor KV cache
    void clear_code_pred_kv_cache();
    
    // Forward pass for text tokens (prefill phase)
    // text_tokens: input text token IDs [n_tokens]
    // speaker_embd: speaker embedding [hidden_size] (optional, can be nullptr)
    // n_past: number of tokens already in KV cache
    // output: hidden states [n_tokens, hidden_size]
    bool forward_text(const int32_t * text_tokens, int32_t n_tokens,
                      const float * speaker_embd, int32_t n_past,
                      std::vector<float> & output);

    bool forward_prefill(const float * prefill_embd, int32_t n_tokens,
                         int32_t n_past, std::vector<float> & output,
                         std::vector<float> * logits_out = nullptr);
    
    // Forward pass for codec tokens (generation phase)
    // codec_token: single codec token for first codebook
    // n_past: number of tokens already in KV cache
    // output: logits for next codec token [codec_vocab_size]
    bool forward_codec(int32_t codec_token, int32_t n_past,
                       std::vector<float> & output);

    bool forward_step(const float * step_embd, int32_t n_past,
                      std::vector<float> & output,
                      std::vector<float> * hidden_out = nullptr);
    
    // Get hidden states from last forward pass (for code predictor)
    bool get_hidden_states(std::vector<float> & hidden) const;
    
    // Run code predictor to get all 16 codebook predictions
    // hidden: hidden states from talker [hidden_size]
    // prev_codes: previous codes for codebooks 1-15 (can be nullptr for first step)
    // output: logits for all 16 codebooks [16, code_pred_vocab_size]
    bool predict_codes(const float * hidden, const int32_t * prev_codes,
                       std::vector<float> & output);
    
    // Run code predictor autoregressively to generate 15 codes (codebooks 1-15)
    // hidden: hidden states from talker [hidden_size]
    // codebook_0_token: the codebook 0 token (used to create 2-token prefill input)
    // output: generated codes for codebooks 1-15 [15]
    bool predict_codes_autoregressive(const float * hidden, int32_t codebook_0_token, 
                                       std::vector<int32_t> & output,
                                       float temperature = 0.9f,
                                       int32_t top_k = 50);
    
    // Generate speech codes autoregressively
    // text_tokens: input text token IDs [n_tokens]
    // speaker_embd: speaker embedding [hidden_size]
    // max_len: maximum number of frames to generate
    // output: generated speech codes [n_frames, n_codebooks]
    bool generate(const int32_t * text_tokens, int32_t n_tokens,
                  const float * speaker_embd, int32_t max_len,
                  std::vector<int32_t> & output,
                  int32_t language_id = 2050,
                  float repetition_penalty = 1.05f,
                  float temperature = 0.9f,
                  int32_t top_k = 50,
                  const int32_t * instruct_tokens = nullptr,
                  int32_t n_instruct_tokens = 0,
                  const int32_t * ref_text_tokens = nullptr,
                  int32_t n_ref_text_tokens = 0,
                  const int32_t * ref_codes = nullptr,
                  int32_t n_ref_frames = 0,
                  // Voice-keyed prefill cache: when non-zero, generate()
                  // looks up cached talker KV state from prior invocations
                  // with the same key (= same instruct + ref_text + speaker
                  // prefix) and skips the prefix forward-pass. Caching is
                  // bounded to the [prefix..end-of-ref_text] window — the
                  // request's new_text and ref_codes section always run.
                  uint64_t prefill_cache_key = 0);

    // Drop all cached prefill KV snapshots. Called by Qwen3TTS::unload_model.
    void clear_prefill_kv_cache();

    // One-line VRAM/host inventory for the talker. Triggered via
    // QWEN3_TTS_LOG_SCHED — emits weights, KV, code-pred KV, and sched
    // buffer sizes so a session can audit where bytes are going.
    void log_vram_breakdown(const char * label) const;

    // Talker KV-cache occupancy (positions written, including any restored
    // prefix snapshot). Used by VRAM probes to correlate with chunk index.
    int32_t get_kv_n_used() const { return state_.cache.n_used; }

    // -- Persistence helpers (used by Qwen3TTS::save/load_voice_warmup) ---

    // Read-only access to a cached entry, if present. Returns nullptr when
    // missing. Caller must NOT mutate or persist the pointer past the
    // next call that could trigger eviction.
    const prefill_kv_snapshot * get_prefill_kv_snapshot(uint64_t key) const;

    // Insert a snapshot into the cache (overwrites any existing entry under
    // the same key, refreshes LRU position). Used at startup when the server
    // imports a previously-saved voice warmup blob.
    void put_prefill_kv_snapshot(uint64_t key, prefill_kv_snapshot snap);

    // Read-only access to the cached icl_codec_section bytes for a voice
    // (keyed by ref_codes hash). Returns nullptr if not cached.
    const std::vector<float> * get_icl_codec_section(uint64_t ref_codes_hash) const;

    // Insert icl_codec_section bytes into the cache.
    void put_icl_codec_section(uint64_t ref_codes_hash, std::vector<float> data);
    
    const tts_transformer_config & get_config() const { return model_.config; }

    // extract a single row from codec_embd for a given token ID
    bool get_codec_embedding(int32_t token_id, std::vector<float> & output);

    const std::string & get_error() const { return error_msg_; }

    // Set abort callback checked before each graph compute (thread-safe)
    void set_abort_callback(ggml_abort_callback callback, void * data);

    // Fired inside generate() after each frame's 16 codebook codes are
    // pushed onto the output vector. The callback receives the frame
    // index and a pointer to the new frame's codes (valid only for the
    // duration of the call). Returning false aborts generation.
    using frame_emit_fn = std::function<bool(int32_t frame_idx,
                                             const int32_t * frame_codes)>;
    void set_frame_callback(frame_emit_fn cb) { frame_cb_ = std::move(cb); }

    // Enable per-stage progress prints inside generate() (prefill + decode loop)
    void set_verbose(bool v) { verbose_ = v; }

    // Reseed the sampling RNG (for reproducible generation).
    void set_seed(uint64_t seed) { rng_.seed(seed); }

    // Check if abort has been requested
    bool is_aborted() const;

    // Stats from the most recent generate() call. Prefill = build_prefill_graph
    // + forward_prefill (everything before the autoregressive loop). Decode
    // timing is the loop itself; wall-clock of generate() = prefill + decode
    // (plus negligible overhead).
    int32_t get_last_n_prefill_tokens() const { return last_n_prefill_tokens_; }
    int64_t get_last_prefill_ms()       const { return last_prefill_ms_; }
    int64_t get_last_decode_ms()        const { return last_decode_ms_; }
    
    // Legacy interface for compatibility
    bool forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                 std::vector<float> & output);
    
    bool forward_with_audio(const int32_t * tokens, int32_t n_tokens,
                            const float * audio_embd, int32_t n_audio,
                            int32_t audio_start_pos, int32_t n_past,
                            std::vector<float> & output);
    
private:
    bool try_init_coreml_code_predictor(const std::string & model_path);
    bool predict_codes_autoregressive_coreml(const float * hidden, int32_t codebook_0_token,
                                             std::vector<int32_t> & output,
                                             float temperature,
                                             int32_t top_k);

    bool build_prefill_graph(const int32_t * text_tokens, int32_t n_tokens,
                             const float * speaker_embd, int32_t language_id,
                             std::vector<float> & prefill_embd,
                             std::vector<float> & trailing_text_hidden,
                             std::vector<float> & tts_pad_embed,
                             const int32_t * instruct_tokens = nullptr,
                             int32_t n_instruct_tokens = 0,
                             const int32_t * ref_text_tokens = nullptr,
                             int32_t n_ref_text_tokens = 0,
                             const int32_t * ref_codes = nullptr,
                             int32_t n_ref_frames = 0,
                             // out: number of leading positions in
                             // prefill_embd whose KV state is voice-bound
                             // and safe to cache (= prefix + ref_text in
                             // ICL mode; prefix only otherwise; 0 means
                             // nothing cacheable).
                             int32_t * cacheable_len = nullptr);

    struct ggml_cgraph * build_prefill_forward_graph(int32_t n_tokens, int32_t n_past);

    struct ggml_cgraph * build_step_graph(int32_t n_past);

    bool project_text_tokens(const int32_t * text_tokens, int32_t n_tokens,
                             std::vector<float> & output);

    bool lookup_embedding_rows(struct ggml_tensor * embedding, const int32_t * token_ids,
                               int32_t n_tokens, const char * input_name,
                               const char * output_name, std::vector<float> & output);
    bool lookup_single_embedding_row(struct ggml_tensor * embedding, int32_t token_id,
                                     float * out_row);
    
    // Build computation graph for code predictor
    struct ggml_cgraph * build_code_pred_graph(int32_t n_prev_codes);
    
    // Build computation graph for single-step autoregressive code predictor
    // n_past: number of tokens already in KV cache (0-14)
    // generation_step: which codebook we're predicting (0-14)
    // temperature/top_k: when temperature > 0 the cgraph appends an on-GPU
    //   Gumbel-max sampling chain (input tensor "inp_gumbel" expected,
    //   shape [top_k] f32). When temperature <= 0 the cgraph appends a
    //   plain ggml_argmax(logits). The sampled token is exposed as the
    //   named output tensor "sampled_token" (shape [1] i32).
    struct ggml_cgraph * build_code_pred_step_graph(int32_t n_past, int32_t generation_step,
                                                     float temperature, int32_t top_k);

    // Build computation graph for 2-token prefill of code predictor
    // Processes [past_hidden, codec_embd(codebook_0_token)] together.
    // Same sampling-chain semantics as build_code_pred_step_graph.
    struct ggml_cgraph * build_code_pred_prefill_graph(float temperature, int32_t top_k);

    // Append a GPU sampling chain to gf for tensor `logits` (shape [V, 1] f32).
    // Returns the named sampled-token tensor (shape [1] i32).
    // - temperature <= 0 → ggml_argmax(logits).
    // - temperature  > 0 → scale(1/T) → top_k → gather → +Gumbel → argmax → gather.
    //   Caller must register a graph input named `gumbel_input_name` with
    //   shape [top_k] f32 and tensor_set it before compute. `gumbel_input_name`
    //   must be unique per cgraph (so per-step in Phase 2's merged cgraph).
    struct ggml_tensor * append_gpu_sampling(struct ggml_context * ctx,
                                              struct ggml_cgraph * gf,
                                              struct ggml_tensor * logits,
                                              float temperature,
                                              int32_t top_k,
                                              const char * gumbel_input_name,
                                              const char * sampled_output_name);

    // Same as append_gpu_sampling but consumes an existing `gumbel` tensor
    // (typically a view into a shared per-frame Gumbel buffer). Used by
    // build_code_pred_full_ar_graph so all 15 sampling chains share one
    // [top_k * 15] f32 input rather than emitting 15 separate inputs.
    struct ggml_tensor * append_gpu_sampling_view(struct ggml_context * ctx,
                                                   struct ggml_cgraph * gf,
                                                   struct ggml_tensor * logits,
                                                   float temperature,
                                                   int32_t top_k,
                                                   struct ggml_tensor * gumbel,
                                                   const char * sampled_output_name);

    // Phase 2: single cgraph for the entire code-pred AR loop (1 prefill +
    // 14 AR steps, 15 codebook outputs total). Each step's sampled-token
    // tensor feeds the next step's embed-lookup directly via graph deps,
    // so ggml-cuda's per-compute graph capture covers the whole AR — no
    // host roundtrip between steps. Outputs: 15 named tensors
    // "sampled_token_0" .. "sampled_token_14" (each [1] i32). Inputs:
    // "inp_hidden", "inp_cb0_embd", "inp_pos_all" [16] i32,
    // "inp_mask_all" [n_ctx, 14] f16, "inp_gumbel_all" [top_k, 15] f32
    // (only when temperature > 0).
    //
    // Caller supplies ctx + gf so the cgraph can be cached across frames
    // (the topology depends only on temperature/top_k; data values flow
    // through the named inputs each frame).
    void build_code_pred_full_ar_graph_into(struct ggml_context * ctx0,
                                              struct ggml_cgraph * gf,
                                              float temperature, int32_t top_k);
    
    // Parse hyperparameters from GGUF
    bool parse_config(struct gguf_context * ctx);
    
    // Create tensor structures
    bool create_tensors(struct gguf_context * ctx);
    
    // Load tensor data from file
    bool load_tensor_data(const std::string & path, struct gguf_context * ctx);
    
    tts_transformer_model model_;
    tts_transformer_state state_;
    std::string error_msg_;
    ggml_abort_callback abort_cb_ = nullptr;
    void * abort_data_ = nullptr;
    bool verbose_ = false;
    frame_emit_fn frame_cb_;

    // Stats populated by generate()
    int32_t last_n_prefill_tokens_ = 0;
    int64_t last_prefill_ms_ = 0;
    int64_t last_decode_ms_ = 0;

    // Voice-keyed prefill KV cache. Looked up + populated inside generate()
    // when the caller passes prefill_cache_key. Bounded eviction (LRU,
    // capacity 16) keeps host-memory use manageable (~7 MB per entry at
    // typical ICL prefix lengths).
    std::map<uint64_t, prefill_kv_snapshot> prefill_kv_cache_;
    std::vector<uint64_t> prefill_kv_cache_lru_;  // front = oldest
    static constexpr size_t kPrefillKvCacheCapacity = 16;

    // Per-voice cache of the icl_codec_section embedding bytes built by
    // build_prefill_graph (codec_bos + Σ ref_code embeddings, overlayed with
    // tts_pad_embed). Keyed by FNV-1a hash of ref_codes content. Avoids the
    // 1072 single-row D2H embedding lookups that dominate voice_clone build
    // cost (~400 ms on Q4_K_M for 67 frames × 16 codebooks).
    std::map<uint64_t, std::vector<float>> icl_codec_section_cache_;

    // Cached hidden states from last forward pass
    std::vector<float> last_hidden_;
    std::vector<ggml_fp16_t> embd_row_fp16_scratch_;
    // Per-frame mask scratch reused across forward_step (talker) and the
    // code-predictor step. They never overlap on the same frame, so one
    // shared buffer is enough; grow on demand, never shrink.
    std::vector<ggml_fp16_t> step_mask_scratch_;
    // Per-frame Gumbel noise scratch for on-GPU code-pred sampling.
    // Sized to top_k * 15 (1 prefill + 14 AR steps) and regenerated from rng_
    // at the top of each predict_codes_autoregressive() call.
    std::vector<float> code_pred_gumbel_scratch_;
    std::mt19937 rng_{std::random_device{}()};
    CoreMLCodePredictor coreml_code_predictor_;
    bool use_coreml_code_predictor_ = false;
    std::string coreml_code_predictor_path_;
    bool skip_ggml_code_pred_layers_ = false;

#ifdef QWEN3_TTS_TIMING
    tts_timing * timing_ = nullptr;
#endif
};

// Free model resources
void free_transformer_model(tts_transformer_model & model);

// Free KV cache resources
void free_tts_kv_cache(tts_kv_cache & cache);

} // namespace qwen3_tts
