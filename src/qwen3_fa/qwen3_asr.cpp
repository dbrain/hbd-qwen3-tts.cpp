// qwen3_asr.cpp — Qwen/Qwen3-ASR-0.6B ggml runtime
//
// STAGE 1 (current commit): loader + audio encoder conv front-end only.
//   - Loads the GGUF produced by models/convert-qwen3-asr-to-gguf.py
//   - Computes the per-chunk Conv2D subsampler (conv2d1/2/3 + GELU) and the
//     conv_out linear projection. Output shape (num_chunks, T_chunk_out, 896).
//   - Exposed via qwen3_asr_run_conv() for differential testing against
//     /tmp/qwen3-asr-ref/jfk/conv_out.npy
//
// Subsequent stages will add the chunked self-attention encoder body, the
// projector head, the Qwen3 0.6B LLM forward, and the audio-injection glue.
//
// See qwen3-asr-todo.md for the full plan.

#include "qwen3_asr.h"
#include "crisp_audio.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct qwen3_asr_hparams {
    // Audio
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 400;
    uint32_t win_length = 400;
    uint32_t hop_length = 160;
    uint32_t audio_n_layers = 18;
    uint32_t audio_d_model = 896;
    uint32_t audio_n_heads = 14;
    uint32_t audio_head_dim = 64;
    uint32_t audio_ff_dim = 3584;
    uint32_t audio_conv_ch = 480;
    uint32_t audio_proj_dim = 1024;
    uint32_t audio_max_pos = 1500;

    // Chunking parameters (from reference impl: n_window=50, n_window_infer=800)
    uint32_t n_window = 50;
    uint32_t n_window_infer = 800;

    // LLM (Qwen3 0.6B)
    uint32_t llm_n_layers = 28;
    uint32_t llm_d_model = 1024;
    uint32_t llm_n_heads = 16;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 3072;
    float llm_rope_theta = 1e6f;
    float llm_rms_eps = 1e-6f;
    uint32_t llm_vocab_size = 151936;
    uint32_t llm_max_pos = 65536;

    // The LM head's actual output dimension. For the standard ASR
    // models this equals llm_vocab_size (152064 for 1.7B, 151936 for
    // 0.6B). For the Qwen3-ForcedAligner-0.6B variant the head
    // outputs 5000 timestamp classes instead — we read the real
    // value from the loaded `output.weight` tensor's shape rather
    // than asserting equality with vocab_size.
    uint32_t llm_lm_head_dim = 0; // 0 = "use vocab_size" sentinel

    // Special tokens
    uint32_t audio_start_token_id = 151669;
    uint32_t audio_end_token_id = 151670;
    uint32_t audio_pad_token_id = 151676;
    uint32_t eos_token_id = 151645;
    uint32_t pad_token_id = 151643;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

struct qwen3_asr_audio_block {
    // Pre-LN self-attention
    ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor *attn_k_w = nullptr, *attn_k_b = nullptr;
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    // Pre-LN FFN (GELU)
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor *ffn_up_w = nullptr, *ffn_up_b = nullptr;
    ggml_tensor *ffn_down_w = nullptr, *ffn_down_b = nullptr;
};

struct qwen3_asr_audio_tower {
    // Conv subsampler front-end (4 stride-2 freq convs as 2D over the mel image)
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;       // (480, 1,   3, 3)
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;       // (480, 480, 3, 3)
    ggml_tensor *conv3_w = nullptr, *conv3_b = nullptr;       // (480, 480, 3, 3)
    ggml_tensor *conv_out_w = nullptr, *conv_out_b = nullptr; // (896, 7680)

    // Encoder body
    std::vector<qwen3_asr_audio_block> blocks;

    // Final norm + projector head (896 → 896 → GELU → 1024)
    ggml_tensor *ln_post_w = nullptr, *ln_post_b = nullptr;
    ggml_tensor *proj1_w = nullptr, *proj1_b = nullptr;
    ggml_tensor *proj2_w = nullptr, *proj2_b = nullptr;

    // Mel preprocessor (baked from WhisperFeatureExtractor by the converter)
    ggml_tensor* mel_filters = nullptr; // (n_freqs=201, n_mels=128) F32
    ggml_tensor* mel_window = nullptr;  // (400,) F32 hann window
};

struct qwen3_asr_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_qkv_w = nullptr; // fused Q+K+V for single matmul
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr; // Qwen3 per-head Q RMSNorm
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct qwen3_asr_llm {
    ggml_tensor* token_embd_w = nullptr; // (151936, 1024)
    std::vector<qwen3_asr_llm_block> blocks;
    ggml_tensor* output_norm_w = nullptr;
    ggml_tensor* output_w = nullptr;
};

struct qwen3_asr_model {
    qwen3_asr_hparams hparams;
    qwen3_asr_audio_tower audio;
    qwen3_asr_llm llm;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    // PLAN #69a: optional second buffer for layers spilled to CPU.
    // Non-null only when CRISPASR_N_GPU_LAYERS triggered a split load.
    ggml_backend_buffer_t buf_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Sinusoidal positional embedding for the audio encoder, computed once
    // at load time. Layout: row-major (max_pos, d_model) where row p is the
    // pos embed for position p.
    std::vector<float> audio_pe; // size = audio_max_pos * audio_d_model
};

struct qwen3_asr_vocab {
    std::vector<std::string> id_to_token;

    // Reverse lookup: byte-encoded vocab string → token id (for BPE encode).
    std::unordered_map<std::string, int32_t> token_to_id;

    // BPE merges loaded from the GGUF. Each merge is a (left, right) pair
    // of byte-encoded strings; the rank is the index in the list (lower = earlier
    // merge = higher priority, matching GPT-2 / Qwen2 BPE convention).
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" → rank
};

// Forward decl from crisp_audio — full type lives in crisp_audio/include/crisp_audio.h
struct crisp_audio_context;

struct qwen3_asr_context {
    qwen3_asr_context_params params;

    qwen3_asr_model model;
    qwen3_asr_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    // Fused QKV weights (optional optimization)
    ggml_context* fused_ctx = nullptr;
    ggml_backend_buffer_t fused_buf = nullptr;

    std::vector<uint8_t> compute_meta;

    // KV cache (Stage 5). Single tensor for K, single for V, both shape
    // (head_dim, max_ctx, n_kv_heads, n_layers). Allocated to backend.
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;

    int n_threads = 4;

    // Shared audio-tower runtime (loaded lazily on first audio call). The
    // qwen3_asr_audio_tower struct above is kept around so existing in-tree
    // tests / fallbacks compile; it is no longer the path used by
    // qwen3_asr_compute_mel / qwen3_asr_run_encoder once `audio_ca` is open.
    crisp_audio_context* audio_ca = nullptr;
    std::string model_path; // remembered for lazy crisp_audio init

    // ---- Streaming forced-alignment state ----
    // Held across partial align calls in the same paragraph so we can
    // skip re-forwarding the audio prefix each pass. Cleared by
    // qwen3_asr_align_streaming_reset() (or implicit reset=true).
    //
    // After each partial, the KV cache holds:
    //   slot 0:                       audio_start
    //   slot [1, N_enc+1):            audio_pad × N_enc
    //   slot [N_enc+1, N_enc+2):      audio_end          (may be stale across partials)
    //   slot [N_enc+2, N_enc+2+nt):   text suffix tokens (may be stale across partials)
    //
    // On the next partial we rewind n_past to (N_enc + 1) so audio_end
    // + text suffix get re-written at their NEW absolute positions. The
    // audio prefix's K/V stays put.
    //
    // Caveat (and the reason this is gated behind QWEN3_FA_STREAMING_ALIGN):
    // the audio encoder runs full bidirectional self-attention across all
    // chunks (attn_window_mode=0 in the cstr/qwen3-forced-aligner-0.6b
    // GGUF), so the audio_pad embeds for old frames are NOT bit-identical
    // between partials — appending new audio shifts the embeds of the
    // already-cached frames. We accept that drift in exchange for the
    // ~N_enc/N_total per-partial wallclock saving; measure word-timing
    // drift vs the one-shot path with /tmp/diff_words.py.
    struct {
        bool                 initialized   = false;
        int                  N_enc_committed = 0;     // audio_pad slots currently in cache
        int                  n_text_tokens   = 0;     // count of audio_end + words + timestamps
        std::vector<int32_t> text_ids;                // [audio_end, w1_toks, ts, ts, w2_toks, ts, ts, ...]
        // Snapshot of the word list as a single newline-joined string, so we
        // can cheaply detect a different paragraph being pushed in and force
        // a reset. (Per-word strings live in the caller; we only need a hash.)
        std::string          words_signature;

        // Cached raw (pre-normalization) log-mel spectrogram in (T, n_mels)
        // row-major layout. Grown incrementally each partial via
        // crisp_audio_compute_log_mel_range. Bit-identical reconstruction
        // to a one-shot mel compute is guaranteed because the STFT + mel
        // filter + log10 steps are time-local; only the GlobalClipMax
        // normalization is global, and that's re-applied to a temp copy
        // of the full buffer on each partial.
        std::vector<float> raw_log_mel_TM;
        int                T_committed_mel = 0;
    } align_stream;
};

// ===========================================================================
// Loader helpers
// ===========================================================================

#include "core/gguf_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static ggml_tensor* try_get(qwen3_asr_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(qwen3_asr_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "qwen3_asr");
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool qwen3_asr_load_model(qwen3_asr_model& model, qwen3_asr_vocab& vocab, const char* path,
                                 ggml_backend_t backend, ggml_backend_t backend_cpu) {
    // ---- pass 1: read hparams + vocab via metadata-only context ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "qwen3asr.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "qwen3asr.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "qwen3asr.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "qwen3asr.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "qwen3asr.hop_length", hp.hop_length);
        hp.audio_n_layers = core_gguf::kv_u32(gctx, "qwen3asr.audio.n_layers", hp.audio_n_layers);
        hp.audio_d_model = core_gguf::kv_u32(gctx, "qwen3asr.audio.d_model", hp.audio_d_model);
        hp.audio_n_heads = core_gguf::kv_u32(gctx, "qwen3asr.audio.n_heads", hp.audio_n_heads);
        hp.audio_head_dim = core_gguf::kv_u32(gctx, "qwen3asr.audio.head_dim", hp.audio_head_dim);
        hp.audio_ff_dim = core_gguf::kv_u32(gctx, "qwen3asr.audio.ff_dim", hp.audio_ff_dim);
        hp.audio_conv_ch = core_gguf::kv_u32(gctx, "qwen3asr.audio.conv_channels", hp.audio_conv_ch);
        hp.audio_proj_dim = core_gguf::kv_u32(gctx, "qwen3asr.audio.proj_dim", hp.audio_proj_dim);
        hp.audio_max_pos = core_gguf::kv_u32(gctx, "qwen3asr.audio.max_source_pos", hp.audio_max_pos);

        hp.llm_n_layers = core_gguf::kv_u32(gctx, "qwen3asr.llm.n_layers", hp.llm_n_layers);
        hp.llm_d_model = core_gguf::kv_u32(gctx, "qwen3asr.llm.d_model", hp.llm_d_model);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "qwen3asr.llm.n_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "qwen3asr.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(gctx, "qwen3asr.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(gctx, "qwen3asr.llm.ff_dim", hp.llm_ff_dim);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "qwen3asr.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(gctx, "qwen3asr.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "qwen3asr.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "qwen3asr.llm.max_pos", hp.llm_max_pos);

        hp.audio_start_token_id = core_gguf::kv_u32(gctx, "qwen3asr.audio_start_token_id", hp.audio_start_token_id);
        hp.audio_end_token_id = core_gguf::kv_u32(gctx, "qwen3asr.audio_end_token_id", hp.audio_end_token_id);
        hp.audio_pad_token_id = core_gguf::kv_u32(gctx, "qwen3asr.audio_pad_token_id", hp.audio_pad_token_id);
        hp.eos_token_id = core_gguf::kv_u32(gctx, "qwen3asr.eos_token_id", hp.eos_token_id);
        hp.pad_token_id = core_gguf::kv_u32(gctx, "qwen3asr.pad_token_id", hp.pad_token_id);

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            vocab.token_to_id.reserve(vocab.id_to_token.size());
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }

        // Register Qwen2/Qwen3 special tokens explicitly. The original
        // vocab.json that the converter pulls strings from only has 151 643
        // regular tokens; the special tokens (<|im_start|>, <|audio_pad|>,
        // ...) live in tokenizer_config.json's added_tokens list, which the
        // converter currently doesn't propagate, so they end up as
        // "[PAD151644]" etc. in vocab.id_to_token. Patch them in here so
        // qwen3_asr_tokenize and qwen3_asr_token_text can find them.
        struct SpecialTok {
            int id;
            const char* text;
        };
        static const SpecialTok specials[] = {
            {151643, "<|endoftext|>"},        {151644, "<|im_start|>"},       {151645, "<|im_end|>"},
            {151646, "<|object_ref_start|>"}, {151647, "<|object_ref_end|>"}, {151648, "<|box_start|>"},
            {151649, "<|box_end|>"},          {151650, "<|quad_start|>"},     {151651, "<|quad_end|>"},
            {151652, "<|vision_start|>"},     {151653, "<|vision_end|>"},     {151654, "<|vision_pad|>"},
            {151655, "<|image_pad|>"},        {151656, "<|video_pad|>"},      {151669, "<|audio_start|>"},
            {151670, "<|audio_end|>"},        {151676, "<|audio_pad|>"},
        };
        for (const auto& sp : specials) {
            if (sp.id < (int)vocab.id_to_token.size()) {
                // Drop the old [PAD<id>] reverse-map entry if present
                auto old_it = vocab.token_to_id.find(vocab.id_to_token[sp.id]);
                if (old_it != vocab.token_to_id.end() && old_it->second == sp.id) {
                    vocab.token_to_id.erase(old_it);
                }
                vocab.id_to_token[sp.id] = sp.text;
                vocab.token_to_id[sp.text] = sp.id;
            }
        }
        // Merges (BPE encode side). Each entry is a "left right" pair string;
        // the ARRAY index is the merge's rank (lowest rank = highest priority).
        auto merges = core_gguf::kv_str_array(gctx, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merges.size(); i++) {
            vocab.merge_rank[merges[i]] = i;
        }

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: tensor data via shared helper ----
    // PLAN #69a: when CRISPASR_N_GPU_LAYERS is set and < total layers,
    // route layers [N..total) onto the CPU backend.
    //
    // HANDOFF-fa-aligner-vram-2 B8: drop `audio.*` tensors here entirely.
    // crisp_audio (audio_ca) loads them separately into its own GPU buffer
    // via load_weights_filtered (B6). Loading them again under qwen3_asr's
    // model.buf duplicated ~176 MiB of Q4_K audio weights for no benefit —
    // the in-tree audio path (model.audio.* graph builders at lines 700+
    // and qwen3_asr_compute_mel's fallback at line 598+) is "tests/
    // fallbacks compile" code per the struct comment, never reached in
    // the production aligner path. The audio struct binding below is
    // wrapped in a try_get to no-op cleanly when the tensors are absent.
    auto drop_audio = +[](const char* tname, void* /*user*/) -> bool {
        return tname && std::strncmp(tname, "audio.", 6) == 0;
    };
    core_gguf::WeightLoad wl;
    int n_gpu_layers_env = -1;
    if (const char* s = std::getenv("CRISPASR_N_GPU_LAYERS")) {
        n_gpu_layers_env = std::atoi(s);
    }
    const int total_layers = (int)model.hparams.llm_n_layers;
    const bool do_split =
        backend_cpu && backend_cpu != backend && n_gpu_layers_env >= 0 && n_gpu_layers_env < total_layers;
    if (do_split) {
        int threshold = n_gpu_layers_env;
        if (!core_gguf::load_weights_split_with_drop(path, backend, backend_cpu, core_gguf::is_gpu_tensor_blk,
                                                     &threshold, drop_audio, nullptr, "qwen3_asr", wl)) {
            return false;
        }
        fprintf(stderr, "qwen3_asr: layer offload: gpu=[0,%d), cpu=[%d,%d) (CRISPASR_N_GPU_LAYERS=%d)\n",
                n_gpu_layers_env, n_gpu_layers_env, total_layers, n_gpu_layers_env);
    } else {
        if (!core_gguf::load_weights_with_drop(path, backend, drop_audio, nullptr, "qwen3_asr", wl)) {
            return false;
        }
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.buf_cpu = wl.buf_cpu;
    model.tensors = std::move(wl.tensors);

    // ---- bind named tensors into the per-layer structs ----
    //
    // The audio.* binding is skipped entirely when audio tensors were
    // dropped at load time (B8). model.audio.* stays default-zero, which
    // is fine because the production aligner path goes through audio_ca
    // (crisp_audio) and never touches model.audio.*. The in-tree fallback
    // graph builders at qwen3_asr_compute_mel / run_conv / run_encoder
    // already guard on `model.audio.mel_filters == nullptr` and return
    // null. Detect-via-presence is simpler than a separate flag.
    auto& a = model.audio;
    const bool have_audio = (try_get(model, "audio.conv.1.weight") != nullptr);
    if (have_audio) {
        a.conv1_w = require(model, "audio.conv.1.weight");
        a.conv1_b = require(model, "audio.conv.1.bias");
        a.conv2_w = require(model, "audio.conv.2.weight");
        a.conv2_b = require(model, "audio.conv.2.bias");
        a.conv3_w = require(model, "audio.conv.3.weight");
        a.conv3_b = require(model, "audio.conv.3.bias");
        a.conv_out_w = require(model, "audio.conv_out.weight");
        a.conv_out_b = try_get(model, "audio.conv_out.bias"); // bias may be absent
        a.ln_post_w = require(model, "audio.ln_post.weight");
        a.ln_post_b = require(model, "audio.ln_post.bias");
        a.proj1_w = require(model, "audio.proj1.weight");
        a.proj1_b = require(model, "audio.proj1.bias");
        a.proj2_w = require(model, "audio.proj2.weight");
        a.proj2_b = require(model, "audio.proj2.bias");
        a.mel_filters = try_get(model, "audio.mel_filters"); // optional (may be missing in older GGUFs)
        a.mel_window = try_get(model, "audio.mel_window");

        a.blocks.resize(model.hparams.audio_n_layers);
        for (uint32_t i = 0; i < model.hparams.audio_n_layers; i++) {
            char buf[128];
            auto& b = a.blocks[i];
            auto get = [&](const char* suf) {
                snprintf(buf, sizeof(buf), "audio.blk.%u.%s", i, suf);
                return require(model, buf);
            };
            b.attn_norm_w = get("attn_norm.weight");
            b.attn_norm_b = get("attn_norm.bias");
            b.attn_q_w = get("attn_q.weight");
            b.attn_q_b = get("attn_q.bias");
            b.attn_k_w = get("attn_k.weight");
            b.attn_k_b = get("attn_k.bias");
            b.attn_v_w = get("attn_v.weight");
            b.attn_v_b = get("attn_v.bias");
            b.attn_out_w = get("attn_out.weight");
            b.attn_out_b = get("attn_out.bias");
            b.ffn_norm_w = get("ffn_norm.weight");
            b.ffn_norm_b = get("ffn_norm.bias");
            b.ffn_up_w = get("ffn_up.weight");
            b.ffn_up_b = get("ffn_up.bias");
            b.ffn_down_w = get("ffn_down.weight");
            b.ffn_down_b = get("ffn_down.bias");
        }
    }

    auto& l = model.llm;
    l.token_embd_w = require(model, "token_embd.weight");
    l.output_norm_w = require(model, "output_norm.weight");
    l.output_w = require(model, "output.weight");
    // Read the actual lm_head output dimension from the loaded tensor
    // shape rather than asserting it equals vocab_size. The standard
    // Qwen3-ASR-{0.6B,1.7B} models have lm_head = (vocab, d), but the
    // Qwen3-ForcedAligner variant has lm_head = (5000, d) — same body,
    // different head. ne[1] is the row count after ggml's [in, out]
    // storage convention.
    model.hparams.llm_lm_head_dim = (uint32_t)l.output_w->ne[1];
    if (model.hparams.llm_lm_head_dim == 0) {
        model.hparams.llm_lm_head_dim = model.hparams.llm_vocab_size;
    }
    l.blocks.resize(model.hparams.llm_n_layers);
    for (uint32_t i = 0; i < model.hparams.llm_n_layers; i++) {
        char buf[128];
        auto& b = l.blocks[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_q_w = get("attn_q.weight");
        b.attn_k_w = get("attn_k.weight");
        b.attn_v_w = get("attn_v.weight");
        b.attn_output_w = get("attn_output.weight");
        b.attn_q_norm_w = get("attn_q_norm.weight");
        b.attn_k_norm_w = get("attn_k_norm.weight");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_gate_w = get("ffn_gate.weight");
        b.ffn_up_w = get("ffn_up.weight");
        b.ffn_down_w = get("ffn_down.weight");
    }

    // ---- precompute sinusoidal positional embedding for the audio encoder ----
    // Reference: SinusoidsPositionEmbedding in modeling_qwen3_asr.py
    //   log_inc = log(10000) / (C/2 - 1)
    //   inv_t   = exp(-log_inc * arange(C/2))
    //   pe[p, :C/2] = sin(p * inv_t)
    //   pe[p, C/2:] = cos(p * inv_t)
    {
        const int C = (int)model.hparams.audio_d_model;
        const int L = (int)model.hparams.audio_max_pos;
        const int half = C / 2;
        const float log_inc = std::log(10000.0f) / (float)(half - 1);
        std::vector<float> inv_t(half);
        for (int i = 0; i < half; i++)
            inv_t[i] = std::exp(-log_inc * (float)i);
        model.audio_pe.assign((size_t)L * C, 0.0f);
        for (int p = 0; p < L; p++) {
            float* row = model.audio_pe.data() + (size_t)p * C;
            for (int i = 0; i < half; i++) {
                float angle = (float)p * inv_t[i];
                row[i] = std::sin(angle);
                row[half + i] = std::cos(angle);
            }
        }
    }

    return true;
}

// ===========================================================================
// FFT (Cooley-Tukey for even sizes, falls back to DFT for odd leaves).
// Handles n_fft=400 (= 2^4 * 25) by recursing down to a 25-point DFT.
// ===========================================================================

static void qwen3_asr_dft(const float* in, int N, float* out) {
    for (int k = 0; k < N; k++) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; n++) {
            float ang = -2.0f * (float)M_PI * (float)k * (float)n / (float)N;
            re += in[n] * std::cos(ang);
            im += in[n] * std::sin(ang);
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

// Real-input FFT, output complex (out has 2*N floats interleaved real/imag).
// in/out are scratch buffers; in must have at least 2*N floats of writable space.
static void qwen3_asr_fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        qwen3_asr_dft(in, N, out);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; i++)
        even[i] = in[2 * i];
    float* even_fft = out + 2 * N;
    qwen3_asr_fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; i++)
        odd[i] = in[2 * i + 1];
    float* odd_fft = even_fft + N;
    qwen3_asr_fft(odd, half_N, odd_fft);

    for (int k = 0; k < half_N; k++) {
        float ang = -2.0f * (float)M_PI * (float)k / (float)N;
        float re = std::cos(ang);
        float im = std::sin(ang);
        float re_odd = odd_fft[2 * k];
        float im_odd = odd_fft[2 * k + 1];
        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + half_N)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

// ===========================================================================
// Whisper-style log-mel spectrogram
//
// Pipeline (matches WhisperFeatureExtractor._np_extract_fbank_features):
//   1. center-pad audio with n_fft/2 zeros on each side
//   2. STFT: hann window length 400, hop 160, n_fft 400 → (n_freqs=201, T)
//   3. power = |STFT|^2
//   4. mel = power @ filters^T → (n_mels=128, T)
//   5. log10(max(mel, 1e-10))
//   6. drop the last frame
//   7. clip: log_spec = max(log_spec, log_spec.max() - 8.0)
//   8. normalize: log_spec = (log_spec + 4) / 4
//
// Returns a flat (n_mels, T) row-major buffer.
// ===========================================================================

#include "core/mel.h"
#include "core/ffn.h"
#include "core/attention.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// qwen3_asr_fft uses its input buffer as scratch during recursion (needs
// ~4N extra floats past the input pointer). Wrap it to match core_mel's
// const-input FftR2C signature, same trick as voxtral / voxtral4b.
static void qwen3_asr_fft_wrapper(const float* in, int N, float* out) {
    static thread_local std::vector<float> scratch_in;
    static thread_local std::vector<float> scratch_out;
    if ((int)scratch_in.size() < 4 * N)
        scratch_in.assign((size_t)4 * N, 0.0f);
    if ((int)scratch_out.size() < 8 * N)
        scratch_out.assign((size_t)8 * N, 0.0f);
    std::memcpy(scratch_in.data(), in, (size_t)N * sizeof(float));
    qwen3_asr_fft(scratch_in.data(), N, scratch_out.data());
    std::memcpy(out, scratch_out.data(), (size_t)(2 * N) * sizeof(float));
}

// Lazily open a crisp_audio context for the audio path. The qwen3-asr GGUF
// uses tensor names under "audio." (the crisp_audio default) and metadata
// under "qwen3asr.audio." (handled by crisp_audio's prefix fallback), so
// passing the defaults here is correct.
static crisp_audio_context* qwen3_asr_get_audio(qwen3_asr_context* ctx) {
    if (!ctx)
        return nullptr;
    if (ctx->audio_ca)
        return ctx->audio_ca;
    if (ctx->model_path.empty())
        return nullptr;
    crisp_audio_params p = crisp_audio_params_default();
    p.n_threads = ctx->n_threads;
    p.verbosity = ctx->params.verbosity;
    p.use_gpu = ctx->params.use_gpu;
    p.tensor_prefix = "audio.";
    p.meta_prefix = "qwen3asr.audio."; // qwen3-asr's hparam namespace
    ctx->audio_ca = crisp_audio_init_from_file(ctx->model_path.c_str(), &p);
    return ctx->audio_ca;
}

extern "C" float* qwen3_asr_compute_mel(qwen3_asr_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                        int* out_T_mel) {
    crisp_audio_context* ca = qwen3_asr_get_audio(ctx);
    if (ca) {
        return crisp_audio_compute_mel(ca, samples, n_samples, out_n_mels, out_T_mel);
    }
    // Fall through to the in-tree implementation below if crisp_audio failed
    // to load (defensive — should not happen on a well-formed qwen3-asr GGUF).
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    if (!ctx->model.audio.mel_filters || !ctx->model.audio.mel_window) {
        fprintf(stderr, "qwen3_asr: model GGUF missing audio.mel_filters / audio.mel_window\n");
        return nullptr;
    }

    const int n_fft = (int)hp.n_fft;    // 400
    const int hop = (int)hp.hop_length; // 160
    const int n_mels = (int)hp.n_mels;  // 128
    const int n_freqs = n_fft / 2 + 1;  // 201

    std::vector<float> hann(n_fft);
    ggml_backend_tensor_get(ctx->model.audio.mel_window, hann.data(), 0, n_fft * sizeof(float));
    std::vector<float> filt((size_t)n_freqs * n_mels);
    ggml_backend_tensor_get(ctx->model.audio.mel_filters, filt.data(), 0, filt.size() * sizeof(float));

    // Qwen3-ASR / Whisper HF feature extractor: log10 + max-clip guard,
    // double-accumulator matmul, drop last STFT frame, fb in (n_freqs, n_mels)
    // layout. No fixed-size padding — output T is whatever the audio yields.
    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Log10;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.norm = core_mel::Normalization::GlobalClipMax;
    p.layout = core_mel::Layout::MelsTime;
    p.fb_layout = core_mel::FbLayout::FreqsMels;
    p.matmul = core_mel::MatmulPrecision::Double;
    p.log_eps = 1e-10f;
    p.center_pad = true;
    p.drop_last_frame = true;

    int T_ret = 0;
    auto mel = core_mel::compute(samples, n_samples, hann.data(), n_fft, filt.data(), n_freqs, qwen3_asr_fft_wrapper, p,
                                 T_ret);

    if (mel.empty())
        return nullptr;

    if (out_n_mels)
        *out_n_mels = n_mels;
    if (out_T_mel)
        *out_T_mel = T_ret;

    float* result = (float*)malloc(mel.size() * sizeof(float));
    std::memcpy(result, mel.data(), mel.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Conv front-end graph (Stage 1)
//
// Input  (set on the CPU side as a contiguous F32 buffer):
//   mel_batched: shape (T_chunk, n_mels, 1, num_chunks)  in ggml ne order
//                = num_chunks chunks of (1, n_mels, T_chunk) per the
//                  reference impl's per-chunk processing
//
// Output:
//   conv_out: shape (audio_d_model, T_chunk_out, num_chunks)
//             = num_chunks chunks of (T_chunk_out, audio_d_model) frames
//
// Each chunk is processed independently through:
//   conv2d1 + bias + GELU      (in_ch=1,   out_ch=480, k=3, stride=2, pad=1)
//   conv2d2 + bias + GELU      (in_ch=480, out_ch=480, k=3, stride=2, pad=1)
//   conv2d3 + bias + GELU      (in_ch=480, out_ch=480, k=3, stride=2, pad=1)
//   permute + flatten freq → (num_chunks * T_chunk_out, 480 * F_out)
//   conv_out linear (480*16=7680 → 896) + optional bias
//
// For our reference test on jfk.wav:
//   T_chunk=100, n_mels=128, num_chunks=11 → conv1: (50,64,480), conv2:
//   (25,32,480), conv3: (13,16,480) → flatten: (13, 7680) → linear: (13, 896)
// ===========================================================================

static const float kLayerNormEps = 1e-5f;

static ggml_cgraph* qwen3_asr_build_graph_conv(qwen3_asr_context* ctx, int T_chunk, int num_chunks) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // Input: ggml 2D conv expects (W, H, C, N) where ne[0]=W (fast), ne[3]=N
    // For per-chunk processing of (1, n_mels=128, T_chunk=100):
    //   ne[0] = T_chunk  (time, varies fastest)
    //   ne[1] = n_mels   (frequency)
    //   ne[2] = 1        (in channels)
    //   ne[3] = num_chunks (batch)
    ggml_tensor* mel = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T_chunk, n_mels, 1, num_chunks);
    ggml_set_name(mel, "mel_batched");
    ggml_set_input(mel);

    auto bias_4d = [&](ggml_context* c0, ggml_tensor* b) {
        // bias is (out_ch,) — broadcast as (1, 1, out_ch, 1) for elementwise add
        return ggml_cast(c0, ggml_reshape_4d(c0, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32);
    };

    // Conv1: in=1, out=480, k=3, stride=2, pad=1
    ggml_tensor* cur = ggml_conv_2d(ctx0, m.audio.conv1_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);

    // Conv2: in=480, out=480, k=3, stride=2, pad=1
    cur = ggml_conv_2d(ctx0, m.audio.conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);

    // Conv3: in=480, out=480, k=3, stride=2, pad=1
    cur = ggml_conv_2d(ctx0, m.audio.conv3_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.audio.conv3_b));
    cur = ggml_gelu_erf(ctx0, cur);

    // After conv3: shape (T_out, F_out, 480, num_chunks)
    // For T_chunk=100, n_mels=128: T_out=13, F_out=16
    const int T_out = (int)cur->ne[0];
    const int F_out = (int)cur->ne[1];
    const int C_out = (int)cur->ne[2]; // 480
    GGML_ASSERT(C_out == (int)hp.audio_conv_ch);

    // Reference does: padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c*f)
    // PyTorch shape (B, C, F, T) → permute(0, 3, 1, 2) → (B, T, C, F) → flatten last two
    // Our ggml shape is (T, F, C, B). We want (T, C*F, B) where C*F is contiguous so
    // that the linear in conv_out (which expects 7680 input dim) gets the right
    // memory layout. PyTorch's view(b, t, c*f) over (B, T, C, F) means C is the
    // outer index, F is the inner index → memory order: f0c0, f1c0, ..., f15c0,
    // f0c1, ... = (F + F*C). To match, our final layout should be (F + F*C) along
    // the fast axis = ne[0] = F*C with inner stride F.
    //
    // Currently: ne = (T, F, C, B). We want ne = (F*C, T, B) with C as inner.
    // Permute to put C inner: (T, F, C, B) → (C, F, T, B)? No, we want C inner of F.
    // Let's permute so axes order becomes (C, F, T, B) — then C is fast (ne[0]),
    // F is next (ne[1]), so memory is c0f0, c1f0, ..., c479f0, c0f1, ...
    // That's the order f outer, c inner = c + C*f. Reshape (C*F, T, B) gives us
    // (c+C*f, t, b) — which equals PyTorch's (b, t, c*F + f) — wait that's NOT
    // what PyTorch does. PyTorch's view(b, t, c*f) treats it as a flat dim where
    // PyTorch's prior layout was (B, T, C, F) → memory: t outer, c middle, f inner
    // → flat index along last dim = c*F + f. So fast-axis index = c*F + f, with
    // c outer and f inner. Our target ggml memory is therefore (f + F*c) along
    // ne[0]. Permute (T, F, C, B) → axes (1, 2, 0, 3): puts F at ne[0], C at ne[1].
    // Memory order: f0c0, f1c0, ..., F-1 c0, f0c1, ... = (f + F*c). YES.
    //
    // ggml_permute(t, p0, p1, p2, p3) semantics: source axis i goes to NEW
    // position p_i. So to get new ne = (F, C, T, B) from source (T, F, C, B):
    //   source 0 (T) → new pos 2  → p0 = 2
    //   source 1 (F) → new pos 0  → p1 = 0
    //   source 2 (C) → new pos 1  → p2 = 1
    //   source 3 (B) → new pos 3  → p3 = 3
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
    cur = ggml_reshape_3d(ctx0, cur, F_out * C_out, T_out, num_chunks);

    // Linear: cur is (F*C, T, B) = (7680, 13, 11). conv_out_w is stored as
    // ggml shape (7680, 896) — i.e. ne[0]=7680, ne[1]=896. ggml_mul_mat(A, B)
    // computes B^T @ A^T with output ne[0] = A->ne[1], ne[1] = B->ne[1].
    // We want output (896, T, B). With cur as B (7680, T*B effectively), and
    // mul_mat(conv_out_w, cur): output ne[0] = conv_out_w->ne[1] = 896,
    // ne[1..] inherit from cur. ✓
    cur = ggml_mul_mat(ctx0, m.audio.conv_out_w, cur);
    if (m.audio.conv_out_b) {
        cur = ggml_add(ctx0, cur, m.audio.conv_out_b);
    }
    // cur shape now: (896, T_out, num_chunks)

    ggml_set_name(cur, "conv_front_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Full encoder graph (Stage 2)
//
// Pipeline (matching modeling_qwen3_asr.Qwen3ASRAudioEncoder.forward):
//   1. Per-chunk Conv2D subsampler  → (896, 13, num_chunks)        [as Stage 1]
//   2. Add sinusoidal pos embed (broadcast over chunks)            → same shape
//   3. Reshape to flat (896, N_padded) where N_padded = 13*num_chunks
//      [Stage-2 simplification: assumes all chunks are full, no padding mask]
//   4. 18 × Whisper-style pre-LN encoder block:
//        residual = x
//        x = LN1(x)
//        Q,K,V = x @ {Wq,Wk,Wv} + bias
//        attn = softmax((Q @ K^T)/sqrt(hd) + window_mask) @ V
//        x = residual + Wo @ attn + bo
//        residual = x
//        x = LN2(x); x = GELU(W1 x + b1); x = W2 x + b2
//        x = residual + x
//   5. ln_post → proj1 → GELU → proj2  →  (1024, N_padded)
//
// The "window_mask" implements the chunked attention from the reference:
// each position only attends within its window of size 104. The mask is
// supplied as an input tensor (N_padded, N_padded) F32 with -inf in
// disallowed positions and 0 in allowed ones.
// ===========================================================================

static ggml_cgraph* qwen3_asr_build_graph_encoder(qwen3_asr_context* ctx, int T_chunk, int num_chunks,
                                                  int T_chunk_out_expected) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;
    const int d = (int)hp.audio_d_model;         // 896
    const int n_heads = (int)hp.audio_n_heads;   // 14
    const int head_dim = (int)hp.audio_head_dim; // 64
    const int proj_dim = (int)hp.audio_proj_dim; // 1024

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // ------- Inputs -------
    // mel_batched ne = (T_chunk, n_mels, 1, num_chunks)
    ggml_tensor* mel = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T_chunk, n_mels, 1, num_chunks);
    ggml_set_name(mel, "mel_batched");
    ggml_set_input(mel);

    // pe_input ne = (d, T_chunk_out, 1, 1)  — broadcasts over batch
    ggml_tensor* pe_in = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d, T_chunk_out_expected, 1);
    ggml_set_name(pe_in, "pe_input");
    ggml_set_input(pe_in);

    // attn_mask ne = (N_padded, N_padded)  F32 with 0 / -inf
    const int N_padded = T_chunk_out_expected * num_chunks;
    ggml_tensor* mask_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, N_padded, N_padded);
    ggml_set_name(mask_in, "attn_mask");
    ggml_set_input(mask_in);

    // ------- Conv front-end (same as Stage 1) -------
    auto bias_4d = [&](ggml_context* c0, ggml_tensor* b) {
        return ggml_cast(c0, ggml_reshape_4d(c0, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32);
    };

    ggml_tensor* cur = ggml_conv_2d(ctx0, m.audio.conv1_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_conv_2d(ctx0, m.audio.conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_conv_2d(ctx0, m.audio.conv3_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.audio.conv3_b));
    cur = ggml_gelu_erf(ctx0, cur);
    // cur ne = (T_out, F_out, 480, num_chunks)
    const int T_out = (int)cur->ne[0];
    const int F_out = (int)cur->ne[1];
    const int C_out = (int)cur->ne[2];
    GGML_ASSERT(T_out == T_chunk_out_expected);

    // Permute (T,F,C,B) → (F,C,T,B): source axis 0(T)→pos 2, 1(F)→0, 2(C)→1, 3(B)→3
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
    cur = ggml_reshape_3d(ctx0, cur, F_out * C_out, T_out, num_chunks);
    cur = ggml_mul_mat(ctx0, m.audio.conv_out_w, cur); // (d, T_out, num_chunks)

    // ------- Add positional embedding (broadcasts over batch) -------
    // pe_in ne = (d, T_out, 1) → broadcast against (d, T_out, num_chunks)
    cur = ggml_add(ctx0, cur, pe_in);

    // ------- Flatten chunks into a single sequence -------
    // cur ne = (d, T_out, num_chunks). Want (d, N_padded). Memory layout for
    // (d, T_out, num_chunks) row-major (d fastest) is identical to
    // (d, N_padded=T_out*num_chunks) where chunk-major order is preserved.
    // ggml_reshape_2d just relabels strides.
    cur = ggml_cont(ctx0, cur); // ensure contiguous before reshape
    cur = ggml_reshape_2d(ctx0, cur, d, N_padded);

    // ------- 18 × encoder blocks -------
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);
    for (uint32_t il = 0; il < hp.audio_n_layers; il++) {
        const auto& b = m.audio.blocks[il];
        ggml_tensor* residual = cur;

        // ---- LN1 (pre-attention) ----
        ggml_tensor* x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);
        x = ggml_add(ctx0, x, b.attn_norm_b);

        // ---- Q, K, V projections (with biases) ----
        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, x), b.attn_q_b);
        ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_k_w, x), b.attn_k_b);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, x), b.attn_v_b);
        // Q/K/V ne = (d, N_padded). Reshape to (head_dim, n_heads, N_padded),
        // then permute to (head_dim, N_padded, n_heads).
        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, N_padded);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_heads, N_padded);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_heads, N_padded);
        // Permute (hd, n_h, N) → (hd, N, n_h): source 0→0, 1→2, 2→1
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        // V layout for the attn @ V step: we want V as (hd, N, n_h) too, then
        // reshape later. Use the same permute.
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        // ---- Scores = (Q @ K^T) ----
        // ggml_mul_mat(K, Q): K ne=(hd, N, n_h), Q ne=(hd, N, n_h)
        // result ne = (N, N, n_h) where result[j, i, h] = dot(K[:, j, h], Q[:, i, h])
        // So result[j, i, h] = sum_d K[d,j,h] * Q[d,i,h] = (Q @ K^T)[i, j, h]
        // ne[0]=j (key index, varies fast), ne[1]=i (query index)
        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);

        // Add window mask. mask_in ne=(N, N) F32. ggml_add broadcasts over the
        // n_heads dim (size 1 in mask, size n_h in scores).
        scores = ggml_add(ctx0, scores, mask_in);

        // Softmax along key axis (ne[0]) with scale baked in.
        scores = ggml_soft_max_ext(ctx0, scores, /*mask*/ nullptr, attn_scale, 0.0f);

        // ---- attn = scores @ V ----
        // We need: out[d, i, h] = sum_j scores[j, i, h] * V[d, j, h]
        // ggml_mul_mat(V_perm, scores) where V_perm is (j, d, h) so dot is over j.
        // Currently V ne=(hd, N, n_h). We want V indexed as (j, d, h) with j fast.
        // Permute V (hd, N, n_h) → (N, hd, n_h): source 0→1, 1→0, 2→2
        ggml_tensor* V2 = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
        // ggml_mul_mat(V2, scores): V2 ne=(N, hd, n_h), scores ne=(N, N, n_h)
        // dot over ne[0]=N (the j axis). Result ne=(hd, N, n_h) where result[d, i, h]
        // = sum_j V2[j, d, h] * scores[j, i, h] = sum_j V[d, j, h] * scores[j, i, h] ✓
        ggml_tensor* attn = ggml_mul_mat(ctx0, V2, scores);
        // attn ne=(hd, N, n_h). Permute back to (hd, n_h, N) and reshape (d, N).
        // src 0(hd)→0, 1(N)→2, 2(n_h)→1
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, d, N_padded);

        // ---- Output projection (with bias) ----
        attn = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_out_w, attn), b.attn_out_b);
        cur = ggml_add(ctx0, residual, attn);

        // ---- LN2 + FFN ----
        residual = cur;
        x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        x = ggml_add(ctx0, x, b.ffn_norm_b);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_up_w, x), b.ffn_up_b);
        x = ggml_gelu_erf(ctx0, x);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_down_w, x), b.ffn_down_b);
        cur = ggml_add(ctx0, residual, x);
    }

    // ------- ln_post → proj1 → GELU → proj2 -------
    {
        ggml_tensor* x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, m.audio.ln_post_w);
        x = ggml_add(ctx0, x, m.audio.ln_post_b);
        cur = x;
    }
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.audio.proj1_w, cur), m.audio.proj1_b);
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.audio.proj2_w, cur), m.audio.proj2_b);
    // cur ne = (proj_dim=1024, N_padded)
    (void)proj_dim;

    ggml_set_name(cur, "encoder_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Qwen3 LLM forward graph (Stage 3)
//
// Architecture: 28 layers, hidden=1024, GQA(16/8), head_dim=128, RMSNorm,
//   SwiGLU FFN, RoPE θ=1e6 NEOX-style, Q-norm/K-norm per-head along head_dim.
//
// Pipeline:
//   x = embed(input_ids)              # (1024, T)
//   for layer in 28 layers:
//     residual = x
//     x = RMSNorm(x) * attn_norm_w
//     Q = q_proj(x).view(head_dim, n_q,  T)
//     K = k_proj(x).view(head_dim, n_kv, T)
//     V = v_proj(x).view(head_dim, n_kv, T)
//     Q = q_norm(Q) along head_dim
//     K = k_norm(K) along head_dim
//     Q = rope_neox(Q, positions)
//     K = rope_neox(K, positions)
//     # GQA: repeat K, V from n_kv to n_q heads
//     K_rep = repeat_each(K, n_q / n_kv)   # (head_dim, n_q, T)
//     V_rep = repeat_each(V, n_q / n_kv)
//     # Standard attention
//     scores = (Q @ K_rep^T) * (1/sqrt(head_dim)) + causal_mask
//     attn   = softmax(scores) @ V_rep
//     attn   = o_proj(attn.reshape(d, T))
//     x = residual + attn
//     residual = x
//     x = RMSNorm(x) * ffn_norm_w
//     x = down_proj(silu(gate_proj(x)) * up_proj(x))
//     x = residual + x
//   x = RMSNorm(x) * output_norm_w
//   logits = lm_head(x)               # (vocab, T)
//
// First iteration: no KV cache, full forward each call. Used for diff testing.
// ===========================================================================

// Internal: builds the 28-layer transformer + lm_head graph starting from
// a (d, T) hidden state. Used by both build_graph_llm (which prepends a
// get_rows token-embed lookup) and build_graph_llm_from_embeds (which takes
// pre-computed embeddings as input).
static void qwen3_asr_build_llm_body(qwen3_asr_context* ctx, ggml_context* ctx0, ggml_cgraph* gf,
                                     ggml_tensor* cur, // (d, T) input hidden state
                                     ggml_tensor* positions, ggml_tensor* causal_mask, int T) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.llm_d_model;
    const int n_q = (int)hp.llm_n_heads;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.llm_rms_eps;
    const float theta = hp.llm_rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    for (uint32_t il = 0; il < hp.llm_n_layers; il++) {
        const auto& b = m.llm.blocks[il];
        ggml_tensor* residual = cur;

        // ---- LN1 (RMSNorm + multiplicative weight, no bias) ----
        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // ---- Q, K, V projections ----
        ggml_tensor* Q = ggml_mul_mat(ctx0, b.attn_q_w, x);
        ggml_tensor* K = ggml_mul_mat(ctx0, b.attn_k_w, x);
        ggml_tensor* V = ggml_mul_mat(ctx0, b.attn_v_w, x);
        Q = ggml_reshape_3d(ctx0, Q, hd, n_q, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_kv, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_kv, T);

        // ---- Q-norm / K-norm ----
        Q = ggml_rms_norm(ctx0, Q, eps);
        Q = ggml_mul(ctx0, Q, b.attn_q_norm_w);
        K = ggml_rms_norm(ctx0, K, eps);
        K = ggml_mul(ctx0, K, b.attn_k_norm_w);

        // ---- RoPE NEOX ----
        Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos, theta, 1.0f, 0.0f,
                          1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos, theta, 1.0f, 0.0f,
                          1.0f, 32.0f, 1.0f);

        // ---- GQA expand ----
        if (n_kv_grp > 1) {
            ggml_tensor* K4 = ggml_reshape_4d(ctx0, K, hd, 1, n_kv, T);
            ggml_tensor* V4 = ggml_reshape_4d(ctx0, V, hd, 1, n_kv, T);
            K4 = ggml_repeat_4d(ctx0, K4, hd, n_kv_grp, n_kv, T);
            V4 = ggml_repeat_4d(ctx0, V4, hd, n_kv_grp, n_kv, T);
            K = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, n_q, T));
            V = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, n_q, T));
        }

        // ---- Permute for attention ----
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);
        scores = ggml_add(ctx0, scores, causal_mask);
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, attn_scale, 0.0f);

        ggml_tensor* V2 = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_mul_mat(ctx0, V2, scores);
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);

        attn = ggml_mul_mat(ctx0, b.attn_output_w, attn);
        cur = ggml_add(ctx0, residual, attn);

        // ---- FFN ----
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    (void)d;
}

static ggml_cgraph* qwen3_asr_build_graph_llm(qwen3_asr_context* ctx, int n_tokens) {
    const auto& m = ctx->model;
    const int T = n_tokens;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // ------- Inputs -------
    ggml_tensor* input_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(input_ids, "input_ids");
    ggml_set_input(input_ids);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    // Token embedding lookup → (d, T)
    ggml_tensor* cur = ggml_get_rows(ctx0, m.llm.token_embd_w, input_ids);
    qwen3_asr_build_llm_body(ctx, ctx0, gf, cur, positions, causal_mask, T);
    ggml_free(ctx0);
    return gf;
}

// Variant: takes pre-computed inputs_embeds (d, T) F32 instead of input_ids.
// Used by the audio-injection path after splicing audio frames into the
// text-token embedding sequence.
static ggml_cgraph* qwen3_asr_build_graph_llm_from_embeds(qwen3_asr_context* ctx, int n_tokens) {
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int T = n_tokens;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    qwen3_asr_build_llm_body(ctx, ctx0, gf, embeds, positions, causal_mask, T);
    ggml_free(ctx0);
    return gf;
}

// Graph builder for the KV-cached LLM forward. Used by both prefill
// (n_past=0, n_tokens=T_prompt) and incremental decode (n_past>0, n_tokens=1).
//
// Inputs:
//   inputs_embeds: F32 (d, n_tokens)
//   positions:     I32 (n_tokens,) — absolute positions n_past, n_past+1, ...
//   causal_mask:   F32 (n_kv_total, n_tokens) where n_kv_total = n_past+n_tokens
//                  mask[k, q] = 0 if k <= n_past+q else -inf
//
// Per layer, the new K/V are written into the persistent cache at positions
// [n_past, n_past+n_tokens) and attention reads from [0, n_past+n_tokens).
static ggml_cgraph* qwen3_asr_build_graph_llm_kv(qwen3_asr_context* ctx, int n_past, int n_tokens,
                                                 bool last_token_only = true) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.llm_d_model;
    const int n_q = (int)hp.llm_n_heads;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.llm_rms_eps;
    const float theta = hp.llm_rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = n_past + T; // total cache length after this call

    GGML_ASSERT(ctx->kv_k && ctx->kv_v);
    GGML_ASSERT(Lk <= ctx->kv_max_ctx);

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Only the prefill path (T > 1) needs the causal mask. Decode (T = 1)
    // uses ggml_flash_attn_ext with no mask — the single new query attends
    // to all cached keys including itself. If we always declared the mask
    // input, the scheduler would optimize it away on the decode path and
    // ggml_graph_get_tensor("causal_mask") would return null.
    //
    // For prefill we use flash_attn_ext too, which requires the mask to
    // be F16 (and contiguous, broadcast-compatible with Q's trailing dims).
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    ggml_tensor* cur = embeds;

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.llm_max_pos,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 32.0f,
        /*rope_beta_slow*/ 1.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    for (uint32_t il = 0; il < hp.llm_n_layers; il++) {
        const auto& b = m.llm.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // KV-cached GQA self-attention with Q/K norm — qwen3 is the only
        // backend currently passing non-null q_norm_w/k_norm_w to the
        // shared helper.
        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w, b.attn_q_norm_w, b.attn_k_norm_w,
            positions, (T == 1) ? nullptr : causal_mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, kvp, b.attn_qkv_w);
        cur = ggml_add(ctx0, residual, attn);

        // ---- FFN ----
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);

    // Last-token-only lm_head (default): slice (d, T) → (d, 1) before
    // the big matmul. The autoregressive decode loop only ever needs the
    // next-token logits, so we save the (152064, 2048) matmul on T-1
    // columns.
    //
    // Forced-alignment mode (last_token_only=false) keeps all T columns
    // because the FA model needs the lm_head output at every position
    // where input == <|timestamp|>. The lm_head shape itself is read
    // from output_w (5000 for FA, vocab_size for ASR).
    if (last_token_only && T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);
    // logits ne = (lm_head_dim, 1)        (decode / last-token mode)
    //         or (lm_head_dim, T)         (FA / full-T mode)

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Tiny standalone graph for token embedding lookup (used by run_embed_tokens).
static ggml_cgraph* qwen3_asr_build_graph_embed(qwen3_asr_context* ctx, int n_tokens) {
    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids");
    ggml_set_input(ids);
    ggml_tensor* out = ggml_get_rows(ctx0, ctx->model.llm.token_embd_w, ids);
    ggml_set_name(out, "embeds");
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" const char* qwen3_asr_token_text(qwen3_asr_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[id].c_str();
}

// ===========================================================================
// BPE tokenizer (GPT-2 byte-level, Qwen2/Qwen3 compatible)
//
// Two-stage encode:
//   1. Pre-tokenize: split the input on `<|special|>` markers and on
//      whitespace boundaries. Special tokens are looked up directly in the
//      vocab as full strings; the remaining text segments are byte-encoded
//      via the GPT-2 byte→unicode mapping and then BPE-merged.
//   2. BPE merge loop: for each pre-token, find the lowest-rank merge in
//      the merge table and apply it, repeating until no merges apply.
//      The final symbols are then looked up in the vocab.
// ===========================================================================

// All four building blocks (byte_encoder, utf8_encode, bytes_to_unicode,
// bpe_one) used to live inline here as `qwen3_*` statics. They now live
// once in src/core/bpe.h — qwen3 calls into them via the namespace alias
// below, granite uses the same primitives, and any future GPT-2-family
// model gets them for free.
#include "core/bpe.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
extern "C" int32_t* qwen3_asr_tokenize(qwen3_asr_context* ctx, const char* text, int* out_n_tokens) {
    if (!ctx || !text) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return nullptr;
    }
    const auto& v = ctx->vocab;
    std::vector<int32_t> result;

    const std::string s = text;
    size_t i = 0;
    while (i < s.size()) {
        // 1. Special-token check: if the next chars are "<|...|>" and the
        //    full token exists in the vocab, emit it directly.
        if (s[i] == '<' && i + 1 < s.size() && s[i + 1] == '|') {
            size_t end = s.find("|>", i + 2);
            if (end != std::string::npos) {
                std::string special = s.substr(i, end + 2 - i);
                auto it = v.token_to_id.find(special);
                if (it != v.token_to_id.end()) {
                    result.push_back(it->second);
                    i = end + 2;
                    continue;
                }
            }
        }

        // 2. Plain text segment: collect chars up to the next "<|...|>" we
        //    can recognize. We treat a "<|" as a candidate boundary only if
        //    it's an actual special token we know — otherwise we keep
        //    extending the plain-text segment past it (a literal "<|" in
        //    user text isn't a special token).
        size_t j = i;
        // Ensure we always advance by at least one char on every outer
        // iteration, even if step 1 just failed on a "<|...|>" lookalike
        // that isn't actually a special token.
        if (s[j] == '<' && j + 1 < s.size() && s[j + 1] == '|')
            j++;
        while (j < s.size()) {
            if (s[j] == '<' && j + 1 < s.size() && s[j + 1] == '|') {
                size_t end = s.find("|>", j + 2);
                if (end != std::string::npos) {
                    std::string special = s.substr(j, end + 2 - j);
                    if (v.token_to_id.find(special) != v.token_to_id.end())
                        break;
                }
            }
            j++;
        }
        std::string chunk = s.substr(i, j - i);
        i = j;
        if (chunk.empty())
            continue;

        // 3. Pre-split the chunk on whitespace boundaries the way GPT-2
        //    does: each pre-token starts at a non-space char and includes
        //    the leading space if present (Ġ marker). We approximate this
        //    by splitting on transitions between "leading space + word",
        //    "word", "punctuation", etc.
        size_t k = 0;
        while (k < chunk.size()) {
            size_t start = k;
            // Optional leading whitespace (single char)
            if (chunk[k] == ' ' || chunk[k] == '\t' || chunk[k] == '\n')
                k++;
            // The "word body" — everything until the next whitespace OR a
            // standalone punctuation transition.
            while (k < chunk.size() && chunk[k] != ' ' && chunk[k] != '\t' && chunk[k] != '\n') {
                k++;
            }
            if (k == start)
                k++; // pure whitespace boundary, advance one
            std::string pre(chunk, start, k - start);
            // 4. Byte-encode and BPE-merge via the shared helpers.
            std::string encoded = core_bpe::bytes_to_unicode(pre.data(), pre.size());
            core_bpe::bpe_one(v.token_to_id, v.merge_rank, encoded, result);
        }
    }

    if (out_n_tokens)
        *out_n_tokens = (int)result.size();
    int32_t* out = (int32_t*)malloc(result.size() * sizeof(int32_t));
    if (!out) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return nullptr;
    }
    std::memcpy(out, result.data(), result.size() * sizeof(int32_t));
    return out;
}

extern "C" qwen3_asr_context_params qwen3_asr_context_default_params(void) {
    qwen3_asr_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

extern "C" qwen3_asr_context* qwen3_asr_init_from_file(const char* path, qwen3_asr_context_params params) {
    qwen3_asr_context* ctx = new qwen3_asr_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    if (path)
        ctx->model_path = path;

    // Try GPU backend first (Metal, CUDA, Vulkan...), fall back to CPU.
    // ggml_backend_init_best() picks the highest-priority available backend.
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu) {
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    }
    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    }

    if (!qwen3_asr_load_model(ctx->model, ctx->vocab, path, ctx->backend, ctx->backend_cpu)) {
        delete ctx;
        return nullptr;
    }

    // ---- Fuse Q+K+V weights for single-matmul LLM attention ----
    // PLAN #60d: type gate dropped May 2026 — concatenating along the
    // output axis is byte-concat for any row-wise quantized format
    // (Q4_K, Q4_0, Q5_K, Q8_0, ...) just as it is for F16/F32.
    // Each output row is a self-contained block group; no requantization.
    // Buffer allocation switched from CPU to default-backend buffer:
    // for Q-format weights on Metal the CPU-buffer path would pay a
    // backend-transfer cost per matmul. See LEARNINGS § "runtime
    // QKV/MLP fusion on row-wise quantized weights is just byte-concat".
    // Opt-out: CRISPASR_QWEN3_ASR_FUSED_QKV=0.
    {
        const char* fuse_env = getenv("CRISPASR_QWEN3_ASR_FUSED_QKV");
        const bool fuse_enabled = (fuse_env == nullptr) || (atoi(fuse_env) != 0);
        auto& hp = ctx->model.hparams;
        auto& blocks = ctx->model.llm.blocks;
        bool can_fuse =
            fuse_enabled && !blocks.empty() && blocks[0].attn_q_w && blocks[0].attn_k_w && blocks[0].attn_v_w;
        if (can_fuse) {
            const ggml_type t0 = blocks[0].attn_q_w->type;
            for (auto& b : blocks) {
                if (!b.attn_q_w || !b.attn_k_w || !b.attn_v_w || b.attn_q_w->type != t0 || b.attn_k_w->type != t0 ||
                    b.attn_v_w->type != t0 || b.attn_q_w->ne[0] != b.attn_k_w->ne[0] ||
                    b.attn_q_w->ne[0] != b.attn_v_w->ne[0]) {
                    can_fuse = false;
                    break;
                }
            }
        }
        if (can_fuse) {
            int q_out = (int)blocks[0].attn_q_w->ne[1];
            int k_out = (int)blocks[0].attn_k_w->ne[1];
            int hidden = (int)blocks[0].attn_q_w->ne[0];
            int qkv_out = q_out + 2 * k_out;
            size_t fused_mem = ggml_tensor_overhead() * blocks.size() + 256;
            ggml_init_params fgp = {fused_mem, nullptr, true};
            ctx->fused_ctx = ggml_init(fgp);
            if (ctx->fused_ctx) {
                for (auto& b : blocks) {
                    b.attn_qkv_w = ggml_new_tensor_2d(ctx->fused_ctx, b.attn_q_w->type, hidden, qkv_out);
                }
                ctx->fused_buf = ggml_backend_alloc_ctx_tensors_from_buft(
                    ctx->fused_ctx, ggml_backend_get_default_buffer_type(ctx->backend));
                if (ctx->fused_buf) {
                    for (auto& b : blocks) {
                        size_t qb = ggml_nbytes(b.attn_q_w), kb = ggml_nbytes(b.attn_k_w);
                        std::vector<uint8_t> tmp(qb + 2 * kb);
                        ggml_backend_tensor_get(b.attn_q_w, tmp.data(), 0, qb);
                        ggml_backend_tensor_get(b.attn_k_w, tmp.data() + qb, 0, kb);
                        ggml_backend_tensor_get(b.attn_v_w, tmp.data() + qb + kb, 0, kb);
                        ggml_backend_tensor_set(b.attn_qkv_w, tmp.data(), 0, tmp.size());
                    }
                    if (params.verbosity >= 1)
                        fprintf(stderr, "qwen3_asr: fused QKV for %zu LLM layers (%d+%d+%d→%d, type=%s)\n",
                                blocks.size(), q_out, k_out, k_out, qkv_out, ggml_type_name(blocks[0].attn_q_w->type));
                } else {
                    ggml_free(ctx->fused_ctx);
                    ctx->fused_ctx = nullptr;
                    for (auto& b : blocks)
                        b.attn_qkv_w = nullptr;
                }
            }
        }
    }

    // Create the backend scheduler once with the worst-case node budget.
    // All compute functions reuse this scheduler via ggml_backend_sched_reset().
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        fprintf(stderr, "qwen3_asr: loaded %s  (audio %u layers, llm %u layers, vocab %u)\n", path,
                ctx->model.hparams.audio_n_layers, ctx->model.hparams.llm_n_layers,
                (uint32_t)ctx->vocab.id_to_token.size());
    }
    return ctx;
}

extern "C" void qwen3_asr_free(qwen3_asr_context* ctx) {
    if (!ctx)
        return;
    if (ctx->audio_ca) {
        crisp_audio_free(ctx->audio_ca);
        ctx->audio_ca = nullptr;
    }
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->fused_buf)
        ggml_backend_buffer_free(ctx->fused_buf);
    if (ctx->fused_ctx)
        ggml_free(ctx->fused_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.buf_cpu)
        ggml_backend_buffer_free(ctx->model.buf_cpu);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    // Free the primary backend last — buffers above were allocated against it,
    // and on Metal an unreleased backend leaves the residency set live and
    // trips ggml_metal_rsets_free's assert at process exit.
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" char* qwen3_asr_transcribe(qwen3_asr_context* /*ctx*/, const float* /*samples*/, int /*n_samples*/) {
    // Stage 1: not yet implemented end-to-end. Use qwen3_asr_run_conv for now.
    return strdup("");
}

extern "C" float* qwen3_asr_run_conv(qwen3_asr_context* ctx, const float* mel_features, int n_mels, int T_mel,
                                     int* out_n_chunks, int* out_T_chunk_out, int* out_d) {
    if (!ctx || !mel_features)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    if (n_mels != (int)hp.n_mels) {
        fprintf(stderr, "qwen3_asr: mel feature mismatch (%d vs %d)\n", n_mels, (int)hp.n_mels);
        return nullptr;
    }

    // Chunking: split T_mel into chunks of n_window*2. The final chunk is
    // padded with zeros to n_window*2 to match the reference impl, which
    // pad_sequences chunks before batching them through the convs.
    const int chunk_T = (int)hp.n_window * 2; // 100
    const int num_chunks = (T_mel + chunk_T - 1) / chunk_T;

    // Build (T_chunk=100, n_mels=128, 1, num_chunks) F32 buffer, padded with zeros.
    std::vector<float> mel_padded((size_t)chunk_T * n_mels * num_chunks, 0.0f);
    // Source layout: mel_features is (n_mels, T_mel), row-major (mel as outer,
    // time as inner). Per the reference dump it's saved that way.
    // Target ggml layout: ne[0]=T_chunk varies fastest, ne[1]=n_mels, ne[3]=batch.
    // Memory index: t + chunk_T*(f + n_mels*chunk_idx)
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int t_start = chunk * chunk_T;
        const int t_end = std::min(t_start + chunk_T, T_mel);
        const int t_len = t_end - t_start;
        for (int f = 0; f < n_mels; f++) {
            for (int t = 0; t < t_len; t++) {
                // src: (f, t_start + t) — mel_features[f * T_mel + (t_start + t)]
                // dst: (t, f, 0, chunk) — mel_padded[t + chunk_T*(f + n_mels*chunk)]
                mel_padded[(size_t)t + chunk_T * ((size_t)f + n_mels * (size_t)chunk)] =
                    mel_features[(size_t)f * T_mel + (size_t)(t_start + t)];
            }
        }
    }

    ggml_cgraph* gf = qwen3_asr_build_graph_conv(ctx, chunk_T, num_chunks);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc conv graph\n");
        return nullptr;
    }

    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel_batched");
    ggml_backend_tensor_set(mel_in, mel_padded.data(), 0, mel_padded.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: conv graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "conv_front_out");
    if (!out) {
        fprintf(stderr, "qwen3_asr: missing conv_front_out tensor\n");
        return nullptr;
    }
    const int d = (int)out->ne[0]; // 896
    const int T = (int)out->ne[1]; // 13
    const int B = (int)out->ne[2]; // num_chunks
    if (out_n_chunks)
        *out_n_chunks = B;
    if (out_T_chunk_out)
        *out_T_chunk_out = T;
    if (out_d)
        *out_d = d;

    const size_t total = (size_t)d * T * B;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" float* qwen3_asr_run_encoder(qwen3_asr_context* ctx, const float* mel_features, int n_mels, int T_mel,
                                        int* out_N_total, int* out_proj_dim) {
    if (!ctx || !mel_features)
        return nullptr;
    crisp_audio_context* ca = qwen3_asr_get_audio(ctx);
    if (ca) {
        return crisp_audio_encode(ca, mel_features, n_mels, T_mel, out_N_total, out_proj_dim);
    }
    // Fall through to the in-tree implementation below if crisp_audio failed.
    const auto& hp = ctx->model.hparams;
    if (n_mels != (int)hp.n_mels) {
        fprintf(stderr, "qwen3_asr: mel feature mismatch (%d vs %d)\n", n_mels, (int)hp.n_mels);
        return nullptr;
    }

    // Chunking. Round T_mel up to the nearest multiple of chunk_T = 100 and
    // zero-pad the trailing partial chunk. The padding shows up as "silence"
    // encoder frames at the end of the sequence; the LLM handles them
    // naturally (it's trained on audio with silence). For long audio there
    // are typically only 0..99 padding frames out of thousands.
    const int chunk_T = (int)hp.n_window * 2;
    const int num_chunks = (T_mel + chunk_T - 1) / chunk_T;

    // After three stride-2 convs the time dim shrinks by 8 (with rounding).
    // Reference: 100 → 50 → 25 → 13.
    auto conv_out_len = [](int in_len) {
        // (in + 2*pad - k)/stride + 1, with pad=1, k=3, stride=2
        return (in_len + 2 - 3) / 2 + 1;
    };
    const int T_chunk_out = conv_out_len(conv_out_len(conv_out_len(chunk_T)));
    const int N_padded = T_chunk_out * num_chunks;

    // Pack mel into the (T_chunk, n_mels, 1, num_chunks) ggml layout.
    std::vector<float> mel_padded((size_t)chunk_T * n_mels * num_chunks, 0.0f);
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int t_start = chunk * chunk_T;
        const int t_end = std::min(t_start + chunk_T, T_mel);
        const int t_len = t_end - t_start; // valid (non-padded) frames in this chunk
        for (int f = 0; f < n_mels; f++) {
            for (int t = 0; t < t_len; t++) {
                mel_padded[(size_t)t + chunk_T * ((size_t)f + n_mels * (size_t)chunk)] =
                    mel_features[(size_t)f * T_mel + (size_t)(t_start + t)];
            }
            // remaining (chunk_T - t_len) entries are already zero from the
            // initial assignment — silence padding for the trailing partial chunk.
        }
    }

    // The reference's eager_attention_forward IGNORES cu_seqlens and uses
    // standard full self-attention with attention_mask=None. cu_seqlens is
    // only consumed by FlashAttention2 on GPU. So on CPU we just need a
    // zero mask. (We keep the input tensor in the graph so the structure
    // is ready when we add real per-chunk padding masking later.)
    std::vector<float> mask((size_t)N_padded * N_padded, 0.0f);

    ggml_cgraph* gf = qwen3_asr_build_graph_encoder(ctx, chunk_T, num_chunks, T_chunk_out);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc encoder graph\n");
        return nullptr;
    }

    // Set inputs
    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel_batched");
    ggml_backend_tensor_set(mel_in, mel_padded.data(), 0, mel_padded.size() * sizeof(float));

    // pe_input ne=(d, T_chunk_out). Pull rows [0, T_chunk_out) from model.audio_pe.
    ggml_tensor* pe_in = ggml_graph_get_tensor(gf, "pe_input");
    {
        const int d = (int)hp.audio_d_model;
        std::vector<float> pe_buf((size_t)d * T_chunk_out);
        // model.audio_pe row p starts at offset p*d. We need to write into ggml
        // ne=(d, T_chunk_out) which has d as ne[0] (fast). Memory layout matches
        // a row-major (T_chunk_out, d) buffer. So just copy [0, T_chunk_out*d).
        std::memcpy(pe_buf.data(), ctx->model.audio_pe.data(), pe_buf.size() * sizeof(float));
        ggml_backend_tensor_set(pe_in, pe_buf.data(), 0, pe_buf.size() * sizeof(float));
    }

    // attn_mask is unreferenced when build_graph_qwen_omni took the
    // flash-attention path (mask is all-zero in this caller, so FA with
    // nullptr mask is equivalent and skips materialising the N²×n_heads
    // scores tensor). In that case the tensor is optimized out and
    // ggml_graph_get_tensor returns NULL.
    ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "attn_mask");
    if (mask_in) {
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: encoder graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "encoder_out");
    if (!out) {
        fprintf(stderr, "qwen3_asr: missing encoder_out tensor\n");
        return nullptr;
    }
    const int pdim = (int)out->ne[0]; // 1024
    const int N = (int)out->ne[1];    // N_padded
    if (out_N_total)
        *out_N_total = N;
    if (out_proj_dim)
        *out_proj_dim = pdim;

    const size_t total = (size_t)pdim * N;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" bool qwen3_asr_kv_init(qwen3_asr_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    // Re-init if the current cache is too small for the requested max_ctx.
    // Old behaviour was "already initialized → no-op", which forced callers
    // to oversize the first init or fail later. For forced alignment the
    // prompt length is known up-front per request, so right-sizing the
    // cache (max_ctx = T_prompt+16) reclaims ~340-390 MiB vs the historical
    // max_ctx=4096 floor. PLAN #69f.
    if (ctx->kv_k) {
        if (max_ctx <= ctx->kv_max_ctx) return true;
        ggml_backend_buffer_free(ctx->kv_buf);
        ggml_free(ctx->kv_ctx);
        ctx->kv_buf = nullptr;
        ctx->kv_ctx = nullptr;
        ctx->kv_k   = nullptr;
        ctx->kv_v   = nullptr;
        ctx->kv_max_ctx = 0;
        ctx->kv_n_used  = 0;
    }

    const auto& hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int n_lay = (int)hp.llm_n_layers;

    ggml_init_params kp = {
        /*mem_size=*/ggml_tensor_overhead() * 4 + 1024,
        /*mem_buffer=*/nullptr,
        /*no_alloc=*/true,
    };
    ctx->kv_ctx = ggml_init(kp);
    // F16 KV cache: halves memory + ~2× cache read bandwidth on decode.
    // Conversion happens at the ggml_cpy() write into the cache view, and
    // ggml_mul_mat handles F16-on-F32 dot products natively for the read path.
    // PLAN #60e + #69e: per-half KV dtype. CRISPASR_KV_QUANT sets both,
    // CRISPASR_KV_QUANT_{K,V} override per half (default f16/f16).
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("qwen3_asr");
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, n_lay);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    // PLAN #69b: optional KV-on-CPU spill for long-context / tight-VRAM users.
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "qwen3_asr");
    // ggml_nbytes() returns the *content* size of a quantised tensor, but
    // the backend buffer requires the *alignment-rounded* size for the
    // allocator's bookkeeping (256 B on CUDA, etc.). For F16/F32 the two
    // are equal; for Q8_0/Q4_0 the alloc size is >= content size, so we
    // must consult the backend's buffer_type alloc_size, not ggml_nbytes.
    // Without this, Q8 KV tripped ggml-backend.cpp:2010 GGML_ASSERT on
    // the v tensor (offset by kbytes content size + v_alloc_size > buffer
    // end). Use buffer_type alignment-aware sizing for both tensors.
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(kv_backend);
    const size_t kbytes = ggml_backend_buft_get_alloc_size(buft, ctx->kv_k);
    const size_t vbytes = ggml_backend_buft_get_alloc_size(buft, ctx->kv_v);
    ctx->kv_buf = ggml_backend_alloc_buffer(kv_backend, kbytes + vbytes);
    if (!ctx->kv_buf) {
        fprintf(stderr, "qwen3_asr: failed to allocate kv buffer\n");
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kbytes);
    ctx->kv_max_ctx = max_ctx;
    ctx->kv_n_used = 0;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "qwen3_asr: kv cache %d MiB (head_dim=%d max_ctx=%d n_kv=%d n_layers=%d)\n",
                (int)((kbytes + vbytes) / 1048576), hd, max_ctx, n_kv, n_lay);
    }
    return true;
}

extern "C" void qwen3_asr_kv_reset(qwen3_asr_context* ctx) {
    if (ctx)
        ctx->kv_n_used = 0;
}

extern "C" float* qwen3_asr_run_llm_kv(qwen3_asr_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                                       int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;
    if (!ctx->kv_k) {
        fprintf(stderr, "qwen3_asr: kv cache not initialized — call qwen3_asr_kv_init first\n");
        return nullptr;
    }
    if (n_past + n_tokens > ctx->kv_max_ctx) {
        fprintf(stderr, "qwen3_asr: kv overflow (n_past=%d + n_tokens=%d > max_ctx=%d)\n", n_past, n_tokens,
                ctx->kv_max_ctx);
        return nullptr;
    }
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    // Use the LM head's actual output dim, not the token vocab size.
    // For ASR models the two are equal; for the ForcedAligner variant
    // the head is (5000, d) while vocab_size stays 152064.
    const int vocab = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);
    const int Lk = n_past + n_tokens;

    // Positions [n_past, n_past+T)
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;

    // Causal mask: only needed for prefill (T > 1). Decode (T = 1) uses
    // ggml_flash_attn_ext with no mask — the single new query attends to
    // all cached keys including itself, so no masking is needed.
    // ggml_flash_attn_ext requires the mask to be F16.
    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        mask.assign((size_t)Lk * n_tokens, zero_h);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = n_past + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neginf_h;
            }
        }
    }

    ggml_cgraph* gf = qwen3_asr_build_graph_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc llm_kv graph\n");
        return nullptr;
    }

    ggml_tensor* embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: llm_kv graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out)
        return nullptr;
    ctx->kv_n_used = n_past + n_tokens;
    // Output is only the last token's logits — see lm_head slice in build_graph_llm_kv.
    if (out_n_tokens)
        *out_n_tokens = 1;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    float* result = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)vocab * sizeof(float));
    return result;
}

extern "C" float* qwen3_asr_embed_tokens(qwen3_asr_context* ctx, const int32_t* input_ids, int n_tokens) {
    if (!ctx || !input_ids || n_tokens <= 0)
        return nullptr;
    const int d = (int)ctx->model.hparams.llm_d_model;

    ggml_cgraph* gf = qwen3_asr_build_graph_embed(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc embed graph\n");
        return nullptr;
    }
    ggml_tensor* ids_in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_in, input_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: embed graph compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    const size_t total = (size_t)d * n_tokens;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" float* qwen3_asr_run_llm_from_embeds(qwen3_asr_context* ctx, const float* inputs_embeds, int n_tokens,
                                                int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int vocab = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = i;
    std::vector<float> mask((size_t)n_tokens * n_tokens, 0.0f);
    for (int i = 0; i < n_tokens; i++)
        for (int j = i + 1; j < n_tokens; j++)
            mask[(size_t)i * n_tokens + j] = -INFINITY;

    ggml_cgraph* gf = qwen3_asr_build_graph_llm_from_embeds(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc llm-from-embeds graph\n");
        return nullptr;
    }

    ggml_tensor* embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: llm-from-embeds graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out) {
        fprintf(stderr, "missing logits\n");
        return nullptr;
    }
    if (out_n_tokens)
        *out_n_tokens = n_tokens;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    const size_t total = (size_t)vocab * n_tokens;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" float* qwen3_asr_run_llm(qwen3_asr_context* ctx, const int32_t* input_ids, int n_tokens, int* out_n_tokens,
                                    int* out_vocab_size) {
    if (!ctx || !input_ids || n_tokens <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int vocab = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);

    // Build positions = [0, 1, ..., T-1]
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = i;

    // Build causal mask: (T, T) F32. mask[i, j] = 0 if j <= i else -inf.
    // ggml ne[0]=j (key, fast), ne[1]=i (query). Disallowed → -inf.
    std::vector<float> mask((size_t)n_tokens * n_tokens, 0.0f);
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < n_tokens; j++) {
            if (j > i)
                mask[(size_t)i * n_tokens + j] = -INFINITY;
        }
    }

    ggml_cgraph* gf = qwen3_asr_build_graph_llm(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc llm graph\n");
        return nullptr;
    }

    ggml_tensor* ids_in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_in, input_ids, 0, (size_t)n_tokens * sizeof(int32_t));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));

    ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: llm graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out) {
        fprintf(stderr, "qwen3_asr: missing logits tensor\n");
        return nullptr;
    }
    if (out_n_tokens)
        *out_n_tokens = n_tokens;
    if (out_vocab_size)
        *out_vocab_size = vocab;

    const size_t total = (size_t)vocab * n_tokens;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" int qwen3_asr_lm_head_dim(struct qwen3_asr_context* ctx) {
    if (!ctx)
        return 0;
    const auto& hp = ctx->model.hparams;
    return (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);
}

// Forward decl — defined alongside the streaming-path helpers further
// down. Both the one-shot and streaming dispatchers run this on the
// argmax timestamp-class vector before converting to ms. `ts_confs`
// (optional, pass nullptr to disable) carries per-row top-1 softmax
// probabilities; when paired with a non-zero threshold (env
// QWEN3_FA_LIS_CONF_THRESHOLD), low-confidence positions get demoted out
// of LIS membership so the per-word boundary is interpolated from
// surrounding high-confidence anchors instead of trusting a noisy
// argmax that happened to fit the monotone chain.
static void qwen3_fa_lis_monotonize(std::vector<int>& ts_classes, int n_words,
                                    const std::vector<float>* ts_confs = nullptr);

// Soft-argmax over one row of logits (H = lm_head_dim classes).
//
// Three passes:
//   pass 1: find max-logit (numerics anchor + argmax index `best`).
//   pass 2: full-row softmax denominator → top-1 probability as `*out_conf`.
//   pass 3: windowed expectation over [best-W, best+W] (clipped to row
//           bounds) using exp(row[k] - mx) weights — same un-normalized
//           weights as the full softmax, so within-window probabilities
//           are just window weights / window denominator.
//
// Why windowed: lm_head_dim is 5000 classes covering 400 s of timestamp
// space, but Tolstoy-class prompts only use the first ~1500. Full-row
// expectation gets dragged thousands of classes high by the tiny but
// numerous masses on out-of-range distant classes. A ±W window around
// argmax keeps the expectation local to the actual peak — for a sharp
// F16 peak it collapses back to argmax (within-window non-best mass is
// negligible), and for a fat Q8 peak it averages the ±1-2 class jitter.
//
// Confidence uses the full-row softmax denominator so it remains
// comparable across windowed/un-windowed analyses; 1/Z ∈ [1/H, 1].
//
// Window radius is overridable via QWEN3_FA_SOFTMAX_WINDOW.
//
// Default 0 = pure argmax (no class-picker change vs pre-patch). The
// confidence value is still emitted because pass 2 always runs the
// full-row softmax denominator. Setting W>0 enables ±W-class windowed
// expectation; empirically W=4 changes 36/670 F16 placeholders by 5-10
// ms mean drift but doesn't close the 3-of-335 Q8 outliers (those are
// 6-class noise spikes, beyond any reasonable W), and lever #3
// (low-conf LIS gate) is a more targeted attack.
static inline int qwen3_fa_softmax_window() {
    static const int W = []() {
        const char* e = std::getenv("QWEN3_FA_SOFTMAX_WINDOW");
        if (!e || !*e) return 0;
        const int v = std::atoi(e);
        return v < 0 ? 0 : v;
    }();
    return W;
}

// Confidence threshold for LIS anchor gating (lever #3). Positions with
// top-1 softmax probability below this threshold are demoted out of the
// LIS chain so their values are reconstructed by interpolation from
// surrounding high-confidence anchors. Default 0 = gate disabled. Set
// QWEN3_FA_LIS_CONF_THRESHOLD=0.4 to activate.
static inline float qwen3_fa_lis_conf_threshold() {
    static const float T = []() {
        const char* e = std::getenv("QWEN3_FA_LIS_CONF_THRESHOLD");
        if (!e || !*e) return 0.0f;
        const float v = (float)std::atof(e);
        return v < 0.0f ? 0.0f : v;
    }();
    return T;
}

// Audio-aware post-LIS boundary refinement (HANDOFF-aligner-audio-peaks).
// After LIS gives interpolated [t0,t1] for each word, we walk ±W bins of
// 40 ms RMS energy and snap each boundary to the nearest speech onset
// (for t0) or offset (for t1). Closes the "highlight in silence"
// failure mode that bench_audio_peaks.py's low_energy_words metric
// catches. Default W=6 (±240 ms) → quality fixes ship without operator
// intervention; pass QWEN3_FA_AUDIO_REFINE=0 to opt out.
static inline int qwen3_fa_audio_refine_window() {
    static const int W = []() {
        const char* e = std::getenv("QWEN3_FA_AUDIO_REFINE");
        if (!e || !*e) return 6;
        const int v = std::atoi(e);
        return v < 0 ? 0 : v;
    }();
    return W;
}

// RMS-vs-median threshold ratio for "speech" classification in the
// audio-refine walker. Bins with RMS > median * thresh are speech;
// below are quiet. Default 0.30 — matches bench_audio_peaks.py's
// gap_quiet_frac threshold so the in-aligner refiner uses the same
// notion of "silence" as the off-line scorer.
static inline float qwen3_fa_audio_refine_thresh() {
    static const float T = []() {
        const char* e = std::getenv("QWEN3_FA_AUDIO_REFINE_THRESH");
        if (!e || !*e) return 0.30f;
        const float v = (float)std::atof(e);
        return v <= 0.0f ? 0.30f : v;
    }();
    return T;
}

// Bin width in ms for the RMS energy ladder. Default 40 — matches the
// off-line bench scorer. Keep aligned with bench_audio_peaks.BIN_MS so
// the metric directly predicts the in-aligner walker's notion of
// "this boundary is in silence."
static inline int qwen3_fa_audio_refine_bin_ms() {
    static const int B = []() {
        const char* e = std::getenv("QWEN3_FA_AUDIO_REFINE_BIN_MS");
        if (!e || !*e) return 40;
        const int v = std::atoi(e);
        return v < 5 ? 40 : v;
    }();
    return B;
}

// Gap-bridge pass: when adjacent words leave a gap larger than this
// many ms AND the gap's audio is mostly loud (≥ gap-loud-frac), the
// model has carved a phantom pause through a continuous speech run.
// Snap both word[i-1].t1 and word[i].t0 to the quietest bin in the
// gap so playback never falls between words. Default 200 ms (~2.5
// 80 ms classes — small enough to ignore real pauses, large enough to
// catch the failure mode).
static inline int qwen3_fa_audio_refine_gap_min_ms() {
    static const int M = []() {
        const char* e = std::getenv("QWEN3_FA_AUDIO_REFINE_GAP_MIN_MS");
        if (!e || !*e) return 200;
        const int v = std::atoi(e);
        return v < 0 ? 200 : v;
    }();
    return M;
}

// Fraction of bins in the gap that must exceed `loud_thresh` to treat
// the gap as "phantom silence inside continuous speech." Default 0.60.
// Tighter (higher) → only bridge gaps that are clearly inside speech.
static inline float qwen3_fa_audio_refine_gap_loud() {
    static const float F = []() {
        const char* e = std::getenv("QWEN3_FA_AUDIO_REFINE_GAP_LOUD");
        if (!e || !*e) return 0.60f;
        const float v = (float)std::atof(e);
        return v <= 0.0f || v > 1.0f ? 0.60f : v;
    }();
    return F;
}

// Highlight lead-time in ms: shift all word boundaries EARLIER by this
// many ms before emitting. Compensates for human-perceptual playback
// lag (audio reaches the ear a few frames after the playhead reports
// it) so the highlight feels "in sync" rather than chasing the audio.
// Default 80 ms — empirical sweet spot on browser audio + AGC tail.
// Boundaries clamp to >= 0 so leading silence words still start at 0.
static inline int qwen3_fa_audio_refine_lead_ms() {
    static const int L = []() {
        const char* e = std::getenv("QWEN3_FA_AUDIO_REFINE_LEAD_MS");
        if (!e) return 80;        // env unset → default lead
        if (!*e) return 0;        // env set to empty → off
        const int v = std::atoi(e);
        return v < 0 ? 0 : v;
    }();
    return L;
}

// Walk per-word [t0,t1] ms boundaries ±W bins on the RMS ladder and
// snap each to the nearest speech edge. Two failure modes attacked:
//
//   t0 in silence (word-start latency, e.g. 'I' w219):
//       → first bin in [t0-W, t0+W] whose RMS > loud_thresh
//   t1 in silence (sentence-end tail, e.g. 'place.' w334):
//       → last  bin in [t1-W, t1+W]+1 whose RMS > loud_thresh
//
// If no qualifying bin found in the window, the boundary is left
// alone (refusal-to-snap = safer than landing on noise).
//
// After per-word snaps, a monotonicity sweep enforces
// out_start_ms[w] <= out_end_ms[w] <= out_start_ms[w+1].
//
// `samples` are mono float32 at `sample_rate` Hz (the aligner uses
// 16 kHz). RMS is computed over non-overlapping bin_ms windows.
static void qwen3_fa_audio_refine_bounds(const float* samples, int n_samples, int sample_rate,
                                         int64_t* out_start_ms, int64_t* out_end_ms, int n_words) {
    if (!samples || n_samples <= 0 || !out_start_ms || !out_end_ms || n_words <= 0)
        return;
    const int W = qwen3_fa_audio_refine_window();
    if (W <= 0) return; // disabled
    const int bin_ms = qwen3_fa_audio_refine_bin_ms();
    const int per_bin = (sample_rate * bin_ms) / 1000;
    if (per_bin <= 0) return;
    const int n_bins = n_samples / per_bin;
    if (n_bins <= 0) return;

    std::vector<float> rms(n_bins);
    for (int i = 0; i < n_bins; i++) {
        double sq = 0.0;
        const float* p = samples + (size_t)i * per_bin;
        for (int k = 0; k < per_bin; k++) sq += (double)p[k] * (double)p[k];
        rms[i] = (float)std::sqrt(sq / per_bin);
    }
    // Median via nth_element.
    std::vector<float> tmp(rms);
    std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
    const float median = tmp[tmp.size() / 2];
    const float loud = median * qwen3_fa_audio_refine_thresh();

    auto bin_of = [&](int64_t ms) -> int {
        int b = (int)(ms / bin_ms);
        if (b < 0) b = 0;
        if (b > n_bins - 1) b = n_bins - 1;
        return b;
    };

    auto loud_at = [&](int b) -> bool { return b >= 0 && b < n_bins && rms[b] > loud; };

    for (int w = 0; w < n_words; w++) {
        // t0: locate the speech onset for this word.
        //   If t0_bin is already inside speech: walk LEFT as long as bins
        //     stay loud, capped at W bins. Stops at the first silence —
        //     never crosses into the previous word's speech body.
        //   If t0_bin is in silence: walk RIGHT for the first loud bin
        //     in [t0_bin+1, t0_bin+W]. This catches "t0 lands before
        //     speech onset" without ever extending across an intervening
        //     silent gap (so the walker can't jump TO the next word).
        const int t0_bin = bin_of(out_start_ms[w]);
        int new_t0 = t0_bin;
        if (loud_at(t0_bin)) {
            int onset = t0_bin;
            for (int i = t0_bin - 1; i >= std::max(0, t0_bin - W); i--) {
                if (loud_at(i)) onset = i;
                else break;
            }
            new_t0 = onset;
        } else {
            for (int i = t0_bin + 1; i <= std::min(n_bins - 1, t0_bin + W); i++) {
                if (loud_at(i)) { new_t0 = i; break; }
            }
        }
        out_start_ms[w] = (int64_t)new_t0 * bin_ms;

        // t1: locate speech offset. Same asymmetry, mirrored:
        //   If t1-1 bin is loud (currently inside speech): walk RIGHT only
        //     while bins stay loud, capped at W. Stops at the first
        //     silence — won't run into the NEXT word's speech body even
        //     when adjacent words are tightly butted.
        //   If t1-1 bin is silent: walk LEFT for the last loud bin in
        //     [t1_bin-W, t1_bin-1]. This trims sentence-end tails.
        const int t1_bin = bin_of(out_end_ms[w] > 0 ? out_end_ms[w] - 1 : 0);
        int new_t1 = t1_bin + 1;
        if (loud_at(t1_bin)) {
            int offset = t1_bin;
            for (int i = t1_bin + 1; i <= std::min(n_bins - 1, t1_bin + W); i++) {
                if (loud_at(i)) offset = i;
                else break;
            }
            new_t1 = offset + 1;
        } else {
            for (int i = t1_bin - 1; i >= std::max(0, t1_bin - W); i--) {
                if (loud_at(i)) { new_t1 = i + 1; break; }
            }
        }
        out_end_ms[w] = (int64_t)new_t1 * bin_ms;
    }

    // Monotonicity sweep: out_start <= out_end, no overlap across words.
    for (int w = 0; w < n_words; w++) {
        if (out_end_ms[w] <= out_start_ms[w])
            out_end_ms[w] = out_start_ms[w] + bin_ms;
        if (w > 0 && out_start_ms[w] < out_end_ms[w - 1])
            out_start_ms[w] = out_end_ms[w - 1];
        if (out_end_ms[w] <= out_start_ms[w])
            out_end_ms[w] = out_start_ms[w] + bin_ms;
    }

    // Gap-bridge pass. When two adjacent words have a gap > gap_min_ms
    // and most of that gap is loud, the model carved a phantom silence
    // through a continuous speech run (the canonical "fields highlights
    // late" failure — model puts the.t1 mid-run and fields.t0 way after
    // the actual onset). Snap both boundaries to the quietest bin in
    // the gap so playback never falls between words.
    const int   gap_min_ms = qwen3_fa_audio_refine_gap_min_ms();
    const float gap_loud_f = qwen3_fa_audio_refine_gap_loud();
    if (gap_min_ms > 0 && gap_loud_f > 0.0f) {
        for (int w = 1; w < n_words; w++) {
            const int64_t gap_start = out_end_ms[w - 1];
            const int64_t gap_end   = out_start_ms[w];
            if (gap_end - gap_start < gap_min_ms) continue;
            int gap_lo = (int)(gap_start / bin_ms);
            int gap_hi = (int)((gap_end - 1) / bin_ms);
            if (gap_lo < 0) gap_lo = 0;
            if (gap_hi > n_bins - 1) gap_hi = n_bins - 1;
            if (gap_hi < gap_lo) continue;
            int n_gap = gap_hi - gap_lo + 1;
            int n_loud = 0;
            int   min_idx = gap_lo;
            float min_rms = rms[gap_lo];
            for (int i = gap_lo; i <= gap_hi; i++) {
                if (rms[i] > loud) n_loud++;
                if (rms[i] < min_rms) { min_rms = rms[i]; min_idx = i; }
            }
            if ((float)n_loud / (float)n_gap < gap_loud_f) continue;
            const int64_t mid_ms = (int64_t)min_idx * bin_ms;
            out_end_ms[w - 1] = mid_ms;
            out_start_ms[w]   = mid_ms;
        }
    }

    // Highlight lead-time. Shift every word's [t0,t1] EARLIER by N ms
    // to compensate for browser-side audio playback lag; the highlight
    // then sits a hair ahead of the audio so it doesn't feel like it's
    // chasing. Clamp to >= 0 so leading words stay non-negative.
    const int lead_ms = qwen3_fa_audio_refine_lead_ms();
    if (lead_ms > 0) {
        for (int w = 0; w < n_words; w++) {
            out_start_ms[w] = std::max<int64_t>(0, out_start_ms[w] - lead_ms);
            out_end_ms[w]   = std::max<int64_t>(0, out_end_ms[w] - lead_ms);
            if (out_end_ms[w] <= out_start_ms[w])
                out_end_ms[w] = out_start_ms[w] + bin_ms;
        }
    }
}

static inline int qwen3_fa_pick_row_class(const float* row, int H, float* out_conf) {
    int best = 0;
    float mx = row[0];
    for (int k = 1; k < H; k++) {
        if (row[k] > mx) { mx = row[k]; best = k; }
    }
    // Confidence: softmax(best) = 1 / Σ_k exp(row[k] - mx). Full-row.
    double sum_full = 0.0;
    for (int k = 0; k < H; k++) sum_full += std::exp((double)(row[k] - mx));
    if (out_conf) {
        float c = (float)(1.0 / sum_full);
        if (c < 0.0f) c = 0.0f;
        else if (c > 1.0f) c = 1.0f;
        *out_conf = c;
    }
    // Windowed expectation. W=0 reduces to argmax exactly.
    const int W = qwen3_fa_softmax_window();
    if (W == 0) return best;
    const int lo = std::max(0, best - W);
    const int hi = std::min(H - 1, best + W);
    double sum_w = 0.0;
    double sum_kw = 0.0;
    for (int k = lo; k <= hi; k++) {
        double w = std::exp((double)(row[k] - mx));
        sum_w  += w;
        sum_kw += (double)k * w;
    }
    long soft = std::lround(sum_kw / sum_w);
    if (soft < 0) soft = 0;
    if (soft > H - 1) soft = H - 1;
    return (int)soft;
}

// High-level forced-alignment dispatch.
//
// Mirrors qwen_asr/inference/qwen3_forced_aligner.py::Qwen3ForcedAligner.align
// minus the language-specific tokenizers (we whitespace-split words and let
// each word go through the standard byte-level BPE encoder; CJK char-level
// tokenization is a follow-up).
extern "C" int qwen3_asr_align_words(struct qwen3_asr_context* ctx, const float* samples, int n_samples,
                                     const char** words, int n_words, int64_t* out_start_ms, int64_t* out_end_ms,
                                     float* out_confidence) {
    if (!ctx || !samples || n_samples <= 0 || !words || n_words <= 0 || !out_start_ms || !out_end_ms)
        return -1;

    constexpr int TIMESTAMP_TOKEN_ID = 151705;
    // Default ms-per-class. Forced-aligner config sets this to 80; the
    // value isn't carried in the GGUF metadata yet, so we hardcode it
    // for now and add a kv field on the next converter bump.
    constexpr float TIMESTAMP_SEGMENT_TIME_MS = 80.0f;

    // Stage-by-stage VRAM probe. Enabled via QWEN3_FA_PROFILE_VRAM=1.
    // Off by default — cheap (one ggml_backend_dev_memory call per stage)
    // but adds 7 lines of stderr per align request when on. Used for
    // attributing the ~1 GiB of non-weight overhead in the FA path.
    const bool probe_vram = []() {
        const char* e = std::getenv("QWEN3_FA_PROFILE_VRAM");
        return e && *e && std::atoi(e) > 0;
    }();
    auto vram_used_mib = [&]() -> double {
        size_t f = 0, t = 0;
        ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (gpu) ggml_backend_dev_memory(gpu, &f, &t);
        return (double)(t - f) / 1048576.0;
    };
    auto probe = [&](const char* label) {
        if (!probe_vram) return;
        fprintf(stderr, "  [fa-vram %-14s] gpu_used=%7.1f MiB\n", label, vram_used_mib());
    };
    probe("align/entry");

    // 1. Mel
    int n_mels = 0, T_mel = 0;
    float* mel = qwen3_asr_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        fprintf(stderr, "qwen3_asr[align]: mel failed\n");
        return -2;
    }
    probe("after-mel");

    // 2. Audio encoder
    int N_enc = 0, pdim = 0;
    float* audio_embeds = qwen3_asr_run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &pdim);
    free(mel);
    if (!audio_embeds) {
        fprintf(stderr, "qwen3_asr[align]: encoder failed\n");
        return -3;
    }
    probe("after-encoder");

    // 3. Build the prompt token-id sequence:
    //      <|audio_start|>  <|audio_pad|>×N_enc  <|audio_end|>
    //      word_1 <timestamp> <timestamp>
    //      word_2 <timestamp> <timestamp>
    //      ...
    //      word_M <timestamp> <timestamp>
    const auto& hp = ctx->model.hparams;
    const int audio_start_id = (int)hp.audio_start_token_id;
    const int audio_end_id = (int)hp.audio_end_token_id;
    const int audio_pad_id = (int)hp.audio_pad_token_id;

    std::vector<int32_t> ids;
    ids.reserve((size_t)(N_enc + n_words * 6 + 4));
    ids.push_back(audio_start_id);
    for (int i = 0; i < N_enc; i++)
        ids.push_back(audio_pad_id);
    ids.push_back(audio_end_id);

    // Tokenize each word separately and append two timestamp markers
    // after it. Whitespace-split English / Latin scripts work fine
    // through the standard BPE encoder; CJK languages need char-level
    // pre-tokenization which is tracked as a follow-up. The leading
    // space convention matches GPT-2 BPE: each non-first word starts
    // with a space so the tokenizer recognises it as a word boundary.
    for (int w = 0; w < n_words; w++) {
        const std::string word = (w == 0) ? std::string(words[w]) : std::string(" ") + words[w];
        int n = 0;
        int32_t* arr = qwen3_asr_tokenize(ctx, word.c_str(), &n);
        if (arr && n > 0) {
            for (int i = 0; i < n; i++)
                ids.push_back(arr[i]);
        }
        free(arr);
        ids.push_back(TIMESTAMP_TOKEN_ID);
        ids.push_back(TIMESTAMP_TOKEN_ID);
    }

    const int T_prompt = (int)ids.size();

    // 4. Embed and splice audio
    float* text_embeds = qwen3_asr_embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) {
        free(audio_embeds);
        fprintf(stderr, "qwen3_asr[align]: embed failed\n");
        return -4;
    }
    probe("after-embed");
    int spliced = 0;
    for (int i = 0; i < T_prompt && spliced < N_enc; i++) {
        if (ids[i] == audio_pad_id) {
            std::memcpy(text_embeds + (size_t)i * pdim, audio_embeds + (size_t)spliced * pdim, pdim * sizeof(float));
            spliced++;
        }
    }
    free(audio_embeds);

    // 5. KV cache + aligner forward
    //
    // Right-size max_ctx to this request's prompt length. Historical
    // floor of 4096 was inherited from the ASR backend (which decodes
    // autoregressively past prefill); forced alignment is one-shot, so
    // the floor is wasted VRAM. Each unused token costs head_dim *
    // n_kv * n_layers * 2 (K+V) * 2 (f16) = 16 KiB per slot, so 4096-T
    // slots at T=650 ≈ 54 MiB recovered per missing token in floor.
    // Env QWEN3_FA_KV_MIN_CTX lets ops pin a minimum floor — useful
    // when the worker churns through wildly different prompt lengths
    // and the realloc cost (4-8 ms) hurts more than the headroom.
    int kv_min_ctx = 0;
    if (const char* e = std::getenv("QWEN3_FA_KV_MIN_CTX")) kv_min_ctx = std::atoi(e);
    const int kv_max_ctx = std::max(T_prompt + 16, kv_min_ctx);
    if (!qwen3_asr_kv_init(ctx, kv_max_ctx)) {
        free(text_embeds);
        fprintf(stderr, "qwen3_asr[align]: kv_init failed\n");
        return -5;
    }
    probe("after-kv_init");
    if (probe_vram) {
        fprintf(stderr, "  [fa-vram T_prompt=%d N_enc=%d n_words=%d]\n", T_prompt, N_enc, n_words);
    }
    int n_t_out = 0, H = 0;
    float* logits = qwen3_asr_run_aligner(ctx, text_embeds, T_prompt, &n_t_out, &H);
    free(text_embeds);
    if (!logits) {
        fprintf(stderr, "qwen3_asr[align]: aligner forward failed\n");
        return -6;
    }
    probe("after-aligner");

    // 6. Softmax-expectation class pick at each <timestamp> placeholder.
    // The graph stores logits as ne[0]=H, ne[1]=T_prompt row-major, i.e.
    // logits[t * H + k] = score(k, t). See qwen3_fa_pick_row_class for the
    // expectation vs raw-argmax rationale (Q8 KV peak-widening robustness).
    std::vector<int>   ts_classes;
    std::vector<float> ts_confs;
    ts_classes.reserve((size_t)(n_words * 2));
    ts_confs.reserve((size_t)(n_words * 2));
    for (int t = 0; t < T_prompt; t++) {
        if (ids[t] != TIMESTAMP_TOKEN_ID)
            continue;
        const float* row = logits + (size_t)t * H;
        float conf = 0.0f;
        const int cls = qwen3_fa_pick_row_class(row, H, &conf);
        ts_classes.push_back(cls);
        ts_confs.push_back(conf);
    }
    free(logits);
    probe("after-argmax");

    if ((int)ts_classes.size() != 2 * n_words) {
        fprintf(stderr,
                "qwen3_asr[align]: timestamp marker count mismatch — "
                "got %zu placeholders, expected %d\n",
                ts_classes.size(), 2 * n_words);
        return -7;
    }

    // 6b. LIS-based monotonicity fix + minimum 1-class duration. Shared
    // with the streaming path; see qwen3_fa_lis_monotonize for the algo.
    // ts_confs is forwarded so the optional low-conf anchor gate can fire.
    qwen3_fa_lis_monotonize(ts_classes, n_words, &ts_confs);

    // 7. Convert classes → ms and write into the caller's parallel arrays.
    //    Per-word confidence is the min of the two boundary placeholders
    //    (a word is only as confident as its weakest boundary).
    for (int w = 0; w < n_words; w++) {
        out_start_ms[w] = (int64_t)((float)ts_classes[2 * w + 0] * TIMESTAMP_SEGMENT_TIME_MS);
        out_end_ms[w] = (int64_t)((float)ts_classes[2 * w + 1] * TIMESTAMP_SEGMENT_TIME_MS);
        if (out_confidence)
            out_confidence[w] = std::min(ts_confs[2 * w + 0], ts_confs[2 * w + 1]);
    }

    // 8. Optional audio-aware boundary refinement. Gated by
    //    QWEN3_FA_AUDIO_REFINE=W (window radius in bins, default 0 = off).
    //    No-op when disabled → bit-identical to pre-patch.
    qwen3_fa_audio_refine_bounds(samples, n_samples, /*sample_rate=*/16000,
                                  out_start_ms, out_end_ms, n_words);
    return 0;
}

// Internal core: full-T forward through the FA LLM body at any n_past.
// The KV cache must already be allocated AND positioned (caller is
// responsible for kv_reset on one-shot, or for setting kv_n_used to
// the prior commit point on streaming). Writes K/V into slots
// [n_past, n_past + n_tokens). Returns malloc'd (n_tokens, H) logits.
static float* qwen3_asr_run_aligner_core(qwen3_asr_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                                         int* out_n_tokens, int* out_lm_head_dim) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;
    if (!ctx->kv_k) {
        fprintf(stderr, "qwen3_asr: kv cache not initialized — call qwen3_asr_kv_init first\n");
        return nullptr;
    }
    if (n_past + n_tokens > ctx->kv_max_ctx) {
        fprintf(stderr, "qwen3_asr: aligner needs %d+%d tokens but kv max_ctx is %d\n", n_past, n_tokens,
                ctx->kv_max_ctx);
        return nullptr;
    }

    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int H = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);
    const int Lk = n_past + n_tokens;

    // Positions n_past..n_past+n_tokens-1.
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;

    // Causal mask (Lk, n_tokens) F16, -inf above the diagonal relative to
    // the absolute slot of each query. For n_past=0 this is a standard
    // upper-triangular mask; for n_past>0 the cached audio prefix is
    // always visible to the new queries (their causal frontier sits at
    // n_past+q, which is >= n_past for any q >= 0).
    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        mask.assign((size_t)Lk * n_tokens, zero_h);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = n_past + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neginf_h;
            }
        }
    }

    ggml_cgraph* gf = qwen3_asr_build_graph_llm_kv(ctx, n_past, n_tokens, /*last_token_only=*/false);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "qwen3_asr: failed to alloc aligner graph\n");
        return nullptr;
    }

    ggml_tensor* embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen3_asr: aligner graph compute failed\n");
        return nullptr;
    }

    ctx->kv_n_used = n_past + n_tokens;

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out)
        return nullptr;
    if (out_n_tokens)
        *out_n_tokens = n_tokens;
    if (out_lm_head_dim)
        *out_lm_head_dim = H;

    const size_t total = (size_t)H * n_tokens;
    float* result = (float*)malloc(total * sizeof(float));
    if (!result)
        return nullptr;
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

// One full-T forward pass for the Qwen3-ForcedAligner. Same KV-cached
// graph the ASR backend uses for prefill, but with last_token_only=false
// so the lm_head sees every position. The KV cache is reset at the start
// because forced alignment is one-shot — no autoregressive decode loop.
extern "C" float* qwen3_asr_run_aligner(struct qwen3_asr_context* ctx, const float* inputs_embeds, int n_tokens,
                                        int* out_n_tokens, int* out_lm_head_dim) {
    if (!ctx)
        return nullptr;
    qwen3_asr_kv_reset(ctx);
    return qwen3_asr_run_aligner_core(ctx, inputs_embeds, n_tokens, /*n_past=*/0, out_n_tokens, out_lm_head_dim);
}

// ============================================================================
// Streaming forced alignment
// ============================================================================
//
// Public API: qwen3_asr_align_words_streaming + qwen3_asr_align_streaming_reset.
// Internal helpers below extract the argmax+LIS+ms-conversion pipeline so
// the streaming path can reuse it without duplicating ~80 lines of
// fragile timestamp post-processing.

constexpr int kQwen3FaTimestampTokenId = 151705;
constexpr float kQwen3FaTimestampSegmentMs = 80.0f;

// Apply the LIS-based monotonicity fix in-place on a 2*n_words vector
// of timestamp class indices, then enforce a minimum 1-class duration
// per word. Same algo as the reference Qwen3-ForcedAligner: find the
// longest non-decreasing subsequence and treat non-LIS elements as
// outliers to be reconstructed.
//
// Outlier handling for non-LIS elements:
//   * Between two LIS anchors: linear interpolation.
//   * Leading (before first LIS): step back by 1 class per index from
//     the first LIS value, clamped at 0. Without this, a low-confidence
//     leading prediction sitting ABOVE the first LIS value would survive
//     unmodified and the per-word end-vs-start guard would collapse the
//     first word to t0 == t1.
//   * Trailing (after last LIS): step forward by 1 class per index from
//     the last LIS value. Without this, a trailing prediction ABOVE the
//     LIS tail would be held flat at prev_val and the end-vs-start guard
//     would collapse the final word to t0 == t1 — by far the most
//     common breakage in prod alignment output.
//
// Finally each word is given at least 1 class of duration, cascading
// forward so a bump on word w doesn't leave w+1 starting before w's
// new end.
static void qwen3_fa_lis_monotonize(std::vector<int>& ts_classes, int n_words,
                                    const std::vector<float>* ts_confs) {
    const int M = (int)ts_classes.size();
    if (M == 0) return;

    std::vector<int> dp;
    std::vector<int> parent(M, -1);
    std::vector<int> idx_map;
    for (int i = 0; i < M; i++) {
        int val = ts_classes[i];
        auto it = std::upper_bound(dp.begin(), dp.end(), val);
        int pos = (int)(it - dp.begin());
        if (pos == (int)dp.size()) {
            dp.push_back(val);
            idx_map.push_back(i);
        } else {
            dp[pos] = val;
            idx_map[pos] = i;
        }
        parent[i] = (pos > 0) ? idx_map[pos - 1] : -1;
    }
    std::vector<bool> in_lis(M, false);
    int first_lis = -1, last_lis = -1;
    if (!idx_map.empty()) {
        int k = idx_map.back();
        last_lis = k;
        while (k >= 0) {
            in_lis[k] = true;
            first_lis = k;
            k = parent[k];
        }
    }

    // Lever #3 low-conf gate: demote LIS members whose row confidence
    // falls below the threshold so they get interpolated from neighbors
    // instead of anchoring the chain to a noisy argmax. After demotion,
    // recompute first_lis / last_lis. If the gate would drop EVERYONE,
    // disable it to preserve the original LIS chain.
    const float conf_thresh = qwen3_fa_lis_conf_threshold();
    if (ts_confs && (int)ts_confs->size() == M && conf_thresh > 0.0f) {
        std::vector<bool> demoted = in_lis;  // working copy
        int kept = 0;
        for (int i = 0; i < M; i++) {
            if (demoted[i] && (*ts_confs)[i] < conf_thresh) demoted[i] = false;
            if (demoted[i]) kept++;
        }
        // Need at least 2 anchors for middle-interp to make sense; if the
        // gate would leave us with 0 or 1, skip the gate this call.
        if (kept >= 2) {
            in_lis = std::move(demoted);
            first_lis = last_lis = -1;
            for (int i = 0; i < M; i++) {
                if (in_lis[i]) {
                    if (first_lis < 0) first_lis = i;
                    last_lis = i;
                }
            }
        }
    }

    if (first_lis < 0) {
        // M >= 1 always yields at least one LIS member, but guard anyway.
        return;
    }

    // Leading non-LIS: step back from first LIS value, clamped at 0.
    for (int j = 0; j < first_lis; j++) {
        int v = ts_classes[first_lis] - (first_lis - j);
        ts_classes[j] = v < 0 ? 0 : v;
    }
    // Middle gaps: linear interpolation between adjacent LIS anchors.
    int prev_lis = first_lis;
    int prev_val = ts_classes[first_lis];
    for (int i = first_lis + 1; i <= last_lis; i++) {
        if (in_lis[i]) {
            for (int j = prev_lis + 1; j < i; j++) {
                float frac = (float)(j - prev_lis) / (float)(i - prev_lis);
                ts_classes[j] = prev_val + (int)(frac * (float)(ts_classes[i] - prev_val));
            }
            prev_lis = i;
            prev_val = ts_classes[i];
        }
    }
    // Trailing non-LIS: step forward by 1 class per index past last LIS.
    for (int j = last_lis + 1; j < M; j++)
        ts_classes[j] = ts_classes[last_lis] + (j - last_lis);

    // Minimum 1-class duration per word, cascading across boundaries.
    for (int w = 0; w < n_words; w++) {
        if (ts_classes[2 * w + 1] <= ts_classes[2 * w])
            ts_classes[2 * w + 1] = ts_classes[2 * w] + 1;
        if (w + 1 < n_words && ts_classes[2 * (w + 1)] < ts_classes[2 * w + 1])
            ts_classes[2 * (w + 1)] = ts_classes[2 * w + 1];
    }
}

// Pull soft-argmax class at each timestamp position out of a
// (n_logit_rows, H) logits buffer, run LIS monotonicity, and convert to ms.
//
// `ts_logit_rows[i]` is the row index inside `logits` for the i-th
// timestamp placeholder (chronological order). `out_confidence` is
// nullptr-safe; when non-null, the per-word value is the min softmax(top1)
// across the two boundary placeholders. Returns 0 on success.
static int qwen3_fa_extract_word_timings(const float* logits, int /*n_logit_rows*/, int H,
                                         const std::vector<int>& ts_logit_rows, int n_words, int64_t* out_start_ms,
                                         int64_t* out_end_ms, float* out_confidence) {
    if ((int)ts_logit_rows.size() != 2 * n_words) {
        fprintf(stderr, "qwen3_asr[align]: timestamp count mismatch — got %zu placeholders, expected %d\n",
                ts_logit_rows.size(), 2 * n_words);
        return -7;
    }
    std::vector<int>   ts_classes;
    std::vector<float> ts_confs;
    ts_classes.reserve(ts_logit_rows.size());
    ts_confs.reserve(ts_logit_rows.size());
    for (int row : ts_logit_rows) {
        const float* r = logits + (size_t)row * H;
        float conf = 0.0f;
        const int cls = qwen3_fa_pick_row_class(r, H, &conf);
        ts_classes.push_back(cls);
        ts_confs.push_back(conf);
    }
    qwen3_fa_lis_monotonize(ts_classes, n_words, &ts_confs);
    for (int w = 0; w < n_words; w++) {
        out_start_ms[w] = (int64_t)((float)ts_classes[2 * w + 0] * kQwen3FaTimestampSegmentMs);
        out_end_ms[w] = (int64_t)((float)ts_classes[2 * w + 1] * kQwen3FaTimestampSegmentMs);
        if (out_confidence)
            out_confidence[w] = std::min(ts_confs[2 * w + 0], ts_confs[2 * w + 1]);
    }
    return 0;
}

// Build the text-suffix token id stream for the FA prompt:
//   [audio_end, w1_tokens..., TS, TS, w2_tokens..., TS, TS, ...]
// `ts_offsets_in_suffix` gets the offset (within text_suffix_ids) of
// each timestamp placeholder in chronological order — used by the
// argmax helper to find the right logits rows.
static void qwen3_fa_build_text_suffix(qwen3_asr_context* ctx, const char** words, int n_words, int audio_end_id,
                                       std::vector<int32_t>& text_suffix_ids, std::vector<int>& ts_offsets_in_suffix) {
    text_suffix_ids.clear();
    ts_offsets_in_suffix.clear();
    text_suffix_ids.reserve((size_t)(n_words * 6 + 4));
    ts_offsets_in_suffix.reserve((size_t)(2 * n_words));
    text_suffix_ids.push_back((int32_t)audio_end_id);
    for (int w = 0; w < n_words; w++) {
        const std::string word = (w == 0) ? std::string(words[w]) : std::string(" ") + words[w];
        int n = 0;
        int32_t* arr = qwen3_asr_tokenize(ctx, word.c_str(), &n);
        if (arr && n > 0) {
            for (int i = 0; i < n; i++)
                text_suffix_ids.push_back(arr[i]);
        }
        free(arr);
        ts_offsets_in_suffix.push_back((int)text_suffix_ids.size());
        text_suffix_ids.push_back((int32_t)kQwen3FaTimestampTokenId);
        ts_offsets_in_suffix.push_back((int)text_suffix_ids.size());
        text_suffix_ids.push_back((int32_t)kQwen3FaTimestampTokenId);
    }
}

extern "C" void qwen3_asr_align_streaming_reset(struct qwen3_asr_context* ctx) {
    if (!ctx)
        return;
    ctx->align_stream.initialized = false;
    ctx->align_stream.N_enc_committed = 0;
    ctx->align_stream.n_text_tokens = 0;
    ctx->align_stream.text_ids.clear();
    ctx->align_stream.words_signature.clear();
    ctx->align_stream.raw_log_mel_TM.clear();
    ctx->align_stream.T_committed_mel = 0;
    // KV cache buffer stays alive — the next call may reuse it.
    qwen3_asr_kv_reset(ctx);
}

extern "C" int qwen3_asr_align_words_streaming(struct qwen3_asr_context* ctx, const float* samples, int n_samples,
                                               const char** words, int n_words, bool reset, int64_t* out_start_ms,
                                               int64_t* out_end_ms, float* out_confidence) {
    if (!ctx || !samples || n_samples <= 0 || !words || n_words <= 0 || !out_start_ms || !out_end_ms)
        return -1;

    const bool probe_vram = []() {
        const char* e = std::getenv("QWEN3_FA_PROFILE_VRAM");
        return e && *e && std::atoi(e) > 0;
    }();
    // Fine-grained timing breakdown for the streaming path — used to
    // determine which inner stage (encoder vs body forward vs other) is
    // the actual bottleneck. The handoff's "body forward dominates"
    // premise turned out to be wrong on this model (q4_k_m FA, RTX 3060
    // host, CPU LLM body), so we keep this in tree as a permanent
    // diagnostic and not just temporary instrumentation.
    const bool probe_time = []() {
        const char* e = std::getenv("QWEN3_FA_PROFILE_TIME");
        return e && *e && std::atoi(e) > 0;
    }();
    using clk_t = std::chrono::steady_clock;
    auto vram_used_mib = [&]() -> double {
        size_t f = 0, t = 0;
        ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (gpu)
            ggml_backend_dev_memory(gpu, &f, &t);
        return (double)(t - f) / 1048576.0;
    };
    auto probe = [&](const char* label) {
        if (!probe_vram)
            return;
        fprintf(stderr, "  [fa-vram-stream %-16s] gpu_used=%7.1f MiB\n", label, vram_used_mib());
    };
    auto t_entry = clk_t::now();
    int64_t t_mel_ms = 0, t_enc_ms = 0, t_embed_ms = 0, t_body_ms = 0, t_argmax_ms = 0;
    probe("entry");

    // Build a cheap stable signature of the word list so we can detect
    // a different paragraph being pushed in mid-stream and force a reset.
    std::string sig;
    sig.reserve((size_t)n_words * 8);
    for (int i = 0; i < n_words; i++) {
        sig.append(words[i]);
        sig.push_back('\n');
    }

    bool need_reset = reset || !ctx->align_stream.initialized || ctx->align_stream.words_signature != sig;

    // K/V cache reuse across partials is a separate (drift-introducing)
    // optimization on top of the bit-identical mel cache.
    //
    // **Default OFF.** Empirically the body forward isn't dominated by
    // per-token cost: dropping from N_enc + n_text (~435) tokens to
    // Δ + n_text (~265) only saves ~10-30 ms while sacrificing strict
    // bit-identity (the FA encoder is fully bidirectional, so cached
    // audio K/V drifts as new chunks shift the encoder's old-frame
    // embeds). The mel cache alone delivers ~150 ms/partial wallclock
    // savings AND zero word-timing drift, which is the right default.
    //
    // Set QWEN3_FA_STREAMING_KV=1 to opt in to the K/V cache reuse
    // (faster by ~10-30 ms per partial at the tail; ~720 ms max
    // word-timing drift on the smoke paragraph). Useful when latency
    // matters more than bit-identical word boundaries.
    const bool kv_reuse = []() {
        const char* e = std::getenv("QWEN3_FA_STREAMING_KV");
        if (!e || !*e)
            return false;
        return std::atoi(e) != 0;
    }();

    if (need_reset) {
        qwen3_asr_align_streaming_reset(ctx);
        ctx->align_stream.words_signature = sig;
    }

    // ---- Mel (incremental) ----
    //
    // Whisper-style STFT + mel-filter + log10 is time-local: frame t only
    // depends on samples [t*hop - n_fft/2, t*hop + n_fft/2). We cache the
    // RAW (pre-normalization) log-mel across partials and only run STFT
    // for the newly-arrived frames. The Whisper GlobalClipMax norm step
    // IS global, so we apply it to a temp copy of the cumulative buffer
    // each partial — that's a flat O(n_mels*T) scan and negligible cost.
    //
    // For a 32 s paragraph this cuts the per-partial mel cost from
    // ~200 ms (full STFT) to ~10-20 ms (just the Δ_audio frames). Mel
    // output is bit-identical to qwen3_asr_compute_mel on the same
    // cumulative samples (we manually verify under
    // QWEN3_FA_MEL_VERIFY=1).
    auto t_stage = clk_t::now();
    crisp_audio_context* ca = qwen3_asr_get_audio(ctx);
    if (!ca) {
        fprintf(stderr, "qwen3_asr[align-stream]: audio tower unavailable\n");
        return -2;
    }
    const int hop = (int)ctx->model.hparams.hop_length;
    const int n_mels = (int)ctx->model.hparams.n_mels;
    const int n_fft = (int)ctx->model.hparams.n_fft;
    const int T_total = n_samples / hop; // matches qwen3_asr_compute_mel's drop_last_frame T
    if (T_total <= 0) {
        fprintf(stderr, "qwen3_asr[align-stream]: audio too short (n_samples=%d < hop=%d)\n", n_samples, hop);
        return -2;
    }
    // Only cache frames whose entire STFT window lies inside the current
    // sample buffer. A frame at index t reads samples [t*hop - n_fft/2,
    // t*hop + n_fft/2); if its right edge extends past n_samples the
    // missing tail is zero-padded — fine for the current call but stale
    // once more audio arrives (the one-shot mel at the larger n_samples
    // would use REAL audio there). T_safe is the largest frame index
    // whose right edge is still strictly inside the buffer.
    const int T_safe_raw = (n_samples - n_fft / 2) / hop;
    const int T_safe = std::max(0, std::min(T_safe_raw, T_total));

    if (need_reset) {
        ctx->align_stream.raw_log_mel_TM.clear();
        ctx->align_stream.T_committed_mel = 0;
    }
    int T_committed = ctx->align_stream.T_committed_mel;
    if (T_committed > T_safe) {
        // Audio shrank or boundary moved backwards — trim cache.
        ctx->align_stream.raw_log_mel_TM.resize((size_t)T_safe * n_mels);
        ctx->align_stream.T_committed_mel = T_safe;
        T_committed = T_safe;
    }
    // 1. Append newly-safe frames [T_committed, T_safe) to the persistent cache.
    if (T_safe > T_committed) {
        int n_mels_got = 0;
        float* new_safe = crisp_audio_compute_log_mel_range(ca, samples, n_samples, T_committed, T_safe, &n_mels_got);
        if (!new_safe || n_mels_got != n_mels) {
            free(new_safe);
            fprintf(stderr, "qwen3_asr[align-stream]: incremental mel failed (safe range [%d, %d))\n", T_committed,
                    T_safe);
            return -2;
        }
        const size_t n_new = (size_t)(T_safe - T_committed) * n_mels;
        ctx->align_stream.raw_log_mel_TM.insert(ctx->align_stream.raw_log_mel_TM.end(), new_safe, new_safe + n_new);
        free(new_safe);
        ctx->align_stream.T_committed_mel = T_safe;
    }
    // 2. Compute the "unsafe" frames [T_safe, T_total) on every call —
    //    their right-context will fill in as more audio arrives, so we
    //    can't cache them bit-identically. Held in a temporary stage
    //    buffer that's freed at end of this call.
    std::vector<float> unsafe_TM;
    if (T_total > T_safe) {
        int n_mels_got = 0;
        float* unsafe = crisp_audio_compute_log_mel_range(ca, samples, n_samples, T_safe, T_total, &n_mels_got);
        if (!unsafe || n_mels_got != n_mels) {
            free(unsafe);
            fprintf(stderr, "qwen3_asr[align-stream]: incremental mel failed (unsafe range [%d, %d))\n", T_safe,
                    T_total);
            return -2;
        }
        const size_t n_unsafe = (size_t)(T_total - T_safe) * n_mels;
        unsafe_TM.assign(unsafe, unsafe + n_unsafe);
        free(unsafe);
    }
    probe("after-mel");
    t_mel_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk_t::now() - t_stage).count();

    // ---- Transpose (T, n_mels) → (n_mels, T) and apply GlobalClipMax ----
    //
    // We stitch two segments here: the cached safe frames [0, T_safe)
    // from raw_log_mel_TM, followed by the freshly-computed unsafe
    // frames [T_safe, T_total) from unsafe_TM. Both are (T, n_mels)
    // row-major; the loop body handles the index → row-major-(n_mels, T)
    // transpose in one pass.
    t_stage = clk_t::now();
    std::vector<float> mel_for_encoder((size_t)n_mels * T_total);
    {
        const float* safe_src = ctx->align_stream.raw_log_mel_TM.data();
        const float* unsafe_src = unsafe_TM.data();
        for (int t = 0; t < T_total; t++) {
            const float* row = (t < T_safe) ? (safe_src + (size_t)t * n_mels)
                                            : (unsafe_src + (size_t)(t - T_safe) * n_mels);
            for (int m = 0; m < n_mels; m++) {
                mel_for_encoder[(size_t)m * T_total + t] = row[m];
            }
        }
        crisp_audio_apply_clip_max_norm(mel_for_encoder.data(), n_mels, T_total);
    }

    // Optional verification: compare against the one-shot mel each call.
    // QWEN3_FA_MEL_VERIFY=1 turns it on. Bails out + logs if drift is
    // non-negligible.
    if (const char* mv = std::getenv("QWEN3_FA_MEL_VERIFY"); mv && std::atoi(mv) > 0) {
        int gold_n_mels = 0, gold_T = 0;
        float* gold = qwen3_asr_compute_mel(ctx, samples, n_samples, &gold_n_mels, &gold_T);
        if (gold && gold_n_mels == n_mels && gold_T == T_total) {
            double max_diff = 0;
            int n_nonzero = 0;
            int first_t = -1, first_m = -1;
            float first_gold = 0, first_mine = 0;
            for (int m = 0; m < n_mels; m++) {
                for (int t = 0; t < T_total; t++) {
                    const size_t i = (size_t)m * T_total + t;
                    double d = std::abs((double)gold[i] - (double)mel_for_encoder[i]);
                    if (d > max_diff)
                        max_diff = d;
                    if (d > 1e-5) {
                        n_nonzero++;
                        if (first_t < 0) {
                            first_t = t;
                            first_m = m;
                            first_gold = gold[i];
                            first_mine = mel_for_encoder[i];
                        }
                    }
                }
            }
            fprintf(stderr,
                    "  [fa-mel-verify] T=%d max_abs_diff=%.6f n_diff_gt_1e5=%d first_diff(t=%d,m=%d) gold=%.6f "
                    "mine=%.6f T_committed_pre=%d\n",
                    T_total, max_diff, n_nonzero, first_t, first_m, first_gold, first_mine,
                    ctx->align_stream.T_committed_mel - (T_total - T_committed));
        }
        free(gold);
    }

    int N_enc = 0, pdim = 0;
    float* audio_embeds = qwen3_asr_run_encoder(ctx, mel_for_encoder.data(), n_mels, T_total, &N_enc, &pdim);
    if (!audio_embeds) {
        fprintf(stderr, "qwen3_asr[align-stream]: encoder failed\n");
        return -3;
    }
    probe("after-encoder");
    t_enc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk_t::now() - t_stage).count();

    // Encoder shrinking would indicate a non-monotonic call — force reset.
    if (!need_reset && N_enc < ctx->align_stream.N_enc_committed) {
        free(audio_embeds);
        qwen3_asr_align_streaming_reset(ctx);
        // Try again from scratch via a tail-recursive call. (One-shot;
        // the recursive call will hit the reset path.)
        return qwen3_asr_align_words_streaming(ctx, samples, n_samples, words, n_words, /*reset=*/true,
                                               out_start_ms, out_end_ms, out_confidence);
    }

    const auto& hp = ctx->model.hparams;
    const int audio_start_id = (int)hp.audio_start_token_id;
    const int audio_end_id = (int)hp.audio_end_token_id;
    const int audio_pad_id = (int)hp.audio_pad_token_id;

    // Build / re-use the text-suffix token list.
    std::vector<int> ts_offsets_in_suffix; // recomputed on reset
    if (need_reset) {
        ctx->align_stream.text_ids.clear();
        qwen3_fa_build_text_suffix(ctx, words, n_words, audio_end_id, ctx->align_stream.text_ids,
                                   ts_offsets_in_suffix);
        ctx->align_stream.n_text_tokens = (int)ctx->align_stream.text_ids.size();
    } else {
        // Reconstruct ts_offsets_in_suffix from the cached text_ids by
        // scanning for TIMESTAMP_TOKEN_ID. Cheaper than caching the
        // vector (we'd have to keep it in sync with text_ids anyway).
        ts_offsets_in_suffix.reserve((size_t)(2 * n_words));
        for (int i = 0; i < (int)ctx->align_stream.text_ids.size(); i++) {
            if (ctx->align_stream.text_ids[i] == kQwen3FaTimestampTokenId)
                ts_offsets_in_suffix.push_back(i);
        }
        if ((int)ts_offsets_in_suffix.size() != 2 * n_words) {
            // Word list shape mismatch — fall back to reset.
            free(audio_embeds);
            qwen3_asr_align_streaming_reset(ctx);
            return qwen3_asr_align_words_streaming(ctx, samples, n_samples, words, n_words, /*reset=*/true,
                                                   out_start_ms, out_end_ms, out_confidence);
        }
    }

    // ---- Decide cache sizing + n_past ----
    const int n_text = ctx->align_stream.n_text_tokens;
    int n_past = 0;
    int n_input_tokens = 0;
    int n_audio_rows_in_input = 0; // how many of the leading input rows are audio pads
    int N_enc_committed_pre = ctx->align_stream.N_enc_committed;

    if (need_reset || !kv_reuse) {
        // Full prompt: [audio_start, N_enc audio_pads, audio_end, words+ts].
        // The `!kv_reuse` branch runs each partial without reusing prior
        // K/V — bit-identical to the one-shot path, but still benefits
        // from the cached mel.
        n_audio_rows_in_input = N_enc;
        n_input_tokens = 1 + N_enc + n_text; // +1 for audio_start
        n_past = 0;
    } else {
        // Just the delta + text suffix re-forward.
        const int delta = N_enc - N_enc_committed_pre;
        n_audio_rows_in_input = delta;
        n_input_tokens = delta + n_text;
        n_past = 1 + N_enc_committed_pre; // audio_start + committed audio_pads
    }

    // Right-size the KV cache. On reset we overshoot generously so
    // subsequent partials never trigger a realloc (which would zero
    // kv_n_used and destroy the audio K/V we just cached). Default
    // ceiling assumes paragraph length ≤ ~60 s of audio:
    //   audio @ ~12.5 Hz post-cnn × 60 s ≈ 750 slots
    //   + audio_start + audio_end + (~200 token text suffix worst-case)
    //   + headroom for the LIS argmax row scan
    //   ≈ 1024
    int kv_min_ctx = 0;
    if (const char* e = std::getenv("QWEN3_FA_KV_MIN_CTX"))
        kv_min_ctx = std::atoi(e);
    const bool full_forward = need_reset || !kv_reuse;
    // Oversize the cache only on the first partial of a kv_reuse session,
    // so subsequent resume partials can grow the prompt without triggering
    // a realloc (which would zero kv_n_used and drop the audio K/V). In
    // every other case (one-shot, !kv_reuse, mid-session resume) we
    // right-size to this call's prompt + 16 — matches the prior
    // PLAN #69f VRAM target (~340-390 MiB recovered vs the historical
    // 4096-slot floor; ~117 MiB more recovered vs an unconditional 1024).
    const int kv_baseline = (need_reset && kv_reuse) ? 1024 : 16;
    const int kv_target = std::max({n_past + n_input_tokens + kv_baseline, kv_min_ctx, ctx->kv_max_ctx});
    if (!full_forward && kv_target > ctx->kv_max_ctx) {
        // Resume path can't grow the cache without losing the audio K/V.
        // Force a reset and re-run from scratch.
        free(audio_embeds);
        qwen3_asr_align_streaming_reset(ctx);
        return qwen3_asr_align_words_streaming(ctx, samples, n_samples, words, n_words, /*reset=*/true, out_start_ms,
                                               out_end_ms, out_confidence);
    }
    if (!qwen3_asr_kv_init(ctx, kv_target)) {
        free(audio_embeds);
        fprintf(stderr, "qwen3_asr[align-stream]: kv_init failed\n");
        return -5;
    }
    if (full_forward) {
        // Full forward path — cache cursor starts at 0. align_streaming_reset
        // already called kv_reset on the reset branch; for !kv_reuse we
        // do it here explicitly.
        qwen3_asr_kv_reset(ctx);
    } else {
        // Rewind the cursor so the upcoming forward writes start at
        // n_past = 1 + N_enc_committed. (This is bookkeeping for the
        // graph builder's GGML_ASSERT(Lk <= kv_max_ctx); the actual
        // cache slots [0, n_past) are preserved.)
        ctx->kv_n_used = n_past;
    }
    probe("after-kv_init");

    // ---- Build the input_ids stream for embedding + later argmax ----
    std::vector<int32_t> input_ids;
    input_ids.reserve((size_t)n_input_tokens);
    if (full_forward) {
        input_ids.push_back((int32_t)audio_start_id);
        for (int i = 0; i < N_enc; i++)
            input_ids.push_back((int32_t)audio_pad_id);
    } else {
        for (int i = 0; i < n_audio_rows_in_input; i++)
            input_ids.push_back((int32_t)audio_pad_id);
    }
    for (int t : ctx->align_stream.text_ids)
        input_ids.push_back(t);
    if ((int)input_ids.size() != n_input_tokens) {
        free(audio_embeds);
        fprintf(stderr, "qwen3_asr[align-stream]: input_ids size mismatch (%zu vs %d)\n", input_ids.size(),
                n_input_tokens);
        return -8;
    }

    // ---- Embed + splice audio into audio_pad slots ----
    t_stage = clk_t::now();
    float* text_embeds = qwen3_asr_embed_tokens(ctx, input_ids.data(), n_input_tokens);
    if (!text_embeds) {
        free(audio_embeds);
        fprintf(stderr, "qwen3_asr[align-stream]: embed failed\n");
        return -4;
    }
    probe("after-embed");
    {
        // Which audio_embeds rows to splice in. For full_forward paths
        // that's [0, N_enc); for the K/V-reuse resume path it's
        // [N_enc_committed, N_enc).
        int splice_src_base = full_forward ? 0 : N_enc_committed_pre;
        int spliced = 0;
        for (int i = 0; i < n_input_tokens; i++) {
            if (input_ids[i] == audio_pad_id) {
                std::memcpy(text_embeds + (size_t)i * pdim, audio_embeds + (size_t)(splice_src_base + spliced) * pdim,
                            (size_t)pdim * sizeof(float));
                spliced++;
            }
        }
        if (spliced != n_audio_rows_in_input) {
            free(text_embeds);
            free(audio_embeds);
            fprintf(stderr, "qwen3_asr[align-stream]: audio splice mismatch (%d vs %d)\n", spliced,
                    n_audio_rows_in_input);
            return -9;
        }
    }
    free(audio_embeds);
    t_embed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk_t::now() - t_stage).count();

    // ---- Forward ----
    t_stage = clk_t::now();
    int n_t_out = 0, H = 0;
    float* logits = qwen3_asr_run_aligner_core(ctx, text_embeds, n_input_tokens, n_past, &n_t_out, &H);
    free(text_embeds);
    if (!logits) {
        fprintf(stderr, "qwen3_asr[align-stream]: aligner forward failed\n");
        return -6;
    }
    probe("after-aligner");
    t_body_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk_t::now() - t_stage).count();

    // ---- Argmax timestamps ----
    t_stage = clk_t::now();
    // Convert ts_offsets_in_suffix (offsets within text_suffix) to
    // logit row indices. The text suffix begins at:
    //   full_forward: 1 + N_enc      (audio_start at row 0, then N_enc audio_pads)
    //   KV resume:    n_audio_rows_in_input  (= Δ, no audio_start in the slice)
    const int text_suffix_logit_base = full_forward ? (1 + N_enc) : n_audio_rows_in_input;
    std::vector<int> ts_logit_rows;
    ts_logit_rows.reserve((size_t)(2 * n_words));
    for (int off : ts_offsets_in_suffix)
        ts_logit_rows.push_back(text_suffix_logit_base + off);

    int rc = qwen3_fa_extract_word_timings(logits, n_input_tokens, H, ts_logit_rows, n_words, out_start_ms, out_end_ms,
                                            out_confidence);
    free(logits);
    probe("after-argmax");
    t_argmax_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk_t::now() - t_stage).count();

    if (rc == 0) {
        // Optional audio-aware boundary refinement on the cumulative
        // PCM. Gated by QWEN3_FA_AUDIO_REFINE; no-op when unset →
        // bit-identical to pre-patch streaming output.
        qwen3_fa_audio_refine_bounds(samples, n_samples, /*sample_rate=*/16000,
                                      out_start_ms, out_end_ms, n_words);
        ctx->align_stream.N_enc_committed = N_enc;
        ctx->align_stream.initialized = true;
    }
    if (probe_time) {
        const int64_t total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk_t::now() - t_entry).count();
        fprintf(stderr,
                "  [fa-time-stream %s] mel=%lld enc=%lld embed=%lld body=%lld argmax=%lld TOTAL=%lld ms  "
                "N_enc=%d Δ=%d n_text=%d n_past=%d n_input=%d\n",
                need_reset ? "reset" : "resume", (long long)t_mel_ms, (long long)t_enc_ms, (long long)t_embed_ms,
                (long long)t_body_ms, (long long)t_argmax_ms, (long long)total_ms, N_enc,
                N_enc - N_enc_committed_pre, n_text, n_past, n_input_tokens);
    }
    return rc;
}

extern "C" void qwen3_asr_get_vram_breakdown(qwen3_asr_context* ctx, qwen3_asr_vram_breakdown* out) {
    if (!out) return;
    *out = {};
    if (!ctx) return;
    const bool gpu = ctx->backend && !ggml_backend_is_cpu(ctx->backend);
    if (ctx->model.buf && gpu) {
        out->model_buf_bytes = ggml_backend_buffer_get_size(ctx->model.buf);
    }
    if (ctx->model.buf_cpu) {
        out->model_buf_cpu_bytes = ggml_backend_buffer_get_size(ctx->model.buf_cpu);
    }
    if (ctx->fused_buf && gpu) {
        out->fused_buf_bytes = ggml_backend_buffer_get_size(ctx->fused_buf);
    }
    if (ctx->kv_buf && gpu) {
        out->kv_buf_bytes = ggml_backend_buffer_get_size(ctx->kv_buf);
    }
    if (ctx->sched && gpu) {
        out->sched_buf_gpu_bytes = ggml_backend_sched_get_buffer_size(ctx->sched, ctx->backend);
    }
    if (ctx->audio_ca) {
        crisp_audio_vram_stats a{};
        crisp_audio_get_vram_stats(ctx->audio_ca, &a);
        out->audio_model_buf_bytes      = a.model_buf_bytes;
        out->audio_conv_galloc_bytes    = a.conv_galloc_bytes;
        out->audio_body_sched_gpu_bytes = a.body_sched_gpu_bytes;
    }
}

extern "C" void* qwen3_asr_get_gpu_backend_handle(qwen3_asr_context* ctx) {
    if (!ctx || !ctx->backend) return nullptr;
    return ggml_backend_is_cpu(ctx->backend) ? nullptr : (void*) ctx->backend;
}
