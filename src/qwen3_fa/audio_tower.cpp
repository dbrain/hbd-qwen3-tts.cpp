// audio_tower.cpp — crisp_audio implementation.
//
// Dialect dispatcher + first concrete implementation (qwen_omni).
//
// First dialect: CRISP_AUDIO_DIALECT_QWEN_OMNI
//   - 3× Conv2D (k=3 s=2 p=1) over the (n_mels, T) mel image with GELU
//   - conv_out: linear (480·F_out → d_model)
//   - sinusoidal positional embedding (max_pos × d_model, broadcast over chunks)
//   - n_layers × pre-LN encoder block (LayerNorm, MHA with biases, GELU FFN)
//   - ln_post → proj1 → GELU → proj2 → (n_frames, output_dim)
//
// Caller-tunable scalars (d_model, n_heads, n_layers, n_window, output_dim,
// n_mels, n_fft, hop_length, audio_max_pos) all come from the GGUF metadata
// under <meta_prefix>. Tensor names live under <tensor_prefix>.
//
// Lifted and parameterized from CrispASR's src/qwen3_asr.cpp audio tower
// (Stage 1 + Stage 2). Numerical equivalence is locked in by
// tests/test_qwen3_audio_tower.cpp.

#include "crisp_audio.h"

#include "core/gguf_loader.h"
#include "core/mel.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// ---------------------------------------------------------------------------
// Hyper-parameters (read from GGUF at load time)
// ---------------------------------------------------------------------------

struct hparams {
    // Mel preprocessor
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 400;
    uint32_t win_length = 400;
    uint32_t hop_length = 160;

    // Encoder
    uint32_t n_layers = 18;
    uint32_t d_model = 896;
    uint32_t n_heads = 14;
    uint32_t head_dim = 64;
    uint32_t ff_dim = 3584;
    uint32_t conv_ch = 480;
    uint32_t output_dim = 1024;
    uint32_t max_source_pos = 1500;

    // Chunking
    uint32_t n_window = 50;        // (full chunk = n_window * 2 mel frames)
    uint32_t n_window_infer = 800; // not used by Stage-2 encoder graph

    // Attention-mask shape:
    //   0 = full (all post-cnn frames attend to each other) — qwen3-asr's
    //       eager_attention_forward ignores cu_seqlens, so this matches HF.
    //   1 = windowed — encoder attention is block-diagonal across windows of
    //       (T_chunk_out * (n_window_infer / (n_window*2))) frames AND
    //       padding-frame keys are masked off. BidirLM-Omni uses this.
    uint32_t attn_window_mode = 0;
};

// ---------------------------------------------------------------------------
// Per-layer + tower tensor containers
// ---------------------------------------------------------------------------

struct layer_block {
    ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor *attn_k_w = nullptr, *attn_k_b = nullptr;
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor *ffn_up_w = nullptr, *ffn_up_b = nullptr;
    ggml_tensor *ffn_down_w = nullptr, *ffn_down_b = nullptr;
};

struct tower {
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;
    ggml_tensor *conv3_w = nullptr, *conv3_b = nullptr;
    ggml_tensor *conv_out_w = nullptr, *conv_out_b = nullptr;

    std::vector<layer_block> blocks;

    ggml_tensor *ln_post_w = nullptr, *ln_post_b = nullptr;
    ggml_tensor *proj1_w = nullptr, *proj1_b = nullptr;
    ggml_tensor *proj2_w = nullptr, *proj2_b = nullptr;

    // Optional — present in GGUFs that bake the WhisperFeatureExtractor
    // mel filterbank + Hann window into the model.
    ggml_tensor* mel_filters = nullptr;
    ggml_tensor* mel_window = nullptr;
};

constexpr float kLayerNormEps = 1e-5f;

} // namespace

// ---------------------------------------------------------------------------
// Public context — opaque to the C ABI
// ---------------------------------------------------------------------------

struct crisp_audio_context {
    crisp_audio_dialect dialect = CRISP_AUDIO_DIALECT_AUTO;

    hparams hp;
    tower w;
    std::vector<float> sin_pe; // (max_source_pos, d_model) row-major

    ggml_context* model_ctx = nullptr;
    ggml_backend_buffer_t model_buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    // Phase-1.5 graph split:
    //   conv front-end is built ONCE at init into a persistent ggml_context
    //   and reserved on a dedicated `ggml_gallocr_t`. Encode reuses the same
    //   cgraph nodes + GPU offsets across every chunk AND every paragraph,
    //   so the ggml-cuda graph-capture cache replays after the first call.
    //   Bypasses the sched (uses `ggml_backend_graph_compute` directly).
    //   body still uses a sched because N_padded varies per paragraph; the
    //   body cgraph is rebuilt into `compute_meta` per encode.
    ggml_context* conv_ctx = nullptr;
    ggml_cgraph* conv_gf = nullptr;
    ggml_gallocr_t conv_galloc = nullptr;
    std::vector<uint8_t> conv_meta;
    ggml_backend_sched_t body_sched = nullptr;
    std::vector<uint8_t> compute_meta;

    int n_threads = 4;
    int verbosity = 1;
};

// ---------------------------------------------------------------------------
// FFT — same Cooley-Tukey routine qwen3_asr.cpp uses, lifted unchanged.
// Handles n_fft=400 (= 2^4 * 25) by recursing down to a 25-point DFT.
// ---------------------------------------------------------------------------

namespace {

void crisp_audio_dft(const float* in, int N, float* out) {
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

void crisp_audio_fft_recursive(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        crisp_audio_dft(in, N, out);
        return;
    }
    float* even = in + N;
    for (int i = 0; i < half_N; i++)
        even[i] = in[2 * i];
    float* even_fft = out + 2 * N;
    crisp_audio_fft_recursive(even, half_N, even_fft);
    float* odd = even;
    for (int i = 0; i < half_N; i++)
        odd[i] = in[2 * i + 1];
    float* odd_fft = even_fft + N;
    crisp_audio_fft_recursive(odd, half_N, odd_fft);
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

void crisp_audio_fft_wrapper(const float* in, int N, float* out) {
    static thread_local std::vector<float> scratch_in;
    static thread_local std::vector<float> scratch_out;
    if ((int)scratch_in.size() < 4 * N)
        scratch_in.assign((size_t)4 * N, 0.0f);
    if ((int)scratch_out.size() < 8 * N)
        scratch_out.assign((size_t)8 * N, 0.0f);
    std::memcpy(scratch_in.data(), in, (size_t)N * sizeof(float));
    crisp_audio_fft_recursive(scratch_in.data(), N, scratch_out.data());
    std::memcpy(out, scratch_out.data(), (size_t)(2 * N) * sizeof(float));
}

const char* default_or(const char* p, const char* d) {
    return (p && *p) ? p : d;
}

// ---------------------------------------------------------------------------
// Model loading — pull weights + hparams from the GGUF.
// ---------------------------------------------------------------------------

bool load_model(crisp_audio_context& ctx, const char* path, const crisp_audio_params& params) {
    const std::string tprefix = default_or(params.tensor_prefix, "audio.");
    const std::string mprefix = default_or(params.meta_prefix, "crisp_audio.");

    // ---- pass 1: hparams via metadata-only context ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx) {
            fprintf(stderr, "crisp_audio: cannot open %s\n", path);
            return false;
        }
        auto& hp = ctx.hp;
        // Pick the key prefix the file actually uses. We try the caller's
        // requested prefix first; if its canonical "d_model" key isn't
        // present we fall back to the qwen3-asr key layout so existing
        // qwen3-asr GGUFs work without re-conversion.
        std::string ap = mprefix; // audio-fields prefix
        std::string tp;           // top-level (sr/n_fft/etc.) prefix
        const std::string canonical_probe = mprefix + "d_model";
        if (gguf_find_key(gctx, canonical_probe.c_str()) < 0) {
            const std::string qwen_probe = "qwen3asr.audio.d_model";
            if (gguf_find_key(gctx, qwen_probe.c_str()) >= 0) {
                ap = "qwen3asr.audio.";
                tp = "qwen3asr.";
            } else {
                tp = mprefix; // both stay at requested prefix; defaults will apply
            }
        } else {
            tp = mprefix;
        }
        auto ua = [&](const char* k, uint32_t d) { return core_gguf::kv_u32(gctx, (ap + k).c_str(), d); };
        auto ut = [&](const char* k, uint32_t d) { return core_gguf::kv_u32(gctx, (tp + k).c_str(), d); };
        // Top-level audio config (sample rate / FFT / window).
        hp.sample_rate = ut("sample_rate", hp.sample_rate);
        hp.n_mels = ut("n_mels", hp.n_mels);
        hp.n_fft = ut("n_fft", hp.n_fft);
        hp.win_length = ut("win_length", hp.win_length);
        hp.hop_length = ut("hop_length", hp.hop_length);
        hp.n_window = ut("n_window", hp.n_window);
        hp.n_window_infer = ut("n_window_infer", hp.n_window_infer);
        hp.attn_window_mode = ua("attn_window_mode", hp.attn_window_mode);
        // Encoder hparams.
        hp.n_layers = ua("n_layers", hp.n_layers);
        hp.d_model = ua("d_model", hp.d_model);
        hp.n_heads = ua("n_heads", hp.n_heads);
        hp.head_dim = ua("head_dim", hp.head_dim);
        hp.ff_dim = ua("ff_dim", hp.ff_dim);
        hp.conv_ch = ua("conv_channels", hp.conv_ch);
        hp.max_source_pos = ua("max_source_pos", hp.max_source_pos);
        // qwen3-asr converter wrote `proj_dim`; BidirLM/crisp_audio writes
        // `output_dim`. Try both so the same loader works on both GGUF
        // dialects without forcing the user to re-run the converter.
        hp.output_dim = ua("output_dim", ua("proj_dim", hp.output_dim));
        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: weights via shared loader ----
    //
    // Filter to `<tensor_prefix>*` only. Multi-component GGUFs (qwen3-asr,
    // qwen3-forced-aligner) hold the LLM body weights under unrelated
    // tensor names; loading the whole file here would duplicate the entire
    // GGUF in VRAM against the parent qwen3_asr's own load (HANDOFF-fa-
    // aligner-vram Phase-A probe found 499 MiB of duplicate Q4_K weights).
    // Phase-A acceptance confirmed by re-running the probe after this lever.
    core_gguf::WeightLoad wl;
    {
        struct PrefixFilter {
            const char* prefix;
            size_t      prefix_len;
        };
        PrefixFilter pf{tprefix.c_str(), tprefix.size()};
        auto match_prefix = +[](const char* tname, void* user) -> bool {
            const auto* p = static_cast<const PrefixFilter*>(user);
            if (!tname) return false;
            return std::strncmp(tname, p->prefix, p->prefix_len) == 0;
        };
        if (!core_gguf::load_weights_filtered(path, ctx.backend, "crisp_audio",
                                              match_prefix, &pf, wl)) {
            return false;
        }
    }
    ctx.model_ctx = wl.ctx;
    ctx.model_buf = wl.buf;
    ctx.tensors = std::move(wl.tensors);

    auto get = [&](const std::string& name) -> ggml_tensor* {
        auto it = ctx.tensors.find(name);
        return it != ctx.tensors.end() ? it->second : nullptr;
    };
    auto require = [&](const std::string& name) -> ggml_tensor* {
        auto t = get(name);
        if (!t) {
            fprintf(stderr, "crisp_audio: required tensor '%s' missing\n", name.c_str());
        }
        return t;
    };

    auto& w = ctx.w;
    w.conv1_w = require(tprefix + "conv.1.weight");
    w.conv1_b = require(tprefix + "conv.1.bias");
    w.conv2_w = require(tprefix + "conv.2.weight");
    w.conv2_b = require(tprefix + "conv.2.bias");
    w.conv3_w = require(tprefix + "conv.3.weight");
    w.conv3_b = require(tprefix + "conv.3.bias");
    w.conv_out_w = require(tprefix + "conv_out.weight");
    w.conv_out_b = get(tprefix + "conv_out.bias");
    w.ln_post_w = require(tprefix + "ln_post.weight");
    w.ln_post_b = require(tprefix + "ln_post.bias");
    w.proj1_w = require(tprefix + "proj1.weight");
    w.proj1_b = require(tprefix + "proj1.bias");
    w.proj2_w = require(tprefix + "proj2.weight");
    w.proj2_b = require(tprefix + "proj2.bias");
    w.mel_filters = get(tprefix + "mel_filters");
    w.mel_window = get(tprefix + "mel_window");

    if (!w.conv1_w || !w.conv2_w || !w.conv3_w || !w.conv_out_w || !w.ln_post_w || !w.proj1_w || !w.proj2_w) {
        return false;
    }

    w.blocks.resize(ctx.hp.n_layers);
    for (uint32_t i = 0; i < ctx.hp.n_layers; i++) {
        auto& b = w.blocks[i];
        char buf[160];
        auto rq = [&](const char* suf) {
            std::snprintf(buf, sizeof(buf), "%sblk.%u.%s", tprefix.c_str(), i, suf);
            return require(buf);
        };
        b.attn_norm_w = rq("attn_norm.weight");
        b.attn_norm_b = rq("attn_norm.bias");
        b.attn_q_w = rq("attn_q.weight");
        b.attn_q_b = rq("attn_q.bias");
        b.attn_k_w = rq("attn_k.weight");
        b.attn_k_b = rq("attn_k.bias");
        b.attn_v_w = rq("attn_v.weight");
        b.attn_v_b = rq("attn_v.bias");
        b.attn_out_w = rq("attn_out.weight");
        b.attn_out_b = rq("attn_out.bias");
        b.ffn_norm_w = rq("ffn_norm.weight");
        b.ffn_norm_b = rq("ffn_norm.bias");
        b.ffn_up_w = rq("ffn_up.weight");
        b.ffn_up_b = rq("ffn_up.bias");
        b.ffn_down_w = rq("ffn_down.weight");
        b.ffn_down_b = rq("ffn_down.bias");
        if (!b.attn_q_w)
            return false;
    }

    // Precompute sinusoidal positional embedding (Whisper-style).
    {
        const int C = (int)ctx.hp.d_model;
        const int L = (int)ctx.hp.max_source_pos;
        const int half = C / 2;
        const float log_inc = std::log(10000.0f) / (float)(half - 1);
        std::vector<float> inv_t(half);
        for (int i = 0; i < half; i++)
            inv_t[i] = std::exp(-log_inc * (float)i);
        ctx.sin_pe.assign((size_t)L * C, 0.0f);
        for (int p = 0; p < L; p++) {
            float* row = ctx.sin_pe.data() + (size_t)p * C;
            for (int i = 0; i < half; i++) {
                float angle = (float)p * inv_t[i];
                row[i] = std::sin(angle);
                row[half + i] = std::cos(angle);
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Encoder graph builders — qwen_omni dialect, split into two subgraphs.
//
// The Phase-1 split swaps the old monolithic build_graph_qwen_omni for a
// per-chunk Conv2D front-end loop + a single body forward. Old graph held
// 18.4 MiB of conv-stage intermediates *per chunk* — paragraph alignment
// at nc=39 needed ~720 MiB of compute scratch. New conv_subgraph runs
// nc=1 (~18 MiB constant); body_subgraph keeps the same encoder body
// (alias-friendly, ~50 MiB at nc=39).
//
//   conv_subgraph (built per encode, run num_chunks times):
//     mel_chunk (T_chunk × n_mels × 1 × 1)
//       → 3× Conv2D s=2 + GELU
//       → flatten freq → conv_out linear → (d_model, T_chunk_out, 1)
//       → + sinusoidal_pos (broadcast)
//       → cont, reshape (d_model, T_chunk_out)
//       → "conv_out"
//
//   body_subgraph (built per encode, run once):
//     body_in (d_model, N_padded)  ← host-side concat of conv outputs
//       → N × pre-LN encoder block (full bidir self-attn)
//       → ln_post → proj1 → GELU → proj2
//       → "encoder_out" (output_dim, N_padded)
// ---------------------------------------------------------------------------

// Caller owns `g` + `gf` (persistent across encodes). The conv topology is
// fixed by hp (chunk_T derived from n_window, T_chunk_out from 3× stride-2),
// so this is called exactly once at crisp_audio_init_from_file.
void build_conv_subgraph_into(ggml_context* g, ggml_cgraph* gf,
                              crisp_audio_context& ctx, int T_chunk, int T_chunk_out_expected) {
    const auto& hp = ctx.hp;
    const auto& w = ctx.w;
    const int n_mels = (int)hp.n_mels;
    const int d = (int)hp.d_model;

    // nc=1 — one chunk at a time. Loop runs num_chunks times in caller.
    ggml_tensor* mel = ggml_new_tensor_4d(g, GGML_TYPE_F32, T_chunk, n_mels, 1, 1);
    ggml_set_name(mel, "mel_chunk");
    ggml_set_input(mel);

    ggml_tensor* pe_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, d, T_chunk_out_expected, 1);
    ggml_set_name(pe_in, "pe_input");
    ggml_set_input(pe_in);

    auto bias_4d = [&](ggml_tensor* b) {
        return ggml_cast(g, ggml_reshape_4d(g, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32);
    };

    // ggml_conv_2d (im2col + mat_mul) is ~9× faster than ggml_conv_2d_direct
    // on RTX 3060 sm_86: 62 ms vs 580 ms for nc=39 paragraph forward. The
    // CUDA conv2d.cu kernel is a naive nested-loop conv with no tensor-core
    // path; the im2col+MUL_MAT route hits MMQ on the F16 weights. The
    // per-chunk loop here amortises the intermediate-slot cost without
    // changing the kernel choice.
    ggml_tensor* cur = ggml_conv_2d(g, w.conv1_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(g, cur, bias_4d(w.conv1_b));
    cur = ggml_gelu_erf(g, cur);
    cur = ggml_conv_2d(g, w.conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(g, cur, bias_4d(w.conv2_b));
    cur = ggml_gelu_erf(g, cur);
    cur = ggml_conv_2d(g, w.conv3_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(g, cur, bias_4d(w.conv3_b));
    cur = ggml_gelu_erf(g, cur);

    const int T_out = (int)cur->ne[0];
    const int F_out = (int)cur->ne[1];
    const int C_out = (int)cur->ne[2];
    GGML_ASSERT(T_out == T_chunk_out_expected);
    GGML_ASSERT(C_out == (int)hp.conv_ch);

    cur = ggml_cont(g, ggml_permute(g, cur, 2, 0, 1, 3));
    cur = ggml_reshape_3d(g, cur, F_out * C_out, T_out, 1);
    cur = ggml_mul_mat(g, w.conv_out_w, cur);
    if (w.conv_out_b)
        cur = ggml_add(g, cur, w.conv_out_b);

    cur = ggml_add(g, cur, pe_in);

    cur = ggml_cont(g, cur);
    cur = ggml_reshape_2d(g, cur, d, T_chunk_out_expected);

    ggml_set_name(cur, "conv_out");
    ggml_build_forward_expand(gf, cur);
}

ggml_cgraph* build_body_subgraph(crisp_audio_context& ctx, int N_padded) {
    const auto& hp = ctx.hp;
    const auto& w = ctx.w;
    const int d = (int)hp.d_model;
    const int n_heads = (int)hp.n_heads;
    const int head_dim = (int)hp.head_dim;

    ggml_init_params ip{
        ctx.compute_meta.size(),
        ctx.compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* g = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 16384, false);

    ggml_tensor* cur = ggml_new_tensor_2d(g, GGML_TYPE_F32, d, N_padded);
    ggml_set_name(cur, "body_in");
    ggml_set_input(cur);

    ggml_tensor* mask_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, N_padded, N_padded);
    ggml_set_name(mask_in, "attn_mask");
    ggml_set_input(mask_in);

    const float attn_scale = 1.0f / std::sqrt((float)head_dim);
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = w.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_norm(g, cur, kLayerNormEps);
        x = ggml_mul(g, x, b.attn_norm_w);
        x = ggml_add(g, x, b.attn_norm_b);

        ggml_tensor* Q = ggml_add(g, ggml_mul_mat(g, b.attn_q_w, x), b.attn_q_b);
        ggml_tensor* K = ggml_add(g, ggml_mul_mat(g, b.attn_k_w, x), b.attn_k_b);
        ggml_tensor* V = ggml_add(g, ggml_mul_mat(g, b.attn_v_w, x), b.attn_v_b);
        Q = ggml_reshape_3d(g, Q, head_dim, n_heads, N_padded);
        K = ggml_reshape_3d(g, K, head_dim, n_heads, N_padded);
        V = ggml_reshape_3d(g, V, head_dim, n_heads, N_padded);
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

        // ggml_flash_attn_ext doesn't materialise the (N_padded × N_padded ×
        // n_heads × 4B) softmax-scores tensor that the mul_mat → softmax →
        // mul_mat path used to. On Tolstoy-length input (N_padded≈1586),
        // that buffer was ~140 MiB of the encoder body-sched alloc. The
        // encoder mask is all-zeros (full bidirectional self-attention, no
        // per-chunk padding mask wired yet), so nullptr is correct here.
        // Opt out via QWEN3_FA_NO_ENCODER_FA=1 to compare against the
        // legacy path.
        static const bool s_no_enc_fa = std::getenv("QWEN3_FA_NO_ENCODER_FA") != nullptr;
        ggml_tensor* attn;
        if (!s_no_enc_fa) {
            (void) mask_in; // unused on FA path; mask is all-zero in caller
            attn = ggml_flash_attn_ext(g, Q, K, V, /*mask*/ nullptr, attn_scale,
                                       /*max_bias*/ 0.0f, /*logit_softcap*/ 0.0f);
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            // FA output: (head_dim, n_heads, N_padded, 1) — already in the
            // permuted-and-cont layout that the legacy code reaches via
            // `ggml_cont(ggml_permute(attn, 0, 2, 1, 3))`. Reshape to (d, N_padded).
            attn = ggml_reshape_2d(g, attn, d, N_padded);
        } else {
            ggml_tensor* scores = ggml_mul_mat(g, K, Q);
            scores = ggml_add(g, scores, mask_in);
            scores = ggml_soft_max_ext(g, scores, nullptr, attn_scale, 0.0f);

            ggml_tensor* V2 = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
            attn = ggml_mul_mat(g, V2, scores);
            attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
            attn = ggml_reshape_2d(g, attn, d, N_padded);
        }

        attn = ggml_add(g, ggml_mul_mat(g, b.attn_out_w, attn), b.attn_out_b);
        cur = ggml_add(g, residual, attn);

        residual = cur;
        x = ggml_norm(g, cur, kLayerNormEps);
        x = ggml_mul(g, x, b.ffn_norm_w);
        x = ggml_add(g, x, b.ffn_norm_b);
        x = ggml_add(g, ggml_mul_mat(g, b.ffn_up_w, x), b.ffn_up_b);
        x = ggml_gelu_erf(g, x);
        x = ggml_add(g, ggml_mul_mat(g, b.ffn_down_w, x), b.ffn_down_b);
        cur = ggml_add(g, residual, x);
    }

    {
        ggml_tensor* x = ggml_norm(g, cur, kLayerNormEps);
        x = ggml_mul(g, x, w.ln_post_w);
        x = ggml_add(g, x, w.ln_post_b);
        cur = x;
    }
    cur = ggml_add(g, ggml_mul_mat(g, w.proj1_w, cur), w.proj1_b);
    cur = ggml_gelu_erf(g, cur);
    cur = ggml_add(g, ggml_mul_mat(g, w.proj2_w, cur), w.proj2_b);

    ggml_set_name(cur, "encoder_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(g);
    return gf;
}

} // namespace

// ===========================================================================
// Public C API
// ===========================================================================

extern "C" {

struct crisp_audio_params crisp_audio_params_default(void) {
    struct crisp_audio_params p {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.tensor_prefix = nullptr;
    p.meta_prefix = nullptr;
    p.dialect = CRISP_AUDIO_DIALECT_AUTO;
    return p;
}

struct crisp_audio_context* crisp_audio_init_from_file(const char* gguf_path, const struct crisp_audio_params* params) {
    if (!gguf_path)
        return nullptr;
    crisp_audio_params eff = params ? *params : crisp_audio_params_default();

    auto* ctx = new crisp_audio_context();
    ctx->n_threads = eff.n_threads;
    ctx->verbosity = eff.verbosity;
    ctx->dialect = (eff.dialect == CRISP_AUDIO_DIALECT_AUTO) ? CRISP_AUDIO_DIALECT_QWEN_OMNI : eff.dialect;

    // Backend selection — GPU if requested + available, fall back to CPU.
    ctx->backend = nullptr;
    if (eff.use_gpu) {
        ggml_backend_dev_t gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (gdev)
            ctx->backend = ggml_backend_dev_init(gdev, nullptr);
    }
    if (!ctx->backend) {
        ctx->backend = ggml_backend_cpu_init();
    }
    ctx->backend_cpu = ggml_backend_is_cpu(ctx->backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx->backend_cpu) {
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    } else {
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    }

    if (!load_model(*ctx, gguf_path, eff)) {
        crisp_audio_free(ctx);
        return nullptr;
    }

    // Sized to match qwen3_asr's pattern (ggml metadata, not actual data —
    // graph holds ~290 ops for an 18-layer encoder, ~390 for 24 layers, so
    // 16384 is comfortable headroom).
    constexpr int kGraphCapacity = 16384;
    ctx->compute_meta.resize(ggml_tensor_overhead() * kGraphCapacity +
                             ggml_graph_overhead_custom(kGraphCapacity, false));

    std::vector<ggml_backend_t> backends;
    backends.push_back(ctx->backend);
    if (ctx->backend_cpu)
        backends.push_back(ctx->backend_cpu);
    // op_offload=false to match qwen3_asr's behavior — keeps each tensor
    // on its assigned backend rather than letting the scheduler migrate.
    // Body still uses a sched (per-encode rebuild because N_padded varies);
    // conv is cached on a dedicated gallocr (see below).
    ctx->body_sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), kGraphCapacity, false, false);

    // ── Phase-1.5: build conv subgraph once into persistent context, reserve
    // dedicated gallocr. Skipping sched eliminates the cuda-graph capture
    // thrash between conv and body schedulers when they share a backend.
    {
        constexpr int kConvGraphCapacity = 1024; // ~40 nodes actual
        ctx->conv_meta.resize(ggml_tensor_overhead() * kConvGraphCapacity +
                              ggml_graph_overhead_custom(kConvGraphCapacity, false));
        ggml_init_params ip{
            ctx->conv_meta.size(),
            ctx->conv_meta.data(),
            /*no_alloc=*/true,
        };
        ctx->conv_ctx = ggml_init(ip);
        if (!ctx->conv_ctx) {
            fprintf(stderr, "crisp_audio: ggml_init for conv_ctx failed\n");
            crisp_audio_free(ctx);
            return nullptr;
        }
        ctx->conv_gf = ggml_new_graph_custom(ctx->conv_ctx, kConvGraphCapacity, false);

        const int chunk_T = (int)ctx->hp.n_window * 2;
        auto conv_out_len = [](int in_len) { return (in_len + 2 - 3) / 2 + 1; };
        const int T_chunk_out = conv_out_len(conv_out_len(conv_out_len(chunk_T)));
        build_conv_subgraph_into(ctx->conv_ctx, ctx->conv_gf, *ctx, chunk_T, T_chunk_out);

        ctx->conv_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ctx->conv_galloc) {
            fprintf(stderr, "crisp_audio: ggml_gallocr_new failed\n");
            crisp_audio_free(ctx);
            return nullptr;
        }
        if (!ggml_gallocr_reserve(ctx->conv_galloc, ctx->conv_gf)) {
            fprintf(stderr, "crisp_audio: conv gallocr reserve failed\n");
            crisp_audio_free(ctx);
            return nullptr;
        }
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr,
                "crisp_audio: loaded dialect=qwen_omni d_model=%u layers=%u "
                "heads=%u head_dim=%u output_dim=%u n_mels=%u n_window=%u\n",
                ctx->hp.d_model, ctx->hp.n_layers, ctx->hp.n_heads, ctx->hp.head_dim, ctx->hp.output_dim,
                ctx->hp.n_mels, ctx->hp.n_window);
    }
    return ctx;
}

void crisp_audio_free(struct crisp_audio_context* ctx) {
    if (!ctx)
        return;
    if (ctx->conv_galloc)
        ggml_gallocr_free(ctx->conv_galloc);
    if (ctx->conv_ctx)
        ggml_free(ctx->conv_ctx);
    if (ctx->body_sched)
        ggml_backend_sched_free(ctx->body_sched);
    if (ctx->model_buf)
        ggml_backend_buffer_free(ctx->model_buf);
    if (ctx->model_ctx)
        ggml_free(ctx->model_ctx);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

float* crisp_audio_compute_mel(struct crisp_audio_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                               int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->hp;
    if (!ctx->w.mel_filters || !ctx->w.mel_window) {
        fprintf(stderr, "crisp_audio: GGUF missing mel_filters / mel_window — "
                        "regenerate the GGUF with the converter that bakes them in\n");
        return nullptr;
    }

    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int n_mels = (int)hp.n_mels;
    const int n_freqs = n_fft / 2 + 1;

    std::vector<float> hann(n_fft);
    ggml_backend_tensor_get(ctx->w.mel_window, hann.data(), 0, (size_t)n_fft * sizeof(float));
    std::vector<float> filt((size_t)n_freqs * n_mels);
    ggml_backend_tensor_get(ctx->w.mel_filters, filt.data(), 0, filt.size() * sizeof(float));

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
    auto mel = core_mel::compute(samples, n_samples, hann.data(), n_fft, filt.data(), n_freqs, crisp_audio_fft_wrapper,
                                 p, T_ret);
    if (mel.empty())
        return nullptr;

    if (out_n_mels)
        *out_n_mels = n_mels;
    if (out_T_mel)
        *out_T_mel = T_ret;
    float* result = (float*)std::malloc(mel.size() * sizeof(float));
    std::memcpy(result, mel.data(), mel.size() * sizeof(float));
    return result;
}

float* crisp_audio_compute_log_mel_range(struct crisp_audio_context* ctx, const float* samples, int n_samples,
                                         int T_start, int T_end, int* out_n_mels) {
    if (!ctx || !samples || n_samples <= 0 || T_end <= T_start)
        return nullptr;
    const auto& hp = ctx->hp;
    if (!ctx->w.mel_filters || !ctx->w.mel_window) {
        fprintf(stderr, "crisp_audio: GGUF missing mel_filters / mel_window\n");
        return nullptr;
    }

    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int n_mels = (int)hp.n_mels;
    const int n_freqs = n_fft / 2 + 1;
    const int half_n_fft = n_fft / 2;
    const int T = T_end - T_start;

    std::vector<float> hann(n_fft);
    ggml_backend_tensor_get(ctx->w.mel_window, hann.data(), 0, (size_t)n_fft * sizeof(float));
    std::vector<float> filt((size_t)n_freqs * n_mels);
    ggml_backend_tensor_get(ctx->w.mel_filters, filt.data(), 0, filt.size() * sizeof(float));

    // For each frame t ∈ [T_start, T_end), the STFT window covers original
    // samples [t*hop - n_fft/2, t*hop + n_fft/2). Out-of-range positions
    // (negative on the left edge, past n_samples on the right edge) read
    // as zero — matches whisper-style center padding.
    std::vector<float> fft_in((size_t)n_fft);
    std::vector<float> fft_out((size_t)n_fft * 2);
    std::vector<float> mel_tn((size_t)T * n_mels, 0.0f);
    constexpr float kLogEps = 1e-10f;
    for (int ti = 0; ti < T; ti++) {
        const int t_abs = T_start + ti;
        const int win_start_abs = t_abs * hop - half_n_fft;
        // Build windowed frame, zero-padding outside [0, n_samples).
        for (int n = 0; n < n_fft; n++) {
            const int idx = win_start_abs + n;
            const float s = (idx >= 0 && idx < n_samples) ? samples[idx] : 0.0f;
            fft_in[n] = s * hann[n];
        }
        crisp_audio_fft_wrapper(fft_in.data(), n_fft, fft_out.data());
        // Power spectrum.
        // (Stack-allocate to avoid an extra heap alloc per frame.)
        // n_freqs == n_fft/2 + 1, capped at 4097 for our n_fft=400.
        std::vector<float> power((size_t)n_freqs);
        for (int k = 0; k < n_freqs; k++) {
            const float re = fft_out[2 * k];
            const float im = fft_out[2 * k + 1];
            power[k] = re * re + im * im;
        }
        // Mel projection in double precision (matches qwen3 / whisper).
        // filt is laid out as FbLayout::FreqsMels: filt[k * n_mels + m].
        float* mp = mel_tn.data() + (size_t)ti * n_mels;
        for (int m = 0; m < n_mels; m++) {
            double s = 0.0;
            for (int k = 0; k < n_freqs; k++) {
                s += (double)power[k] * (double)filt[(size_t)k * n_mels + m];
            }
            // log10 with MaxClip guard.
            float v = (float)s;
            if (v < kLogEps)
                v = kLogEps;
            mp[m] = std::log10(v);
        }
    }

    // Output layout: (T, n_mels) row-major — frames are contiguous so the
    // streaming caller can append successive ranges with vector::insert /
    // memcpy. Conversion to (n_mels, T) happens on the way into the
    // encoder via a separate transpose.
    float* out = (float*)std::malloc((size_t)T * n_mels * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, mel_tn.data(), (size_t)T * n_mels * sizeof(float));
    if (out_n_mels)
        *out_n_mels = n_mels;
    return out;
}

void crisp_audio_apply_clip_max_norm(float* log_mel, int n_mels, int T_mel) {
    if (!log_mel || n_mels <= 0 || T_mel <= 0)
        return;
    const size_t N = (size_t)n_mels * T_mel;
    float mx = -1e30f;
    for (size_t i = 0; i < N; i++)
        if (log_mel[i] > mx)
            mx = log_mel[i];
    const float floor_v = mx - 8.0f;
    for (size_t i = 0; i < N; i++) {
        float v = log_mel[i];
        if (v < floor_v)
            v = floor_v;
        log_mel[i] = (v + 4.0f) / 4.0f;
    }
}

float* crisp_audio_encode(struct crisp_audio_context* ctx, const float* mel_features, int n_mels, int T_mel,
                          int* out_n_frames, int* out_dim) {
    if (!ctx || !mel_features)
        return nullptr;
    const auto& hp = ctx->hp;
    if (n_mels != (int)hp.n_mels) {
        fprintf(stderr, "crisp_audio: mel mismatch (%d vs %d)\n", n_mels, (int)hp.n_mels);
        return nullptr;
    }

    const int chunk_T = (int)hp.n_window * 2;
    const int num_chunks = (T_mel + chunk_T - 1) / chunk_T;
    const int d = (int)hp.d_model;
    auto conv_out_len = [](int in_len) { return (in_len + 2 - 3) / 2 + 1; };
    const int T_chunk_out = conv_out_len(conv_out_len(conv_out_len(chunk_T)));
    const int N_padded = T_chunk_out * num_chunks;

    std::vector<float> mel_padded((size_t)chunk_T * n_mels * num_chunks, 0.0f);
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int t_start = chunk * chunk_T;
        const int t_end = std::min(t_start + chunk_T, T_mel);
        const int t_len = t_end - t_start;
        for (int f = 0; f < n_mels; f++) {
            for (int t = 0; t < t_len; t++) {
                mel_padded[(size_t)t + chunk_T * ((size_t)f + n_mels * (size_t)chunk)] =
                    mel_features[(size_t)f * T_mel + (size_t)(t_start + t)];
            }
        }
    }

    std::vector<float> mask((size_t)N_padded * N_padded, 0.0f);
    if (hp.attn_window_mode == 1) {
        // BidirLM-style windowed mask: block-diagonal over inference windows
        // of `chunks_per_window * T_chunk_out` valid post-cnn frames. Padding
        // frames at chunk tails are masked off as KEYS (no real query attends
        // to them) but as QUERIES they get a zero row — softmax over -inf row
        // would produce NaN, so we leave padding rows unmasked. Their outputs
        // are discarded by the BidirLM wrapper's pooling step anyway.
        const float kNegInf = -std::numeric_limits<float>::infinity();
        const int chunks_per_window = (hp.n_window > 0) ? std::max(1, (int)(hp.n_window_infer / (hp.n_window * 2))) : 1;
        const int window_aftercnn = chunks_per_window * T_chunk_out;
        std::vector<int> valid_per_chunk(num_chunks);
        for (int c = 0; c < num_chunks; c++) {
            const int t_len = std::min(chunk_T, T_mel - c * chunk_T);
            int v = t_len;
            for (int s = 0; s < 3; s++)
                v = (v - 1) / 2 + 1;
            valid_per_chunk[c] = v;
        }
        // Determine which global frame indices are valid (and therefore have
        // a window assignment).
        std::vector<int> window_id(N_padded, -1);
        int next_window = 0;
        int frames_in_current_window = 0;
        for (int c = 0; c < num_chunks; c++) {
            for (int f = 0; f < valid_per_chunk[c]; f++) {
                if (frames_in_current_window == window_aftercnn) {
                    next_window++;
                    frames_in_current_window = 0;
                }
                window_id[c * T_chunk_out + f] = next_window;
                frames_in_current_window++;
            }
        }
        // Build the mask: for each valid query row i, allow attention only
        // to keys in the same window. For padding query rows (window_id == -1)
        // leave full zero attention — their outputs are pooled out.
        for (int i = 0; i < N_padded; i++) {
            if (window_id[i] < 0)
                continue; // padding query: keep zero row
            for (int j = 0; j < N_padded; j++) {
                if (window_id[j] != window_id[i]) {
                    mask[(size_t)i * N_padded + j] = kNegInf;
                }
            }
        }
    }

    // VRAM probe — dumps allocator buffer, alias_ratio, top-N tensors and
    // per-slot occupancy. `buf_bytes` is passed in by the caller (sched vs
    // gallocr have different size APIs). Called twice (conv + body) when
    // QWEN3_FA_PROFILE_VRAM=1+, with level≥2 producing the detail block.
    auto probe_graph = [&](const char* label, ggml_cgraph* gf, size_t buf_bytes, int level) {
        if (level <= 0)
            return;
        const int nodes = ggml_graph_n_nodes(gf);
        fprintf(stderr,
                "  [fa-vram %-12s] alloc=%6.1f MiB nodes=%d chunk_T=%d num_chunks=%d "
                "N_padded=%d (T_chunk_out=%d)\n",
                label, (double)buf_bytes / 1048576.0, nodes, chunk_T, num_chunks, N_padded, T_chunk_out);
        if (level < 2)
            return;
        size_t sum_bytes = 0;
        size_t sum_bytes_compute = 0;
        std::map<uintptr_t, size_t> unique_offsets;
        int n_compute = 0, n_view = 0, n_leaf_or_input = 0;
        for (int i = 0; i < nodes; i++) {
            ggml_tensor* t = ggml_graph_node(gf, i);
            if (!t) continue;
            size_t nb = ggml_nbytes(t);
            sum_bytes += nb;
            if (ggml_is_view(t) || t->view_src) { n_view++; continue; }
            if (t->flags & GGML_TENSOR_FLAG_INPUT) { n_leaf_or_input++; continue; }
            n_compute++;
            sum_bytes_compute += nb;
            unique_offsets[(uintptr_t)t->data] = nb;
        }
        size_t unique_bytes = 0;
        for (auto& [off, nb] : unique_offsets) unique_bytes += nb;
        fprintf(stderr,
                "  [fa-vram %-12s] sum_all=%.1f MiB sum_compute=%.1f MiB unique_offsets=%zu unique_bytes=%.1f MiB "
                "(compute=%d view=%d input=%d) alias_ratio=%.2fx\n",
                label,
                (double)sum_bytes / 1048576.0, (double)sum_bytes_compute / 1048576.0,
                unique_offsets.size(), (double)unique_bytes / 1048576.0,
                n_compute, n_view, n_leaf_or_input,
                buf_bytes > 0 ? (double)sum_bytes_compute / (double)buf_bytes : 0.0);
        struct probe_node { ggml_tensor* t; size_t nb; int idx; };
        std::vector<probe_node> sorted_nodes;
        sorted_nodes.reserve(n_compute);
        for (int i = 0; i < nodes; i++) {
            ggml_tensor* t = ggml_graph_node(gf, i);
            if (!t) continue;
            if (ggml_is_view(t) || t->view_src) continue;
            if (t->flags & GGML_TENSOR_FLAG_INPUT) continue;
            sorted_nodes.push_back({t, ggml_nbytes(t), i});
        }
        std::sort(sorted_nodes.begin(), sorted_nodes.end(),
                  [](const probe_node& a, const probe_node& b) { return a.nb > b.nb; });
        const int top_nodes = std::min((int)sorted_nodes.size(), 12);
        for (int i = 0; i < top_nodes; i++) {
            auto& pn = sorted_nodes[i];
            fprintf(stderr,
                    "  [fa-vram %-12s]   #%-3d %-14s %-22s ne=[%5lld,%5lld,%5lld,%5lld] %s bytes=%.1f MiB off=0x%lx\n",
                    label, pn.idx,
                    ggml_op_name(pn.t->op),
                    pn.t->name[0] ? pn.t->name : "(unnamed)",
                    (long long)pn.t->ne[0], (long long)pn.t->ne[1],
                    (long long)pn.t->ne[2], (long long)pn.t->ne[3],
                    ggml_type_name(pn.t->type),
                    (double)pn.nb / 1048576.0,
                    (unsigned long)(uintptr_t)pn.t->data);
        }
        std::map<uintptr_t, std::pair<size_t, int>> slot_info;
        for (int i = 0; i < nodes; i++) {
            ggml_tensor* t = ggml_graph_node(gf, i);
            if (!t) continue;
            if (ggml_is_view(t) || t->view_src) continue;
            if (t->flags & GGML_TENSOR_FLAG_INPUT) continue;
            size_t nb = ggml_nbytes(t);
            auto& s2 = slot_info[(uintptr_t)t->data];
            s2.first = std::max(s2.first, nb);
            s2.second++;
        }
        std::vector<std::pair<uintptr_t, std::pair<size_t, int>>> slot_sorted(slot_info.begin(), slot_info.end());
        std::sort(slot_sorted.begin(), slot_sorted.end(),
                  [](auto& a, auto& b) { return a.second.first > b.second.first; });
        for (auto& [off, info] : slot_sorted) {
            fprintf(stderr, "  [fa-vram %-12s]   slot off=0x%lx max=%.1f MiB tensors=%d\n",
                    label, (unsigned long)off,
                    (double)info.first / 1048576.0, info.second);
        }
    };

    const int profile_level = []() {
        const char* e = std::getenv("QWEN3_FA_PROFILE_VRAM");
        return (e && *e) ? std::atoi(e) : 0;
    }();
    const bool prof = profile_level > 0;

    // ── Conv subgraph: cached at init. gallocr_alloc_graph is a no-op
    // when topology unchanged → tensor data pointers are stable across
    // calls so the ggml-cuda graph cache replays after the first chunk
    // and survives across encode invocations.
    if (!ggml_gallocr_alloc_graph(ctx->conv_galloc, ctx->conv_gf)) {
        fprintf(stderr, "crisp_audio: conv gallocr_alloc_graph failed\n");
        return nullptr;
    }
    probe_graph("enc-conv", ctx->conv_gf, ggml_gallocr_get_buffer_size(ctx->conv_galloc, 0), profile_level);

    // pe_input is identical across chunks — set once before the loop.
    {
        std::vector<float> pe_buf((size_t)d * T_chunk_out);
        std::memcpy(pe_buf.data(), ctx->sin_pe.data(), pe_buf.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(ctx->conv_gf, "pe_input"), pe_buf.data(), 0,
                                pe_buf.size() * sizeof(float));
    }

    // Host-side flat accumulator: each chunk writes its (T_chunk_out, d)
    // F32 conv_out into the corresponding slice. body_in is set from this
    // in one shot below. d * N_padded * 4 ≈ 2 MiB at nc=39.
    std::vector<float> flat((size_t)d * N_padded);

    ggml_tensor* mel_in   = ggml_graph_get_tensor(ctx->conv_gf, "mel_chunk");
    ggml_tensor* conv_out = ggml_graph_get_tensor(ctx->conv_gf, "conv_out");

    auto enc_t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < num_chunks; c++) {
        const float* chunk_src = mel_padded.data() + (size_t)c * chunk_T * n_mels;
        ggml_backend_tensor_set(mel_in, chunk_src, 0, (size_t)chunk_T * n_mels * sizeof(float));
        if (ggml_backend_graph_compute(ctx->backend, ctx->conv_gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "crisp_audio: conv graph compute failed (chunk %d/%d)\n", c, num_chunks);
            return nullptr;
        }
        ggml_backend_tensor_get(conv_out, flat.data() + (size_t)c * T_chunk_out * d, 0,
                                (size_t)T_chunk_out * d * sizeof(float));
    }
    double conv_ms = 0.0;
    if (prof) {
        ggml_backend_synchronize(ctx->backend);
        conv_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - enc_t0).count();
    }

    // ── Body subgraph: rebuild per encode (N_padded varies). Built into
    // the per-encode `compute_meta` arena; gallocr lives inside body_sched.
    ggml_cgraph* body_gf = build_body_subgraph(*ctx, N_padded);
    ggml_backend_sched_reset(ctx->body_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->body_sched, body_gf)) {
        fprintf(stderr, "crisp_audio: failed to alloc body graph\n");
        return nullptr;
    }
    probe_graph("enc-body", body_gf,
                ctx->backend ? ggml_backend_sched_get_buffer_size(ctx->body_sched, ctx->backend) : 0,
                profile_level);

    ggml_backend_tensor_set(ggml_graph_get_tensor(body_gf, "body_in"), flat.data(), 0,
                            flat.size() * sizeof(float));
    // attn_mask is unreferenced when build_body_subgraph took the
    // flash-attention path (mask is all-zero in this caller, so FA with
    // nullptr mask is equivalent and skips materialising the N²×n_heads
    // scores tensor). In that case the tensor is optimised out and
    // ggml_graph_get_tensor returns NULL.
    if (ggml_tensor* mask_in = ggml_graph_get_tensor(body_gf, "attn_mask")) {
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));
    }

    auto body_t0 = std::chrono::steady_clock::now();
    if (ggml_backend_sched_graph_compute(ctx->body_sched, body_gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "crisp_audio: body graph compute failed\n");
        return nullptr;
    }
    if (prof) {
        ggml_backend_synchronize(ctx->backend);
        const double body_ms  = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - body_t0).count();
        const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - enc_t0).count();
        fprintf(stderr,
                "  [fa-vram enc-time     ] conv=%.2f ms (%.2f ms/chunk × %d) body=%.2f ms total=%.2f ms\n",
                conv_ms, conv_ms / std::max(1, num_chunks), num_chunks, body_ms, total_ms);
    }

    ggml_tensor* out = ggml_graph_get_tensor(body_gf, "encoder_out");
    if (!out)
        return nullptr;
    const int pdim = (int)out->ne[0];
    const int N = (int)out->ne[1];
    if (out_n_frames)
        *out_n_frames = N;
    if (out_dim)
        *out_dim = pdim;

    const size_t total = (size_t)pdim * N;
    float* result = (float*)std::malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

int crisp_audio_d_model(struct crisp_audio_context* ctx) {
    return ctx ? (int)ctx->hp.d_model : 0;
}
int crisp_audio_output_dim(struct crisp_audio_context* ctx) {
    return ctx ? (int)ctx->hp.output_dim : 0;
}
int crisp_audio_n_layers(struct crisp_audio_context* ctx) {
    return ctx ? (int)ctx->hp.n_layers : 0;
}
int crisp_audio_n_window(struct crisp_audio_context* ctx) {
    return ctx ? (int)ctx->hp.n_window : 0;
}
enum crisp_audio_dialect crisp_audio_dialect_of(struct crisp_audio_context* ctx) {
    return ctx ? ctx->dialect : CRISP_AUDIO_DIALECT_AUTO;
}

void crisp_audio_get_vram_stats(struct crisp_audio_context* ctx, struct crisp_audio_vram_stats* out) {
    if (!out) return;
    out->model_buf_bytes      = 0;
    out->conv_galloc_bytes    = 0;
    out->body_sched_gpu_bytes = 0;
    if (!ctx) return;
    const bool gpu = ctx->backend && !ggml_backend_is_cpu(ctx->backend);
    if (ctx->model_buf && gpu) {
        out->model_buf_bytes = ggml_backend_buffer_get_size(ctx->model_buf);
    }
    if (ctx->conv_galloc) {
        out->conv_galloc_bytes = ggml_gallocr_get_buffer_size(ctx->conv_galloc, 0);
    }
    if (ctx->body_sched && ctx->backend && gpu) {
        out->body_sched_gpu_bytes = ggml_backend_sched_get_buffer_size(ctx->body_sched, ctx->backend);
    }
}

void* crisp_audio_get_gpu_backend_handle(struct crisp_audio_context* ctx) {
    if (!ctx || !ctx->backend) return nullptr;
    return ggml_backend_is_cpu(ctx->backend) ? nullptr : (void*) ctx->backend;
}

} // extern "C"
