#include "higgs_lm.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace {
    inline bool higgs_prof_on() { static const bool on = std::getenv("HIGGS_PROF") != nullptr; return on; }
    using prof_clk = std::chrono::steady_clock;
    inline double prof_ms(prof_clk::time_point t){ return std::chrono::duration<double,std::milli>(prof_clk::now()-t).count(); }
}

#define HIGGS_LM_MAX_NODES 8192
#define HIGGS_LM_FATTN_STRIDE 256

namespace higgs {

using qwen3_tts::GGUFLoader;
using qwen3_tts::load_tensor_data_from_file;
using qwen3_tts::init_preferred_backend;
using qwen3_tts::release_preferred_backend;

HiggsLM::~HiggsLM() { unload_model(); }

void HiggsLM::unload_model() {
    if (sched_)      { ggml_backend_sched_free(sched_); sched_ = nullptr; }
    if (kv_buffer_)  { ggml_backend_buffer_free(kv_buffer_); kv_buffer_ = nullptr; }
    if (kv_ctx_)     { ggml_free(kv_ctx_); kv_ctx_ = nullptr; }
    if (aux_buffer_) { ggml_backend_buffer_free(aux_buffer_); aux_buffer_ = nullptr; }
    if (aux_ctx_)    { ggml_free(aux_ctx_); aux_ctx_ = nullptr; }
    if (buffer_)     { ggml_backend_buffer_free(buffer_); buffer_ = nullptr; }
    if (ctx_)        { ggml_free(ctx_); ctx_ = nullptr; }
    if (backend_)     { release_preferred_backend(backend_); backend_ = nullptr; }
    if (backend_cpu_) { ggml_backend_free(backend_cpu_); backend_cpu_ = nullptr; }
    tensors_.clear(); layers_.clear(); k_cache_.clear(); v_cache_.clear();
    compute_meta_.clear();
    n_ctx_ = 0; n_past_ = 0;
}

#define HIGGS_LM_KV_INITIAL 1024  // covers a ~40s single utterance with no grow

static inline int higgs_round_up_stride(int n) {
    return ((n + HIGGS_LM_FATTN_STRIDE - 1) / HIGGS_LM_FATTN_STRIDE) * HIGGS_LM_FATTN_STRIDE;
}

void HiggsLM::reset() {
    n_past_ = 0;
    // Shrink an over-grown slab back to the small idle size between utterances so
    // a long read doesn't pin a big KV slab forever (dynamic KV, task #9).
    if (kv_buffer_ && kv_alloc_ > kv_initial_) realloc_kv_slab(kv_initial_, /*copy=*/false);
}

bool HiggsLM::load_model(const std::string & backbone_gguf, const std::string & aux_gguf, int n_ctx) {
    unload_model();

    GGUFLoader bb;
    if (!bb.open(backbone_gguf)) { error_msg_ = bb.get_error(); return false; }
    gguf_context * gc = bb.get_ctx();
    ggml_context * meta = bb.get_meta_ctx();

    cfg_.n_layers  = bb.get_u32("qwen3.block_count", 36);
    cfg_.n_embd    = bb.get_u32("qwen3.embedding_length", 2560);
    cfg_.n_head    = bb.get_u32("qwen3.attention.head_count", 32);
    cfg_.n_kv_head = bb.get_u32("qwen3.attention.head_count_kv", 8);
    cfg_.head_dim  = bb.get_u32("qwen3.attention.key_length", 128);
    cfg_.n_ff      = bb.get_u32("qwen3.feed_forward_length", 9728);
    cfg_.vocab     = bb.get_u32("qwen3.vocab_size", 151936);
    cfg_.rope_theta = bb.get_f32("qwen3.rope.freq_base", 1000000.0f);
    cfg_.rms_eps   = bb.get_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);

    // Backend (prefer CUDA).
    backend_ = init_preferred_backend("HiggsLM", &error_msg_, false, nullptr);
    if (!backend_) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
    const bool on_cpu = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (!on_cpu) {
        backend_cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu_) { error_msg_ = "cpu fallback init failed"; return false; }
    }
    enum ggml_backend_dev_type want = on_cpu ? GGML_BACKEND_DEVICE_TYPE_CPU : ggml_backend_dev_type(dev);
    fprintf(stderr, "  HiggsLM backend: %s  (%d L, d%d, %dh/%dkv, hd%d, ff%d, θ%.0f)\n",
            dev ? ggml_backend_dev_name(dev) : "?", cfg_.n_layers, cfg_.n_embd,
            cfg_.n_head, cfg_.n_kv_head, cfg_.head_dim, cfg_.n_ff, cfg_.rope_theta);

    // ---- backbone weights ----
    const int64_t n_t = bb.get_n_tensors();
    struct ggml_init_params p = { ggml_tensor_overhead() * (n_t + 8), nullptr, true };
    ctx_ = ggml_init(p);
    for (int64_t i = 0; i < n_t; ++i) {
        const char * name = bb.get_tensor_name(i);
        struct ggml_tensor * mt = ggml_get_tensor(meta, name);
        if (!mt) continue;
        struct ggml_tensor * t = ggml_dup_tensor(ctx_, mt);
        ggml_set_name(t, name);
        tensors_[name] = t;
    }
    if (!load_tensor_data_from_file(backbone_gguf, gc, ctx_, tensors_, buffer_, error_msg_, want))
        return false;

    auto get = [&](const std::string & n)->struct ggml_tensor *{
        auto it = tensors_.find(n); return it==tensors_.end()?nullptr:it->second; };
    tok_embd_ = get("token_embd.weight");
    out_norm_ = get("output_norm.weight");
    if (!tok_embd_ || !out_norm_) { error_msg_ = "missing token_embd/output_norm"; return false; }
    layers_.resize(cfg_.n_layers);
    for (int i = 0; i < cfg_.n_layers; ++i) {
        char nm[64]; auto & L = layers_[i];
        snprintf(nm,sizeof(nm),"blk.%d.attn_norm.weight",i);   L.attn_norm = get(nm);
        snprintf(nm,sizeof(nm),"blk.%d.attn_q.weight",i);      L.wq = get(nm);
        snprintf(nm,sizeof(nm),"blk.%d.attn_k.weight",i);      L.wk = get(nm);
        snprintf(nm,sizeof(nm),"blk.%d.attn_v.weight",i);      L.wv = get(nm);
        snprintf(nm,sizeof(nm),"blk.%d.attn_output.weight",i); L.wo = get(nm);
        snprintf(nm,sizeof(nm),"blk.%d.attn_q_norm.weight",i); L.q_norm = get(nm);
        snprintf(nm,sizeof(nm),"blk.%d.attn_k_norm.weight",i); L.k_norm = get(nm);
        snprintf(nm,sizeof(nm),"blk.%d.ffn_norm.weight",i);    L.ffn_norm = get(nm);
        snprintf(nm,sizeof(nm),"blk.%d.ffn_gate.weight",i);    L.ffn_gate = get(nm);
        snprintf(nm,sizeof(nm),"blk.%d.ffn_up.weight",i);      L.ffn_up = get(nm);
        snprintf(nm,sizeof(nm),"blk.%d.ffn_down.weight",i);    L.ffn_down = get(nm);
        if (!L.wq||!L.wk||!L.wv||!L.wo||!L.q_norm||!L.k_norm||!L.ffn_gate||!L.ffn_up||!L.ffn_down) {
            error_msg_ = "missing layer tensor at blk " + std::to_string(i); return false;
        }
    }

    // ---- audio_embd from aux sidecar ----
    GGUFLoader aux;
    if (!aux.open(aux_gguf)) { error_msg_ = "aux: " + aux.get_error(); return false; }
    {
        ggml_context * am = aux.get_meta_ctx();
        struct ggml_tensor * mt = ggml_get_tensor(am, "audio_embd.weight");
        if (!mt) { error_msg_ = "aux missing audio_embd.weight"; return false; }
        struct ggml_init_params ap = { ggml_tensor_overhead() * 4, nullptr, true };
        aux_ctx_ = ggml_init(ap);
        audio_embd_ = ggml_dup_tensor(aux_ctx_, mt);
        ggml_set_name(audio_embd_, "audio_embd.weight");
        std::map<std::string, struct ggml_tensor *> am_map = {{"audio_embd.weight", audio_embd_}};
        if (!load_tensor_data_from_file(aux_gguf, aux.get_ctx(), aux_ctx_, am_map, aux_buffer_, error_msg_, want))
            return false;
        cfg_.audio_vocab = (int)audio_embd_->ne[1];   // 8208
        cfg_.cb_vocab    = aux.get_u32("higgs-codec.audio_codebook_stride", 1026);
        cfg_.n_codebooks = aux.get_u32("higgs-codec.n_codebooks", 8);
        if (cfg_.audio_vocab != cfg_.n_codebooks * cfg_.cb_vocab) {
            fprintf(stderr, "  WARN audio_vocab %d != %d*%d\n", cfg_.audio_vocab, cfg_.n_codebooks, cfg_.cb_vocab);
        }
    }

    // ---- KV cache ----
    const char * env_kv = std::getenv("HIGGS_LM_KV");
    ggml_type kvt = GGML_TYPE_F16;
    if (env_kv && std::strcmp(env_kv, "q8") == 0) kvt = GGML_TYPE_Q8_0;
    else if (env_kv && std::strcmp(env_kv, "f32") == 0) kvt = GGML_TYPE_F32;
    if (!alloc_kv(n_ctx, kvt)) return false;

    // ---- scheduler ----
    std::vector<ggml_backend_t> backends = { backend_ };
    if (backend_cpu_) backends.push_back(backend_cpu_);
    sched_ = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(),
                                    HIGGS_LM_MAX_NODES, false, true);
    if (!sched_) { error_msg_ = "sched_new failed"; return false; }
    compute_meta_.resize(ggml_tensor_overhead() * HIGGS_LM_MAX_NODES +
                         ggml_graph_overhead_custom(HIGGS_LM_MAX_NODES, false));
    return true;
}

// Initial allocation: cap is n_ctx; start with a small slab (dynamic KV, #9).
bool HiggsLM::alloc_kv(int n_ctx, ggml_type kv_type) {
    kv_type_ = kv_type; n_ctx_ = n_ctx;
    kv_initial_ = std::min(higgs_round_up_stride(HIGGS_LM_KV_INITIAL), higgs_round_up_stride(n_ctx));
    // Debug/budget knob: force the full-cap KV slab at load to measure the
    // worst-case (KV grown to n_ctx) VRAM peak without a 320s continuous read.
    { const char * pe = std::getenv("HIGGS_LM_KV_PREALLOC");
      if (pe && pe[0] && pe[0] != '0') kv_initial_ = higgs_round_up_stride(n_ctx); }
    kv_alloc_ = 0;
    if (!realloc_kv_slab(kv_initial_, /*copy=*/false)) return false;
    fprintf(stderr, "  HiggsLM KV (dynamic): %d L × %s, initial %d / cap %d rows = %.1f MiB now\n",
            cfg_.n_layers, ggml_type_name(kv_type), kv_alloc_, n_ctx_,
            ggml_backend_buffer_get_size(kv_buffer_) / (1024.0*1024.0));
    return true;
}

// (Re)allocate the KV slab to n_alloc rows. Mirrors the vocoder
// ensure_stream_kv_cache grow-with-copy pattern (host-bounce save/restore;
// block-quant-aware row_bytes). When copy=false the contents are discarded.
bool HiggsLM::realloc_kv_slab(int n_alloc, bool copy) {
    const int nl = cfg_.n_layers, nkv = cfg_.n_kv_head, hd = cfg_.head_dim;
    if (n_alloc < HIGGS_LM_FATTN_STRIDE) n_alloc = HIGGS_LM_FATTN_STRIDE;
    if (n_alloc == kv_alloc_ && kv_buffer_) return true;

    // Save populated [0..n_past_) rows if growing in-place.
    std::vector<std::vector<uint8_t>> saved_k, saved_v;
    int saved_n_past = 0;
    const size_t row_bytes = (size_t)ggml_type_size(kv_type_) * hd * nkv / (size_t)ggml_blck_size(kv_type_);
    if (copy && kv_buffer_ && n_past_ > 0) {
        saved_n_past = std::min(n_past_, std::min(kv_alloc_, n_alloc));
        const size_t bytes = row_bytes * (size_t)saved_n_past;
        saved_k.assign(nl, std::vector<uint8_t>(bytes));
        saved_v.assign(nl, std::vector<uint8_t>(bytes));
        for (int i = 0; i < nl; ++i) {
            ggml_backend_tensor_get(k_cache_[i], saved_k[i].data(), 0, bytes);
            ggml_backend_tensor_get(v_cache_[i], saved_v[i].data(), 0, bytes);
        }
    }

    if (kv_buffer_) { ggml_backend_buffer_free(kv_buffer_); kv_buffer_ = nullptr; }
    if (kv_ctx_)    { ggml_free(kv_ctx_); kv_ctx_ = nullptr; }

    struct ggml_init_params p = { ggml_tensor_overhead() * (nl * 2 + 4), nullptr, true };
    kv_ctx_ = ggml_init(p);
    k_cache_.resize(nl); v_cache_.resize(nl);
    for (int i = 0; i < nl; ++i) {
        k_cache_[i] = ggml_new_tensor_3d(kv_ctx_, kv_type_, hd, nkv, n_alloc);
        ggml_format_name(k_cache_[i], "k_cache_%d", i);
        v_cache_[i] = ggml_new_tensor_3d(kv_ctx_, kv_type_, hd, nkv, n_alloc);
        ggml_format_name(v_cache_[i], "v_cache_%d", i);
    }
    kv_buffer_ = ggml_backend_alloc_ctx_tensors(kv_ctx_, backend_);
    if (!kv_buffer_) { error_msg_ = "kv alloc failed"; return false; }
    for (int i = 0; i < nl; ++i) {
        ggml_backend_tensor_memset(k_cache_[i], 0, 0, ggml_nbytes(k_cache_[i]));
        ggml_backend_tensor_memset(v_cache_[i], 0, 0, ggml_nbytes(v_cache_[i]));
    }
    // Restore previously-populated rows (growing path).
    if (saved_n_past > 0) {
        const size_t bytes = row_bytes * (size_t)saved_n_past;
        for (int i = 0; i < nl; ++i) {
            ggml_backend_tensor_set(k_cache_[i], saved_k[i].data(), 0, bytes);
            ggml_backend_tensor_set(v_cache_[i], saved_v[i].data(), 0, bytes);
        }
    }
    kv_alloc_ = n_alloc;
    return true;
}

// Grow the slab geometrically so it can hold need_pos positions. cap = n_ctx_.
bool HiggsLM::ensure_kv(int need_pos) {
    const int target = higgs_round_up_stride(need_pos);
    if (target <= kv_alloc_) return true;
    if (target > higgs_round_up_stride(n_ctx_)) { error_msg_ = "context overflow"; return false; }
    int n_alloc = std::max(target, kv_alloc_ * 2);
    n_alloc = std::min(n_alloc, higgs_round_up_stride(n_ctx_));
    fprintf(stderr, "  HiggsLM KV grow: %d -> %d rows (need %d, cap %d)\n", kv_alloc_, n_alloc, need_pos, n_ctx_);
    return realloc_kv_slab(n_alloc, /*copy=*/true);
}

struct ggml_tensor * HiggsLM::rms_norm(struct ggml_context * c, struct ggml_tensor * x,
                                       struct ggml_tensor * w) {
    return ggml_mul(c, ggml_rms_norm(c, x, cfg_.rms_eps), w);
}

struct ggml_tensor * HiggsLM::apply_layer(struct ggml_context * c, struct ggml_cgraph * gf,
                                          struct ggml_tensor * x, const lm_layer & L, int il,
                                          int n, struct ggml_tensor * pos,
                                          struct ggml_tensor * mask) {
    const int hd = cfg_.head_dim, nh = cfg_.n_head, nkv = cfg_.n_kv_head;
    const float scale = 1.0f / sqrtf((float)hd);

    struct ggml_tensor * res = x;
    struct ggml_tensor * h = rms_norm(c, x, L.attn_norm);

    struct ggml_tensor * q = ggml_mul_mat(c, L.wq, h);   // [nh*hd, n]
    struct ggml_tensor * k = ggml_mul_mat(c, L.wk, h);   // [nkv*hd, n]
    struct ggml_tensor * v = ggml_mul_mat(c, L.wv, h);   // [nkv*hd, n]
    q = ggml_reshape_3d(c, q, hd, nh, n);
    k = ggml_reshape_3d(c, k, hd, nkv, n);
    v = ggml_reshape_3d(c, v, hd, nkv, n);

    // Qwen3 per-head q/k RMSNorm over head_dim, then RoPE.
    q = ggml_mul(c, ggml_rms_norm(c, q, cfg_.rms_eps), L.q_norm);
    k = ggml_mul(c, ggml_rms_norm(c, k, cfg_.rms_eps), L.k_norm);
    q = ggml_rope_ext(c, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      cfg_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(c, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      cfg_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // write K/V into the slab at [n_past .. n_past+n)
    struct ggml_tensor * kslab = k_cache_[il];
    struct ggml_tensor * vslab = v_cache_[il];
    struct ggml_tensor * kslab2d = ggml_view_2d(c, kslab, hd*nkv, kv_alloc_, kslab->nb[2], 0);
    struct ggml_tensor * vslab2d = ggml_view_2d(c, vslab, hd*nkv, kv_alloc_, vslab->nb[2], 0);
    struct ggml_tensor * kcur2d = ggml_view_2d(c, k, hd*nkv, n, k->nb[2], 0);
    struct ggml_tensor * vcur2d = ggml_view_2d(c, v, hd*nkv, n, v->nb[2], 0);
    ggml_build_forward_expand(gf, ggml_set_rows(c, kslab2d, kcur2d, pos));
    ggml_build_forward_expand(gf, ggml_set_rows(c, vslab2d, vcur2d, pos));

    const int n_kv_eff = ((n_past_ + n + HIGGS_LM_FATTN_STRIDE - 1) / HIGGS_LM_FATTN_STRIDE)
                         * HIGGS_LM_FATTN_STRIDE;
    struct ggml_tensor * kview = ggml_view_3d(c, kslab, hd, nkv, n_kv_eff, kslab->nb[1], kslab->nb[2], 0);
    struct ggml_tensor * vview = ggml_view_3d(c, vslab, hd, nkv, n_kv_eff, vslab->nb[1], vslab->nb[2], 0);
    struct ggml_tensor * Q = ggml_permute(c, q, 0, 2, 1, 3);       // [hd, n, nh]
    struct ggml_tensor * K = ggml_permute(c, kview, 0, 2, 1, 3);   // [hd, n_kv_eff, nkv]
    struct ggml_tensor * V = ggml_permute(c, vview, 0, 2, 1, 3);

    struct ggml_tensor * attn = ggml_flash_attn_ext(c, Q, K, V, mask, scale, 0.0f, 0.0f);
    attn = ggml_cont_2d(c, attn, nh*hd, n);
    attn = ggml_mul_mat(c, L.wo, attn);

    x = ggml_add(c, res, attn);
    res = x;
    h = rms_norm(c, x, L.ffn_norm);
    struct ggml_tensor * g = ggml_silu(c, ggml_mul_mat(c, L.ffn_gate, h));
    struct ggml_tensor * u = ggml_mul_mat(c, L.ffn_up, h);
    h = ggml_mul_mat(c, L.ffn_down, ggml_mul(c, g, u));
    return ggml_add(c, res, h);
}

struct ggml_cgraph * HiggsLM::build_graph(struct ggml_context * c, int n, bool is_audio) {
    struct ggml_cgraph * gf = ggml_new_graph_custom(c, HIGGS_LM_MAX_NODES, false);

    struct ggml_tensor * emb;  // [n_embd, n]
    if (is_audio) {
        // Σ_c get_rows(audio_embd, codes_c + c*cb_vocab). codes pre-offset on host.
        emb = nullptr;
        for (int cb = 0; cb < cfg_.n_codebooks; ++cb) {
            char nm[32]; snprintf(nm, sizeof(nm), "acodes_%d", cb);
            struct ggml_tensor * idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
            ggml_set_name(idx, nm); ggml_set_input(idx);
            struct ggml_tensor * r = ggml_get_rows(c, audio_embd_, idx);  // [n_embd, n]
            emb = emb ? ggml_add(c, emb, r) : r;
        }
    } else {
        struct ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
        ggml_set_name(ids, "ids"); ggml_set_input(ids);
        emb = ggml_get_rows(c, tok_embd_, ids);  // [n_embd, n]
    }
    // get_rows on a quantized matrix yields F32; ensure F32 contiguous.
    struct ggml_tensor * x = emb;

    struct ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    ggml_set_name(pos, "pos"); ggml_set_input(pos);

    const int n_kv_eff = ((n_past_ + n + HIGGS_LM_FATTN_STRIDE - 1) / HIGGS_LM_FATTN_STRIDE)
                         * HIGGS_LM_FATTN_STRIDE;
    struct ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv_eff, n);
    ggml_set_name(mask, "mask"); ggml_set_input(mask);

    for (int il = 0; il < cfg_.n_layers; ++il)
        x = apply_layer(c, gf, x, layers_[il], il, n, pos, mask);

    x = rms_norm(c, x, out_norm_);   // [n_embd, n]

    // audio logits for the LAST token: h_last @ audio_embdᵀ -> [audio_vocab]
    struct ggml_tensor * last = ggml_view_2d(c, x, cfg_.n_embd, 1, x->nb[1],
                                             (size_t)(n - 1) * x->nb[1]);
    last = ggml_cont(c, last);
    struct ggml_tensor * logits = ggml_mul_mat(c, audio_embd_, last);  // [audio_vocab, 1]
    ggml_set_name(logits, "audio_logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

bool HiggsLM::run(int n, bool is_audio, const int32_t * ids_or_codes, std::vector<float> & out) {
    const bool prof = higgs_prof_on();
    prof_clk::time_point t0;
    if (!ensure_kv(n_past_ + n)) return false;   // grow the slab on demand (dynamic KV)

    if (prof) t0 = prof_clk::now();
    struct ggml_init_params p = { compute_meta_.size(), compute_meta_.data(), true };
    struct ggml_context * c = ggml_init(p);
    struct ggml_cgraph * gf = build_graph(c, n, is_audio);
    if (prof) { prof_build_ms += prof_ms(t0); t0 = prof_clk::now(); }

    if (!ggml_backend_sched_alloc_graph(sched_, gf)) { error_msg_="alloc graph"; ggml_free(c); return false; }
    if (prof) { prof_alloc_ms += prof_ms(t0); t0 = prof_clk::now(); }

    // positions
    std::vector<int32_t> posv(n);
    for (int i = 0; i < n; ++i) posv[i] = n_past_ + i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf,"pos"), posv.data(), 0, n*sizeof(int32_t));

    // inputs
    if (is_audio) {
        std::vector<int32_t> col(n);
        for (int cb = 0; cb < cfg_.n_codebooks; ++cb) {
            char nm[32]; snprintf(nm, sizeof(nm), "acodes_%d", cb);
            for (int i = 0; i < n; ++i) col[i] = ids_or_codes[i*cfg_.n_codebooks + cb] + cb*cfg_.cb_vocab;
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf,nm), col.data(), 0, n*sizeof(int32_t));
        }
    } else {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf,"ids"), ids_or_codes, 0, n*sizeof(int32_t));
    }

    // causal mask (F16): mask[j,i] = 0 if j <= n_past+i else -inf
    {
        const int n_kv_eff = ((n_past_ + n + HIGGS_LM_FATTN_STRIDE - 1)/HIGGS_LM_FATTN_STRIDE)*HIGGS_LM_FATTN_STRIDE;
        std::vector<ggml_fp16_t> m((size_t)n_kv_eff * n);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n_kv_eff; ++j)
                m[(size_t)i*n_kv_eff + j] = (j <= n_past_ + i) ? zero : ninf;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf,"mask"), m.data(), 0, m.size()*sizeof(ggml_fp16_t));
    }
    if (prof) { prof_upload_ms += prof_ms(t0); t0 = prof_clk::now(); }

    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        error_msg_="compute"; ggml_backend_sched_reset(sched_); ggml_free(c); return false;
    }
    if (prof) { prof_compute_ms += prof_ms(t0); t0 = prof_clk::now(); }

    struct ggml_tensor * lt = ggml_graph_get_tensor(gf, "audio_logits");
    out.resize(ggml_nelements(lt));
    ggml_backend_tensor_get(lt, out.data(), 0, ggml_nbytes(lt));
    ggml_backend_sched_reset(sched_);
    ggml_free(c);
    n_past_ += n;
    if (prof) { prof_get_ms += prof_ms(t0); prof_calls++; }
    return true;
}

void HiggsLM::dump_prof_reset(const char * label) {
    if (!prof_calls) return;
    const double n = (double)prof_calls;
    const double tot = prof_build_ms + prof_alloc_ms + prof_upload_ms + prof_compute_ms + prof_get_ms;
    fprintf(stderr,
        "  [higgs-prof %-10s] %ld calls | per-call ms: build=%.3f alloc=%.3f upload=%.3f compute=%.3f get=%.3f | sum=%.3f\n"
        "  [higgs-prof %-10s] share: build=%.1f%% alloc=%.1f%% upload=%.1f%% compute(GPU)=%.1f%% get=%.1f%%\n",
        label, prof_calls, prof_build_ms/n, prof_alloc_ms/n, prof_upload_ms/n, prof_compute_ms/n, prof_get_ms/n, tot/n,
        label, 100*prof_build_ms/tot, 100*prof_alloc_ms/tot, 100*prof_upload_ms/tot, 100*prof_compute_ms/tot, 100*prof_get_ms/tot);
    prof_build_ms = prof_alloc_ms = prof_upload_ms = prof_compute_ms = prof_get_ms = 0; prof_calls = 0;
}

bool HiggsLM::prefill(const int32_t * ids, int n, std::vector<float> & out) {
    return run(n, /*is_audio=*/false, ids, out);
}
bool HiggsLM::decode_step(const int32_t * frame_codes, std::vector<float> & out) {
    return run(1, /*is_audio=*/true, frame_codes, out);
}
bool HiggsLM::prefill_audio(const int32_t * delayed_codes, int n_frames, std::vector<float> & out) {
    if (n_frames <= 0) { out.clear(); return true; }
    return run(n_frames, /*is_audio=*/true, delayed_codes, out);
}

void HiggsLM::log_vram(const char * label) const {
    auto mib = [](ggml_backend_buffer_t b){ return b ? ggml_backend_buffer_get_size(b)/(1024.0*1024.0) : 0.0; };
    double w = mib(buffer_), a = mib(aux_buffer_), kv = mib(kv_buffer_);
    double sc = (sched_&&backend_) ? ggml_backend_sched_get_buffer_size(sched_, backend_)/(1024.0*1024.0) : 0.0;
    fprintf(stderr, "  [vram-lm %-12s] weights=%.1f audio_embd=%.1f kv=%.1f sched=%.1f total=%.1f MiB\n",
            label, w, a, kv, sc, w+a+kv+sc);
}

} // namespace higgs
