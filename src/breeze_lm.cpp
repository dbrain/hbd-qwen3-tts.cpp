#include "breeze_lm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define BZ_LM_MAX_NODES   8192
#define BZ_LM_FA_STRIDE   256
#define BZ_LM_KV_INITIAL  512

namespace breeze {

static inline int round_up_stride(int n) {
    return ((n + BZ_LM_FA_STRIDE - 1) / BZ_LM_FA_STRIDE) * BZ_LM_FA_STRIDE;
}

size_t Backbone::sched_bytes() const {
    if (!w_) return 0;
    size_t n = sched_ ? ggml_backend_sched_get_buffer_size(sched_, w_->backend()) : 0;
    if (dec_.sched) n += ggml_backend_sched_get_buffer_size(dec_.sched, w_->backend());
    return n;
}

void Backbone::drop_decode_slot() {
    if (dec_.sched) { ggml_backend_sched_free(dec_.sched); dec_.sched = nullptr; }
    if (dec_.ctx)   { ggml_free(dec_.ctx); dec_.ctx = nullptr; }
    dec_.gf = nullptr; dec_.n_kv_eff = -1; dec_.kv_alloc = -1; dec_.meta.clear();
}

void Backbone::free_state() {
    drop_decode_slot();
    if (sched_)     { ggml_backend_sched_free(sched_); sched_ = nullptr; }
    if (kv_buffer_) { ggml_backend_buffer_free(kv_buffer_); kv_buffer_ = nullptr; }
    if (kv_ctx_)    { ggml_free(kv_ctx_); kv_ctx_ = nullptr; }
    k_cache_.clear(); v_cache_.clear(); layers_.clear(); compute_meta_.clear();
    n_ctx_ = kv_alloc_ = n_past_ = 0;
}

bool Backbone::init(BreezeWeights * w, int n_ctx) {
    free_state();
    w_ = w;
    cfg_ = w->cfg().bb;

    out_norm_   = w->get("backbone.output_norm.weight");
    lm_head_    = w->get("backbone.lm_head.weight");
    // tie_codebooks_embeddings: the backbone's embed_audio_tokens IS the depth
    // decoder's embed_tokens, so the checkpoint ships exactly one copy.
    audio_embd_ = w->get("depth.token_embd.weight");
    if (!out_norm_ || !lm_head_ || !audio_embd_) {
        error_msg_ = "backbone: missing output_norm/lm_head/token_embd";
        return false;
    }

    layers_.resize(cfg_.n_layers);
    for (int i = 0; i < cfg_.n_layers; ++i) {
        auto & L = layers_[i];
        L.attn_norm = w->getf("backbone.blk.%d.attn_norm.weight", i);
        L.wq        = w->getf("backbone.blk.%d.attn_q.weight", i);
        L.wk        = w->getf("backbone.blk.%d.attn_k.weight", i);
        L.wv        = w->getf("backbone.blk.%d.attn_v.weight", i);
        L.wo        = w->getf("backbone.blk.%d.attn_output.weight", i);
        L.q_norm    = w->getf("backbone.blk.%d.attn_q_norm.weight", i);
        L.k_norm    = w->getf("backbone.blk.%d.attn_k_norm.weight", i);
        L.ffn_norm  = w->getf("backbone.blk.%d.ffn_norm.weight", i);
        L.ffn_gate  = w->getf("backbone.blk.%d.ffn_gate.weight", i);
        L.ffn_up    = w->getf("backbone.blk.%d.ffn_up.weight", i);
        L.ffn_down  = w->getf("backbone.blk.%d.ffn_down.weight", i);
        if (!L.attn_norm || !L.wq || !L.wk || !L.wv || !L.wo || !L.q_norm ||
            !L.k_norm || !L.ffn_norm || !L.ffn_gate || !L.ffn_up || !L.ffn_down) {
            error_msg_ = "backbone: missing tensor in blk " + std::to_string(i);
            return false;
        }
        char nm[64];
        snprintf(nm, sizeof(nm), "backbone.blk.%d.ffn_gate_up.weight", i);
        L.ffn_gate_up = w->fuse_rows(L.ffn_gate, L.ffn_up, nm);
        snprintf(nm, sizeof(nm), "backbone.blk.%d.attn_qk.weight", i);
        if (struct ggml_tensor * qk = w->fuse_rows(L.wq, L.wk, nm)) {
            snprintf(nm, sizeof(nm), "backbone.blk.%d.attn_qkv.weight", i);
            L.wqkv = w->fuse_rows(qk, L.wv, nm);
        }
    }

    n_ctx_ = std::min(n_ctx, cfg_.max_pos);
    kv_initial_ = std::min(round_up_stride(BZ_LM_KV_INITIAL), round_up_stride(n_ctx_));
    { const char * e = std::getenv("BREEZE_KV");
      if (e && !std::strcmp(e, "q8"))  kv_type_ = GGML_TYPE_Q8_0;
      if (e && !std::strcmp(e, "f32")) kv_type_ = GGML_TYPE_F32; }
    if (!alloc_kv(kv_initial_, false)) return false;

    std::vector<ggml_backend_t> backends = { w->backend() };
    if (w->backend_cpu()) backends.push_back(w->backend_cpu());
    sched_ = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(),
                                    BZ_LM_MAX_NODES, false, true);
    if (!sched_) { error_msg_ = "backbone: sched_new failed"; return false; }
    compute_meta_.resize(ggml_tensor_overhead() * BZ_LM_MAX_NODES +
                         ggml_graph_overhead_custom(BZ_LM_MAX_NODES, false));
    return true;
}

void Backbone::reset() {
    n_past_ = 0;
    if (kv_buffer_ && kv_alloc_ > kv_initial_) alloc_kv(kv_initial_, false);
}

bool Backbone::alloc_kv(int n_alloc, bool copy) {
    const int nl = cfg_.n_layers, nkv = cfg_.n_kv_head, hd = cfg_.head_dim;
    if (n_alloc < BZ_LM_FA_STRIDE) n_alloc = BZ_LM_FA_STRIDE;
    if (n_alloc == kv_alloc_ && kv_buffer_) return true;

    std::vector<std::vector<uint8_t>> sk, sv;
    int saved = 0;
    const size_t row = (size_t) ggml_type_size(kv_type_) * hd * nkv / (size_t) ggml_blck_size(kv_type_);
    if (copy && kv_buffer_ && n_past_ > 0) {
        saved = std::min(n_past_, std::min(kv_alloc_, n_alloc));
        sk.assign(nl, std::vector<uint8_t>(row * saved));
        sv.assign(nl, std::vector<uint8_t>(row * saved));
        for (int i = 0; i < nl; ++i) {
            ggml_backend_tensor_get(k_cache_[i], sk[i].data(), 0, row * saved);
            ggml_backend_tensor_get(v_cache_[i], sv[i].data(), 0, row * saved);
        }
    }
    if (kv_buffer_) { ggml_backend_buffer_free(kv_buffer_); kv_buffer_ = nullptr; }
    if (kv_ctx_)    { ggml_free(kv_ctx_); kv_ctx_ = nullptr; }

    struct ggml_init_params p = { ggml_tensor_overhead() * (nl * 2 + 4), nullptr, true };
    kv_ctx_ = ggml_init(p);
    k_cache_.resize(nl); v_cache_.resize(nl);
    for (int i = 0; i < nl; ++i) {
        k_cache_[i] = ggml_new_tensor_3d(kv_ctx_, kv_type_, hd, nkv, n_alloc);
        ggml_format_name(k_cache_[i], "bb_k_%d", i);
        v_cache_[i] = ggml_new_tensor_3d(kv_ctx_, kv_type_, hd, nkv, n_alloc);
        ggml_format_name(v_cache_[i], "bb_v_%d", i);
    }
    kv_buffer_ = ggml_backend_alloc_ctx_tensors(kv_ctx_, w_->backend());
    if (!kv_buffer_) { error_msg_ = "backbone: kv alloc failed"; return false; }
    for (int i = 0; i < nl; ++i) {
        ggml_backend_tensor_memset(k_cache_[i], 0, 0, ggml_nbytes(k_cache_[i]));
        ggml_backend_tensor_memset(v_cache_[i], 0, 0, ggml_nbytes(v_cache_[i]));
    }
    if (saved > 0) {
        for (int i = 0; i < nl; ++i) {
            ggml_backend_tensor_set(k_cache_[i], sk[i].data(), 0, row * saved);
            ggml_backend_tensor_set(v_cache_[i], sv[i].data(), 0, row * saved);
        }
    }
    kv_alloc_ = n_alloc;
    drop_decode_slot();   // the cached decode graph views the old slab
    return true;
}

bool Backbone::ensure_kv(int need) {
    const int target = round_up_stride(need);
    if (target <= kv_alloc_) return true;
    if (target > round_up_stride(n_ctx_)) { error_msg_ = "backbone: context overflow"; return false; }
    int n_alloc = std::min(std::max(target, kv_alloc_ * 2), round_up_stride(n_ctx_));
    return alloc_kv(n_alloc, true);
}

struct ggml_cgraph * Backbone::build_graph(struct ggml_context * c, int n, bool from_codes) {
    struct ggml_cgraph * gf = ggml_new_graph_custom(c, BZ_LM_MAX_NODES, false);

    const int hd = cfg_.head_dim, nh = cfg_.n_head, nkv = cfg_.n_kv_head;
    const float scale = 1.0f / std::sqrt((float) hd);

    struct ggml_tensor * x;
    if (from_codes) {
        // BreezeBackboneModelEmbeddings: sum_c embed[code_c + c*audio_vocab].
        // The per-codebook offsets are applied host-side.
        x = nullptr;
        for (int cb = 0; cb < cfg_.n_codebooks; ++cb) {
            char nm[32]; snprintf(nm, sizeof(nm), "codes_%d", cb);
            struct ggml_tensor * idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
            ggml_set_name(idx, nm); ggml_set_input(idx);
            struct ggml_tensor * r = ggml_get_rows(c, audio_embd_, idx);
            x = x ? ggml_add(c, x, r) : r;
        }
    } else {
        x = ggml_new_tensor_2d(c, GGML_TYPE_F32, cfg_.n_embd, n);
        ggml_set_name(x, "embeds"); ggml_set_input(x);
    }

    struct ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    ggml_set_name(pos, "pos"); ggml_set_input(pos);

    const int n_kv_eff = round_up_stride(n_past_ + n);
    struct ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv_eff, n);
    ggml_set_name(mask, "mask"); ggml_set_input(mask);

    for (int il = 0; il < cfg_.n_layers; ++il) {
        const layer & L = layers_[il];
        struct ggml_tensor * res = x;
        struct ggml_tensor * h = ggml_mul(c, ggml_rms_norm(c, x, cfg_.rms_eps), L.attn_norm);

        struct ggml_tensor * q, * k, * v;
        if (L.wqkv && n == 1) {
            // One MMVQ + one q8_1 quantisation of `h` instead of three. n == 1
            // leaves each slice contiguous, so the reshapes need no ggml_cont.
            struct ggml_tensor * qkv = ggml_mul_mat(c, L.wqkv, h);
            const size_t es = ggml_type_size(qkv->type);
            const int64_t nq = hd * nh, nk = hd * nkv;
            q = ggml_reshape_3d(c, ggml_view_2d(c, qkv, nq, n, nq * es, 0), hd, nh, n);
            k = ggml_reshape_3d(c, ggml_view_2d(c, qkv, nk, n, nk * es, (size_t) nq * es), hd, nkv, n);
            v = ggml_reshape_3d(c, ggml_view_2d(c, qkv, nk, n, nk * es, (size_t) (nq + nk) * es), hd, nkv, n);
        } else {
            q = ggml_reshape_3d(c, ggml_mul_mat(c, L.wq, h), hd, nh,  n);
            k = ggml_reshape_3d(c, ggml_mul_mat(c, L.wk, h), hd, nkv, n);
            v = ggml_reshape_3d(c, ggml_mul_mat(c, L.wv, h), hd, nkv, n);
        }
        q = ggml_mul(c, ggml_rms_norm(c, q, cfg_.rms_eps), L.q_norm);
        k = ggml_mul(c, ggml_rms_norm(c, k, cfg_.rms_eps), L.k_norm);
        q = ggml_rope_ext(c, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                          cfg_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(c, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                          cfg_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        struct ggml_tensor * ks = k_cache_[il], * vs = v_cache_[il];
        ggml_build_forward_expand(gf, ggml_set_rows(c,
            ggml_view_2d(c, ks, hd * nkv, kv_alloc_, ks->nb[2], 0),
            ggml_view_2d(c, k,  hd * nkv, n, k->nb[2], 0), pos));
        ggml_build_forward_expand(gf, ggml_set_rows(c,
            ggml_view_2d(c, vs, hd * nkv, kv_alloc_, vs->nb[2], 0),
            ggml_view_2d(c, v,  hd * nkv, n, v->nb[2], 0), pos));

        struct ggml_tensor * kview = ggml_view_3d(c, ks, hd, nkv, n_kv_eff, ks->nb[1], ks->nb[2], 0);
        struct ggml_tensor * vview = ggml_view_3d(c, vs, hd, nkv, n_kv_eff, vs->nb[1], vs->nb[2], 0);
        struct ggml_tensor * Q = ggml_permute(c, q, 0, 2, 1, 3);
        struct ggml_tensor * K = ggml_permute(c, kview, 0, 2, 1, 3);
        struct ggml_tensor * V = ggml_permute(c, vview, 0, 2, 1, 3);
        struct ggml_tensor * attn = ggml_flash_attn_ext(c, Q, K, V, mask, scale, 0.0f, 0.0f);
        attn = ggml_cont_2d(c, attn, nh * hd, n);
        attn = ggml_mul_mat(c, L.wo, attn);

        x = ggml_add(c, res, attn);
        res = x;
        h = ggml_mul(c, ggml_rms_norm(c, x, cfg_.rms_eps), L.ffn_norm);
        struct ggml_tensor * g, * u;
        if (L.ffn_gate_up && n == 1) {
            // One MMVQ over [n_embd, 2*ffn] instead of two, and one q8_1
            // quantisation of `h` instead of two. n == 1 keeps both halves
            // contiguous, so no ggml_cont is needed to split them.
            struct ggml_tensor * gu = ggml_mul_mat(c, L.ffn_gate_up, h);
            const int64_t F = L.ffn_gate->ne[1];
            g = ggml_silu(c, ggml_view_2d(c, gu, F, n, gu->nb[1], 0));
            u = ggml_view_2d(c, gu, F, n, gu->nb[1], (size_t) F * gu->nb[0]);
        } else {
            g = ggml_silu(c, ggml_mul_mat(c, L.ffn_gate, h));
            u = ggml_mul_mat(c, L.ffn_up, h);
        }
        h = ggml_mul_mat(c, L.ffn_down, ggml_mul(c, g, u));
        x = ggml_add(c, res, h);
    }

    x = ggml_mul(c, ggml_rms_norm(c, x, cfg_.rms_eps), out_norm_);
    struct ggml_tensor * last = ggml_cont(c, ggml_view_2d(c, x, cfg_.n_embd, 1, x->nb[1],
                                                          (size_t) (n - 1) * x->nb[1]));
    ggml_set_name(last, "hidden_last");
    ggml_set_output(last);
    ggml_build_forward_expand(gf, last);

    struct ggml_tensor * logits = ggml_mul_mat(c, lm_head_, last);   // [lm_head_size, 1]
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

namespace { struct bb_acc { double build=0, alloc=0, upload=0, compute=0, get=0; long calls=0; }; }
static bb_acc g_bb;
static bool bb_prof() { static const bool v = std::getenv("BREEZE_PROF") != nullptr; return v; }
void backbone_prof_dump() {
    if (!g_bb.calls) return;
    const double n = (double) g_bb.calls;
    const double tot = g_bb.build + g_bb.alloc + g_bb.upload + g_bb.compute + g_bb.get;
    fprintf(stderr, "  [breeze-prof %-8s] %5ld calls | per-call ms build=%.3f alloc=%.3f "
            "upload=%.3f compute=%.3f get=%.3f | sum=%.3f (%.1f%% CPU-side)\n",
            "backbone", g_bb.calls, g_bb.build/n, g_bb.alloc/n, g_bb.upload/n,
            g_bb.compute/n, g_bb.get/n, tot/n,
            100.0*(g_bb.build+g_bb.alloc+g_bb.upload+g_bb.get)/tot);
    g_bb = bb_acc{};
}

bool Backbone::run(int n, const float * embeds, const int32_t * codes,
                   std::vector<float> & hidden_last, std::vector<float> & logits) {
    if (!ensure_kv(n_past_ + n)) return false;
    // Single-frame decode: reuse the pre-allocated graph while the KV block and
    // slab size hold. Anything else (the prompt prefill, a ref-audio run) builds
    // a throwaway graph on the shared scheduler.
    const bool cached = (n == 1 && codes != nullptr);
    const int want_kv = round_up_stride(n_past_ + n);
    const bool P = bb_prof();
    std::chrono::steady_clock::time_point t;
    auto lap = [&]() {
        const double d = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t).count();
        t = std::chrono::steady_clock::now();
        return d;
    };
    if (P) t = std::chrono::steady_clock::now();

    struct ggml_context * c = nullptr;
    struct ggml_cgraph * gf = nullptr;
    ggml_backend_sched_t sch = sched_;
    if (cached) {
        if (dec_.gf && (dec_.n_kv_eff != want_kv || dec_.kv_alloc != kv_alloc_)) drop_decode_slot();
        if (!dec_.gf) {
            std::vector<ggml_backend_t> backends = { w_->backend() };
            if (w_->backend_cpu()) backends.push_back(w_->backend_cpu());
            dec_.sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(),
                                                BZ_LM_MAX_NODES, false, true);
            if (!dec_.sched) { error_msg_ = "backbone: sched_new failed"; return false; }
            dec_.meta.resize(compute_meta_.size());
            struct ggml_init_params dp = { dec_.meta.size(), dec_.meta.data(), true };
            dec_.ctx = ggml_init(dp);
            dec_.gf = build_graph(dec_.ctx, n, true);
            if (!ggml_backend_sched_alloc_graph(dec_.sched, dec_.gf)) {
                error_msg_ = "backbone: alloc decode graph"; drop_decode_slot(); return false;
            }
            dec_.n_kv_eff = want_kv;
            dec_.kv_alloc = kv_alloc_;
        }
        gf = dec_.gf;
        sch = dec_.sched;
        if (P) { g_bb.build += lap(); g_bb.alloc += lap(); }
    } else {
        struct ggml_init_params p = { compute_meta_.size(), compute_meta_.data(), true };
        c = ggml_init(p);
        gf = build_graph(c, n, codes != nullptr);
        if (P) g_bb.build += lap();
        if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
            error_msg_ = "backbone: alloc graph"; ggml_free(c); return false;
        }
        if (P) g_bb.alloc += lap();
    }

    std::vector<int32_t> posv(n);
    for (int i = 0; i < n; ++i) posv[i] = n_past_ + i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), posv.data(), 0, n * sizeof(int32_t));

    if (codes) {
        std::vector<int32_t> col(n);
        for (int cb = 0; cb < cfg_.n_codebooks; ++cb) {
            char nm[32]; snprintf(nm, sizeof(nm), "codes_%d", cb);
            for (int i = 0; i < n; ++i)
                col[i] = codes[(size_t) i * cfg_.n_codebooks + cb] + cb * cfg_.audio_vocab;
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, nm), col.data(), 0, n * sizeof(int32_t));
        }
    } else {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "embeds"), embeds, 0,
                                (size_t) n * cfg_.n_embd * sizeof(float));
    }
    {
        const int n_kv_eff = round_up_stride(n_past_ + n);
        std::vector<ggml_fp16_t> m((size_t) n_kv_eff * n);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n_kv_eff; ++j)
                m[(size_t) i * n_kv_eff + j] = (j <= n_past_ + i) ? zero : ninf;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask"), m.data(), 0,
                                m.size() * sizeof(ggml_fp16_t));
    }

    if (P) g_bb.upload += lap();
    if (ggml_backend_sched_graph_compute(sch, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "backbone: compute";
        if (!cached) { ggml_backend_sched_reset(sched_); ggml_free(c); }
        return false;
    }
    if (P) g_bb.compute += lap();
    struct ggml_tensor * hl = ggml_graph_get_tensor(gf, "hidden_last");
    hidden_last.resize(ggml_nelements(hl));
    ggml_backend_tensor_get(hl, hidden_last.data(), 0, ggml_nbytes(hl));
    struct ggml_tensor * lg = ggml_graph_get_tensor(gf, "logits");
    logits.resize(ggml_nelements(lg));
    ggml_backend_tensor_get(lg, logits.data(), 0, ggml_nbytes(lg));

    if (!cached) { ggml_backend_sched_reset(sched_); ggml_free(c); }
    n_past_ += n;
    if (P) { g_bb.get += lap(); g_bb.calls++; }
    return true;
}

bool Backbone::embed_frames(const int32_t * codes, int n, std::vector<float> & out) {
    out.clear();
    if (n <= 0) return true;
    struct ggml_init_params p = { compute_meta_.size(), compute_meta_.data(), true };
    struct ggml_context * c = ggml_init(p);
    struct ggml_cgraph * gf = ggml_new_graph_custom(c, BZ_LM_MAX_NODES, false);
    struct ggml_tensor * x = nullptr;
    for (int cb = 0; cb < cfg_.n_codebooks; ++cb) {
        char nm[32]; snprintf(nm, sizeof(nm), "codes_%d", cb);
        struct ggml_tensor * idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
        ggml_set_name(idx, nm); ggml_set_input(idx);
        struct ggml_tensor * r = ggml_get_rows(c, audio_embd_, idx);
        x = x ? ggml_add(c, x, r) : r;
    }
    ggml_set_name(x, "emb"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        error_msg_ = "backbone: alloc embed graph"; ggml_free(c); return false;
    }
    std::vector<int32_t> col(n);
    for (int cb = 0; cb < cfg_.n_codebooks; ++cb) {
        char nm[32]; snprintf(nm, sizeof(nm), "codes_%d", cb);
        for (int i = 0; i < n; ++i)
            col[i] = codes[(size_t) i * cfg_.n_codebooks + cb] + cb * cfg_.audio_vocab;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, nm), col.data(), 0, n * sizeof(int32_t));
    }
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "backbone: embed compute";
        ggml_backend_sched_reset(sched_); ggml_free(c); return false;
    }
    struct ggml_tensor * o = ggml_graph_get_tensor(gf, "emb");
    out.resize(ggml_nelements(o));
    ggml_backend_tensor_get(o, out.data(), 0, ggml_nbytes(o));
    ggml_backend_sched_reset(sched_);
    ggml_free(c);
    return true;
}

bool Backbone::prefill_embeds(const float * e, int n, std::vector<float> & h, std::vector<float> & l) {
    return run(n, e, nullptr, h, l);
}
bool Backbone::decode_frame(const int32_t * codes, std::vector<float> & h, std::vector<float> & l) {
    return run(1, nullptr, codes, h, l);
}
bool Backbone::prefill_frames(const int32_t * codes, int n, std::vector<float> & h, std::vector<float> & l) {
    if (n <= 0) { h.clear(); l.clear(); return true; }
    return run(n, nullptr, codes, h, l);
}

} // namespace breeze
