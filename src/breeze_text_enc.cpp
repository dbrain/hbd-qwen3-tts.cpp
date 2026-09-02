#include "breeze_text_enc.h"

#include <cmath>
#include <cstring>
#include <cstdio>

#define BZ_TE_MAX_NODES 8192

namespace breeze {

namespace {

// Gemma RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w). The (1 + w) convention
// is why the weights cannot be fed straight to ggml_rms_norm+mul.
inline struct ggml_tensor * gemma_norm(struct ggml_context * c, struct ggml_tensor * x,
                                       struct ggml_tensor * w, float eps) {
    struct ggml_tensor * n = ggml_rms_norm(c, x, eps);
    return ggml_add(c, n, ggml_mul(c, n, w));
}

} // namespace

void TextEncoder::free_state() {
    if (sched_) { ggml_backend_sched_free(sched_); sched_ = nullptr; }
    compute_meta_.clear();
    layers_.clear();
}

bool TextEncoder::init(BreezeWeights * w) {
    free_state();
    w_ = w;
    cfg_ = w->cfg().te;

    tok_embd_ = w->get("text_enc.token_embd.weight");
    eoi_embd_ = w->get("text_enc.eoi_embd.weight");
    out_norm_ = w->get("text_enc.output_norm.weight");
    proj_     = w->get("text_enc.proj.weight");
    if (!tok_embd_ || !out_norm_ || !proj_) {
        error_msg_ = "text encoder: missing token_embd/output_norm/proj";
        return false;
    }

    layers_.resize(cfg_.n_layers);
    for (int i = 0; i < cfg_.n_layers; ++i) {
        auto & L = layers_[i];
        L.attn_norm      = w->getf("text_enc.blk.%d.attn_norm.weight", i);
        L.attn_post_norm = w->getf("text_enc.blk.%d.attn_post_norm.weight", i);
        L.wq             = w->getf("text_enc.blk.%d.attn_q.weight", i);
        L.wk             = w->getf("text_enc.blk.%d.attn_k.weight", i);
        L.wv             = w->getf("text_enc.blk.%d.attn_v.weight", i);
        L.wo             = w->getf("text_enc.blk.%d.attn_output.weight", i);
        L.q_norm         = w->getf("text_enc.blk.%d.attn_q_norm.weight", i);
        L.k_norm         = w->getf("text_enc.blk.%d.attn_k_norm.weight", i);
        L.ffn_norm       = w->getf("text_enc.blk.%d.ffn_norm.weight", i);
        L.ffn_post_norm  = w->getf("text_enc.blk.%d.ffn_post_norm.weight", i);
        L.ffn_gate       = w->getf("text_enc.blk.%d.ffn_gate.weight", i);
        L.ffn_up         = w->getf("text_enc.blk.%d.ffn_up.weight", i);
        L.ffn_down       = w->getf("text_enc.blk.%d.ffn_down.weight", i);
        L.full           = cfg_.layer_is_full[i] != 0;
        if (!L.attn_norm || !L.attn_post_norm || !L.wq || !L.wk || !L.wv || !L.wo ||
            !L.q_norm || !L.k_norm || !L.ffn_norm || !L.ffn_post_norm ||
            !L.ffn_gate || !L.ffn_up || !L.ffn_down) {
            error_msg_ = "text encoder: missing tensor in blk " + std::to_string(i);
            return false;
        }
    }

    std::vector<ggml_backend_t> backends = { w->backend() };
    if (w->backend_cpu()) backends.push_back(w->backend_cpu());
    sched_ = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(),
                                    BZ_TE_MAX_NODES, false, true);
    if (!sched_) { error_msg_ = "text encoder: sched_new failed"; return false; }
    compute_meta_.resize(ggml_tensor_overhead() * BZ_TE_MAX_NODES +
                         ggml_graph_overhead_custom(BZ_TE_MAX_NODES, false));
    return true;
}

struct ggml_cgraph * TextEncoder::build_graph(struct ggml_context * c, int n,
                                              bool project, bool has_eoi) {
    struct ggml_cgraph * gf = ggml_new_graph_custom(c, BZ_TE_MAX_NODES, false);

    const int hd  = cfg_.head_dim;
    const int nh  = cfg_.n_head;
    const int nkv = cfg_.n_kv_head;
    const float scale = 1.0f / std::sqrt((float) cfg_.qk_scalar);

    struct ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    ggml_set_name(ids, "ids"); ggml_set_input(ids);

    struct ggml_tensor * x = ggml_get_rows(c, tok_embd_, ids);       // [n_embd, n]
    x = ggml_scale(c, x, cfg_.embed_scale);
    if (has_eoi && eoi_embd_) {
        // T5Gemma2TextScaledWordEmbedding: the end-of-image id bypasses the
        // table AND the sqrt(hidden) scale. Only built when an eoi id is
        // actually present, so the common path pays nothing.
        struct ggml_tensor * m = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, n);
        ggml_set_name(m, "eoi_mask"); ggml_set_input(m);
        struct ggml_tensor * keep = ggml_scale_bias(c, m, -1.0f, 1.0f);
        x = ggml_add(c, ggml_mul(c, x, keep),
                     ggml_mul(c, ggml_repeat(c, eoi_embd_, x), m));
    }

    struct ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    ggml_set_name(pos, "pos"); ggml_set_input(pos);

    // One mask per attention flavour. `full` layers over a single unpadded
    // segment need no mask at all; sliding layers get the bidirectional window.
    struct ggml_tensor * mask_slide = ggml_new_tensor_2d(c, GGML_TYPE_F32, n, n);
    ggml_set_name(mask_slide, "mask_slide"); ggml_set_input(mask_slide);

    for (int il = 0; il < cfg_.n_layers; ++il) {
        const layer & L = layers_[il];
        struct ggml_tensor * res = x;
        struct ggml_tensor * h = gemma_norm(c, x, L.attn_norm, cfg_.rms_eps);

        struct ggml_tensor * q = ggml_mul_mat(c, L.wq, h);   // [nh*hd, n]
        struct ggml_tensor * k = ggml_mul_mat(c, L.wk, h);   // [nkv*hd, n]
        struct ggml_tensor * v = ggml_mul_mat(c, L.wv, h);
        q = ggml_reshape_3d(c, q, hd, nh,  n);
        k = ggml_reshape_3d(c, k, hd, nkv, n);
        v = ggml_reshape_3d(c, v, hd, nkv, n);

        q = gemma_norm(c, q, L.q_norm, cfg_.rms_eps);
        k = gemma_norm(c, k, L.k_norm, cfg_.rms_eps);

        // rope_type "linear" with factor F == positions scaled by 1/F, which is
        // exactly ggml's freq_scale. Sliding layers are plain default RoPE.
        const float theta = L.full ? cfg_.rope_theta_full : cfg_.rope_theta_sliding;
        const float fscale = (L.full && cfg_.rope_full_kind == rope_kind::linear &&
                              cfg_.rope_factor_full > 0.0f)
                             ? 1.0f / cfg_.rope_factor_full : 1.0f;
        q = ggml_rope_ext(c, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                          theta, fscale, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(c, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                          theta, fscale, 0.0f, 1.0f, 0.0f, 0.0f);

        struct ggml_tensor * Q = ggml_permute(c, q, 0, 2, 1, 3);     // [hd, n, nh]
        struct ggml_tensor * K = ggml_permute(c, k, 0, 2, 1, 3);     // [hd, n, nkv]
        struct ggml_tensor * kq = ggml_mul_mat(c, K, Q);             // [n, n, nh]
        kq = ggml_soft_max_ext(c, kq, L.full ? nullptr : mask_slide, scale, 0.0f);
        struct ggml_tensor * V = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3)); // [n, hd, nkv]
        struct ggml_tensor * kqv = ggml_mul_mat(c, V, kq);           // [hd, n, nh]
        kqv = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), hd * nh, n);
        struct ggml_tensor * attn = ggml_mul_mat(c, L.wo, kqv);      // [n_embd, n]

        attn = gemma_norm(c, attn, L.attn_post_norm, cfg_.rms_eps);
        x = ggml_add(c, res, attn);

        res = x;
        h = gemma_norm(c, x, L.ffn_norm, cfg_.rms_eps);
        struct ggml_tensor * g = ggml_gelu(c, ggml_mul_mat(c, L.ffn_gate, h));
        struct ggml_tensor * u = ggml_mul_mat(c, L.ffn_up, h);
        h = ggml_mul_mat(c, L.ffn_down, ggml_mul(c, g, u));
        h = gemma_norm(c, h, L.ffn_post_norm, cfg_.rms_eps);
        x = ggml_add(c, res, h);
    }

    x = gemma_norm(c, x, out_norm_, cfg_.rms_eps);
    if (project) x = ggml_mul_mat(c, proj_, x);   // [2048, n]
    ggml_set_name(x, "out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

bool TextEncoder::run(const std::vector<int32_t> & ids, bool project,
                      std::vector<float> & out) {
    const int n = (int) ids.size();
    if (n <= 0) { out.clear(); return true; }

    bool has_eoi = false;
    for (int32_t id : ids) if (id == cfg_.eoi_token) { has_eoi = true; break; }

    struct ggml_init_params p = { compute_meta_.size(), compute_meta_.data(), true };
    struct ggml_context * c = ggml_init(p);
    struct ggml_cgraph * gf = build_graph(c, n, project, has_eoi);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        error_msg_ = "text encoder: alloc graph"; ggml_free(c); return false;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ids"), ids.data(), 0,
                            n * sizeof(int32_t));
    {
        std::vector<int32_t> posv(n);
        for (int i = 0; i < n; ++i) posv[i] = i;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), posv.data(), 0,
                                n * sizeof(int32_t));
    }
    if (has_eoi) {
        std::vector<float> m(n, 0.0f);
        for (int i = 0; i < n; ++i) if (ids[i] == cfg_.eoi_token) m[i] = 1.0f;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "eoi_mask"), m.data(), 0,
                                n * sizeof(float));
    }
    {
        // Bidirectional sliding window, exactly as the reference builds it:
        //   left  = (W + 1) / 2      allowed q-kv distance  [0, left)
        //   right = W / 2 + 1        allowed kv-q distance  [0, right)
        // With W = 512 that is dist in [-256, 255].
        const int W = cfg_.sliding_window;
        const int left = (W + 1) / 2, right = W / 2 + 1;
        std::vector<float> m((size_t) n * n, 0.0f);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const int d = i - j;
                const bool ok = (d >= 0 && d < left) || (d < 0 && -d < right);
                m[(size_t) i * n + j] = ok ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask_slide"), m.data(), 0,
                                m.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "text encoder: compute";
        ggml_backend_sched_reset(sched_); ggml_free(c); return false;
    }
    struct ggml_tensor * o = ggml_graph_get_tensor(gf, "out");
    out.resize(ggml_nelements(o));
    ggml_backend_tensor_get(o, out.data(), 0, ggml_nbytes(o));
    ggml_backend_sched_reset(sched_);
    ggml_free(c);
    return true;
}

bool TextEncoder::encode(const std::vector<int32_t> & ids, std::vector<float> & out) {
    return run(ids, /*project=*/true, out);
}
bool TextEncoder::encode_hidden(const std::vector<int32_t> & ids, std::vector<float> & out) {
    return run(ids, /*project=*/false, out);
}

} // namespace breeze
