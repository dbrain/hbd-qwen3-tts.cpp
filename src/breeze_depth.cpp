#include "breeze_depth.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <cstdlib>

#define BZ_DD_MAX_NODES 4096

namespace breeze {

// BREEZE_PROF=1 splits each graph launch into build / alloc / upload / compute
// / readback so a launch-bound loop is distinguishable from a FLOP-bound one.
namespace prof {
using clk = std::chrono::steady_clock;
inline bool on() { static const bool v = std::getenv("BREEZE_PROF") != nullptr; return v; }
inline double ms(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}
struct acc { double build=0, alloc=0, upload=0, compute=0, get=0; long calls=0;
    void dump(const char * label) {
        if (!calls) return;
        const double n = (double) calls;
        const double tot = build + alloc + upload + compute + get;
        fprintf(stderr, "  [breeze-prof %-8s] %5ld calls | per-call ms build=%.3f alloc=%.3f "
                "upload=%.3f compute=%.3f get=%.3f | sum=%.3f (%.1f%% CPU-side)\n",
                label, calls, build/n, alloc/n, upload/n, compute/n, get/n, tot/n,
                100.0*(build+alloc+upload+get)/tot);
        build=alloc=upload=compute=get=0; calls=0;
    }
};
} // namespace prof

namespace {
template <class G> void drop_graph_slot(G & g) {
    if (g.sched) { ggml_backend_sched_free(g.sched); g.sched = nullptr; }
    if (g.ctx)   { ggml_free(g.ctx); g.ctx = nullptr; }
    g.gf = nullptr; g.allocated = false; g.meta.clear();
}
} // namespace

void DepthDecoder::drop_pair_slots() {
    drop_graph_slot(pair_pre_);
    for (auto & g : pair_step_) drop_graph_slot(g);
    pair_step_.clear();
    if (pair_kv_buf_) { ggml_backend_buffer_free(pair_kv_buf_); pair_kv_buf_ = nullptr; }
    if (pair_kv_ctx_) { ggml_free(pair_kv_ctx_); pair_kv_ctx_ = nullptr; }
    pk_.clear(); pv_.clear();
}

void DepthDecoder::free_state() {
    auto drop = [](graph_slot & g) { drop_graph_slot(g); };
    drop_pair_slots();
    drop(pre_);
    for (auto & g : step_) drop(g);
    step_.clear();
    if (kv_buf_)  { ggml_backend_buffer_free(kv_buf_); kv_buf_ = nullptr; }
    if (kv_ctx_)  { ggml_free(kv_ctx_); kv_ctx_ = nullptr; }
    if (aux_buf_) { ggml_backend_buffer_free(aux_buf_); aux_buf_ = nullptr; }
    if (aux_ctx_) { ggml_free(aux_ctx_); aux_ctx_ = nullptr; }
    layers_.clear(); k_cache_.clear(); v_cache_.clear();
    heads_ = nullptr;
}

bool DepthDecoder::init(BreezeWeights * w) {
    free_state();
    w_ = w;
    cfg_ = w->cfg().dd;

    tok_embd_ = w->get("depth.token_embd.weight");
    in_proj_  = w->get("depth.in_proj.weight");
    out_norm_ = w->get("depth.output_norm.weight");
    if (!tok_embd_ || !in_proj_ || !out_norm_) {
        error_msg_ = "depth: missing token_embd/in_proj/output_norm";
        return false;
    }
    heads_ = w->get("depth.codebooks_head.weight");
    if (!heads_) {
        error_msg_ = "depth: missing depth.codebooks_head.weight -- re-run "
                     "scripts/convert_breeze_to_gguf.py";
        return false;
    }
    layers_.resize(cfg_.n_layers);
    for (int i = 0; i < cfg_.n_layers; ++i) {
        auto & L = layers_[i];
        L.attn_norm = w->getf("depth.blk.%d.attn_norm.weight", i);
        L.wq        = w->getf("depth.blk.%d.attn_q.weight", i);
        L.wk        = w->getf("depth.blk.%d.attn_k.weight", i);
        L.wv        = w->getf("depth.blk.%d.attn_v.weight", i);
        L.wo        = w->getf("depth.blk.%d.attn_output.weight", i);
        L.ffn_norm  = w->getf("depth.blk.%d.ffn_norm.weight", i);
        L.ffn_gate  = w->getf("depth.blk.%d.ffn_gate.weight", i);
        L.ffn_up    = w->getf("depth.blk.%d.ffn_up.weight", i);
        L.ffn_down  = w->getf("depth.blk.%d.ffn_down.weight", i);
        if (!L.attn_norm || !L.wq || !L.wk || !L.wv || !L.wo || !L.ffn_norm ||
            !L.ffn_gate || !L.ffn_up || !L.ffn_down) {
            error_msg_ = "depth: missing tensor in blk " + std::to_string(i);
            return false;
        }
        char nm[64];
        snprintf(nm, sizeof(nm), "depth.blk.%d.ffn_gate_up.weight", i);
        L.ffn_gate_up = w->fuse_rows(L.ffn_gate, L.ffn_up, nm);
        snprintf(nm, sizeof(nm), "depth.blk.%d.attn_qk.weight", i);
        if (struct ggml_tensor * qk = w->fuse_rows(L.wq, L.wk, nm)) {
            snprintf(nm, sizeof(nm), "depth.blk.%d.attn_qkv.weight", i);
            L.wqkv = w->fuse_rows(qk, L.wv, nm);
        }
    }

    // llama3 RoPE, as the divisor ggml_rope_ext applies per frequency pair.
    // orig_ctx is 16 here (the codebook axis), so 35 of the 64 pairs move --
    // ignoring rope_scaling would detune most of the codebook positions.
    std::vector<float> ff;
    if (cfg_.rope == rope_kind::llama3) {
        ff = build_llama3_freq_factors(cfg_.head_dim, cfg_.rope_theta, cfg_.rope_factor,
                                       cfg_.rope_low_freq, cfg_.rope_high_freq,
                                       cfg_.rope_orig_ctx);
    }
    // Exactly the positions the frame uses. Fixed (not grown) so every step
    // attends over the same n_kv and the graph shape never moves.
    kv_slots_ = cfg_.n_codebooks + 1;
    {
        const int nl = cfg_.n_layers;
        struct ggml_init_params p = { ggml_tensor_overhead() * (nl * 2 + 8), nullptr, true };
        kv_ctx_ = ggml_init(p);
        k_cache_.resize(nl); v_cache_.resize(nl);
        for (int i = 0; i < nl; ++i) {
            k_cache_[i] = ggml_new_tensor_3d(kv_ctx_, GGML_TYPE_F32, cfg_.head_dim,
                                             cfg_.n_kv_head, kv_slots_);
            ggml_format_name(k_cache_[i], "dd_k_%d", i);
            v_cache_[i] = ggml_new_tensor_3d(kv_ctx_, GGML_TYPE_F32, cfg_.head_dim,
                                             cfg_.n_kv_head, kv_slots_);
            ggml_format_name(v_cache_[i], "dd_v_%d", i);
        }
        kv_buf_ = ggml_backend_alloc_ctx_tensors(kv_ctx_, w->backend());
        if (!kv_buf_) { error_msg_ = "depth: kv alloc failed"; return false; }
    }
    if (!ff.empty()) {
        struct ggml_init_params p = { ggml_tensor_overhead() * 4 + ff.size() * 4 + 1024,
                                      nullptr, true };
        aux_ctx_ = ggml_init(p);
        rope_ff_ = ggml_new_tensor_1d(aux_ctx_, GGML_TYPE_F32, (int64_t) ff.size());
        ggml_set_name(rope_ff_, "rope_freq_factors");
        aux_buf_ = ggml_backend_alloc_ctx_tensors(aux_ctx_, w->backend());
        if (!aux_buf_) { error_msg_ = "depth: aux alloc failed"; return false; }
        ggml_backend_tensor_set(rope_ff_, ff.data(), 0, ff.size() * sizeof(float));
    }

    return true;
}

size_t DepthDecoder::sched_bytes() const {
    if (!w_) return 0;
    size_t n = pre_.sched ? ggml_backend_sched_get_buffer_size(pre_.sched, w_->backend()) : 0;
    for (const auto & g : step_)
        if (g.sched) n += ggml_backend_sched_get_buffer_size(g.sched, w_->backend());
    if (pair_pre_.sched) n += ggml_backend_sched_get_buffer_size(pair_pre_.sched, w_->backend());
    for (const auto & g : pair_step_)
        if (g.sched) n += ggml_backend_sched_get_buffer_size(g.sched, w_->backend());
    return n;
}

void DepthDecoder::log_sched_detail() const {
    if (!w_) return;
    const double M = 1024.0 * 1024.0;
    const double pre = pre_.sched
        ? ggml_backend_sched_get_buffer_size(pre_.sched, w_->backend()) / M : 0.0;
    double each = 0.0; int n = 0;
    for (const auto & g : step_)
        if (g.sched) { each += ggml_backend_sched_get_buffer_size(g.sched, w_->backend()) / M; ++n; }
    fprintf(stderr, "  [vram-breeze depth-sched] prefill=%.2f MiB + %d step graphs = %.2f MiB "
            "(%.2f MiB each)\n", pre, n, each, n ? each / n : 0.0);
}

bool DepthDecoder::init_slot(graph_slot & g, int n, int n_past, int head_idx) {
    std::vector<ggml_backend_t> backends = { w_->backend() };
    if (w_->backend_cpu()) backends.push_back(w_->backend_cpu());
    g.sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(),
                                     BZ_DD_MAX_NODES, false, true);
    if (!g.sched) { error_msg_ = "depth: sched_new failed"; return false; }
    g.meta.resize(ggml_tensor_overhead() * BZ_DD_MAX_NODES +
                  ggml_graph_overhead_custom(BZ_DD_MAX_NODES, false));
    struct ggml_init_params p = { g.meta.size(), g.meta.data(), true };
    g.ctx = ggml_init(p);
    g.gf = build_graph(g.ctx, n, n_past, head_idx);
    if (!ggml_backend_sched_alloc_graph(g.sched, g.gf)) {
        error_msg_ = "depth: alloc graph"; return false;
    }
    g.allocated = true;
    return true;
}

struct ggml_cgraph * DepthDecoder::build_graph(struct ggml_context * c, int n,
                                               int n_past, int head_idx) {
    struct ggml_cgraph * gf = ggml_new_graph_custom(c, BZ_DD_MAX_NODES, false);
    const int hd = cfg_.head_dim, nh = cfg_.n_head, nkv = cfg_.n_kv_head;
    const int n_kv = kv_slots_;   // fixed; the mask hides the unwritten slots
    const float scale = 1.0f / std::sqrt((float) hd);

    // audio_embed_size-wide inputs, then a single 2048->1024 projector.
    struct ggml_tensor * e;
    {
        struct ggml_tensor * idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
        ggml_set_name(idx, "codes"); ggml_set_input(idx);
        e = ggml_get_rows(c, tok_embd_, idx);            // [audio_embd, n]
        if (n_past == 0) {
            // Position 0 is the backbone hidden state, not a codebook embedding.
            struct ggml_tensor * bh = ggml_new_tensor_2d(c, GGML_TYPE_F32, cfg_.audio_embd_size, 1);
            ggml_set_name(bh, "backbone_hidden"); ggml_set_input(bh);
            struct ggml_tensor * rest = ggml_view_2d(c, e, cfg_.audio_embd_size, n - 1,
                                                     e->nb[1], e->nb[1]);
            e = ggml_concat(c, bh, ggml_cont(c, rest), 1);
        }
    }
    struct ggml_tensor * x = ggml_mul_mat(c, in_proj_, e);   // [n_embd, n]

    struct ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    ggml_set_name(pos, "pos"); ggml_set_input(pos);
    struct ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_kv, n);
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
        // No q_norm/k_norm here: the depth decoder is Qwen2-style attention.
        q = ggml_rope_ext(c, q, pos, rope_ff_, hd, GGML_ROPE_TYPE_NEOX, 0,
                          cfg_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(c, k, pos, rope_ff_, hd, GGML_ROPE_TYPE_NEOX, 0,
                          cfg_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        struct ggml_tensor * ks = k_cache_[il], * vs = v_cache_[il];
        ggml_build_forward_expand(gf, ggml_set_rows(c,
            ggml_view_2d(c, ks, hd * nkv, kv_slots_, ks->nb[2], 0),
            ggml_view_2d(c, k,  hd * nkv, n, k->nb[2], 0), pos));
        ggml_build_forward_expand(gf, ggml_set_rows(c,
            ggml_view_2d(c, vs, hd * nkv, kv_slots_, vs->nb[2], 0),
            ggml_view_2d(c, v,  hd * nkv, n, v->nb[2], 0), pos));

        struct ggml_tensor * kview = ggml_view_3d(c, ks, hd, nkv, n_kv, ks->nb[1], ks->nb[2], 0);
        struct ggml_tensor * vview = ggml_view_3d(c, vs, hd, nkv, n_kv, vs->nb[1], vs->nb[2], 0);
        struct ggml_tensor * Q = ggml_permute(c, q, 0, 2, 1, 3);
        struct ggml_tensor * K = ggml_permute(c, kview, 0, 2, 1, 3);
        struct ggml_tensor * V = ggml_permute(c, vview, 0, 2, 1, 3);
        struct ggml_tensor * kq = ggml_mul_mat(c, K, Q);              // [n_kv, n, nh]
        kq = ggml_soft_max_ext(c, kq, mask, scale, 0.0f);
        V = ggml_cont(c, ggml_permute(c, vview, 1, 2, 0, 3));         // [n_kv, hd, nkv]
        struct ggml_tensor * kqv = ggml_mul_mat(c, V, kq);            // [hd, n, nh]
        kqv = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), nh * hd, n);
        struct ggml_tensor * attn = ggml_mul_mat(c, L.wo, kqv);

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
    // This step's codebook head only: a [n_embd, vocab] slice of the packed
    // [n_embd, vocab, n_codebooks-1] tensor, chosen at graph-build time.
    struct ggml_tensor * head = ggml_view_2d(c, heads_, heads_->ne[0], heads_->ne[1],
                                             heads_->nb[1], (size_t) head_idx * heads_->nb[2]);
    struct ggml_tensor * logits = ggml_mul_mat(c, head, last);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

prof::acc g_dd_prof;
void depth_prof_dump() { g_dd_prof.dump("depth"); }

bool DepthDecoder::step(int n, int n_past, int head_idx, const float * hidden,
                        const int32_t * codes_in, std::vector<float> & logits) {
    const bool P = prof::on();
    prof::clk::time_point t;
    if (P) t = prof::clk::now();
    if (step_.empty()) step_.resize(cfg_.n_codebooks - 1);
    graph_slot & g = (n_past == 0) ? pre_ : step_[head_idx];
    if (!g.allocated && !init_slot(g, n, n_past, head_idx)) return false;
    struct ggml_cgraph * gf = g.gf;
    if (P) { g_dd_prof.build += prof::ms(t); t = prof::clk::now(); }
    if (P) { g_dd_prof.alloc += prof::ms(t); t = prof::clk::now(); }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "codes"), codes_in, 0, n * sizeof(int32_t));
    if (n_past == 0) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "backbone_hidden"), hidden, 0,
                                (size_t) cfg_.audio_embd_size * sizeof(float));
    }
    {
        std::vector<int32_t> posv(n);
        for (int i = 0; i < n; ++i) posv[i] = n_past + i;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), posv.data(), 0,
                                n * sizeof(int32_t));
        // MUST match the graph's fixed n_kv (kv_slots_), not n_past+n: the mask
        // tensor is sized for the whole slab and a short upload leaves the tail
        // uninitialised, which reads as "attend to stale KV".
        const int n_kv = kv_slots_;
        std::vector<float> m((size_t) n_kv * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n_kv; ++j)
                m[(size_t) i * n_kv + j] = (j <= n_past + i) ? 0.0f : -INFINITY;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask"), m.data(), 0,
                                m.size() * sizeof(float));
    }
    if (P) { g_dd_prof.upload += prof::ms(t); t = prof::clk::now(); }
    // No reset: the graph stays allocated so this call skips split+alloc.
    if (ggml_backend_sched_graph_compute(g.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "depth: compute"; return false;
    }
    if (P) { g_dd_prof.compute += prof::ms(t); t = prof::clk::now(); }
    struct ggml_tensor * lt = ggml_graph_get_tensor(gf, "logits");
    logits.resize(ggml_nelements(lt));
    ggml_backend_tensor_get(lt, logits.data(), 0, ggml_nbytes(lt));
    if (P) { g_dd_prof.get += prof::ms(t); g_dd_prof.calls++; }
    return true;
}

namespace {
inline float rng_next(uint32_t & s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (float) ((s >> 8) & 0xFFFFFF) / (float) 0x1000000;
}
int sample_from(std::vector<float> & lg, int V, float temp, int top_k, float top_p,
                uint32_t & rng) {
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < V; ++i) if (lg[i] > lg[best]) best = i;
        return best;
    }
    std::vector<std::pair<float, int>> p(V);
    float mx = -INFINITY;
    for (int i = 0; i < V; ++i) { p[i] = { lg[i] / temp, i }; mx = std::max(mx, p[i].first); }
    double sum = 0;
    for (auto & e : p) { e.first = std::exp(e.first - mx); sum += e.first; }
    for (auto & e : p) e.first = (float) (e.first / sum);
    int keep = (top_k > 0 && top_k < V) ? top_k : V;
    // Only the top `keep` are ever read: partition them out (O(V)) and sort
    // just those, instead of sorting all ~2051 per codebook (16x a frame).
    auto desc = [](const auto & a, const auto & b) { return a.first > b.first; };
    if (keep < V) {
        std::nth_element(p.begin(), p.begin() + keep - 1, p.end(), desc);
        std::sort(p.begin(), p.begin() + keep, desc);
    } else {
        std::sort(p.begin(), p.end(), desc);
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        double c = 0; int i = 0;
        for (; i < keep; ++i) { c += p[i].first; if (c > top_p) { ++i; break; } }
        keep = std::max(1, i);
    }
    double tot = 0;
    for (int i = 0; i < keep; ++i) tot += p[i].first;
    double r = rng_next(rng) * tot, acc = 0;
    for (int i = 0; i < keep; ++i) { acc += p[i].first; if (r <= acc) return p[i].second; }
    return p[0].second;
}
} // namespace

bool DepthDecoder::run_frame(const float * backbone_hidden, int32_t * codes,
                             float temperature, int top_k, float top_p,
                             uint32_t & rng, std::vector<float> * logits_out) {
    const int NC = cfg_.n_codebooks;
    const int V  = cfg_.vocab;
    const int CB = w_->cfg().cc.codebook_size;   // 2048 valid codes; 2048..V-1 reserved
    if (logits_out) logits_out->clear();

    std::vector<float> logits;
    // Prefill: [placeholder(0), c0]. Position 0 carries the backbone hidden
    // state; position 1 embeds c0 with codebook-0's offset (clamp(p-1,0) == 0).
    int32_t prefill[2] = { 0, codes[0] };
    if (!step(2, 0, 0, backbone_hidden, prefill, logits)) return false;

    for (int cb = 1; cb < NC; ++cb) {
        if ((int) logits.size() < V) { error_msg_ = "depth: short logits"; return false; }
        for (int i = CB; i < V; ++i) logits[i] = -INFINITY;   // reserved codec ids
        if (logits_out) logits_out->insert(logits_out->end(), logits.begin(), logits.begin() + V);
        codes[cb] = sample_from(logits, V, temperature, top_k, top_p, rng);
        if (cb == NC - 1) break;
        // Next step: cache_position cb+1, whose embedding offset is cb*vocab
        // and whose head index is cb.
        int32_t tok = codes[cb] + cb * V;
        if (!step(1, cb + 1, cb, nullptr, &tok, logits)) return false;
    }
    return true;
}

// Columns [0, n_per) are sequence A, [n_per, 2*n_per) sequence B (CFG: the
// conditional and unconditional branch; concurrency: two requests). Both sit at
// the same codebook positions, so the pair shares one position vector and one
// mask, and their KV lives in ONE [hd, nkv, slots, 2] slab owned by this
// decoder -- attention then runs once with a batch dimension of 2 instead of
// once per sequence. The step is launch-bound (16 tiny graphs per frame), so
// what a batch-2 step costs is mostly its kernel COUNT: one attention, and the
// fused QKV / gate-up matmuls the single-sequence graph already uses.
struct ggml_cgraph * DepthDecoder::build_pair_graph(struct ggml_context * c, int n_per,
                                                    int n_past, int head_idx,
                                                    const DepthDecoder & /*other*/) {
    struct ggml_cgraph * gf = ggml_new_graph_custom(c, BZ_DD_MAX_NODES, false);
    const int hd = cfg_.head_dim, nh = cfg_.n_head, nkv = cfg_.n_kv_head;
    const int n_kv = kv_slots_;
    const int n = 2 * n_per;
    const float scale = 1.0f / std::sqrt((float) hd);

    // `codes` holds both sequences' input ids, A's n_per then B's (CFG uploads
    // the same ids twice).
    struct ggml_tensor * e;
    {
        struct ggml_tensor * idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
        ggml_set_name(idx, "codes"); ggml_set_input(idx);
        struct ggml_tensor * e1 = ggml_get_rows(c, tok_embd_, idx);   // [audio_embd, n]
        if (n_past == 0) {
            struct ggml_tensor * bh = ggml_new_tensor_2d(c, GGML_TYPE_F32, cfg_.audio_embd_size, 2);
            ggml_set_name(bh, "backbone_hidden"); ggml_set_input(bh);
            struct ggml_tensor * part[2];
            for (int sq = 0; sq < 2; ++sq) {
                struct ggml_tensor * rest = ggml_cont(c, ggml_view_2d(c, e1, cfg_.audio_embd_size,
                                                      n_per - 1, e1->nb[1],
                                                      (size_t) (sq * n_per + 1) * e1->nb[1]));
                struct ggml_tensor * b = ggml_view_2d(c, bh, cfg_.audio_embd_size, 1, bh->nb[1],
                                                      (size_t) sq * bh->nb[1]);
                part[sq] = ggml_concat(c, b, rest, 1);
            }
            e = ggml_concat(c, part[0], part[1], 1);
        } else {
            e = e1;
        }
    }
    struct ggml_tensor * x = ggml_mul_mat(c, in_proj_, e);   // [n_embd, n]

    // Positions per column (A's then B's, identical) for RoPE; row indices
    // into the slab per sequence for the KV writes.
    struct ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    ggml_set_name(pos, "pos"); ggml_set_input(pos);
    struct ggml_tensor * kvi = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_per);
    ggml_set_name(kvi, "kvi"); ggml_set_input(kvi);
    struct ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_kv, n_per);
    ggml_set_name(mask, "mask"); ggml_set_input(mask);

    for (int il = 0; il < cfg_.n_layers; ++il) {
        const layer & L = layers_[il];
        struct ggml_tensor * res = x;
        struct ggml_tensor * h = ggml_mul(c, ggml_rms_norm(c, x, cfg_.rms_eps), L.attn_norm);

        struct ggml_tensor * q, * k, * v;   // v: [hd*nkv, n] rows, contiguous per row
        if (L.wqkv) {
            struct ggml_tensor * qkv = ggml_mul_mat(c, L.wqkv, h);
            const size_t es = ggml_type_size(qkv->type);
            const int64_t nq = hd * nh, nk = hd * nkv;
            q = ggml_view_3d(c, qkv, hd, nh,  n, hd * es, qkv->nb[1], 0);
            k = ggml_view_3d(c, qkv, hd, nkv, n, hd * es, qkv->nb[1], (size_t) nq * es);
            v = ggml_view_3d(c, qkv, nk, n_per, 2, qkv->nb[1], qkv->nb[1] * n_per, (size_t) (nq + nk) * es);
        } else {
            q = ggml_reshape_3d(c, ggml_mul_mat(c, L.wq, h), hd, nh,  n);
            k = ggml_reshape_3d(c, ggml_mul_mat(c, L.wk, h), hd, nkv, n);
            v = ggml_reshape_3d(c, ggml_mul_mat(c, L.wv, h), hd * nkv, n_per, 2);
        }
        q = ggml_rope_ext(c, q, pos, rope_ff_, hd, GGML_ROPE_TYPE_NEOX, 0,
                          cfg_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(c, k, pos, rope_ff_, hd, GGML_ROPE_TYPE_NEOX, 0,
                          cfg_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        struct ggml_tensor * ks = pk_[il], * vs = pv_[il];   // [hd, nkv, slots, 2]
        ggml_build_forward_expand(gf, ggml_set_rows(c,
            ggml_reshape_3d(c, ks, hd * nkv, kv_slots_, 2),
            ggml_reshape_3d(c, k, hd * nkv, n_per, 2), kvi));
        ggml_build_forward_expand(gf, ggml_set_rows(c,
            ggml_reshape_3d(c, vs, hd * nkv, kv_slots_, 2), v, kvi));

        struct ggml_tensor * Q = ggml_permute(c, ggml_reshape_4d(c, q, hd, nh, n_per, 2), 0, 2, 1, 3);
        struct ggml_tensor * K = ggml_permute(c, ks, 0, 2, 1, 3);       // [hd, slots, nkv, 2]
        struct ggml_tensor * kq = ggml_mul_mat(c, K, Q);                 // [slots, n_per, nh, 2]
        kq = ggml_soft_max_ext(c, kq, mask, scale, 0.0f);
        struct ggml_tensor * V = ggml_cont(c, ggml_permute(c, vs, 1, 2, 0, 3));   // [slots, hd, nkv, 2]
        struct ggml_tensor * kqv = ggml_mul_mat(c, V, kq);               // [hd, n_per, nh, 2]
        kqv = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), nh * hd, n);
        struct ggml_tensor * attn = ggml_mul_mat(c, L.wo, kqv);

        x = ggml_add(c, res, attn);
        res = x;
        h = ggml_mul(c, ggml_rms_norm(c, x, cfg_.rms_eps), L.ffn_norm);
        if (L.ffn_gate_up) {
            struct ggml_tensor * gu = ggml_mul_mat(c, L.ffn_gate_up, h);
            const int64_t F = L.ffn_gate->ne[1];
            h = ggml_swiglu_split(c, ggml_view_2d(c, gu, F, n, gu->nb[1], 0),
                                     ggml_view_2d(c, gu, F, n, gu->nb[1], (size_t) F * gu->nb[0]));
        } else {
            h = ggml_swiglu_split(c, ggml_mul_mat(c, L.ffn_gate, h), ggml_mul_mat(c, L.ffn_up, h));
        }
        h = ggml_mul_mat(c, L.ffn_down, h);
        x = ggml_add(c, res, h);
    }

    x = ggml_mul(c, ggml_rms_norm(c, x, cfg_.rms_eps), out_norm_);
    struct ggml_tensor * l0 = ggml_view_2d(c, x, cfg_.n_embd, 1, x->nb[1], (size_t) (n_per - 1) * x->nb[1]);
    struct ggml_tensor * l1 = ggml_view_2d(c, x, cfg_.n_embd, 1, x->nb[1], (size_t) (n - 1) * x->nb[1]);
    struct ggml_tensor * last = ggml_concat(c, ggml_cont(c, l0), ggml_cont(c, l1), 1);
    struct ggml_tensor * head = ggml_view_2d(c, heads_, heads_->ne[0], heads_->ne[1],
                                             heads_->nb[1], (size_t) head_idx * heads_->nb[2]);
    struct ggml_tensor * logits = ggml_mul_mat(c, head, last);   // [vocab, 2]
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

bool DepthDecoder::ensure_pair_kv() {
    if (pair_kv_buf_) return true;
    const int nl = cfg_.n_layers;
    struct ggml_init_params p = { ggml_tensor_overhead() * (nl * 2 + 8), nullptr, true };
    pair_kv_ctx_ = ggml_init(p);
    pk_.resize(nl); pv_.resize(nl);
    for (int i = 0; i < nl; ++i) {
        pk_[i] = ggml_new_tensor_4d(pair_kv_ctx_, GGML_TYPE_F32, cfg_.head_dim, cfg_.n_kv_head, kv_slots_, 2);
        ggml_format_name(pk_[i], "dd_pk_%d", i);
        pv_[i] = ggml_new_tensor_4d(pair_kv_ctx_, GGML_TYPE_F32, cfg_.head_dim, cfg_.n_kv_head, kv_slots_, 2);
        ggml_format_name(pv_[i], "dd_pv_%d", i);
    }
    pair_kv_buf_ = ggml_backend_alloc_ctx_tensors(pair_kv_ctx_, w_->backend());
    if (!pair_kv_buf_) { error_msg_ = "depth: pair kv alloc failed"; return false; }
    return true;
}

bool DepthDecoder::step_pair(DepthDecoder & other, int n_per, int n_past, int head_idx,
                             const float * hidden2, const int32_t * codes_in,
                             std::vector<float> & logits2) {
    if (!ensure_pair_kv()) return false;
    if (pair_step_.empty()) pair_step_.resize(cfg_.n_codebooks - 1);
    graph_slot & g = (n_past == 0) ? pair_pre_ : pair_step_[head_idx];
    if (!g.allocated) {
        std::vector<ggml_backend_t> backends = { w_->backend() };
        if (w_->backend_cpu()) backends.push_back(w_->backend_cpu());
        g.sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(),
                                         BZ_DD_MAX_NODES, false, true);
        if (!g.sched) { error_msg_ = "depth: sched_new failed"; return false; }
        g.meta.resize(ggml_tensor_overhead() * BZ_DD_MAX_NODES +
                      ggml_graph_overhead_custom(BZ_DD_MAX_NODES, false));
        struct ggml_init_params p = { g.meta.size(), g.meta.data(), true };
        g.ctx = ggml_init(p);
        g.gf = build_pair_graph(g.ctx, n_per, n_past, head_idx, other);
        if (!ggml_backend_sched_alloc_graph(g.sched, g.gf)) {
            error_msg_ = "depth: alloc pair graph"; return false;
        }
        g.allocated = true;
    }
    struct ggml_cgraph * gf = g.gf;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "codes"), codes_in, 0, 2 * n_per * sizeof(int32_t));
    if (n_past == 0) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "backbone_hidden"), hidden2, 0,
                                (size_t) 2 * cfg_.audio_embd_size * sizeof(float));
    }
    std::vector<int32_t> posv(2 * n_per), kvi(n_per);
    for (int i = 0; i < n_per; ++i) posv[i] = posv[n_per + i] = kvi[i] = n_past + i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), posv.data(), 0, posv.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "kvi"), kvi.data(), 0, kvi.size() * sizeof(int32_t));
    const int n_kv = kv_slots_;
    std::vector<float> m((size_t) n_kv * n_per);
    for (int i = 0; i < n_per; ++i)
        for (int j = 0; j < n_kv; ++j)
            m[(size_t) i * n_kv + j] = (j <= n_past + i) ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask"), m.data(), 0, m.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(g.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "depth: pair compute"; return false;
    }
    struct ggml_tensor * lt = ggml_graph_get_tensor(gf, "logits");
    logits2.resize(ggml_nelements(lt));
    ggml_backend_tensor_get(lt, logits2.data(), 0, ggml_nbytes(lt));
    return true;
}

bool DepthDecoder::run_frame_cfg(DepthDecoder & neg, const float * h_cond, const float * h_uncond,
                                 int32_t * codes, float cfg_scale, bool batched,
                                 float temperature, int top_k, float top_p, uint32_t & rng) {
    const int NC = cfg_.n_codebooks;
    const int V  = cfg_.vocab;
    const int CB = w_->cfg().cc.codebook_size;

    std::vector<float> lc, lu, l2, mixed(V);
    auto run = [&](int n, int n_past, int head_idx, const int32_t * in) -> bool {
        if (batched) {
            std::vector<float> h2;
            if (n_past == 0) {
                const int E = cfg_.audio_embd_size;
                h2.resize((size_t) 2 * E);
                std::memcpy(h2.data(),     h_cond,   (size_t) E * sizeof(float));
                std::memcpy(h2.data() + E, h_uncond, (size_t) E * sizeof(float));
            }
            int32_t in2[4];
            for (int i = 0; i < n; ++i) in2[i] = in2[n + i] = in[i];
            if (!step_pair(neg, n, n_past, head_idx, n_past == 0 ? h2.data() : nullptr, in2, l2))
                return false;
            lc.assign(l2.begin(), l2.begin() + V);
            lu.assign(l2.begin() + V, l2.begin() + 2 * V);
            return true;
        }
        if (!step(n, n_past, head_idx, n_past == 0 ? h_cond : nullptr, in, lc)) return false;
        if (!neg.step(n, n_past, head_idx, n_past == 0 ? h_uncond : nullptr, in, lu)) {
            error_msg_ = neg.error_msg_; return false;
        }
        return true;
    };

    int32_t prefill[2] = { 0, codes[0] };
    if (!run(2, 0, 0, prefill)) return false;
    for (int cb = 1; cb < NC; ++cb) {
        if ((int) lc.size() < V || (int) lu.size() < V) { error_msg_ = "depth: short logits"; return false; }
        for (int i = 0; i < V; ++i) mixed[i] = lu[i] + cfg_scale * (lc[i] - lu[i]);
        for (int i = CB; i < V; ++i) mixed[i] = -INFINITY;
        codes[cb] = sample_from(mixed, V, temperature, top_k, top_p, rng);
        if (cb == NC - 1) break;
        int32_t tok = codes[cb] + cb * V;
        if (!run(1, cb + 1, cb, &tok)) return false;
    }
    return true;
}

bool DepthDecoder::run_frame_pair(DepthDecoder & other, const float * h_a, const float * h_b,
                                  int32_t * codes_a, int32_t * codes_b,
                                  const sample_params & sa, const sample_params & sb,
                                  uint32_t & rng_a, uint32_t & rng_b) {
    const int NC = cfg_.n_codebooks;
    const int V  = cfg_.vocab;
    const int CB = w_->cfg().cc.codebook_size;
    const int E  = cfg_.audio_embd_size;

    std::vector<float> h2((size_t) 2 * E), l2, la(V), lb(V);
    std::memcpy(h2.data(),     h_a, (size_t) E * sizeof(float));
    std::memcpy(h2.data() + E, h_b, (size_t) E * sizeof(float));
    int32_t pre[4] = { 0, codes_a[0], 0, codes_b[0] };
    if (!step_pair(other, 2, 0, 0, h2.data(), pre, l2)) return false;
    for (int cb = 1; cb < NC; ++cb) {
        if ((int) l2.size() < 2 * V) { error_msg_ = "depth: short pair logits"; return false; }
        std::memcpy(la.data(), l2.data(),     (size_t) V * sizeof(float));
        std::memcpy(lb.data(), l2.data() + V, (size_t) V * sizeof(float));
        for (int i = CB; i < V; ++i) la[i] = lb[i] = -INFINITY;
        codes_a[cb] = sample_from(la, V, sa.temperature, sa.top_k, sa.top_p, rng_a);
        codes_b[cb] = sample_from(lb, V, sb.temperature, sb.top_k, sb.top_p, rng_b);
        if (cb == NC - 1) break;
        int32_t tok[2] = { codes_a[cb] + cb * V, codes_b[cb] + cb * V };
        if (!step_pair(other, 1, cb + 1, cb, nullptr, tok, l2)) return false;
    }
    return true;
}

} // namespace breeze
