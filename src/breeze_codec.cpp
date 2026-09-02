#include "breeze_codec.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#define BZ_CODEC_MAX_NODES 16384

namespace breeze {

namespace {

// Prefer the fused direct kernel for F16 weights (tensor cores, no im2col).
// Callers pre-pad with ggml_pad_ext, so p_left = p_right = 0 here.
inline struct ggml_tensor * conv1d(struct ggml_context * c, struct ggml_tensor * w,
                                   struct ggml_tensor * x, int stride, int dilation) {
    return (w->type == GGML_TYPE_F16)
        ? ggml_conv_1d_direct(c, w, x, stride, 0, 0, dilation)
        : ggml_conv_1d(c, w, x, stride, 0, dilation);
}

// `ch` is explicit because Conv1d weights are [k, in, out] while
// ConvTranspose1d weights are [k, out, in] -- reading ne[2] for both silently
// broadcasts the wrong channel count on the transpose path.
inline struct ggml_tensor * add_bias(struct ggml_context * c, struct ggml_tensor * x,
                                     struct ggml_tensor * b, int64_t ch) {
    if (!b) return x;
    return ggml_add(c, x, ggml_reshape_3d(c, b, 1, ch, 1));
}

// Mimi causal Conv1d, stride 1: left pad = (k-1)*dilation, no right pad
// (extra_padding is provably 0 when stride == 1).
inline struct ggml_tensor * causal_conv(struct ggml_context * c, struct ggml_tensor * w,
                                        struct ggml_tensor * x, int dilation) {
    const int k = (int) w->ne[0];
    x = ggml_pad_ext(c, x, (k - 1) * dilation, 0, 0, 0, 0, 0, 0, 0);
    return conv1d(c, w, x, 1, dilation);
}

inline struct ggml_tensor * layer_norm(struct ggml_context * c, struct ggml_tensor * x,
                                       struct ggml_tensor * w, struct ggml_tensor * b,
                                       float eps) {
    struct ggml_tensor * n = ggml_norm(c, x, eps);
    n = ggml_mul(c, n, w);
    return b ? ggml_add(c, n, b) : n;
}

} // namespace

void MimiCodec::free_state() {
    if (sched_)   { ggml_backend_sched_free(sched_); sched_ = nullptr; }
    if (aux_buf_) { ggml_backend_buffer_free(aux_buf_); aux_buf_ = nullptr; }
    if (aux_ctx_) { ggml_free(aux_ctx_); aux_ctx_ = nullptr; }
    dec_tfm_.clear(); enc_tfm_.clear(); dec_res_.clear(); enc_res_.clear();
    dec_up_w_.clear(); dec_up_b_.clear(); enc_ds_w_.clear(); enc_ds_b_.clear();
    up_tap_.clear(); cb_host_.clear(); compute_meta_.clear();
}

bool MimiCodec::init(BreezeWeights * w) {
    free_state();
    w_ = w;
    cfg_ = w->cfg().cc;

    // ── quantizer ────────────────────────────────────────────────────────
    vq_sem_embed_ = w->get("codec.vq_first.0.embed_sum");
    vq_sem_usage_ = w->get("codec.vq_first.0.cluster_usage");
    vq_sem_in_    = w->get("codec.vq_first.input_proj.weight");
    vq_sem_out_   = w->get("codec.vq_first.output_proj.weight");
    vq_ac_in_     = w->get("codec.vq_rest.input_proj.weight");
    vq_ac_out_    = w->get("codec.vq_rest.output_proj.weight");
    const int n_ac = cfg_.n_quantizers - 1;
    vq_ac_embed_.resize(n_ac); vq_ac_usage_.resize(n_ac);
    for (int i = 0; i < n_ac; ++i) {
        vq_ac_embed_[i] = w->getf("codec.vq_rest.%d.embed_sum", i);
        vq_ac_usage_[i] = w->getf("codec.vq_rest.%d.cluster_usage", i);
        if (!vq_ac_embed_[i] || !vq_ac_usage_[i]) {
            error_msg_ = "codec: missing acoustic codebook " + std::to_string(i);
            return false;
        }
    }
    if (!vq_sem_embed_ || !vq_sem_usage_ || !vq_sem_out_ || !vq_ac_out_) {
        error_msg_ = "codec: missing quantizer tensors";
        return false;
    }

    // ── SEANet layer indices ─────────────────────────────────────────────
    // Encoder: 0 conv, then per ratio {resnet, ELU, downsample}, then ELU + conv.
    // Decoder: 0 conv, then per ratio {ELU, convT, resnet}, then ELU + conv.
    // With num_residual_layers = 1 both sides land on the fixed index sets below,
    // which the GGUF's own name table confirms.
    const int R = (int) cfg_.upsample_rates.size();
    dec_in_w_  = w->get("codec.decoder.layers.0.conv.weight");
    dec_in_b_  = w->get("codec.decoder.layers.0.conv.bias");
    dec_up_w_.resize(R); dec_up_b_.resize(R); dec_res_.resize(R);
    for (int i = 0; i < R; ++i) {
        const int li = 2 + 3 * i;                 // 2, 5, 8, 11
        dec_up_w_[i] = w->getf("codec.decoder.layers.%d.conv.weight", li);
        dec_up_b_[i] = w->getf("codec.decoder.layers.%d.conv.bias", li);
        dec_res_[i].c1w = w->getf("codec.decoder.layers.%d.block.1.conv.weight", li + 1);
        dec_res_[i].c1b = w->getf("codec.decoder.layers.%d.block.1.conv.bias", li + 1);
        dec_res_[i].c2w = w->getf("codec.decoder.layers.%d.block.3.conv.weight", li + 1);
        dec_res_[i].c2b = w->getf("codec.decoder.layers.%d.block.3.conv.bias", li + 1);
        dec_res_[i].dilation = 1;                 // dilation_growth_rate ** 0
        if (!dec_up_w_[i] || !dec_res_[i].c1w || !dec_res_[i].c2w) {
            error_msg_ = "codec: missing decoder layer " + std::to_string(li);
            return false;
        }
    }
    dec_out_w_ = w->getf("codec.decoder.layers.%d.conv.weight", 3 * R + 2);
    dec_out_b_ = w->getf("codec.decoder.layers.%d.conv.bias", 3 * R + 2);
    if (!dec_in_w_ || !dec_out_w_) { error_msg_ = "codec: missing decoder head/tail conv"; return false; }

    enc_in_w_ = w->get("codec.encoder.layers.0.conv.weight");
    enc_in_b_ = w->get("codec.encoder.layers.0.conv.bias");
    enc_ds_w_.resize(R); enc_ds_b_.resize(R); enc_res_.resize(R);
    for (int i = 0; i < R; ++i) {
        const int ri = 1 + 3 * i;                 // 1, 4, 7, 10
        enc_res_[i].c1w = w->getf("codec.encoder.layers.%d.block.1.conv.weight", ri);
        enc_res_[i].c1b = w->getf("codec.encoder.layers.%d.block.1.conv.bias", ri);
        enc_res_[i].c2w = w->getf("codec.encoder.layers.%d.block.3.conv.weight", ri);
        enc_res_[i].c2b = w->getf("codec.encoder.layers.%d.block.3.conv.bias", ri);
        enc_res_[i].dilation = 1;
        enc_ds_w_[i] = w->getf("codec.encoder.layers.%d.conv.weight", ri + 2);
        enc_ds_b_[i] = w->getf("codec.encoder.layers.%d.conv.bias", ri + 2);
    }
    enc_out_w_ = w->getf("codec.encoder.layers.%d.conv.weight", 3 * R + 2);
    enc_out_b_ = w->getf("codec.encoder.layers.%d.conv.bias", 3 * R + 2);
    downsample_w_ = w->get("codec.downsample.conv.weight");

    auto load_tfm = [&](const char * pfx, std::vector<tfm_layer> & out) -> bool {
        out.resize(cfg_.n_tfm_layers);
        for (int i = 0; i < cfg_.n_tfm_layers; ++i) {
            auto & L = out[i];
            L.attn_norm_w = w->getf("%s.blk.%d.input_layernorm.weight", pfx, i);
            L.attn_norm_b = w->getf("%s.blk.%d.input_layernorm.bias", pfx, i);
            L.wq   = w->getf("%s.blk.%d.self_attn.q_proj.weight", pfx, i);
            L.wk   = w->getf("%s.blk.%d.self_attn.k_proj.weight", pfx, i);
            L.wv   = w->getf("%s.blk.%d.self_attn.v_proj.weight", pfx, i);
            L.wo   = w->getf("%s.blk.%d.self_attn.o_proj.weight", pfx, i);
            L.attn_scale = w->getf("%s.blk.%d.self_attn_layer_scale.scale", pfx, i);
            L.ffn_norm_w = w->getf("%s.blk.%d.post_attention_layernorm.weight", pfx, i);
            L.ffn_norm_b = w->getf("%s.blk.%d.post_attention_layernorm.bias", pfx, i);
            L.fc1  = w->getf("%s.blk.%d.mlp.fc1.weight", pfx, i);
            L.fc2  = w->getf("%s.blk.%d.mlp.fc2.weight", pfx, i);
            L.ffn_scale = w->getf("%s.blk.%d.mlp_layer_scale.scale", pfx, i);
            if (!L.wq || !L.wk || !L.wv || !L.wo || !L.fc1 || !L.fc2 ||
                !L.attn_norm_w || !L.ffn_norm_w) {
                error_msg_ = std::string("codec: missing ") + pfx + " layer " + std::to_string(i);
                return false;
            }
        }
        return true;
    };
    if (!load_tfm("codec.decoder_tfm", dec_tfm_)) return false;
    if (!load_tfm("codec.encoder_tfm", enc_tfm_)) return false;

    // ── the depthwise upsample, split into per-phase taps ────────────────
    // self.upsample is ConvTranspose1d(512, 512, k=4, s=2, groups=512): a
    // per-channel 4-tap filter. ggml has no grouped transpose conv, but with
    // stride 2 the operation is exactly two 2-tap depthwise FIRs interleaved:
    //   out[2i+p] = x[i]*w[p] + x[i-1]*w[p+2],  p in {0,1}
    // so pull the four taps out once, as [1, C] F32 vectors.
    if (downsample_w_ == nullptr) { error_msg_ = "codec: missing downsample conv"; return false; }
    {
        struct ggml_tensor * uw = w->get("codec.upsample.conv.weight");
        if (!uw) { error_msg_ = "codec: missing upsample conv"; return false; }
        const int K = (int) uw->ne[0];
        const int C = (int) uw->ne[2];
        std::vector<uint8_t> raw(ggml_nbytes(uw));
        ggml_backend_tensor_get(uw, raw.data(), 0, raw.size());
        std::vector<float> f32((size_t) K * C);
        if (uw->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), (int64_t) K * C);
        } else if (uw->type == GGML_TYPE_F32) {
            std::memcpy(f32.data(), raw.data(), raw.size());
        } else {
            error_msg_ = "codec: unexpected upsample weight dtype";
            return false;
        }
        struct ggml_init_params p = { ggml_tensor_overhead() * (K + 4) + (size_t) K * C * 4 + 4096,
                                      nullptr, true };
        aux_ctx_ = ggml_init(p);
        up_tap_.resize(K);
        for (int j = 0; j < K; ++j) {
            up_tap_[j] = ggml_new_tensor_2d(aux_ctx_, GGML_TYPE_F32, 1, C);
            ggml_format_name(up_tap_[j], "up_tap_%d", j);
        }
        aux_buf_ = ggml_backend_alloc_ctx_tensors(aux_ctx_, w->backend());
        if (!aux_buf_) { error_msg_ = "codec: aux alloc failed"; return false; }
        std::vector<float> col(C);
        for (int j = 0; j < K; ++j) {
            for (int c = 0; c < C; ++c) col[c] = f32[(size_t) c * K + j];
            ggml_backend_tensor_set(up_tap_[j], col.data(), 0, col.size() * sizeof(float));
        }
    }

    // ── polyphase repack of the SEANet ConvTranspose1d layers ────────────
    // ggml's conv_transpose_1d kernel measured (nsys) at 49.6% of ALL GPU time
    // in a synth -- ~7 ms per layer for a few hundred MFLOPs, i.e. ~1% of the
    // card. A stride-s transposed conv with kernel 2s is exactly s interleaved
    // kernel-2 regular convs:
    //     out[co, i*s + p] = SUM_ci ( x[ci,i]*w[ci,co,p] + x[ci,i-1]*w[ci,co,p+s] )
    // so repack the weights once into s kernel-2 filters per layer and run them
    // through ggml_conv_1d_direct (the tensor-core path everything else uses),
    // then interleave. Identical arithmetic, ~10x faster, and the trimmed output
    // length L*s falls out for free instead of needing a view.
    {
        size_t poly_elems = 0, poly_tensors = 0;
        for (size_t i = 0; i < dec_up_w_.size(); ++i) {
            const int s = cfg_.upsample_rates[i];
            poly_tensors += s;
            poly_elems += (size_t) 2 * dec_up_w_[i]->ne[1] * dec_up_w_[i]->ne[2] * s;
        }
        struct ggml_init_params pp = {
            ggml_tensor_overhead() * (poly_tensors + up_tap_.size() + 8) + poly_elems * 2 + 8192,
            nullptr, true };
        // aux_ctx_ already holds the upsample taps; extend it for the polyphase set.
        std::vector<std::vector<std::vector<uint16_t>>> host;   // [layer][phase][elems]
        host.resize(dec_up_w_.size());
        for (size_t i = 0; i < dec_up_w_.size(); ++i) {
            struct ggml_tensor * tw = dec_up_w_[i];
            const int K = (int) tw->ne[0], CO = (int) tw->ne[1], CI = (int) tw->ne[2];
            const int s = cfg_.upsample_rates[i];
            if (K != 2 * s) { error_msg_ = "codec: transpose kernel is not 2*stride"; return false; }
            std::vector<uint8_t> raw(ggml_nbytes(tw));
            ggml_backend_tensor_get(tw, raw.data(), 0, raw.size());
            std::vector<float> f((size_t) K * CO * CI);
            if (tw->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f.data(), (int64_t) f.size());
            } else if (tw->type == GGML_TYPE_F32) {
                std::memcpy(f.data(), raw.data(), raw.size());
            } else { error_msg_ = "codec: unexpected transpose weight dtype"; return false; }
            host[i].resize(s);
            for (int ph = 0; ph < s; ++ph) {
                // target ne = [2, CI, CO]; index (t, ci, co)
                host[i][ph].assign((size_t) 2 * CI * CO, 0);
                for (int co = 0; co < CO; ++co)
                    for (int ci = 0; ci < CI; ++ci) {
                        const float w_prev = f[((size_t) ci * CO + co) * K + ph + s];
                        const float w_cur  = f[((size_t) ci * CO + co) * K + ph];
                        const size_t base = ((size_t) co * CI + ci) * 2;
                        host[i][ph][base + 0] = ggml_fp32_to_fp16(w_prev);
                        host[i][ph][base + 1] = ggml_fp32_to_fp16(w_cur);
                    }
            }
        }
        // Rebuild aux_ctx_ with room for both the taps and the polyphase filters.
        std::vector<std::vector<float>> tap_host(up_tap_.size());
        for (size_t j = 0; j < up_tap_.size(); ++j) {
            tap_host[j].resize(ggml_nelements(up_tap_[j]));
            ggml_backend_tensor_get(up_tap_[j], tap_host[j].data(), 0, ggml_nbytes(up_tap_[j]));
        }
        const int64_t tap_c = up_tap_.empty() ? 0 : up_tap_[0]->ne[1];
        if (aux_buf_) { ggml_backend_buffer_free(aux_buf_); aux_buf_ = nullptr; }
        if (aux_ctx_) { ggml_free(aux_ctx_); aux_ctx_ = nullptr; }
        aux_ctx_ = ggml_init(pp);
        for (size_t j = 0; j < up_tap_.size(); ++j) {
            up_tap_[j] = ggml_new_tensor_2d(aux_ctx_, GGML_TYPE_F32, 1, tap_c);
            ggml_format_name(up_tap_[j], "up_tap_%d", (int) j);
        }
        dec_up_poly_.resize(dec_up_w_.size());
        for (size_t i = 0; i < dec_up_w_.size(); ++i) {
            const int s = cfg_.upsample_rates[i];
            dec_up_poly_[i].resize(s);
            for (int ph = 0; ph < s; ++ph) {
                dec_up_poly_[i][ph] = ggml_new_tensor_3d(aux_ctx_, GGML_TYPE_F16, 2,
                                                         dec_up_w_[i]->ne[2], dec_up_w_[i]->ne[1]);
                ggml_format_name(dec_up_poly_[i][ph], "up_poly_%d_%d", (int) i, ph);
            }
        }
        aux_buf_ = ggml_backend_alloc_ctx_tensors(aux_ctx_, w->backend());
        if (!aux_buf_) { error_msg_ = "codec: aux alloc failed"; return false; }
        for (size_t j = 0; j < up_tap_.size(); ++j)
            ggml_backend_tensor_set(up_tap_[j], tap_host[j].data(), 0, tap_host[j].size() * sizeof(float));
        for (size_t i = 0; i < dec_up_poly_.size(); ++i)
            for (size_t ph = 0; ph < dec_up_poly_[i].size(); ++ph)
                ggml_backend_tensor_set(dec_up_poly_[i][ph], host[i][ph].data(), 0,
                                        host[i][ph].size() * sizeof(uint16_t));
    }

    // Both are overridable so the exactness of the span decode stays testable:
    // BREEZE_CODEC_CTX=0 must AUDIBLY break the output. If it does not, the
    // left-context argument is wrong and the spans are exact by luck.
    if (const char * e = std::getenv("BREEZE_CODEC_SPAN")) seanet_span_ = std::max(1, atoi(e));
    if (const char * e = std::getenv("BREEZE_CODEC_CTX"))  seanet_ctx_  = std::max(0, atoi(e));

    std::vector<ggml_backend_t> backends = { w->backend() };
    if (w->backend_cpu()) backends.push_back(w->backend_cpu());
    sched_ = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(),
                                    BZ_CODEC_MAX_NODES, false, true);
    if (!sched_) { error_msg_ = "codec: sched_new failed"; return false; }
    compute_meta_.resize(ggml_tensor_overhead() * BZ_CODEC_MAX_NODES +
                         ggml_graph_overhead_custom(BZ_CODEC_MAX_NODES, false));
    return true;
}

struct ggml_tensor * MimiCodec::tfm_stack(struct ggml_context * c, struct ggml_tensor * x,
                                          const std::vector<tfm_layer> & layers,
                                          int seq, struct ggml_tensor * pos) {
    const int nh = cfg_.n_heads, hd = cfg_.head_dim;
    for (const auto & L : layers) {
        struct ggml_tensor * res = x;
        struct ggml_tensor * h = layer_norm(c, x, L.attn_norm_w, L.attn_norm_b, cfg_.norm_eps);

        struct ggml_tensor * Q = ggml_reshape_3d(c, ggml_mul_mat(c, L.wq, h), hd, nh, seq);
        struct ggml_tensor * K = ggml_reshape_3d(c, ggml_mul_mat(c, L.wk, h), hd, nh, seq);
        struct ggml_tensor * V = ggml_reshape_3d(c, ggml_mul_mat(c, L.wv, h), hd, nh, seq);
        Q = ggml_rope_ext(c, Q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                          cfg_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(c, K, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                          cfg_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        Q = ggml_permute(c, Q, 0, 2, 1, 3);
        K = ggml_permute(c, K, 0, 2, 1, 3);
        V = ggml_permute(c, V, 0, 2, 1, 3);

        struct ggml_tensor * kq = ggml_mul_mat(c, K, Q);
        kq = ggml_scale(c, kq, 1.0f / std::sqrt((float) hd));
        // MimiTransformerModel uses create_causal_mask -- plain causal. The
        // config's sliding_window=250 is NOT applied by transformers' mask
        // factory for this model, so do not window here either.
        kq = ggml_diag_mask_inf_inplace(c, kq, 0);
        kq = ggml_soft_max(c, kq);
        V = ggml_cont(c, ggml_transpose(c, V));
        struct ggml_tensor * kqv = ggml_mul_mat(c, V, kq);
        kqv = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), nh * hd, seq);
        struct ggml_tensor * o = ggml_mul_mat(c, L.wo, kqv);
        if (L.attn_scale) o = ggml_mul(c, o, L.attn_scale);
        x = ggml_add(c, res, o);

        res = x;
        h = layer_norm(c, x, L.ffn_norm_w, L.ffn_norm_b, cfg_.norm_eps);
        h = ggml_gelu(c, ggml_mul_mat(c, L.fc1, h));
        h = ggml_mul_mat(c, L.fc2, h);
        if (L.ffn_scale) h = ggml_mul(c, h, L.ffn_scale);
        x = ggml_add(c, res, h);
    }
    return x;
}

// ELU -> conv(k=3, dilation d) -> ELU -> conv(k=1) -> + residual
struct ggml_tensor * MimiCodec::resnet_block(struct ggml_context * c, struct ggml_tensor * x,
                                             const resnet & r) {
    struct ggml_tensor * res = x;
    x = ggml_elu(c, x);
    x = causal_conv(c, r.c1w, x, r.dilation);
    x = add_bias(c, x, r.c1b, r.c1w->ne[2]);
    x = ggml_elu(c, x);
    x = causal_conv(c, r.c2w, x, 1);
    x = add_bias(c, x, r.c2b, r.c2w->ne[2]);
    return ggml_add(c, res, x);
}

struct ggml_cgraph * MimiCodec::build_latent_graph(struct ggml_context * c, int T) {
    struct ggml_cgraph * gf = ggml_new_graph_custom(c, BZ_CODEC_MAX_NODES, false);
    const int D = cfg_.codebook_dim, H = cfg_.hidden;

    // ── split-RVQ decode. embed = embed_sum / clamp(cluster_usage, 1e-5) is
    // applied per lookup rather than materialised, so no extra weight buffer.
    auto lookup = [&](struct ggml_tensor * embed, struct ggml_tensor * usage,
                      const char * idx_name) {
        struct ggml_tensor * idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
        ggml_set_name(idx, idx_name); ggml_set_input(idx);
        struct ggml_tensor * e = ggml_get_rows(c, embed, idx);              // [D, T]
        struct ggml_tensor * u = ggml_get_rows(c, ggml_reshape_2d(c, usage, 1, usage->ne[0]), idx);
        u = ggml_clamp(c, u, 1e-5f, INFINITY);                              // [1, T]
        return ggml_div(c, e, ggml_repeat(c, u, e));
    };

    struct ggml_tensor * sem = lookup(vq_sem_embed_, vq_sem_usage_, "code_0");
    struct ggml_tensor * ac = nullptr;
    for (int i = 0; i < (int) vq_ac_embed_.size(); ++i) {
        char nm[16]; snprintf(nm, sizeof(nm), "code_%d", i + 1);
        struct ggml_tensor * e = lookup(vq_ac_embed_[i], vq_ac_usage_[i], nm);
        ac = ac ? ggml_add(c, ac, e) : e;
    }
    struct ggml_tensor * q = ggml_mul_mat(c, ggml_reshape_2d(c, vq_sem_out_, D, H), sem);
    if (ac) q = ggml_add(c, q, ggml_mul_mat(c, ggml_reshape_2d(c, vq_ac_out_, D, H), ac));
    // [H, T] -> conv layout [T, H, 1]
    struct ggml_tensor * x = ggml_cont(c, ggml_transpose(c, q));
    x = ggml_reshape_3d(c, x, T, H, 1);

    // ── depthwise x2 upsample, as two interleaved 2-tap FIRs
    {
        struct ggml_tensor * shifted = ggml_cont(c, ggml_view_3d(c,
            ggml_pad_ext(c, x, 1, 0, 0, 0, 0, 0, 0, 0), T, H, 1,
            (size_t) (T + 1) * sizeof(float), (size_t) (T + 1) * H * sizeof(float), 0));
        struct ggml_tensor * y0 = ggml_add(c, ggml_mul(c, x, up_tap_[0]),
                                              ggml_mul(c, shifted, up_tap_[2]));
        struct ggml_tensor * y1 = ggml_add(c, ggml_mul(c, x, up_tap_[1]),
                                              ggml_mul(c, shifted, up_tap_[3]));
        y0 = ggml_reshape_3d(c, y0, 1, T, H);
        y1 = ggml_reshape_3d(c, y1, 1, T, H);
        x = ggml_concat(c, y0, y1, 0);                 // [2, T, H]
        x = ggml_reshape_3d(c, x, 2 * T, H, 1);
    }
    const int S = 2 * T;

    // ── decoder_transformer (channels-first token layout [H, S])
    {
        struct ggml_tensor * t = ggml_cont(c, ggml_transpose(c, ggml_reshape_2d(c, x, S, H)));
        struct ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, S);
        ggml_set_name(pos, "pos"); ggml_set_input(pos);
        t = tfm_stack(c, t, dec_tfm_, S, pos);
        x = ggml_reshape_3d(c, ggml_cont(c, ggml_transpose(c, t)), S, H, 1);
    }

    ggml_set_name(x, "latents");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// The SEANet conv stack over `n_lat` latent frames -> n_lat * 960 samples.
struct ggml_cgraph * MimiCodec::build_seanet_graph(struct ggml_context * c, int n_lat) {
    struct ggml_cgraph * gf = ggml_new_graph_custom(c, BZ_CODEC_MAX_NODES, false);
    const int H = cfg_.hidden;

    struct ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, n_lat, H, 1);
    ggml_set_name(x, "lat"); ggml_set_input(x);

    x = causal_conv(c, dec_in_w_, x, 1);
    x = add_bias(c, x, dec_in_b_, dec_in_w_->ne[2]);
    for (size_t i = 0; i < dec_up_w_.size(); ++i) {
        const int stride = cfg_.upsample_rates[i];
        x = ggml_elu(c, x);
        // Polyphase transposed conv: s kernel-2 causal convs, interleaved.
        // Equivalent to MimiConvTranspose1d with trim_right_ratio 1.0 (the
        // right-hand trim of `stride` samples is what makes the output exactly
        // L*s, which is what the interleave produces directly).
        const int64_t L = x->ne[0];
        const int64_t CO = dec_up_w_[i]->ne[1];
        struct ggml_tensor * xp = ggml_pad_ext(c, x, 1, 0, 0, 0, 0, 0, 0, 0);
        struct ggml_tensor * inter = nullptr;
        for (int ph = 0; ph < stride; ++ph) {
            struct ggml_tensor * y = conv1d(c, dec_up_poly_[i][ph], xp, 1, 1);  // [L, CO, 1]
            y = ggml_reshape_3d(c, y, 1, L, CO);
            inter = inter ? ggml_concat(c, inter, y, 0) : y;
        }
        x = ggml_reshape_3d(c, inter, L * stride, CO, 1);
        x = add_bias(c, x, dec_up_b_[i], CO);
        x = resnet_block(c, x, dec_res_[i]);
    }
    x = ggml_elu(c, x);
    x = causal_conv(c, dec_out_w_, x, 1);
    x = add_bias(c, x, dec_out_b_, dec_out_w_->ne[2]);

    ggml_set_name(x, "pcm");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

bool MimiCodec::decode(const int32_t * codes, int T, std::vector<float> & pcm) {
    pcm.clear();
    if (T <= 0) return true;
    const int nq = cfg_.n_quantizers;
    const int H  = cfg_.hidden;
    const int S  = 2 * T;                       // latent frames, 25 Hz
    const int spf = cfg_.samples_per_frame();   // 1920 samples per codec frame
    const int spl = spf / 2;                    // 960 samples per latent frame

    // ── graph A: codes -> transformer latents [S, H] ─────────────────────
    std::vector<float> lat((size_t) S * H);
    {
        struct ggml_init_params p = { compute_meta_.size(), compute_meta_.data(), true };
        struct ggml_context * c = ggml_init(p);
        struct ggml_cgraph * gf = build_latent_graph(c, T);
        if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
            error_msg_ = "codec: alloc latent graph"; ggml_free(c); return false;
        }
        std::vector<int32_t> col(T);
        for (int q = 0; q < nq; ++q) {
            char nm[16]; snprintf(nm, sizeof(nm), "code_%d", q);
            for (int t = 0; t < T; ++t) col[t] = codes[(size_t) t * nq + q];
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, nm), col.data(), 0, T * sizeof(int32_t));
        }
        std::vector<int32_t> posv(S);
        for (int i = 0; i < S; ++i) posv[i] = i;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), posv.data(), 0,
                                posv.size() * sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "codec: latent compute";
            ggml_backend_sched_reset(sched_); ggml_free(c); return false;
        }
        struct ggml_tensor * o = ggml_graph_get_tensor(gf, "latents");
        ggml_backend_tensor_get(o, lat.data(), 0, ggml_nbytes(o));
        ggml_backend_sched_reset(sched_);
        ggml_free(c);
    }

    // ── graph B: the SEANet stack, in bounded spans ──────────────────────
    // Every intermediate here is (latents x upsampled channels), so a
    // whole-clip decode grows at ~1.5 MiB per codec frame and a 60 s
    // paragraph would want over a gigabyte of scratch that ggml's allocator
    // then never gives back. Spans cap it at a constant.
    pcm.assign((size_t) T * spf, 0.0f);
    const int span_lat = std::max(2, seanet_span_ * 2);
    std::vector<float> in;
    for (int s0 = 0; s0 < S; s0 += span_lat) {
        const int s1   = std::min(S, s0 + span_lat);
        const int warm = std::min(s0, seanet_ctx_);   // real history, not zeros
        const int n_in = (s1 - s0) + warm;

        struct ggml_init_params p = { compute_meta_.size(), compute_meta_.data(), true };
        struct ggml_context * c = ggml_init(p);
        struct ggml_cgraph * gf = build_seanet_graph(c, n_in);
        if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
            error_msg_ = "codec: alloc seanet graph"; ggml_free(c); return false;
        }
        // [S, H] row-major over H -> the span's [n_in, H] block.
        in.resize((size_t) n_in * H);
        for (int ch = 0; ch < H; ++ch)
            std::memcpy(in.data() + (size_t) ch * n_in,
                        lat.data() + (size_t) ch * S + (s0 - warm),
                        (size_t) n_in * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "lat"), in.data(), 0,
                                in.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "codec: seanet compute";
            ggml_backend_sched_reset(sched_); ggml_free(c); return false;
        }
        struct ggml_tensor * o = ggml_graph_get_tensor(gf, "pcm");
        const size_t got = ggml_nelements(o);
        if (got != (size_t) n_in * spl) {
            error_msg_ = "codec: seanet span produced the wrong sample count";
            ggml_backend_sched_reset(sched_); ggml_free(c); return false;
        }
        std::vector<float> out(got);
        ggml_backend_tensor_get(o, out.data(), 0, ggml_nbytes(o));
        std::memcpy(pcm.data() + (size_t) s0 * spl,
                    out.data() + (size_t) warm * spl,
                    (size_t) (s1 - s0) * spl * sizeof(float));
        ggml_backend_sched_reset(sched_);
        ggml_free(c);
    }
    return true;
}

// ─── encode (voice cloning) ──────────────────────────────────────────────

struct ggml_cgraph * MimiCodec::build_encode_graph(struct ggml_context * c, int n_samples) {
    struct ggml_cgraph * gf = ggml_new_graph_custom(c, BZ_CODEC_MAX_NODES, false);
    const int H = cfg_.hidden;

    struct ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, n_samples, 1, 1);
    ggml_set_name(x, "audio"); ggml_set_input(x);

    x = causal_conv(c, enc_in_w_, x, 1);
    x = add_bias(c, x, enc_in_b_, enc_in_w_->ne[2]);

    // Strided causal convs need the right-hand extra_padding that makes the
    // output ceil(L/S) -- the only place Mimi's padding is not just (k-1).
    auto extra_pad = [](int64_t L, int S) { return (int) (((L + S - 1) / S) * S - L); };
    int64_t seq = n_samples;
    for (size_t i = 0; i < enc_ds_w_.size(); ++i) {
        x = resnet_block(c, x, enc_res_[i]);
        x = ggml_elu(c, x);
        const int stride = cfg_.upsample_rates[cfg_.upsample_rates.size() - 1 - i];
        x = ggml_pad_ext(c, x, stride, extra_pad(seq, stride), 0, 0, 0, 0, 0, 0);
        x = conv1d(c, enc_ds_w_[i], x, stride, 1);
        x = add_bias(c, x, enc_ds_b_[i], enc_ds_w_[i]->ne[2]);
        seq = (seq + stride - 1) / stride;
    }
    x = ggml_elu(c, x);
    x = causal_conv(c, enc_out_w_, x, 1);
    x = add_bias(c, x, enc_out_b_, enc_out_w_->ne[2]);

    const int64_t S = x->ne[0];
    {
        struct ggml_tensor * t = ggml_cont(c, ggml_transpose(c, ggml_reshape_2d(c, x, S, H)));
        struct ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, S);
        ggml_set_name(pos, "pos"); ggml_set_input(pos);
        t = tfm_stack(c, t, enc_tfm_, (int) S, pos);
        x = ggml_reshape_3d(c, ggml_cont(c, ggml_transpose(c, t)), S, H, 1);
    }

    // self.downsample: Conv1d k=4 s=2 bias=False, pad_mode="replicate".
    {
        const int right = extra_pad(S, 2);
        struct ggml_tensor * first = ggml_cont(c, ggml_view_3d(c, x, 1, x->ne[1], x->ne[2],
                                                               x->nb[1], x->nb[2], 0));
        struct ggml_tensor * padded = ggml_concat(c, ggml_concat(c, first, first, 0), x, 0);
        if (right > 0) {
            struct ggml_tensor * last = ggml_cont(c, ggml_view_3d(c, x, 1, x->ne[1], x->ne[2],
                                                                  x->nb[1], x->nb[2],
                                                                  (S - 1) * x->nb[0]));
            for (int i = 0; i < right; ++i) padded = ggml_concat(c, padded, last, 0);
        }
        x = conv1d(c, downsample_w_, padded, 2, 1);
    }

    const int64_t F = x->ne[0];
    x = ggml_cont(c, ggml_transpose(c, ggml_reshape_2d(c, x, F, H)));   // [H, F]
    // Both RVQ branches read the SAME embeddings; the acoustic residual loop
    // runs in the 256-d projected space, so project once per branch here.
    struct ggml_tensor * es = ggml_mul_mat(c, ggml_reshape_2d(c, vq_sem_in_, H, cfg_.codebook_dim), x);
    struct ggml_tensor * ea = ggml_mul_mat(c, ggml_reshape_2d(c, vq_ac_in_,  H, cfg_.codebook_dim), x);
    ggml_set_name(es, "proj_sem"); ggml_set_output(es);
    ggml_set_name(ea, "proj_ac");  ggml_set_output(ea);
    ggml_build_forward_expand(gf, es);
    ggml_build_forward_expand(gf, ea);
    return gf;
}

bool MimiCodec::encode(const float * pcm, int n_samples,
                       std::vector<int32_t> & codes, int & n_frames) {
    codes.clear(); n_frames = 0;
    if (n_samples <= 0) return true;
    if (!vq_sem_in_ || !vq_ac_in_ || !enc_in_w_ || !downsample_w_) {
        error_msg_ = "codec: encoder tensors missing (voice cloning unavailable)";
        return false;
    }

    // Normalised codebooks, materialised on the host once -- the nearest-
    // neighbour search is a CPU scan over 16 x 2048 x 256.
    if (cb_host_.empty()) {
        const int nq = cfg_.n_quantizers, CB = cfg_.codebook_size, D = cfg_.codebook_dim;
        cb_host_.resize(nq);
        for (int q = 0; q < nq; ++q) {
            struct ggml_tensor * e = (q == 0) ? vq_sem_embed_ : vq_ac_embed_[q - 1];
            struct ggml_tensor * u = (q == 0) ? vq_sem_usage_ : vq_ac_usage_[q - 1];
            std::vector<float> ev((size_t) CB * D), uv(CB);
            ggml_backend_tensor_get(e, ev.data(), 0, ev.size() * sizeof(float));
            ggml_backend_tensor_get(u, uv.data(), 0, uv.size() * sizeof(float));
            cb_host_[q].resize(ev.size());
            for (int i = 0; i < CB; ++i) {
                const float inv = 1.0f / std::max(uv[i], 1e-5f);
                for (int d = 0; d < D; ++d) cb_host_[q][(size_t) i * D + d] = ev[(size_t) i * D + d] * inv;
            }
        }
    }

    struct ggml_init_params p = { compute_meta_.size(), compute_meta_.data(), true };
    struct ggml_context * c = ggml_init(p);
    struct ggml_cgraph * gf = build_encode_graph(c, n_samples);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        error_msg_ = "codec: alloc encode graph"; ggml_free(c); return false;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "audio"), pcm, 0,
                            (size_t) n_samples * sizeof(float));
    struct ggml_tensor * pt = ggml_graph_get_tensor(gf, "pos");
    {
        std::vector<int32_t> posv(pt->ne[0]);
        for (int64_t i = 0; i < pt->ne[0]; ++i) posv[i] = (int32_t) i;
        ggml_backend_tensor_set(pt, posv.data(), 0, posv.size() * sizeof(int32_t));
    }
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "codec: encode compute";
        ggml_backend_sched_reset(sched_); ggml_free(c); return false;
    }
    struct ggml_tensor * ts = ggml_graph_get_tensor(gf, "proj_sem");
    struct ggml_tensor * ta = ggml_graph_get_tensor(gf, "proj_ac");
    const int D = cfg_.codebook_dim;
    const int F = (int) ts->ne[1];
    std::vector<float> ps((size_t) D * F), pa((size_t) D * F);
    ggml_backend_tensor_get(ts, ps.data(), 0, ps.size() * sizeof(float));
    ggml_backend_tensor_get(ta, pa.data(), 0, pa.size() * sizeof(float));
    ggml_backend_sched_reset(sched_);
    ggml_free(c);

    const int nq = cfg_.n_quantizers, CB = cfg_.codebook_size;
    auto nearest = [&](const float * v, int q) {
        const float * cb = cb_host_[q].data();
        int best = 0; float bd = 1e30f;
        for (int i = 0; i < CB; ++i) {
            const float * e = cb + (size_t) i * D;
            float d = 0.0f;
            for (int k = 0; k < D; ++k) { const float t = v[k] - e[k]; d += t * t; }
            if (d < bd) { bd = d; best = i; }
        }
        return best;
    };

    n_frames = F;
    codes.assign((size_t) F * nq, 0);
    std::vector<float> resid(D);
    for (int t = 0; t < F; ++t) {
        codes[(size_t) t * nq + 0] = nearest(ps.data() + (size_t) t * D, 0);
        std::memcpy(resid.data(), pa.data() + (size_t) t * D, D * sizeof(float));
        for (int q = 1; q < nq; ++q) {
            const int idx = nearest(resid.data(), q);
            codes[(size_t) t * nq + q] = idx;
            const float * e = cb_host_[q].data() + (size_t) idx * D;
            for (int k = 0; k < D; ++k) resid[k] -= e[k];
        }
    }
    return true;
}

} // namespace breeze
