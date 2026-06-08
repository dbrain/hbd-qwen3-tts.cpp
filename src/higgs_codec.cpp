#include "higgs_codec.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#define HIGGS_CODEC_MAX_NODES 8192

namespace higgs {

using qwen3_tts::GGUFLoader;
using qwen3_tts::load_tensor_data_from_file;
using qwen3_tts::init_preferred_backend;
using qwen3_tts::release_preferred_backend;

HiggsCodecDecoder::~HiggsCodecDecoder() { unload_model(); }

void HiggsCodecDecoder::unload_model() {
    if (sched_)       { ggml_backend_sched_free(sched_); sched_ = nullptr; }
    if (model_.buffer)       { ggml_backend_buffer_free(model_.buffer); model_.buffer = nullptr; }
    if (model_.snake_buffer) { ggml_backend_buffer_free(model_.snake_buffer); model_.snake_buffer = nullptr; }
    if (model_.ctx)       { ggml_free(model_.ctx); model_.ctx = nullptr; }
    if (model_.snake_ctx) { ggml_free(model_.snake_ctx); model_.snake_ctx = nullptr; }
    if (backend_)     { release_preferred_backend(backend_); backend_ = nullptr; }
    if (backend_cpu_) { ggml_backend_free(backend_cpu_); backend_cpu_ = nullptr; }
    model_.tensors.clear();
    model_.vq_embed.clear();
    model_.vq_proj_out_w.clear();
    model_.vq_proj_out_b.clear();
    model_.blocks.clear();
    compute_meta_.clear();
}

// Read an integer array metadata key (decoder_strides) from the GGUF.
static std::vector<int> read_int_array(gguf_context * gc, const char * key) {
    std::vector<int> out;
    int id = gguf_find_key(gc, key);
    if (id < 0 || gguf_get_kv_type(gc, id) != GGUF_TYPE_ARRAY) return out;
    const enum gguf_type elt = gguf_get_arr_type(gc, id);
    const int64_t n = gguf_get_arr_n(gc, id);
    const void * data = gguf_get_arr_data(gc, id);
    for (int64_t i = 0; i < n; ++i) {
        int32_t v = 0;
        switch (elt) {
            case GGUF_TYPE_UINT8:  v = ((const uint8_t  *)data)[i]; break;
            case GGUF_TYPE_INT8:   v = ((const int8_t   *)data)[i]; break;
            case GGUF_TYPE_UINT16: v = ((const uint16_t *)data)[i]; break;
            case GGUF_TYPE_INT16:  v = ((const int16_t  *)data)[i]; break;
            case GGUF_TYPE_UINT32: v = (int32_t)((const uint32_t *)data)[i]; break;
            case GGUF_TYPE_INT32:  v = ((const int32_t  *)data)[i]; break;
            case GGUF_TYPE_UINT64: v = (int32_t)((const uint64_t *)data)[i]; break;
            case GGUF_TYPE_INT64:  v = (int32_t)((const int64_t  *)data)[i]; break;
            default: break;
        }
        out.push_back(v);
    }
    return out;
}

bool HiggsCodecDecoder::load_model(const std::string & gguf_path) {
    unload_model();

    GGUFLoader loader;
    if (!loader.open(gguf_path)) { error_msg_ = loader.get_error(); return false; }
    gguf_context * gc = loader.get_ctx();
    ggml_context * meta = loader.get_meta_ctx();

    auto & cfg = model_.config;
    cfg.n_codebooks   = loader.get_u32("higgs-codec.n_codebooks",   8);
    cfg.codebook_size = loader.get_u32("higgs-codec.codebook_size", 1024);
    cfg.codebook_dim  = loader.get_u32("higgs-codec.codebook_dim",  64);
    cfg.quantizer_dim = loader.get_u32("higgs-codec.quantizer_dim", 1024);
    cfg.acoustic_dim  = loader.get_u32("higgs-codec.acoustic_dim",  256);
    cfg.decoder_chan  = loader.get_u32("higgs-codec.decoder_channels", 1024);
    cfg.sample_rate   = loader.get_u32("higgs-codec.sample_rate",   24000);
    cfg.n_blocks      = loader.get_u32("higgs-codec.n_decoder_blocks", 5);
    cfg.strides       = read_int_array(gc, "higgs-codec.decoder_strides");
    if ((int)cfg.strides.size() != cfg.n_blocks) {
        // fall back to the known higgs-v3 cascade
        cfg.strides = {8, 5, 4, 2, 3};
        cfg.n_blocks = (int)cfg.strides.size();
    }

    // Create the weight metadata context covering every tensor in the file.
    const int64_t n_tensors = loader.get_n_tensors();
    size_t ctx_size = ggml_tensor_overhead() * (n_tensors + 8);
    struct ggml_init_params p = { ctx_size, nullptr, true };
    model_.ctx = ggml_init(p);
    if (!model_.ctx) { error_msg_ = "ggml_init (weights) failed"; return false; }

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        struct ggml_tensor * mt = ggml_get_tensor(meta, name);
        if (!mt) continue;
        struct ggml_tensor * t = ggml_dup_tensor(model_.ctx, mt);
        ggml_set_name(t, name);
        model_.tensors[name] = t;
    }

    // Backend (prefer CUDA, CPU fallback). Weights live on the compute backend.
    backend_ = init_preferred_backend("HiggsCodec", &error_msg_, false, nullptr);
    if (!backend_) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
    const bool on_cpu = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (!on_cpu) {
        backend_cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu_) { error_msg_ = "CPU fallback backend init failed"; return false; }
    }
    fprintf(stderr, "  HiggsCodec backend: %s\n", dev ? ggml_backend_dev_name(dev) : "?");

    // Load tensor data onto the compute backend.
    enum ggml_backend_dev_type want = on_cpu ? GGML_BACKEND_DEVICE_TYPE_CPU
                                             : ggml_backend_dev_type(dev);
    if (!load_tensor_data_from_file(gguf_path, gc, model_.ctx, model_.tensors,
                                    model_.buffer, error_msg_, want)) {
        return false;
    }

    // Resolve named tensors.
    auto get = [&](const std::string & n) -> struct ggml_tensor * {
        auto it = model_.tensors.find(n);
        return it == model_.tensors.end() ? nullptr : it->second;
    };
    auto need = [&](const std::string & n) -> struct ggml_tensor * {
        struct ggml_tensor * t = get(n);
        if (!t) { error_msg_ = "missing tensor: " + n; }
        return t;
    };

    model_.audio_embd = get("audio_embd.weight");  // optional here

    model_.vq_embed.resize(cfg.n_codebooks);
    model_.vq_proj_out_w.resize(cfg.n_codebooks);
    model_.vq_proj_out_b.resize(cfg.n_codebooks);
    for (int i = 0; i < cfg.n_codebooks; ++i) {
        char nm[64];
        snprintf(nm, sizeof(nm), "vq.%d.embed", i);            model_.vq_embed[i] = need(nm);
        snprintf(nm, sizeof(nm), "vq.%d.project_out.weight", i); model_.vq_proj_out_w[i] = need(nm);
        snprintf(nm, sizeof(nm), "vq.%d.project_out.bias", i);   model_.vq_proj_out_b[i] = need(nm);
    }
    model_.fc2_w = need("fc2.weight");
    model_.fc2_b = need("fc2.bias");
    model_.conv1_w = need("dec.conv1.weight");
    model_.conv1_b = need("dec.conv1.bias");
    model_.conv2_w = need("dec.conv2.weight");
    model_.conv2_b = need("dec.conv2.bias");
    struct ggml_tensor * snake_final_raw = need("dec.snake_final.alpha");

    model_.blocks.resize(cfg.n_blocks);
    std::vector<struct ggml_tensor *> snake_raw; // raw α tensors to host-precompute
    for (int b = 0; b < cfg.n_blocks; ++b) {
        auto & blk = model_.blocks[b];
        blk.stride = cfg.strides[b];
        char nm[96];
        snprintf(nm, sizeof(nm), "dec.block.%d.snake_in.alpha", b); blk.snake_in_alpha = need(nm);
        snprintf(nm, sizeof(nm), "dec.block.%d.conv_t.weight", b);  blk.conv_t_w = need(nm);
        snprintf(nm, sizeof(nm), "dec.block.%d.conv_t.bias", b);    blk.conv_t_b = need(nm);
        snake_raw.push_back(blk.snake_in_alpha);
        for (int r = 1; r <= 3; ++r) {
            auto & ru = blk.res[r-1];
            ru.dilation = (r == 1) ? 1 : (r == 2) ? 3 : 9;
            snprintf(nm, sizeof(nm), "dec.block.%d.res%d.conv1.weight", b, r); ru.conv1_w = need(nm);
            snprintf(nm, sizeof(nm), "dec.block.%d.res%d.conv1.bias", b, r);   ru.conv1_b = need(nm);
            snprintf(nm, sizeof(nm), "dec.block.%d.res%d.conv2.weight", b, r); ru.conv2_w = need(nm);
            snprintf(nm, sizeof(nm), "dec.block.%d.res%d.conv2.bias", b, r);   ru.conv2_b = need(nm);
            snprintf(nm, sizeof(nm), "dec.block.%d.res%d.snake1.alpha", b, r); ru.snake1_alpha = need(nm);
            snprintf(nm, sizeof(nm), "dec.block.%d.res%d.snake2.alpha", b, r); ru.snake2_alpha = need(nm);
            snake_raw.push_back(ru.snake1_alpha);
            snake_raw.push_back(ru.snake2_alpha);
        }
    }
    snake_raw.push_back(snake_final_raw);
    if (!error_msg_.empty()) return false;

    // ---- host-precompute snake α (F32) and inv = 1/(α+1e-9) (F32) ----
    // The GGUF stores raw α as F16 [1,C,1]. Manual snake needs a per-channel
    // F32 α and its reciprocal; both broadcast over the time axis.
    {
        size_t n_snake = snake_raw.size();
        struct ggml_init_params sp = { ggml_tensor_overhead() * (n_snake * 2 + 4), nullptr, true };
        model_.snake_ctx = ggml_init(sp);
        if (!model_.snake_ctx) { error_msg_ = "ggml_init (snake) failed"; return false; }

        // Build alpha_f32 / inv_f32 tensors mirroring each raw α's shape.
        std::vector<struct ggml_tensor *> a_t(n_snake), inv_t(n_snake);
        for (size_t i = 0; i < n_snake; ++i) {
            struct ggml_tensor * raw = snake_raw[i];
            a_t[i]   = ggml_new_tensor(model_.snake_ctx, GGML_TYPE_F32, ggml_n_dims(raw), raw->ne);
            inv_t[i] = ggml_new_tensor(model_.snake_ctx, GGML_TYPE_F32, ggml_n_dims(raw), raw->ne);
        }
        model_.snake_buffer = ggml_backend_alloc_ctx_tensors(model_.snake_ctx, backend_);
        if (!model_.snake_buffer) { error_msg_ = "snake buffer alloc failed"; return false; }

        for (size_t i = 0; i < n_snake; ++i) {
            struct ggml_tensor * raw = snake_raw[i];
            const int64_t ne = ggml_nelements(raw);
            std::vector<float> a(ne), inv(ne);
            // raw is F16 on device; bounce to host and convert.
            if (raw->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> h(ne);
                ggml_backend_tensor_get(raw, h.data(), 0, ne * sizeof(ggml_fp16_t));
                for (int64_t k = 0; k < ne; ++k) a[k] = ggml_fp16_to_fp32(h[k]);
            } else {
                ggml_backend_tensor_get(raw, a.data(), 0, ne * sizeof(float));
            }
            for (int64_t k = 0; k < ne; ++k) inv[k] = 1.0f / (a[k] + 1e-9f);
            ggml_backend_tensor_set(a_t[i],   a.data(),   0, ne * sizeof(float));
            ggml_backend_tensor_set(inv_t[i], inv.data(), 0, ne * sizeof(float));
        }
        // Re-point block structs at the F32 copies.
        size_t idx = 0;
        for (int b = 0; b < cfg.n_blocks; ++b) {
            auto & blk = model_.blocks[b];
            blk.snake_in_alpha = a_t[idx]; blk.snake_in_inv = inv_t[idx]; ++idx;
            for (int r = 0; r < 3; ++r) {
                blk.res[r].snake1_alpha = a_t[idx]; blk.res[r].snake1_inv = inv_t[idx]; ++idx;
                blk.res[r].snake2_alpha = a_t[idx]; blk.res[r].snake2_inv = inv_t[idx]; ++idx;
            }
        }
        model_.snake_final_alpha = a_t[idx]; model_.snake_final_inv = inv_t[idx]; ++idx;
    }

    // Scheduler.
    std::vector<ggml_backend_t> backends = { backend_ };
    if (backend_cpu_) backends.push_back(backend_cpu_);
    sched_ = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(),
                                    HIGGS_CODEC_MAX_NODES, false, true);
    if (!sched_) { error_msg_ = "sched_new failed"; return false; }
    compute_meta_.resize(ggml_tensor_overhead() * HIGGS_CODEC_MAX_NODES +
                         ggml_graph_overhead_custom(HIGGS_CODEC_MAX_NODES, false));

    fprintf(stderr, "  HiggsCodec loaded: %d codebooks, strides {", cfg.n_codebooks);
    for (size_t i = 0; i < cfg.strides.size(); ++i)
        fprintf(stderr, "%d%s", cfg.strides[i], i + 1 < cfg.strides.size() ? "," : "");
    fprintf(stderr, "}, %d fps, %d Hz\n",
            cfg.sample_rate / 960, cfg.sample_rate);
    return true;
}

// y = x + sin²(α·x) / (α+1e-9), α per-channel along ne1, broadcast over ne0/ne2.
struct ggml_tensor * HiggsCodecDecoder::apply_snake(struct ggml_context * ctx,
                                                    struct ggml_tensor * x,
                                                    struct ggml_tensor * alpha,
                                                    struct ggml_tensor * inv) {
    struct ggml_tensor * ax = ggml_mul(ctx, x, alpha);
    struct ggml_tensor * s  = ggml_sin(ctx, ax);
    struct ggml_tensor * s2 = ggml_sqr(ctx, s);
    struct ggml_tensor * t  = ggml_mul(ctx, s2, inv);
    return ggml_add(ctx, x, t);
}

// DAC ResidualUnit: snake1 -> conv1(k7,dil) -> snake2 -> conv2(k1); residual add.
// Length is preserved (symmetric padding), so the torch center-crop is a no-op.
struct ggml_tensor * HiggsCodecDecoder::apply_res_unit(struct ggml_context * ctx,
                                                       struct ggml_tensor * x,
                                                       const dac_res_unit & ru) {
    struct ggml_tensor * residual = x;
    const int64_t C = x->ne[1];
    struct ggml_tensor * y = apply_snake(ctx, x, ru.snake1_alpha, ru.snake1_inv);
    const int k1 = (int)ru.conv1_w->ne[0];               // 7
    const int pad1 = ((k1 - 1) * ru.dilation) / 2;       // 3*dil
    y = ggml_conv_1d_direct(ctx, ru.conv1_w, y, 1, pad1, pad1, ru.dilation);
    y = ggml_add(ctx, y, ggml_reshape_3d(ctx, ru.conv1_b, 1, ru.conv1_b->ne[0], 1));
    y = apply_snake(ctx, y, ru.snake2_alpha, ru.snake2_inv);
    y = ggml_conv_1d_direct(ctx, ru.conv2_w, y, 1, 0, 0, 1);   // k1, no pad
    y = ggml_add(ctx, y, ggml_reshape_3d(ctx, ru.conv2_b, 1, ru.conv2_b->ne[0], 1));
    (void)C;
    return ggml_add(ctx, residual, y);
}

struct ggml_cgraph * HiggsCodecDecoder::build_graph(struct ggml_context * ctx0, int32_t n_frames,
                                                    struct ggml_tensor ** out_rvq,
                                                    struct ggml_tensor ** out_fc2) {
    const auto & cfg = model_.config;
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, HIGGS_CODEC_MAX_NODES, false);

    // ---- RVQ decode: Σ_i project_out_i(embed_i[codes_i]) ----
    struct ggml_tensor * quantized = nullptr;  // [quantizer_dim, T]
    for (int i = 0; i < cfg.n_codebooks; ++i) {
        char nm[32];
        snprintf(nm, sizeof(nm), "codes_cb%d", i);
        struct ggml_tensor * codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
        ggml_set_name(codes, nm);
        ggml_set_input(codes);

        struct ggml_tensor * emb = ggml_get_rows(ctx0, model_.vq_embed[i], codes); // [64,T]
        struct ggml_tensor * q   = ggml_mul_mat(ctx0, model_.vq_proj_out_w[i], emb); // [1024,T]
        q = ggml_add(ctx0, q, model_.vq_proj_out_b[i]);
        quantized = quantized ? ggml_add(ctx0, quantized, q) : q;
    }
    ggml_set_name(quantized, "post_rvq");
    if (out_rvq) *out_rvq = quantized;

    // ---- fc2: 1024 -> 256 (channel-mixing matmul, contracts ne0) ----
    struct ggml_tensor * acoustic = ggml_mul_mat(ctx0, model_.fc2_w, quantized); // [256,T]
    acoustic = ggml_add(ctx0, acoustic, model_.fc2_b);
    ggml_set_name(acoustic, "post_fc2");
    if (out_fc2) *out_fc2 = acoustic;

    // ---- to time-major [T, 256] for conv1 ----
    struct ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, acoustic)); // [T,256]
    x = ggml_reshape_3d(ctx0, x, n_frames, cfg.acoustic_dim, 1);

    // conv1: 256 -> 1024, k7 pad3
    x = ggml_conv_1d_direct(ctx0, model_.conv1_w, x, 1, 3, 3, 1);
    x = ggml_add(ctx0, x, ggml_reshape_3d(ctx0, model_.conv1_b, 1, cfg.decoder_chan, 1));

    // ---- 5 DecoderBlocks ----
    for (int b = 0; b < cfg.n_blocks; ++b) {
        const auto & blk = model_.blocks[b];
        const int s = blk.stride;

        x = apply_snake(ctx0, x, blk.snake_in_alpha, blk.snake_in_inv);

        const int64_t Lin   = x->ne[0];
        const int64_t Cin   = x->ne[1];
        const int64_t Cout  = blk.conv_t_w->ne[1];
        const int     k     = (int)blk.conv_t_w->ne[0];   // 2*s
        struct ggml_tensor * x2 = ggml_reshape_2d(ctx0, x, Lin, Cin);
        x2 = ggml_conv_transpose_1d(ctx0, blk.conv_t_w, x2, s, 0, 1);  // [Lfull, Cout]
        const int64_t Lfull = x2->ne[0];                  // (Lin-1)*s + k

        // torch ConvTranspose1d(padding=ceil(s/2), output_padding=s%2):
        //   keep [left_trim : Lfull - right_trim]
        const int left_trim  = (s + 1) / 2;               // ceil(s/2)
        const int right_trim = (s + 1) / 2 - (s % 2);
        const int64_t out_len = Lfull - left_trim - right_trim;
        struct ggml_tensor * xv = ggml_view_2d(ctx0, x2, out_len, Cout,
                                               x2->nb[1], (size_t)left_trim * x2->nb[0]);
        x = ggml_cont(ctx0, xv);
        x = ggml_reshape_3d(ctx0, x, out_len, Cout, 1);
        x = ggml_add(ctx0, x, ggml_reshape_3d(ctx0, blk.conv_t_b, 1, Cout, 1));
        (void)k;

        for (int r = 0; r < 3; ++r) x = apply_res_unit(ctx0, x, blk.res[r]);
    }

    // ---- final snake -> conv2 -> 1ch (NO tanh) ----
    x = apply_snake(ctx0, x, model_.snake_final_alpha, model_.snake_final_inv);
    x = ggml_conv_1d_direct(ctx0, model_.conv2_w, x, 1, 3, 3, 1);   // -> [L,1,1]
    x = ggml_add(ctx0, x, ggml_reshape_3d(ctx0, model_.conv2_b, 1, 1, 1));
    x = ggml_reshape_1d(ctx0, x, x->ne[0]);
    ggml_set_name(x, "audio");
    ggml_set_output(x);

    ggml_build_forward_expand(gf, x);
    // Debug intermediates: cont() into fresh leaf tensors so the scheduler
    // can't alias their storage to later scratch (mid-graph tensors marked
    // output are still eligible for reuse otherwise).
    if (out_rvq && *out_rvq) {
        *out_rvq = ggml_cont(ctx0, *out_rvq); ggml_set_name(*out_rvq, "post_rvq_out");
        ggml_set_output(*out_rvq); ggml_build_forward_expand(gf, *out_rvq);
    }
    if (out_fc2 && *out_fc2) {
        *out_fc2 = ggml_cont(ctx0, *out_fc2); ggml_set_name(*out_fc2, "post_fc2_out");
        ggml_set_output(*out_fc2); ggml_build_forward_expand(gf, *out_fc2);
    }
    return gf;
}

bool HiggsCodecDecoder::decode(const int32_t * codes, int32_t n_frames, std::vector<float> & samples) {
    std::vector<float> a, b;
    return decode_debug(codes, n_frames, samples, a, b);
}

bool HiggsCodecDecoder::decode_debug(const int32_t * codes, int32_t n_frames,
                                     std::vector<float> & samples,
                                     std::vector<float> & post_rvq,
                                     std::vector<float> & post_fc2) {
    if (!model_.ctx) { error_msg_ = "model not loaded"; return false; }
    const auto & cfg = model_.config;

    struct ggml_init_params p = { compute_meta_.size(), compute_meta_.data(), true };
    struct ggml_context * ctx0 = ggml_init(p);
    struct ggml_tensor * rvq_t = nullptr, * fc2_t = nullptr;
    struct ggml_cgraph * gf = build_graph(ctx0, n_frames, &rvq_t, &fc2_t);

    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        error_msg_ = "sched_alloc_graph failed"; ggml_free(ctx0); return false;
    }

    // set codes inputs (one I32 tensor per codebook; codes row-major [T, n_cb])
    std::vector<int32_t> col(n_frames);
    for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
        char nm[32]; snprintf(nm, sizeof(nm), "codes_cb%d", cb);
        struct ggml_tensor * t = ggml_graph_get_tensor(gf, nm);
        for (int f = 0; f < n_frames; ++f) col[f] = codes[f * cfg.n_codebooks + cb];
        ggml_backend_tensor_set(t, col.data(), 0, n_frames * sizeof(int32_t));
    }

    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "graph_compute failed"; ggml_backend_sched_reset(sched_); ggml_free(ctx0); return false;
    }

    struct ggml_tensor * audio = ggml_graph_get_tensor(gf, "audio");
    samples.resize(audio->ne[0]);
    ggml_backend_tensor_get(audio, samples.data(), 0, audio->ne[0] * sizeof(float));

    if (rvq_t) {
        post_rvq.resize(ggml_nelements(rvq_t));
        ggml_backend_tensor_get(rvq_t, post_rvq.data(), 0, ggml_nbytes(rvq_t));
    }
    if (fc2_t) {
        post_fc2.resize(ggml_nelements(fc2_t));
        ggml_backend_tensor_get(fc2_t, post_fc2.data(), 0, ggml_nbytes(fc2_t));
    }

    ggml_backend_sched_reset(sched_);
    ggml_free(ctx0);
    return true;
}

} // namespace higgs
