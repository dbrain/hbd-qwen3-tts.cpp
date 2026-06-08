#include "higgs_encode.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define HIGGS_ENC_MAX_NODES 16384

namespace higgs {

using qwen3_tts::GGUFLoader;
using qwen3_tts::load_tensor_data_from_file;
using qwen3_tts::init_preferred_backend;
using qwen3_tts::release_preferred_backend;

HiggsCodecEncoder::~HiggsCodecEncoder() { unload_model(); }

void HiggsCodecEncoder::unload_model() {
    if (sched_)              { ggml_backend_sched_free(sched_); sched_ = nullptr; }
    if (model_.buffer)       { ggml_backend_buffer_free(model_.buffer); model_.buffer = nullptr; }
    if (model_.snake_buffer) { ggml_backend_buffer_free(model_.snake_buffer); model_.snake_buffer = nullptr; }
    if (model_.ctx)          { ggml_free(model_.ctx); model_.ctx = nullptr; }
    if (model_.snake_ctx)    { ggml_free(model_.snake_ctx); model_.snake_ctx = nullptr; }
    if (backend_)            { release_preferred_backend(backend_); backend_ = nullptr; }
    if (backend_cpu_)        { ggml_backend_free(backend_cpu_); backend_cpu_ = nullptr; }
    model_.tensors.clear();
    model_.layers.clear();
    model_.sem_blocks.clear();
    model_.aenc_blocks.clear();
    model_.vq_proj_in_w.clear();
    model_.vq_proj_in_b.clear();
    model_.vq_embed.clear();
    model_.vq_proj_out_w.clear();
    model_.vq_proj_out_b.clear();
    compute_meta_.clear();
}

// ---------------------------------------------------------------------------
// Kaiser-windowed sinc resampler matching torchaudio.functional.resample
// defaults (lowpass_filter_width=6, rolloff=0.99, beta=14.7696566). Mirrors
// ref_encode.py::_resample exactly (polyphase, gcd-reduced ratio).
// ---------------------------------------------------------------------------
static double bessel_i0(double x) {
    // series expansion of the modified Bessel function I0
    double sum = 1.0, term = 1.0, y = x * x / 4.0;
    for (int k = 1; k < 64; ++k) {
        term *= y / (double)(k * k);
        sum += term;
        if (term < 1e-18 * sum) break;
    }
    return sum;
}

std::vector<float> HiggsCodecEncoder::resample(const std::vector<float> & in, int orig_sr, int new_sr) {
    if (orig_sr == new_sr || in.empty()) return in;
    const double lowpass_filter_width = 6.0;
    const double rolloff = 0.99;
    const double beta = 14.769656459379492;

    auto gcd = [](int a, int b){ while (b) { int t = a % b; a = b; b = t; } return a; };
    int g = gcd(orig_sr, new_sr);
    int of = orig_sr / g;   // orig_freq (reduced)
    int nf = new_sr  / g;   // new_freq  (reduced)

    double base_freq = std::min(of, nf) * rolloff;
    int width = (int)std::ceil(lowpass_filter_width * of / base_freq);
    double i0_beta = bessel_i0(beta);

    // kernel[i][j] : i in [0,nf), j in [0, 2*width+of)
    int klen = 2 * width + of;
    std::vector<double> kernel((size_t)nf * klen);
    double scale = base_freq / (double)of;
    for (int i = 0; i < nf; ++i) {
        for (int j = 0; j < klen; ++j) {
            // idx = (-width + j) / of ;  t = (-i)/nf + idx ;  t *= base_freq
            double idx = (double)(-width + j) / (double)of;
            double t = (double)(-i) / (double)nf + idx;
            t *= base_freq;
            if (t < -lowpass_filter_width) t = -lowpass_filter_width;
            if (t >  lowpass_filter_width) t =  lowpass_filter_width;
            double winarg = 1.0 - (t / lowpass_filter_width) * (t / lowpass_filter_width);
            if (winarg < 0) winarg = 0;
            double win = bessel_i0(beta * std::sqrt(winarg)) / i0_beta;
            double tp = t * M_PI;
            double sinc = (tp == 0.0) ? 1.0 : std::sin(tp) / tp;
            kernel[(size_t)i * klen + j] = sinc * win * scale;
        }
    }

    int length = (int)in.size();
    // pad (width, width + of) then conv with stride = of
    int plen = width + length + (width + of);
    std::vector<double> pad((size_t)plen, 0.0);
    for (int i = 0; i < length; ++i) pad[(size_t)width + i] = in[i];

    // conv1d stride=of: out frames = floor((plen - klen)/of) + 1, each producing nf samples
    int frames = (plen - klen) / of + 1;
    if (frames < 0) frames = 0;
    int target_length = (int)std::ceil((double)nf * (double)length / (double)of);
    std::vector<float> out;
    out.reserve((size_t)frames * nf);
    for (int fr = 0; fr < frames; ++fr) {
        int start = fr * of;
        for (int i = 0; i < nf; ++i) {
            double acc = 0.0;
            const double * kr = &kernel[(size_t)i * klen];
            for (int j = 0; j < klen; ++j) acc += pad[(size_t)start + j] * kr[j];
            out.push_back((float)acc);
        }
    }
    if ((int)out.size() > target_length) out.resize(target_length);
    return out;
}

bool HiggsCodecEncoder::load_model(const std::string & gguf_path) {
    unload_model();

    GGUFLoader loader;
    if (!loader.open(gguf_path)) { error_msg_ = loader.get_error(); return false; }
    gguf_context * gc = loader.get_ctx();
    ggml_context * meta = loader.get_meta_ctx();

    auto & cfg = model_.config;
    cfg.sample_rate          = loader.get_u32("higgs-codec.sample_rate", 24000);
    cfg.semantic_sample_rate = loader.get_u32("higgs-codec.semantic_sample_rate", 16000);
    cfg.hop_length           = loader.get_u32("higgs-codec.hop_length", 960);
    cfg.downsample_factor    = loader.get_u32("higgs-codec.downsample_factor", 320);
    cfg.semantic_downsample_factor = loader.get_u32("higgs-codec.semantic_downsample_factor", 2);
    cfg.model_pad            = loader.get_u32("higgs-codec.model_pad", 480);
    cfg.n_codebooks          = loader.get_u32("higgs-codec.n_codebooks", 8);
    cfg.codebook_size        = loader.get_u32("higgs-codec.codebook_size", 1024);
    cfg.codebook_dim         = loader.get_u32("higgs-codec.codebook_dim", 64);
    cfg.quantizer_dim        = loader.get_u32("higgs-codec.quantizer_dim", 1024);
    cfg.acoustic_dim         = loader.get_u32("higgs-codec.acoustic_dim", 256);
    cfg.semantic_dim         = loader.get_u32("higgs-codec.semantic_dim", 768);
    cfg.hubert_hidden        = loader.get_u32("higgs-codec.hubert_hidden", 512);
    cfg.n_semantic_layers    = loader.get_u32("higgs-codec.n_semantic_layers", 12);
    cfg.semantic_heads       = loader.get_u32("higgs-codec.semantic_heads", 12);
    cfg.pos_conv_groups      = loader.get_u32("higgs-codec.pos_conv_groups", 16);
    cfg.pos_conv_kernel      = loader.get_u32("higgs-codec.pos_conv_kernel", 128);

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

    backend_ = init_preferred_backend("HiggsEnc", &error_msg_, false, nullptr);
    if (!backend_) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
    const bool on_cpu = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (!on_cpu) {
        backend_cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu_) { error_msg_ = "CPU fallback backend init failed"; return false; }
    }
    fprintf(stderr, "  HiggsEnc backend: %s\n", dev ? ggml_backend_dev_name(dev) : "?");

    enum ggml_backend_dev_type want = on_cpu ? GGML_BACKEND_DEVICE_TYPE_CPU
                                             : ggml_backend_dev_type(dev);
    if (!load_tensor_data_from_file(gguf_path, gc, model_.ctx, model_.tensors,
                                    model_.buffer, error_msg_, want)) {
        return false;
    }

    auto get = [&](const std::string & n) -> struct ggml_tensor * {
        auto it = model_.tensors.find(n);
        return it == model_.tensors.end() ? nullptr : it->second;
    };
    auto need = [&](const std::string & n) -> struct ggml_tensor * {
        struct ggml_tensor * t = get(n);
        if (!t) error_msg_ = "missing tensor: " + n;
        return t;
    };

    char nm[96];
    // feature extractor
    for (int i = 0; i < 7; ++i) {
        snprintf(nm, sizeof(nm), "enc.sem.feat.conv.%d.weight", i);
        model_.feat_conv[i] = need(nm);
    }
    model_.gn0_w = need("enc.sem.feat.gn0.weight");
    model_.gn0_b = need("enc.sem.feat.gn0.bias");
    model_.fproj_ln_w = need("enc.sem.fproj.ln.weight");
    model_.fproj_ln_b = need("enc.sem.fproj.ln.bias");
    model_.fproj_w    = need("enc.sem.fproj.proj.weight");
    model_.fproj_b    = need("enc.sem.fproj.proj.bias");
    model_.pos_conv_w = need("enc.sem.pos_conv.weight");
    model_.pos_conv_b = need("enc.sem.pos_conv.bias");
    model_.enc_ln_w   = need("enc.sem.enc_ln.weight");
    model_.enc_ln_b   = need("enc.sem.enc_ln.bias");

    model_.layers.resize(cfg.n_semantic_layers);
    for (int i = 0; i < cfg.n_semantic_layers; ++i) {
        auto & L = model_.layers[i];
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.attn.q.weight", i);   L.q_w = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.attn.q.bias", i);     L.q_b = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.attn.k.weight", i);   L.k_w = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.attn.k.bias", i);     L.k_b = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.attn.v.weight", i);   L.v_w = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.attn.v.bias", i);     L.v_b = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.attn.out.weight", i); L.o_w = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.attn.out.bias", i);   L.o_b = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.ln.weight", i);       L.ln_w = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.ln.bias", i);         L.ln_b = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.ff.fc1.weight", i);   L.ff1_w = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.ff.fc1.bias", i);     L.ff1_b = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.ff.fc2.weight", i);   L.ff2_w = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.ff.fc2.bias", i);     L.ff2_b = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.final_ln.weight", i); L.fln_w = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.layers.%d.final_ln.bias", i);   L.fln_b = need(nm);
    }

    // encoder_semantic
    model_.sem_conv_w = need("enc.sem.enc.conv.weight");
    model_.sem_blocks.resize(2);
    for (int b = 0; b < 2; ++b) {
        auto & blk = model_.sem_blocks[b];
        snprintf(nm, sizeof(nm), "enc.sem.enc.block.%d.conv.weight", b); blk.conv_w = need(nm);
        snprintf(nm, sizeof(nm), "enc.sem.enc.block.%d.conv.bias", b);   blk.conv_b = need(nm);
        for (int r = 0; r < 2; ++r) {
            snprintf(nm, sizeof(nm), "enc.sem.enc.block.%d.res%d.conv1.weight", b, r); blk.res[r].conv1_w = need(nm);
            snprintf(nm, sizeof(nm), "enc.sem.enc.block.%d.res%d.conv2.weight", b, r); blk.res[r].conv2_w = need(nm);
        }
    }

    // acoustic_encoder
    model_.aenc_conv1_w = need("enc.aenc.conv1.weight");
    model_.aenc_conv1_b = need("enc.aenc.conv1.bias");
    model_.aenc_conv2_w = need("enc.aenc.conv2.weight");
    model_.aenc_conv2_b = need("enc.aenc.conv2.bias");
    struct ggml_tensor * aenc_snake_post_raw = need("enc.aenc.snake_post.alpha");
    const int dac_strides[5] = {8, 5, 4, 2, 3};
    model_.aenc_blocks.resize(5);
    std::vector<struct ggml_tensor *> snake_raw;   // collect F16 α tensors to host-precompute
    for (int b = 0; b < 5; ++b) {
        auto & blk = model_.aenc_blocks[b];
        blk.stride = dac_strides[b];
        snprintf(nm, sizeof(nm), "enc.aenc.block.%d.conv1.weight", b);    blk.conv1_w = need(nm);
        snprintf(nm, sizeof(nm), "enc.aenc.block.%d.conv1.bias", b);      blk.conv1_b = need(nm);
        snprintf(nm, sizeof(nm), "enc.aenc.block.%d.snake_pre.alpha", b); blk.snake_pre_alpha = need(nm);
        snake_raw.push_back(blk.snake_pre_alpha);
        for (int r = 1; r <= 3; ++r) {
            auto & ru = blk.res[r-1];
            ru.dilation = (r == 1) ? 1 : (r == 2) ? 3 : 9;
            snprintf(nm, sizeof(nm), "enc.aenc.block.%d.res%d.conv1.weight", b, r); ru.conv1_w = need(nm);
            snprintf(nm, sizeof(nm), "enc.aenc.block.%d.res%d.conv1.bias", b, r);   ru.conv1_b = need(nm);
            snprintf(nm, sizeof(nm), "enc.aenc.block.%d.res%d.conv2.weight", b, r); ru.conv2_w = need(nm);
            snprintf(nm, sizeof(nm), "enc.aenc.block.%d.res%d.conv2.bias", b, r);   ru.conv2_b = need(nm);
            snprintf(nm, sizeof(nm), "enc.aenc.block.%d.res%d.snake1.alpha", b, r); ru.snake1_alpha = need(nm);
            snprintf(nm, sizeof(nm), "enc.aenc.block.%d.res%d.snake2.alpha", b, r); ru.snake2_alpha = need(nm);
            snake_raw.push_back(ru.snake1_alpha);
            snake_raw.push_back(ru.snake2_alpha);
        }
    }
    snake_raw.push_back(aenc_snake_post_raw);

    // projection + RVQ
    model_.fc_w = need("enc.fc.weight");
    model_.fc_b = need("enc.fc.bias");
    model_.vq_proj_in_w.resize(cfg.n_codebooks);
    model_.vq_proj_in_b.resize(cfg.n_codebooks);
    model_.vq_embed.resize(cfg.n_codebooks);
    model_.vq_proj_out_w.resize(cfg.n_codebooks);
    model_.vq_proj_out_b.resize(cfg.n_codebooks);
    for (int i = 0; i < cfg.n_codebooks; ++i) {
        snprintf(nm, sizeof(nm), "vq.%d.project_in.weight", i); model_.vq_proj_in_w[i] = need(nm);
        snprintf(nm, sizeof(nm), "vq.%d.project_in.bias", i);   model_.vq_proj_in_b[i] = need(nm);
        snprintf(nm, sizeof(nm), "vq.%d.embed", i);             model_.vq_embed[i]     = need(nm);
        snprintf(nm, sizeof(nm), "vq.%d.project_out.weight", i); model_.vq_proj_out_w[i] = need(nm);
        snprintf(nm, sizeof(nm), "vq.%d.project_out.bias", i);   model_.vq_proj_out_b[i] = need(nm);
    }
    if (!error_msg_.empty()) return false;

    // ---- host-precompute snake α (F32) and inv = 1/(α+1e-9) ----
    {
        size_t n_snake = snake_raw.size();
        struct ggml_init_params sp = { ggml_tensor_overhead() * (n_snake * 2 + 4), nullptr, true };
        model_.snake_ctx = ggml_init(sp);
        if (!model_.snake_ctx) { error_msg_ = "ggml_init (snake) failed"; return false; }
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
        size_t idx = 0;
        for (int b = 0; b < 5; ++b) {
            auto & blk = model_.aenc_blocks[b];
            blk.snake_pre_alpha = a_t[idx]; blk.snake_pre_inv = inv_t[idx]; ++idx;
            for (int r = 0; r < 3; ++r) {
                blk.res[r].snake1_alpha = a_t[idx]; blk.res[r].snake1_inv = inv_t[idx]; ++idx;
                blk.res[r].snake2_alpha = a_t[idx]; blk.res[r].snake2_inv = inv_t[idx]; ++idx;
            }
        }
        model_.aenc_snake_post_alpha = a_t[idx]; model_.aenc_snake_post_inv = inv_t[idx]; ++idx;
    }

    std::vector<ggml_backend_t> backends = { backend_ };
    if (backend_cpu_) backends.push_back(backend_cpu_);
    sched_ = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(),
                                    HIGGS_ENC_MAX_NODES, false, true);
    if (!sched_) { error_msg_ = "sched_new failed"; return false; }
    compute_meta_.resize(ggml_tensor_overhead() * HIGGS_ENC_MAX_NODES +
                         ggml_graph_overhead_custom(HIGGS_ENC_MAX_NODES, false));

    fprintf(stderr, "  HiggsEnc loaded: %d sem layers, %d codebooks, sr=%d sem_sr=%d hop=%d\n",
            cfg.n_semantic_layers, cfg.n_codebooks, cfg.sample_rate,
            cfg.semantic_sample_rate, cfg.hop_length);
    return true;
}

// y = x + sin²(α·x)/(α+1e-9). α [1,C,1] broadcast over time (ne0) & batch.
struct ggml_tensor * HiggsCodecEncoder::apply_snake(struct ggml_context * ctx, struct ggml_tensor * x,
                                                    struct ggml_tensor * alpha, struct ggml_tensor * inv) {
    struct ggml_tensor * ax = ggml_mul(ctx, x, alpha);
    struct ggml_tensor * s  = ggml_sin(ctx, ax);
    struct ggml_tensor * s2 = ggml_sqr(ctx, s);
    struct ggml_tensor * t  = ggml_mul(ctx, s2, inv);
    return ggml_add(ctx, x, t);
}

// LayerNorm over ne0 (feature axis) with affine w,b [ne0].
static struct ggml_tensor * layer_norm(struct ggml_context * c, struct ggml_tensor * x,
                                       struct ggml_tensor * w, struct ggml_tensor * b, float eps) {
    x = ggml_norm(c, x, eps);
    x = ggml_mul(c, x, w);
    x = ggml_add(c, x, b);
    return x;
}

// ---------------------------------------------------------------------------
// HuBERT semantic feature extraction. Input wav16 (already resampled+padded
// outside). Output semfeat [T_sem, 768] in row-major (idx = t*768 + d).
// ---------------------------------------------------------------------------
bool HiggsCodecEncoder::run_semantic(const std::vector<float> & wav16,
                                     std::vector<float> & semfeat, int & T_sem) {
    const auto & cfg = model_.config;
    const float eps = 1e-5f;
    const int H = cfg.semantic_dim;      // 768
    const int nh = cfg.semantic_heads;   // 12
    const int hd = H / nh;               // 64
    const int Lin = (int)wav16.size();

    struct ggml_init_params p = { compute_meta_.size(), compute_meta_.data(), true };
    struct ggml_context * c = ggml_init(p);
    struct ggml_cgraph * gf = ggml_new_graph_custom(c, HIGGS_ENC_MAX_NODES, false);

    // input wav as [Lin, 1, 1]
    struct ggml_tensor * inp = ggml_new_tensor_3d(c, GGML_TYPE_F32, Lin, 1, 1);
    ggml_set_name(inp, "wav16"); ggml_set_input(inp);

    // ---- feature extractor: 7 convs, valid (no pad). layer 0 GroupNorm + gelu.
    struct ggml_tensor * x = inp;
    const int fe_k[7] = {10, 3, 3, 3, 3, 2, 2};
    const int fe_s[7] = {5, 2, 2, 2, 2, 2, 2};
    for (int i = 0; i < 7; ++i) {
        x = ggml_conv_1d_direct(c, model_.feat_conv[i], x, fe_s[i], 0, 0, 1); // [L', 512, 1]
        if (i == 0) {
            // GroupNorm(num_groups=512, channels=512) == per-channel norm over time.
            // ggml layout here is [time, channel, 1]; ggml_norm normalizes over ne0
            // (time) for each channel row -> exactly InstanceNorm per channel.
            x = ggml_norm(c, x, eps);                                   // norm over time
            // affine γ,β are per-channel (ne1). reshape to [1,512,1] to broadcast.
            struct ggml_tensor * g = ggml_reshape_3d(c, model_.gn0_w, 1, model_.gn0_w->ne[0], 1);
            struct ggml_tensor * bbias = ggml_reshape_3d(c, model_.gn0_b, 1, model_.gn0_b->ne[0], 1);
            x = ggml_mul(c, x, g);
            x = ggml_add(c, x, bbias);
        }
        x = ggml_gelu(c, x);
        (void)fe_k;
    }
    // x: [T, 512, 1] (time-major). transpose -> [512, T] feature-major.
    const int64_t T = x->ne[0];
    x = ggml_reshape_2d(c, x, x->ne[0], x->ne[1]);            // [T, 512]
    struct ggml_tensor * h = ggml_cont(c, ggml_transpose(c, x)); // [512, T]

    // ---- feature projection: LayerNorm(512) -> Linear 512->768
    h = layer_norm(c, h, model_.fproj_ln_w, model_.fproj_ln_b, eps);
    h = ggml_mul_mat(c, model_.fproj_w, h);                  // [768, T]
    h = ggml_add(c, h, model_.fproj_b);

    // ---- positional conv embedding (grouped conv, 16 groups) ----
    // h is [768, T]; conv wants [T, 768, 1]. groups=16 -> 48 ch/group.
    {
        struct ggml_tensor * ht = ggml_cont(c, ggml_transpose(c, h)); // [T, 768]
        ht = ggml_reshape_3d(c, ht, T, H, 1);
        const int groups = cfg.pos_conv_groups;       // 16
        const int cpg = H / groups;                   // 48
        const int K = cfg.pos_conv_kernel;            // 128, padding=64
        struct ggml_tensor * pos_full = nullptr;      // [T, 768, 1]
        for (int gi = 0; gi < groups; ++gi) {
            // input slice channels [gi*cpg .. +cpg)
            struct ggml_tensor * xin = ggml_view_3d(c, ht, T, cpg, 1,
                                                    ht->nb[1], ht->nb[2],
                                                    (size_t)gi * cpg * ht->nb[1]);
            xin = ggml_cont(c, xin);
            // weight slice out channels [gi*cpg .. +cpg) : pos_conv_w [K, cpg, 768]
            struct ggml_tensor * wslice = ggml_view_3d(c, model_.pos_conv_w, K, cpg, cpg,
                                                       model_.pos_conv_w->nb[1], model_.pos_conv_w->nb[2],
                                                       (size_t)gi * cpg * model_.pos_conv_w->nb[2]);
            wslice = ggml_cont(c, wslice);
            struct ggml_tensor * yg = ggml_conv_1d_direct(c, wslice, xin, 1, K/2, K/2, 1); // [T+1, cpg, 1]
            pos_full = pos_full ? ggml_concat(c, pos_full, yg, 1) : yg;
        }
        // add bias [768] -> reshape [1,768,1]
        pos_full = ggml_add(c, pos_full, ggml_reshape_3d(c, model_.pos_conv_b, 1, H, 1));
        // SamePad: even kernel -> drop last time step
        const int64_t Lp = pos_full->ne[0];
        pos_full = ggml_view_3d(c, pos_full, Lp - 1, H, 1, pos_full->nb[1], pos_full->nb[2], 0);
        pos_full = ggml_cont(c, pos_full);
        pos_full = ggml_gelu(c, pos_full);
        // back to [768, T]
        struct ggml_tensor * pos2d = ggml_reshape_2d(c, pos_full, Lp - 1, H);   // [T, 768]
        struct ggml_tensor * pos_fm = ggml_cont(c, ggml_transpose(c, pos2d));    // [768, T]
        h = ggml_add(c, h, pos_fm);
    }

    // ---- encoder.layer_norm (pre-layers) ----
    h = layer_norm(c, h, model_.enc_ln_w, model_.enc_ln_b, eps);

    // collect 13 hidden states (entry 0 = post pos_conv+LN input; then 12 layer outputs)
    std::vector<struct ggml_tensor *> hidden;
    hidden.push_back(h);

    const float scale = 1.0f / sqrtf((float)hd);
    for (int il = 0; il < cfg.n_semantic_layers; ++il) {
        auto & L = model_.layers[il];
        struct ggml_tensor * res = h;
        // attention (bidirectional, no mask)
        struct ggml_tensor * q = ggml_add(c, ggml_mul_mat(c, L.q_w, h), L.q_b); // [768, T]
        struct ggml_tensor * k = ggml_add(c, ggml_mul_mat(c, L.k_w, h), L.k_b);
        struct ggml_tensor * v = ggml_add(c, ggml_mul_mat(c, L.v_w, h), L.v_b);
        q = ggml_reshape_3d(c, q, hd, nh, T);
        k = ggml_reshape_3d(c, k, hd, nh, T);
        v = ggml_reshape_3d(c, v, hd, nh, T);
        struct ggml_tensor * Q = ggml_permute(c, q, 0, 2, 1, 3); // [hd, T, nh]
        struct ggml_tensor * K = ggml_permute(c, k, 0, 2, 1, 3); // [hd, T, nh]
        // scores = K^T Q scaled -> [T_k, T_q, nh]
        struct ggml_tensor * scores = ggml_mul_mat(c, K, Q);
        scores = ggml_soft_max_ext(c, scores, nullptr, scale, 0.0f);
        // out = scores * V : need V as [T_k, hd, nh]
        struct ggml_tensor * Vt = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3)); // [T, hd, nh]
        struct ggml_tensor * o = ggml_mul_mat(c, Vt, scores);   // [hd, T_q, nh]
        o = ggml_permute(c, o, 0, 2, 1, 3);                     // [hd, nh, T]
        o = ggml_cont_2d(c, o, H, T);                           // [768, T]
        o = ggml_add(c, ggml_mul_mat(c, L.o_w, o), L.o_b);
        h = ggml_add(c, res, o);
        h = layer_norm(c, h, L.ln_w, L.ln_b, eps);
        // feed forward
        res = h;
        struct ggml_tensor * ff = ggml_add(c, ggml_mul_mat(c, L.ff1_w, h), L.ff1_b);
        ff = ggml_gelu(c, ff);
        ff = ggml_add(c, ggml_mul_mat(c, L.ff2_w, ff), L.ff2_b);
        h = ggml_add(c, res, ff);
        h = layer_norm(c, h, L.fln_w, L.fln_b, eps);
        hidden.push_back(h);
    }

    // mean of 13 hidden states -> [768, T]
    struct ggml_tensor * mean = hidden[0];
    for (size_t i = 1; i < hidden.size(); ++i) mean = ggml_add(c, mean, hidden[i]);
    mean = ggml_scale(c, mean, 1.0f / (float)hidden.size());

    // mean is [768, T] feature-major. We want semfeat row-major [T,768] = idx t*768+d.
    // transpose to [T, 768] (time as ne0) then it is laid out d-major... careful:
    // ggml [768,T] has idx = t*768 + d already (ne0=768=d, ne1=t). So contiguous
    // copy gives row-major [T,768]. Just cont it.
    mean = ggml_cont(c, mean);
    ggml_set_name(mean, "semfeat_full");
    ggml_set_output(mean);
    ggml_build_forward_expand(gf, mean);

    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        error_msg_ = "sem: sched_alloc_graph failed"; ggml_free(c); return false;
    }
    struct ggml_tensor * inT = ggml_graph_get_tensor(gf, "wav16");
    ggml_backend_tensor_set(inT, wav16.data(), 0, (size_t)Lin * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "sem: graph_compute failed"; ggml_backend_sched_reset(sched_); ggml_free(c); return false;
    }

    struct ggml_tensor * out = ggml_graph_get_tensor(gf, "semfeat_full");
    int Tfull = (int)out->ne[1];     // [768, Tfull]
    std::vector<float> full((size_t)H * Tfull);
    ggml_backend_tensor_get(out, full.data(), 0, ggml_nbytes(out));
    ggml_backend_sched_reset(sched_);
    ggml_free(c);

    // stride-2 subsample along time (keep frames 0,2,4,...): semfeat[ts, :]
    int ds = cfg.semantic_downsample_factor;
    int Ts = (Tfull + ds - 1) / ds;
    semfeat.assign((size_t)Ts * H, 0.0f);
    for (int ts = 0, tf = 0; ts < Ts; ++ts, tf += ds) {
        // full layout: idx = tf*H + d  (ne0=H=d)
        std::memcpy(&semfeat[(size_t)ts * H], &full[(size_t)tf * H], (size_t)H * sizeof(float));
    }
    T_sem = Ts;
    return true;
}

// encoder_semantic residual unit (ELU activations) over feature-major [768, T].
static struct ggml_tensor * sem_res(struct ggml_context * c, struct ggml_tensor * x,
                                     const sem_res_unit & ru) {
    // operate on time-major [T, 768] for conv. caller passes [T, C, 1].
    struct ggml_tensor * y = ggml_elu(c, x);
    y = ggml_conv_1d_direct(c, ru.conv1_w, y, 1, 1, 1, 1);    // k3 pad1 -> same length
    y = ggml_elu(c, y);
    y = ggml_conv_1d_direct(c, ru.conv2_w, y, 1, 0, 0, 1);    // k1
    return ggml_add(c, x, y);
}

bool HiggsCodecEncoder::run_codes(const std::vector<float> & semfeat, int T_sem,
                                  const float * wav24k, int n_samples,
                                  std::vector<int32_t> & codes_TN, int & T_out,
                                  std::vector<float> & semantic, std::vector<float> & acoustic,
                                  std::vector<float> & prefc, std::vector<float> & postfc) {
    const auto & cfg = model_.config;
    const int H = cfg.semantic_dim;     // 768
    const int A = cfg.acoustic_dim;     // 256
    const int Q = cfg.quantizer_dim;    // 1024

    // ---- frame-count reconciliation for the acoustic branch ----
    // T_codes = floor over the DAC conv cascade. Compute acoustic out-length for
    // both raw and padded input; pad iff raw != T_sem (mirror oracle).
    auto dac_outlen = [&](int Lin) {
        // conv1 k7 s1 p3, then 5 blocks: each block has res-units (length-preserving)
        // then a strided conv1 (kernel=2s, stride=s, padding=ceil(s/2)). then conv2 k3 s1 p1.
        // torch formula: floor((L + 2p - d(k-1) -1)/s + 1).
        auto cl = [](int L, int k, int s, int p, int d) {
            return (L + 2*p - d*(k-1) - 1) / s + 1;
        };
        int L = cl(Lin, 7, 1, 3, 1);              // conv1
        const int strides[5] = {8,5,4,2,3};
        for (int b = 0; b < 5; ++b) {
            int s = strides[b];
            int k = 2*s;
            int p = (int)std::ceil(s / 2.0);
            L = cl(L, k, s, p, 1);                // block strided conv1
        }
        L = cl(L, 3, 1, 1, 1);                    // conv2
        return L;
    };
    int raw_T = dac_outlen(n_samples);
    bool pad_acoustic = (raw_T != T_sem);
    int aenc_pad = pad_acoustic ? cfg.model_pad : 0;   // 480 each side

    struct ggml_init_params p = { compute_meta_.size(), compute_meta_.data(), true };
    struct ggml_context * c = ggml_init(p);
    struct ggml_cgraph * gf = ggml_new_graph_custom(c, HIGGS_ENC_MAX_NODES, false);

    // ---- encoder_semantic on semfeat ----
    // semfeat host layout row-major [T_sem, 768] (idx = t*768+d). Load as [768, T_sem]
    // feature-major (ne0=768=d, ne1=t) which is the SAME contiguous bytes.
    struct ggml_tensor * sf = ggml_new_tensor_2d(c, GGML_TYPE_F32, H, T_sem);
    ggml_set_name(sf, "semfeat_in"); ggml_set_input(sf);
    // transpose to time-major [T_sem, 768, 1] for conv
    struct ggml_tensor * es = ggml_cont(c, ggml_transpose(c, sf));   // [T_sem, 768]
    es = ggml_reshape_3d(c, es, T_sem, H, 1);
    es = ggml_conv_1d_direct(c, model_.sem_conv_w, es, 1, 1, 1, 1);  // k3 pad1, bias-less
    for (int b = 0; b < 2; ++b) {
        auto & blk = model_.sem_blocks[b];
        es = sem_res(c, es, blk.res[0]);
        es = sem_res(c, es, blk.res[1]);
        es = ggml_conv_1d_direct(c, blk.conv_w, es, 1, 1, 1, 1);     // k3 pad1
        es = ggml_add(c, es, ggml_reshape_3d(c, blk.conv_b, 1, H, 1));
    }
    // es: [T, 768, 1] time-major. semantic output (feature-major [768,T] for concat)
    const int64_t T = es->ne[0];
    struct ggml_tensor * sem2d = ggml_reshape_2d(c, es, T, H);       // [T, 768]
    struct ggml_tensor * e_semantic = ggml_cont(c, ggml_transpose(c, sem2d)); // [768, T]
    ggml_set_name(e_semantic, "e_semantic"); ggml_set_output(e_semantic);

    // ---- acoustic_encoder on raw 24k wav ----
    int Lwav = n_samples + 2*aenc_pad;
    struct ggml_tensor * aw = ggml_new_tensor_3d(c, GGML_TYPE_F32, Lwav, 1, 1);
    ggml_set_name(aw, "wav24"); ggml_set_input(aw);
    struct ggml_tensor * a = ggml_conv_1d_direct(c, model_.aenc_conv1_w, aw, 1, 3, 3, 1); // k7 pad3 -> [L,64]
    a = ggml_add(c, a, ggml_reshape_3d(c, model_.aenc_conv1_b, 1, 64, 1));
    const int strides[5] = {8,5,4,2,3};
    for (int b = 0; b < 5; ++b) {
        auto & blk = model_.aenc_blocks[b];
        // 3 res units (snake1 -> conv1 k7 dil -> snake2 -> conv2 k1; residual)
        for (int r = 0; r < 3; ++r) {
            auto & ru = blk.res[r];
            struct ggml_tensor * res = a;
            struct ggml_tensor * y = apply_snake(c, a, ru.snake1_alpha, ru.snake1_inv);
            int pad1 = (7 - 1) * ru.dilation / 2;
            y = ggml_conv_1d_direct(c, ru.conv1_w, y, 1, pad1, pad1, ru.dilation);
            y = ggml_add(c, y, ggml_reshape_3d(c, ru.conv1_b, 1, ru.conv1_b->ne[0], 1));
            y = apply_snake(c, y, ru.snake2_alpha, ru.snake2_inv);
            y = ggml_conv_1d_direct(c, ru.conv2_w, y, 1, 0, 0, 1);   // k1
            y = ggml_add(c, y, ggml_reshape_3d(c, ru.conv2_b, 1, ru.conv2_b->ne[0], 1));
            a = ggml_add(c, res, y);
        }
        // snake_pre -> strided conv1 (kernel=2s, stride=s, padding=ceil(s/2))
        a = apply_snake(c, a, blk.snake_pre_alpha, blk.snake_pre_inv);
        int s = strides[b];
        int k = 2*s;
        int padc = (int)std::ceil(s / 2.0);
        int Cout = (int)blk.conv1_w->ne[2];
        a = ggml_conv_1d_direct(c, blk.conv1_w, a, s, padc, padc, 1);
        a = ggml_add(c, a, ggml_reshape_3d(c, blk.conv1_b, 1, Cout, 1));
        (void)k;
    }
    // post snake -> conv2 (k3 pad1) -> [L,256]
    a = apply_snake(c, a, model_.aenc_snake_post_alpha, model_.aenc_snake_post_inv);
    a = ggml_conv_1d_direct(c, model_.aenc_conv2_w, a, 1, 1, 1, 1);
    a = ggml_add(c, a, ggml_reshape_3d(c, model_.aenc_conv2_b, 1, A, 1));
    // a: [Ta, 256, 1]. feature-major [256, Ta]
    const int64_t Ta = a->ne[0];
    struct ggml_tensor * a2d = ggml_reshape_2d(c, a, Ta, A);
    struct ggml_tensor * e_acoustic = ggml_cont(c, ggml_transpose(c, a2d));  // [256, Ta]
    ggml_set_name(e_acoustic, "e_acoustic"); ggml_set_output(e_acoustic);

    // ---- concat [acoustic(256); semantic(768)] along feature axis (ne0) ----
    // Both feature-major [C, T]; concat over ne0 yields [1024, T]. acoustic first.
    // (Requires Ta == T; the pad reconciliation should ensure this.)
    struct ggml_tensor * cat = ggml_concat(c, e_acoustic, e_semantic, 0);   // [1024, T]
    ggml_set_name(cat, "prefc"); ggml_set_output(cat);

    // ---- fc 1024->1024 ----
    struct ggml_tensor * pf = ggml_add(c, ggml_mul_mat(c, model_.fc_w, cat), model_.fc_b); // [1024, T]
    ggml_set_name(pf, "postfc"); ggml_set_output(pf);

    ggml_build_forward_expand(gf, e_semantic);
    ggml_build_forward_expand(gf, e_acoustic);
    ggml_build_forward_expand(gf, cat);
    ggml_build_forward_expand(gf, pf);

    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        error_msg_ = "codes: sched_alloc_graph failed"; ggml_free(c); return false;
    }
    // set inputs
    struct ggml_tensor * sfT = ggml_graph_get_tensor(gf, "semfeat_in");
    ggml_backend_tensor_set(sfT, semfeat.data(), 0, (size_t)H * T_sem * sizeof(float));
    struct ggml_tensor * awT = ggml_graph_get_tensor(gf, "wav24");
    {
        std::vector<float> padded((size_t)Lwav, 0.0f);
        std::memcpy(&padded[aenc_pad], wav24k, (size_t)n_samples * sizeof(float));
        ggml_backend_tensor_set(awT, padded.data(), 0, (size_t)Lwav * sizeof(float));
    }
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "codes: graph_compute failed"; ggml_backend_sched_reset(sched_); ggml_free(c); return false;
    }

    // pull stage tensors (feature-major [C,T]: ggml idx = t*C + c)
    auto pull = [&](const char * nm) -> std::vector<float> {
        struct ggml_tensor * t = ggml_graph_get_tensor(gf, nm);
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        return v;
    };
    std::vector<float> sem_tc = pull("e_semantic");   // [768, T] -> idx t*768+c
    std::vector<float> aco_tc = pull("e_acoustic");
    std::vector<float> pre_tc = pull("prefc");
    std::vector<float> post   = pull("postfc");       // [1024, T]
    int Tfin = (int)T;

    // convert stage tensors to oracle channel-major [C, T] (idx = c*T + t)
    auto to_ct = [&](const std::vector<float> & in, int C, int Tn) {
        std::vector<float> o((size_t)C * Tn);
        for (int t = 0; t < Tn; ++t) for (int cc = 0; cc < C; ++cc) o[(size_t)cc*Tn + t] = in[(size_t)t*C + cc];
        return o;
    };
    semantic = to_ct(sem_tc, H, Tfin);
    acoustic = to_ct(aco_tc, A, Tfin);
    prefc    = to_ct(pre_tc, Q, Tfin);
    postfc   = to_ct(post,   Q, Tfin);   // ggml fc is correct (postfc==oracle 0.9994 numpy-verified)

    ggml_backend_sched_reset(sched_);
    ggml_free(c);

    // ---- RVQ encode (host) on postfc ----
    // Read project_in / codebook / project_out matrices to host (F16/F32).
    auto read_mat = [&](struct ggml_tensor * t, std::vector<float> & dst) {
        int64_t ne = ggml_nelements(t);
        dst.resize(ne);
        if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h(ne);
            ggml_backend_tensor_get(t, h.data(), 0, ne * sizeof(ggml_fp16_t));
            for (int64_t i = 0; i < ne; ++i) dst[i] = ggml_fp16_to_fp32(h[i]);
        } else {
            ggml_backend_tensor_get(t, dst.data(), 0, ne * sizeof(float));
        }
    };

    const int D = cfg.codebook_dim;     // 64
    const int CB = cfg.codebook_size;   // 1024
    codes_TN.assign((size_t)Tfin * cfg.n_codebooks, 0);

    // residual starts as postfc, channel-major [Q, T] (idx = q*T + t) -> easier to
    // work per-frame; build a frame-major residual [T][Q].
    std::vector<float> resid((size_t)Tfin * Q);
    for (int t = 0; t < Tfin; ++t)
        for (int q = 0; q < Q; ++q)
            resid[(size_t)t*Q + q] = postfc[(size_t)q*Tfin + t];

    std::vector<float> pin_w, pin_b, emb, pout_w, pout_b;
    for (int qi = 0; qi < cfg.n_codebooks; ++qi) {
        read_mat(model_.vq_proj_in_w[qi], pin_w);   // ggml [Q, D] -> ne0=Q, row r in [D] is r*Q.. (torch [64,1024])
        read_mat(model_.vq_proj_in_b[qi], pin_b);   // [D]
        read_mat(model_.vq_embed[qi],     emb);     // ggml [D, CB] -> idx = e*D + d (torch [1024,64])
        read_mat(model_.vq_proj_out_w[qi], pout_w); // ggml [D, Q] (torch [1024,64]) idx = q*D + d
        read_mat(model_.vq_proj_out_b[qi], pout_b); // [Q]

        // precompute codebook squared norms
        std::vector<float> cb_sq(CB, 0.0f);
        for (int e = 0; e < CB; ++e) {
            float s = 0; const float * cr = &emb[(size_t)e*D];
            for (int d = 0; d < D; ++d) s += cr[d]*cr[d];
            cb_sq[e] = s;
        }
        for (int t = 0; t < Tfin; ++t) {
            const float * rx = &resid[(size_t)t*Q];
            // project_in: proj[d] = sum_q W[d,q]*rx[q] + b[d].
            // ggml pin_w is [Q, D] (ne0=Q): element (d,q) at idx d*Q + q  (== torch [D,Q] row-major)
            float proj[64];
            for (int d = 0; d < D; ++d) {
                const float * wr = &pin_w[(size_t)d*Q];
                float s = pin_b[d];
                for (int q = 0; q < Q; ++q) s += wr[q]*rx[q];
                proj[d] = s;
            }
            // nearest codebook by L2: argmin ||proj - cb_e||^2 = argmax(2*dot - cb_sq)
            int best = 0; float best_score = -1e30f;
            for (int e = 0; e < CB; ++e) {
                const float * cr = &emb[(size_t)e*D];
                float dot = 0;
                for (int d = 0; d < D; ++d) dot += proj[d]*cr[d];
                float score = 2.0f*dot - cb_sq[e];
                if (score > best_score) { best_score = score; best = e; }
            }
            codes_TN[(size_t)t*cfg.n_codebooks + qi] = best;
            // decode: q64 = emb[best] ; q1024 = project_out(q64) ; resid -= q1024
            const float * cr = &emb[(size_t)best*D];
            // project_out: out[q] = sum_d Wo[q,d]*cr[d] + bo[q]; pout_w ggml [D,Q] (ne0=D): (q,d) at q*D+d
            for (int q = 0; q < Q; ++q) {
                const float * wr = &pout_w[(size_t)q*D];
                float s = pout_b[q];
                for (int d = 0; d < D; ++d) s += wr[d]*cr[d];
                resid[(size_t)t*Q + q] -= s;
            }
        }
    }

    T_out = Tfin;
    return true;
}

bool HiggsCodecEncoder::encode_debug(const float * wav24k, int n_samples,
                                     std::vector<int32_t> & codes_TN, int & T,
                                     std::vector<float> & semfeat,
                                     std::vector<float> & semantic,
                                     std::vector<float> & acoustic,
                                     std::vector<float> & prefc,
                                     std::vector<float> & postfc) {
    if (!model_.ctx) { error_msg_ = "model not loaded"; return false; }
    const auto & cfg = model_.config;

    // resample 24k -> 16k for the semantic branch
    std::vector<float> wav24(wav24k, wav24k + n_samples);
    std::vector<float> wav16 = resample(wav24, cfg.sample_rate, cfg.semantic_sample_rate);
    // pad(160,160)
    std::vector<float> wav16p((size_t)wav16.size() + 320, 0.0f);
    std::memcpy(&wav16p[160], wav16.data(), wav16.size() * sizeof(float));

    int T_sem = 0;
    if (!run_semantic(wav16p, semfeat, T_sem)) return false;

    if (!run_codes(semfeat, T_sem, wav24k, n_samples, codes_TN, T,
                   semantic, acoustic, prefc, postfc)) return false;
    return true;
}

bool HiggsCodecEncoder::encode(const float * wav24k, int n_samples,
                               std::vector<int32_t> & codes_TN, int & T) {
    std::vector<float> a, b, cc, d, e;
    return encode_debug(wav24k, n_samples, codes_TN, T, a, b, cc, d, e);
}

bool HiggsCodecEncoder::encode(const float * wav, int n_samples, int in_sr,
                               std::vector<int32_t> & codes_TN, int & T) {
    const int target = model_.config.sample_rate;   // 24000
    if (in_sr <= 0 || in_sr == target) return encode(wav, n_samples, codes_TN, T);
    std::vector<float> in(wav, wav + n_samples);
    std::vector<float> wav24 = resample(in, in_sr, target);
    return encode(wav24.data(), (int)wav24.size(), codes_TN, T);
}

} // namespace higgs
