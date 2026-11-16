// FuseAlign + Dia text encoder on a ggml backend. Mirrors fusealign_hub/model.py, text_encoder.py and
// FuseAlignAligner._forward_group for a one-clip batch (no padding, so every mask in the Python forward is a no-op).
//
// Precision. Operands that torch holds in bf16 are bf16 here; accumulated activations stay F32:
// - Dia (bf16 in Python on every device): BF16 weights (ggml rounds the matmul inputs to bf16), bf16 RoPE tables, and
//   the encoder output rounded to bf16. Rounding every intermediate activation as torch does was measured not to bring
//   the outputs closer (torch's own bf16 GEMM accumulation order dominates the differences), so it is not done.
// - FuseAlign on CPU: fp32 throughout, as the Python CPU path.
// - FuseAlign on a GPU (Python: CUDA bf16 autocast with cast_autocast_weights): the Linear/Conv weights Python stores in
//   bf16 are BF16 (biases bf16-valued F32), so cuBLAS runs those GEMMs on bf16 operands with F32 accumulation; the RoPE
//   tables and pooling weights autocast rounds to bf16 are rounded too. Residual stream, norms, attention softmax, word
//   MLPs and logits are F32.
// RoPE tables are computed on the host as torch computes them (fp32 angles, fractional text positions).

#include "model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fusealign {
namespace {

constexpr const char * GGUF_FORMAT = "fusealign-gguf/1";
constexpr int GRAPH_SIZE = 8192;
// RefinementHead: ConvTranspose1d(kernel 4, stride 2, padding 1), then Conv1d(kernel 5, padding 2).
constexpr int UP_KERNEL = 4, UP_STRIDE = 2, UP_PAD = 1, SMOOTH_KERNEL = 5, SMOOTH_PAD = 2;
// FuseAlign RMSNorm epsilon (model.py RMSNorm default) and the _standardize epsilon.
constexpr float NORM_EPS = 1e-6f, STD_EPS = 1e-5f;

struct DiaParams {
    int hidden = 0, n_layers = 0, n_heads = 0, head_dim = 0, inter = 0, vocab = 0, max_pos = 0;
    float norm_eps = 0.0f, rope_theta = 0.0f;
};

// Round to the nearest bf16 (ties to even), as torch's .to(torch.bfloat16).
float round_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return x;  // NaN
    }
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    float y;
    std::memcpy(&y, &u, sizeof(y));
    return y;
}

struct GgufFile {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    ~GgufFile() {
        if (gguf) gguf_free(gguf);
        if (meta) ggml_free(meta);
    }
};

int64_t require_key(const gguf_context * g, const std::string & key, gguf_type type) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) {
        throw std::runtime_error("GGUF is missing key " + key);
    }
    if (gguf_get_kv_type(g, id) != type) {
        throw std::runtime_error("GGUF key " + key + " has the wrong type");
    }
    return id;
}

int get_u32(const gguf_context * g, const std::string & key) {
    return (int) gguf_get_val_u32(g, require_key(g, key, GGUF_TYPE_UINT32));
}

float get_f32(const gguf_context * g, const std::string & key) {
    return gguf_get_val_f32(g, require_key(g, key, GGUF_TYPE_FLOAT32));
}

std::string get_str(const gguf_context * g, const std::string & key) {
    return gguf_get_val_str(g, require_key(g, key, GGUF_TYPE_STRING));
}

struct Block {
    ggml_tensor * attn_norm, * qkv, * out, * ffn_norm, * w1, * w2, * w3;
};

struct DiaLayer {
    ggml_tensor * pre_norm, * q, * k, * v, * o, * post_norm, * gate_up, * down;
};

struct Head {
    ggml_tensor * in_w, * in_b;                     // Conv1d(dim, feat, 1): [dim, feat]
    std::vector<ggml_tensor *> up_w, up_b;          // ConvTranspose1d, transposed to [feat, K * feat]
    ggml_tensor * smooth_w, * smooth_b;             // Conv1d(feat, feat, 5): [5, feat, feat]
    ggml_tensor * out_w, * out_b;                   // Conv1d(feat, feat, 1): [feat, feat]
};

struct Mlp {
    ggml_tensor * w0, * b0, * w2, * b2;
};

}  // namespace

struct Model::Impl {
    HParams hp;
    DiaParams dia;
    ggml_backend_t backend = nullptr;
    bool autocast = false;  // GPU: FuseAlign weights and operands as under Python's CUDA bf16 autocast
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::vector<uint8_t> graph_mem;

    std::unordered_map<std::string, ggml_tensor *> weights;
    ggml_tensor * audio_patch_w = nullptr, * audio_patch_b = nullptr, * text_proj_w = nullptr,
                * text_proj_b = nullptr, * mod_emb = nullptr, * final_norm = nullptr;
    std::vector<Block> blocks;
    Head start_head, end_head;
    Mlp start_mlp, end_mlp, presence_mlp;
    ggml_tensor * dia_embedding = nullptr, * dia_norm = nullptr;
    std::vector<DiaLayer> dia_layers;
    std::vector<float> fa_inv_freq, dia_inv_freq;

    ~Impl() {
        if (galloc) ggml_gallocr_free(galloc);
        if (buf_w) ggml_backend_buffer_free(buf_w);
        if (ctx_w) ggml_free(ctx_w);
        if (backend) ggml_backend_free(backend);
    }

    void init_backend(const BackendOptions & options);
    void load(const std::string & path);
    ggml_tensor * weight(const std::string & name) const;
    Placement forward(const float * wav, size_t n_samples, const std::vector<std::string> & words);
};

void Model::Impl::init_backend(const BackendOptions & options) {
    const std::string & device = options.device;
    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(dev);
        }
    }
    ggml_backend_dev_t dev = nullptr;
    if (device == "cpu" || (device == "auto" && gpus.empty())) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    } else if (device == "auto" || device == "cuda") {
        if (gpus.empty()) {
            throw std::runtime_error("device \"cuda\" requested but no CUDA device is available");
        }
        dev = gpus[0];
    } else if (device.rfind("cuda:", 0) == 0) {
        size_t index = 0, used = 0;
        try {
            index = std::stoul(device.substr(5), &used);
        } catch (const std::exception &) {
            used = 0;
        }
        if (used == 0 || used != device.size() - 5) {
            throw std::runtime_error("invalid device \"" + device + "\"");
        }
        if (index >= gpus.size()) {
            throw std::runtime_error("device \"" + device + "\" requested but only " + std::to_string(gpus.size()) +
                                     " CUDA device(s) are available");
        }
        dev = gpus[index];
    } else {
        throw std::runtime_error("unknown device \"" + device + "\" (expected auto, cpu, cuda or cuda:N)");
    }
    if (!dev) {
        throw std::runtime_error("no ggml device for \"" + device + "\"");
    }
    backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        throw std::runtime_error(std::string("failed to initialize ggml device ") + ggml_backend_dev_name(dev));
    }
    autocast = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
    if (ggml_backend_is_cpu(backend)) {
        const int threads = options.n_threads > 0 ? options.n_threads
                                                  : (int) std::max(1u, std::thread::hardware_concurrency());
        ggml_backend_cpu_set_n_threads(backend, threads);
    }
}

ggml_tensor * Model::Impl::weight(const std::string & name) const {
    auto it = weights.find(name);
    if (it == weights.end()) {
        throw std::runtime_error("GGUF is missing tensor " + name);
    }
    return it->second;
}

void Model::Impl::load(const std::string & path) {
    GgufFile file;
    gguf_init_params params = {true, &file.meta};
    file.gguf = gguf_init_from_file(path.c_str(), params);
    if (!file.gguf) {
        throw std::runtime_error("cannot read GGUF file " + path);
    }
    const gguf_context * g = file.gguf;
    if (get_str(g, "general.architecture") != "fusealign" || get_str(g, "fusealign.format") != GGUF_FORMAT) {
        throw std::runtime_error(path + " is not a " + GGUF_FORMAT + " file");
    }
    hp.dim = get_u32(g, "fusealign.dim");
    hp.n_layers = get_u32(g, "fusealign.n_layers");
    hp.n_heads = get_u32(g, "fusealign.n_heads");
    hp.ffn_dim = get_u32(g, "fusealign.ffn_dim");
    hp.feat_dim = get_u32(g, "fusealign.feat_dim");
    hp.patch_window = get_u32(g, "fusealign.patch_window");
    hp.patch_hop = get_u32(g, "fusealign.patch_hop");
    hp.n_upsample = get_u32(g, "fusealign.n_upsample");
    hp.text_dim = get_u32(g, "fusealign.text_dim");
    hp.rope_theta = get_f32(g, "fusealign.rope_theta");
    hp.presence_threshold = get_f32(g, "fusealign.presence_threshold");
    hp.max_text_bytes = get_u32(g, "fusealign.max_text_bytes");
    hp.max_seconds = get_f32(g, "fusealign.max_seconds");
    dia.hidden = get_u32(g, "dia.hidden_size");
    dia.n_layers = get_u32(g, "dia.num_layers");
    dia.n_heads = get_u32(g, "dia.num_heads");
    dia.head_dim = get_u32(g, "dia.head_dim");
    dia.inter = get_u32(g, "dia.intermediate_size");
    dia.vocab = get_u32(g, "dia.vocab_size");
    dia.max_pos = get_u32(g, "dia.max_position_embeddings");
    dia.norm_eps = get_f32(g, "dia.norm_eps");
    dia.rope_theta = get_f32(g, "dia.rope_theta");
    if (hp.dim <= 0 || hp.n_heads <= 0 || hp.dim % hp.n_heads != 0 || (hp.dim / hp.n_heads) % 2 != 0 ||
        dia.head_dim % 2 != 0 || hp.text_dim != dia.hidden || hp.patch_window <= 0 || hp.patch_hop <= 0 ||
        dia.vocab < 256 || hp.max_text_bytes + 2 > dia.max_pos) {
        throw std::runtime_error(path + ": inconsistent hyperparameters");
    }

    const int dim = hp.dim, feat = hp.feat_dim, D = dia.hidden, HD = dia.n_heads * dia.head_dim;

    // Target tensors: name -> (expected GGUF ne, uploaded ne). Only model.* and text_encoder.* are loaded.
    struct Spec {
        std::string name;
        std::vector<int64_t> ne;  // expected ne in the file
        enum Kind { AsIs, ToF32, Autocast, AutocastBias, UpConv } kind;
    };
    std::vector<Spec> specs;
    auto add = [&](const std::string & name, std::vector<int64_t> ne, Spec::Kind kind) {
        specs.push_back({name, std::move(ne), kind});
    };
    add("model.audio_patch.weight", {hp.patch_window, 1, dim}, Spec::Autocast);
    add("model.audio_patch.bias", {dim}, Spec::AutocastBias);
    add("model.text_proj.weight", {hp.text_dim, dim}, Spec::Autocast);
    add("model.text_proj.bias", {dim}, Spec::AutocastBias);
    add("model.mod_emb", {dim, 2}, Spec::AsIs);
    for (int i = 0; i < hp.n_layers; ++i) {
        const std::string p = "model.blocks." + std::to_string(i) + ".";
        add(p + "attn_norm.weight", {dim}, Spec::AsIs);
        add(p + "attn.qkv.weight", {dim, 3 * dim}, Spec::Autocast);
        add(p + "attn.out.weight", {dim, dim}, Spec::Autocast);
        add(p + "ffn_norm.weight", {dim}, Spec::AsIs);
        add(p + "ffn.w1.weight", {dim, hp.ffn_dim}, Spec::Autocast);
        add(p + "ffn.w3.weight", {dim, hp.ffn_dim}, Spec::Autocast);
        add(p + "ffn.w2.weight", {hp.ffn_dim, dim}, Spec::Autocast);
    }
    add("model.final_norm.weight", {dim}, Spec::AsIs);
    for (const char * head : {"start", "end"}) {
        const std::string p = std::string("model.") + head + "_refinement.";
        add(p + "0.weight", {1, dim, feat}, Spec::Autocast);
        add(p + "0.bias", {feat}, Spec::AutocastBias);
        for (int i = 0; i < hp.n_upsample; ++i) {
            add(p + std::to_string(1 + 2 * i) + ".weight", {UP_KERNEL, feat, feat}, Spec::UpConv);
            add(p + std::to_string(1 + 2 * i) + ".bias", {feat}, Spec::AutocastBias);
        }
        add(p + std::to_string(1 + 2 * hp.n_upsample) + ".weight", {SMOOTH_KERNEL, feat, feat}, Spec::Autocast);
        add(p + std::to_string(1 + 2 * hp.n_upsample) + ".bias", {feat}, Spec::AutocastBias);
        add(p + std::to_string(3 + 2 * hp.n_upsample) + ".weight", {1, feat, feat}, Spec::Autocast);
        add(p + std::to_string(3 + 2 * hp.n_upsample) + ".bias", {feat}, Spec::AutocastBias);
    }
    for (const char * mlp : {"start", "end", "presence"}) {
        const std::string p = std::string("model.") + mlp + "_mlp.";
        const int out = std::string(mlp) == "presence" ? 1 : feat;
        add(p + "0.weight", {dim, dim}, Spec::AsIs);
        add(p + "0.bias", {dim}, Spec::AsIs);
        add(p + "2.weight", {dim, out}, Spec::AsIs);
        add(p + "2.bias", {out}, Spec::AsIs);
    }
    add("text_encoder.embedding.weight", {D, dia.vocab}, Spec::ToF32);
    for (int i = 0; i < dia.n_layers; ++i) {
        const std::string p = "text_encoder.layers." + std::to_string(i) + ".";
        add(p + "pre_sa_norm.weight", {D}, Spec::ToF32);
        add(p + "self_attention.q_proj.weight", {D, HD}, Spec::AsIs);
        add(p + "self_attention.k_proj.weight", {D, HD}, Spec::AsIs);
        add(p + "self_attention.v_proj.weight", {D, HD}, Spec::AsIs);
        add(p + "self_attention.o_proj.weight", {HD, D}, Spec::AsIs);
        add(p + "post_sa_norm.weight", {D}, Spec::ToF32);
        add(p + "mlp.gate_up_proj.weight", {D, 2 * dia.inter}, Spec::AsIs);
        add(p + "mlp.down_proj.weight", {dia.inter, D}, Spec::AsIs);
    }
    add("text_encoder.norm.weight", {D}, Spec::ToF32);

    // Room for the tensors plus the reshaped views taken of them below.
    ggml_init_params wp = {ggml_tensor_overhead() * 2 * specs.size(), nullptr, true};
    ctx_w = ggml_init(wp);
    if (!ctx_w) {
        throw std::runtime_error("ggml_init failed for the weight context");
    }
    std::vector<ggml_tensor *> sources(specs.size());
    for (size_t s = 0; s < specs.size(); ++s) {
        const Spec & spec = specs[s];
        ggml_tensor * src = ggml_get_tensor(file.meta, spec.name.c_str());
        if (!src) {
            throw std::runtime_error(path + ": missing tensor " + spec.name);
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            const int64_t want = d < (int) spec.ne.size() ? spec.ne[d] : 1;
            if (src->ne[d] != want) {
                throw std::runtime_error(path + ": tensor " + spec.name + " has an unexpected shape");
            }
        }
        const bool quantized = ggml_is_quantized(src->type);
        if (!quantized && src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_BF16 && src->type != GGML_TYPE_F16) {
            throw std::runtime_error(path + ": tensor " + spec.name + " has an unsupported type");
        }
        ggml_type type = src->type;
        int64_t ne[GGML_MAX_DIMS] = {src->ne[0], src->ne[1], src->ne[2], src->ne[3]};
        switch (spec.kind) {
            case Spec::AsIs:
                break;
            case Spec::ToF32:
            case Spec::AutocastBias:
                type = GGML_TYPE_F32;
                break;
            case Spec::Autocast:
                if (!quantized) type = autocast ? GGML_TYPE_BF16 : GGML_TYPE_F32;
                break;
            case Spec::UpConv:
                // torch [in, out, k] (ne [k, out, in]) -> [in, k * out] (ne0 = in) for mul_mat + col2im_1d.
                if (quantized) {
                    throw std::runtime_error(path + ": tensor " + spec.name + " cannot be quantized");
                }
                type = autocast ? GGML_TYPE_BF16 : GGML_TYPE_F32;
                ne[0] = src->ne[2];
                ne[1] = src->ne[0] * src->ne[1];
                ne[2] = 1;
                break;
        }
        ggml_tensor * t = ggml_new_tensor(ctx_w, type, 4, ne);
        ggml_set_name(t, spec.name.c_str());
        weights[spec.name] = t;
        sources[s] = src;
    }
    buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (!buf_w) {
        throw std::runtime_error("failed to allocate the model weights on " + std::string(ggml_backend_name(backend)));
    }
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    const size_t data_offset = gguf_get_data_offset(g);
    std::vector<uint8_t> raw;
    std::vector<float> f32, moved;
    std::vector<ggml_bf16_t> bf16;
    for (size_t s = 0; s < specs.size(); ++s) {
        const Spec & spec = specs[s];
        ggml_tensor * src = sources[s];
        ggml_tensor * dst = weights[spec.name];
        const int64_t id = gguf_find_tensor(g, spec.name.c_str());
        const size_t nbytes = ggml_nbytes(src);
        raw.resize(nbytes);
        in.seekg((std::streamoff) (data_offset + gguf_get_tensor_offset(g, id)));
        in.read(reinterpret_cast<char *>(raw.data()), (std::streamsize) nbytes);
        if (!in) {
            throw std::runtime_error(path + ": truncated tensor data for " + spec.name);
        }
        if (dst->type == src->type && spec.kind != Spec::UpConv) {
            ggml_backend_tensor_set(dst, raw.data(), 0, nbytes);
            continue;
        }
        const int64_t n = ggml_nelements(src);
        f32.resize(n);
        if (src->type == GGML_TYPE_F32) {
            std::memcpy(f32.data(), raw.data(), n * sizeof(float));
        } else {
            ggml_get_type_traits(src->type)->to_float(raw.data(), f32.data(), n);
        }
        if (spec.kind == Spec::UpConv) {
            const int64_t K = src->ne[0], OC = src->ne[1], IC = src->ne[2];
            moved.resize(n);
            for (int64_t ic = 0; ic < IC; ++ic) {
                for (int64_t r = 0; r < K * OC; ++r) {
                    moved[r * IC + ic] = f32[ic * K * OC + r];
                }
            }
            f32.swap(moved);
        }
        if (spec.kind == Spec::AutocastBias && autocast) {
            for (float & x : f32) x = round_bf16(x);
        }
        if (dst->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(dst, f32.data(), 0, n * sizeof(float));
        } else {
            bf16.resize(n);
            ggml_fp32_to_bf16_row_ref(f32.data(), bf16.data(), n);
            ggml_backend_tensor_set(dst, bf16.data(), 0, n * sizeof(ggml_bf16_t));
        }
    }

    audio_patch_w = ggml_reshape_2d(ctx_w, weight("model.audio_patch.weight"), hp.patch_window, dim);
    audio_patch_b = weight("model.audio_patch.bias");
    text_proj_w = weight("model.text_proj.weight");
    text_proj_b = weight("model.text_proj.bias");
    mod_emb = weight("model.mod_emb");
    final_norm = weight("model.final_norm.weight");
    for (int i = 0; i < hp.n_layers; ++i) {
        const std::string p = "model.blocks." + std::to_string(i) + ".";
        blocks.push_back({weight(p + "attn_norm.weight"), weight(p + "attn.qkv.weight"), weight(p + "attn.out.weight"),
                          weight(p + "ffn_norm.weight"), weight(p + "ffn.w1.weight"), weight(p + "ffn.w2.weight"),
                          weight(p + "ffn.w3.weight")});
    }
    for (Head * head : {&start_head, &end_head}) {
        const std::string p = std::string("model.") + (head == &start_head ? "start" : "end") + "_refinement.";
        head->in_w = ggml_reshape_2d(ctx_w, weight(p + "0.weight"), dim, feat);
        head->in_b = weight(p + "0.bias");
        for (int i = 0; i < hp.n_upsample; ++i) {
            head->up_w.push_back(weight(p + std::to_string(1 + 2 * i) + ".weight"));
            head->up_b.push_back(ggml_reshape_2d(ctx_w, weight(p + std::to_string(1 + 2 * i) + ".bias"), 1, feat));
        }
        head->smooth_w = ggml_reshape_2d(ctx_w, weight(p + std::to_string(1 + 2 * hp.n_upsample) + ".weight"),
                                         SMOOTH_KERNEL * feat, feat);
        head->smooth_b = weight(p + std::to_string(1 + 2 * hp.n_upsample) + ".bias");
        head->out_w = ggml_reshape_2d(ctx_w, weight(p + std::to_string(3 + 2 * hp.n_upsample) + ".weight"), feat, feat);
        head->out_b = weight(p + std::to_string(3 + 2 * hp.n_upsample) + ".bias");
    }
    for (Mlp * mlp : {&start_mlp, &end_mlp, &presence_mlp}) {
        const std::string p = std::string("model.") +
                              (mlp == &start_mlp ? "start" : mlp == &end_mlp ? "end" : "presence") + "_mlp.";
        *mlp = {weight(p + "0.weight"), weight(p + "0.bias"), weight(p + "2.weight"), weight(p + "2.bias")};
    }
    dia_embedding = weight("text_encoder.embedding.weight");
    dia_norm = weight("text_encoder.norm.weight");
    for (int i = 0; i < dia.n_layers; ++i) {
        const std::string p = "text_encoder.layers." + std::to_string(i) + ".";
        dia_layers.push_back({weight(p + "pre_sa_norm.weight"), weight(p + "self_attention.q_proj.weight"),
                              weight(p + "self_attention.k_proj.weight"), weight(p + "self_attention.v_proj.weight"),
                              weight(p + "self_attention.o_proj.weight"), weight(p + "post_sa_norm.weight"),
                              weight(p + "mlp.gate_up_proj.weight"), weight(p + "mlp.down_proj.weight")});
    }

    // RotaryEmbedding (model.py): theta ** (-arange(0, hd, 2) / hd) in fp32.
    const int fa_hd = dim / hp.n_heads;
    for (int i = 0; i < fa_hd / 2; ++i) {
        const float exponent = -(float) (2 * i) / (float) fa_hd;
        fa_inv_freq.push_back((float) std::pow((double) hp.rope_theta, (double) exponent));
    }
    // transformers' default RoPE: 1 / base ** (arange(0, hd, 2) / hd) in fp32.
    for (int i = 0; i < dia.head_dim / 2; ++i) {
        const float exponent = (float) (2 * i) / (float) dia.head_dim;
        dia_inv_freq.push_back(1.0f / (float) std::pow((double) dia.rope_theta, (double) exponent));
    }

    graph_mem.resize(ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false));
    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc) {
        throw std::runtime_error("failed to create the graph allocator");
    }
}

namespace {

// Graph helpers shared by the Dia and FuseAlign stacks.
struct GraphBuilder {
    ggml_context * ctx;

    // x * cos + rotate_half(x) * sin over ne0 = head_dim, with sin's first half negated (rotate_half = cat(-x2, x1)).
    ggml_tensor * rope(ggml_tensor * x, ggml_tensor * cos, ggml_tensor * sin_signed) const {
        const int64_t half = x->ne[0] / 2;
        ggml_tensor * x1 = ggml_view_3d(ctx, x, half, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
        ggml_tensor * x2 = ggml_view_3d(ctx, x, half, x->ne[1], x->ne[2], x->nb[1], x->nb[2], half * x->nb[0]);
        ggml_tensor * rotated = ggml_concat(ctx, x2, x1, 0);
        return ggml_add(ctx, ggml_mul(ctx, x, cos), ggml_mul(ctx, rotated, sin_signed));
    }

    // Full softmax attention. q, k, v: [head_dim, n_heads, n]. Returns [head_dim * n_heads, n].
    //
    // Fused, with K/V in F16. The upstream form (mul_mat on permuted f32 views, soft_max, mul_mat) missed the CPU
    // GEMM path because the views are not contiguous: on a 27 s clip (2104 tokens) attention was 1.7 s of a 5.4 s
    // forward at 6 threads, and is 0.59 s fused. Every edge stays within 4 ms of the upstream graph.
    ggml_tensor * attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, float scale) const {
        const int64_t hd = q->ne[0], nh = q->ne[1], n = q->ne[2];
        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);                                       // [hd, n, nh]
        ggml_tensor * kp = ggml_cast(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), GGML_TYPE_F16);        // [hd, n, nh]
        ggml_tensor * vp = ggml_cast(ctx, ggml_permute(ctx, v, 0, 2, 1, 3), GGML_TYPE_F16);        // [hd, n, nh]
        ggml_tensor * out = ggml_flash_attn_ext(ctx, qp, kp, vp, nullptr, scale, 0.0f, 0.0f);   // [hd, nh, n]
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        return ggml_reshape_2d(ctx, out, hd * nh, n);
    }
};

}  // namespace

Placement Model::Impl::forward(const float * wav, size_t n_samples, const std::vector<std::string> & words) {
    Placement result;
    if (words.empty()) {
        return result;  // _forward_batch: an item without words gets no forward
    }
    if (n_samples < (size_t) hp.patch_window) {
        throw std::invalid_argument("Waveform must contain at least one complete audio patch");
    }
    // Transcript bytes and word spans (excluding the two-token prefix).
    std::vector<int32_t> ids = {1, 32};
    std::vector<std::pair<int, int>> spans;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i) ids.push_back(32);
        const int start = (int) ids.size() - 2;
        for (unsigned char c : words[i]) ids.push_back(c);
        spans.emplace_back(start, (int) ids.size() - 2);
    }
    const int T = (int) ids.size() - 2;
    if (T > hp.max_text_bytes) {
        throw std::invalid_argument("Use long-form alignment for transcripts over " +
                                    std::to_string(hp.max_text_bytes) + " UTF-8 bytes");
    }
    const int W = (int) words.size();
    const int L = T + 2;
    const int A = (int) ((n_samples - hp.patch_window) / hp.patch_hop + 1);
    const int N = A + T;
    const int dim = hp.dim, nh = hp.n_heads, hd = dim / nh, feat = hp.feat_dim;
    const int D = dia.hidden, dnh = dia.n_heads, dhd = dia.head_dim;

    // FuseAlign._standardize in fp32 (sums accumulated in double, then rounded as torch's fp32 reductions are).
    std::vector<float> standardized(n_samples);
    {
        double sum = 0.0;
        for (size_t i = 0; i < n_samples; ++i) sum += wav[i];
        const float count = (float) n_samples;
        const float mean = (float) sum / count;
        double sq = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            standardized[i] = wav[i] - mean;
            sq += (double) standardized[i] * standardized[i];
        }
        const float denom = std::sqrt((float) sq / count) + STD_EPS;
        for (float & x : standardized) x /= denom;
    }
    std::vector<float> frames((size_t) hp.patch_window * A);
    for (int a = 0; a < A; ++a) {
        std::memcpy(&frames[(size_t) a * hp.patch_window], &standardized[(size_t) a * hp.patch_hop],
                    hp.patch_window * sizeof(float));
    }

    // RoPE tables: cos and sin with the first half negated, ne [head_dim, 1, n].
    auto tables = [](const std::vector<float> & positions, const std::vector<float> & inv_freq, bool round,
                     std::vector<float> & cos, std::vector<float> & sin) {
        const size_t half = inv_freq.size(), hdim = 2 * half;
        cos.resize(positions.size() * hdim);
        sin.resize(positions.size() * hdim);
        for (size_t p = 0; p < positions.size(); ++p) {
            for (size_t i = 0; i < hdim; ++i) {
                const float angle = positions[p] * inv_freq[i % half];
                float c = std::cos(angle), s = std::sin(angle);
                if (round) {
                    c = round_bf16(c);
                    s = round_bf16(s);
                }
                cos[p * hdim + i] = c;
                sin[p * hdim + i] = i < half ? -s : s;
            }
        }
    };
    std::vector<float> positions(L);
    for (int i = 0; i < L; ++i) positions[i] = (float) i;
    std::vector<float> dia_cos, dia_sin, fa_cos, fa_sin;
    tables(positions, dia_inv_freq, true, dia_cos, dia_sin);
    positions.resize(N);
    const float text_scale = (float) A / (float) std::max(T - 1, 1);
    for (int a = 0; a < A; ++a) positions[a] = (float) a;
    for (int j = 0; j < T; ++j) positions[A + j] = (float) j * text_scale;
    tables(positions, fa_inv_freq, autocast, fa_cos, fa_sin);

    // Mean pooling over each word's bytes, ne [T, W].
    std::vector<float> pooling((size_t) T * W, 0.0f);
    for (int w = 0; w < W; ++w) {
        const int n = spans[w].second - spans[w].first;
        float value = 1.0f / (float) std::max(n, 1);
        if (autocast) value = round_bf16(value);
        for (int j = spans[w].first; j < spans[w].second; ++j) pooling[(size_t) w * T + j] = value;
    }

    ggml_init_params gp = {graph_mem.size(), graph_mem.data(), true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) {
        throw std::runtime_error("ggml_init failed for the graph context");
    }
    struct CtxGuard {
        ggml_context * ctx;
        ~CtxGuard() { ggml_free(ctx); }
    } guard{ctx};
    GraphBuilder gb{ctx};
    auto input = [&](ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        ggml_set_input(t);
        return t;
    };
    ggml_tensor * in_ids = input(GGML_TYPE_I32, L, 1, 1);
    ggml_tensor * in_dia_cos = input(GGML_TYPE_F32, dhd, 1, L);
    ggml_tensor * in_dia_sin = input(GGML_TYPE_F32, dhd, 1, L);
    ggml_tensor * in_frames = input(GGML_TYPE_F32, hp.patch_window, A, 1);
    ggml_tensor * in_fa_cos = input(GGML_TYPE_F32, hd, 1, N);
    ggml_tensor * in_fa_sin = input(GGML_TYPE_F32, hd, 1, N);
    ggml_tensor * in_pool = input(GGML_TYPE_F32, T, W, 1);

    // ── Dia text encoder ─────────────────────────────────────────────────────
    auto dia_rms_norm = [&](ggml_tensor * t, ggml_tensor * w) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, t, dia.norm_eps), w);
    };
    ggml_tensor * x = ggml_get_rows(ctx, dia_embedding, in_ids);  // [D, L]
    for (const DiaLayer & layer : dia_layers) {
        ggml_tensor * h = dia_rms_norm(x, layer.pre_norm);
        ggml_tensor * q = gb.rope(ggml_reshape_3d(ctx, ggml_mul_mat(ctx, layer.q, h), dhd, dnh, L), in_dia_cos,
                                  in_dia_sin);
        ggml_tensor * k = gb.rope(ggml_reshape_3d(ctx, ggml_mul_mat(ctx, layer.k, h), dhd, dnh, L), in_dia_cos,
                                  in_dia_sin);
        ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, layer.v, h), dhd, dnh, L);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, layer.o, gb.attention(q, k, v, 1.0f)));  // DiaSelfAttention scaling 1
        h = dia_rms_norm(x, layer.post_norm);
        ggml_tensor * gate_up = ggml_mul_mat(ctx, layer.gate_up, h);  // [2 * inter, L]: gate, then up
        ggml_tensor * gate = ggml_view_2d(ctx, gate_up, dia.inter, L, gate_up->nb[1], 0);
        ggml_tensor * up = ggml_view_2d(ctx, gate_up, dia.inter, L, gate_up->nb[1], dia.inter * gate_up->nb[0]);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, layer.down, ggml_mul(ctx, up, ggml_silu(ctx, gate))));
    }
    // The encoder's bf16 output, without the two-token [S1] prefix.
    x = ggml_cast(ctx, ggml_cast(ctx, dia_rms_norm(x, dia_norm), GGML_TYPE_BF16), GGML_TYPE_F32);
    ggml_tensor * text_feats = ggml_view_2d(ctx, x, D, T, x->nb[1], 2 * x->nb[1]);

    // ── FuseAlign ────────────────────────────────────────────────────────────
    ggml_tensor * audio = ggml_add(ctx, ggml_mul_mat(ctx, audio_patch_w, in_frames), audio_patch_b);  // [dim, A]
    audio = ggml_add(ctx, audio, ggml_view_1d(ctx, mod_emb, dim, 0));
    ggml_tensor * text = ggml_add(ctx, ggml_mul_mat(ctx, text_proj_w, text_feats), text_proj_b);      // [dim, T]
    text = ggml_add(ctx, text, ggml_view_1d(ctx, mod_emb, dim, mod_emb->nb[1]));
    x = ggml_concat(ctx, audio, text, 1);  // [dim, N]
    const float scale = 1.0f / std::sqrt((float) hd);
    for (const Block & block : blocks) {
        ggml_tensor * h = ggml_mul(ctx, ggml_rms_norm(ctx, x, NORM_EPS), block.attn_norm);
        ggml_tensor * qkv = ggml_mul_mat(ctx, block.qkv, h);  // [3 * dim, N]: q, k, v, each [hd, nh]
        auto part = [&](int i) {
            return ggml_view_3d(ctx, qkv, hd, nh, N, hd * qkv->nb[0], qkv->nb[1], (size_t) i * dim * qkv->nb[0]);
        };
        ggml_tensor * q = gb.rope(part(0), in_fa_cos, in_fa_sin);
        ggml_tensor * k = gb.rope(part(1), in_fa_cos, in_fa_sin);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, block.out, gb.attention(q, k, part(2), scale)));
        h = ggml_mul(ctx, ggml_rms_norm(ctx, x, NORM_EPS), block.ffn_norm);
        ggml_tensor * gated = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, block.w1, h)),
                                       ggml_mul_mat(ctx, block.w3, h));
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, block.w2, gated));
    }
    x = ggml_mul(ctx, ggml_rms_norm(ctx, x, NORM_EPS), final_norm);
    ggml_tensor * audio_out = ggml_view_2d(ctx, x, dim, A, x->nb[1], 0);
    ggml_tensor * text_out = ggml_view_2d(ctx, x, dim, T, x->nb[1], (size_t) A * x->nb[1]);

    // Word mean pooling, then the word MLPs.
    ggml_tensor * text_t = ggml_cont(ctx, ggml_transpose(ctx, text_out));  // [T, dim]
    ggml_tensor * word_vectors = ggml_mul_mat(ctx, text_t, in_pool);      // [dim, W]
    auto mlp = [&](const Mlp & m) {
        ggml_tensor * h = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, m.w0, word_vectors), m.b0));
        return ggml_add(ctx, ggml_mul_mat(ctx, m.w2, h), m.b2);
    };
    // RefinementHead. Every conv sees the whole clip, so its per-sample length masking is a no-op here.
    auto refine = [&](const Head & head) {
        ggml_tensor * h = ggml_add(ctx, ggml_mul_mat(ctx, head.in_w, audio_out), head.in_b);  // [feat, A]
        for (int i = 0; i < hp.n_upsample; ++i) {
            if (i) h = ggml_cont(ctx, ggml_transpose(ctx, h));      // [feat, t]
            ggml_tensor * cols = ggml_mul_mat(ctx, head.up_w[i], h);  // [K * feat, t]
            h = ggml_col2im_1d(ctx, cols, UP_STRIDE, feat, UP_PAD);   // [2t, feat]
            h = ggml_gelu_erf(ctx, ggml_add(ctx, h, head.up_b[i]));
        }
        // h: [F, feat] (time along ne0), the layout im2col reads.
        ggml_tensor * cols = ggml_im2col(ctx, ggml_reshape_3d(ctx, head.smooth_w, SMOOTH_KERNEL, feat, feat), h, 1, 0,
                                         SMOOTH_PAD, 0, 1, 0, false, GGML_TYPE_F32);  // [5 * feat, F]
        h = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, head.smooth_w, cols), head.smooth_b));  // [feat, F]
        return ggml_add(ctx, ggml_mul_mat(ctx, head.out_w, h), head.out_b);                            // [feat, F]
    };
    ggml_tensor * start_logits = ggml_mul_mat(ctx, refine(start_head), mlp(start_mlp));  // [F, W]
    ggml_tensor * end_logits = ggml_mul_mat(ctx, refine(end_head), mlp(end_mlp));
    ggml_tensor * start_idx = ggml_argmax(ctx, start_logits);
    ggml_tensor * end_idx = ggml_argmax(ctx, end_logits);
    ggml_tensor * presence = mlp(presence_mlp);  // [1, W]
    for (ggml_tensor * out : {start_idx, end_idx, presence}) ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
    for (ggml_tensor * out : {start_idx, end_idx, presence}) ggml_build_forward_expand(graph, out);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (!ggml_backend_supports_op(backend, node)) {
            throw std::runtime_error(std::string(ggml_backend_name(backend)) + " does not support op " +
                                     ggml_op_desc(node) + " (" + ggml_get_name(node) + ")");
        }
    }
    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        throw std::runtime_error("failed to allocate the compute graph on " + std::string(ggml_backend_name(backend)));
    }
    ggml_backend_tensor_set(in_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(in_dia_cos, dia_cos.data(), 0, dia_cos.size() * sizeof(float));
    ggml_backend_tensor_set(in_dia_sin, dia_sin.data(), 0, dia_sin.size() * sizeof(float));
    ggml_backend_tensor_set(in_frames, frames.data(), 0, frames.size() * sizeof(float));
    ggml_backend_tensor_set(in_fa_cos, fa_cos.data(), 0, fa_cos.size() * sizeof(float));
    ggml_backend_tensor_set(in_fa_sin, fa_sin.data(), 0, fa_sin.size() * sizeof(float));
    ggml_backend_tensor_set(in_pool, pooling.data(), 0, pooling.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("graph compute failed on " + std::string(ggml_backend_name(backend)));
    }

    std::vector<int32_t> start(W), end(W);
    std::vector<float> logits(W);
    ggml_backend_tensor_get(start_idx, start.data(), 0, W * sizeof(int32_t));
    ggml_backend_tensor_get(end_idx, end.data(), 0, W * sizeof(int32_t));
    ggml_backend_tensor_get(presence, logits.data(), 0, W * sizeof(float));
    result.intervals.resize(W);
    result.presence.resize(W);
    for (int w = 0; w < W; ++w) {
        float p = 1.0f / (1.0f + std::exp(-logits[w]));
        if (std::isnan(p)) p = 0.0f;
        result.presence[w] = p;
        if (p >= hp.presence_threshold) {
            const int s = start[w], e = std::max(end[w], start[w]);
            result.intervals[w] = Interval(s / 1000.0, e / 1000.0);
        }
    }
    return result;
}

Model::Model(const std::string & gguf_path, const BackendOptions & options) : impl_(std::make_unique<Impl>()) {
    impl_->init_backend(options);
    impl_->load(gguf_path);
}

Model::~Model() = default;

const HParams & Model::hparams() const { return impl_->hp; }

std::string Model::backend_name() const { return ggml_backend_name(impl_->backend); }

Placement Model::forward(const float * wav, size_t n_samples, const std::vector<std::string> & words) {
    return impl_->forward(wav, n_samples, words);
}

}  // namespace fusealign
