#include "breeze_weights.h"
#include "gguf_loader.h"

#include <cstdarg>
#include <cstring>
#include <set>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace breeze {

using qwen3_tts::GGUFLoader;
using qwen3_tts::load_tensor_data_from_file;
using qwen3_tts::init_preferred_backend;
using qwen3_tts::release_preferred_backend;

void BreezeWeights::unload() {
    if (buffer_)      { ggml_backend_buffer_free(buffer_); buffer_ = nullptr; }
    if (buffer_cpu_)  { ggml_backend_buffer_free(buffer_cpu_); buffer_cpu_ = nullptr; }
    if (ctx_)         { ggml_free(ctx_); ctx_ = nullptr; }
    if (ctx_cpu_)     { ggml_free(ctx_cpu_); ctx_cpu_ = nullptr; }
    if (backend_)     { release_preferred_backend(backend_); backend_ = nullptr; }
    if (backend_cpu_) { ggml_backend_free(backend_cpu_); backend_cpu_ = nullptr; }
    tensors_.clear(); gpu_tensors_.clear(); cpu_tensors_.clear();
    te_on_cpu_ = false;
}

struct ggml_tensor * BreezeWeights::getf(const char * fmt, ...) const {
    char nm[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(nm, sizeof(nm), fmt, ap);
    va_end(ap);
    return get(nm);
}

struct ggml_tensor * BreezeWeights::fuse_rows(struct ggml_tensor * a, struct ggml_tensor * b,
                                              const char * name) {
    if (!ctx_ || !a || !b) return nullptr;
    const bool ok = a->type == b->type &&
                    a->ne[0] == b->ne[0] &&
                    a->ne[2] == 1 && a->ne[3] == 1 && b->ne[2] == 1 && b->ne[3] == 1 &&
                    a->nb[1] == b->nb[1] &&
                    ggml_is_contiguous(a) && ggml_is_contiguous(b) &&
                    a->buffer && a->buffer == b->buffer &&
                    (char *) b->data == (char *) a->data + ggml_nbytes(a);
    if (!ok) {
        // Not fatal -- the caller keeps the unfused path -- but it silently
        // costs ~7% of decode, so say so rather than regressing quietly.
        static int warned = 0;
        if (warned++ < 4)
            fprintf(stderr, "  Breeze: %s not fused (%s / %s are not adjacent in the "
                    "weight buffer); falling back to separate matmuls\n",
                    name, ggml_get_name(a), ggml_get_name(b));
        return nullptr;
    }

    struct ggml_tensor * t = ggml_new_tensor_2d(ctx_, a->type, a->ne[0], a->ne[1] + b->ne[1]);

    if (!t) return nullptr;
    t->data   = a->data;
    t->buffer = a->buffer;
    ggml_set_name(t, name);
    tensors_[name] = t;
    return t;
}

bool BreezeWeights::load(const std::string & path) {
    unload();

    GGUFLoader ld;
    if (!ld.open(path)) { error_msg_ = ld.get_error(); return false; }
    gguf_context * gc = ld.get_ctx();
    ggml_context * meta = ld.get_meta_ctx();

    if (!load_breeze_config(gc, cfg_, error_msg_)) return false;
    if (!tok_.load_from_gguf(gc)) {
        error_msg_ = "tokenizer: " + tok_.get_error();
        return false;
    }

    backend_ = init_preferred_backend("Breeze", &error_msg_, false, nullptr);
    if (!backend_) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
    dev_type_ = dev ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_CPU;
    if (dev_type_ != GGML_BACKEND_DEVICE_TYPE_CPU) {
        backend_cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu_) { error_msg_ = "cpu fallback init failed"; return false; }
    }

    const int64_t nt = ld.get_n_tensors();
    struct ggml_init_params p = { ggml_tensor_overhead() * (size_t) (nt + 128), nullptr, true };
    ctx_ = ggml_init(p);
    if (!ctx_) { error_msg_ = "ggml_init failed"; return false; }
    ctx_cpu_ = ggml_init(p);
    if (!ctx_cpu_) { error_msg_ = "ggml_init (cpu) failed"; return false; }
    // ggml_backend_alloc_ctx_tensors lays tensors out in ctx-creation order, and
    // load_tensor_data_from_file fills them by name, so the creation order is a
    // free choice of buffer layout. Emit attn_q/attn_k/attn_v back-to-back and
    // ffn_gate/ffn_up back-to-back so fuse_rows() can span them with a plain
    // descriptor -- one MMVQ and one q8_1 activation quantisation instead of
    // three (QKV) / two (gate+up), at zero extra VRAM and bit-identical output.
    // The GGUF's own order is alphabetical within a block, which separates them.
    {
        std::vector<std::string> order;
        std::set<std::string> done;
        order.reserve((size_t) nt);
        auto push = [&](const std::string & nm) {
            if (!done.count(nm) && ggml_get_tensor(meta, nm.c_str())) {
                order.push_back(nm); done.insert(nm);
            }
        };
        // Hitting ANY member of a group emits the whole group, in group order --
        // the GGUF lists them alphabetically (attn_k, attn_output, attn_q,
        // attn_v), so keying off the first member alone would not reorder them.
        static const char * const kGroups[][3] = {
            { ".attn_q.weight",   ".attn_k.weight", ".attn_v.weight" },
            { ".ffn_gate.weight", ".ffn_up.weight", nullptr },
        };
        for (int64_t i = 0; i < nt; ++i) {
            const std::string name = ld.get_tensor_name(i);
            if (!ggml_get_tensor(meta, name.c_str())) continue;
            bool grouped = false;
            for (const auto & grp : kGroups) {
                for (int j = 0; j < 3 && grp[j]; ++j) {
                    const size_t sfx = strlen(grp[j]);
                    if (name.size() <= sfx || name.compare(name.size() - sfx, sfx, grp[j]) != 0)
                        continue;
                    const std::string stem = name.substr(0, name.size() - sfx);
                    for (int m = 0; m < 3 && grp[m]; ++m) push(stem + grp[m]);
                    grouped = true;
                    break;
                }
                if (grouped) break;
            }
            if (!grouped) push(name);
        }
        // BREEZE_TEXT_ENC_CPU=1 keeps the text encoder's weights in host RAM.
        // It is 40% of the weight bytes for ~1% of the runtime -- it runs once
        // per request, not once per frame -- so on a shared card that is a very
        // cheap ~926 MiB. No change is needed in breeze_text_enc.cpp: its
        // scheduler already lists the CPU backend, and ggml-sched places an op
        // on the backend its weights live on.
        const char * te_cpu_env = std::getenv("BREEZE_TEXT_ENC_CPU");
        te_on_cpu_ = backend_cpu_ && te_cpu_env && *te_cpu_env && *te_cpu_env != '0';

        for (const auto & name : order) {
            const bool to_cpu = te_on_cpu_ && name.rfind("text_enc.", 0) == 0;
            ggml_context * dst = to_cpu ? ctx_cpu_ : ctx_;
            struct ggml_tensor * t = ggml_dup_tensor(dst, ggml_get_tensor(meta, name.c_str()));
            ggml_set_name(t, name.c_str());
            tensors_[name] = t;
            (to_cpu ? cpu_tensors_ : gpu_tensors_)[name] = t;
        }
    }
    if (!load_tensor_data_from_file(path, gc, ctx_, gpu_tensors_, buffer_, error_msg_, dev_type_))
        return false;
    if (te_on_cpu_ && !cpu_tensors_.empty()) {
        if (!load_tensor_data_from_file(path, gc, ctx_cpu_, cpu_tensors_, buffer_cpu_, error_msg_,
                                        GGML_BACKEND_DEVICE_TYPE_CPU))
            return false;
    }

    fprintf(stderr, "  Breeze weights: %s  %lld tensors, %.1f MiB on %s\n",
            path.c_str(), (long long) nt,
            ggml_backend_buffer_get_size(buffer_) / (1024.0 * 1024.0),
            dev ? ggml_backend_dev_name(dev) : "?");
    if (te_on_cpu_ && buffer_cpu_) {
        fprintf(stderr, "  Breeze weights: text encoder on CPU, %.1f MiB moved off the device\n",
                ggml_backend_buffer_get_size(buffer_cpu_) / (1024.0 * 1024.0));
    }
    return true;
}

} // namespace breeze
