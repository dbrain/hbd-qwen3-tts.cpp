#include "breeze_weights.h"
#include "gguf_loader.h"

#include <cstdarg>
#include <cstdio>

namespace breeze {

using qwen3_tts::GGUFLoader;
using qwen3_tts::load_tensor_data_from_file;
using qwen3_tts::init_preferred_backend;
using qwen3_tts::release_preferred_backend;

void BreezeWeights::unload() {
    if (buffer_)      { ggml_backend_buffer_free(buffer_); buffer_ = nullptr; }
    if (ctx_)         { ggml_free(ctx_); ctx_ = nullptr; }
    if (backend_)     { release_preferred_backend(backend_); backend_ = nullptr; }
    if (backend_cpu_) { ggml_backend_free(backend_cpu_); backend_cpu_ = nullptr; }
    tensors_.clear();
}

struct ggml_tensor * BreezeWeights::getf(const char * fmt, ...) const {
    char nm[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(nm, sizeof(nm), fmt, ap);
    va_end(ap);
    return get(nm);
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
    struct ggml_init_params p = { ggml_tensor_overhead() * (size_t) (nt + 16), nullptr, true };
    ctx_ = ggml_init(p);
    if (!ctx_) { error_msg_ = "ggml_init failed"; return false; }
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = ld.get_tensor_name(i);
        struct ggml_tensor * mt = ggml_get_tensor(meta, name);
        if (!mt) continue;
        struct ggml_tensor * t = ggml_dup_tensor(ctx_, mt);
        ggml_set_name(t, name);
        tensors_[name] = t;
    }
    if (!load_tensor_data_from_file(path, gc, ctx_, tensors_, buffer_, error_msg_, dev_type_))
        return false;

    fprintf(stderr, "  Breeze weights: %s  %lld tensors, %.1f MiB on %s\n",
            path.c_str(), (long long) nt,
            ggml_backend_buffer_get_size(buffer_) / (1024.0 * 1024.0),
            dev ? ggml_backend_dev_name(dev) : "?");
    return true;
}

} // namespace breeze
