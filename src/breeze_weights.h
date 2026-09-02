#pragma once

// One GGUF, four models. BreezeWeights opens breeze-tts-2-*.gguf once, builds a
// single ggml_context holding every tensor, and uploads them into ONE backend
// buffer. The text encoder / backbone / depth decoder / codec then just borrow
// tensors by name -- no four-way re-open, no four weight buffers, and the whole
// checkpoint lands on one device (which is what the koblem GPU gate wants).

#include "breeze_model.h"
#include "breeze_tokenizer.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <map>
#include <string>

namespace breeze {

class BreezeWeights {
public:
    ~BreezeWeights() { unload(); }

    // `gpu_uuid` empty = default device order (IGPU -> GPU -> ACCEL -> CPU).
    bool load(const std::string & gguf_path);
    void unload();

    struct ggml_tensor * get(const std::string & name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : it->second;
    }
    // printf-style convenience for the blk.%d.* names.
    struct ggml_tensor * getf(const char * fmt, ...) const;

    const breeze_config & cfg() const { return cfg_; }
    const Tokenizer & tok() const { return tok_; }
    ggml_backend_t backend() const { return backend_; }
    ggml_backend_t backend_cpu() const { return backend_cpu_; }
    enum ggml_backend_dev_type dev_type() const { return dev_type_; }
    size_t weights_bytes() const {
        return buffer_ ? ggml_backend_buffer_get_size(buffer_) : 0;
    }
    const std::string & get_error() const { return error_msg_; }

private:
    breeze_config cfg_;
    Tokenizer     tok_;
    std::map<std::string, struct ggml_tensor *> tensors_;
    struct ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t backend_cpu_ = nullptr;
    enum ggml_backend_dev_type dev_type_ = GGML_BACKEND_DEVICE_TYPE_CPU;
    std::string error_msg_;
};

} // namespace breeze
