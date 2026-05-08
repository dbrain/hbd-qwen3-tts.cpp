#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

namespace qwen3_tts {

// Generic GGUF model loader class
// This is a simplified loader that can be extended for specific model types
class GGUFLoader {
public:
    GGUFLoader();
    ~GGUFLoader();
    
    // Open GGUF file and parse metadata
    bool open(const std::string & path);
    
    // Close file and free resources
    void close();
    
    // Get error message if operation failed
    const std::string & get_error() const { return error_msg_; }
    
    // Get number of tensors in file
    int64_t get_n_tensors() const;
    
    // Get tensor name by index
    const char * get_tensor_name(int64_t idx) const;
    
    // Get tensor type by index
    enum ggml_type get_tensor_type(int64_t idx) const;
    
    // Get tensor offset by index
    size_t get_tensor_offset(int64_t idx) const;
    
    // Get tensor size by index
    size_t get_tensor_size(int64_t idx) const;
    
    // Get metadata value (returns -1 if not found)
    int32_t get_u32(const char * key, int32_t default_val = 0) const;
    float get_f32(const char * key, float default_val = 0.0f) const;
    
    // Get data offset (start of tensor data in file)
    size_t get_data_offset() const;
    
    // Get GGUF context (for advanced usage)
    struct gguf_context * get_ctx() const { return ctx_; }
    
    // Get metadata context
    struct ggml_context * get_meta_ctx() const { return meta_ctx_; }
    
protected:
    struct gguf_context * ctx_ = nullptr;
    struct ggml_context * meta_ctx_ = nullptr;
    std::string error_msg_;
    std::string file_path_;
};

// Helper function to allocate and load tensor data from GGUF file.
// preferred_backend_type is required (no default) — silently defaulting to
// CPU caused the speaker encoder to ride the op_offload churn path with the
// reviewer flagging it as MED-3. Callers must now pick explicitly so the
// CPU-vs-GPU intent is visible at the call site.
bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    enum ggml_backend_dev_type preferred_backend_type
);

// Helper to initialize backend with GPU preference and CPU fallback.
// When prefer_dedicated=true, bypasses the process-wide shared backend
// cache and creates a fresh ggml_backend instance (its own ggml-cuda
// context + streams). Required when a component must avoid stream-capture
// conflicts with another component on the same device — e.g. the vocoder
// in async-dispatch mode while the talker is in CUDA-graph capture mode.
// release_preferred_backend handles both cached and dedicated backends.
ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg,
                                      bool prefer_dedicated = false);
void release_preferred_backend(ggml_backend_t backend);

// Helper function to free model resources
void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer);

} // namespace qwen3_tts
