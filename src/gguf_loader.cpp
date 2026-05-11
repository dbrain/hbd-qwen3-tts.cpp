#include "gguf_loader.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

namespace qwen3_tts {

namespace {
struct shared_backend_state {
    ggml_backend_t backend = nullptr;
    int32_t ref_count = 0;
};

shared_backend_state & get_shared_backend_state() {
    static shared_backend_state state;
    return state;
}

// Guards both backend pointer and ref_count for concurrent
// init_preferred_backend / release_preferred_backend callers (e.g.
// the C-API can be driven from worker threads).
std::mutex & get_shared_backend_mutex() {
    static std::mutex m;
    return m;
}
}

GGUFLoader::GGUFLoader() = default;

GGUFLoader::~GGUFLoader() {
    close();
}

static ggml_backend_t init_preferred_backend_uncached(const char * component_name,
                                                       std::string * error_msg,
                                                       const char * backend_priority = nullptr) {
    const char * force_cpu = std::getenv("QWEN3_TTS_FORCE_CPU");
    // Build "priority=<low|high>" if a priority hint was passed. The
    // device init parses this; non-CUDA backends ignore the params arg.
    std::string params_buf;
    const char * params = nullptr;
    if (backend_priority && backend_priority[0] != '\0' &&
        std::strcmp(backend_priority, "default") != 0) {
        params_buf = "priority=";
        params_buf += backend_priority;
        params = params_buf.c_str();
    }
    ggml_backend_t backend = nullptr;
    if (!(force_cpu && force_cpu[0] == '1')) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, params);
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, params);
        }
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, params);
        }
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, params);
    }

    if (!backend && error_msg) {
        const char * name = component_name ? component_name : "component";
        *error_msg = "Failed to initialize backend (IGPU/GPU/ACCEL/CPU) for " + std::string(name);
    }
    return backend;
}

ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg,
                                      bool prefer_dedicated,
                                      const char * backend_priority) {
    if (error_msg) error_msg->clear();

    if (prefer_dedicated) {
        // Skip the shared cache: caller wants its own backend (separate
        // ggml-cuda context + streams). release_preferred_backend's else
        // branch will free the returned backend.
        return init_preferred_backend_uncached(component_name, error_msg, backend_priority);
    }

    // First caller's backend_priority sticks — subsequent shared-cache
    // callers inherit whatever priority the talker (or whoever inits
    // first) requested. Documented at the call site so siblings don't
    // assume their hint will apply.
    std::lock_guard<std::mutex> lock(get_shared_backend_mutex());

    auto & shared = get_shared_backend_state();
    if (shared.backend) {
        shared.ref_count++;
        return shared.backend;
    }

    ggml_backend_t backend = init_preferred_backend_uncached(component_name, error_msg, backend_priority);

    if (backend) {
        shared.backend = backend;
        shared.ref_count = 1;
    }

    return backend;
}

void release_preferred_backend(ggml_backend_t backend) {
    if (!backend) {
        return;
    }

    std::lock_guard<std::mutex> lock(get_shared_backend_mutex());

    auto & shared = get_shared_backend_state();
    if (shared.backend == backend) {
        shared.ref_count--;
        if (shared.ref_count <= 0) {
            ggml_backend_free(shared.backend);
            shared.backend = nullptr;
            shared.ref_count = 0;
        }
        return;
    }

    ggml_backend_free(backend);
}

bool GGUFLoader::open(const std::string & path) {
    close();  // Close any previously opened file
    
    file_path_ = path;
    
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx_,
    };
    
    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) {
        error_msg_ = "Failed to open GGUF file: " + path;
        return false;
    }
    
    return true;
}

void GGUFLoader::close() {
    if (ctx_) {
        gguf_free(ctx_);
        ctx_ = nullptr;
    }
    if (meta_ctx_) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
    file_path_.clear();
}

int64_t GGUFLoader::get_n_tensors() const {
    if (!ctx_) return 0;
    return gguf_get_n_tensors(ctx_);
}

const char * GGUFLoader::get_tensor_name(int64_t idx) const {
    if (!ctx_) return nullptr;
    return gguf_get_tensor_name(ctx_, idx);
}

enum ggml_type GGUFLoader::get_tensor_type(int64_t idx) const {
    if (!ctx_) return GGML_TYPE_F32;
    return gguf_get_tensor_type(ctx_, idx);
}

size_t GGUFLoader::get_tensor_offset(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_offset(ctx_, idx);
}

size_t GGUFLoader::get_tensor_size(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_size(ctx_, idx);
}

int32_t GGUFLoader::get_u32(const char * key, int32_t default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return (int32_t)gguf_get_val_u32(ctx_, idx);
}

float GGUFLoader::get_f32(const char * key, float default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return gguf_get_val_f32(ctx_, idx);
}

size_t GGUFLoader::get_data_offset() const {
    if (!ctx_) return 0;
    return gguf_get_data_offset(ctx_);
}

bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    enum ggml_backend_dev_type preferred_backend_type
) {
    // Try the requested device type first; if it doesn't exist (e.g. IGPU on
    // a discrete-only machine), fall back to GPU then CPU. The previous
    // IGPU-then-CPU path silently loaded the WavTokenizer vocoder weights into
    // CPU buffer on systems with only a discrete GPU, which then forced the
    // entire decoder graph to run on CPU.
    ggml_backend_t backend = ggml_backend_init_by_type(preferred_backend_type, nullptr);
    if (!backend && preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_CPU
                 && preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_GPU) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!backend && preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_CPU) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!backend) {
        error_msg = "Failed to initialize backend for GGUF tensor loader";
        return false;
    }

    // Allocate buffer for all tensors
    buffer = ggml_backend_alloc_ctx_tensors(model_ctx, backend);
    if (!buffer) {
        error_msg = "Failed to allocate tensor buffer";
        ggml_backend_free(backend);
        return false;
    }
    // Mark buffer as WEIGHTS so ggml's scheduler treats this as the source-of-
    // backend for ops that consume it (per ggml-backend.cpp:819). Without this
    // tag the scheduler defaults to CPU for ops where it can't infer otherwise.
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    
    // Open file for reading tensor data
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg = "Failed to open file for reading: " + path;
        ggml_backend_free(backend);
        return false;
    }
    
    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;
    std::vector<float> f32_buf;     // F16 → F32 staging for quant
    std::vector<uint8_t> quant_buf; // F32 → Q8_0 staging for upload

    int64_t t_io_us = 0, t_h2d_us = 0, t_quant_us = 0;
    size_t total_h2d_bytes = 0;
    int n_loaded = 0;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);

        auto it = tensors.find(name);
        if (it == tensors.end()) {
            continue;  // Skip tensors not in our map
        }

        struct ggml_tensor * tensor = it->second;
        const enum ggml_type src_type = gguf_get_tensor_type(ctx, i);
        const enum ggml_type dst_type = tensor->type;

        // Source bytes from file are sized by the on-disk type.
        const int64_t n_elem = ggml_nelements(tensor);
        size_t src_nbytes;
        if (src_type == dst_type) {
            src_nbytes = ggml_nbytes(tensor);
        } else {
            // Compute on-disk size from src_type.
            const size_t blck_size  = (size_t) ggml_blck_size(src_type);
            const size_t type_size  = ggml_type_size(src_type);
            src_nbytes = (size_t) n_elem * type_size / blck_size;
        }

        read_buf.resize(src_nbytes);

        if (fseek(f, (long)(data_offset + offset), SEEK_SET) != 0) {
            error_msg = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }

        auto t_io0 = std::chrono::steady_clock::now();
        if (fread(read_buf.data(), 1, src_nbytes, f) != src_nbytes) {
            error_msg = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
        auto t_io1 = std::chrono::steady_clock::now();
        t_io_us += std::chrono::duration_cast<std::chrono::microseconds>(t_io1 - t_io0).count();

        if (src_type == dst_type) {
            ggml_backend_tensor_set(tensor, read_buf.data(), 0, src_nbytes);
            auto t_h2d1 = std::chrono::steady_clock::now();
            t_h2d_us += std::chrono::duration_cast<std::chrono::microseconds>(t_h2d1 - t_io1).count();
            total_h2d_bytes += src_nbytes;
            ++n_loaded;
        } else if (src_type == GGML_TYPE_F16 && dst_type == GGML_TYPE_Q8_0) {
            // Decoder requested Q8_0 for selected mat-mul-only weights to
            // shrink vocoder VRAM (~28 MiB on the V1 12Hz tokenizer). On-disk
            // is F16; convert F16 → F32 → Q8_0 via tmp buffers, then upload
            // the quantized payload.
            f32_buf.resize((size_t) n_elem);
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(read_buf.data()),
                                  f32_buf.data(), n_elem);

            const int64_t n_per_row = tensor->ne[0];
            const int64_t nrows     = n_elem / n_per_row;
            const size_t  q_bytes   = ggml_nbytes(tensor);
            quant_buf.resize(q_bytes);
            const size_t actual = ggml_quantize_chunk(
                GGML_TYPE_Q8_0, f32_buf.data(), quant_buf.data(),
                /*start=*/0, nrows, n_per_row, /*imatrix=*/nullptr);
            if (actual != q_bytes) {
                error_msg = "Q8_0 quantize size mismatch for " + std::string(name);
                fclose(f);
                ggml_backend_free(backend);
                return false;
            }
            auto t_q1 = std::chrono::steady_clock::now();
            t_quant_us += std::chrono::duration_cast<std::chrono::microseconds>(t_q1 - t_io1).count();
            ggml_backend_tensor_set(tensor, quant_buf.data(), 0, q_bytes);
            auto t_h2d1 = std::chrono::steady_clock::now();
            t_h2d_us += std::chrono::duration_cast<std::chrono::microseconds>(t_h2d1 - t_q1).count();
            total_h2d_bytes += q_bytes;
            ++n_loaded;
        } else {
            error_msg = std::string("Unsupported dtype conversion ") +
                        ggml_type_name(src_type) + "->" + ggml_type_name(dst_type) +
                        " for tensor " + name;
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
    }

    fclose(f);
    ggml_backend_free(backend);

    fprintf(stderr, "  [load-bw %s] %d tensors, %.1f MiB H2D: fread=%lld ms, quant=%lld ms, H2D=%lld ms (%.2f GiB/s)\n",
            path.c_str(), n_loaded, total_h2d_bytes / 1048576.0,
            (long long)(t_io_us / 1000), (long long)(t_quant_us / 1000),
            (long long)(t_h2d_us / 1000),
            t_h2d_us > 0 ? (total_h2d_bytes / 1073741824.0) / (t_h2d_us / 1e6) : 0.0);
    return true;
}

void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer) {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

} // namespace qwen3_tts
