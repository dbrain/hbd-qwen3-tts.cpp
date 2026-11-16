// FuseAlign + Dia text encoder on a ggml backend: GGUF loading, graphs, decoding.
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fusealign {

using Interval = std::pair<double, double>;  // seconds

// Per word: the model's interval (nullopt when presence < threshold) and its presence probability.
struct Placement {
    std::vector<std::optional<Interval>> intervals;
    std::vector<float> presence;
};

struct HParams {
    int dim = 0, n_layers = 0, n_heads = 0, ffn_dim = 0, feat_dim = 0;
    int patch_window = 0, patch_hop = 0, n_upsample = 0, text_dim = 0;
    float rope_theta = 0.0f;
    float presence_threshold = 0.1f;
    int max_text_bytes = 1022;
    float max_seconds = 30.0f;
};

struct BackendOptions {
    std::string device = "auto";  // "auto" (first CUDA device, else CPU) | "cpu" | "cuda" | "cuda:N"
    int n_threads = 0;            // CPU threads; 0 = hardware concurrency
};

class Model {
public:
    // Loads a fusealign-gguf/1 file (model.* and text_encoder.* tensors) onto the chosen backend.
    // Throws std::runtime_error on a missing file, wrong format, or unavailable device.
    Model(const std::string & gguf_path, const BackendOptions & options);
    ~Model();
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    const HParams & hparams() const;
    std::string backend_name() const;

    // One forward over a 16 kHz mono clip and its words: Python FuseAlignAligner._forward_group for a one-item
    // batch. Requires n_samples >= patch_window and the words joined by single spaces to be at most max_text_bytes
    // UTF-8 bytes (std::invalid_argument otherwise). Words are non-empty and carry no whitespace.
    Placement forward(const float * wav, size_t n_samples, const std::vector<std::string> & words);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace fusealign
