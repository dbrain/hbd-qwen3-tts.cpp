// Silero VAD v5 (16 kHz) from the vad.* tensors of a fusealign GGUF.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fusealign {

class Vad {
public:
    explicit Vad(const std::string & gguf_path);  // throws std::runtime_error
    ~Vad();
    Vad(const Vad &) = delete;
    Vad & operator=(const Vad &) = delete;

    // Per-512-sample-frame speech probabilities of a 16 kHz waveform (fusealign_hub.vad.silero_probs): zero context
    // before the first frame, the last frame zero-padded. CPU, float32.
    std::vector<float> probs(const float * wav, size_t n_samples) const;

    // fusealign_hub.vad.speech_regions: silero_vad.get_speech_timestamps_from_probs(probs, sampling_rate=16000,
    // return_seconds=True, audio_length_samples=n) with its default parameters; (start, end) in seconds, rounded as
    // silero rounds them.
    std::vector<std::pair<double, double>> speech_regions(const float * wav, size_t n_samples) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace fusealign
