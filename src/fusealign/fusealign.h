// FuseAlign word-level forced aligner on ggml: the public API.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "longform.h"
#include "model.h"

namespace fusealign {

constexpr int SAMPLE_RATE = 16000;

struct Options {
    std::string model_path;     // fusealign-gguf/1 file
    std::string device = "auto";
    int n_threads = 0;
};

// One transcript token; times in seconds. placed is false for the uniform fallback (presence below the threshold,
// audio under one patch, a token no long-form chunk could hold) and for punctuation-only tokens, which are
// zero-width at the previous token's end.
struct WordAlignment {
    std::string word;
    double start = 0.0;
    double end = 0.0;
    float presence = 0.0f;
    bool placed = false;
};

class Aligner {
public:
    explicit Aligner(const Options & options);
    ~Aligner();
    Aligner(const Aligner &) = delete;
    Aligner & operator=(const Aligner &) = delete;

    const Model & model() const;

    // FuseAlignAligner.intervals_batch for one item: 16 kHz mono waveform and its words. Under one patch every word
    // is unplaced; up to max_seconds and max_text_bytes one forward; otherwise long-form, anchored on `anchors` when
    // given, else on the VAD character-time map (Python asr_words=[]).
    Placement intervals(const std::vector<float> & wav16k, const std::vector<std::string> & words,
                        const std::vector<AsrWord> * anchors = nullptr);

    // FuseAlignAligner.align: tokens (non-empty, no whitespace; std::invalid_argument otherwise), any sample rate
    // (resampled to 16 kHz), one WordAlignment per token.
    std::vector<WordAlignment> align(const std::vector<float> & wav, int sample_rate,
                                     const std::vector<std::string> & tokens,
                                     const std::vector<AsrWord> * anchors = nullptr);
    // Same with transcript.split().
    std::vector<WordAlignment> align(const std::vector<float> & wav, int sample_rate, const std::string & transcript,
                                     const std::vector<AsrWord> * anchors = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace fusealign
