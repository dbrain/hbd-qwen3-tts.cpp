// Routing on top of the model: fusealign_hub/aligner.py (intervals_batch, _long, align_batch) for one item, with
// Python's batch_size=1 semantics (one clip or long-form chunk per forward).

#include "fusealign.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "longform.h"
#include "resample.h"
#include "text.h"
#include "vad.h"

namespace fusealign {

struct Aligner::Impl {
    Model model;
    Vad vad;

    explicit Impl(const Options & options)
        : model(options.model_path, BackendOptions{options.device, options.n_threads}), vad(options.model_path) {}

    // ASR anchors (or the VAD character-time map), VAD cores and context stitching: FuseAlignAligner._long.
    Placement long_form(const std::vector<float> & wav, const std::vector<std::string> & words,
                        const std::vector<AsrWord> & anchors) {
        const size_t n_samples = wav.size();
        const double duration = static_cast<double>(n_samples) / SAMPLE_RATE;
        const std::vector<Interval> speech = vad.speech_regions(wav.data(), n_samples);
        const std::vector<Interval> approx = approximate_times(words, anchors, speech, duration);
        // Deviation kept from the Python port: words no chunk can hold are left unplaced instead of failing.
        const std::set<size_t> skipped = unroutable(words, approx);
        std::vector<size_t> keep;
        std::vector<std::string> kept_words;
        std::vector<Interval> kept_approx;
        for (size_t i = 0; i < words.size(); ++i) {
            if (skipped.count(i)) continue;
            keep.push_back(i);
            kept_words.push_back(words[i]);
            kept_approx.push_back(approx[i]);
        }
        std::vector<Chunk> chunks = plan_chunks(kept_words, kept_approx, speech, duration);
        for (Chunk & chunk : chunks) {
            for (size_t & index : chunk.indices) index = keep[index];
        }

        std::vector<std::vector<std::optional<Interval>>> predictions;
        std::vector<std::vector<float>> presences;
        predictions.reserve(chunks.size());
        presences.reserve(chunks.size());
        for (const Chunk & chunk : chunks) {
            // Python round() rounds half to even, as nearbyint does in the default rounding mode.
            const double total = static_cast<double>(n_samples);
            const double start_sample = std::max(0.0, std::min(total, std::nearbyint(chunk.audio.first * SAMPLE_RATE)));
            const double end_sample =
                std::max(start_sample, std::min(total, std::nearbyint(chunk.audio.second * SAMPLE_RATE)));
            const double offset = start_sample / SAMPLE_RATE;
            std::vector<std::string> chunk_words;
            for (size_t index : chunk.indices) chunk_words.push_back(words[index]);
            Placement placed;
            if (!chunk_words.empty()) {
                const size_t lo = static_cast<size_t>(start_sample);
                placed = model.forward(wav.data() + lo, static_cast<size_t>(end_sample) - lo, chunk_words);
            }
            std::vector<std::optional<Interval>> shifted;
            shifted.reserve(placed.intervals.size());
            for (const auto & iv : placed.intervals) {
                if (iv) shifted.emplace_back(Interval{iv->first + offset, iv->second + offset});
                else shifted.emplace_back(std::nullopt);
            }
            predictions.push_back(std::move(shifted));
            presences.push_back(std::move(placed.presence));
        }
        Placement result;
        result.intervals = stitch_predictions(words.size(), chunks, predictions);

        result.presence.assign(words.size(), 0.0f);
        std::vector<bool> found(words.size(), false);
        for (size_t c = 0; c < chunks.size(); ++c) {
            for (size_t j = 0; j < chunks[c].indices.size(); ++j) {
                const size_t index = chunks[c].indices[j];
                const float p = presences[c][j];
                if (result.intervals[index]) {
                    if (!found[index] && predictions[c][j] == result.intervals[index]) {
                        result.presence[index] = p;
                        found[index] = true;
                    }
                } else {
                    result.presence[index] = std::max(result.presence[index], p);
                }
            }
        }
        return result;
    }
};

static void check_tokens(const std::vector<std::string> & tokens) {
    for (const std::string & token : tokens) {
        check_utf8(token);
        if (token.empty() || has_whitespace(token)) {
            throw std::invalid_argument("tokens must be non-empty strings without whitespace");
        }
    }
}

Aligner::Aligner(const Options & options) : impl_(std::make_unique<Impl>(options)) {}

Aligner::~Aligner() = default;

const Model & Aligner::model() const { return impl_->model; }

Placement Aligner::intervals(const std::vector<float> & wav16k, const std::vector<std::string> & words,
                             const std::vector<AsrWord> * anchors) {
    check_tokens(words);
    const size_t n = words.size();
    if (n == 0) return {};
    const HParams & hp = impl_->model.hparams();
    if (wav16k.size() < static_cast<size_t>(hp.patch_window)) {
        return Placement{std::vector<std::optional<Interval>>(n), std::vector<float>(n, 0.0f)};
    }
    size_t text_bytes = n - 1;
    for (const std::string & word : words) text_bytes += word.size();
    if (static_cast<double>(wav16k.size()) <= static_cast<double>(hp.max_seconds) * SAMPLE_RATE &&
        text_bytes <= static_cast<size_t>(hp.max_text_bytes)) {
        return impl_->model.forward(wav16k.data(), wav16k.size(), words);
    }
    static const std::vector<AsrWord> no_anchors;
    return impl_->long_form(wav16k, words, anchors ? *anchors : no_anchors);
}

std::vector<WordAlignment> Aligner::align(const std::vector<float> & wav, int sample_rate,
                                          const std::vector<std::string> & tokens,
                                          const std::vector<AsrWord> * anchors) {
    check_tokens(tokens);
    if (tokens.empty()) return {};
    if (sample_rate <= 0) throw std::invalid_argument("sample_rate must be positive");
    std::vector<float> resampled;
    const std::vector<float> * wav16k = &wav;
    if (sample_rate != SAMPLE_RATE) {
        resampled = resample(wav.data(), wav.size(), sample_rate, SAMPLE_RATE);
        wav16k = &resampled;
    }
    std::vector<size_t> wordy;
    std::vector<std::string> wordy_tokens;
    for (size_t k = 0; k < tokens.size(); ++k) {
        if (is_wordy(tokens[k])) {
            wordy.push_back(k);
            wordy_tokens.push_back(tokens[k]);
        }
    }
    const Placement placed = intervals(*wav16k, wordy_tokens, anchors);
    const double duration = static_cast<double>(wav16k->size()) / SAMPLE_RATE;
    const std::vector<Interval> filled = fill_unplaced(placed.intervals, duration);

    std::vector<WordAlignment> out;
    out.reserve(tokens.size());
    const double first_start = filled.empty() ? 0.0 : filled[0].first;
    std::optional<double> previous_end;
    size_t j = 0;
    for (size_t k = 0; k < tokens.size(); ++k) {
        if (j < wordy.size() && wordy[j] == k) {
            out.push_back({tokens[k], filled[j].first, filled[j].second, placed.presence[j],
                           placed.intervals[j].has_value()});
            previous_end = filled[j].second;
            ++j;
        } else {
            const double t = previous_end ? *previous_end : first_start;
            out.push_back({tokens[k], t, t, 0.0f, false});
            previous_end = t;
        }
    }
    return out;
}

std::vector<WordAlignment> Aligner::align(const std::vector<float> & wav, int sample_rate,
                                          const std::string & transcript, const std::vector<AsrWord> * anchors) {
    check_utf8(transcript);
    return align(wav, sample_rate, split_whitespace(transcript), anchors);
}

}  // namespace fusealign
