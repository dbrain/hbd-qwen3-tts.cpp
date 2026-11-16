// Long-form routing (FuseAlign paper, Appendix G.3): a port of fusealign_hub/longform.py. Same names, same semantics;
// Python ValueError becomes std::invalid_argument.
#pragma once

#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace fusealign {

using Interval = std::pair<double, double>;  // seconds

constexpr int MAX_TEXT_BYTES = 1022;
constexpr double CORE_SECONDS = 26.0;
constexpr double CONTEXT_SECONDS = 2.0;

// One approximate-time anchor (Whisper-style word). Entries without a start or end are ignored, as in Python.
struct AsrWord {
    std::string word;
    std::optional<double> start;
    std::optional<double> end;
};

struct Chunk {
    Interval core;
    Interval audio;
    std::vector<size_t> indices;  // original word indices
};

// approximate_times(words, asr_words, speech_regions, duration). The ASR match key is Python's
// re.sub(r"[^0-9a-z]", "", word.lower()), and the matcher is difflib.SequenceMatcher(autojunk=False)'s
// get_matching_blocks, ported exactly.
std::vector<Interval> approximate_times(const std::vector<std::string> & words, const std::vector<AsrWord> & asr_words,
                                        const std::vector<Interval> & speech_regions, double duration);

// unroutable(words, approx): indices of the words no chunk can hold.
std::set<size_t> unroutable(const std::vector<std::string> & words, const std::vector<Interval> & approx);

// plan_chunks(words, approx, speech_regions, duration).
std::vector<Chunk> plan_chunks(const std::vector<std::string> & words, const std::vector<Interval> & approx,
                               const std::vector<Interval> & speech_regions, double duration);

// stitch_predictions(n_words, chunks, predictions): predictions[c][j] is chunk c's interval for chunks[c].indices[j],
// already offset to recording time.
std::vector<std::optional<Interval>> stitch_predictions(size_t n_words, const std::vector<Chunk> & chunks,
                                                        const std::vector<std::vector<std::optional<Interval>>> & predictions);

// fill_unplaced(intervals, duration).
std::vector<Interval> fill_unplaced(const std::vector<std::optional<Interval>> & intervals, double duration);

}  // namespace fusealign
