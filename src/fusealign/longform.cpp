// Port of fusealign_hub/longform.py. Floating-point expressions keep Python's operand order so the results are the
// same doubles; max/min follow Python's builtins (the first argument wins ties), bisect_left/right are
// std::lower_bound/upper_bound.
#include "longform.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "text.h"

namespace fusealign {

namespace {

double py_max(double a, double b) { return b > a ? b : a; }
double py_min(double a, double b) { return b < a ? b : a; }

size_t bisect_left(const std::vector<double> & v, double x, size_t lo, size_t hi) {
    return static_cast<size_t>(std::lower_bound(v.begin() + lo, v.begin() + hi, x) - v.begin());
}
size_t bisect_left(const std::vector<double> & v, double x) { return bisect_left(v, x, 0, v.size()); }
size_t bisect_right(const std::vector<double> & v, double x) {
    return static_cast<size_t>(std::upper_bound(v.begin(), v.end(), x) - v.begin());
}

// difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks() over interned elements. Without isjunk
// and autojunk there is no junk and no popular set, so b2j holds every element of b.
class SequenceMatcher {
public:
    struct Match {
        size_t a, b, size;
    };

    SequenceMatcher(const std::vector<int> & a, const std::vector<int> & b) : a_(a), b_(b) {
        for (size_t j = 0; j < b_.size(); ++j) {
            b2j_[b_[j]].push_back(j);
        }
        len_[0].assign(b_.size(), 0);
        len_[1].assign(b_.size(), 0);
        stamp_[0].assign(b_.size(), 0);
        stamp_[1].assign(b_.size(), 0);
    }

    std::vector<Match> get_matching_blocks() {
        const size_t la = a_.size(), lb = b_.size();
        std::vector<std::tuple<size_t, size_t, size_t, size_t>> queue = {{0, la, 0, lb}};
        std::vector<Match> blocks;
        while (!queue.empty()) {
            const auto [alo, ahi, blo, bhi] = queue.back();
            queue.pop_back();
            const Match x = find_longest_match(alo, ahi, blo, bhi);
            const size_t i = x.a, j = x.b, k = x.size;
            if (k) {
                blocks.push_back(x);
                if (alo < i && blo < j) {
                    queue.emplace_back(alo, i, blo, j);
                }
                if (i + k < ahi && j + k < bhi) {
                    queue.emplace_back(i + k, ahi, j + k, bhi);
                }
            }
        }
        std::sort(blocks.begin(), blocks.end(), [](const Match & l, const Match & r) {
            return std::tie(l.a, l.b, l.size) < std::tie(r.a, r.b, r.size);
        });
        size_t i1 = 0, j1 = 0, k1 = 0;
        std::vector<Match> non_adjacent;
        for (const Match & m : blocks) {
            if (i1 + k1 == m.a && j1 + k1 == m.b) {
                k1 += m.size;
            } else {
                if (k1) {
                    non_adjacent.push_back({i1, j1, k1});
                }
                i1 = m.a;
                j1 = m.b;
                k1 = m.size;
            }
        }
        if (k1) {
            non_adjacent.push_back({i1, j1, k1});
        }
        non_adjacent.push_back({la, lb, 0});
        return non_adjacent;
    }

private:
    Match find_longest_match(size_t alo, size_t ahi, size_t blo, size_t bhi) {
        size_t besti = alo, bestj = blo, bestsize = 0;
        // j2len / newj2len dicts as two stamped rows: an entry is present iff its stamp is the row's.
        stamp_counter_ += 2;  // nothing from an earlier call matches this call's first "previous row"
        int prev = 0;
        uint64_t prev_stamp = stamp_counter_ - 1;
        for (size_t i = alo; i < ahi; ++i) {
            const int cur = 1 - prev;
            const uint64_t cur_stamp = prev_stamp + 1;
            const auto found = b2j_.find(a_[i]);
            if (found != b2j_.end()) {
                const std::vector<size_t> & js = found->second;
                for (auto it = std::lower_bound(js.begin(), js.end(), blo); it != js.end(); ++it) {
                    const size_t j = *it;
                    if (j >= bhi) {
                        break;
                    }
                    const size_t before = j > 0 && stamp_[prev][j - 1] == prev_stamp ? len_[prev][j - 1] : 0;
                    const size_t k = before + 1;
                    len_[cur][j] = k;
                    stamp_[cur][j] = cur_stamp;
                    if (k > bestsize) {
                        besti = i - k + 1;
                        bestj = j - k + 1;
                        bestsize = k;
                    }
                }
            }
            prev = cur;
            prev_stamp = cur_stamp;
        }
        stamp_counter_ = prev_stamp + 1;
        while (besti > alo && bestj > blo && a_[besti - 1] == b_[bestj - 1]) {
            --besti;
            --bestj;
            ++bestsize;
        }
        while (besti + bestsize < ahi && bestj + bestsize < bhi && a_[besti + bestsize] == b_[bestj + bestsize]) {
            ++bestsize;
        }
        // The junk-extension loops of difflib never run: bjunk is empty.
        return {besti, bestj, bestsize};
    }

    const std::vector<int> & a_;
    const std::vector<int> & b_;
    std::unordered_map<int, std::vector<size_t>> b2j_;
    std::vector<size_t> len_[2];
    std::vector<uint64_t> stamp_[2];
    uint64_t stamp_counter_ = 1;
};

// re.sub(r"[^0-9a-z]", "", word.lower())
std::string match_key(const std::string & word) {
    std::string key;
    for (const char32_t cp : decode_utf8(word)) {
        const char c = ascii_key(cp);
        if (c) {
            key.push_back(c);
        }
    }
    return key;
}

bool oversized_start_group(const std::vector<std::string> & words, const std::vector<Interval> & approx) {
    bool has_previous = false;
    double previous = 0.0;
    size_t size = 0;
    const size_t n = std::min(words.size(), approx.size());
    for (size_t i = 0; i < n; ++i) {
        const double start = approx[i].first;
        size = has_previous && start == previous ? size + 1 : 0;
        size += words[i].size();
        if (size > static_cast<size_t>(MAX_TEXT_BYTES)) {
            return true;
        }
        previous = start;
        has_previous = true;
    }
    return false;
}

std::vector<Interval> speech_intervals(const std::vector<Interval> & speech_regions, double duration) {
    std::vector<Interval> intervals;
    for (const auto & region : speech_regions) {
        const double start = py_max(0.0, region.first), end = py_min(duration, region.second);
        if (end > start) {
            intervals.emplace_back(start, end);
        }
    }
    std::sort(intervals.begin(), intervals.end());
    std::vector<Interval> merged;
    for (const auto & [start, end] : intervals) {
        if (!merged.empty() && start <= merged.back().second) {
            merged.back() = {merged.back().first, py_max(merged.back().second, end)};
        } else {
            merged.emplace_back(start, end);
        }
    }
    return merged;
}

std::vector<Interval> speech_fallback(const std::vector<std::string> & words, const std::vector<Interval> & speech_regions,
                                      double duration) {
    const std::vector<Interval> regions = speech_intervals(speech_regions, duration);
    if (regions.empty()) {
        return std::vector<Interval>(words.size(), Interval{0.0, 0.0});
    }
    std::vector<double> cumulative = {0.0};
    for (const auto & [start, end] : regions) {
        cumulative.push_back(cumulative.back() + end - start);
    }
    std::vector<size_t> lengths;  // len(word): code points
    size_t total_chars = 0;
    for (const auto & word : words) {
        lengths.push_back(decode_utf8(word).size());
        total_chars += lengths.back();
    }
    if (!total_chars) {
        lengths.assign(words.size(), 1);
        total_chars = words.size();
    }

    // At a speech-region seam, starts belong to the next region and ends to the preceding one, so a word ending
    // at a seam excludes silence.
    const auto recording_time = [&](double speech_time, bool is_end) {
        const size_t found = is_end ? bisect_left(cumulative, speech_time) : bisect_right(cumulative, speech_time);
        const int64_t last = static_cast<int64_t>(regions.size()) - 1;
        const size_t index = static_cast<size_t>(std::max<int64_t>(0, std::min<int64_t>(last, static_cast<int64_t>(found) - 1)));
        return regions[index].first + speech_time - cumulative[index];
    };

    std::vector<Interval> times;
    times.reserve(words.size());
    size_t chars = 0;
    for (const size_t length : lengths) {
        const double start = recording_time(cumulative.back() * static_cast<double>(chars) / static_cast<double>(total_chars), false);
        chars += length;
        const double end = recording_time(cumulative.back() * static_cast<double>(chars) / static_cast<double>(total_chars), true);
        times.emplace_back(start, py_max(start, end));
    }
    return times;
}

struct Tiers {
    std::vector<double> lists[3];  // safe silences, silences, word gaps
};

Tiers cut_tiers(const std::vector<Interval> & approx, const std::vector<Interval> & speech_regions, double duration) {
    std::vector<double> silences;
    double previous_end = 0.0;
    for (const auto & [start, end] : speech_intervals(speech_regions, duration)) {
        if (start - previous_end >= 0.5) {
            silences.push_back((start + previous_end) / 2.0);
        }
        previous_end = end;
    }
    if (duration - previous_end >= 0.5) {
        silences.push_back((duration + previous_end) / 2.0);
    }

    // Test silence midpoints against the union of approximate word intervals. Keeping cumulative maximum ends also
    // handles overlapping word maps.
    std::vector<Interval> ordered = approx;
    std::sort(ordered.begin(), ordered.end());
    std::vector<double> starts, furthest_ends;
    double furthest = -std::numeric_limits<double>::infinity();
    for (const auto & [start, end] : ordered) {
        starts.push_back(start);
        furthest = py_max(furthest, end);
        furthest_ends.push_back(furthest);
    }
    std::vector<double> safe_silences;
    for (const double point : silences) {
        const size_t right = bisect_right(starts, point);
        if (right == 0 || furthest_ends[right - 1] < point) {
            safe_silences.push_back(point);
        }
    }
    // sorted(set(...)): a std::set keeps the first of equal values (0.0 / -0.0), as Python's set does.
    std::set<double> gaps;
    for (size_t i = 0; i + 1 < approx.size(); ++i) {
        gaps.insert((approx[i].second + approx[i + 1].first) / 2.0);
    }
    Tiers tiers;
    tiers.lists[0] = std::move(safe_silences);
    tiers.lists[1] = std::move(silences);
    tiers.lists[2].assign(gaps.begin(), gaps.end());
    return tiers;
}

}  // namespace

std::vector<Interval> approximate_times(const std::vector<std::string> & words, const std::vector<AsrWord> & asr_words,
                                        const std::vector<Interval> & speech_regions, double duration) {
    if (words.empty()) {
        return {};
    }
    std::vector<const AsrWord *> asr;
    for (const auto & word : asr_words) {
        if (word.start && word.end) {
            asr.push_back(&word);
        }
    }
    std::unordered_map<std::string, int> ids;
    const auto intern = [&](const std::string & word) {
        return ids.emplace(match_key(word), static_cast<int>(ids.size())).first->second;
    };
    std::vector<int> a, b;
    a.reserve(words.size());
    b.reserve(asr.size());
    for (const auto & word : words) {
        a.push_back(intern(word));
    }
    for (const AsrWord * word : asr) {
        b.push_back(intern(word->word));
    }
    std::vector<std::pair<size_t, size_t>> candidates;
    for (const auto & m : SequenceMatcher(a, b).get_matching_blocks()) {
        for (size_t offset = 0; offset < m.size; ++offset) {
            candidates.emplace_back(m.a + offset, m.b + offset);
        }
    }
    struct Anchor {
        size_t target;
        double start, end;
    };
    std::vector<Anchor> anchors;
    for (const auto & [target, source] : candidates) {
        const double start = *asr[source]->start, end = *asr[source]->end;
        if (anchors.empty() ||
            start - anchors.back().end <=
                10.0 + 0.5 * static_cast<double>(static_cast<int64_t>(target) - static_cast<int64_t>(anchors.back().target) - 1)) {
            anchors.push_back({target, start, end});
        }
    }
    const size_t minimum = words.size() <= 2 ? 1 : 2;
    if (anchors.size() < minimum || static_cast<double>(anchors.size()) < 0.5 * static_cast<double>(candidates.size())) {
        return speech_fallback(words, speech_regions, duration);
    }

    std::vector<Interval> times(words.size());
    for (const auto & anchor : anchors) {
        times[anchor.target] = {anchor.start, anchor.end};
    }
    const auto spread = [&](int64_t lo, int64_t hi, double start, double end) {
        const int64_t count = hi - lo;
        for (int64_t offset = 0; offset < count; ++offset) {
            times[static_cast<size_t>(lo + offset)] = {
                start + (end - start) * static_cast<double>(offset) / static_cast<double>(count),
                start + (end - start) * static_cast<double>(offset + 1) / static_cast<double>(count),
            };
        }
    };
    const Anchor & first = anchors.front();
    const Anchor & last = anchors.back();
    spread(0, static_cast<int64_t>(first.target), *asr.front()->start, first.start);
    for (size_t k = 0; k + 1 < anchors.size(); ++k) {
        spread(static_cast<int64_t>(anchors[k].target) + 1, static_cast<int64_t>(anchors[k + 1].target), anchors[k].end,
               anchors[k + 1].start);
    }
    spread(static_cast<int64_t>(last.target) + 1, static_cast<int64_t>(words.size()), last.end, *asr.back()->end);
    double previous_start = -std::numeric_limits<double>::infinity();
    std::vector<Interval> result;
    result.reserve(words.size());
    for (const auto & [time_start, end] : times) {
        const double start = py_max(time_start, previous_start);
        result.emplace_back(start, py_max(start, end));
        previous_start = start;
    }
    if (oversized_start_group(words, result)) {
        // Time cuts cannot separate words that share one approximate start; a run of them over
        // MAX_TEXT_BYTES falls back to the VAD character-time map.
        return speech_fallback(words, speech_regions, duration);
    }
    return result;
}

std::set<size_t> unroutable(const std::vector<std::string> & words, const std::vector<Interval> & approx) {
    const size_t limit = static_cast<size_t>(MAX_TEXT_BYTES);
    std::set<size_t> out;
    for (size_t i = 0; i < words.size(); ++i) {
        if (words[i].size() > limit) {
            out.insert(i);
        }
    }
    std::vector<size_t> run;
    size_t size = 0;
    bool has_previous = false;
    double previous = 0.0;
    const size_t n = std::min(words.size(), approx.size());
    for (size_t i = 0; i < n; ++i) {
        if (out.count(i)) {
            continue;
        }
        const double start = approx[i].first;
        const size_t length = words[i].size();
        if (!run.empty() && has_previous && start == previous) {
            run.push_back(i);
            size += 1 + length;
        } else {
            if (size > limit) {
                out.insert(run.begin(), run.end());
            }
            run.assign(1, i);
            size = length;
        }
        previous = start;
        has_previous = true;
    }
    if (size > limit) {
        out.insert(run.begin(), run.end());
    }
    return out;
}

std::vector<Chunk> plan_chunks(const std::vector<std::string> & words, const std::vector<Interval> & approx,
                               const std::vector<Interval> & speech_regions, double duration) {
    if (words.empty()) {
        return {};
    }
    if (words.size() != approx.size()) {
        throw std::invalid_argument("Word and approximate-time counts differ");
    }
    const size_t limit = static_cast<size_t>(MAX_TEXT_BYTES);
    std::vector<double> starts;
    starts.reserve(approx.size());
    for (const auto & interval : approx) {
        starts.push_back(interval.first);
    }
    std::vector<size_t> byte_prefix = {0};
    for (const auto & word : words) {
        if (word.size() > limit) {
            throw std::invalid_argument("A single word exceeds the 1022-byte transcript limit");
        }
        byte_prefix.push_back(byte_prefix.back() + word.size());
    }
    const auto text_bytes = [&](size_t lo, size_t hi) {
        return byte_prefix[hi] - byte_prefix[lo] + (hi > lo + 1 ? hi - lo - 1 : 0);
    };

    // Equal approximate starts cannot be separated by any time-based cut.
    if (oversized_start_group(words, approx)) {
        throw std::invalid_argument("Words sharing an approximate start exceed the 1022-byte limit");
    }

    const Tiers tiers = cut_tiers(approx, speech_regions, duration);
    std::vector<Interval> cores;
    double core_start = 0.0;
    while (duration - core_start > CORE_SECONDS) {
        double core_end = core_start + CORE_SECONDS;
        for (const auto & tier : tiers.lists) {
            const size_t right = bisect_right(tier, core_end);
            if (right && tier[right - 1] > core_start) {
                core_end = tier[right - 1];
                break;
            }
        }
        cores.emplace_back(core_start, core_end);
        core_start = core_end;
    }
    cores.emplace_back(core_start, duration);

    const auto route = [&](const Interval & core, Chunk & chunk) {
        const auto [a, b] = core;
        const Interval audio = {py_max(0.0, a - CONTEXT_SECONDS), py_min(duration, b + CONTEXT_SECONDS)};
        // The first/last cores also own starts on the recording's outer edges.
        const size_t owned_lo = a == 0.0 ? 0 : bisect_left(starts, a);
        const size_t owned_hi = b == duration ? words.size() : bisect_left(starts, b);
        size_t lo = owned_lo, hi = owned_hi;
        const auto overlaps = [&](size_t index) {
            const auto [start, end] = approx[index];
            return start < audio.second && end > audio.first;
        };
        while (lo > 0 && overlaps(lo - 1)) {
            --lo;
        }
        while (hi < words.size() && overlaps(hi)) {
            ++hi;
        }
        if (text_bytes(lo, hi) > limit) {
            lo = owned_lo;
            hi = owned_hi;
        }
        if (text_bytes(lo, hi) <= limit) {
            chunk.core = core;
            chunk.audio = audio;
            chunk.indices.clear();
            for (size_t index = lo; index < hi; ++index) {
                chunk.indices.push_back(index);
            }
            return true;
        }
        return false;
    };

    std::vector<Chunk> result;
    std::vector<Interval> pending(cores.rbegin(), cores.rend());
    while (!pending.empty()) {
        const Interval core = pending.back();
        pending.pop_back();
        Chunk chunk;
        if (route(core, chunk)) {
            result.push_back(std::move(chunk));
            continue;
        }
        const auto [a, b] = core;
        const double midpoint = (a + b) / 2.0;
        double cut = midpoint;
        for (const auto & tier : tiers.lists) {
            const size_t lo = bisect_right(tier, a), hi = bisect_left(tier, b);
            if (lo < hi) {
                const size_t near = bisect_left(tier, midpoint, lo, hi);
                const size_t from = std::max(lo, near > 0 ? near - 1 : 0), to = std::min(hi, near + 1);
                // min(choices, key=lambda point: (abs(point - midpoint), -point)): the first smallest key.
                cut = tier[from];
                for (size_t k = from + 1; k < to; ++k) {
                    const double point = tier[k];
                    const double dk = std::fabs(point - midpoint), dc = std::fabs(cut - midpoint);
                    if (dk < dc || (dk == dc && -point < -cut)) {
                        cut = point;
                    }
                }
                break;
            }
        }
        if (!(a < cut && cut < b)) {
            throw std::invalid_argument("Cannot split an oversized transcript at distinct recording times");
        }
        pending.emplace_back(cut, b);
        pending.emplace_back(a, cut);
    }
    return result;
}

std::vector<std::optional<Interval>> stitch_predictions(size_t n_words, const std::vector<Chunk> & chunks,
                                                        const std::vector<std::vector<std::optional<Interval>>> & predictions) {
    std::vector<std::optional<Interval>> result(n_words);
    if (!n_words) {
        return result;
    }
    if (chunks.size() != predictions.size()) {
        throw std::invalid_argument("stitch_predictions: chunk and prediction counts differ");
    }
    // Python's (rank, value) priority tuples.
    std::vector<std::optional<std::pair<int, double>>> priorities(n_words);
    for (size_t c = 0; c < chunks.size(); ++c) {
        const auto [a, b] = chunks[c].core;
        const auto & indices = chunks[c].indices;
        const auto & placed = predictions[c];
        if (indices.size() != placed.size()) {
            throw std::invalid_argument("stitch_predictions: chunk indices and predictions differ in length");
        }
        for (size_t k = 0; k < indices.size(); ++k) {
            if (!placed[k]) {
                continue;
            }
            const size_t index = indices[k];
            if (index >= n_words) {
                throw std::invalid_argument("stitch_predictions: word index out of range");
            }
            const auto [start, end] = *placed[k];
            const double center = (start + end) / 2.0;
            const std::pair<int, double> priority = a <= center && center <= b
                ? std::pair<int, double>{1, py_min(center - a, b - center)}
                : std::pair<int, double>{0, -py_max(a - center, center - b)};
            const auto & best = priorities[index];
            const bool better = !best || priority.first > best->first ||
                                (priority.first == best->first && priority.second > best->second);
            if (better) {
                priorities[index] = priority;
                result[index] = Interval{start, end};
            }
        }
    }
    return result;
}

std::vector<Interval> fill_unplaced(const std::vector<std::optional<Interval>> & intervals, double duration) {
    std::vector<std::optional<Interval>> filled = intervals;
    size_t index = 0;
    while (index < filled.size()) {
        if (filled[index]) {
            ++index;
            continue;
        }
        const size_t lo = index;
        while (index < filled.size() && !filled[index]) {
            ++index;
        }
        const double left = lo ? filled[lo - 1]->second : 0.0;
        const double right = index < filled.size() ? filled[index]->first : duration;
        const double width = (right - left) / static_cast<double>(index - lo);
        for (size_t offset = 0; offset < index - lo; ++offset) {
            filled[lo + offset] = Interval{left + static_cast<double>(offset) * width,
                                           lo + offset + 1 == index ? right : left + static_cast<double>(offset + 1) * width};
        }
    }
    std::vector<Interval> out;
    out.reserve(filled.size());
    for (const auto & interval : filled) {
        out.push_back(*interval);
    }
    return out;
}

}  // namespace fusealign
