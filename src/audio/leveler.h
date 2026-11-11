// leveler.h -- slow speech-level leveller for streamed TTS output.
//
// Breeze drifts in level sentence to sentence inside ONE generation (no chunk
// seam involved): a calm "Today they gave me comfort, of a kind." at -22.6 dBFS
// was followed by "I was wearing the right clothes for a hard day." at -17.4,
// +5 dB, which reads as the narrator suddenly raising their voice. Whole takes
// of the same text also land anywhere from -21 to -25 dBFS by seed, which is
// audible page to page across a book.
//
// This is a gentle AGC, not a compressor: it tracks the level of SPEECH (10 ms
// frames above -45 dBFS; pauses are ignored) over ~2.5 s behind and 0.6 s ahead,
// and steers a gain towards `target_db` minus that level, capped at +/-6 dB.
// The gain moves at most 4 dB/s while speech is sounding -- slow enough to keep
// a phrase's natural rise and fall -- and up to 40 dB/s in pauses, so a level
// step between sentences is mostly absorbed in the gap before the next one
// starts. A soft knee above -1 dBFS catches the rare peak a boost pushes over.
//
// Measured on the offending take: per-sentence spread 5.2 dB -> 1.3 dB, with a
// gain standard deviation of 1.5 dB over the clip.
//
// Lookahead is a hold-back: output runs up to 0.6 s behind input. The hold
// grows from zero (half of the audio seen so far), so time-to-first-audio is
// unchanged; by the time it reaches 0.6 s the client already holds seconds of
// buffered audio, since generation runs faster than realtime.
//
// Length-preserving and sample-aligned: the n-th sample out is the n-th sample
// in, scaled. So the forced aligner (which is fed the raw audio) and every
// timestamp the server emits stay valid for the levelled stream.

#ifndef QWEN3_TTS_LEVELER_H
#define QWEN3_TTS_LEVELER_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <vector>

namespace qwen3_tts_audio {

class Leveler {
public:
    explicit Leveler(int sample_rate, double target_db = -20.0)
        : sr_(sample_rate), hop_(std::max(1, sample_rate / 100)), target_(target_db) {}

    // Appends the levelled samples that are ready to `out`.
    void push(const float * x, size_t n, std::vector<float> & out) {
        in_.insert(in_.end(), x, x + n);
        total_in_ += n;
        while (in_frames_end() >= (size_t) hop_ * (e_.size() + 1)) add_frame((size_t) hop_);
        const size_t hold_frames = std::min<size_t>(kAheadFrames, total_in_ / 2 / hop_);
        const size_t ready = e_.size() > hold_frames ? e_.size() - hold_frames : 0;
        compute_gains(ready, false);
        emit(gains_.size() * (size_t) hop_, out);
    }

    // End of stream: level and return everything still held.
    void flush(std::vector<float> & out) {
        const size_t tail = total_in_ - (size_t) hop_ * e_.size();
        if (tail > 0) add_frame(tail);
        compute_gains(e_.size(), true);
        emit(total_in_, out);
    }

private:
    static constexpr size_t kBackFrames  = 250;    // 2.5 s
    static constexpr size_t kAheadFrames = 60;     // 0.6 s
    static constexpr double kGateDb      = -45.0;
    static constexpr double kMaxGainDb   = 6.0;
    static constexpr double kSlewSpeech  = 4.0 / 100.0;   // dB per 10 ms frame
    static constexpr double kSlewGap     = 40.0 / 100.0;
    static constexpr size_t kMinVoiced   = 10;     // frames before a level is trusted

    size_t in_frames_end() const { return total_in_; }

    void add_frame(size_t len) {
        // Frame k covers input samples [k*hop, k*hop + len).
        const size_t start = (size_t) hop_ * e_.size() - out_pos_;
        double s = 0;
        for (size_t i = 0; i < len; ++i) { const double v = in_[start + i]; s += v * v; }
        const double e = s / (double) len;
        e_.push_back(e);
        voiced_.push_back(10.0 * std::log10(e + 1e-12) > kGateDb);
    }

    void compute_gains(size_t upto, bool at_end) {
        while (gains_.size() < upto) {
            const size_t k = gains_.size();
            const size_t lo = k >= kBackFrames ? k - kBackFrames : 0;
            const size_t hi = std::min(e_.size(), k + kAheadFrames + 1);
            (void) at_end;
            double sum = 0; size_t nv = 0;
            for (size_t j = lo; j < hi; ++j) if (voiced_[j]) { sum += e_[j]; ++nv; }
            if (nv >= kMinVoiced) {
                const double lvl = 10.0 * std::log10(sum / (double) nv + 1e-12);
                target_gain_ = std::clamp(target_ - lvl, -kMaxGainDb, kMaxGainDb);
            }
            const double rate = voiced_[k] ? kSlewSpeech : kSlewGap;
            g_ += std::clamp(target_gain_ - g_, -rate, rate);
            gains_.push_back(g_);
        }
    }

    // Emits input samples [out_pos_, upto): within frame k the gain ramps from
    // frame k-1's to frame k's, so there is no step at a frame edge.
    void emit(size_t upto, std::vector<float> & out) {
        upto = std::min(upto, total_in_);
        while (out_pos_ < upto) {
            const size_t k = out_pos_ / (size_t) hop_;
            const size_t f0 = k * (size_t) hop_;
            const size_t flen = std::min((size_t) hop_, total_in_ - f0);
            const double ga = k > 0 ? gains_[k - 1] : 0.0, gb = gains_[k];
            const size_t end = std::min(upto, f0 + flen);
            for (; out_pos_ < end; ++out_pos_) {
                const double t = (double) (out_pos_ - f0 + 1) / (double) flen;
                const double g = std::pow(10.0, (ga + (gb - ga) * t) / 20.0);
                out.push_back(soft_clip((float) (in_.front() * g)));
                in_.pop_front();
            }
        }
    }

    static float soft_clip(float y) {
        const float c = 0.891f;   // -1 dBFS
        const float a = std::fabs(y);
        if (a <= c) return y;
        const float v = c + (1.0f - c) * std::tanh((a - c) / (1.0f - c));
        return y < 0 ? -v : v;
    }

    int sr_, hop_;
    double target_;
    std::deque<float> in_;            // samples not yet emitted, from out_pos_
    size_t total_in_ = 0, out_pos_ = 0;
    std::vector<double> e_;           // per-frame mean square, all frames so far
    std::vector<bool> voiced_;
    std::vector<double> gains_;       // per-frame gain (dB), computed so far
    double g_ = 0.0, target_gain_ = 0.0;
};

} // namespace qwen3_tts_audio

#endif
