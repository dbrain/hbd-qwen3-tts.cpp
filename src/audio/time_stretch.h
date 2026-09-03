// time_stretch.h — streaming WSOLA time-scale modification.
//
// `speed` on /v1/audio/speech changes DURATION without changing PITCH. Plain
// resampling would be a one-liner and would also raise the pitch, so it is not
// an option; the model has no speed conditioning either. WSOLA (waveform
// similarity overlap-add) is the standard answer: slide a window along the
// input at an analysis hop of `Hs * speed`, but before each overlap-add, search
// +/- a pitch period for the offset that best continues the previous frame.
// Aligning on waveform similarity is what stops the periodicity tearing and
// gives it the "no reverb, no phasiness" character a phase vocoder lacks on
// speech.
//
// Streaming by construction, because the default path is streamed: push() emits
// as soon as enough input has arrived and keeps only the samples still reachable
// by the search window, so time-to-first-audio is unaffected.
//
// Deliberately NOT applied to the audio the forced aligner sees. The aligner
// works on the natural-rate waveform -- its acoustic model was trained on
// natural speech, and stretched audio would only degrade it -- and the server
// divides the resulting timestamps by `speed` instead. That mapping is exact
// up to WSOLA's own +/- search radius (~16 ms), well inside the aligner's
// 30 ms median error.

#ifndef QWEN3_TTS_TIME_STRETCH_H
#define QWEN3_TTS_TIME_STRETCH_H

#include <cmath>
#include <cstdint>
#include <vector>

namespace qwen3_tts_audio {

class TimeStretch {
public:
    // speed > 1 is faster and shorter; 0.5 is half speed. Values outside
    // [0.25, 4.0] are clamped -- the caller validates and reports.
    TimeStretch(double speed, int sample_rate);

    // 1.0 within float noise: the caller should skip the stretcher entirely.
    static bool is_identity(double speed) { return std::fabs(speed - 1.0) < 1e-6; }

    // Append the stretched form of the newly-arrived samples to `out`. May
    // append nothing (still filling the first window) or several frames.
    void push(const float * in, size_t n, std::vector<float> & out);

    // Drain the tail. Zero-pads so the final frames can still be laid down,
    // then flushes the overlap accumulator, which fades the last ~43 ms.
    void flush(std::vector<float> & out);

    double speed() const { return speed_; }

private:
    bool emit_frame(std::vector<float> & out);   // false = need more input
    void trim();

    double speed_;
    int    sr_;
    int    N_;        // window length
    int    Hs_;       // synthesis hop (N_/2 -> Hann is COLA)
    int    L_;        // similarity template length (== N_ - Hs_)
    int    D_;        // search radius, >= one pitch period of a low male voice
    double Ha_;       // analysis hop = Hs_ * speed

    std::vector<float> win_;
    std::vector<float> buf_;      // input; absolute index of buf_[i] is base_ + i
    std::vector<float> acc_;      // overlap-add accumulator, N_ long
    std::vector<float> target_;   // natural continuation of the previous frame
    std::vector<double> psum_;    // prefix sums of squares over the search span
    int64_t base_     = 0;
    double  pos_      = 0;        // desired analysis position, absolute
    int64_t prev_     = -1;       // chosen analysis start of the previous frame
    int64_t total_in_ = 0;        // samples pushed, for the exact output length
    int64_t emitted_  = 0;        // samples handed back
    bool    flushing_ = false;
    bool    ended_    = false;
};

}  // namespace qwen3_tts_audio

#endif
