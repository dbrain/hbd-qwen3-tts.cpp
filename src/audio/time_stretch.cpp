#include "time_stretch.h"

#include <algorithm>

namespace qwen3_tts_audio {

TimeStretch::TimeStretch(double speed, int sample_rate) {
    speed_ = std::min(4.0, std::max(0.25, speed));
    sr_    = sample_rate > 0 ? sample_rate : 24000;

    // ~43 ms window at 24 kHz. Long enough to hold several periods of a low
    // voice (85 Hz -> 12 ms), short enough that a transient is not smeared.
    N_  = 1 << 10;
    Hs_ = N_ / 2;
    L_  = N_ - Hs_;
    // The search must be able to reach a whole pitch period or it cannot phase
    // align the lowest voices: 24 kHz / 70 Hz = 343 samples.
    D_  = std::max(1, (int) (sr_ / 70));
    Ha_ = (double) Hs_ * speed_;

    // Periodic Hann: w[k] + w[k + Hs_] == 1 at 50 % overlap, so the overlap-add
    // reconstructs unity gain without a normalisation pass.
    win_.resize(N_);
    for (int k = 0; k < N_; k++)
        win_[k] = 0.5f - 0.5f * (float) std::cos(2.0 * M_PI * k / (double) N_);

    acc_.assign(N_, 0.0f);
    target_.assign(L_, 0.0f);
}

void TimeStretch::trim() {
    // Everything before the earliest sample the next frame can still reach is
    // dead: the search may look back to pos_ - D_, and the similarity target
    // starts at prev_ + Hs_.
    int64_t keep = (int64_t) std::llround(pos_) - D_;
    if (prev_ >= 0) keep = std::min(keep, prev_ + Hs_);
    if (keep <= base_) return;
    const size_t drop = (size_t) (keep - base_);
    if (drop >= buf_.size()) { buf_.clear(); base_ = keep; return; }
    // Amortised: only compact once the dead prefix is worth the move.
    if (drop < (size_t) (4 * N_)) return;
    buf_.erase(buf_.begin(), buf_.begin() + (long) drop);
    base_ = keep;
}

bool TimeStretch::emit_frame(std::vector<float> & out) {
    const int64_t end = base_ + (int64_t) buf_.size();
    const int64_t p   = (int64_t) std::llround(pos_);

    // Clamp the search to what actually exists rather than refusing outright.
    // The first frames sit at p == 0, where p - D_ is behind the start of the
    // stream; rejecting those stalls the stretcher permanently, because pos_
    // only advances when a frame is emitted.
    // Emit only once the FULL search range has arrived. Emitting early would
    // let the stretcher pick from a truncated candidate set, which makes the
    // output depend on how the caller happened to chunk its input -- measured
    // as streamed != one-shot before this guard. The first frame does not
    // search (no previous frame to match), so it need not wait.
    if (!flushing_ && prev_ >= 0 && p + D_ + N_ > end) return false;

    const int64_t lo = std::max<int64_t>(p - D_, base_);
    const int64_t hi = std::min<int64_t>(p + D_, end - N_);
    if (hi < lo) return false;                                  // need more input
    if (prev_ >= 0 && prev_ + Hs_ + L_ > end) return false;

    int64_t chosen = std::min(std::max(p, lo), hi);
    if (prev_ >= 0) {
        // Target: what would naturally have followed the frame we laid down
        // last time. Choosing the candidate most similar to it is what keeps
        // the waveform continuous across the splice.
        const float * tgt = &buf_[(size_t) (prev_ + Hs_ - base_)];

        // Candidate energies in O(1) each, from one pass of prefix sums-of-
        // squares over the search span. Without the energy term the search just
        // walks to the loudest offset rather than the best-matching one.
        const size_t span = (size_t) (hi - lo) + (size_t) L_ + 1;
        psum_.resize(span + 1);
        psum_[0] = 0.0;
        {
            const float * b = &buf_[(size_t) (lo - base_)];
            for (size_t k = 0; k < span; k++) psum_[k + 1] = psum_[k] + (double) b[k] * b[k];
        }
        auto score_at = [&](int64_t c) {
            const float * cand = &buf_[(size_t) (c - base_)];
            double dot = 0.0;
            for (int k = 0; k < L_; k++) dot += (double) tgt[k] * cand[k];
            const size_t o = (size_t) (c - lo);
            const double e = psum_[o + (size_t) L_] - psum_[o];
            return dot / std::sqrt(e + 1e-9);
        };

        // Coarse-to-fine. A full-resolution sweep is (2D+1) dot products of
        // length L -- 350k multiply-adds per frame, which measured 8.6 % of the
        // synth budget at speed 0.5. The correlation peak against a pitch
        // period (~200 samples here) is far too broad to be missed at stride 8,
        // and the fine pass recovers the exact offset, for ~5x less work.
        constexpr int kStride = 8;
        double best = -1e30;
        for (int64_t c = lo; c <= hi; c += kStride) {
            const double sc = score_at(c);
            if (sc > best) { best = sc; chosen = c; }
        }
        const int64_t flo = std::max(lo, chosen - (kStride - 1));
        const int64_t fhi = std::min(hi, chosen + (kStride - 1));
        for (int64_t c = flo; c <= fhi; c++) {
            const double sc = score_at(c);
            if (sc > best) { best = sc; chosen = c; }
        }
    }

    const float * src = &buf_[(size_t) (chosen - base_)];
    for (int k = 0; k < N_; k++) acc_[k] += win_[k] * src[k];

    out.insert(out.end(), acc_.begin(), acc_.begin() + Hs_);
    std::move(acc_.begin() + Hs_, acc_.end(), acc_.begin());
    std::fill(acc_.begin() + L_, acc_.end(), 0.0f);

    prev_ = chosen;
    pos_ += Ha_;
    trim();
    return true;
}

void TimeStretch::push(const float * in, size_t n, std::vector<float> & out) {
    if (ended_) return;
    if (n) { buf_.insert(buf_.end(), in, in + n); total_in_ += (int64_t) n; }
    const size_t before = out.size();
    while (emit_frame(out)) {}
    emitted_ += (int64_t) (out.size() - before);
}

void TimeStretch::flush(std::vector<float> & out) {
    if (ended_) return;
    // Pad so the frames covering the real tail can still be laid down. The
    // padding itself only ever lands in the fading half of the last window.
    flushing_ = true;
    buf_.insert(buf_.end(), (size_t) (N_ + 2 * D_), 0.0f);
    std::vector<float> tail;
    while (emit_frame(tail)) {}
    tail.insert(tail.end(), acc_.begin(), acc_.end());

    // Trim to the exact duration the request asked for. Draining the padded
    // frames overshoots by however much padding was needed (measured up to
    // +5.6 % at speed 3), and a client that scales alignment timestamps by
    // `speed` would then find them drifting against a too-long waveform.
    const int64_t want = (int64_t) std::llround((double) total_in_ / speed_) - emitted_;
    if (want > 0 && (int64_t) tail.size() > want) tail.resize((size_t) want);
    // Truncation can land mid-waveform; a short fade keeps that from clicking.
    const int fade = std::min<int>((int) tail.size(), sr_ / 200);   // 5 ms
    for (int k = 0; k < fade; k++)
        tail[tail.size() - fade + k] *= 1.0f - (float) k / (float) fade;

    emitted_ += (int64_t) tail.size();
    out.insert(out.end(), tail.begin(), tail.end());
    ended_ = true;
}

}  // namespace qwen3_tts_audio
