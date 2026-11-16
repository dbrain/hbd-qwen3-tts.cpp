// torchaudio.functional.resample (torchaudio 2.9.1) for a float32 waveform with its defaults: sinc_interp_hann,
// lowpass_filter_width=6, rolloff=0.99. The kernel is built as torchaudio builds it for a float32 input (every
// tensor op in float32, Python scalars rounded to float32), then applied as its strided conv1d.
#include "resample.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sched.h>
#include <stdexcept>
#include <thread>

namespace fusealign {

namespace {

constexpr int LOWPASS_FILTER_WIDTH = 6;
constexpr double ROLLOFF = 0.99;
constexpr double PI = 3.141592653589793;

int worker_count() {
    cpu_set_t set;
    int n = 0;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        n = CPU_COUNT(&set);
    }
    if (n <= 0) {
        n = static_cast<int>(std::thread::hardware_concurrency());
    }
    return std::clamp(n, 1, 16);
}

float dot(const float * a, const float * b, size_t n) {
    float lanes[8] = {};
    size_t k = 0;
    for (; k + 8 <= n; k += 8) {
        for (int l = 0; l < 8; ++l) {
            lanes[l] += a[k + l] * b[k + l];
        }
    }
    float sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (; k < n; ++k) {
        sum += a[k] * b[k];
    }
    return sum;
}

}  // namespace

std::vector<float> resample(const float * wav, size_t n_samples, int orig_sample_rate, int new_sample_rate) {
    if (orig_sample_rate <= 0 || new_sample_rate <= 0) {
        throw std::invalid_argument("Original frequency and desired frequecy should be positive");
    }
    if (orig_sample_rate == new_sample_rate) {
        return std::vector<float>(wav, wav + n_samples);
    }
    const int gcd = std::gcd(orig_sample_rate, new_sample_rate);
    const int orig = orig_sample_rate / gcd;
    const int target = new_sample_rate / gcd;
    const double base_freq = std::min(orig, target) * ROLLOFF;
    const int width = static_cast<int>(std::ceil(LOWPASS_FILTER_WIDTH * orig / base_freq));
    const size_t taps = static_cast<size_t>(2 * width + orig);

    // kernels[phase][k]: torchaudio's _get_sinc_resample_kernel with dtype=float32.
    const float base_f = static_cast<float>(base_freq);
    const float pi_f = static_cast<float>(PI);
    const float scale_f = static_cast<float>(base_freq / orig);
    const float limit = static_cast<float>(LOWPASS_FILTER_WIDTH);
    std::vector<float> kernels(static_cast<size_t>(target) * taps);
    for (int phase = 0; phase < target; ++phase) {
        const float shift = static_cast<float>(-phase) / static_cast<float>(target);
        for (size_t k = 0; k < taps; ++k) {
            const float idx = static_cast<float>(static_cast<int>(k) - width) / static_cast<float>(orig);
            float t = (shift + idx) * base_f;
            t = std::min(std::max(t, -limit), limit);
            const float c = std::cos(t * pi_f / limit / 2.0f);
            const float window = c * c;
            t *= pi_f;
            const float sinc = t == 0.0f ? 1.0f : std::sin(t) / t;
            kernels[static_cast<size_t>(phase) * taps + k] = sinc * (window * scale_f);
        }
    }

    // Pad (width, width + orig), conv1d with stride orig, interleave the phases, keep ceil(new * length / orig).
    std::vector<float> padded(static_cast<size_t>(width) + n_samples + static_cast<size_t>(width + orig), 0.0f);
    std::copy(wav, wav + n_samples, padded.begin() + width);
    const size_t frames = (padded.size() - taps) / static_cast<size_t>(orig) + 1;
    const size_t out_len = static_cast<size_t>(
        std::ceil(static_cast<double>(static_cast<int64_t>(target) * static_cast<int64_t>(n_samples)) / orig));
    std::vector<float> out(std::min(out_len, frames * static_cast<size_t>(target)));

    const auto run = [&](size_t frame_lo, size_t frame_hi) {
        for (size_t frame = frame_lo; frame < frame_hi; ++frame) {
            const float * x = padded.data() + frame * static_cast<size_t>(orig);
            for (int phase = 0; phase < target; ++phase) {
                const size_t o = frame * static_cast<size_t>(target) + static_cast<size_t>(phase);
                if (o >= out.size()) {
                    return;
                }
                out[o] = dot(kernels.data() + static_cast<size_t>(phase) * taps, x, taps);
            }
        }
    };
    const size_t work = frames * static_cast<size_t>(target) * taps;
    const int n_threads = work < (size_t{1} << 22) ? 1 : worker_count();
    if (n_threads == 1) {
        run(0, frames);
    } else {
        std::vector<std::thread> pool;
        const size_t per = (frames + n_threads - 1) / n_threads;
        for (int t = 0; t < n_threads; ++t) {
            const size_t lo = std::min(frames, per * t), hi = std::min(frames, lo + per);
            pool.emplace_back(run, lo, hi);
        }
        for (auto & thread : pool) {
            thread.join();
        }
    }
    return out;
}

}  // namespace fusealign
