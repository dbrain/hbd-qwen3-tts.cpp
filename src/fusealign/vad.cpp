// Silero VAD v5, 16 kHz, as fusealign_hub.vad.silero_probs runs it. The STFT and the encoder see one 512-sample
// frame plus the previous frame's last 64 samples, so frames are independent there and run on threads; only the
// LSTM steps through the frames.
//
// The convolutions round as PyTorch's CPU path does (oneDNN jit:avx512_core direct convolution, checked against
// torch 2.9.1): per output value, the input channels go in blocks of 16; each block is a fused multiply-add chain
// from zero over (tap, channel in block); the first block's sum gets the bias added, each later block's sum the
// running value. The STFT (one input channel, no bias) is a single chain over its 256 taps. With that, the encoder
// features are bit-identical to PyTorch's on AVX-512 machines; the LSTM and the activations differ by float rounding
// only. Every fused multiply-add is a correctly rounded fma, so the AVX2 path and the scalar fallback agree bit for
// bit; nothing else may be contracted into an fma.
#include "vad.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <sched.h>
#include <stdexcept>
#include <thread>

#include "ggml.h"
#include "gguf.h"

#if defined(__x86_64__)
#include <immintrin.h>
#define FUSEALIGN_VAD_X86 1
#endif

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace fusealign {

namespace {

constexpr int SAMPLE_RATE = 16000;
constexpr size_t FRAME = 512;          // samples per probability
constexpr size_t CONTEXT = 64;         // previous samples seen with each frame
constexpr size_t PADDED = 640;         // context + frame + 64 reflected samples
constexpr size_t N_FFT = 256;          // STFT window
constexpr size_t HOP = 128;            // STFT stride
constexpr size_t STEPS = 4;            // STFT columns per frame: (640 - 256) / 128 + 1
constexpr size_t BINS = 129;           // magnitude bins (258 basis rows = real + imaginary)
constexpr size_t SPEC_LD = 264;        // 258 rounded up to a multiple of 8
constexpr size_t HIDDEN = 128;         // LSTM width
constexpr size_t GATES = 4 * HIDDEN;   // i, f, g, o
constexpr size_t IC_BLOCK = 16;        // oneDNN's input-channel block
constexpr size_t BLOCK_FRAMES = 2048;  // frames per front-end pass; bounds memory, does not change results

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

// acc[o] = fma(w[k * ld + o], x[k], acc[o]) for k = 0 .. k_count - 1 in order, o = 0 .. o_count - 1.
using FmaChain = void (*)(float * acc, const float * w, size_t ld, const float * x, size_t k_count, size_t o_count);

void fma_chain_scalar(float * acc, const float * w, size_t ld, const float * x, size_t k_count, size_t o_count) {
    for (size_t k = 0; k < k_count; ++k) {
        const float xk = x[k];
        const float * wk = w + k * ld;
        for (size_t o = 0; o < o_count; ++o) {
            acc[o] = std::fma(wk[o], xk, acc[o]);
        }
    }
}

#ifdef FUSEALIGN_VAD_X86
__attribute__((target("avx2,fma"))) void fma_chain_avx2(float * acc, const float * w, size_t ld, const float * x,
                                                         size_t k_count, size_t o_count) {
    size_t o = 0;
    for (; o + 32 <= o_count; o += 32) {
        __m256 a0 = _mm256_loadu_ps(acc + o), a1 = _mm256_loadu_ps(acc + o + 8);
        __m256 a2 = _mm256_loadu_ps(acc + o + 16), a3 = _mm256_loadu_ps(acc + o + 24);
        for (size_t k = 0; k < k_count; ++k) {
            const __m256 xk = _mm256_broadcast_ss(x + k);
            const float * wk = w + k * ld + o;
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(wk), xk, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(wk + 8), xk, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(wk + 16), xk, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(wk + 24), xk, a3);
        }
        _mm256_storeu_ps(acc + o, a0);
        _mm256_storeu_ps(acc + o + 8, a1);
        _mm256_storeu_ps(acc + o + 16, a2);
        _mm256_storeu_ps(acc + o + 24, a3);
    }
    for (; o + 8 <= o_count; o += 8) {
        __m256 a = _mm256_loadu_ps(acc + o);
        for (size_t k = 0; k < k_count; ++k) {
            a = _mm256_fmadd_ps(_mm256_loadu_ps(w + k * ld + o), _mm256_broadcast_ss(x + k), a);
        }
        _mm256_storeu_ps(acc + o, a);
    }
    if (o < o_count) {
        fma_chain_scalar(acc + o, w + o, ld, x, k_count, o_count - o);
    }
}
#endif

FmaChain pick_fma_chain() {
#ifdef FUSEALIGN_VAD_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return fma_chain_avx2;
    }
#endif
    return fma_chain_scalar;
}

// Conv1d with kernel 3 and padding 1, weights as [tap][cin][ld] (ld = cout rounded up to 8, zero columns).
struct Conv {
    size_t cin = 0, cout = 0, ld = 0;
    std::vector<float> w;
    std::vector<float> bias;  // [ld]
};

// One output position from up to three input columns (nullptr = padding), oneDNN's rounding order, then ReLU.
// partial has ld floats of scratch; out gets cout values.
void conv_point(const Conv & conv, FmaChain fma_chain, const float * const (&taps)[3], float * partial, float * out) {
    for (size_t b0 = 0; b0 < conv.cin; b0 += IC_BLOCK) {
        const size_t nb = std::min(IC_BLOCK, conv.cin - b0);
        std::fill(partial, partial + conv.ld, 0.0f);
        for (size_t tap = 0; tap < 3; ++tap) {
            if (taps[tap]) {
                fma_chain(partial, conv.w.data() + (tap * conv.cin + b0) * conv.ld, conv.ld, taps[tap] + b0, nb, conv.ld);
            }
        }
        for (size_t o = 0; o < conv.cout; ++o) {
            out[o] = partial[o] + (b0 == 0 ? conv.bias[o] : out[o]);
        }
    }
    for (size_t o = 0; o < conv.cout; ++o) {
        out[o] = std::max(out[o], 0.0f);
    }
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Python's round(x, 1): the double nearest to x correctly rounded (half to even) to one decimal. glibc's printf is
// exact, as CPython's _Py_dg_dtoa is.
double round1(double x) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.1f", x);
    return std::strtod(text, nullptr);
}

int64_t floor_div2(int64_t x) { return x >= 0 ? x / 2 : -((-x + 1) / 2); }

}  // namespace

struct Vad::Impl {
    std::vector<float> basis_t;  // [256][SPEC_LD]: STFT basis transposed, zero-padded columns
    Conv enc[4];
    std::vector<float> w_ih_t;   // [128][512]
    std::vector<float> b_ih;
    std::vector<float> w_hh_t;   // [128][512]
    std::vector<float> b_hh;
    std::vector<float> dec_w;    // [128]
    float dec_b = 0.0f;
    FmaChain fma_chain = fma_chain_scalar;
    int n_threads = 1;
};

Vad::Vad(const std::string & gguf_path) : impl_(std::make_unique<Impl>()) {
    gguf_init_params params = {true, nullptr};
    gguf_context * ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx) {
        throw std::runtime_error("vad: cannot read GGUF " + gguf_path);
    }
    std::ifstream file(gguf_path, std::ios::binary);
    const auto load = [&](const std::string & name, std::vector<int64_t> torch_shape) {
        const int64_t id = gguf_find_tensor(ctx, name.c_str());
        if (id < 0) {
            gguf_free(ctx);
            throw std::runtime_error("vad: tensor " + name + " missing from " + gguf_path);
        }
        if (gguf_get_tensor_type(ctx, id) != GGML_TYPE_F32) {
            gguf_free(ctx);
            throw std::runtime_error("vad: tensor " + name + " is not F32");
        }
        const int64_t * ne = gguf_get_tensor_ne(ctx, id);
        size_t count = 1;
        for (size_t d = 0; d < GGML_MAX_DIMS; ++d) {
            const int64_t want = d < torch_shape.size() ? torch_shape[torch_shape.size() - 1 - d] : 1;
            if (ne[d] != want) {
                gguf_free(ctx);
                throw std::runtime_error("vad: tensor " + name + " has an unexpected shape");
            }
            count *= static_cast<size_t>(want);
        }
        std::vector<float> data(count);
        file.seekg(static_cast<std::streamoff>(gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, id)));
        file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
        if (!file) {
            gguf_free(ctx);
            throw std::runtime_error("vad: cannot read tensor " + name + " from " + gguf_path);
        }
        return data;
    };
    const auto transpose = [](const std::vector<float> & w, size_t rows, size_t cols, size_t ld) {
        std::vector<float> t(cols * ld, 0.0f);  // [cols][ld]
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                t[c * ld + r] = w[r * cols + c];
            }
        }
        return t;
    };

    Impl & m = *impl_;
    m.basis_t = transpose(load("vad.stft.forward_basis_buffer", {2 * BINS, 1, N_FFT}), 2 * BINS, N_FFT, SPEC_LD);
    const size_t cin[4] = {BINS, 128, 64, 64}, cout[4] = {128, 64, 64, 128};
    for (int layer = 0; layer < 4; ++layer) {
        const std::string prefix = "vad.encoder." + std::to_string(layer) + ".reparam_conv.";
        Conv & conv = m.enc[layer];
        conv.cin = cin[layer];
        conv.cout = cout[layer];
        conv.ld = (conv.cout + 7) / 8 * 8;
        const std::vector<float> w =
            load(prefix + "weight", {static_cast<int64_t>(conv.cout), static_cast<int64_t>(conv.cin), 3});
        conv.w.assign(3 * conv.cin * conv.ld, 0.0f);
        for (size_t o = 0; o < conv.cout; ++o) {
            for (size_t c = 0; c < conv.cin; ++c) {
                for (size_t tap = 0; tap < 3; ++tap) {
                    conv.w[(tap * conv.cin + c) * conv.ld + o] = w[(o * conv.cin + c) * 3 + tap];
                }
            }
        }
        conv.bias = load(prefix + "bias", {static_cast<int64_t>(conv.cout)});
        conv.bias.resize(conv.ld, 0.0f);
    }
    m.w_ih_t = transpose(load("vad.decoder.rnn.weight_ih", {GATES, HIDDEN}), GATES, HIDDEN, GATES);
    m.w_hh_t = transpose(load("vad.decoder.rnn.weight_hh", {GATES, HIDDEN}), GATES, HIDDEN, GATES);
    m.b_ih = load("vad.decoder.rnn.bias_ih", {GATES});
    m.b_hh = load("vad.decoder.rnn.bias_hh", {GATES});
    m.dec_w = load("vad.decoder.decoder.2.weight", {1, HIDDEN, 1});
    m.dec_b = load("vad.decoder.decoder.2.bias", {1})[0];
    gguf_free(ctx);
    m.fma_chain = pick_fma_chain();
    m.n_threads = worker_count();
}

Vad::~Vad() = default;

std::vector<float> Vad::probs(const float * wav, size_t n_samples) const {
    const Impl & m = *impl_;
    const FmaChain fma_chain = m.fma_chain;
    const size_t n_frames = (n_samples + FRAME - 1) / FRAME;
    std::vector<float> out(n_frames);
    const auto sample = [&](int64_t i) { return i >= 0 && static_cast<size_t>(i) < n_samples ? wav[i] : 0.0f; };

    // Frame f of the recording -> its LSTM input projection xw[f - first] (W_ih x + b_ih).
    std::vector<float> xw;
    const auto front_end = [&](size_t first, size_t lo, size_t hi) {
        float buf[PADDED], spec[STEPS][SPEC_LD], mag[STEPS][BINS], y0[STEPS][128], y1[2][64], y2[64], y3[128];
        float partial[SPEC_LD];
        for (size_t f = lo; f < hi; ++f) {
            // [previous 64 samples (zeros before the first frame) | frame (zero-padded at the end) | reflect 64]
            const int64_t base = static_cast<int64_t>(f * FRAME);
            for (size_t i = 0; i < CONTEXT; ++i) {
                buf[i] = f == 0 ? 0.0f : sample(base - static_cast<int64_t>(CONTEXT) + static_cast<int64_t>(i));
            }
            for (size_t i = 0; i < FRAME; ++i) {
                buf[CONTEXT + i] = sample(base + static_cast<int64_t>(i));
            }
            for (size_t i = 0; i < PADDED - CONTEXT - FRAME; ++i) {
                buf[CONTEXT + FRAME + i] = buf[CONTEXT + FRAME - 2 - i];
            }
            for (size_t t = 0; t < STEPS; ++t) {
                std::fill(spec[t], spec[t] + SPEC_LD, 0.0f);
                fma_chain(spec[t], m.basis_t.data(), SPEC_LD, buf + t * HOP, N_FFT, SPEC_LD);
                for (size_t b = 0; b < BINS; ++b) {
                    const float re2 = spec[t][b] * spec[t][b];
                    const float im2 = spec[t][BINS + b] * spec[t][BINS + b];
                    mag[t][b] = std::sqrt(re2 + im2);
                }
            }
            // Encoder 0: stride 1 over the 4 steps; 1: stride 2, 4 -> 2; 2: stride 2, 2 -> 1; 3: stride 1, 1 -> 1.
            for (size_t o = 0; o < STEPS; ++o) {
                const float * const taps[3] = {o > 0 ? mag[o - 1] : nullptr, mag[o], o + 1 < STEPS ? mag[o + 1] : nullptr};
                conv_point(m.enc[0], fma_chain, taps, partial, y0[o]);
            }
            for (size_t o = 0; o < 2; ++o) {
                const float * const taps[3] = {o > 0 ? y0[2 * o - 1] : nullptr, y0[2 * o], y0[2 * o + 1]};
                conv_point(m.enc[1], fma_chain, taps, partial, y1[o]);
            }
            const float * const taps2[3] = {nullptr, y1[0], y1[1]};
            conv_point(m.enc[2], fma_chain, taps2, partial, y2);
            const float * const taps3[3] = {nullptr, y2, nullptr};
            conv_point(m.enc[3], fma_chain, taps3, partial, y3);

            float * x = xw.data() + (f - first) * GATES;
            std::fill(x, x + GATES, 0.0f);
            fma_chain(x, m.w_ih_t.data(), GATES, y3, HIDDEN, GATES);
            for (size_t r = 0; r < GATES; ++r) {
                x[r] = x[r] + m.b_ih[r];
            }
        }
    };

    float h[HIDDEN] = {}, c[HIDDEN] = {}, gates[GATES], hidden[HIDDEN], partial[IC_BLOCK];
    for (size_t first = 0; first < n_frames; first += BLOCK_FRAMES) {
        const size_t n = std::min(BLOCK_FRAMES, n_frames - first);
        xw.resize(n * GATES);
        const size_t n_threads = std::min(static_cast<size_t>(m.n_threads), (n + 63) / 64);
        if (n_threads <= 1) {
            front_end(first, first, first + n);
        } else {
            std::vector<std::thread> pool;
            const size_t per = (n + n_threads - 1) / n_threads;
            for (size_t t = 0; t < n_threads; ++t) {
                const size_t lo = first + std::min(n, t * per), hi = first + std::min(n, (t + 1) * per);
                pool.emplace_back(front_end, first, lo, hi);
            }
            for (auto & thread : pool) {
                thread.join();
            }
        }

        // LSTM (gates i, f, g, o as torch.nn.LSTM), ReLU, the 1x1 decoder conv in oneDNN's order, sigmoid.
        for (size_t f = 0; f < n; ++f) {
            std::copy(m.b_hh.begin(), m.b_hh.end(), gates);
            fma_chain(gates, m.w_hh_t.data(), GATES, h, HIDDEN, GATES);
            const float * x = xw.data() + f * GATES;
            for (size_t j = 0; j < HIDDEN; ++j) {
                const float gi = sigmoid(gates[j] + x[j]);
                const float gf = sigmoid(gates[HIDDEN + j] + x[HIDDEN + j]);
                const float gg = std::tanh(gates[2 * HIDDEN + j] + x[2 * HIDDEN + j]);
                const float go = sigmoid(gates[3 * HIDDEN + j] + x[3 * HIDDEN + j]);
                const float fc = gf * c[j], ig = gi * gg;
                c[j] = fc + ig;
                h[j] = go * std::tanh(c[j]);
                hidden[j] = std::max(h[j], 0.0f);
            }
            float logit = 0.0f;
            for (size_t b0 = 0; b0 < HIDDEN; b0 += IC_BLOCK) {
                partial[0] = 0.0f;
                fma_chain(partial, m.dec_w.data() + b0, 1, hidden + b0, IC_BLOCK, 1);
                logit = partial[0] + (b0 == 0 ? m.dec_b : logit);
            }
            out[first + f] = sigmoid(logit);
        }
    }
    return out;
}

std::vector<std::pair<double, double>> Vad::speech_regions(const float * wav, size_t n_samples) const {
    const std::vector<float> speech_probs = probs(wav, n_samples);

    // silero_vad.utils_vad.get_speech_timestamps_from_probs with its defaults, sampling_rate=16000,
    // return_seconds=True, audio_length_samples=n_samples.
    const double sampling_rate = SAMPLE_RATE;
    const int64_t window_size_samples = 512;
    const int64_t audio_length_samples = static_cast<int64_t>(n_samples);
    const double threshold = 0.5;
    const double min_speech_duration_ms = 250, min_silence_duration_ms = 100, speech_pad_ms = 30;
    const double max_speech_duration_s = std::numeric_limits<double>::infinity();
    const double min_silence_at_max_speech = 98;
    const bool use_max_poss_sil_at_max_speech = true;  // time_resolution=1: round1 below

    const double min_speech_samples = sampling_rate * min_speech_duration_ms / 1000;
    const double speech_pad_samples = sampling_rate * speech_pad_ms / 1000;
    const double max_speech_samples =
        sampling_rate * max_speech_duration_s - static_cast<double>(window_size_samples) - 2 * speech_pad_samples;
    const double min_silence_samples = sampling_rate * min_silence_duration_ms / 1000;
    const double min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000;
    const double neg_threshold = std::max(threshold - 0.15, 0.01);

    struct Speech {
        int64_t start = 0, end = 0;
    };
    std::vector<Speech> speeches;
    bool triggered = false;
    std::optional<int64_t> current;  // current_speech['start']; empty dict when unset
    int64_t temp_end = 0, prev_end = 0, next_start = 0;
    std::vector<std::pair<int64_t, int64_t>> possible_ends;

    for (size_t i = 0; i < speech_probs.size(); ++i) {
        const double speech_prob = speech_probs[i];
        const int64_t cur_sample = window_size_samples * static_cast<int64_t>(i);

        // If speech returns after a temp_end, record candidate silence if long enough and clear temp_end.
        if (speech_prob >= threshold && temp_end) {
            const int64_t sil_dur = cur_sample - temp_end;
            if (static_cast<double>(sil_dur) > min_silence_samples_at_max_speech) {
                possible_ends.emplace_back(temp_end, sil_dur);
            }
            temp_end = 0;
            if (next_start < prev_end) {
                next_start = cur_sample;
            }
        }

        // Start of speech.
        if (speech_prob >= threshold && !triggered) {
            triggered = true;
            current = cur_sample;
            continue;
        }

        // Max speech length reached: decide where to cut.
        if (triggered && static_cast<double>(cur_sample - *current) > max_speech_samples) {
            if (use_max_poss_sil_at_max_speech && !possible_ends.empty()) {
                auto best = possible_ends.begin();  // max(..., key=duration): the first longest
                for (auto it = possible_ends.begin(); it != possible_ends.end(); ++it) {
                    if (it->second > best->second) {
                        best = it;
                    }
                }
                prev_end = best->first;
                const int64_t dur = best->second;
                speeches.push_back({*current, prev_end});
                current.reset();
                next_start = prev_end + dur;
                if (next_start < prev_end + cur_sample) {
                    current = next_start;
                } else {
                    triggered = false;
                }
                prev_end = next_start = temp_end = 0;
                possible_ends.clear();
            } else if (prev_end) {
                speeches.push_back({*current, prev_end});
                current.reset();
                if (next_start < prev_end) {
                    triggered = false;
                } else {
                    current = next_start;
                }
                prev_end = next_start = temp_end = 0;
                possible_ends.clear();
            } else {
                speeches.push_back({*current, cur_sample});
                current.reset();
                prev_end = next_start = temp_end = 0;
                triggered = false;
                possible_ends.clear();
                continue;
            }
        }

        // Silence detection while in speech.
        if (speech_prob < neg_threshold && triggered) {
            if (!temp_end) {
                temp_end = cur_sample;
            }
            const int64_t sil_dur_now = cur_sample - temp_end;
            if (!use_max_poss_sil_at_max_speech && static_cast<double>(sil_dur_now) > min_silence_samples_at_max_speech) {
                prev_end = temp_end;
            }
            if (static_cast<double>(sil_dur_now) < min_silence_samples) {
                continue;
            }
            if (static_cast<double>(temp_end - *current) > min_speech_samples) {
                speeches.push_back({*current, temp_end});
            }
            current.reset();
            prev_end = next_start = temp_end = 0;
            triggered = false;
            possible_ends.clear();
            continue;
        }
    }

    if (current && static_cast<double>(audio_length_samples - *current) > min_speech_samples) {
        speeches.push_back({*current, audio_length_samples});
    }

    for (size_t i = 0; i < speeches.size(); ++i) {
        Speech & speech = speeches[i];
        if (i == 0) {
            speech.start = static_cast<int64_t>(std::max(0.0, static_cast<double>(speech.start) - speech_pad_samples));
        }
        if (i != speeches.size() - 1) {
            const int64_t silence_duration = speeches[i + 1].start - speech.end;
            if (static_cast<double>(silence_duration) < 2 * speech_pad_samples) {
                speech.end += floor_div2(silence_duration);
                speeches[i + 1].start = std::max<int64_t>(0, speeches[i + 1].start - floor_div2(silence_duration));
            } else {
                speech.end = static_cast<int64_t>(
                    std::min(static_cast<double>(audio_length_samples), static_cast<double>(speech.end) + speech_pad_samples));
                speeches[i + 1].start =
                    static_cast<int64_t>(std::max(0.0, static_cast<double>(speeches[i + 1].start) - speech_pad_samples));
            }
        } else {
            speech.end = static_cast<int64_t>(
                std::min(static_cast<double>(audio_length_samples), static_cast<double>(speech.end) + speech_pad_samples));
        }
    }

    const double audio_length_seconds = static_cast<double>(audio_length_samples) / sampling_rate;
    std::vector<std::pair<double, double>> regions;
    regions.reserve(speeches.size());
    for (const Speech & speech : speeches) {
        const double start = round1(static_cast<double>(speech.start) / sampling_rate);
        const double end = round1(static_cast<double>(speech.end) / sampling_rate);
        regions.emplace_back(0.0 > start ? 0.0 : start, audio_length_seconds < end ? audio_length_seconds : end);
    }
    return regions;
}

}  // namespace fusealign
