#include "audio_tokenizer_decoder.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

namespace {

void print_usage(const char * argv0) {
    fprintf(stderr, "usage: %s --tokenizer <path.gguf> [--frames N] [--chunk K] [--first K0] [--sched a,b,c] [--repeat R] [--dump out.f32] [--seed S] [--tol F]\n", argv0);
}

}

int main(int argc, char ** argv) {
    const char * tokenizer_path = nullptr;
    int n_frames = 64;
    int chunk = 16;
    int first = 0;
    int repeat = 1;
    std::vector<int> sched;
    const char * dump = nullptr;
    int seed = 1;
    float tol = 1e-4f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
        else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) n_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) chunk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--first") == 0 && i + 1 < argc) first = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sched") == 0 && i + 1 < argc) {
            for (char * t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) sched.push_back(atoi(t));
        }
        else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) dump = argv[++i];
        else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc) tol = (float) atof(argv[++i]);
        else { print_usage(argv[0]); return 1; }
    }
    if (!tokenizer_path) { print_usage(argv[0]); return 1; }

    qwen3_tts::AudioTokenizerDecoder decoder;
    if (!decoder.load_model(tokenizer_path)) {
        fprintf(stderr, "load failed: %s\n", decoder.get_error().c_str());
        return 2;
    }
    const auto & cfg = decoder.get_config();

    // random but deterministic codes in [0, codebook_size).
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(0, cfg.codebook_size - 1);
    std::vector<int32_t> codes((size_t) n_frames * cfg.n_codebooks);
    for (auto & c : codes) c = dist(rng);

    std::vector<float> pcm_oneshot;
    if (!decoder.decode(codes.data(), n_frames, pcm_oneshot)) {
        fprintf(stderr, "one-shot decode failed: %s\n", decoder.get_error().c_str());
        return 3;
    }
    printf("one-shot: %zu samples\n", pcm_oneshot.size());

    // causality check: decode(first half) should equal decode(full)[0:prefix_len]
    // if the decoder is strictly causal. this isolates streaming bugs from
    // non-causal behavior in the underlying decoder.
    if (n_frames >= 2) {
        int half = n_frames / 2;
        std::vector<float> pcm_half;
        if (decoder.decode(codes.data(), half, pcm_half)) {
            size_t cmp_n = pcm_half.size();
            if (cmp_n > pcm_oneshot.size()) cmp_n = pcm_oneshot.size();
            double max_diff = 0.0;
            size_t first_bad = SIZE_MAX;
            for (size_t i = 0; i < cmp_n; ++i) {
                double d = std::fabs((double) pcm_half[i] - (double) pcm_oneshot[i]);
                if (d > max_diff) max_diff = d;
                if (d > tol && first_bad == SIZE_MAX) first_bad = i;
            }
            printf("causality: decode(%d)[0:%zu] vs decode(%d)[0:%zu] max_diff=%.3e%s\n",
                   half, cmp_n, n_frames, cmp_n, max_diff,
                   (first_bad != SIZE_MAX) ? " (first deviation exists)" : "");
            if (first_bad != SIZE_MAX) {
                printf("  first deviation at %zu: half=%.6f full=%.6f\n",
                       first_bad, pcm_half[first_bad], pcm_oneshot[first_bad]);
            }
        }
    }

    std::vector<float> pcm_stream, pcm_prev;
    for (int r = 0; r < repeat; ++r) {
    pcm_prev.swap(pcm_stream);
    pcm_stream.clear();
    decoder.stream_reset();
    for (int off = 0, ci = 0; off < n_frames; ++ci) {
        int k = !sched.empty() ? (ci < (int) sched.size() ? sched[ci] : sched.back())
                               : ((off == 0 && first > 0) ? first : chunk);
        k = std::min(k, n_frames - off);
        const int32_t * ptr = codes.data() + (size_t) off * cfg.n_codebooks;
        if (!decoder.stream_decode(ptr, k, pcm_stream)) {
            fprintf(stderr, "stream_decode failed at off=%d: %s\n", off, decoder.get_error().c_str());
            return 4;
        }
        off += k;
    }
    if (r > 0) {
        size_t nd = 0, fd = SIZE_MAX;
        for (size_t i = 0; i < pcm_stream.size() && i < pcm_prev.size(); ++i)
            if (pcm_stream[i] != pcm_prev[i]) { ++nd; if (fd == SIZE_MAX) fd = i; }
        printf("repeat %d vs %d: %zu samples differ (first at %lld)\n", r, r - 1, nd, fd == SIZE_MAX ? -1LL : (long long) fd);
    }
    }
    if (dump) {
        FILE * f = fopen(dump, "wb");
        if (f) { fwrite(pcm_stream.data(), sizeof(float), pcm_stream.size(), f); fclose(f); }
    }
    printf("streaming: %zu samples (%d chunks of %d frames)\n", pcm_stream.size(),
           (n_frames + chunk - 1) / chunk, chunk);

    if (pcm_stream.size() != pcm_oneshot.size()) {
        fprintf(stderr, "FAIL: length mismatch: stream=%zu oneshot=%zu\n",
                pcm_stream.size(), pcm_oneshot.size());
        return 5;
    }

    double max_abs = 0.0, sum_sq = 0.0;
    size_t first_bad = SIZE_MAX;
    for (size_t i = 0; i < pcm_oneshot.size(); ++i) {
        double d = std::fabs((double) pcm_stream[i] - (double) pcm_oneshot[i]);
        if (d > max_abs) max_abs = d;
        sum_sq += d * d;
        if (d > tol && first_bad == SIZE_MAX) first_bad = i;
    }
    double rms = std::sqrt(sum_sq / pcm_oneshot.size());
    {
        const size_t W = 1920 * 4;
        printf("per-4-frame SNR(dB):");
        for (size_t b = 0; b < pcm_oneshot.size(); b += W) {
            double se = 0, sn = 0;
            for (size_t i = b; i < std::min(b + W, pcm_oneshot.size()); ++i) {
                double d = (double) pcm_stream[i] - pcm_oneshot[i];
                se += (double) pcm_oneshot[i] * pcm_oneshot[i]; sn += d * d;
            }
            printf(" %.1f", 10.0 * std::log10((se + 1e-30) / (sn + 1e-30)));
        }
        printf("\n");
    }
    printf("max_abs=%.3e rms=%.3e tol=%.3e\n", max_abs, rms, (double) tol);
    if (first_bad != SIZE_MAX) {
        printf("first deviation at sample %zu: stream=%.6f oneshot=%.6f\n",
               first_bad, pcm_stream[first_bad], pcm_oneshot[first_bad]);
    }
    if (max_abs > tol) {
        fprintf(stderr, "FAIL: max-abs-diff %.3e exceeds tol %.3e\n", max_abs, (double) tol);
        return 6;
    }
    printf("PASS\n");
    return 0;
}
