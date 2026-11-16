// fa_bench — drive the read-along aligner exactly as breeze-server does (one
// AlignerSession, one FINAL pass per engine chunk, over the same socketpair
// child), without a TTS engine or a GPU.
//
//   fa_bench <aligner.gguf> <cpu|gpu> <chunks.jsonl> <out.jsonl>
//
// chunks.jsonl: one {"clip_id":..,"audio":"x.wav","text":".."} per line; the
// WAV is 16-bit mono at any rate (breeze hands the aligner 24 kHz).
// out.jsonl: {"clip_id","align_ms","wall_ms","words":[{"word","start","end","presence"}]},
// times in seconds as fusealign-cli writes them, so the same scorers read both.

#include "fa_session.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static bool read_wav_s16(const std::string & path, std::vector<float> & out, int & sr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) || std::memcmp(b.data() + 8, "WAVE", 4)) return false;
    size_t p = 12;
    int ch = 0, bits = 0;
    while (p + 8 <= b.size()) {
        uint32_t len;
        std::memcpy(&len, b.data() + p + 4, 4);
        const char * id = b.data() + p;
        if (!std::memcmp(id, "fmt ", 4)) {
            uint16_t c, bp; uint32_t r;
            std::memcpy(&c, b.data() + p + 10, 2); std::memcpy(&r, b.data() + p + 12, 4);
            std::memcpy(&bp, b.data() + p + 22, 2);
            ch = c; sr = (int) r; bits = bp;
        } else if (!std::memcmp(id, "data", 4)) {
            if (ch != 1 || bits != 16) return false;
            const size_t n = std::min<size_t>(len, b.size() - p - 8) / 2;
            out.resize(n);
            const int16_t * s = reinterpret_cast<const int16_t *>(b.data() + p + 8);
            for (size_t i = 0; i < n; i++) out[i] = s[i] / 32768.0f;
            return true;
        }
        p += 8 + len + (len & 1);
    }
    return false;
}

int main(int argc, char ** argv) {
    for (int i = 1; i + 1 < argc; i++)
        if (!std::strcmp(argv[i], "--fa-aligner")) return fa::run_aligner_loop(std::atoi(argv[i + 1]));
    if (argc != 5) {
        fprintf(stderr, "usage: %s <aligner.gguf> <cpu|gpu> <chunks.jsonl> <out.jsonl>\n", argv[0]);
        return 1;
    }
    const std::string model = argv[1], device = argv[2];
    std::ifstream in(argv[3]);
    std::ofstream out(argv[4]);
    fa::AlignerSession s(argv[0]);
    using clk = std::chrono::steady_clock;
    int64_t total_align = 0, total_wall = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const json row = json::parse(line);
        std::vector<float> pcm;
        int sr = 0;
        if (!read_wav_s16(row["audio"].get<std::string>(), pcm, sr)) {
            fprintf(stderr, "bad wav %s\n", row["audio"].get<std::string>().c_str());
            return 2;
        }
        std::vector<std::string> words;
        std::istringstream ws(row["text"].get<std::string>());
        for (std::string w; ws >> w;) words.push_back(w);

        const auto t0 = clk::now();
        std::vector<fa::AlignedWord> got;
        fa::AlignProfile prof;
        const int64_t ms = (int64_t) pcm.size() * 1000 / sr;
        if (!s.ensure_loaded(model, device) || !s.begin_streaming_align(words, sr) ||
            !s.finalize_streaming_align(pcm.data(), pcm.size(), ms, got, prof)) {
            fprintf(stderr, "align failed: %s\n", s.last_error().c_str());
            return 3;
        }
        const int64_t wall = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
        total_align += prof.t_aligner_ms;
        total_wall += wall;
        json wj = json::array();
        for (const auto & w : got)
            wj.push_back({{"word", w.text}, {"start", w.t0_ms / 1000.0}, {"end", w.t1_ms / 1000.0},
                          {"presence", w.confidence}});
        out << json{{"clip_id", row["clip_id"]}, {"align_ms", prof.t_aligner_ms}, {"load_ms", prof.t_load_ms},
                    {"wall_ms", wall}, {"words", wj}}.dump()
            << "\n";
    }
    s.shutdown();
    fprintf(stderr, "fa_bench: align %lld ms, wall %lld ms\n", (long long) total_align, (long long) total_wall);
    return 0;
}
