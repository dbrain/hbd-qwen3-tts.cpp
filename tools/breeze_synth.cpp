// Standalone Breeze-TTS-2 synth CLI: text (+ optional reference wav) -> wav.
// Also the benchmark driver -- `--repeat N` reports per-stage ms, RTF and TTFA
// with the median over N warm runs.
//
//   breeze_synth --model M.gguf --text "..." [--instruction "..."]
//                [--ref-wav r.wav --ref-text "..."] [--speaker S0]
//                [--temp 0.9] [--seed 42] [--repeat 5] [--stream 12] -o out.wav

#include "breeze_tts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static const char * argval(int argc, char ** argv, const char * k, const char * def) {
    for (int i = 1; i < argc - 1; ++i) if (!strcmp(argv[i], k)) return argv[i + 1];
    return def;
}
static bool hasarg(int argc, char ** argv, const char * k) {
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], k)) return true;
    return false;
}

static void write_wav(const std::string & path, const std::vector<float> & pcm, int sr) {
    std::ofstream f(path, std::ios::binary);
    const uint32_t n = (uint32_t) pcm.size();
    const uint32_t data_bytes = n * 2;
    auto u32 = [&](uint32_t v) { f.write((const char *) &v, 4); };
    auto u16 = [&](uint16_t v) { f.write((const char *) &v, 2); };
    f.write("RIFF", 4); u32(36 + data_bytes); f.write("WAVE", 4);
    f.write("fmt ", 4); u32(16); u16(1); u16(1); u32(sr); u32(sr * 2); u16(2); u16(16);
    f.write("data", 4); u32(data_bytes);
    for (float v : pcm) {
        const int s = (int) std::lround(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f);
        u16((uint16_t) (int16_t) s);
    }
}

// 16-bit PCM WAV reader; resamples to 24 kHz with linear interpolation.
static bool read_wav_24k(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (b.size() < 44 || b.compare(0, 4, "RIFF") || b.compare(8, 4, "WAVE")) return false;
    size_t p = 12; int sr = 24000, ch = 1, bits = 16;
    std::vector<float> raw;
    while (p + 8 <= b.size()) {
        const std::string id = b.substr(p, 4);
        uint32_t sz; memcpy(&sz, b.data() + p + 4, 4);
        const size_t body = p + 8;
        if (id == "fmt " && body + 16 <= b.size()) {
            uint16_t c, bp; uint32_t s;
            memcpy(&c, b.data() + body + 2, 2);
            memcpy(&s, b.data() + body + 4, 4);
            memcpy(&bp, b.data() + body + 14, 2);
            ch = c; sr = (int) s; bits = bp;
        } else if (id == "data") {
            const size_t avail = std::min<size_t>(sz, b.size() - body);
            if (bits != 16) return false;
            const size_t ns = avail / 2;
            raw.resize(ns / ch);
            for (size_t i = 0; i < raw.size(); ++i) {
                int32_t acc = 0;
                for (int c = 0; c < ch; ++c) {
                    int16_t v; memcpy(&v, b.data() + body + (i * ch + c) * 2, 2);
                    acc += v;
                }
                raw[i] = (float) acc / (ch * 32768.0f);
            }
            break;
        }
        p = body + sz + (sz & 1);
    }
    if (raw.empty()) return false;
    if (sr == 24000) { out = std::move(raw); return true; }
    const double ratio = 24000.0 / sr;
    out.resize((size_t) (raw.size() * ratio));
    for (size_t i = 0; i < out.size(); ++i) {
        const double s = i / ratio;
        const size_t i0 = (size_t) s, i1 = std::min(i0 + 1, raw.size() - 1);
        const double t = s - i0;
        out[i] = (float) (raw[i0] * (1 - t) + raw[i1] * t);
    }
    return true;
}

int main(int argc, char ** argv) {
    const std::string model = argval(argc, argv, "--model", "");
    const std::string text  = argval(argc, argv, "--text", "The quick brown fox jumps over the lazy dog.");
    const std::string outp  = argval(argc, argv, "-o", "breeze_out.wav");
    if (model.empty()) { fprintf(stderr, "usage: breeze_synth --model M.gguf --text \"...\"\n"); return 2; }

    breeze::gen_params gp;
    gp.instruction = argval(argc, argv, "--instruction", "Speak clearly and naturally.");
    gp.speaker     = argval(argc, argv, "--speaker", "S0");
    gp.temperature = (float) atof(argval(argc, argv, "--temp", "0.9"));
    gp.depth_temperature = (float) atof(argval(argc, argv, "--depth-temp",
                                               argval(argc, argv, "--temp", "0.9")));
    gp.top_k       = atoi(argval(argc, argv, "--top-k", "50"));
    gp.top_p       = (float) atof(argval(argc, argv, "--top-p", "1.0"));
    gp.depth_top_k = gp.top_k;
    gp.depth_top_p = gp.top_p;
    gp.repetition_penalty = (float) atof(argval(argc, argv, "--rep-penalty", "1.1"));
    gp.seed        = (uint32_t) atoi(argval(argc, argv, "--seed", "42"));
    gp.max_new_frames = atoi(argval(argc, argv, "--max-frames", "750"));
    if (hasarg(argc, argv, "--no-instruction")) gp.instruction.clear();

    const int repeat = atoi(argval(argc, argv, "--repeat", "1"));
    const int stream_chunk = atoi(argval(argc, argv, "--stream", "0"));

    breeze::BreezeTTS eng;
    auto t0 = std::chrono::steady_clock::now();
    if (!eng.load(model, atoi(argval(argc, argv, "--n-ctx", "2048")))) {
        fprintf(stderr, "load failed: %s\n", eng.get_error().c_str()); return 1;
    }
    printf("load: %.0f ms\n", std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count());
    eng.log_vram("loaded");

    breeze::ref_voice ref;
    const std::string refw = argval(argc, argv, "--ref-wav", "");
    if (!refw.empty()) {
        std::vector<float> pcm;
        if (!read_wav_24k(refw, pcm)) { fprintf(stderr, "cannot read %s\n", refw.c_str()); return 1; }
        auto te = std::chrono::steady_clock::now();
        if (!eng.encode_voice(pcm.data(), (int) pcm.size(), ref.codes, ref.T)) {
            fprintf(stderr, "encode_voice: %s\n", eng.get_error().c_str()); return 1;
        }
        ref.ref_text = argval(argc, argv, "--ref-text", "");
        printf("ref: %zu samples -> %d frames in %.0f ms\n", pcm.size(), ref.T,
               std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - te).count());
    }

    std::vector<double> rtfs, ttfas, walls;
    breeze::gen_result r;
    for (int i = 0; i < repeat; ++i) {
        auto tw = std::chrono::steady_clock::now();
        bool ok;
        if (stream_chunk > 0) {
            int nblocks = 0;
            ok = eng.synthesize_stream(text, gp, ref.T ? &ref : nullptr, stream_chunk,
                                       [&](const float *, int n, bool fin) {
                                           if (n > 0) ++nblocks;
                                           (void) fin;
                                       }, r);
            if (ok && i == repeat - 1) printf("stream blocks: %d\n", nblocks);
        } else {
            ok = eng.synthesize(text, gp, ref.T ? &ref : nullptr, r);
        }
        if (!ok) { fprintf(stderr, "synth failed: %s\n", eng.get_error().c_str()); return 1; }
        const double wall = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - tw).count();
        const double dur_ms = 1000.0 * r.pcm.size() / eng.sample_rate();
        walls.push_back(wall);
        rtfs.push_back(wall / dur_ms);
        ttfas.push_back(r.ttfa_ms);
        printf("run %d: %d frames, %.2f s audio | prompt=%d tok text_enc=%.0f prefill=%.0f "
               "decode=%.0f codec=%.0f wall=%.0f ms | RTF=%.4f TTFA=%.0f ms\n",
               i, r.T, dur_ms / 1000.0, r.n_prompt_tokens, r.text_enc_ms, r.prefill_ms,
               r.decode_ms, r.codec_ms, wall, wall / dur_ms, r.ttfa_ms);
    }
    if (repeat > 1) {
        auto med = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        printf("MEDIAN over %d: wall=%.0f ms RTF=%.4f TTFA=%.0f ms\n",
               repeat, med(walls), med(rtfs), med(ttfas));
    }
    breeze::prof_dump_all();   // BREEZE_PROF=1
    eng.log_vram("after");

    write_wav(outp, r.pcm, eng.sample_rate());
    printf("wrote %s (%zu samples)\n", outp.c_str(), r.pcm.size());
    // A 44-byte WAV is what a crashed worker produces; fail loudly instead.
    if (r.pcm.size() < 1000) { fprintf(stderr, "*** SUSPICIOUSLY SHORT AUDIO ***\n"); return 1; }
    return 0;
}
