#include "breeze_tts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>

namespace breeze {

void depth_prof_dump();
void backbone_prof_dump();
void prof_dump_all() { backbone_prof_dump(); depth_prof_dump(); }

namespace {

using clk = std::chrono::steady_clock;
inline double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

inline float rng_next(uint32_t & s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (float) ((s >> 8) & 0xFFFFFF) / (float) 0x1000000;
}

// Sampling order mirrors HF's processor list: repetition penalty (applied by
// the caller), then temperature, top_k, top_p, multinomial.
int sample_logits(std::vector<float> & lg, int V, float temp, int top_k, float top_p,
                  uint32_t & rng) {
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < V; ++i) if (lg[i] > lg[best]) best = i;
        return best;
    }
    std::vector<std::pair<float, int>> p;
    p.reserve(V);
    float mx = -INFINITY;
    for (int i = 0; i < V; ++i) {
        const float v = lg[i] / temp;
        if (v > mx) mx = v;
        p.emplace_back(v, i);
    }
    double sum = 0;
    for (auto & e : p) { e.first = std::exp(e.first - mx); sum += e.first; }
    for (auto & e : p) e.first = (float) (e.first / sum);
    std::sort(p.begin(), p.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    int keep = (top_k > 0 && top_k < V) ? top_k : V;
    if (top_p > 0.0f && top_p < 1.0f) {
        double c = 0; int i = 0;
        for (; i < keep; ++i) { c += p[i].first; if (c > top_p) { ++i; break; } }
        keep = std::max(1, i);
    }
    double tot = 0;
    for (int i = 0; i < keep; ++i) tot += p[i].first;
    double r = rng_next(rng) * tot, acc = 0;
    for (int i = 0; i < keep; ++i) { acc += p[i].first; if (r <= acc) return p[i].second; }
    return p[0].second;
}

} // namespace

bool BreezeTTS::load(const std::string & path, int n_ctx) {
    unload();
    if (!w_.load(path))        { error_msg_ = w_.get_error();  return false; }
    if (!te_.init(&w_))        { error_msg_ = te_.get_error(); return false; }
    if (!bb_.init(&w_, n_ctx)) { error_msg_ = bb_.get_error(); return false; }
    if (!dd_.init(&w_))        { error_msg_ = dd_.get_error(); return false; }
    if (!cc_.init(&w_))        { error_msg_ = cc_.get_error(); return false; }
    loaded_ = true;
    return true;
}

void BreezeTTS::unload() {
    loaded_ = false;
    w_.unload();
}

std::string BreezeTTS::speaker_tag(const std::string & spk) const {
    if (spk.empty()) return "";
    if (spk.front() == '[' && spk.back() == ']') return spk;
    return "[" + spk + "]";
}

bool BreezeTTS::build_prompt(const std::string & text, const gen_params & gp,
                             const ref_voice * ref,
                             std::vector<int32_t> & ids, std::vector<float> & embeds,
                             double & text_enc_ms) {
    const auto & sp = w_.cfg().sp;
    const int n_embd = w_.cfg().bb.n_embd;
    const int NC = w_.cfg().bb.n_codebooks;
    const std::string tag = speaker_tag(gp.speaker);
    const bool has_ref = ref && ref->T > 0;

    struct segment { bool is_text; std::string text; };
    std::vector<segment> segs;
    if (has_ref) {
        segs.push_back({ true,  tag + ref->ref_text });
        segs.push_back({ false, {} });
    }
    segs.push_back({ true, gp.instruction.empty()
                         ? tag + text
                         : tag + "<ins_bos>" + gp.instruction + "<ins_eos>" + text });

    ids.clear();
    embeds.clear();
    text_enc_ms = 0;

    for (const auto & s : segs) {
        if (s.is_text) {
            const std::vector<int32_t> tids = w_.tok().encode(s.text, /*add_special=*/true);
            std::vector<float> e;
            auto t0 = clk::now();
            if (!te_.encode(tids, e)) { error_msg_ = te_.get_error(); return false; }
            text_enc_ms += ms_since(t0);
            if ((int) e.size() != (int) tids.size() * n_embd) {
                error_msg_ = "prompt: text-encoder output size mismatch";
                return false;
            }
            ids.insert(ids.end(), tids.begin(), tids.end());
            embeds.insert(embeds.end(), e.begin(), e.end());
        } else {
            // One <|AUDIO|> per reference frame, then <|audio_eos|>, whose
            // embedding is the frame of all codebook_eos ids.
            std::vector<int32_t> frames((size_t) (ref->T + 1) * NC);
            std::memcpy(frames.data(), ref->codes.data(),
                        (size_t) ref->T * NC * sizeof(int32_t));
            for (int c = 0; c < NC; ++c) frames[(size_t) ref->T * NC + c] = sp.codebook_eos;
            std::vector<float> e;
            if (!bb_.embed_frames(frames.data(), ref->T + 1, e)) {
                error_msg_ = bb_.get_error(); return false;
            }
            for (int t = 0; t < ref->T; ++t) ids.push_back(sp.audio);
            ids.push_back(sp.audio_eos);
            embeds.insert(embeds.end(), e.begin(), e.end());
        }
    }
    return true;
}

bool BreezeTTS::run_ar(const gen_params & gp, int chunk_frames, const pcm_cb * on_chunk,
                       gen_result & out) {
    const auto & bbc = w_.cfg().bb;
    const int NC  = bbc.n_codebooks;
    const int CB  = w_.cfg().cc.codebook_size;    // 0..2047 are real codes
    const int V   = bbc.audio_vocab;              // 2051
    const int LM  = bbc.lm_head_size;             // 2052; class 2051 is EOS
    const int EOS = V;

    uint32_t rng = gp.seed ? gp.seed : 0x9E3779B9u;
    std::vector<float> hidden, logits, frame_pcm;
    std::vector<int32_t> c0_hist;

    // The prompt has already been prefilled by the caller; `hidden`/`logits`
    // arrive through out.codes being empty and the members below.
    hidden = std::move(prefill_hidden_);
    logits = std::move(prefill_logits_);

    auto t_dec = clk::now();
    int emitted_frames = 0;
    int next_chunk = chunk_frames > 0 ? chunk_frames : 0;
    const int max_chunk = chunk_frames > 0 ? std::max(chunk_frames, 48) : 0;
    bool first_chunk = true;
    const auto t_start = t_ar_start_;

    for (int step = 0; step < gp.max_new_frames; ++step) {
        if (cancelled()) break;
        if ((int) logits.size() < LM) { error_msg_ = "ar: short logits"; return false; }

        // GeneratedTokenRepetitionPenaltyLogitsProcessor: only over generated,
        // in-vocab codebook-0 ids, and only from the second step on.
        if (gp.repetition_penalty != 1.0f && !c0_hist.empty()) {
            for (int id : c0_hist) {
                if (id < 0 || id >= LM) continue;
                logits[id] = logits[id] < 0 ? logits[id] * gp.repetition_penalty
                                            : logits[id] / gp.repetition_penalty;
            }
        }
        for (int i = CB; i < V; ++i) logits[i] = -INFINITY;   // reserved codec ids

        const int c0 = sample_logits(logits, LM, gp.temperature, gp.top_k, gp.top_p, rng);
        if (c0 == EOS) break;
        c0_hist.push_back(c0);

        std::vector<int32_t> frame(NC, 0);
        frame[0] = c0;
        if (!dd_.run_frame(hidden.data(), frame.data(), gp.depth_temperature,
                           gp.depth_top_k, gp.depth_top_p, rng)) {
            error_msg_ = dd_.get_error(); return false;
        }
        out.codes.insert(out.codes.end(), frame.begin(), frame.end());
        out.T++;

        // Mimi decode is fully causal, so decoding a PREFIX reproduces the
        // earlier samples exactly -- no cross-chunk artifacts, no hold-back.
        // The cost is that each emit re-decodes everything so far, which is
        // O(n^2) at a fixed chunk size. Keep the first chunk small (that IS the
        // TTFA) and then double up to a cap: same exact audio, ~5x less codec
        // work on a long paragraph.
        if (on_chunk && chunk_frames > 0 && out.T - emitted_frames >= next_chunk) {
            auto t_c = clk::now();
            std::vector<float> pcm;
            if (!cc_.decode(out.codes.data(), out.T, pcm)) { error_msg_ = cc_.get_error(); return false; }
            out.codec_ms += ms_since(t_c);
            const size_t done = (size_t) emitted_frames * w_.cfg().cc.samples_per_frame();
            if (pcm.size() > done) {
                (*on_chunk)(pcm.data() + done, (int) (pcm.size() - done), false);
                if (first_chunk) { out.ttfa_ms = ms_since(t_start); first_chunk = false; }
            }
            out.pcm = std::move(pcm);
            emitted_frames = out.T;
            next_chunk = std::min(next_chunk * 2, max_chunk);
        }

        if (step + 1 < gp.max_new_frames) {
            if (!bb_.decode_frame(frame.data(), hidden, logits)) {
                error_msg_ = bb_.get_error(); return false;
            }
        }
    }
    out.decode_ms = ms_since(t_dec) - out.codec_ms;

    auto t_c = clk::now();
    if (out.T > 0) {
        std::vector<float> pcm;
        if (!cc_.decode(out.codes.data(), out.T, pcm)) { error_msg_ = cc_.get_error(); return false; }
        out.codec_ms += ms_since(t_c);
        if (on_chunk) {
            const size_t done = (size_t) emitted_frames * w_.cfg().cc.samples_per_frame();
            if (pcm.size() > done) {
                (*on_chunk)(pcm.data() + done, (int) (pcm.size() - done), true);
                if (first_chunk) { out.ttfa_ms = ms_since(t_start); first_chunk = false; }
            } else {
                (*on_chunk)(nullptr, 0, true);
            }
        }
        out.pcm = std::move(pcm);
    } else if (on_chunk) {
        (*on_chunk)(nullptr, 0, true);
    }
    return true;
}

bool BreezeTTS::synthesize(const std::string & text, const gen_params & gp,
                           const ref_voice * ref, gen_result & out) {
    return synthesize_stream(text, gp, ref, 0, nullptr, out);
}

bool BreezeTTS::synthesize_stream(const std::string & text, const gen_params & gp,
                                  const ref_voice * ref, int chunk_frames,
                                  const pcm_cb & on_chunk, gen_result & out) {
    out = gen_result{};
    if (!loaded_) { error_msg_ = "engine not loaded"; return false; }

    t_ar_start_ = std::chrono::steady_clock::now();
    bb_.reset();

    std::vector<int32_t> ids;
    std::vector<float> embeds;
    if (!build_prompt(text, gp, ref, ids, embeds, out.text_enc_ms)) return false;
    out.n_prompt_tokens = (int) ids.size();

    auto t0 = std::chrono::steady_clock::now();
    if (!bb_.prefill_embeds(embeds.data(), (int) ids.size(), prefill_hidden_, prefill_logits_)) {
        error_msg_ = bb_.get_error(); return false;
    }
    out.prefill_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count();

    return run_ar(gp, chunk_frames, on_chunk ? &on_chunk : nullptr, out);
}

bool BreezeTTS::encode_voice(const float * pcm, int n, std::vector<int32_t> & codes, int & T) {
    if (!cc_.encode(pcm, n, codes, T)) { error_msg_ = cc_.get_error(); return false; }
    return true;
}

void BreezeTTS::log_vram(const char * label) const {
    const double w = w_.weights_bytes() / (1024.0 * 1024.0);
    const double kv = (bb_.kv_bytes() + dd_.kv_bytes()) / (1024.0 * 1024.0);
    const double sc = (bb_.sched_bytes() + dd_.sched_bytes() + cc_.sched_bytes()) / (1024.0 * 1024.0);
    fprintf(stderr, "  [vram-breeze %-10s] weights=%.1f kv=%.1f sched=%.1f total=%.1f MiB\n",
            label, w, kv, sc, w + kv + sc);
    if (getenv("BREEZE_PROF")) {
        const double M = 1024.0 * 1024.0;
        fprintf(stderr, "  [vram-breeze %-10s]   sched: backbone=%.1f depth=%.1f codec=%.1f "
                "text_enc=%.1f MiB | kv: backbone=%.1f depth=%.1f MiB\n", label,
                bb_.sched_bytes() / M, dd_.sched_bytes() / M, cc_.sched_bytes() / M,
                te_.sched_bytes() / M, bb_.kv_bytes() / M, dd_.kv_bytes() / M);
        dd_.log_sched_detail();
    }
}

} // namespace breeze
