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
    // Same GGUF, opened again by the vocoder/encoder: they own their own ggml
    // context and backend (the decoder runs on a dedicated low-priority stream),
    // and each reads only its own `tok_dec.*` / `tok_enc.*` prefix.
    if (!cc_.load_model(path))  { error_msg_ = "audio tokenizer decoder failed to load"; return false; }
    if (!ce_.load_model(path))  { error_msg_ = ce_.get_error(); return false; }
    loaded_ = true;
    return true;
}

void BreezeTTS::unload() {
    loaded_ = false;
    cc_.unload_model();
    ce_.unload_model();
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

    // Frames are [T, NC] row-major; the vocoder takes the same layout.
    const int n_cb = NC;
    if (on_chunk && chunk_frames > 0) cc_.stream_reset(gp.max_new_frames);

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

        // The vocoder carries real streaming state (pre-transformer KV slab +
        // causal-conv tail rings), so each chunk decodes only its OWN frames and
        // still produces bit-identical audio to a single decode(). The previous
        // Mimi path re-decoded the whole prefix every emit, which was O(n^2) in
        // frames and made a long audiobook paragraph quadratically expensive.
        if (on_chunk && chunk_frames > 0 && out.T - emitted_frames >= next_chunk) {
            auto t_c = clk::now();
            std::vector<float> chunk;
            const int n_new = out.T - emitted_frames;
            if (!cc_.stream_decode(out.codes.data() + (size_t) emitted_frames * n_cb,
                                   n_new, chunk)) {
                error_msg_ = "vocoder stream_decode failed"; return false;
            }
            out.codec_ms += ms_since(t_c);
            if (!chunk.empty()) {
                (*on_chunk)(chunk.data(), (int) chunk.size(), false);
                if (first_chunk) { out.ttfa_ms = ms_since(t_start); first_chunk = false; }
                out.pcm.insert(out.pcm.end(), chunk.begin(), chunk.end());
            }
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
        if (on_chunk) {
            const int n_new = out.T - emitted_frames;
            if (n_new > 0 && !cc_.stream_decode(out.codes.data() + (size_t) emitted_frames * n_cb,
                                                n_new, pcm)) {
                error_msg_ = "vocoder stream_decode failed"; return false;
            }
        } else if (!cc_.decode(out.codes.data(), out.T, pcm)) {
            error_msg_ = "vocoder decode failed"; return false;
        }
        out.codec_ms += ms_since(t_c);
        if (on_chunk) {
            if (!pcm.empty()) {
                (*on_chunk)(pcm.data(), (int) pcm.size(), true);
                if (first_chunk) { out.ttfa_ms = ms_since(t_start); first_chunk = false; }
            } else {
                (*on_chunk)(nullptr, 0, true);
            }
            // Streaming decodes each chunk once, so out.pcm is built by
            // appending -- assigning here would keep only the final chunk.
            out.pcm.insert(out.pcm.end(), pcm.begin(), pcm.end());
        } else {
            out.pcm = std::move(pcm);
        }
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

bool BreezeTTS::teacher_force(const std::string & text, const gen_params & gp,
                              const int32_t * codes, int T,
                              std::vector<int> & ranks, std::vector<float> & logprobs) {
    ranks.clear(); logprobs.clear();
    if (!loaded_) { error_msg_ = "engine not loaded"; return false; }
    bb_.reset();
    std::vector<int32_t> ids;
    std::vector<float> embeds;
    double te_ms = 0;
    if (!build_prompt(text, gp, nullptr, ids, embeds, te_ms)) return false;

    const int NC = w_.cfg().bb.n_codebooks;
    const int LM = w_.cfg().bb.lm_head_size;
    std::vector<float> hidden, logits;
    if (!bb_.prefill_embeds(embeds.data(), (int) ids.size(), hidden, logits)) {
        error_msg_ = bb_.get_error(); return false;
    }
    for (int t = 0; t < T; ++t) {
        const int true_c0 = codes[(size_t) t * NC];
        double mx = -1e30;
        for (int i = 0; i < LM; ++i) mx = std::max(mx, (double) logits[i]);
        double sum = 0;
        int rank = 0;
        for (int i = 0; i < LM; ++i) {
            sum += std::exp(logits[i] - mx);
            if (logits[i] > logits[true_c0]) ++rank;
        }
        ranks.push_back(rank);
        logprobs.push_back((float) (logits[true_c0] - mx - std::log(sum)));
        if (t + 1 < T) {
            if (!bb_.decode_frame(codes + (size_t) t * NC, hidden, logits)) {
                error_msg_ = bb_.get_error(); return false;
            }
        }
    }
    return true;
}

bool BreezeTTS::decode_codes(const int32_t * codes, int T, std::vector<float> & pcm) {
    if (!cc_.decode(codes, T, pcm)) { error_msg_ = "vocoder decode failed"; return false; }
    return true;
}

std::vector<std::string> BreezeTTS::chunk_text(const std::string & text, int chunk_words) {
    if (chunk_words <= 0) chunk_words = 120;
    // Split on sentence enders first so a seam never lands mid-clause; pack
    // sentences up to the word budget. An over-long single sentence ships whole
    // rather than being cut at an arbitrary word.
    std::vector<std::string> sents;
    size_t start = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '.' || c == '!' || c == '?' || c == '\n') {
            size_t j = i + 1;
            while (j < text.size() && (text[j] == '"' || text[j] == '\'' || text[j] == ')')) ++j;
            if (j >= text.size() || text[j] == ' ' || text[j] == '\n') {
                std::string sn = text.substr(start, j - start);
                size_t a = sn.find_first_not_of(" \t\r\n");
                if (a != std::string::npos) sents.push_back(sn.substr(a));
                start = j;
            }
        }
    }
    if (start < text.size()) {
        std::string sn = text.substr(start);
        size_t a = sn.find_first_not_of(" \t\r\n");
        if (a != std::string::npos) sents.push_back(sn.substr(a));
    }
    if (sents.empty() && !text.empty()) sents.push_back(text);

    auto words_in = [](const std::string & s) {
        int n = 0; bool in = false;
        for (char c : s) { const bool sp = (c==' '||c=='\t'||c=='\n'||c=='\r');
                           if (!sp && !in) { ++n; in = true; } else if (sp) in = false; }
        return n;
    };
    std::vector<std::string> out;
    std::string cur; int cur_w = 0;
    for (const auto & sn : sents) {
        const int w = words_in(sn);
        if (cur_w > 0 && cur_w + w > chunk_words) { out.push_back(cur); cur.clear(); cur_w = 0; }
        if (!cur.empty()) cur += " ";
        cur += sn; cur_w += w;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

bool BreezeTTS::synthesize_long(const std::string & text, const gen_params & gp,
                                int chunk_words, int ref_max_frames,
                                int stream_chunk_frames,
                                const pcm_cb & on_chunk, gen_result & out) {
    if (!loaded_) { error_msg_ = "not loaded"; return false; }
    if (ref_max_frames <= 0) ref_max_frames = 250;
    const int NC = w_.cfg().bb.n_codebooks;

    auto chunks = chunk_text(text, chunk_words);
    if (chunks.empty()) { error_msg_ = "no chunks"; return false; }

    out = gen_result{};
    // Rolling history: the previous chunk's generated codes and its text, which
    // together are exactly the clone template's (ref_text, ref_audio) pair.
    ref_voice hist;
    std::string hist_text;

    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        if (cancelled()) break;
        gen_result r;
        const bool have_hist = hist.T > 0;
        const int used_ref_T = have_hist ? hist.T : 0;
        if (have_hist) { hist.ref_text = hist_text; }
        const bool last = (ci + 1 == chunks.size());
        bool ok;
        if (stream_chunk_frames > 0 && on_chunk) {
            // Forward each codec block as it lands, but only report is_final on
            // the last block of the last chunk -- a mid-document "final" would
            // close the caller's stream early.
            auto fwd = [&](const float * pcm, int n, bool fin) {
                on_chunk(pcm, n, fin && last);
            };
            ok = synthesize_stream(chunks[ci], gp, have_hist ? &hist : nullptr,
                                   stream_chunk_frames, fwd, r);
        } else {
            ok = synthesize(chunks[ci], gp, have_hist ? &hist : nullptr, r);
            if (ok && on_chunk && !r.pcm.empty())
                on_chunk(r.pcm.data(), (int) r.pcm.size(), last);
        }
        if (!ok) {
            error_msg_ = "chunk " + std::to_string(ci) + ": " + error_msg_;
            return false;
        }
        out.pcm.insert(out.pcm.end(), r.pcm.begin(), r.pcm.end());
        out.codes.insert(out.codes.end(), r.codes.begin(), r.codes.end());
        out.T += r.T;
        out.prefill_ms += r.prefill_ms; out.decode_ms += r.decode_ms;
        out.codec_ms += r.codec_ms;     out.text_enc_ms += r.text_enc_ms;
        if (ci == 0) out.ttfa_ms = r.ttfa_ms;

        // Carry only the TAIL of this chunk. Prompt tokens + ref frames + new
        // frames share n_ctx, so an unbounded history starves the generation.
        // Carry only the TAIL of this chunk: prompt tokens, reference frames and
        // new frames all share n_ctx, so an unbounded history starves the
        // generation it is meant to protect.
        const int keep = std::min(r.T, ref_max_frames);
        hist.codes.assign(r.codes.end() - (size_t) keep * NC, r.codes.end());
        hist.T = keep;
        // The clone template pairs a transcript with THE AUDIO OF THAT
        // TRANSCRIPT. Once the codes are trimmed to a tail, the chunk's full
        // text no longer describes them, so pairing the two would hand the model
        // a transcript for audio it cannot hear. Fall back to voice-only
        // conditioning in that case -- an empty ref_text, which is what higgs
        // does for its rolling context anyway.
        hist_text = (keep == r.T) ? chunks[ci] : std::string();
        fprintf(stderr, "  [breeze long-form] chunk %zu/%zu: prompt=%d tok -> %d frames (%.1fs), "
                "ref_T used=%d%s, carry=%d\n",
                ci + 1, chunks.size(), r.n_prompt_tokens, r.T, r.T / 12.5, used_ref_T,
                used_ref_T && hist_text.empty() ? " (voice-only)" : "", keep);
    }
    return true;
}

bool BreezeTTS::encode_voice(const float * pcm, int n, std::vector<int32_t> & codes, int & T) {
    int32_t nf = 0;
    if (!ce_.encode(pcm, n, codes, nf)) { error_msg_ = ce_.get_error(); return false; }
    T = (int) nf;
    return true;
}

void BreezeTTS::log_vram(const char * label) const {
    const double w = w_.weights_bytes() / (1024.0 * 1024.0);
    const double kv = (bb_.kv_bytes() + dd_.kv_bytes()) / (1024.0 * 1024.0);
    const double sc = (bb_.sched_bytes() + dd_.sched_bytes() + cc_.sched_bytes()
                       + ce_.sched_bytes() + cc_.stream_kv_bytes()) / (1024.0 * 1024.0);
    fprintf(stderr, "  [vram-breeze %-10s] weights=%.1f kv=%.1f sched=%.1f total=%.1f MiB\n",
            label, w, kv, sc, w + kv + sc);
    if (getenv("BREEZE_PROF")) {
        const double M = 1024.0 * 1024.0;
        fprintf(stderr, "  [vram-breeze %-10s]   sched: backbone=%.1f depth=%.1f codec=%.1f "
                "text_enc=%.1f MiB | kv: backbone=%.1f depth=%.1f MiB\n", label,
                bb_.sched_bytes() / M, dd_.sched_bytes() / M,
                (cc_.sched_bytes() + ce_.sched_bytes() + cc_.stream_kv_bytes()) / M,
                te_.sched_bytes() / M, bb_.kv_bytes() / M, dd_.kv_bytes() / M);
        dd_.log_sched_detail();
    }
}

} // namespace breeze
