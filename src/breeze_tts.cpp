#include "breeze_tts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

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
    int keep = (top_k > 0 && top_k < V) ? top_k : V;
    // Only the top `keep` are ever read: partition them out (O(V)) and sort
    // just those, instead of sorting all ~2051 per codebook (16x a frame).
    auto desc = [](const auto & a, const auto & b) { return a.first > b.first; };
    if (keep < V) {
        std::nth_element(p.begin(), p.begin() + keep - 1, p.end(), desc);
        std::sort(p.begin(), p.begin() + keep, desc);
    } else {
        std::sort(p.begin(), p.end(), desc);
    }
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
    n_ctx_ = n_ctx;
    cfg_ready_ = false;
    dd_pair_ready_ = false;
    if (!dd_.init(&w_))        { error_msg_ = dd_.get_error(); return false; }
    // Same GGUF, opened again by the vocoder/encoder: they own their own ggml
    // context and backend (the decoder runs on a dedicated low-priority stream),
    // and each reads only its own `tok_dec.*` / `tok_enc.*` prefix.
    if (!cc_.load_model(path))  { error_msg_ = "audio tokenizer decoder failed to load"; return false; }
    if (!ce_.load_model(path))  { error_msg_ = ce_.get_error(); return false; }
    loaded_ = true;
    return true;
}

bool BreezeTTS::ensure_dd_pair() {
    if (dd_pair_ready_) return true;
    if (!dd_neg_.init(&w_)) { error_msg_ = "depth pair: " + dd_neg_.get_error(); return false; }
    dd_pair_ready_ = true;
    return true;
}

bool BreezeTTS::ensure_cfg() {
    if (cfg_ready_) return true;
    if (!bb_neg_.init(&w_, n_ctx_, &bb_)) { error_msg_ = "cfg: " + bb_neg_.get_error(); return false; }
    if (!ensure_dd_pair()) return false;
    cfg_ready_ = true;
    return true;
}

void BreezeTTS::unload() {
    loaded_ = false;
    cfg_ready_ = false;
    dd_pair_ready_ = false;
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
                             double & text_enc_ms, const text_seg_fn * text_seg) {
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
            std::vector<int32_t> tids;
            std::vector<float> e;
            if (text_seg) {
                double ms = 0;
                if (!(*text_seg)(s.text, tids, e, ms)) { error_msg_ = "text segment not encoded"; return false; }
                text_enc_ms += ms;
            } else {
                tids = w_.tok().encode(s.text, /*add_special=*/true);
                auto t0 = clk::now();
                if (!te_.encode(tids, e)) { error_msg_ = te_.get_error(); return false; }
                text_enc_ms += ms_since(t0);
            }
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
    std::vector<float> neg_hidden = std::move(neg_hidden_);
    std::vector<float> neg_logits = std::move(neg_logits_);
    static const bool cfg_batch = [] {
        const char * e = std::getenv("BREEZE_CFG_BATCH");
        return !(e && !std::strcmp(e, "0"));
    }();

    // Frames are [T, NC] row-major; the vocoder takes the same layout.
    const int n_cb = NC;
    if (on_chunk && chunk_frames > 0) cc_.stream_reset(gp.max_new_frames);

    auto t_dec = clk::now();
    int emitted_frames = 0;
    int next_chunk = chunk_frames > 0 ? chunk_frames : 0;
    const int max_chunk = chunk_frames > 0 ? std::max(chunk_frames, 48) : 0;
    bool first_chunk = true;
    const auto t_start = t_ar_start_;

    bool hit_eos = false;
    for (int step = 0; step < gp.max_new_frames; ++step) {
        if (cancelled()) { hit_eos = true; break; }   // a cancel is not a truncation

        if ((int) logits.size() < LM) { error_msg_ = "ar: short logits"; return false; }
        if (use_cfg_) {
            if ((int) neg_logits.size() < LM) { error_msg_ = "ar: short cfg logits"; return false; }
            for (int i = 0; i < LM; ++i)
                logits[i] = neg_logits[i] + gp.cfg_scale * (logits[i] - neg_logits[i]);
        }

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
        if (c0 == EOS) { hit_eos = true; break; }
        c0_hist.push_back(c0);

        std::vector<int32_t> frame(NC, 0);
        frame[0] = c0;
        if (use_cfg_) {
            if (!dd_.run_frame_cfg(dd_neg_, hidden.data(), neg_hidden.data(), frame.data(),
                                   gp.cfg_scale, cfg_batch, gp.depth_temperature,
                                   gp.depth_top_k, gp.depth_top_p, rng)) {
                error_msg_ = dd_.get_error(); return false;
            }
        } else if (!dd_.run_frame(hidden.data(), frame.data(), gp.depth_temperature,
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
            if (use_cfg_ && cfg_batch) {
                if (!bb_.decode_frame_pair(bb_neg_, frame.data(), hidden, logits,
                                           neg_hidden, neg_logits)) {
                    error_msg_ = bb_.get_error(); return false;
                }
            } else {
                if (!bb_.decode_frame(frame.data(), hidden, logits)) {
                    error_msg_ = bb_.get_error(); return false;
                }
                if (use_cfg_ && !bb_neg_.decode_frame(frame.data(), neg_hidden, neg_logits)) {
                    error_msg_ = bb_neg_.get_error(); return false;
                }
            }
        }
    }
    out.truncated = !hit_eos;
    if (out.truncated)
        fprintf(stderr, "breeze: TRUNCATED at max_new_frames=%d (%.1f s) -- the text did not "
                        "finish; use long-form chunking or raise max_new_frames\n",
                gp.max_new_frames, gp.max_new_frames / 12.5);
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
    use_cfg_ = gp.cfg_scale != 1.0f && !gp.instruction.empty();
    if (gp.cfg_scale != 1.0f && gp.instruction.empty()) {
        static int warned = 0;
        if (warned++ < 4)
            fprintf(stderr, "breeze: cfg_scale=%.2f ignored -- CFG needs an instruction "
                            "(the negative prompt is the request without it)\n", gp.cfg_scale);
    }
    if (use_cfg_ && !ensure_cfg()) return false;

    std::vector<int32_t> ids;
    std::vector<float> embeds;
    if (!build_prompt(text, gp, ref, ids, embeds, out.text_enc_ms)) return false;
    out.n_prompt_tokens = (int) ids.size();

    auto t0 = std::chrono::steady_clock::now();
    if (!bb_.prefill_embeds(embeds.data(), (int) ids.size(), prefill_hidden_, prefill_logits_)) {
        error_msg_ = bb_.get_error(); return false;
    }
    if (use_cfg_) {
        gen_params neg = gp;
        neg.instruction.clear();
        std::vector<int32_t> nids;
        std::vector<float> nembeds;
        double nte_ms = 0;
        if (!build_prompt(text, neg, ref, nids, nembeds, nte_ms)) return false;
        out.text_enc_ms += nte_ms;
        bb_neg_.reset();
        if (!bb_neg_.prefill_embeds(nembeds.data(), (int) nids.size(), neg_hidden_, neg_logits_)) {
            error_msg_ = bb_neg_.get_error(); return false;
        }
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

std::vector<std::string> BreezeTTS::chunk_text(const std::string & text, int chunk_words,
                                               int first_chunk_words) {
    if (chunk_words <= 0) chunk_words = 120;
    int budget = (first_chunk_words > 0 && first_chunk_words < chunk_words) ? first_chunk_words
                                                                            : chunk_words;
    // Split on sentence enders first so a seam never lands mid-clause; pack
    // sentences up to the word budget. An over-long single sentence ships whole
    // rather than being cut at an arbitrary word.
    // Each sentence carries whether a seam after it is a good one: never
    // inside an open quotation -- '"Blah blah. Blah blah blah," Joe said.'
    // split after the first sentence renders the rest of the line in a fresh
    // generation that has not heard how the quote began -- and preferably at
    // a paragraph break. A paragraph break closes any quote (fiction leaves a
    // quote open across paragraphs of one speaker; that is a fine seam).
    struct Sent { std::string s; bool in_quote; bool para; };
    std::vector<Sent> sents;
    auto push = [&](size_t from, size_t to, bool in_q, bool para) {
        std::string sn = text.substr(from, to - from);
        size_t a = sn.find_first_not_of(" \t\r\n");
        if (a != std::string::npos) sents.push_back({sn.substr(a), in_q, para});
    };
    size_t start = 0;
    bool in_q = false;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '"') in_q = !in_q;
        else if ((unsigned char) c == 0xE2 && i + 2 < text.size() && (unsigned char) text[i + 1] == 0x80) {
            if ((unsigned char) text[i + 2] == 0x9C) in_q = true;        // left double quote
            else if ((unsigned char) text[i + 2] == 0x9D) in_q = false;  // right double quote
        }
        if (c == '.' || c == '!' || c == '?' || c == '\n') {
            size_t j = i + 1;
            while (j < text.size()) {
                const unsigned char d = (unsigned char) text[j];
                if (d == '"') { in_q = !in_q; ++j; }
                else if (d == '\'' || d == ')') ++j;
                else if (d == 0xE2 && j + 2 < text.size() && (unsigned char) text[j + 1] == 0x80 &&
                         ((unsigned char) text[j + 2] == 0x9D || (unsigned char) text[j + 2] == 0x99)) {
                    if ((unsigned char) text[j + 2] == 0x9D) in_q = false;
                    j += 3;
                } else break;
            }
            // The closing marks just scanned (and their effect on in_q) are
            // consumed here, not again by the outer loop.
            const size_t i_next = j - 1;
            if (j >= text.size() || text[j] == ' ' || text[j] == '\n' || text[j] == '\r') {
                size_t k = j;
                while (k < text.size() && (text[k] == ' ' || text[k] == '\r')) ++k;
                const bool para = c == '\n' || k >= text.size() || text[k] == '\n';
                if (para) in_q = false;
                push(start, j, in_q, para);
                start = j;
            }
            i = i_next;
        }
    }
    if (start < text.size()) push(start, text.size(), false, true);
    if (sents.empty() && !text.empty()) sents.push_back({text, false, true});

    auto words_in = [](const std::string & s) {
        int n = 0; bool in = false;
        for (char c : s) { const bool sp = (c==' '||c=='\t'||c=='\n'||c=='\r');
                           if (!sp && !in) { ++n; in = true; } else if (sp) in = false; }
        return n;
    };
    std::vector<std::string> out;
    std::string cur; int cur_w = 0;
    bool cur_safe = true;            // a seam after `cur` is outside any quote
    for (size_t si = 0; si < sents.size(); ++si) {
        const auto & sn = sents[si];
        const int w = words_in(sn.s);
        // Over budget: seam here unless it would split a quote, in which case
        // run on (up to half a budget more) to the quote's end.
        if (cur_w > 0 && cur_w + w > budget && (cur_safe || cur_w + w > budget + budget / 2)) {
            out.push_back(cur); cur.clear(); cur_w = 0;
            budget = std::min(budget * 2, chunk_words);
        }
        if (!cur.empty()) cur += " ";
        cur += sn.s; cur_w += w;
        cur_safe = !sn.in_quote;
        // A paragraph end past half the budget is a better seam than a
        // sentence end inside the next paragraph, when that paragraph will
        // not fit whole.
        int next_para_w = 0;
        if (sn.para)
            for (size_t k = si + 1; k < sents.size(); ++k) {
                next_para_w += words_in(sents[k].s);
                if (sents[k].para) break;
            }
        if (sn.para && cur_safe && si + 1 < sents.size() && cur_w * 2 >= budget &&
            cur_w + next_para_w > budget) {
            out.push_back(cur); cur.clear(); cur_w = 0;
            budget = std::min(budget * 2, chunk_words);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

bool BreezeTTS::synthesize_long(const std::string & text, const gen_params & gp,
                                const ref_voice * ref,
                                int chunk_words, int ref_max_frames,
                                int stream_chunk_frames, int gap_ms,
                                const pcm_cb & on_chunk, gen_result & out,
                                const chunk_cb & on_chunk_start,
                                int first_chunk_words) {
    if (!loaded_) { error_msg_ = "not loaded"; return false; }
    // Big enough that a default-sized chunk (120 words, ~500 frames) is carried
    // WHOLE. That matters far more than the memory it costs: the rolling history
    // is a clone template, and a template only works when the transcript
    // describes the audio. Trim the audio to a tail and the transcript has to be
    // dropped (below), which leaves every chunk after the first conditioned on
    // 20 s of the model's own speech with nothing saying what it said. Measured
    // on 912 words of prose, that produced real non-words by chunk 4 ("but
    // consider your daughters" -> "Paparier daughters", "impossible for us" ->
    // "impossible for Nums for Evertex") and drifted worse from there; carrying
    // the transcript rendered the same passage clean at the same RTF.
    if (ref_max_frames <= 0) ref_max_frames = 600;
    if (gap_ms < 0) gap_ms = 0;
    const std::vector<float> gap((size_t) (gap_ms * w_.cfg().cc.sample_rate / 1000), 0.0f);
    const int NC = w_.cfg().bb.n_codebooks;

    auto chunks = chunk_text(text, chunk_words, first_chunk_words);
    if (chunks.empty()) { error_msg_ = "no chunks"; return false; }

    out = gen_result{};
    // Rolling history: the previous chunk's generated codes and its text, which
    // together are exactly the clone template's (ref_text, ref_audio) pair.
    // Seed from the caller's voice so chunk 0 speaks in it; from chunk 1 the
    // rolling history takes over and carries that same voice forward.
    ref_voice hist;
    std::string hist_text;
    if (ref && ref->T > 0) { hist = *ref; hist_text = ref->ref_text; }

    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        if (cancelled()) break;
        gen_result r;
        const bool have_hist = hist.T > 0;
        const int used_ref_T = have_hist ? hist.T : 0;
        // Captured BEFORE hist_text is reassigned for the next chunk, which is
        // what the log used to read -- so the line described the conditioning of
        // the chunk after the one it named.
        const bool used_voice_only = have_hist && hist_text.empty();
        if (have_hist) hist.ref_text = hist_text;
        const bool last = (ci + 1 == chunks.size());
        if (ci > 0 && !gap.empty()) {
            if (on_chunk) on_chunk(gap.data(), (int) gap.size(), false);
            out.pcm.insert(out.pcm.end(), gap.begin(), gap.end());
        }
        // After the seam gap, so the position handed out is where this chunk's
        // own speech starts rather than where the silence before it does.
        if (on_chunk_start) on_chunk_start(ci, chunks.size(), chunks[ci]);
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
        out.truncated |= r.truncated;   // any short chunk truncates the whole read
        if (ci == 0) out.ttfa_ms = r.ttfa_ms;

        // Carry only the TAIL of this chunk: prompt tokens, reference frames and
        // new frames all share n_ctx, so an unbounded history starves the
        // generation it is meant to protect.
        //
        // The clone template pairs a transcript with THE AUDIO OF THAT
        // TRANSCRIPT. Once the codes are trimmed to a tail, the chunk's full
        // text no longer describes them, and pairing the two would hand the
        // model a transcript for audio it cannot hear. The budget is set so a
        // default-sized chunk is carried whole, but a chunk of unusually long
        // sentences can still overrun it -- and when it does, the rolling
        // history has to be abandoned rather than sent untranscribed, because
        // audio with no transcript is what drifts into non-words.
        //
        // Abandoning it costs nothing when there is a named voice: the caller's
        // reference IS a valid template and is what holds the speaker across the
        // whole read anyway. Only an unnamed voice has nothing to fall back to,
        // and there the untranscribed tail is still better than starting over
        // with a freshly sampled speaker.
        const int keep = std::min(r.T, ref_max_frames);
        if (keep == r.T && !(anchor_long_ref() && ref && ref->T > 0)) {
            hist.codes.assign(r.codes.end() - (size_t) keep * NC, r.codes.end());
            hist.T = keep;
            hist_text = chunks[ci];
        } else if (ref && ref->T > 0) {
            hist = *ref;
            hist_text = ref->ref_text;
        } else {
            hist.codes.assign(r.codes.end() - (size_t) keep * NC, r.codes.end());
            hist.T = keep;
            hist_text.clear();
        }
        fprintf(stderr, "  [breeze long-form] chunk %zu/%zu: prompt=%d tok -> %d frames (%.1fs), "
                "ref_T used=%d%s, carry=%d\n",
                ci + 1, chunks.size(), r.n_prompt_tokens, r.T, r.T / 12.5, used_ref_T,
                used_voice_only ? " (voice-only)" : "", hist.T);
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
    const double kv = (bb_.kv_bytes() + dd_.kv_bytes() + bb_neg_.kv_bytes() + dd_neg_.kv_bytes())
                      / (1024.0 * 1024.0);
    const double sc = (bb_.sched_bytes() + dd_.sched_bytes() + bb_neg_.sched_bytes()
                       + dd_neg_.sched_bytes() + cc_.sched_bytes()
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
