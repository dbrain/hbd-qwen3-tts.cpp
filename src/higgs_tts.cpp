#include "higgs_tts.h"
#include "higgs_encode.h"
#include "gguf_loader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <regex>

namespace higgs {

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

bool HiggsTTS::load(const std::string & backbone_gguf, const std::string & aux_gguf, int n_ctx) {
    if (!lm_.load_model(backbone_gguf, aux_gguf, n_ctx)) { error_msg_ = "lm: " + lm_.get_error(); return false; }
    if (!codec_.load_model(aux_gguf)) { error_msg_ = "codec: " + codec_.get_error(); return false; }
    // tokenizer: reuse the Qwen3 BPE vocab/merges packed in the backbone GGUF.
    {
        qwen3_tts::GGUFLoader gl;
        if (gl.open(backbone_gguf) && tok_.load_from_gguf(gl.get_ctx())) {
            tok_loaded_ = true;
        } else {
            fprintf(stderr, "  HiggsTTS: tokenizer load failed (%s) — text synth disabled, ids-only OK\n",
                    tok_.get_error().c_str());
        }
    }
    return true;
}

// BPE-encode text, emitting any "<|...|>" substring present in the vocab as its
// single token id (control-token / added-token handling).
std::vector<int32_t> HiggsTTS::encode_with_specials(const std::string & text) const {
    std::vector<int32_t> out;
    size_t i = 0, n = text.size();
    std::string buf;
    auto flush = [&]() { if (!buf.empty()) { auto t = tok_.encode(buf); out.insert(out.end(), t.begin(), t.end()); buf.clear(); } };
    while (i < n) {
        if (text[i] == '<' && i + 1 < n && text[i+1] == '|') {
            size_t end = text.find("|>", i + 2);
            if (end != std::string::npos) {
                std::string sp = text.substr(i, end + 2 - i);
                int32_t id = tok_.token_to_id(sp);
                if (id >= 0) { flush(); out.push_back(id); i = end + 2; continue; }
            }
        }
        buf.push_back(text[i++]);
    }
    flush();
    return out;
}

std::vector<int32_t> HiggsTTS::build_prompt(const std::string & text) const {
    std::vector<int32_t> ids;
    ids.push_back(sp_.tts);
    ids.push_back(sp_.text);
    auto t = encode_with_specials(text);
    ids.insert(ids.end(), t.begin(), t.end());
    ids.push_back(sp_.audio);
    return ids;
}

bool HiggsTTS::synthesize(const std::string & text, const gen_params & gp, gen_result & out) {
    if (!tok_loaded_) { error_msg_ = "tokenizer not loaded"; return false; }
    return synthesize_from_ids(build_prompt(text), gp, out, true);
}

bool HiggsTTS::encode_voice(const float * wav, int n_samples, int sr,
                            std::vector<int32_t> & codes_TN, int & T, int & N) {
    if (aux_enc_path_.empty()) { error_msg_ = "voice-clone encoder not configured (--aux-enc)"; return false; }
    HiggsCodecEncoder enc;
    if (!enc.load_model(aux_enc_path_)) { error_msg_ = "encoder load: " + enc.get_error(); return false; }
    N = enc.get_config().n_codebooks;
    bool ok = enc.encode(wav, n_samples, sr, codes_TN, T);
    if (!ok) error_msg_ = "encode: " + enc.get_error();
    enc.unload_model();   // free the 818 MB F32 encoder immediately (idle VRAM stays flat)
    return ok;
}

// Sample one codebook from its [V] logits (mirrors sglang _sample_independent).
int HiggsTTS::sample_one(const float * logits, int V, const gen_params & gp) {
    if (gp.temperature <= 1e-5f) {  // greedy
        int best = 0; float bv = logits[0];
        for (int v = 1; v < V; ++v) if (logits[v] > bv) { bv = logits[v]; best = v; }
        return best;
    }
    // scaled logits
    std::vector<float> l(logits, logits + V);
    const float inv_t = 1.0f / gp.temperature;
    for (float & x : l) x *= inv_t;

    // top_k
    if (gp.top_k > 0 && gp.top_k < V) {
        std::vector<float> tmp = l;
        std::nth_element(tmp.begin(), tmp.begin() + (V - gp.top_k), tmp.end());
        float kth = tmp[V - gp.top_k];
        for (float & x : l) if (x < kth) x = -INFINITY;
    }
    // softmax
    float mx = -INFINITY; for (float x : l) mx = std::max(mx, x);
    double sum = 0; std::vector<float> pr(V);
    for (int v = 0; v < V; ++v) { pr[v] = std::exp(l[v] - mx); sum += pr[v]; }
    for (float & x : pr) x /= (float)sum;
    // top_p (nucleus) on the prob vector
    if (gp.top_p < 1.0f) {
        std::vector<int> idx(V); std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return pr[a] > pr[b]; });
        double cum = 0; std::vector<char> keep(V, 0);
        for (int rank = 0; rank < V; ++rank) {
            keep[idx[rank]] = 1;
            cum += pr[idx[rank]];
            if (cum > gp.top_p) break;   // include the token that crosses the threshold
        }
        double s2 = 0;
        for (int v = 0; v < V; ++v) { if (!keep[v]) pr[v] = 0; s2 += pr[v]; }
        for (float & x : pr) x /= (float)s2;
    }
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double r = uni(rng_), acc = 0;
    for (int v = 0; v < V; ++v) { acc += pr[v]; if (r <= acc) return v; }
    return V - 1;
}

// Sample from the raw-logit softmax (no temperature scaling) — the RAS resample
// path (mirrors sglang/boson: logits.softmax(-1).multinomial(1)).
int HiggsTTS::sample_softmax_raw(const float * logits, int V) {
    float mx = -INFINITY; for (int v = 0; v < V; ++v) mx = std::max(mx, logits[v]);
    std::vector<float> pr(V); double sum = 0;
    for (int v = 0; v < V; ++v) { pr[v] = std::exp(logits[v] - mx); sum += pr[v]; }
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double r = uni(rng_) * sum, acc = 0;
    for (int v = 0; v < V; ++v) { acc += pr[v]; if (r <= acc) return v; }
    return V - 1;
}

void HiggsTTS::sampler_step(const std::vector<float> & logits, sampler_state & st,
                            const gen_params & gp, std::vector<int32_t> & codes,
                            const std::vector<std::vector<int32_t>> & hist) {
    const int N = st.n, V = lm_.get_config().cb_vocab;
    codes.assign(N, 0);
    if (st.done) { std::fill(codes.begin(), codes.end(), -1); return; }

    for (int c = 0; c < N; ++c) codes[c] = sample_one(&logits[(size_t)c * V], V, gp);

    // RAS: if a codebook's just-sampled token already appears >= max_repeat times
    // in the last ras_win_len generated frames, resample it from the raw softmax.
    if (gp.ras_win_len > 0 && !hist.empty()) {
        const int win = gp.ras_win_len;
        const int start = std::max(0, (int)hist.size() - win);
        for (int c = 0; c < N; ++c) {
            int cnt = 0;
            for (int j = start; j < (int)hist.size(); ++j) if (hist[j][c] == codes[c]) ++cnt;
            if (cnt >= gp.ras_max_repeat) codes[c] = sample_softmax_raw(&logits[(size_t)c * V], V);
        }
    }

    if (st.delay_count < N) {
        int next_cb = st.delay_count + 1;
        if (next_cb < N) for (int c = next_cb; c < N; ++c) codes[c] = BOC_ID;
        st.delay_count++;
    } else if (st.eoc_countdown >= 0) {
        st.eoc_countdown--;
        if (st.eoc_countdown <= 0) st.done = true;
    } else if (codes[0] == EOC_ID) {
        if (N <= 2) st.done = true;
        else st.eoc_countdown = N - 2;
    }
}

// AR loop from the first-step `logits` -> codes + reverse delay + codec.
bool HiggsTTS::run_ar(std::vector<float> & logits, const gen_params & gp,
                      gen_result & out, bool decode_audio) {
    const int N = lm_.get_config().n_codebooks;
    sampler_state st; st.n = N;
    std::vector<std::vector<int32_t>> delayed;
    std::vector<int32_t> codes;

    lm_.dump_prof_reset("prefill");   // report+discard the prefill call(s)
    auto td = clk::now();
    for (int s = 0; s < gp.max_new; ++s) {
        if (cancel_requested_.load(std::memory_order_relaxed)) break;  // cooperative cancel
        sampler_step(logits, st, gp, codes, delayed);
        if (st.done) break;
        delayed.push_back(codes);
        if (!lm_.decode_step(codes.data(), logits)) { error_msg_ = "decode_step: " + lm_.get_error(); return false; }
    }
    out.decode_ms = ms_since(td);
    out.steps = (int)delayed.size();
    lm_.dump_prof_reset("decode");

    // reverse delay pattern: T = L - (N-1); out[t][c] = delayed[t+c][c]; clip 0..1023.
    const int L = (int)delayed.size();
    out.L = L;
    out.delayed_flat.resize((size_t)L * N);
    for (int t = 0; t < L; ++t) for (int c = 0; c < N; ++c) out.delayed_flat[(size_t)t*N+c] = delayed[t][c];
    const int T = L - (N - 1);
    if (T <= 0) { error_msg_ = "generated too few frames"; return false; }
    out.T = T;
    out.codes_TN.assign((size_t)T * N, 0);
    for (int c = 0; c < N; ++c)
        for (int t = 0; t < T; ++t) {
            int v = delayed[t + c][c];
            if (v < 0) v = 0; if (v > 1023) v = 1023;
            out.codes_TN[(size_t)t * N + c] = v;
        }

    if (decode_audio) {
        auto tc = clk::now();
        if (!codec_.decode(out.codes_TN.data(), T, out.pcm)) { error_msg_ = "codec: " + codec_.get_error(); return false; }
        out.codec_ms = ms_since(tc);
    }
    return true;
}

bool HiggsTTS::synthesize_from_ids(const std::vector<int32_t> & ids, const gen_params & gp,
                                   gen_result & out, bool decode_audio) {
    rng_.seed(gp.seed ? gp.seed : 0xC0FFEEu);
    lm_.reset();
    auto t0 = clk::now();
    std::vector<float> logits;
    if (!lm_.prefill(ids.data(), (int)ids.size(), logits)) { error_msg_ = "prefill: " + lm_.get_error(); return false; }
    out.prefill_ms = ms_since(t0);
    return run_ar(logits, gp, out, decode_audio);
}

// MusicGen delay pattern: clean [T][N] -> delayed [T+N-1][N].
std::vector<int32_t> HiggsTTS::apply_delay_pattern(const int32_t * codes_TN, int T, int N) {
    const int L = T + N - 1;
    std::vector<int32_t> d((size_t)L * N);
    for (int l = 0; l < L; ++l)
        for (int c = 0; c < N; ++c) {
            const int t = l - c;
            int v = (t < 0) ? BOC_ID : (t >= T ? EOC_ID : codes_TN[(size_t)t * N + c]);
            d[(size_t)l * N + c] = v;
        }
    return d;
}

bool HiggsTTS::synthesize_with_ref(const std::string & text,
                                   const int32_t * ref_codes_TN, int ref_T,
                                   const std::string & ref_text,
                                   const gen_params & gp, gen_result & out, bool decode_audio) {
    if (!tok_loaded_) { error_msg_ = "tokenizer not loaded"; return false; }
    const int N = lm_.get_config().n_codebooks;
    rng_.seed(gp.seed ? gp.seed : 0xC0FFEEu);
    lm_.reset();

    // No reference -> zero-shot prompt (the validated path).
    if (!ref_codes_TN || ref_T <= 0) return synthesize_from_ids(build_prompt(text), gp, out, decode_audio);

    auto t0 = clk::now();
    std::vector<float> logits;

    // Segment 1 (text): <|tts|> [<|ref_text|> tok(ref_text)] <|ref_audio|>
    std::vector<int32_t> seg1; seg1.push_back(sp_.tts);
    if (!ref_text.empty() && sp_.ref_text >= 0) {
        seg1.push_back(sp_.ref_text);
        auto rt = encode_with_specials(ref_text);
        seg1.insert(seg1.end(), rt.begin(), rt.end());
    }
    seg1.push_back(sp_.ref_audio);
    if (!lm_.prefill(seg1.data(), (int)seg1.size(), logits)) { error_msg_ = "ref prefill seg1: " + lm_.get_error(); return false; }

    // Segment 2 (audio): delayed reference codes in the <|ref_audio|> slot.
    auto delayed_ref = apply_delay_pattern(ref_codes_TN, ref_T, N);
    if (!lm_.prefill_audio(delayed_ref.data(), (int)delayed_ref.size() / N, logits)) {
        error_msg_ = "ref prefill audio: " + lm_.get_error(); return false;
    }

    // Segment 3 (text): <|text|> tok(text) <|audio|>
    std::vector<int32_t> seg3; seg3.push_back(sp_.text);
    auto tt = encode_with_specials(text);
    seg3.insert(seg3.end(), tt.begin(), tt.end());
    seg3.push_back(sp_.audio);
    if (!lm_.prefill(seg3.data(), (int)seg3.size(), logits)) { error_msg_ = "ref prefill seg3: " + lm_.get_error(); return false; }

    out.prefill_ms = ms_since(t0);
    return run_ar(logits, gp, out, decode_audio);
}

// English word chunker (boson prepare_chunk_text method="word").
std::vector<std::string> HiggsTTS::prepare_chunk_text(const std::string & text, int max_words) {
    std::vector<std::string> chunks;
    if (max_words < 1) max_words = 1;
    // split into paragraphs on "\n\n"
    std::vector<std::string> paras;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t nn = text.find("\n\n", pos);
        if (nn == std::string::npos) { paras.push_back(text.substr(pos)); break; }
        paras.push_back(text.substr(pos, nn - pos));
        pos = nn + 2;
    }
    for (const auto & para : paras) {
        // split on whitespace into words
        std::vector<std::string> words;
        size_t i = 0, n = para.size();
        while (i < n) {
            while (i < n && (para[i] == ' ' || para[i] == '\t' || para[i] == '\n' || para[i] == '\r')) ++i;
            size_t s = i;
            while (i < n && !(para[i] == ' ' || para[i] == '\t' || para[i] == '\n' || para[i] == '\r')) ++i;
            if (i > s) words.push_back(para.substr(s, i - s));
        }
        if (words.empty()) continue;
        size_t first = chunks.size();
        for (size_t w = 0; w < words.size(); w += max_words) {
            std::string chunk;
            for (size_t k = w; k < std::min(words.size(), w + (size_t)max_words); ++k) {
                if (!chunk.empty()) chunk += ' ';
                chunk += words[k];
            }
            chunks.push_back(chunk);
        }
        if (chunks.size() > first) chunks.back() += "\n\n";  // mark paragraph end
    }
    return chunks;
}

bool HiggsTTS::synthesize_long(const std::string & text, const gen_params & gp,
                               int buffer, int chunk_words,
                               const pcm_cb & on_chunk, gen_result & out) {
    if (!tok_loaded_) { error_msg_ = "tokenizer not loaded"; return false; }
    if (buffer < 0) buffer = 0;
    auto chunks = prepare_chunk_text(text, chunk_words);
    if (chunks.empty()) { error_msg_ = "no chunks"; return false; }

    // rolling buffer of the last `buffer` chunks' clean codes (each [T_i][N]).
    std::vector<std::vector<int32_t>> buf_codes;   // flattened [T_i*N]
    std::vector<int> buf_T;

    out = gen_result{};
    std::vector<float> pcm_all;
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        if (cancel_requested_.load(std::memory_order_relaxed)) break;  // cancel between chunks
        // build concatenated reference codes from buffered chunks (concat along T)
        std::vector<int32_t> ref;
        int ref_T = 0;
        for (size_t b = 0; b < buf_codes.size(); ++b) {
            ref.insert(ref.end(), buf_codes[b].begin(), buf_codes[b].end());
            ref_T += buf_T[b];
        }
        gen_result r;
        bool ok = ref_T > 0
            ? synthesize_with_ref(chunks[ci], ref.data(), ref_T, std::string(), gp, r, true)
            : synthesize_with_ref(chunks[ci], nullptr, 0, std::string(), gp, r, true);
        if (!ok) {
            if (cancel_requested_.load(std::memory_order_relaxed)) break;  // cancelled mid-chunk: keep partial
            error_msg_ = "chunk " + std::to_string(ci) + ": " + error_msg_; return false;
        }

        if (on_chunk) on_chunk(r.pcm.data(), (int)r.pcm.size(), ci + 1 == chunks.size());
        pcm_all.insert(pcm_all.end(), r.pcm.begin(), r.pcm.end());
        out.prefill_ms += r.prefill_ms; out.decode_ms += r.decode_ms; out.codec_ms += r.codec_ms;
        out.steps += r.steps; out.T += r.T;

        // push this chunk's clean codes; trim to last `buffer`
        buf_codes.push_back(r.codes_TN); buf_T.push_back(r.T);
        while ((int)buf_codes.size() > buffer) { buf_codes.erase(buf_codes.begin()); buf_T.erase(buf_T.begin()); }
        fprintf(stderr, "  [long-form] chunk %zu/%zu: T=%d (%.2fs), ref_T=%d\n",
                ci + 1, chunks.size(), r.T, r.T / 25.0, ref_T);
    }
    out.pcm = std::move(pcm_all);
    return true;
}

// Parse "[Speaker_N]:" / "[Speaker N]:" tagged dialogue into ordered turns.
std::vector<HiggsTTS::dialogue_turn> HiggsTTS::parse_dialogue(const std::string & text) {
    std::vector<dialogue_turn> turns;
    static const std::regex re(R"(\[\s*(Speaker[ _]?\w+)\s*\]\s*:?)", std::regex::icase);
    auto trim = [](std::string s){
        size_t a = s.find_first_not_of(" \t\r\n");
        size_t b = s.find_last_not_of(" \t\r\n");
        return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
    };
    auto norm = [](std::string s){ for (char & c : s) if (c == ' ') c = '_'; return s; };
    std::string cur_spk;
    auto it = std::sregex_iterator(text.begin(), text.end(), re);
    auto end = std::sregex_iterator();
    size_t pos = 0;
    for (; it != end; ++it) {
        const auto & m = *it;
        std::string pre = trim(text.substr(pos, m.position() - pos));
        if (!pre.empty()) turns.push_back({cur_spk, pre});
        cur_spk = norm(m[1].str());
        pos = m.position() + m.length();
    }
    std::string tail = trim(text.substr(pos));
    if (!tail.empty()) turns.push_back({cur_spk, tail});
    return turns;
}

bool HiggsTTS::synthesize_multispeaker(const std::string & text,
                                       const std::map<std::string, named_voice> & voices,
                                       const gen_params & gp, int gap_ms, bool rolling,
                                       const pcm_cb & on_chunk, gen_result & out) {
    if (!tok_loaded_) { error_msg_ = "tokenizer not loaded"; return false; }
    auto turns = parse_dialogue(text);
    if (turns.empty()) { error_msg_ = "no dialogue turns"; return false; }
    out = gen_result{};
    const int gap = std::max(0, gap_ms) * 24;   // 24 kHz mono samples
    std::map<std::string, std::vector<int32_t>> prev_codes;   // speaker -> last gen codes [T,N]
    std::map<std::string, int> prev_T;
    bool any = false;
    for (size_t i = 0; i < turns.size(); ++i) {
        if (cancel_requested_.load(std::memory_order_relaxed)) break;  // cancel between turns
        const auto & turn = turns[i];
        auto vit = voices.find(turn.speaker);
        if (vit == voices.end()) {
            fprintf(stderr, "  [multispeaker] turn %zu: no voice for '%s', skipping\n", i, turn.speaker.c_str());
            continue;
        }
        const named_voice & nv = vit->second;
        // ref = speaker voice codes (+ this speaker's prior generated codes if rolling)
        std::vector<int32_t> ref = nv.codes_TN;
        int ref_T = nv.T;
        if (rolling) {
            auto pit = prev_codes.find(turn.speaker);
            if (pit != prev_codes.end()) {
                ref.insert(ref.end(), pit->second.begin(), pit->second.end());
                ref_T += prev_T[turn.speaker];
            }
        }
        gen_result r;
        if (!synthesize_with_ref(turn.text, ref.data(), ref_T, nv.ref_text, gp, r, true)) {
            if (cancel_requested_.load(std::memory_order_relaxed)) break;  // cancelled mid-turn: keep partial
            error_msg_ = "turn " + std::to_string(i) + ": " + error_msg_; return false;
        }
        if (any && gap > 0) out.pcm.insert(out.pcm.end(), (size_t)gap, 0.0f);
        out.pcm.insert(out.pcm.end(), r.pcm.begin(), r.pcm.end());
        out.prefill_ms += r.prefill_ms; out.decode_ms += r.decode_ms; out.codec_ms += r.codec_ms;
        out.steps += r.steps; out.T += r.T;
        if (rolling) { prev_codes[turn.speaker] = r.codes_TN; prev_T[turn.speaker] = r.T; }
        if (on_chunk) on_chunk(r.pcm.data(), (int)r.pcm.size(), i + 1 == turns.size());
        fprintf(stderr, "  [multispeaker] turn %zu [%s]: T=%d (%.2fs)\n", i, turn.speaker.c_str(), r.T, r.T/25.0);
        any = true;
    }
    if (!any) { error_msg_ = "no turns matched a mapped voice"; return false; }
    return true;
}

bool HiggsTTS::synthesize_stream(const std::string & text, const gen_params & gp,
                                 int chunk_frames, const pcm_cb & on_chunk, gen_result & out) {
    if (!tok_loaded_) { error_msg_ = "tokenizer not loaded"; return false; }
    const int N = lm_.get_config().n_codebooks;
    rng_.seed(gp.seed ? gp.seed : 0xC0FFEEu);
    lm_.reset();
    auto ids = build_prompt(text);

    auto t0 = clk::now();
    std::vector<float> logits;
    if (!lm_.prefill(ids.data(), (int)ids.size(), logits)) { error_msg_ = "prefill: " + lm_.get_error(); return false; }
    out.prefill_ms = ms_since(t0);

    sampler_state st; st.n = N;
    std::vector<std::vector<int32_t>> delayed;
    std::vector<int32_t> codes;
    const int hold = 3;             // frames held back near each block boundary
    size_t emitted = 0;             // samples already emitted
    std::vector<float> pcm_all;
    if (chunk_frames < 1) chunk_frames = 25;

    auto emit_upto = [&](int t_emit, bool final) {
        if (t_emit <= 0) return true;
        std::vector<int32_t> codes_TN((size_t)t_emit * N);
        for (int c = 0; c < N; ++c)
            for (int t = 0; t < t_emit; ++t) {
                int v = delayed[t + c][c]; if (v < 0) v = 0; if (v > 1023) v = 1023;
                codes_TN[(size_t)t * N + c] = v;
            }
        if (!codec_.decode(codes_TN.data(), t_emit, pcm_all)) { error_msg_ = "codec: " + codec_.get_error(); return false; }
        if (pcm_all.size() > emitted) {
            on_chunk(pcm_all.data() + emitted, (int)(pcm_all.size() - emitted), final);
            emitted = pcm_all.size();
        }
        return true;
    };

    auto td = clk::now();
    int last_decoded_T = 0;
    for (int s = 0; s < gp.max_new; ++s) {
        if (cancel_requested_.load(std::memory_order_relaxed)) break;  // cooperative cancel
        sampler_step(logits, st, gp, codes, delayed);
        if (st.done) break;
        delayed.push_back(codes);
        if (s + 1 < gp.max_new) {
            if (!lm_.decode_step(codes.data(), logits)) { error_msg_ = "decode_step: " + lm_.get_error(); return false; }
        }
        int T_now = (int)delayed.size() - (N - 1);
        if (T_now - last_decoded_T >= chunk_frames && T_now - hold > (int)(emitted / 960)) {
            if (!emit_upto(T_now - hold, false)) return false;
            last_decoded_T = T_now;
        }
    }
    out.decode_ms = ms_since(td);
    out.steps = (int)delayed.size();

    const int L = (int)delayed.size();
    const int T = L - (N - 1);
    if (T <= 0) { error_msg_ = "generated too few frames"; return false; }
    out.T = T;
    auto tc = clk::now();
    if (!emit_upto(T, true)) return false;   // final block, decode all
    out.codec_ms = ms_since(tc);
    out.pcm = pcm_all;
    out.codes_TN.assign((size_t)T * N, 0);
    for (int c = 0; c < N; ++c) for (int t = 0; t < T; ++t) {
        int v = delayed[t + c][c]; if (v < 0) v = 0; if (v > 1023) v = 1023;
        out.codes_TN[(size_t)t * N + c] = v;
    }
    return true;
}

} // namespace higgs
