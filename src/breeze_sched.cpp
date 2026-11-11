#include "breeze_sched.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <set>

#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "ggml-cpu.h"

namespace breeze {

// Long-form reference. Anchored (default): every chunk is cloned from the
// caller's saved voice, as the reference implementation's batch API does for
// a list of texts. Rolling (BREEZE_LONG_REF=roll): each chunk is cloned from
// the previous chunk's own output. Measured over 36 seeds of a 316-word
// passage: rolling copies whatever level the model landed on forward, so takes
// spread 2.4x wider (sd 0.95 vs 0.42 dB), fade 1.3 dB by the end (anchored
// 0.1), and the first chunk after the short opening chunk -- cloned from that
// one short take -- dipped up to 8.4 dB (anchored 5.2). Without a saved voice
// there is nothing to anchor to, and rolling is used either way.
bool anchor_long_ref() {
    static const bool v = [] {
        const char * e = getenv("BREEZE_LONG_REF");
        return !(e && std::string(e) == "roll");
    }();
    return v;
}


namespace {
using clk = std::chrono::steady_clock;
inline double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}
// Same generator and sampler as breeze_tts.cpp / breeze_depth.cpp; a lane must
// consume its RNG exactly as run_ar does or a solo request stops being
// bit-identical to the unbatched engine.
inline float rng_next(uint32_t & s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (float) ((s >> 8) & 0xFFFFFF) / (float) 0x1000000;
}
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

struct BreezeScheduler::Lane {
    LaneRequest req;
    int slot = 0;
    std::vector<std::string> chunks;
    size_t ci = 0;
    ref_voice hist;
    std::string hist_text;
    std::vector<float> gap;
    gen_result out;   // the whole request
    gen_result r;     // the chunk being generated

    bool need_prefill = true;
    bool streaming = false;
    bool use_cfg = false;
    std::vector<float> hidden, logits, neg_hidden, neg_logits;
    std::vector<int32_t> c0_hist;
    std::vector<int32_t> frame;
    uint32_t rng = 0;
    int step = 0, emitted = 0, next_block = 0, max_block = 0;
    int min_frames = 0;   // EOS floor for this chunk (see start_chunk)
    bool first_block = true;
    clk::time_point t_start, t_dec;
    // Text prefetch; null when the encoder runs on the device (no prefetch).
    std::shared_ptr<Prefetch> pf;
    bool text_was_ready = false;
    clk::time_point t_ready;
    clk::time_point t_admit;
    // Audio handed to the caller, for the admission slack check.
    bool has_audio = false;
    clk::time_point t_first_audio;
    double audio_out_s = 0;
    qwen3_tts::AudioTokenizerDecoder::stream_slot voc;

    bool cancelled() const { return req.cancel && req.cancel->load(std::memory_order_relaxed); }
    bool last_chunk() const { return ci + 1 == chunks.size(); }
};

BreezeScheduler::BreezeScheduler(BreezeTTS & tts, int max_lanes)
    : tts_(tts), max_lanes_(std::max(1, std::min(2, max_lanes))) {
    th_ = std::thread([this] { loop(); });
    text_th_ = std::thread([this] { text_loop(false); });
    text_urgent_th_ = std::thread([this] { text_loop(true); });
}

BreezeScheduler::~BreezeScheduler() {
    { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
    cv_.notify_all();
    text_cv_.notify_all();
    if (th_.joinable()) th_.join();
    if (text_th_.joinable()) text_th_.join();
    if (text_urgent_th_.joinable()) text_urgent_th_.join();
}

void BreezeScheduler::text_loop(bool urgent_only) {
#if defined(__linux__)
    // Below the GPU thread, whose kernel launches are what a decode step waits
    // on; ggml's worker threads are created from here and inherit it.
    setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), 5);
#endif
    std::unique_ptr<TextEncoder> te;
    ggml_backend_t be = nullptr;
    struct Free { ggml_backend_t & b; std::unique_ptr<TextEncoder> & t;
                  ~Free() { t.reset(); if (b) ggml_backend_free(b); } } free_guard{be, te};
    for (;;) {
        TextJob job;
        {
            std::unique_lock<std::mutex> lk(m_);
            // Urgent jobs sit at the front of the queue.
            text_cv_.wait(lk, [&] {
                return stop_ || (!text_q_.empty() && (!urgent_only || text_q_.front().urgent));
            });
            if (stop_) return;
            job = std::move(text_q_.front());
            text_q_.pop_front();
        }
        auto seg = std::make_shared<EncSeg>();
        auto t0 = clk::now();
        if (!te && !te_failed_) {
            int nt = 4;
            if (const char * e = std::getenv("BREEZE_TEXT_ENC_THREADS")) nt = std::max(1, atoi(e));
            be = ggml_backend_cpu_init();
            if (be) ggml_backend_cpu_set_n_threads(be, nt);
            te = std::make_unique<TextEncoder>();
            if (!be || !te->init(&tts_.w_, be)) {
                fprintf(stderr, "breeze-sched: CPU text encoder init failed: %s\n", te->get_error().c_str());
                te_failed_ = true;
            }
        }
        if (!te_failed_ && te) {
            seg->ids = tts_.w_.tok().encode(job.seg, /*add_special=*/true);
            seg->ok = te->encode(seg->ids, seg->emb);
            if (!seg->ok) fprintf(stderr, "breeze-sched: text encode failed: %s\n", te->get_error().c_str());
        }
        seg->ms = ms_since(t0);
        if (job.cache && seg->ok) {
            std::lock_guard<std::mutex> lk(vcache_m_);
            vcache_.emplace_back(job.seg, seg);
            if (vcache_.size() > 8) vcache_.pop_front();
        }
        { std::lock_guard<std::mutex> lk(job.pf->m); job.pf->segs[job.seg] = seg; }
        cv_.notify_all();
    }
}

// The text segments build_prompt will encode for chunk `ci` -- must mirror it.
std::vector<std::string> BreezeScheduler::segs_for(const Lane & L, size_t ci,
                                                   const std::string & hist_text,
                                                   bool have_hist) const {
    const auto & gp = L.req.gp;
    const std::string tag = tts_.speaker_tag(gp.speaker);
    const std::string & text = L.chunks[ci];
    std::vector<std::string> v;
    if (have_hist) v.push_back(tag + hist_text);
    v.push_back(gp.instruction.empty() ? tag + text
                                       : tag + "<ins_bos>" + gp.instruction + "<ins_eos>" + text);
    if (gp.cfg_scale != 1.0f && !gp.instruction.empty()) v.push_back(tag + text);
    return v;
}

// segs[0 .. cacheable) may be served from / stored in the voice cache.
void BreezeScheduler::want_text(Lane & L, const std::vector<std::string> & segs, size_t cacheable,
                                bool urgent) {
    if (!L.pf) return;
    std::vector<TextJob> todo;
    for (size_t i = 0; i < segs.size(); ++i) {
        const std::string & sg = segs[i];
        std::shared_ptr<EncSeg> hit;
        if (i < cacheable) {
            std::lock_guard<std::mutex> lk(vcache_m_);
            for (auto & e : vcache_) if (e.first == sg) { hit = std::make_shared<EncSeg>(*e.second); hit->ms = 0; }
        }
        std::lock_guard<std::mutex> lk(L.pf->m);
        if (hit) { L.pf->segs.emplace(sg, hit); continue; }
        if (L.pf->segs.emplace(sg, nullptr).second) todo.push_back({L.pf, sg, i < cacheable});
    }
    if (todo.empty()) return;
    {
        std::lock_guard<std::mutex> lk(m_);
        if (urgent) for (auto it = todo.rbegin(); it != todo.rend(); ++it) { it->urgent = true; text_q_.push_front(std::move(*it)); }
        else        for (auto & j : todo) text_q_.push_back(std::move(j));
    }
    text_cv_.notify_all();
}

bool BreezeScheduler::text_ready(Lane & L) {
    if (!L.pf) return true;
    const auto need = segs_for(L, L.ci, L.hist_text, L.hist.T > 0);
    want_text(L, need, 0, true);   // a missed speculation (the carry was not the whole chunk)
    std::lock_guard<std::mutex> lk(L.pf->m);
    for (const auto & sg : need) {
        auto it = L.pf->segs.find(sg);
        if (it == L.pf->segs.end() || !it->second) return false;
    }
    return true;
}

// Seconds of audio the caller holds beyond a 1x playhead started at its first
// sample. Only meaningful once the lane has produced audio.
double BreezeScheduler::slack_s(const Lane & L) const {
    if (!L.has_audio) return 1e9;
    return L.audio_out_s - std::chrono::duration<double>(clk::now() - L.t_first_audio).count();
}

void BreezeScheduler::emit(Lane & L, const float * pcm, int n, bool fin) {
    if (n > 0) {
        if (!L.has_audio) { L.has_audio = true; L.t_first_audio = clk::now(); }
        L.audio_out_s += (double) n / (double) tts_.w_.cfg().cc.sample_rate;
    }
    if (L.req.on_pcm) L.req.on_pcm(pcm, n, fin);
}

void BreezeScheduler::submit(LaneRequest r) {
    { std::lock_guard<std::mutex> lk(m_); queue_.push_back(std::move(r)); }
    cv_.notify_all();
}

void BreezeScheduler::run_sync(const std::function<void()> & job) {
    std::promise<void> done;
    auto fut = done.get_future();
    {
        std::lock_guard<std::mutex> lk(m_);
        jobs_.push_back([&] { job(); done.set_value(); });
    }
    cv_.notify_all();
    fut.wait();
}

Backbone & BreezeScheduler::bb(int slot)     { return slot == 0 ? tts_.bb_ : *bb1_; }
Backbone & BreezeScheduler::bb_neg(int slot) { return slot == 0 ? tts_.bb_neg_ : *bb1_neg_; }

bool BreezeScheduler::ensure_slot(int slot, bool cfg) {
    if (slot == 0) return !cfg || tts_.ensure_cfg();
    if (!bb1_ready_) {
        bb1_ = std::make_unique<Backbone>();
        // Shares slot 0's prefill scheduler: prefills are serialised on this
        // thread, so a second prompt-sized compute buffer would only sit idle.
        if (!bb1_->init(&tts_.w_, tts_.n_ctx_, &tts_.bb_)) {
            tts_.error_msg_ = "lane 1: " + bb1_->get_error(); return false;
        }
        bb1_ready_ = true;
        // The depth decoder's pair partner is the CFG negative decoder.
        if (!tts_.ensure_dd_pair()) return false;
    }
    if (cfg && !bb1_neg_ready_) {
        if (!tts_.ensure_dd_pair()) return false;
        bb1_neg_ = std::make_unique<Backbone>();
        if (!bb1_neg_->init(&tts_.w_, tts_.n_ctx_, &tts_.bb_)) {
            tts_.error_msg_ = "lane 1 cfg: " + bb1_neg_->get_error(); return false;
        }
        bb1_neg_ready_ = true;
    }
    return true;
}

void BreezeScheduler::fail(Lane & L, const std::string & err) {
    fprintf(stderr, "breeze-sched: lane %d failed: %s\n", L.slot, err.c_str());
    L.need_prefill = false;
    L.ci = L.chunks.size();
    if (L.req.on_done) L.req.on_done(false, L.out, err);
    L.req.on_done = nullptr;
}

// synthesize_long's per-chunk preamble + synthesize_stream's prefill.
bool BreezeScheduler::start_chunk(Lane & L, bool sync_text) {
    BreezeTTS & T = tts_;
    const auto & gp = L.req.gp;
    const bool have_hist = L.hist.T > 0;
    if (have_hist) L.hist.ref_text = L.hist_text;
    if (L.ci > 0 && !L.gap.empty()) {
        if (L.streaming) emit(L, L.gap.data(), (int) L.gap.size(), false);
        L.out.pcm.insert(L.out.pcm.end(), L.gap.begin(), L.gap.end());
    }
    if (L.streaming && L.req.long_form && L.req.on_chunk_start)
        L.req.on_chunk_start(L.ci, L.chunks.size(), L.chunks[L.ci]);

    L.r = gen_result{};
    L.t_start = clk::now();
    L.use_cfg = gp.cfg_scale != 1.0f && !gp.instruction.empty();
    if (!ensure_slot(L.slot, L.use_cfg)) { fail(L, T.error_msg_); return false; }
    Backbone & B = bb(L.slot);
    B.reset();

    std::vector<int32_t> ids;
    std::vector<float> embeds;
    const ref_voice * ref = have_hist ? &L.hist : nullptr;
    BreezeTTS::text_seg_fn lookup;
    if (L.pf) {
        lookup = [&L, &T, sync_text](const std::string & sg, std::vector<int32_t> & tids,
                                     std::vector<float> & e, double & ms) {
            {
                std::lock_guard<std::mutex> lk(L.pf->m);
                auto it = L.pf->segs.find(sg);
                if (it != L.pf->segs.end() && it->second && it->second->ok) {
                    tids = it->second->ids; e = it->second->emb;
                    ms = it->second->ms;
                    it->second->ms = 0;   // counted once even when reused as the next transcript
                    return true;
                }
            }
            if (!sync_text) return false;
            auto t0 = clk::now();
            tids = T.w_.tok().encode(sg, /*add_special=*/true);
            if (!T.te_.encode(tids, e)) return false;
            ms = ms_since(t0);
            return true;
        };
    }
    if (!T.build_prompt(L.chunks[L.ci], gp, ref, ids, embeds, L.r.text_enc_ms, L.pf ? &lookup : nullptr)) {
        fail(L, T.error_msg_); return false;
    }
    L.r.n_prompt_tokens = (int) ids.size();
    auto t0 = clk::now();
    if (!B.prefill_embeds(embeds.data(), (int) ids.size(), L.hidden, L.logits)) {
        fail(L, B.get_error()); return false;
    }
    if (L.use_cfg) {
        gen_params neg = gp;
        neg.instruction.clear();
        std::vector<int32_t> nids;
        std::vector<float> nembeds;
        double nte = 0;
        if (!T.build_prompt(L.chunks[L.ci], neg, ref, nids, nembeds, nte, L.pf ? &lookup : nullptr)) {
            fail(L, T.error_msg_); return false;
        }
        L.r.text_enc_ms += nte;
        Backbone & N = bb_neg(L.slot);
        N.reset();
        if (!N.prefill_embeds(nembeds.data(), (int) nids.size(), L.neg_hidden, L.neg_logits)) {
            fail(L, N.get_error()); return false;
        }
    }
    L.r.prefill_ms = ms_since(t0);

    L.rng = gp.seed ? gp.seed : 0x9E3779B9u;
    L.c0_hist.clear();
    L.step = 0;
    // EOS floor. Breeze occasionally ends a chunk early -- the last two
    // sentences of a 64-word chunk never spoken, the request still ok, and the
    // aligner squeezing the missing words into the last 40 ms so the
    // read-along highlight races to the end (1 in 72 long-form takes; with the
    // floor, 1 in 24 tried to end at 13 s of ~21 s and was held, transcript
    // complete). It is the model's: single generations do it too. The voice's
    // own reference gives its pace (frames per word); a chunk of 12+ words may
    // not end before 70% of that pace's length. Takes of the same text vary
    // about +/-7% in length, so a genuine ending never comes near the floor.
    {
        auto words_in = [](const std::string & t) {
            int n = 0; bool in = false;
            for (char ch : t) { const bool sp = ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r';
                                if (!sp && !in) { ++n; in = true; } else if (sp) in = false; }
            return n;
        };
        const int nw = words_in(L.chunks[L.ci]);
        const auto & rv = L.req.ref;
        const int rw = rv.T > 0 ? words_in(rv.ref_text) : 0;
        const double fpw = (rv.T > 0 && rw >= 10) ? (double) rv.T / rw : 4.0;
        static const double frac = [] {
            const char * e = getenv("BREEZE_EOS_FLOOR");
            return e ? atof(e) : 0.7;
        }();
        L.min_frames = (nw >= 12 && frac > 0) ? (int) (frac * nw * fpw) : 0;
    }
    L.emitted = 0;
    L.next_block = L.streaming ? L.req.stream_chunk_frames : 0;
    L.max_block  = L.streaming ? std::max(L.req.stream_chunk_frames, 48) : 0;
    L.first_block = true;
    if (L.streaming) {
        T.cc_.swap_stream(L.voc);
        T.cc_.stream_reset(gp.max_new_frames);
        T.cc_.swap_stream(L.voc);
    }
    L.t_dec = clk::now();
    L.need_prefill = false;
    L.text_was_ready = false;

    // Encode the next chunk while this one generates, guessing that this
    // chunk will be carried whole (it is unless it overruns ref_max_frames,
    // and text_ready() re-requests on a miss). Drop what no longer applies.
    if (L.pf) {
        std::set<std::string> keep;
        if (L.ci + 1 < L.chunks.size()) {
            // Anchored, the next chunk's reference is the voice again.
            const bool anchored = anchor_long_ref() && L.req.ref.T > 0;
            const auto nx = segs_for(L, L.ci + 1,
                                     anchored ? L.req.ref.ref_text : L.chunks[L.ci], true);
            keep.insert(nx.begin(), nx.end());
            want_text(L, nx);
        }
        std::lock_guard<std::mutex> lk(L.pf->m);
        for (auto it = L.pf->segs.begin(); it != L.pf->segs.end();)
            if (it->second && !keep.count(it->first)) it = L.pf->segs.erase(it); else ++it;
    }
    return true;
}

bool BreezeScheduler::pre_step(Lane & L) {
    const auto & bbc = tts_.w_.cfg().bb;
    const int NC = bbc.n_codebooks, CB = tts_.w_.cfg().cc.codebook_size;
    const int V = bbc.audio_vocab, LM = bbc.lm_head_size, EOS = V;
    const auto & gp = L.req.gp;

    if (L.cancelled()) { finish_chunk(L, true); return false; }   // a cancel is not a truncation
    if (L.step >= gp.max_new_frames) { finish_chunk(L, false); return false; }
    if ((int) L.logits.size() < LM) { fail(L, "ar: short logits"); return false; }
    auto & logits = L.logits;
    if (L.use_cfg) {
        if ((int) L.neg_logits.size() < LM) { fail(L, "ar: short cfg logits"); return false; }
        for (int i = 0; i < LM; ++i)
            logits[i] = L.neg_logits[i] + gp.cfg_scale * (logits[i] - L.neg_logits[i]);
    }
    if (gp.repetition_penalty != 1.0f && !L.c0_hist.empty()) {
        for (int id : L.c0_hist) {
            if (id < 0 || id >= LM) continue;
            logits[id] = logits[id] < 0 ? logits[id] * gp.repetition_penalty
                                        : logits[id] / gp.repetition_penalty;
        }
    }
    for (int i = CB; i < V; ++i) logits[i] = -INFINITY;
    int c0 = sample_logits(logits, LM, gp.temperature, gp.top_k, gp.top_p, L.rng);
    if (c0 == EOS && L.step < L.min_frames) {
        fprintf(stderr, "breeze-sched: lane %d chunk %zu: EOS at frame %d held (floor %d)\n",
                L.slot, L.ci, L.step, L.min_frames);
        logits[EOS] = -INFINITY;
        c0 = sample_logits(logits, LM, gp.temperature, gp.top_k, gp.top_p, L.rng);
    }
    if (c0 == EOS) { finish_chunk(L, true); return false; }
    L.c0_hist.push_back(c0);
    L.frame.assign(NC, 0);
    L.frame[0] = c0;
    return true;
}

// After the depth decoder filled L.frame: record it and emit any due codec
// block (run_ar's order: depth, vocoder, then the backbone step).
void BreezeScheduler::post_frame(Lane & L, const std::vector<int32_t> & frame) {
    BreezeTTS & T = tts_;
    const int NC = T.w_.cfg().bb.n_codebooks;
    L.r.codes.insert(L.r.codes.end(), frame.begin(), frame.end());
    L.r.T++;
    if (L.streaming && L.r.T - L.emitted >= L.next_block) {
        auto t_c = clk::now();
        std::vector<float> chunk;
        const int n_new = L.r.T - L.emitted;
        T.cc_.swap_stream(L.voc);
        const bool ok = T.cc_.stream_decode(L.r.codes.data() + (size_t) L.emitted * NC, n_new, chunk);
        T.cc_.swap_stream(L.voc);
        if (!ok) { fail(L, "vocoder stream_decode failed"); return; }
        L.r.codec_ms += ms_since(t_c);
        if (!chunk.empty()) {
            emit(L, chunk.data(), (int) chunk.size(), false);
            if (L.first_block) { L.r.ttfa_ms = ms_since(L.t_start); L.first_block = false; }
            L.r.pcm.insert(L.r.pcm.end(), chunk.begin(), chunk.end());
        }
        L.emitted = L.r.T;
        L.next_block = std::min(L.next_block * 2, L.max_block);
    }
}

void BreezeScheduler::finish_chunk(Lane & L, bool hit_eos) {
    BreezeTTS & T = tts_;
    const int NC = T.w_.cfg().bb.n_codebooks;
    const auto & gp = L.req.gp;
    gen_result & r = L.r;
    const bool last = L.last_chunk();

    r.truncated = !hit_eos;
    if (r.truncated)
        fprintf(stderr, "breeze: TRUNCATED at max_new_frames=%d (%.1f s) -- the text did not "
                        "finish; use long-form chunking or raise max_new_frames\n",
                gp.max_new_frames, gp.max_new_frames / 12.5);
    r.decode_ms = ms_since(L.t_dec) - r.codec_ms;

    auto t_c = clk::now();
    if (r.T > 0) {
        std::vector<float> pcm;
        T.cc_.swap_stream(L.voc);
        bool ok = true;
        if (L.streaming) {
            const int n_new = r.T - L.emitted;
            if (n_new > 0) ok = T.cc_.stream_decode(r.codes.data() + (size_t) L.emitted * NC, n_new, pcm);
        } else {
            ok = T.cc_.decode(r.codes.data(), r.T, pcm);
        }
        T.cc_.swap_stream(L.voc);
        if (!ok) { fail(L, "vocoder decode failed"); return; }
        r.codec_ms += ms_since(t_c);
        if (L.streaming) {
            if (!pcm.empty()) emit(L, pcm.data(), (int) pcm.size(), last);
            else              emit(L, nullptr, 0, last);
            if (!pcm.empty() && L.first_block) { r.ttfa_ms = ms_since(L.t_start); L.first_block = false; }
            r.pcm.insert(r.pcm.end(), pcm.begin(), pcm.end());
        } else {
            r.pcm = std::move(pcm);
        }
    } else if (L.streaming) {
        emit(L, nullptr, 0, last);
    }

    // synthesize_long's aggregation and rolling carry.
    gen_result & out = L.out;
    out.pcm.insert(out.pcm.end(), r.pcm.begin(), r.pcm.end());
    out.codes.insert(out.codes.end(), r.codes.begin(), r.codes.end());
    out.T += r.T;
    out.prefill_ms += r.prefill_ms; out.decode_ms += r.decode_ms;
    out.codec_ms += r.codec_ms;     out.text_enc_ms += r.text_enc_ms;
    out.truncated |= r.truncated;
    if (L.ci == 0) { out.ttfa_ms = r.ttfa_ms; out.n_prompt_tokens = r.n_prompt_tokens; }

    if (L.req.long_form) {
        const int used_ref_T = L.hist.T;
        const bool used_voice_only = L.hist.T > 0 && L.hist_text.empty();
        const ref_voice * ref = L.req.ref.T > 0 ? &L.req.ref : nullptr;
        const int keep = std::min(r.T, L.req.ref_max_frames);
        if (keep == r.T && !(anchor_long_ref() && ref)) {
            L.hist.codes.assign(r.codes.end() - (size_t) keep * NC, r.codes.end());
            L.hist.T = keep;
            L.hist_text = L.chunks[L.ci];
        } else if (ref) {
            L.hist = *ref;
            L.hist_text = ref->ref_text;
        } else {
            L.hist.codes.assign(r.codes.end() - (size_t) keep * NC, r.codes.end());
            L.hist.T = keep;
            L.hist_text.clear();
        }
        fprintf(stderr, "  [breeze long-form lane %d] chunk %zu/%zu: prompt=%d tok -> %d frames "
                "(%.1fs), ref_T used=%d%s, carry=%d | text_enc %.0f ms, prefill %.0f ms\n",
                L.slot, L.ci + 1, L.chunks.size(), r.n_prompt_tokens, r.T, r.T / 12.5, used_ref_T,
                used_voice_only ? " (voice-only)" : "", L.hist.T, r.text_enc_ms, r.prefill_ms);
    }

    L.ci++;
    if (L.ci >= L.chunks.size() || L.cancelled()) {
        L.ci = L.chunks.size();
        if (L.req.on_done) L.req.on_done(true, out, std::string());
        L.req.on_done = nullptr;
    } else {
        L.need_prefill = true;
    }
}

namespace {
// BREEZE_SCHED_PROF=1: per-stage wall time of the scheduler loop, dumped every
// 500 ticks, split by how many lanes were generating.
struct SchedProf {
    double prefill = 0, sample = 0, depth = 0, voc = 0, backbone = 0;
    long ticks[3] = {0, 0, 0};
    static bool on() { static const bool v = std::getenv("BREEZE_SCHED_PROF") != nullptr; return v; }
    void dump() {
        const long n = ticks[0] + ticks[1] + ticks[2];
        if (!n) return;
        const double t = prefill + sample + depth + voc + backbone;
        fprintf(stderr, "  [breeze-sched-prof] %ld ticks (1 lane %ld, 2 lanes %ld) | total %.0f ms: "
                "prefill %.0f, sample %.0f, depth %.0f, vocoder %.0f, backbone %.0f | "
                "per tick %.2f ms\n", n, ticks[1], ticks[2], t, prefill, sample, depth, voc, backbone,
                t / (double) n);
        *this = SchedProf{};
    }
};
SchedProf g_sp;
} // namespace

bool BreezeScheduler::tick() {
    BreezeTTS & T = tts_;
    const bool P = SchedProf::on();
    clk::time_point tp = clk::now();
    auto lap = [&](double & acc) { if (P) { acc += ms_since(tp); tp = clk::now(); } };
    bool worked = false;
    // One prefill per tick, round robin, so two arrivals never stall a running
    // lane twice back to back.
    {
        const size_t n = lanes_.size();
        for (size_t k = 0; k < n; ++k) {
            Lane & L = *lanes_[(prefill_rr_ + k) % n];
            if (!L.need_prefill) continue;
            if (L.cancelled()) {   // cancelled between chunks (or before starting)
                prefill_rr_ = (prefill_rr_ + k + 1) % n;
                L.ci = L.chunks.size(); L.need_prefill = false;
                if (L.req.on_done) L.req.on_done(true, L.out, std::string());
                L.req.on_done = nullptr;
                worked = true;
                break;
            }
            double min_slack = 1e9;
            for (auto & o : lanes_) if (o.get() != &L) min_slack = std::min(min_slack, slack_s(*o));
            bool sync_text = false;
            if (!text_ready(L)) {
                // A newcomer whose first text is still with the CPU encoder
                // (which competes with the CPU aligner): when everyone else has
                // audio to spare, encoding it here costs them ~0.1-0.3 s and
                // gets this listener going a second sooner.
                static const bool newcomer_gpu = [] {
                    const char * e = std::getenv("BREEZE_NEWCOMER_GPU_TEXT");
                    return !(e && !std::strcmp(e, "0"));
                }();
                if (!(newcomer_gpu && L.ci == 0 && !L.has_audio && min_slack >= 1.5)) continue;
                sync_text = true;
            }
            if (!L.text_was_ready) { L.text_was_ready = true; L.t_ready = clk::now(); }
            if (!L.has_audio && !sync_text) {
                // A newcomer: let lanes that have just started playing build a
                // little buffer before stalling them for this prefill.
                const double waited = std::chrono::duration<double>(clk::now() - L.t_ready).count();
                if (min_slack < 1.0 && waited < 2.5) continue;
            }
            prefill_rr_ = (prefill_rr_ + k + 1) % n;
            if (L.ci == 0)
                fprintf(stderr, "breeze-sched: lane %d first prefill %.0f ms after admission "
                        "(text ready at %.0f ms)\n", L.slot, ms_since(L.t_admit),
                        std::chrono::duration<double, std::milli>(L.t_ready - L.t_admit).count());
            start_chunk(L, sync_text);
            worked = true;
            break;
        }
    }
    lap(g_sp.prefill);

    // Lanes generating this tick: sample codebook 0 for each.
    std::vector<Lane *> act;
    for (auto & lp : lanes_) {
        Lane & L = *lp;
        if (L.need_prefill || L.ci >= L.chunks.size()) continue;
        if (pre_step(L)) act.push_back(&L);
    }
    lap(g_sp.sample);
    if (act.empty()) return worked;
    if (P) {
        g_sp.ticks[std::min<size_t>(act.size(), 2)]++;
        if (g_sp.ticks[1] + g_sp.ticks[2] >= 500) g_sp.dump();
    }
    // Slot order, so the batched backbone step is always slot 0's pair graph
    // with slot 1 as the partner -- one cached graph, not one per ordering.
    std::sort(act.begin(), act.end(), [](const Lane * a, const Lane * b) { return a->slot < b->slot; });

    static const bool cfg_batch = [] {
        const char * e = std::getenv("BREEZE_CFG_BATCH");
        return !(e && !std::strcmp(e, "0"));
    }();
    static const bool lane_batch = [] {
        const char * e = std::getenv("BREEZE_LANE_BATCH");
        return !(e && !std::strcmp(e, "0"));
    }();
    const bool paired = act.size() == 2 && lane_batch && !act[0]->use_cfg && !act[1]->use_cfg;

    // Depth decoder.
    if (paired) {
        Lane & A = *act[0]; Lane & B = *act[1];
        const DepthDecoder::sample_params sa{A.req.gp.depth_temperature, A.req.gp.depth_top_k, A.req.gp.depth_top_p};
        const DepthDecoder::sample_params sb{B.req.gp.depth_temperature, B.req.gp.depth_top_k, B.req.gp.depth_top_p};
        if (!T.dd_.run_frame_pair(T.dd_neg_, A.hidden.data(), B.hidden.data(), A.frame.data(),
                                  B.frame.data(), sa, sb, A.rng, B.rng)) {
            const std::string e = T.dd_.get_error();
            fail(A, e); fail(B, e); return true;
        }
    } else {
        for (Lane * Lp : act) {
            Lane & L = *Lp;
            const auto & gp = L.req.gp;
            bool ok;
            if (L.use_cfg)
                ok = T.dd_.run_frame_cfg(T.dd_neg_, L.hidden.data(), L.neg_hidden.data(), L.frame.data(),
                                         gp.cfg_scale, cfg_batch, gp.depth_temperature,
                                         gp.depth_top_k, gp.depth_top_p, L.rng);
            else
                ok = T.dd_.run_frame(L.hidden.data(), L.frame.data(), gp.depth_temperature,
                                     gp.depth_top_k, gp.depth_top_p, L.rng);
            if (!ok) { fail(L, T.dd_.get_error()); L.frame.clear(); }
        }
    }

    lap(g_sp.depth);
    std::vector<Lane *> dec;
    for (Lane * Lp : act) {
        Lane & L = *Lp;
        if (L.frame.empty() || L.ci >= L.chunks.size()) continue;   // failed
        post_frame(L, L.frame);
        if (L.ci >= L.chunks.size()) continue;                       // vocoder failed
        // run_ar skips the backbone step after the last allowed frame.
        if (L.step + 1 < L.req.gp.max_new_frames) dec.push_back(&L);
        L.step++;
    }

    lap(g_sp.voc);
    // Backbone.
    struct BbLap { decltype(lap) & f; ~BbLap() { f(g_sp.backbone); } } bb_lap{lap};
    if (dec.size() == 2 && paired) {
        Lane & A = *dec[0]; Lane & B = *dec[1];
        Backbone & BA = bb(A.slot);
        if (!BA.decode_frame_pair(bb(B.slot), A.frame.data(), B.frame.data(), A.hidden, A.logits,
                                  B.hidden, B.logits)) {
            const std::string e = BA.get_error();
            fail(A, e); fail(B, e);
        }
        return true;
    }
    for (Lane * Lp : dec) {
        Lane & L = *Lp;
        Backbone & B = bb(L.slot);
        if (L.use_cfg && cfg_batch) {
            if (!B.decode_frame_pair(bb_neg(L.slot), L.frame.data(), L.hidden, L.logits,
                                     L.neg_hidden, L.neg_logits))
                fail(L, B.get_error());
        } else {
            if (!B.decode_frame(L.frame.data(), L.hidden, L.logits)) { fail(L, B.get_error()); continue; }
            if (L.use_cfg && !bb_neg(L.slot).decode_frame(L.frame.data(), L.neg_hidden, L.neg_logits))
                fail(L, bb_neg(L.slot).get_error());
        }
    }
    return true;
}

void BreezeScheduler::loop() {
    for (;;) {
        std::deque<std::function<void()>> jobs;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] {
                return stop_ || !jobs_.empty() || !lanes_.empty() ||
                       (!queue_.empty() && (int) lanes_.size() < max_lanes_);
            });
            if (stop_) break;
            jobs.swap(jobs_);
            // Admit into free slots.
            while (!queue_.empty() && (int) lanes_.size() < max_lanes_) {
                auto L = std::make_unique<Lane>();
                L->req = std::move(queue_.front());
                queue_.pop_front();
                bool used[2] = { false, false };
                for (auto & o : lanes_) used[o->slot] = true;
                L->slot = used[0] ? 1 : 0;
                lanes_.push_back(std::move(L));
            }
        }
        for (auto & j : jobs) j();

        for (auto & lp : lanes_) {
            Lane & L = *lp;
            if (!L.chunks.empty() || L.ci > 0) continue;
            // Freshly admitted.
            if (!tts_.loaded()) {
                L.chunks.push_back(L.req.text);   // so fail() has a range to close
                fail(L, "engine not loaded");
                continue;
            }
            L.streaming = L.req.stream_chunk_frames > 0 && L.req.on_pcm != nullptr;
            L.chunks = L.req.long_form
                ? BreezeTTS::chunk_text(L.req.text, L.req.chunk_words, L.req.first_chunk_words)
                : std::vector<std::string>{ L.req.text };
            if (L.chunks.empty()) {
                L.chunks.push_back(std::string());
                fail(L, "no chunks");
                continue;
            }
            if (L.req.ref_max_frames <= 0) L.req.ref_max_frames = 600;
            const int gap_ms = L.req.long_form ? std::max(0, L.req.gap_ms) : 0;
            L.gap.assign((size_t) (gap_ms * tts_.w_.cfg().cc.sample_rate / 1000), 0.0f);
            if (L.req.ref.T > 0) { L.hist = L.req.ref; L.hist_text = L.req.ref.ref_text; }
            static const bool prefetch_on = [] {
                const char * e = std::getenv("BREEZE_TEXT_PREFETCH");
                return !(e && !std::strcmp(e, "0"));
            }();
            if (prefetch_on && tts_.w_.te_on_cpu() && !te_failed_) {
                L.pf = std::make_shared<Prefetch>();
                // The voice's transcript (segment 0 when there is a voice) is cacheable.
                want_text(L, segs_for(L, 0, L.hist_text, L.hist.T > 0), L.hist.T > 0 ? 1 : 0, true);
            }
            L.t_admit = clk::now();
            fprintf(stderr, "breeze-sched: admitted lane %d (%zu chunk%s, %zu active)\n", L.slot,
                    L.chunks.size(), L.chunks.size() == 1 ? "" : "s", lanes_.size());
        }

        const bool worked = tick();

        // Retire finished lanes.
        for (size_t i = 0; i < lanes_.size();) {
            Lane & L = *lanes_[i];
            if (L.ci >= L.chunks.size() && !L.need_prefill) {
                tts_.cc_.free_stream_slot(L.voc);
                lanes_.erase(lanes_.begin() + (long) i);
                if (prefill_rr_ >= lanes_.size()) prefill_rr_ = 0;
            } else {
                ++i;
            }
        }
        // Lanes waiting on text or on admission: sleep until the text thread
        // or a submit wakes us (a 5 ms poll covers the slack check).
        if (!worked && !lanes_.empty()) {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait_for(lk, std::chrono::milliseconds(5));
        }
    }
    for (auto & lp : lanes_) tts_.cc_.free_stream_slot(lp->voc);
    lanes_.clear();
}

} // namespace breeze
