#pragma once

// Concurrent requests on ONE engine: a single GPU thread steps up to
// `max_lanes` requests ("lanes") frame by frame and, when two are generating
// at once, runs their backbone and depth-decoder steps as batch-2 graphs, so
// the weights -- which is what a step costs -- are read once for both.
//
// Why not two engines or two threads: one stream already saturates the card,
// so two processes just time-slice it (measured on the 3060: 0.96 RTF each,
// worse than running them back to back), and two threads on one engine would
// still serialise on every graph. Batching is the only way two streams cost
// less than twice one.
//
// What a lane owns: its backbone KV (slot 0 uses the engine's own, slot 1 a
// second Backbone sharing the prefill scheduler), its streaming-vocoder state
// (swapped into the shared vocoder around each decode), RNG, repetition
// history and long-form carry. Shared: weights, text encoder, depth decoder
// (its KV holds nothing between frames), vocoder weights and scheduler.
//
// Exactness: a request that runs alone takes exactly the calls BreezeTTS's
// own synthesize_long/run_ar make, so its audio is bit-identical to the
// unbatched engine. While two run together the batched kernels reassociate
// the arithmetic, so their audio is equivalent, not identical; each lane
// still consumes its own RNG in the same order, so no lane's sampling depends
// on the other's.
//
// A CFG request (cfg_scale != 1 with an instruction) is already a batch-2
// decode with its own negative branch; it is not batched again with another
// request (that would be batch 3/4 with graphs we do not have). The lanes
// then step one after the other inside a tick -- correct, just not faster.
//
// Text encoding is off the GPU thread. The T5Gemma2 text encoder lives in host
// RAM (BREEZE_TEXT_ENC_CPU=1) and took 400-580 ms per chunk -- ten times the
// prompt prefill -- while every lane waited. A second, CPU-only encoder on its
// own thread now encodes each lane's NEXT chunk while the current one is being
// generated (a chunk's text segments are known ahead: its own text, and the
// previous chunk's text as the rolling reference's transcript), so a chunk
// boundary costs only the GPU prefill (~30-70 ms).
//
// Admission: a new request's prompt prefill runs between ticks and pauses the
// other lane for that long. If the other lane has only just started playing
// (under ~1 s of audio buffered ahead of a 1x playhead), the newcomer waits --
// at most 2.5 s -- so a second listener arriving never makes the first one's
// opening stutter. One prefill per tick.

#include "breeze_tts.h"

#include <condition_variable>
#include <deque>
#include <map>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace breeze {

struct LaneRequest {
    std::string text;
    gen_params  gp;
    ref_voice   ref;                     // T == 0 => no clone voice
    bool long_form = true;               // false => the whole text is one chunk
    int  chunk_words = 120, ref_max_frames = 600, gap_ms = 180, first_chunk_words = 0;
    int  stream_chunk_frames = 0;        // > 0 streams codec blocks; 0 = buffered
    BreezeTTS::pcm_cb   on_pcm;          // streaming only
    BreezeTTS::chunk_cb on_chunk_start;  // streaming only
    // Fired exactly once, on the scheduler thread. `cancelled` reflects the
    // lane's cancel flag at the end (a cancel is a successful partial render).
    std::function<void(bool ok, gen_result & r, const std::string & err)> on_done;
    std::shared_ptr<std::atomic<bool>> cancel;
};

class BreezeScheduler {
public:
    BreezeScheduler(BreezeTTS & tts, int max_lanes);
    ~BreezeScheduler();

    void submit(LaneRequest r);
    // Runs `job` on the GPU thread between ticks and waits for it.
    void run_sync(const std::function<void()> & job);

    int max_lanes() const { return max_lanes_; }

private:
    struct Lane;
    void loop();
    // sync_text: encode any text segment not yet prefetched right here, with
    // the engine's own (device-offloaded) encoder.
    bool start_chunk(Lane & L, bool sync_text = false);
    // Samples codebook 0. false => the chunk ended (EOS, cap or cancel) and
    // finish_chunk has run.
    bool pre_step(Lane & L);
    void post_frame(Lane & L, const std::vector<int32_t> & frame);
    void finish_chunk(Lane & L, bool hit_eos);
    void fail(Lane & L, const std::string & err);
    bool tick();
    // Text prefetch (see header comment).
    struct EncSeg { std::vector<int32_t> ids; std::vector<float> emb; double ms = 0; bool ok = false; };
    struct Prefetch {
        std::mutex m;
        std::map<std::string, std::shared_ptr<EncSeg>> segs;   // null = queued
    };
    // Two encoder threads: one takes any job, the other only urgent ones, so a
    // newcomer's first chunk never queues behind an in-progress speculative
    // encode (measured: 0.3 s of encoding waited 1.2 s).
    void text_loop(bool urgent_only);
    std::vector<std::string> segs_for(const Lane & L, size_t ci, const std::string & hist_text,
                                      bool have_hist) const;
    // urgent: a lane is (or is about to be) blocked on it -- ahead of the
    // speculative next-chunk encodes in the queue.
    void want_text(Lane & L, const std::vector<std::string> & segs, size_t cacheable = 0,
                   bool urgent = false);
    bool text_ready(Lane & L);
    double slack_s(const Lane & L) const;
    void emit(Lane & L, const float * pcm, int n, bool fin);

    Backbone & bb(int slot);
    Backbone & bb_neg(int slot);
    bool ensure_slot(int slot, bool cfg);

    BreezeTTS & tts_;
    const int max_lanes_;
    std::unique_ptr<Backbone> bb1_, bb1_neg_;
    bool bb1_ready_ = false, bb1_neg_ready_ = false;

    std::vector<std::unique_ptr<Lane>> lanes_;   // active
    size_t prefill_rr_ = 0;

    std::mutex m_;
    std::condition_variable cv_;
    std::deque<LaneRequest> queue_;
    std::deque<std::function<void()>> jobs_;
    bool stop_ = false;
    std::thread th_;

    std::atomic<bool> te_failed_{false};
    struct TextJob { std::shared_ptr<Prefetch> pf; std::string seg; bool cache = false; bool urgent = false; };
    std::deque<TextJob> text_q_;
    // A voice's reference transcript opens every request in that voice, so its
    // encoding is kept across requests (a few voices, LRU by insertion).
    std::mutex vcache_m_;
    std::deque<std::pair<std::string, std::shared_ptr<EncSeg>>> vcache_;
    std::condition_variable text_cv_;
    std::thread text_th_, text_urgent_th_;
};

} // namespace breeze
