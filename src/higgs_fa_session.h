// higgs_fa_session.h — forced-alignment (word-highlight read-along) sibling for
// the Higgs-Audio-v3 server. Self-contained (namespace `higgs`); the only
// shared code is the genuinely engine-agnostic `qwen3_fa` aligner library
// (cstr/qwen3-forced-aligner-0.6b): it takes (audio PCM + word list) → per-word
// t0/t1 and aligns ANY audio against text, so it serves higgs unchanged.
//
// Architecture mirrors qwen3-tts's aligner sibling exactly:
//   - A SEPARATE aligner-only subprocess (spawned via "--higgs-aligner <fd>")
//     with its OWN CUDA context, so its encode+align passes run concurrently
//     with the higgs synth worker on the same GPU (no shared-stream churn).
//   - The parent streams synthesised PCM deltas (24 kHz on higgs — fed to the
//     aligner as pcm_sample_rate=24000 so its mel front-end resamples right)
//     and gets back per-word timings; word timings are deterministic and
//     bit-identical between partial and one-shot (single argmax, no sampling).
//   - Aligner pid VRAM ~0.6-1.1 GB (Q4_K, 28 LLM layers on CPU by default via
//     CRISPASR_N_GPU_LAYERS); SIGKILL on idle-unload → VRAM true-0.
//
// The gate makes the two TTS engines (qwen3/higgs) mutually exclusive, so at
// most one TTS engine + one aligner is ever resident.

#ifndef HIGGS_FA_SESSION_H
#define HIGGS_FA_SESSION_H

#include "higgs_worker_ipc.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace higgs {

// One word's forced-aligned position in the synthesised audio. Times are ms
// from start-of-audio. `confidence` is the softmax-top1 probability of the
// noisier of the two timestamp boundaries, in [1/H, 1]; -1 if unavailable.
struct AlignedWord {
    std::string text;
    int64_t     t0_ms = 0;
    int64_t     t1_ms = 0;
    float       confidence = -1.0f;
};

// Per-stage timing returned alongside a streaming-align result (ms).
struct AlignProfile {
    int64_t t_load_ms     = 0;
    int64_t t_resample_ms = 0;
    int64_t t_aligner_ms  = 0;
    int64_t t_total_ms    = 0;
    int     n_words       = 0;
};

// Parent-side handle on the aligner-only sibling subprocess. One instance per
// server process. Lazy-spawned on first ensure_loaded(); SIGKILL on shutdown()
// (idle-unload) reclaims VRAM true-0. Only the streaming-align methods are
// valid — there is no synth surface here.
class HiggsAlignerSession {
public:
    explicit HiggsAlignerSession(const char * argv0,
                                 std::vector<std::string> extra_argv = {});
    ~HiggsAlignerSession();

    // Spawn + load the FA GGUF if not already loaded with this model path;
    // respawn on path change. Eager-loads the model in the child (overlaps with
    // the synth worker's own cold load). Returns true on success.
    bool ensure_loaded(const std::string & aligner_model);

    // SIGKILL + waitpid. Idempotent. Next ensure_loaded() respawns.
    void shutdown();

    bool is_alive() const { return pid_ > 0; }
    pid_t pid() const     { return pid_; }
    const std::string & last_error() const { return last_error_; }

    // ── Streaming alignment ─────────────────────────────────────────────────
    //   1. begin_streaming_align(words, pcm_sr) — reset accumulator, stash words.
    //   2. push_partial_pcm(pcm, n, audio_seen_ms) — send PCM delta; the worker
    //      re-encodes the cumulative audio and returns updated word timings on
    //      the same fd. Caller services responses via drain_partial_alignments
    //      (typically from a reader thread).
    //   3. finalize_streaming_align(tail, n, audio_total_ms, ...) — last call;
    //      blocks until ALIGN_FINAL_RESP. Resets session state on exit.
    // Single in-flight stream per session; caller serialises externally.
    bool begin_streaming_align(const std::vector<std::string> & words,
                               int pcm_sample_rate);

    bool push_partial_pcm(const float * pcm, size_t n_samples,
                          int64_t audio_seen_ms);

    using PartialAlignCallback = std::function<void(
        int64_t audio_seen_ms,
        const std::vector<AlignedWord> & words)>;
    bool drain_partial_alignments(const PartialAlignCallback & cb);

    bool finalize_streaming_align(const float * tail_pcm, size_t n_tail_samples,
                                  int64_t audio_total_ms,
                                  std::vector<AlignedWord> & out_words,
                                  AlignProfile & out_profile);

private:
    void kill_worker_locked();

    std::string              argv0_;
    std::vector<std::string> extra_argv_;
    std::string              loaded_model_;
    bool                     loaded_ok_ = false;

    pid_t                    pid_ = -1;
    int                      fd_  = -1;
    mutable std::mutex       io_mutex_;
    std::string              last_error_;
    std::atomic<uint32_t>    next_req_id_{1};

    std::vector<std::string> stream_align_words_;
    int                      stream_align_pcm_sr_ = 0;
    bool                     stream_align_active_ = false;
    bool                     stream_align_has_sent_any_ = false;
};

// Aligner-only dispatch loop. Called from main() when "--higgs-aligner <fd>"
// is passed. Loads the FA GGUF (eager on LOAD_REQ or lazy on first
// ALIGN_PARTIAL_REQ), keeps a growing PCM accumulator, re-runs the FA encode +
// LLM body per pass. No higgs engine — only the aligner is in VRAM. Returns
// the process exit code.
int run_higgs_aligner_worker_loop(int fd);

} // namespace higgs

#endif // HIGGS_FA_SESSION_H
