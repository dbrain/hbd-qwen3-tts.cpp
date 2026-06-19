// worker_session.h — parent-side handle on a subprocess worker, plus the
// child-side dispatch loop. See HANDOFF-worker.md (kobbler repo) for the
// design.
//
// In P1 we only support the non-streaming synth path:
//   /v1/audio/speech (response_format=wav, no stream_format)
// Streaming, voice archive, abort handling come in P2-P4.

#ifndef QWEN3_TTS_WORKER_SESSION_H
#define QWEN3_TTS_WORKER_SESSION_H

#include "qwen3_tts.h"
#include "worker_ipc.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <vector>

namespace qwen3_tts {

// Subset of server_params the worker needs to load the model. Kept as
// a flat POD so JSON serialization stays trivial.
struct WorkerLoadConfig {
    std::string model;             // talker GGUF
    std::string vocoder;           // vocoder GGUF
    std::string speaker_encoder;   // optional speaker-encoder GGUF
    std::string voice_archive_dir; // /app/voice-archive — worker scans for *.warmup
    std::string model_id;          // for warmup model_id-tag matching
    std::string aligner_model;     // optional FA GGUF; empty = align endpoint disabled
    bool        lazy_load = false; // worker defers load_model_files() until first synth
    // True when this worker is the parent's *aligner-only* sibling. The
    // worker skips talker/vocoder/spk-encoder loads entirely and only
    // services ALIGN_PARTIAL_REQ / ALIGN_FINAL_REQ / LOAD_REQ / PING /
    // SHUTDOWN. Spawned via `--worker-aligner <fd>` so the dispatch loop
    // can refuse synth requests up front. See HANDOFF-streaming-aligned-tts.md
    // Phase 2 for the architecture rationale (shared GPU, separate CUDA
    // context, no cross-talk between synth and align flows).
    bool        aligner_only = false;
};

// One word's forced-aligned position in the synthesised audio. Times are
// milliseconds from start-of-audio. Filled by WorkerSession::align_words.
// `confidence` is the softmax-top1 probability of the noisier of the two
// timestamp-class predictions for this word, in [1/H, 1]; ~1 = sharp
// peak, near 1/H = near-uniform. -1.0f if the aligner didn't supply one.
struct AlignedWord {
    std::string text;
    int64_t     t0_ms = 0;
    int64_t     t1_ms = 0;
    float       confidence = -1.0f;
};

// Per-stage timing returned alongside an alignment result. All fields in
// milliseconds; zero means the stage wasn't run or wasn't profiled.
struct AlignProfile {
    int64_t t_load_ms     = 0;   // first-call cold load (0 if cached)
    int64_t t_resample_ms = 0;
    int64_t t_mel_ms      = 0;
    int64_t t_encoder_ms  = 0;
    int64_t t_aligner_ms  = 0;   // body forward (embed + kv + run_aligner)
    int64_t t_total_ms    = 0;
    int     n_enc         = 0;   // audio-pad tokens spliced in
    int     n_prompt      = 0;   // total prompt tokens through the LLM
    int     n_words       = 0;
};

// Parent-side handle. One instance per server process. Thread-safe:
// synthesize() takes io_mutex_ for the full request/response round-trip.
class WorkerSession {
public:
    // Configure but don't spawn yet. argv0 must be argv[0] of the
    // running process — used for execv() so the child runs the same
    // binary in --worker mode. (Streaming forced-alignment now lives in the
    // engine-agnostic fa::AlignerSession; this class is synth-only. The
    // one-shot align_words() below still runs in the synth worker.)
    WorkerSession(const char * argv0,
                  std::vector<std::string> extra_argv = {});
    ~WorkerSession();

    // Spawn the worker if it isn't running. If `cfg` differs from the
    // currently-loaded config, the worker is killed + respawned so we
    // load the right model. Returns true on success; sets last_error_.
    bool ensure_loaded(const WorkerLoadConfig & cfg);

    // SIGKILL + waitpid. Idempotent. Subsequent ensure_loaded() respawns.
    void shutdown();

    bool is_alive() const { return pid_ > 0; }
    pid_t pid() const     { return pid_; }
    const std::string & last_error() const { return last_error_; }

    // Default GPU (UUID) for un-targeted/direct requests. Empty = inherit
    // container CUDA_VISIBLE_DEVICES. Standard env across kob services.
    void set_default_gpu(std::string gpu) { default_gpu_ = std::move(gpu); }
    // Per-request GPU override (from the request's `gpu` field). Only updates
    // on a non-empty value, so a worker stays put for un-targeted requests.
    // The next ensure_loaded() relocates (kill+respawn) if it differs.
    void set_next_gpu(const std::string & gpu) {
        if (!gpu.empty()) { std::lock_guard<std::mutex> lk(io_mutex_); next_gpu_ = gpu; }
    }

    // Vocoder output sample rate, populated from LOAD_RESP. The parent
    // doesn't have a Qwen3TTS instance to query directly. Returns 0 if
    // the worker hasn't loaded yet.
    int sample_rate() const { return sample_rate_; }

    // Synthesize via the worker. Mirrors Qwen3TTS::synthesize{,_with_embedding}
    // signatures so the HTTP handler call site is a 1-line swap.
    //
    // P1 limitations:
    //  - no streaming (the result is the whole audio in tts_result.audio).
    //  - request is serialized: one synth at a time per worker.
    //  - on IPC error / worker crash, the worker is reaped and the next
    //    call respawns it (but the in-flight call returns failure).
    tts_result synthesize(const std::string & text, const tts_params & params);

    // ref_codes points to a buffer of `n_ref_codes` int32s (= n_ref_frames *
    // n_codebooks). n_ref_frames is the timestep count consumed by the
    // synth API. Tracking both separately because the IPC needs the full
    // byte length AND the synth API takes the frame count.
    tts_result synthesize_with_embedding(
        const std::string & text,
        const float * embedding, int32_t embedding_size,
        const tts_params & params,
        const int32_t * ref_codes = nullptr,
        int32_t n_ref_codes = 0,
        int32_t n_ref_frames = 0);

    // P2 — streaming synth dispatch. Worker installs a streaming_opts
    // with an on_pcm that sends AUDIO_FRAME chunks back; parent's
    // `on_pcm` callback receives them and forwards to the HTTP sink.
    // The aggregate audio is NOT accumulated in the returned tts_result
    // (`audio` is empty); the caller streams via `on_pcm` and uses the
    // returned tts_result purely for metadata + cache keys.
    //
    // If `on_pcm` returns false (e.g. HTTP client disconnect → sink
    // write fails), this method keeps draining frames from the worker
    // until SYNTH_DONE/SYNTH_ERR — the worker's send_frame back-pressures
    // through the socket; we don't want to leak a half-consumed stream.
    using StreamCallback = std::function<bool(const float * pcm, size_t n_samples)>;

    tts_result synthesize_with_embedding_streaming(
        const std::string & text,
        const float * embedding, int32_t embedding_size,
        const tts_params & params,
        int32_t stream_batch_size, int32_t stream_first_batch_size,
        const int32_t * ref_codes, int32_t n_ref_codes,
        int32_t n_ref_frames,
        StreamCallback on_pcm);

    tts_result synthesize_streaming(
        const std::string & text,
        const tts_params & params,
        int32_t stream_batch_size, int32_t stream_first_batch_size,
        StreamCallback on_pcm);

    // P3 — voice registration via worker. Both methods serialize on
    // io_mutex_ like the synth path. Failure modes: IPC error → kill +
    // respawn-on-next-call (returns false). last_error() carries detail.
    //
    // extract_speaker_embedding takes a filesystem path readable by
    // the worker (safe in our case: parent + worker share container fs,
    // tmpfile under /tmp lives in both views).
    bool extract_speaker_embedding(const std::string & wav_path,
                                   std::vector<float> & out_embedding);

    bool encode_speech_codes(const float * samples, int32_t n_samples,
                             std::vector<int32_t> & out_codes,
                             int32_t & out_n_ref_frames);

    // P3 — fire-and-forget warmup blob persistence. Worker calls
    // tts.save_voice_warmup() on its in-memory prefill snapshot. Best-
    // effort; failure is logged but doesn't propagate to the user.
    bool save_voice_warmup(const std::string & voice_id,
                           uint64_t prefill_cache_key,
                           uint64_t ref_codes_hash,
                           const std::string & path,
                           const std::string & model_id);

    // Forced alignment against the worker's last-synth PCM buffer. The
    // call is only valid immediately after a synthesize{,_streaming,
    // _with_embedding,...} on the same session — the worker keeps the
    // PCM from the most recent SYNTH_REQ in an internal scratch buffer
    // and discards it on the next SYNTH_REQ or on shutdown.
    //
    // `words` is the whitespace-split word list to align; the caller is
    // responsible for splitting the input text the same way the UI will
    // display it. Returns false on IPC error / aligner load failure /
    // word-count mismatch (timestamp markers != 2*N); last_error() carries
    // detail. On success, out_words gets one entry per input word and
    // out_profile is populated with per-stage timings (zeros if the
    // worker didn't profile this call).
    bool align_words(const std::vector<std::string> & words,
                     std::vector<AlignedWord>        & out_words,
                     AlignProfile                    & out_profile);

    // (Streaming partial-highlight alignment moved to fa::AlignerSession — see
    // src/fa_session.h. It is engine-agnostic and shared with higgs-server.)

    // Send CANCEL_REQ to the worker for the currently in-flight synth.
    // Idempotent: if no synth is in flight, this is a no-op. Safe to
    // call from any thread (uses a separate send_mutex_ so it doesn't
    // contend with the synth call holding io_mutex_). Worker matches
    // req_id, sets request_cancel() on its Qwen3TTS instance, and the
    // talker AR loop / ggml abort callback bails within ~one graph
    // eval. Worker still emits SYNTH_DONE (with success=false) so the
    // parent drain loop terminates cleanly.
    void cancel_in_flight();

private:
    // Send LOAD_REQ, wait for LOAD_RESP. Caller must hold io_mutex_.
    bool send_load_req_locked(const WorkerLoadConfig & cfg);

    // Build SYNTH_REQ payload + send + receive SYNTH_RESP. Caller must
    // hold io_mutex_. On failure, marks the worker dead and returns a
    // tts_result with success=false.
    tts_result do_synth_locked(
        const std::string & text,
        const float * embedding, int32_t embedding_size,
        const tts_params & params,
        const int32_t * ref_codes, int32_t n_ref_codes,
        int32_t n_ref_frames,
        int32_t stream_batch_size,
        int32_t stream_first_batch_size,
        StreamCallback on_pcm);

    // SIGKILL + waitpid. Caller must hold io_mutex_.
    void kill_worker_locked();

    std::string                argv0_;
    std::vector<std::string>   extra_argv_;
    WorkerLoadConfig           loaded_cfg_;
    bool                       loaded_ok_ = false;
    std::string                default_gpu_;   // CVD for un-targeted spawns
    std::string                next_gpu_;      // pending per-request target
    std::string                worker_gpu_;    // GPU the live worker is pinned to

    pid_t                      pid_ = -1;
    int                        fd_  = -1;
    int                        sample_rate_ = 0;
    mutable std::mutex         io_mutex_;
    // Held only by cancel_in_flight() while it writes CANCEL_REQ to fd_.
    // Distinct from io_mutex_ so cancel can be issued from another
    // thread while io_mutex_ is held by the synth's recv loop. The
    // synth path itself only writes SYNTH_REQ once at the very start
    // (before recv begins), so in practice the send_mutex_ is uncontended.
    mutable std::mutex         send_mutex_;
    std::string                last_error_;
    std::atomic<uint32_t>      next_req_id_{1};
    // Set by do_synth_locked while a SYNTH_REQ is in flight (between the
    // SYNTH_REQ send and the final SYNTH_DONE/SYNTH_RESP/SYNTH_ERR).
    // Read by cancel_in_flight() to know which req_id to target. 0 means
    // no synth in flight → cancel is a no-op.
    std::atomic<uint32_t>      current_synth_req_id_{0};
};

// Worker-side dispatch loop. Called from main() when --worker <fd> is
// passed. Owns a Qwen3TTS instance, services LOAD_REQ + SYNTH_REQ
// + SHUTDOWN, exits on EOF or SHUTDOWN.
//
// Returns the process exit code (0 on clean shutdown).
int run_worker_loop(int fd);

} // namespace qwen3_tts

#endif // QWEN3_TTS_WORKER_SESSION_H
