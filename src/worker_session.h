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
    std::string model;          // talker GGUF
    std::string vocoder;        // vocoder GGUF
    std::string speaker_encoder;// optional speaker-encoder GGUF
    bool        lazy_load = false; // worker defers load_model_files() until first synth
};

// Parent-side handle. One instance per server process. Thread-safe:
// synthesize() takes io_mutex_ for the full request/response round-trip.
class WorkerSession {
public:
    // Configure but don't spawn yet. argv0 must be argv[0] of the
    // running process — used for execv() so the child runs the same
    // binary in --worker mode.
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

    pid_t                      pid_ = -1;
    int                        fd_  = -1;
    int                        sample_rate_ = 0;
    mutable std::mutex         io_mutex_;
    std::string                last_error_;
    std::atomic<uint32_t>      next_req_id_{1};
};

// Worker-side dispatch loop. Called from main() when --worker <fd> is
// passed. Owns a Qwen3TTS instance, services LOAD_REQ + SYNTH_REQ
// + SHUTDOWN, exits on EOF or SHUTDOWN.
//
// Returns the process exit code (0 on clean shutdown).
int run_worker_loop(int fd);

} // namespace qwen3_tts

#endif // QWEN3_TTS_WORKER_SESSION_H
