#pragma once

// Parent-side handle on the Breeze GPU subprocess, plus the child dispatch loop.
//
//   Parent = HTTP server + filesystem VoiceStore, NO CUDA context. Survives an
//            idle-unload, so list/delete/from-codes voice ops keep working with
//            the model gone.
//   Worker = owns BreezeTTS + ggml-cuda; SIGKILL on idle reclaims ALL VRAM
//            (true 0), and a wedged or OOM-ing worker is recoverable by killing
//            a pid instead of restarting the container.
//
// Concurrency: the worker runs up to BREEZE_MAX_STREAMS (default 2) requests
// at once on its BreezeScheduler, batching their decode steps. Frames for all
// of them share the one socketpair, tagged by req_id. On the parent side a
// reader thread demultiplexes them into a mailbox per call, and each call
// drains its own mailbox on its own thread -- so a client that stops reading
// (the read-along paces itself) stalls only its own response, never the other
// request's audio or the worker.
//
// The frame protocol itself is shared with the higgs server (higgs_worker_ipc):
// it is a generic length-prefixed socketpair transport with no engine-specific
// content, so it is reused rather than duplicated. Only the role flag differs
// (`--breeze-worker`).

#include "breeze_tts.h"
#include "higgs_worker_ipc.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace breeze {

struct WorkerConfig {
    std::string model;      // the one Breeze GGUF
    int         n_ctx = 2048;
};

class WorkerSession {
public:
    explicit WorkerSession(const char * argv0, std::vector<std::string> extra_argv = {});
    ~WorkerSession();

    bool ensure_loaded(const WorkerConfig & cfg);
    void shutdown();

    bool  is_alive() const { return pid_ > 0; }
    pid_t pid() const      { return pid_; }
    int   sample_rate() const { return sample_rate_; }
    // The error of the last failed call made ON THIS THREAD -- concurrent
    // calls each see their own.
    const std::string & get_error() const;
    const std::string & worker_gpu() const { return worker_gpu_; }

    void set_default_gpu(std::string gpu) { default_gpu_ = std::move(gpu); }
    void set_next_gpu(const std::string & gpu) {
        if (!gpu.empty()) { std::lock_guard<std::mutex> lk(io_mutex_); next_gpu_ = gpu; }
    }

    bool synthesize(const std::string & text, const gen_params & gp,
                    const ref_voice * ref, gen_result & out, CallHandle * h = nullptr);
    // Rolling-context long-form. Runs inside the worker so the per-chunk codes
    // never have to cross the IPC.
    bool synthesize_long(const std::string & text, const gen_params & gp,
                         const ref_voice * ref, int chunk_words, int ref_max_frames,
                         int stream_chunk_frames, int gap_ms,
                         const BreezeTTS::pcm_cb & on_chunk, gen_result & out,
                         const BreezeTTS::chunk_cb & on_chunk_start = nullptr,
                         int first_chunk_words = 0, CallHandle * h = nullptr);
    bool synthesize_stream(const std::string & text, const gen_params & gp,
                           const ref_voice * ref, int chunk_frames,
                           const BreezeTTS::pcm_cb & on_chunk, gen_result & out,
                           CallHandle * h = nullptr);
    bool encode_voice(const float * pcm, int n_samples,
                      std::vector<int32_t> & out_codes, int & out_T);

    // Cancels that one call (a cancelled long-form render returns SUCCESS with
    // partial audio, so the handle's flag is how a finished read is told from
    // an abandoned one).
    void request_cancel(CallHandle & h);
    static bool is_cancel_requested(const CallHandle & h) {
        return h.cancel.load(std::memory_order_relaxed);
    }

private:
    struct Mailbox {
        std::mutex m;
        std::condition_variable cv;
        std::deque<std::pair<higgs::FrameHeader, std::vector<uint8_t>>> q;
        bool dead = false;
    };

    bool send_load_req_locked(const WorkerConfig & cfg);
    void kill_worker_locked();
    void reader_loop(int fd);
    bool call(higgs::WFrame type, const std::vector<uint8_t> & payload,
              std::shared_ptr<Mailbox> & mb, uint32_t & req_id, CallHandle * h);
    void drop_mailbox(uint32_t req_id);
    bool send_speech(const std::string & meta_json, const std::vector<int32_t> & codes_blob,
                     const BreezeTTS::pcm_cb & on_chunk, gen_result & out,
                     const BreezeTTS::chunk_cb & on_chunk_start, CallHandle * h);
    void set_error(const std::string & e);

    std::string              argv0_;
    std::vector<std::string> extra_argv_;
    WorkerConfig             loaded_cfg_;
    bool                     loaded_ok_ = false;
    std::string              default_gpu_, next_gpu_, worker_gpu_;

    pid_t                    pid_ = -1;
    int                      fd_  = -1;
    int                      sample_rate_ = 24000;
    mutable std::mutex       io_mutex_;     // lifecycle: spawn / load / kill
    mutable std::mutex       send_mutex_;   // one frame on the wire at a time
    std::atomic<uint32_t>    next_req_id_{1};
    std::atomic<int>         active_calls_{0};
    std::atomic<bool>        reader_dead_{false};

    std::thread              reader_;
    std::mutex               mb_mutex_;
    std::map<uint32_t, std::shared_ptr<Mailbox>> mailboxes_;
};

// Child dispatch loop; entered from main() on "--breeze-worker <fd>".
int run_breeze_worker_loop(int fd);

} // namespace breeze
