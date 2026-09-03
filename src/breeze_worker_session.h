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
// The frame protocol itself is shared with the higgs server (higgs_worker_ipc):
// it is a generic length-prefixed socketpair transport with no engine-specific
// content, so it is reused rather than duplicated. Only the role flag differs
// (`--breeze-worker`).
//
// The public synth surface mirrors BreezeTTS so the server's handlers can be
// written once and instantiated against either backend.

#include "breeze_tts.h"
#include "higgs_worker_ipc.h"

#include <atomic>
#include <mutex>
#include <string>
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
    const std::string & get_error() const { return last_error_; }
    const std::string & worker_gpu() const { return worker_gpu_; }

    void set_default_gpu(std::string gpu) { default_gpu_ = std::move(gpu); }
    void set_next_gpu(const std::string & gpu) {
        if (!gpu.empty()) { std::lock_guard<std::mutex> lk(io_mutex_); next_gpu_ = gpu; }
    }

    bool synthesize(const std::string & text, const gen_params & gp,
                    const ref_voice * ref, gen_result & out);
    // Rolling-context long-form. Runs inside the worker so the per-chunk codes
    // never have to cross the IPC.
    bool synthesize_long(const std::string & text, const gen_params & gp,
                         int chunk_words, int ref_max_frames,
                         int stream_chunk_frames,
                         const BreezeTTS::pcm_cb & on_chunk, gen_result & out);
    bool synthesize_stream(const std::string & text, const gen_params & gp,
                           const ref_voice * ref, int chunk_frames,
                           const BreezeTTS::pcm_cb & on_chunk, gen_result & out);
    bool encode_voice(const float * pcm, int n_samples,
                      std::vector<int32_t> & out_codes, int & out_T);

    void request_cancel();
    void clear_cancel() {}   // the child clears its own flag per SPEECH_REQ

private:
    bool send_load_req_locked(const WorkerConfig & cfg);
    void kill_worker_locked();
    bool send_speech_locked(const std::string & meta_json,
                            const std::vector<int32_t> & codes_blob,
                            bool streaming, const BreezeTTS::pcm_cb & on_chunk,
                            gen_result & out);

    std::string              argv0_;
    std::vector<std::string> extra_argv_;
    WorkerConfig             loaded_cfg_;
    bool                     loaded_ok_ = false;
    std::string              default_gpu_, next_gpu_, worker_gpu_;

    pid_t                    pid_ = -1;
    int                      fd_  = -1;
    int                      sample_rate_ = 24000;
    mutable std::mutex       io_mutex_;
    mutable std::mutex       send_mutex_;
    std::string              last_error_;
    std::atomic<uint32_t>    next_req_id_{1};
    std::atomic<uint32_t>    current_synth_req_id_{0};
};

// Child dispatch loop; entered from main() on "--breeze-worker <fd>".
int run_breeze_worker_loop(int fd);

} // namespace breeze
