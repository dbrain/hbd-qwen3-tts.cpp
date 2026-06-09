// fa_session.h — the single, engine-agnostic forced-alignment (read-along)
// sibling, shared by BOTH qwen3-tts-server and higgs-server.
//
// History: qwen3-tts grew its aligner methods INTO its engine-specific
// WorkerSession; higgs then copy-pasted them into a standalone
// HiggsAlignerSession. The aligner only needs (PCM + word list) → per-word
// timings, so it is engine-agnostic — that is why the `qwen3_fa` *library* was
// already shared. This module finishes the job: one parent-side session + one
// child loop, used by both binaries, so the IPC/backpressure logic lives in
// exactly one place.
//
// Self-contained transport: the aligner sibling talks to its parent over its
// OWN AF_UNIX SOCK_STREAM socketpair with its OWN frame enum. It deliberately
// does NOT reuse worker_ipc / higgs_worker_ipc (those define clashing global
// symbols and are per-engine); the framing here lives in `namespace fa`.
//
// Architecture (unchanged from before): a SEPARATE aligner-only subprocess with
// its OWN CUDA context, so its encode+align runs concurrently with the synth on
// the same GPU and idle-unloads independently (aligner VRAM → true-0 when not
// aligning). SIGKILL on shutdown reclaims VRAM.

#ifndef FA_SESSION_H
#define FA_SESSION_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <vector>

namespace fa {

// ── Wire protocol ───────────────────────────────────────────────────────────
enum class Frame : uint32_t {
    HELLO        = 0x01,  // W→P  {"pid":int,"role":"fa-aligner"}
    LOAD_REQ     = 0x10,  // P→W  {"aligner_model":str,"eager_load_aligner":bool}
    LOAD_RESP    = 0x11,  // W→P  {"ok":bool,"error":str,"t_load_ms":int}
    PING         = 0x50,  // P→W  {"t_send_ns":u64}
    PONG         = 0x51,  // W→P  echo
    // *_REQ payload  = pack_audio_payload(json{words,pcm_sample_rate,
    //                   audio_seen_ms|audio_total_ms,reset}, pcm_f32_delta)
    // *_RESP payload = pure JSON {ok,error,audio_seen_ms,words[],profile{}}
    PARTIAL_REQ  = 0x60,  // P→W
    PARTIAL_RESP = 0x61,  // W→P
    FINAL_REQ    = 0x62,  // P→W
    FINAL_RESP   = 0x63,  // W→P
    SHUTDOWN     = 0xFF,  // P→W  exit cleanly
};

enum class IpcError { OK, EofClean, EofMidFrame, SocketError, PayloadTooBig };
const char * ipc_error_str(IpcError e);

struct FrameHeader {
    uint32_t type;     // Frame
    uint32_t len;      // payload bytes that follow
    uint32_t req_id;   // 0 = unsolicited / no correlation
};

inline constexpr size_t MAX_FRAME_PAYLOAD = 256u * 1024u * 1024u; // 256 MiB (long-form PCM)

IpcError send_frame(int fd, Frame type, uint32_t req_id, const void * payload, size_t len);
IpcError send_frame(int fd, Frame type, uint32_t req_id, const std::string & json);
IpcError recv_frame(int fd, FrameHeader * out_hdr, std::vector<uint8_t> * out_payload);

// [u32 meta_len][meta_json bytes][raw f32 pcm] — header+payload in one buffer.
std::vector<uint8_t> pack_audio_payload(const std::string & meta, const float * samples, size_t n);
bool unpack_audio_payload(const std::vector<uint8_t> & payload, std::string * out_meta,
                          std::vector<float> * out_pcm);

// fork()+execv() argv0 with `--fa-aligner <child_fd>` appended to extra_argv.
// Returns child pid (>0) and sets *out_parent_fd; -1 on failure.
pid_t spawn_aligner(const char * self_argv0, const std::vector<std::string> & extra_argv,
                    int * out_parent_fd);

// ── Aligned result types (superset of both engines' fields) ─────────────────
struct AlignedWord {
    std::string text;
    int64_t     t0_ms = 0;
    int64_t     t1_ms = 0;
    float       confidence = -1.0f;
};

struct AlignProfile {
    int64_t t_load_ms     = 0;
    int64_t t_resample_ms = 0;
    int64_t t_aligner_ms  = 0;
    int64_t t_total_ms    = 0;
    int     n_words       = 0;
};

// ── Parent-side handle on the aligner-only sibling subprocess ────────────────
// One instance per server process. Lazy-spawned on first ensure_loaded();
// SIGKILL on shutdown(). Only the streaming-align surface is valid.
class AlignerSession {
public:
    explicit AlignerSession(const char * argv0, std::vector<std::string> extra_argv = {});
    ~AlignerSession();

    // Spawn + (lazy or eager) load the FA GGUF; respawn on path change.
    bool ensure_loaded(const std::string & aligner_model);
    void shutdown();  // SIGKILL + waitpid; idempotent

    bool is_alive() const { return pid_ > 0; }
    pid_t pid() const     { return pid_; }
    const std::string & last_error() const { return last_error_; }

    // 1. begin — reset accumulator, stash words.
    // 2. push_partial_pcm — never blocks the caller (bounded-in-flight=1 +
    //    coalescing); the aligner re-encodes cumulative audio and returns
    //    updated timings on the same fd. Service via drain_partial_alignments.
    // 3. finalize — last call; folds any un-sent coalesced audio into FINAL,
    //    blocks until FINAL_RESP. Resets session state on exit.
    bool begin_streaming_align(const std::vector<std::string> & words, int pcm_sample_rate);
    bool push_partial_pcm(const float * pcm, size_t n_samples, int64_t audio_seen_ms);

    using PartialAlignCallback =
        std::function<void(int64_t audio_seen_ms, const std::vector<AlignedWord> & words)>;
    bool drain_partial_alignments(const PartialAlignCallback & cb);

    bool finalize_streaming_align(const float * tail_pcm, size_t n_tail_samples,
                                  int64_t audio_total_ms, std::vector<AlignedWord> & out_words,
                                  AlignProfile & out_profile);

private:
    void kill_worker_locked();
    bool flush_pending_locked();  // send pending_pcm_ iff a slot is free; holds io_mutex_

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

    // Bounded-in-flight backpressure (the long-render deadlock fix). The synth
    // thread feeds PCM inline; the transport is a BLOCKING socketpair and the
    // send holds io_mutex_, so a slow aligner would fill the send buffer, block
    // the synth in writev WHILE holding io_mutex_, starve the reader thread, and
    // deadlock both processes. We keep at most kMaxInflight unacked partials in
    // the socket; surplus audio coalesces into pending_pcm_ and flushes when the
    // aligner frees a slot (also cuts redundant re-encodes — one larger delta).
    static constexpr int     kMaxInflight = 1;
    std::vector<float>       pending_pcm_;
    int64_t                  pending_seen_ms_ = 0;
    int                      inflight_ = 0;
};

// Aligner-only child dispatch loop. Called from main() when `--fa-aligner <fd>`
// is passed. Loads the FA GGUF (eager on LOAD_REQ or lazy on first REQ), keeps a
// growing PCM accumulator, re-runs the qwen3_fa encode + LLM body per pass.
// Returns the process exit code.
int run_aligner_loop(int fd);

} // namespace fa

#endif // FA_SESSION_H
