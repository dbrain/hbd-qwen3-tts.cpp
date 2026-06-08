// higgs_worker_session.h — parent-side handle on a Higgs subprocess worker,
// plus the child-side dispatch loop. Mirrors the qwen3-tts WorkerSession but
// trimmed for Higgs (no speaker-embedding; voice-clone = codes the parent
// resolves from its filesystem VoiceStore and ships in the SPEECH_REQ blob).
//
//   Parent  = HTTP server + VoiceStore (filesystem, NO GPU). Survives an
//             idle-unload (worker SIGKILLed) so list/delete/from-codes voice
//             ops keep working with the model unloaded.
//   Worker  = owns HiggsTTS + ggml-cuda; stays warm; SIGKILL on idle-unload
//             reclaims ALL VRAM (true-0). Respawns + reloads on next request.
//
// The public synth methods MIRROR HiggsTTS's signatures so the server's speech
// handler can be written once against either the in-process engine or this
// session (templated). request_cancel() is named to match HiggsTTS so the
// streaming disconnect-watchdog calls it uniformly.

#ifndef HIGGS_WORKER_SESSION_H
#define HIGGS_WORKER_SESSION_H

#include "higgs_tts.h"
#include "higgs_worker_ipc.h"

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace higgs {

struct HiggsWorkerConfig {
    std::string backbone;     // backbone GGUF
    std::string aux;          // codec-aux GGUF
    int         n_ctx = 8192;
    std::string kv    = "q8"; // "f16" | "q8" → HIGGS_LM_KV in the child
    std::string aux_enc;      // optional F32 voice-clone encoder GGUF ("" = wav-upload disabled)
};

class HiggsWorkerSession {
public:
    explicit HiggsWorkerSession(const char * argv0, std::vector<std::string> extra_argv = {});
    ~HiggsWorkerSession();

    // Spawn + load if not already loaded with this cfg; respawn on cfg change.
    bool ensure_loaded(const HiggsWorkerConfig & cfg);
    // SIGKILL + waitpid. Idempotent. Next ensure_loaded() respawns. (idle-unload)
    void shutdown();

    bool  is_alive() const { return pid_ > 0; }
    pid_t pid() const      { return pid_; }
    int   sample_rate() const { return sample_rate_; }
    const std::string & get_error() const { return last_error_; }

    // ── synth surface mirroring HiggsTTS (engine-agnostic server handler) ──
    bool synthesize(const std::string & text, const gen_params & gp, gen_result & out);

    bool synthesize_with_ref(const std::string & text,
                             const int32_t * ref_codes_TN, int ref_T,
                             const std::string & ref_text,
                             const gen_params & gp, gen_result & out, bool decode_audio = true);

    bool synthesize_long(const std::string & text, const gen_params & gp,
                         int buffer, int chunk_words,
                         const HiggsTTS::pcm_cb & on_chunk, gen_result & out);

    bool synthesize_multispeaker(const std::string & text,
                                 const std::map<std::string, HiggsTTS::named_voice> & voices,
                                 const gen_params & gp, int gap_ms, bool rolling,
                                 const HiggsTTS::pcm_cb & on_chunk, gen_result & out);

    bool synthesize_stream(const std::string & text, const gen_params & gp,
                           int chunk_frames, const HiggsTTS::pcm_cb & on_chunk, gen_result & out);

    // Trial-and-save: render zero-shot, return OUTPUT codes (parent persists).
    bool trial_codes(const std::string & text, const gen_params & gp,
                     std::vector<int32_t> & out_codes, int & out_T, int & out_N);

    // Voice-clone encode: arbitrary-rate mono PCM → codes (the child loads +
    // frees the F32 encoder). Mirrors HiggsTTS::encode_voice for the templated
    // server handler.
    bool encode_voice(const float * wav, int n_samples, int sr,
                      std::vector<int32_t> & out_codes, int & out_T, int & out_N);

    // Cooperative cancel for the in-flight synth (no-op if none). Thread-safe;
    // uses send_mutex_ so it can fire while the synth call holds io_mutex_.
    void request_cancel();
    // No-op: the child clears its own cancel flag on each SPEECH_REQ. Present
    // so the server's speech handler is uniform across HiggsTTS / session.
    void clear_cancel() {}

private:
    bool send_load_req_locked(const HiggsWorkerConfig & cfg);
    void kill_worker_locked();
    // Build SPEECH_REQ (json meta + int32 codes blob), send, drain RESP/FRAMES.
    // Caller holds io_mutex_. on_chunk!=null & streaming → relays AUDIO_FRAMEs.
    bool send_speech_locked(const std::string & meta_json,
                            const std::vector<int32_t> & codes_blob,
                            bool streaming, const HiggsTTS::pcm_cb & on_chunk,
                            gen_result & out);

    std::string              argv0_;
    std::vector<std::string> extra_argv_;
    HiggsWorkerConfig        loaded_cfg_;
    bool                     loaded_ok_ = false;

    pid_t                    pid_ = -1;
    int                      fd_  = -1;
    int                      sample_rate_ = 0;
    mutable std::mutex       io_mutex_;
    mutable std::mutex       send_mutex_;   // guards cancel's wire send
    std::string              last_error_;
    std::atomic<uint32_t>    next_req_id_{1};
    std::atomic<uint32_t>    current_synth_req_id_{0};
};

// Child dispatch loop: called from main() when "--higgs-worker <fd>" is passed.
// Owns a HiggsTTS, services LOAD_REQ/SPEECH_REQ/TRIAL_REQ/PING + CANCEL_REQ
// (reader thread) + SHUTDOWN. Returns the process exit code.
int run_higgs_worker_loop(int fd);

} // namespace higgs

#endif // HIGGS_WORKER_SESSION_H
