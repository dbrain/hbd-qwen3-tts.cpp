// higgs_worker_ipc.h — length-prefixed frame protocol over a Unix-domain
// socketpair for the Higgs-Audio-v3 server's subprocess-worker model.
//
// Self-contained (namespace `higgs`) so the Higgs eval engine stays decoupled
// from the production qwen3-tts isolation stack. Same shape as qwen3-tts's
// worker_ipc, trimmed to what Higgs needs (no speaker-embedding frames — Higgs
// voice-clone is codes, no spk-embed) plus a trial-and-save frame.
//
// Parent role: HTTP server + filesystem VoiceStore, NO GPU. Owns the parent end
// of the socket. Spawns the child via fork()+execv("--higgs-worker <fd>").
//
// Worker role: owns the HiggsTTS engine + ggml-cuda context + model weights;
// stays warm. Started via fork()+execv so the address space is fresh (no
// copy-on-write of the parent's HTTP state into the CUDA process).
//
// Wire format: fixed 12-byte header followed by `len` payload bytes.
//   [u32 frame_type][u32 payload_len][u32 req_id][u8 payload[payload_len]]
//
// Structured frames carry nlohmann::json text. Frames that also carry a binary
// blob (PCM f32 / codes i32) use the pack_payload() layout below so the blob
// isn't base64-stuffed through JSON.

#ifndef HIGGS_WORKER_IPC_H
#define HIGGS_WORKER_IPC_H

#include <cstdint>
#include <string>
#include <vector>
#include <sys/types.h>

namespace higgs {

enum class WFrame : uint32_t {
    HELLO       = 0x01,  // W→P  {"pid":int,"role":str}
    LOAD_REQ    = 0x10,  // P→W  {"backbone":str,"aux":str,"n_ctx":int,"kv":str}
    LOAD_RESP   = 0x11,  // W→P  {"ok":bool,"error":str,"sample_rate":int}
    // SPEECH_REQ: pack_payload(json, codes_i32). json carries the resolved
    // request (mode + params + voice slice descriptors); the blob is the
    // concatenated int32 reference codes the parent resolved from its
    // VoiceStore. See higgs_worker_session for the exact json schema.
    SPEECH_REQ  = 0x20,  // P→W
    SPEECH_RESP = 0x21,  // W→P  pack_payload(json{meta}, pcm_f32) — whole audio
    AUDIO_FRAME = 0x22,  // W→P  pack_payload(json{}, pcm_f32) — streaming chunk
    SPEECH_DONE = 0x23,  // W→P  {"cancelled":bool,...meta} — end of stream
    SPEECH_ERR  = 0x2F,  // W→P  {"error":str}
    // CANCEL_REQ: parent asks the worker to abort the in-flight synth. req_id
    // is matched against the active SPEECH_REQ's req_id (mismatch = no-op so a
    // stale cancel can't kill the next request). Empty payload. The worker
    // flips HiggsTTS::request_cancel(); the AR loop bails within ~one step and
    // the worker still emits SPEECH_DONE/RESP so the parent drains cleanly.
    CANCEL_REQ  = 0x30,  // P→W   empty; hdr.req_id = target synth's req_id
    // TRIAL_REQ: render a zero-shot sample and return its OUTPUT codes so the
    // parent can persist a voice ("audition then keep"). No encoder needed.
    TRIAL_REQ   = 0x40,  // P→W  {"input":str, gp params}
    TRIAL_RESP  = 0x41,  // W→P  pack_payload(json{ok,error,T,N,seed}, codes_i32)
    // ENCODE_VOICE: encode a recorded/uploaded clip → codes (voice clone). The
    // child loads the F32 encoder on demand and frees it (idle VRAM flat).
    ENCODE_REQ  = 0x42,  // P→W  pack_payload(json{sample_rate:int}, wav_f32)
    ENCODE_RESP = 0x43,  // W→P  pack_payload(json{ok,error,T,N}, codes_i32)
    PING        = 0x50,  // P→W  {"t_send_ns":u64}
    PONG        = 0x51,  // W→P  echo
    // ── Forced-alignment sibling (word-highlight reader). A SEPARATE aligner
    // subprocess (spawned via "--higgs-aligner <fd>") owns the engine-agnostic
    // qwen3-forced-aligner model. The parent streams synthesised PCM deltas as
    // they're produced and gets back per-word t0/t1 timings. Same wire shape as
    // qwen3-tts's ALIGN_PARTIAL/FINAL frames so the SSE event JSON matches.
    //   *_REQ payload  = pack_audio_payload(json{words,pcm_sample_rate,
    //                     audio_seen_ms|audio_total_ms,reset}, f32 PCM delta)
    //   *_RESP payload = json{ok,error,audio_seen_ms,words:[{word_index,text,
    //                     t0_ms,t1_ms,confidence}],profile}
    ALIGN_PARTIAL_REQ  = 0x60,  // P→W
    ALIGN_PARTIAL_RESP = 0x61,  // W→P
    ALIGN_FINAL_REQ    = 0x62,  // P→W
    ALIGN_FINAL_RESP   = 0x63,  // W→P
    SHUTDOWN    = 0xFF,  // P→W  exit cleanly
};

struct FrameHeader {
    uint32_t type;     // WFrame
    uint32_t len;      // payload bytes that follow
    uint32_t req_id;   // 0 = unsolicited / no correlation
};
static_assert(sizeof(FrameHeader) == 12, "FrameHeader must stay 12 bytes");

inline constexpr size_t HEADER_BYTES      = sizeof(FrameHeader);
inline constexpr size_t MAX_FRAME_PAYLOAD = 256u * 1024u * 1024u; // 256 MiB cap (long-form PCM)

enum class IpcError {
    OK = 0,
    EofClean,        // peer closed cleanly before any frame bytes
    EofMidFrame,     // peer closed mid-frame (crash)
    SocketError,
    ProtocolError,
    PayloadTooBig,
};

const char * ipc_error_str(IpcError e);

// Blocking. OK iff `len` bytes fully received/sent.
IpcError read_exact(int fd, void * buf, size_t len);
IpcError write_exact(int fd, const void * buf, size_t len);

// Send a full frame (header+payload coalesced via writev).
IpcError send_frame(int fd, WFrame type, uint32_t req_id,
                    const void * payload, size_t payload_len);
IpcError send_frame(int fd, WFrame type, uint32_t req_id, const std::string & json);
IpcError send_frame(int fd, WFrame type, uint32_t req_id, const std::vector<uint8_t> & payload);

// Receive a full frame; on OK out_payload holds exactly hdr.len bytes.
IpcError recv_frame(int fd, FrameHeader * out_hdr, std::vector<uint8_t> * out_payload);

// JSON-meta + binary-blob payload: [u32 json_len][json bytes][raw blob bytes].
// pack_payload copies `blob_bytes` raw bytes (caller picks f32/i32). The
// typed overloads are conveniences. unpack_payload returns a zero-copy view
// of the blob region (valid as long as `payload` lives).
std::vector<uint8_t> pack_payload(const std::string & meta, const void * blob, size_t blob_bytes);
std::vector<uint8_t> pack_audio_payload(const std::string & meta, const float * samples, size_t n_samples);
std::vector<uint8_t> pack_codes_payload(const std::string & meta, const int32_t * codes, size_t n_codes);

bool unpack_payload(const std::vector<uint8_t> & payload,
                    std::string * out_meta,
                    const uint8_t ** out_blob, size_t * out_blob_bytes);

// Spawn helper: socketpair()+fork()+execv(argv0, "--higgs-worker <fd>" extra...).
// Returns child pid and writes the parent-side fd to *out_parent_fd; -1 on fail.
pid_t spawn_worker(const char * self_argv0,
                   const std::vector<std::string> & extra_argv,
                   int * out_parent_fd,
                   const char * role_flag = "--higgs-worker");

} // namespace higgs

#endif // HIGGS_WORKER_IPC_H
