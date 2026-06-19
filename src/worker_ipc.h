// worker_ipc.h — length-prefixed frame protocol over Unix-domain
// socketpair, used by the qwen3-tts-server subprocess-worker model.
//
// Parent role: HTTP server, voice archive, no GPU. Owns the parent end
// of the socket. Spawns child via fork()+execv("--worker <fd>").
//
// Worker role: owns Qwen3TTS, ggml-cuda context, model weights. Started
// via fork()+execv from parent so the address space is fresh (no
// copy-on-write inheritance of parent state into the CUDA process).
// Reads frames from the inherited socket fd, dispatches.
//
// Protocol: fixed 12-byte header followed by `payload_len` bytes.
//
//   [u32 frame_type][u32 payload_len][u32 req_id][u8 payload[payload_len]]
//
// Strings + structured payloads use nlohmann::json. AUDIO_FRAME bypasses
// JSON for the float32 sample bytes — JSON header in payload[0..json_end],
// raw bytes in payload[json_end+1..len]. `req_id` lets the parent match
// SYNTH_DONE / AUDIO_FRAME / SYNTH_ERR back to the original SYNTH_REQ
// across the pipeline (one in-flight synth at a time today, but the
// header stays so we don't have to break the wire later).

#ifndef QWEN3_TTS_WORKER_IPC_H
#define QWEN3_TTS_WORKER_IPC_H

#include <cstdint>
#include <string>
#include <vector>

namespace qwen3_tts {

enum class WorkerFrame : uint32_t {
    HELLO        = 0x01,  // W→P  {"pid": int, "build": str}
    LOAD_REQ     = 0x10,  // P→W  {"model": str, "vocoder": str, "spk_enc": str}
    LOAD_RESP    = 0x11,  // W→P  {"ok": bool, "error": str}
    SYNTH_REQ    = 0x20,  // P→W  full synth args (json)
    SYNTH_RESP   = 0x21,  // W→P  whole-audio response (P1: not streamed)
                          //      json header + raw float32 samples
    AUDIO_FRAME  = 0x22,  // W→P  streaming chunk (P2)
    SYNTH_DONE   = 0x23,  // W→P  end-of-stream + usage stats (P2)
    SYNTH_ERR    = 0x2F,  // W→P  {"error": str}
    // CANCEL_REQ: parent asks worker to abort the currently-running synth
    // (req_id is matched against the in-flight SYNTH_REQ's req_id; a
    // mismatch is treated as a no-op so stale cancels can't accidentally
    // kill the next request). Empty payload; req_id field carries the
    // target. Worker still emits SYNTH_DONE (with success=false) so the
    // parent drain loop terminates cleanly. The worker is multiplexed
    // via a reader thread so CANCEL_REQ is observed while synth is
    // actively running on the main thread.
    CANCEL_REQ   = 0x30,  // P→W   empty payload; hdr.req_id = target synth's req_id
    EXTRACT_EMBED_REQ  = 0x50, // P→W  {"filepath": str}  (voice rego)
    EXTRACT_EMBED_RESP = 0x51, // W→P  {"ok": bool, "error": str, "n_floats": int} + raw f32
    ENCODE_CODES_REQ   = 0x52, // P→W  {"n_samples": int} + raw f32 samples
    ENCODE_CODES_RESP  = 0x53, // W→P  {"ok": bool, "error": str, "n_frames": int} + raw i32 codes
    SAVE_WARMUP_REQ    = 0x54, // P→W  {"voice_id", "prefill_key", "ref_hash", "path", "model_id"}
    SAVE_WARMUP_RESP   = 0x55, // W→P  {"ok": bool, "error": str}
    ALIGN_REQ          = 0x60, // P→W  {"words": [str,...]} — aligns against last synth's PCM
    ALIGN_RESP         = 0x61, // W→P  {"ok": bool, "error": str, "words":[{"text","t0_ms","t1_ms"}], "duration_ms", "profile"{...}}
    // Streaming-alignment frames (P2). Parent sends PCM deltas as audio is
    // generated; aligner worker re-runs the full FA encode + aligner LLM
    // pass on accumulated audio and emits updated word timings. PARTIAL is
    // an interim result emitted while audio is still arriving; FINAL is the
    // last call after TTS completes — locks in timings.
    //
    // ALIGN_PARTIAL_REQ payload: pack_audio_payload() with json:
    //   {"words":[str,...], "pcm_sample_rate":int, "audio_seen_ms":int,
    //    "reset":bool}   followed by raw f32 PCM samples (the *delta*; the
    //   aligner appends to its internal accumulator).
    // ALIGN_PARTIAL_RESP / ALIGN_FINAL_RESP payload (pure JSON):
    //   {"ok":bool, "error":str, "audio_seen_ms":int,
    //    "words":[{"word_index","char_offset","text","t0_ms","t1_ms"}],
    //    "profile":{...}}
    ALIGN_PARTIAL_REQ  = 0x62,
    ALIGN_PARTIAL_RESP = 0x63,
    ALIGN_FINAL_REQ    = 0x64,
    ALIGN_FINAL_RESP   = 0x65,
    PING         = 0x40,  // either {"t_send_ns": u64}
    PONG         = 0x41,  // either {"t_send_ns": u64, "t_recv_ns": u64}
    SHUTDOWN     = 0xFF,  // P→W   ask worker to exit cleanly
};

struct FrameHeader {
    uint32_t type;     // WorkerFrame
    uint32_t len;      // payload bytes that follow
    uint32_t req_id;   // 0 = unsolicited / no correlation
};
static_assert(sizeof(FrameHeader) == 12, "FrameHeader must stay 12 bytes");

// Wire format constants
inline constexpr size_t HEADER_BYTES = sizeof(FrameHeader);
inline constexpr size_t MAX_FRAME_PAYLOAD = 64u * 1024u * 1024u; // 64 MiB safety cap

// Errors. Return-coded rather than thrown so worker dispatch loop can
// log + continue / shutdown deterministically.
enum class IpcError {
    OK = 0,
    EofClean,        // peer closed cleanly mid-read
    EofMidFrame,     // peer closed in the middle of a frame
    SocketError,     // read/write returned -1 with non-recoverable errno
    ProtocolError,   // bad header (len > MAX, etc.)
    PayloadTooBig,
};

const char * ipc_error_str(IpcError e);

// Blocking. Returns OK iff `len` bytes were fully received. EofClean
// means peer closed before any bytes of the frame arrived (orderly
// shutdown); EofMidFrame means partial read (peer crashed).
IpcError read_exact(int fd, void * buf, size_t len);
IpcError write_exact(int fd, const void * buf, size_t len);

// Send / receive a full frame. `payload` is appended after the header
// in a single writev() to keep latency tight on small frames.
IpcError send_frame(int fd,
                    WorkerFrame type,
                    uint32_t req_id,
                    const void * payload,
                    size_t payload_len);

// Convenience overloads
IpcError send_frame(int fd,
                    WorkerFrame type,
                    uint32_t req_id,
                    const std::string & json);

IpcError send_frame(int fd,
                    WorkerFrame type,
                    uint32_t req_id,
                    const std::vector<uint8_t> & payload);

// Receive a full frame. On OK, `out_payload` contains exactly hdr.len
// bytes. The header is also written to `*out_hdr`.
IpcError recv_frame(int fd,
                    FrameHeader * out_hdr,
                    std::vector<uint8_t> * out_payload);

// AUDIO_FRAME / SYNTH_RESP carry a small JSON header followed by raw
// f32 samples. Encode/decode helpers so the binary blob doesn't get
// base64-stuffed through JSON (multi-MiB per synth).
//
// On the wire: [u32 json_len][json bytes][raw f32 bytes].
// json_len + json_bytes give us metadata (sample_rate, n_samples,
// req_id-mirror, error fields). Raw bytes follow.
std::vector<uint8_t> pack_audio_payload(const std::string & json_meta,
                                        const float * samples,
                                        size_t n_samples);

// Returns OK and fills out_meta/out_samples on success.
bool unpack_audio_payload(const std::vector<uint8_t> & payload,
                          std::string * out_meta,
                          std::vector<float> * out_samples);

// Spawn helper: socketpair() + fork() + execv(argv[0], "--worker <fd>").
// The child's inherited fd is passed as `--worker N` where N is the
// numeric fd (the close-on-exec flag is cleared so it survives execv).
// Returns the child pid on success and writes the parent-side fd to
// `*out_parent_fd`. Returns -1 on failure.
//
// `extra_argv` is appended after "--worker N" so the child can be
// invoked with the same load-time arguments (--hf-repo, etc.) it would
// have seen if started normally.
pid_t spawn_worker(const char * self_argv0,
                   const std::vector<std::string> & extra_argv,
                   int * out_parent_fd,
                   const char * role_flag = "--worker",
                   const std::string & cuda_visible_devices = "");

} // namespace qwen3_tts

#endif // QWEN3_TTS_WORKER_IPC_H
