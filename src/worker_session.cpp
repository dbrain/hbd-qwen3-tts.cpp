// worker_session.cpp — see worker_session.h for design.

#include "worker_session.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__linux__)
#  include <sys/prctl.h>
#endif

#if QWEN3_TTS_HAS_FORCED_ALIGNMENT
#  include "qwen3_asr.h" // vendored from parakeet.cpp; in src/qwen3_fa/
#  include "ggml-cuda.h" // ggml_backend_cuda_get_graph_cache_stats — probe-only
#endif

using nlohmann::json;

namespace qwen3_tts {

// ────────────────────────────── helpers ──────────────────────────────

static json params_to_json(const tts_params & p) {
    return {
        {"max_audio_tokens",   p.max_audio_tokens},
        {"temperature",        p.temperature},
        {"top_p",              p.top_p},
        {"top_k",              p.top_k},
        {"n_threads",          p.n_threads},
        {"print_progress",     p.print_progress},
        {"print_timing",       p.print_timing},
        {"repetition_penalty", p.repetition_penalty},
        {"language_id",        p.language_id},
        {"instructions",       p.instructions},
        {"ref_text",           p.ref_text},
        {"seed",               p.seed},
    };
}

static tts_params params_from_json(const json & j) {
    tts_params p;
    p.max_audio_tokens   = j.value("max_audio_tokens",   p.max_audio_tokens);
    p.temperature        = j.value("temperature",        p.temperature);
    p.top_p              = j.value("top_p",              p.top_p);
    p.top_k              = j.value("top_k",              p.top_k);
    p.n_threads          = j.value("n_threads",          p.n_threads);
    p.print_progress     = j.value("print_progress",     p.print_progress);
    p.print_timing       = j.value("print_timing",       p.print_timing);
    p.repetition_penalty = j.value("repetition_penalty", p.repetition_penalty);
    p.language_id        = j.value("language_id",        p.language_id);
    p.instructions       = j.value("instructions",       std::string{});
    p.ref_text           = j.value("ref_text",           std::string{});
    p.seed               = j.value("seed",               p.seed);
    return p;
}

static json result_metadata_to_json(const tts_result & r) {
    return {
        {"success",          r.success},
        {"error_msg",        r.error_msg},
        {"sample_rate",      r.sample_rate},
        {"n_text_tokens",    r.n_text_tokens},
        {"n_prefill_tokens", r.n_prefill_tokens},
        {"n_audio_tokens",   r.n_audio_tokens},
        {"t_total_ms",       r.t_total_ms},
        {"t_load_ms",        r.t_load_ms},
        {"t_tokenize_ms",    r.t_tokenize_ms},
        {"t_encode_ms",      r.t_encode_ms},
        {"t_generate_ms",    r.t_generate_ms},
        {"t_prefill_ms",     r.t_prefill_ms},
        {"t_decode_ms",      r.t_decode_ms},
        {"prefill_cache_key",r.prefill_cache_key},
        {"ref_codes_hash",   r.ref_codes_hash},
    };
}

static void result_metadata_from_json(const json & j, tts_result & r) {
    r.success          = j.value("success",          false);
    r.error_msg        = j.value("error_msg",        std::string{});
    r.sample_rate      = j.value("sample_rate",      r.sample_rate);
    r.n_text_tokens    = j.value("n_text_tokens",    0);
    r.n_prefill_tokens = j.value("n_prefill_tokens", 0);
    r.n_audio_tokens   = j.value("n_audio_tokens",   0);
    r.t_total_ms       = j.value("t_total_ms",       (int64_t) 0);
    r.t_load_ms        = j.value("t_load_ms",        (int64_t) 0);
    r.t_tokenize_ms    = j.value("t_tokenize_ms",    (int64_t) 0);
    r.t_encode_ms      = j.value("t_encode_ms",      (int64_t) 0);
    r.t_generate_ms    = j.value("t_generate_ms",    (int64_t) 0);
    r.t_prefill_ms     = j.value("t_prefill_ms",     (int64_t) 0);
    r.t_decode_ms      = j.value("t_decode_ms",      (int64_t) 0);
    r.prefill_cache_key = j.value("prefill_cache_key", (uint64_t) 0);
    r.ref_codes_hash   = j.value("ref_codes_hash",   (uint64_t) 0);
}

// SYNTH_REQ payload layout: JSON header followed by raw blobs.
//   [u32 json_len][json bytes][float32 embedding[emb_size]][int32 ref_codes[n_ref_codes]]
// JSON carries embedding_size, n_ref_codes (= n_frames * n_codebooks total int32s),
// and n_ref_frames (timestep count used by the synth API).
static std::vector<uint8_t> pack_synth_payload(
        const std::string & text,
        const float * embedding, int32_t embedding_size,
        const tts_params & params,
        const int32_t * ref_codes, int32_t n_ref_codes,
        int32_t n_ref_frames,
        int32_t stream_batch_size,
        int32_t stream_first_batch_size) {
    json meta = {
        {"text",                     text},
        {"embedding_size",           embedding_size},
        {"n_ref_codes",              n_ref_codes},
        {"n_ref_frames",             n_ref_frames},
        {"stream_batch_size",        stream_batch_size},
        {"stream_first_batch_size",  stream_first_batch_size},
        {"params",                   params_to_json(params)},
    };
    std::string meta_str = meta.dump();
    uint32_t mlen = static_cast<uint32_t>(meta_str.size());
    size_t emb_bytes  = embedding_size > 0 ? (size_t) embedding_size * sizeof(float)   : 0;
    size_t code_bytes = n_ref_codes    > 0 ? (size_t) n_ref_codes    * sizeof(int32_t) : 0;
    std::vector<uint8_t> out;
    out.resize(sizeof(mlen) + meta_str.size() + emb_bytes + code_bytes);
    std::memcpy(out.data(),                         &mlen,           sizeof(mlen));
    std::memcpy(out.data() + sizeof(mlen),          meta_str.data(), meta_str.size());
    if (emb_bytes) {
        std::memcpy(out.data() + sizeof(mlen) + meta_str.size(),
                    embedding, emb_bytes);
    }
    if (code_bytes) {
        std::memcpy(out.data() + sizeof(mlen) + meta_str.size() + emb_bytes,
                    ref_codes, code_bytes);
    }
    return out;
}

static bool unpack_synth_payload(
        const std::vector<uint8_t> & payload,
        std::string * out_text,
        std::vector<float> * out_embedding,
        std::vector<int32_t> * out_ref_codes,
        int32_t * out_n_ref_frames,
        int32_t * out_stream_batch_size,
        int32_t * out_stream_first_batch_size,
        tts_params * out_params,
        std::string * err) {
    if (payload.size() < sizeof(uint32_t)) {
        if (err) *err = "payload too small for json_len header";
        return false;
    }
    uint32_t mlen = 0;
    std::memcpy(&mlen, payload.data(), sizeof(mlen));
    if (sizeof(mlen) + mlen > payload.size()) {
        if (err) *err = "json_len exceeds payload";
        return false;
    }
    json meta;
    try {
        meta = json::parse(payload.data() + sizeof(mlen),
                           payload.data() + sizeof(mlen) + mlen);
    } catch (const std::exception & e) {
        if (err) *err = std::string("synth_req json parse failed: ") + e.what();
        return false;
    }
    *out_text   = meta.value("text", std::string{});
    *out_params = params_from_json(meta.value("params", json::object()));
    int32_t emb_size              = meta.value("embedding_size",          0);
    int32_t n_ref_codes           = meta.value("n_ref_codes",             0);
    *out_n_ref_frames             = meta.value("n_ref_frames",            0);
    *out_stream_batch_size        = meta.value("stream_batch_size",       0);
    *out_stream_first_batch_size  = meta.value("stream_first_batch_size", 0);
    size_t expected = sizeof(mlen) + mlen
                    + (size_t) emb_size    * sizeof(float)
                    + (size_t) n_ref_codes * sizeof(int32_t);
    if (expected != payload.size()) {
        if (err) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "synth_req payload size mismatch (expected %zu = hdr+json+%dx4f+%dx4i, got %zu)",
                          expected, emb_size, n_ref_codes, payload.size());
            *err = buf;
        }
        return false;
    }
    out_embedding->resize(emb_size);
    out_ref_codes->resize(n_ref_codes);
    size_t off = sizeof(mlen) + mlen;
    if (emb_size) {
        std::memcpy(out_embedding->data(), payload.data() + off,
                    (size_t) emb_size * sizeof(float));
        off += (size_t) emb_size * sizeof(float);
    }
    if (n_ref_codes) {
        std::memcpy(out_ref_codes->data(), payload.data() + off,
                    (size_t) n_ref_codes * sizeof(int32_t));
    }
    return true;
}

// SYNTH_RESP payload layout: JSON metadata header, then raw float32 audio.
//   [u32 json_len][json bytes][float32 audio[n_samples]]
// n_samples derives from the trailing byte count, but n_samples in JSON
// is sent as a sanity check.
static std::vector<uint8_t> pack_synth_resp(const tts_result & r) {
    json meta = result_metadata_to_json(r);
    meta["n_samples"] = (uint64_t) r.audio.size();
    std::string meta_str = meta.dump();
    return pack_audio_payload(meta_str, r.audio.data(), r.audio.size());
}

static bool unpack_synth_resp(const std::vector<uint8_t> & payload,
                              tts_result * out, std::string * err) {
    std::string meta_str;
    std::vector<float> samples;
    if (!unpack_audio_payload(payload, &meta_str, &samples)) {
        if (err) *err = "unpack_audio_payload failed";
        return false;
    }
    json meta;
    try {
        meta = json::parse(meta_str);
    } catch (const std::exception & e) {
        if (err) *err = std::string("synth_resp meta parse: ") + e.what();
        return false;
    }
    result_metadata_from_json(meta, *out);
    out->audio = std::move(samples);
    return true;
}

// ──────────────────────── WorkerSession (parent) ──────────────────────

WorkerSession::WorkerSession(const char * argv0,
                             std::vector<std::string> extra_argv,
                             bool aligner_only)
    : argv0_(argv0 ? argv0 : ""), extra_argv_(std::move(extra_argv)),
      aligner_only_(aligner_only) {}

WorkerSession::~WorkerSession() {
    shutdown();
}

void WorkerSession::kill_worker_locked() {
    if (pid_ > 0) {
        // SIGKILL is the headline feature: it tears down the CUDA primary
        // context and reclaims ALL VRAM, no graceful-shutdown handshake
        // needed.
        ::kill(pid_, SIGKILL);
        int wstat = 0;
        ::waitpid(pid_, &wstat, 0);
        fprintf(stderr, "worker-session: killed worker pid=%d (wstat=0x%x)\n",
                (int) pid_, wstat);
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    pid_ = -1;
    loaded_ok_ = false;
    loaded_cfg_ = {};
}

void WorkerSession::shutdown() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    kill_worker_locked();
}

bool WorkerSession::send_load_req_locked(const WorkerLoadConfig & cfg) {
    json req = {
        {"model",              cfg.model},
        {"vocoder",            cfg.vocoder},
        {"speaker_encoder",    cfg.speaker_encoder},
        {"voice_archive_dir",  cfg.voice_archive_dir},
        {"model_id",           cfg.model_id},
        {"aligner_model",      cfg.aligner_model},
        {"lazy_load",          cfg.lazy_load},
        // When the parent spawns an aligner-only sibling (eager-spawn
        // path), it expects the subprocess to load the FA GGUF
        // synchronously in LOAD_REQ rather than lazily on the first
        // ALIGN_PARTIAL_REQ. That way the GGUF mmap + buffer alloc
        // overlaps with the parent's own synth cold load.
        {"eager_load_aligner", cfg.aligner_only},
    };
    IpcError e = send_frame(fd_, WorkerFrame::LOAD_REQ, 0, req.dump());
    if (e != IpcError::OK) {
        last_error_ = std::string("LOAD_REQ send failed: ") + ipc_error_str(e);
        return false;
    }
    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK) {
        last_error_ = std::string("LOAD_RESP recv failed: ") + ipc_error_str(e);
        return false;
    }
    if (hdr.type != static_cast<uint32_t>(WorkerFrame::LOAD_RESP)) {
        last_error_ = std::string("expected LOAD_RESP, got type=0x")
                    + std::to_string(hdr.type);
        return false;
    }
    json resp;
    try {
        resp = json::parse(std::string(payload.begin(), payload.end()));
    } catch (const std::exception & ex) {
        last_error_ = std::string("LOAD_RESP json parse: ") + ex.what();
        return false;
    }
    if (!resp.value("ok", false)) {
        last_error_ = std::string("worker load failed: ")
                    + resp.value("error", std::string{"(no msg)"});
        return false;
    }
    sample_rate_ = resp.value("sample_rate", 0);
    return true;
}

bool WorkerSession::ensure_loaded(const WorkerLoadConfig & cfg) {
    std::lock_guard<std::mutex> lock(io_mutex_);

    // If config matches and worker is alive, we're good. voice_archive_dir
    // and model_id aren't part of the cache key — they only affect the
    // worker's startup warmup scan, which is idempotent on respawn.
    if (pid_ > 0 && loaded_ok_
        && loaded_cfg_.model == cfg.model
        && loaded_cfg_.vocoder == cfg.vocoder
        && loaded_cfg_.speaker_encoder == cfg.speaker_encoder
        && loaded_cfg_.aligner_model == cfg.aligner_model) {
        return true;
    }

    // Different config (or first load) → kill any existing worker, respawn.
    if (pid_ > 0) kill_worker_locked();

    pid_t child = spawn_worker(argv0_.c_str(), extra_argv_, &fd_,
                                aligner_only_ ? "--worker-aligner" : "--worker");
    if (child < 0) {
        last_error_ = "spawn_worker failed";
        return false;
    }
    pid_ = child;

    // Expect HELLO before LOAD_REQ.
    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    IpcError e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK || hdr.type != static_cast<uint32_t>(WorkerFrame::HELLO)) {
        last_error_ = std::string("worker HELLO failed: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    fprintf(stderr, "worker-session: child pid=%d HELLO: %.*s\n",
            (int) pid_, (int) payload.size(), (const char *) payload.data());

    if (!send_load_req_locked(cfg)) {
        kill_worker_locked();
        return false;
    }
    loaded_cfg_ = cfg;
    loaded_ok_  = true;
    return true;
}

tts_result WorkerSession::do_synth_locked(
        const std::string & text,
        const float * embedding, int32_t embedding_size,
        const tts_params & params,
        const int32_t * ref_codes, int32_t n_ref_codes,
        int32_t n_ref_frames,
        int32_t stream_batch_size,
        int32_t stream_first_batch_size,
        StreamCallback on_pcm) {
    tts_result fail;
    fail.success = false;

    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) {
        fail.error_msg = "worker not ready (call ensure_loaded first)";
        return fail;
    }

    const bool streaming = (stream_batch_size > 0);

    auto payload = pack_synth_payload(text, embedding, embedding_size,
                                      params, ref_codes, n_ref_codes,
                                      n_ref_frames,
                                      streaming ? stream_batch_size : 0,
                                      streaming ? stream_first_batch_size : 0);
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame(fd_, WorkerFrame::SYNTH_REQ, req_id, payload);
    if (e != IpcError::OK) {
        fail.error_msg = std::string("SYNTH_REQ send failed: ") + ipc_error_str(e);
        kill_worker_locked();
        return fail;
    }

    if (!streaming) {
        // Non-streaming: one SYNTH_RESP frame contains the whole audio.
        FrameHeader hdr{};
        std::vector<uint8_t> resp_payload;
        e = recv_frame(fd_, &hdr, &resp_payload);
        if (e != IpcError::OK) {
            fail.error_msg = std::string("SYNTH_RESP recv failed: ") + ipc_error_str(e);
            kill_worker_locked();
            return fail;
        }
        if (hdr.type == static_cast<uint32_t>(WorkerFrame::SYNTH_ERR)) {
            try {
                json j = json::parse(std::string(resp_payload.begin(), resp_payload.end()));
                fail.error_msg = j.value("error", std::string{"unknown worker error"});
            } catch (...) {
                fail.error_msg = "worker reported error (unparseable)";
            }
            return fail;
        }
        if (hdr.type != static_cast<uint32_t>(WorkerFrame::SYNTH_RESP)) {
            fail.error_msg = std::string("expected SYNTH_RESP, got type=0x")
                           + std::to_string(hdr.type);
            kill_worker_locked();
            return fail;
        }
        tts_result out;
        std::string err;
        if (!unpack_synth_resp(resp_payload, &out, &err)) {
            fail.error_msg = std::string("SYNTH_RESP unpack: ") + err;
            kill_worker_locked();
            return fail;
        }
        return out;
    }

    // Streaming: loop AUDIO_FRAME → on_pcm; SYNTH_DONE → return; SYNTH_ERR → fail.
    // We always drain to SYNTH_DONE / SYNTH_ERR even after on_pcm returns
    // false (HTTP client disconnect) — otherwise the worker's send_frame
    // back-pressures on a full kernel buffer and we leak a half-stream.
    bool client_alive = true;
    tts_result out;
    while (true) {
        FrameHeader hdr{};
        std::vector<uint8_t> p;
        e = recv_frame(fd_, &hdr, &p);
        if (e != IpcError::OK) {
            fail.error_msg = std::string("AUDIO_FRAME recv failed: ") + ipc_error_str(e);
            kill_worker_locked();
            return fail;
        }
        if (hdr.type == static_cast<uint32_t>(WorkerFrame::AUDIO_FRAME)) {
            std::string meta_str;
            std::vector<float> samples;
            if (!unpack_audio_payload(p, &meta_str, &samples)) {
                fail.error_msg = "AUDIO_FRAME unpack failed";
                kill_worker_locked();
                return fail;
            }
            if (client_alive && on_pcm) {
                if (!on_pcm(samples.data(), samples.size())) {
                    client_alive = false;
                    // keep draining
                }
            }
            continue;
        }
        if (hdr.type == static_cast<uint32_t>(WorkerFrame::SYNTH_DONE)) {
            // SYNTH_DONE payload is JSON metadata; no audio bytes.
            try {
                json j = json::parse(std::string(p.begin(), p.end()));
                result_metadata_from_json(j, out);
            } catch (const std::exception & ex) {
                fail.error_msg = std::string("SYNTH_DONE meta parse: ") + ex.what();
                kill_worker_locked();
                return fail;
            }
            // audio stays empty — caller streamed via on_pcm.
            return out;
        }
        if (hdr.type == static_cast<uint32_t>(WorkerFrame::SYNTH_ERR)) {
            try {
                json j = json::parse(std::string(p.begin(), p.end()));
                fail.error_msg = j.value("error", std::string{"unknown worker error"});
            } catch (...) {
                fail.error_msg = "worker reported error (unparseable)";
            }
            return fail;
        }
        // Unknown frame — skip and continue.
        fprintf(stderr, "worker-session: unexpected frame type=0x%x during stream\n", hdr.type);
    }
}

tts_result WorkerSession::synthesize(const std::string & text,
                                     const tts_params & params) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    return do_synth_locked(text, nullptr, 0, params,
                           nullptr, 0, 0,
                           /*stream_batch_size=*/0, /*stream_first_batch_size=*/0,
                           /*on_pcm=*/{});
}

tts_result WorkerSession::synthesize_with_embedding(
        const std::string & text,
        const float * embedding, int32_t embedding_size,
        const tts_params & params,
        const int32_t * ref_codes, int32_t n_ref_codes,
        int32_t n_ref_frames) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    return do_synth_locked(text, embedding, embedding_size, params,
                           ref_codes, n_ref_codes, n_ref_frames,
                           /*stream_batch_size=*/0, /*stream_first_batch_size=*/0,
                           /*on_pcm=*/{});
}

tts_result WorkerSession::synthesize_with_embedding_streaming(
        const std::string & text,
        const float * embedding, int32_t embedding_size,
        const tts_params & params,
        int32_t stream_batch_size, int32_t stream_first_batch_size,
        const int32_t * ref_codes, int32_t n_ref_codes,
        int32_t n_ref_frames,
        StreamCallback on_pcm) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    return do_synth_locked(text, embedding, embedding_size, params,
                           ref_codes, n_ref_codes, n_ref_frames,
                           stream_batch_size, stream_first_batch_size,
                           std::move(on_pcm));
}

tts_result WorkerSession::synthesize_streaming(
        const std::string & text,
        const tts_params & params,
        int32_t stream_batch_size, int32_t stream_first_batch_size,
        StreamCallback on_pcm) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    return do_synth_locked(text, nullptr, 0, params,
                           nullptr, 0, 0,
                           stream_batch_size, stream_first_batch_size,
                           std::move(on_pcm));
}

bool WorkerSession::extract_speaker_embedding(const std::string & wav_path,
                                              std::vector<float> & out_embedding) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) {
        last_error_ = "worker not ready (call ensure_loaded first)";
        return false;
    }
    json req = { {"filepath", wav_path} };
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame(fd_, WorkerFrame::EXTRACT_EMBED_REQ, req_id, req.dump());
    if (e != IpcError::OK) {
        last_error_ = std::string("EXTRACT_EMBED_REQ send: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    FrameHeader hdr{};
    std::vector<uint8_t> p;
    e = recv_frame(fd_, &hdr, &p);
    if (e != IpcError::OK) {
        last_error_ = std::string("EXTRACT_EMBED_RESP recv: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    if (hdr.type != static_cast<uint32_t>(WorkerFrame::EXTRACT_EMBED_RESP)) {
        last_error_ = std::string("expected EXTRACT_EMBED_RESP, got 0x")
                    + std::to_string(hdr.type);
        kill_worker_locked();
        return false;
    }
    std::string meta_str;
    if (!unpack_audio_payload(p, &meta_str, &out_embedding)) {
        last_error_ = "EXTRACT_EMBED_RESP unpack failed";
        kill_worker_locked();
        return false;
    }
    json meta;
    try { meta = json::parse(meta_str); }
    catch (const std::exception & ex) {
        last_error_ = std::string("EXTRACT_EMBED_RESP meta parse: ") + ex.what();
        return false;
    }
    if (!meta.value("ok", false)) {
        last_error_ = std::string("worker extract failed: ")
                    + meta.value("error", std::string{"(no msg)"});
        out_embedding.clear();
        return false;
    }
    return true;
}

bool WorkerSession::encode_speech_codes(const float * samples, int32_t n_samples,
                                        std::vector<int32_t> & out_codes,
                                        int32_t & out_n_ref_frames) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) {
        last_error_ = "worker not ready";
        return false;
    }
    json meta = { {"n_samples", n_samples} };
    auto payload = pack_audio_payload(meta.dump(), samples, (size_t) n_samples);
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame(fd_, WorkerFrame::ENCODE_CODES_REQ, req_id, payload);
    if (e != IpcError::OK) {
        last_error_ = std::string("ENCODE_CODES_REQ send: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    FrameHeader hdr{};
    std::vector<uint8_t> p;
    e = recv_frame(fd_, &hdr, &p);
    if (e != IpcError::OK) {
        last_error_ = std::string("ENCODE_CODES_RESP recv: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    if (hdr.type != static_cast<uint32_t>(WorkerFrame::ENCODE_CODES_RESP)) {
        last_error_ = std::string("expected ENCODE_CODES_RESP, got 0x")
                    + std::to_string(hdr.type);
        kill_worker_locked();
        return false;
    }
    // Payload layout: [u32 json_len][json][i32 codes...]
    if (p.size() < sizeof(uint32_t)) {
        last_error_ = "ENCODE_CODES_RESP too small";
        return false;
    }
    uint32_t mlen = 0;
    std::memcpy(&mlen, p.data(), sizeof(mlen));
    if (sizeof(mlen) + mlen > p.size()) {
        last_error_ = "ENCODE_CODES_RESP json_len exceeds payload";
        return false;
    }
    json rmeta;
    try {
        rmeta = json::parse(p.data() + sizeof(mlen), p.data() + sizeof(mlen) + mlen);
    } catch (const std::exception & ex) {
        last_error_ = std::string("ENCODE_CODES_RESP meta parse: ") + ex.what();
        return false;
    }
    if (!rmeta.value("ok", false)) {
        last_error_ = std::string("worker encode failed: ")
                    + rmeta.value("error", std::string{"(no msg)"});
        return false;
    }
    out_n_ref_frames = rmeta.value("n_frames", 0);
    size_t code_bytes = p.size() - sizeof(mlen) - mlen;
    if (code_bytes % sizeof(int32_t) != 0) {
        last_error_ = "ENCODE_CODES_RESP misaligned codes payload";
        return false;
    }
    out_codes.resize(code_bytes / sizeof(int32_t));
    if (code_bytes) {
        std::memcpy(out_codes.data(),
                    p.data() + sizeof(mlen) + mlen,
                    code_bytes);
    }
    return true;
}

bool WorkerSession::save_voice_warmup(const std::string & voice_id,
                                      uint64_t prefill_cache_key,
                                      uint64_t ref_codes_hash,
                                      const std::string & path,
                                      const std::string & model_id) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) {
        last_error_ = "worker not ready";
        return false;
    }
    json req = {
        {"voice_id",    voice_id},
        {"prefill_key", prefill_cache_key},
        {"ref_hash",    ref_codes_hash},
        {"path",        path},
        {"model_id",    model_id},
    };
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame(fd_, WorkerFrame::SAVE_WARMUP_REQ, req_id, req.dump());
    if (e != IpcError::OK) {
        last_error_ = std::string("SAVE_WARMUP_REQ send: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    FrameHeader hdr{};
    std::vector<uint8_t> p;
    e = recv_frame(fd_, &hdr, &p);
    if (e != IpcError::OK) {
        last_error_ = std::string("SAVE_WARMUP_RESP recv: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    if (hdr.type != static_cast<uint32_t>(WorkerFrame::SAVE_WARMUP_RESP)) {
        last_error_ = std::string("expected SAVE_WARMUP_RESP, got 0x")
                    + std::to_string(hdr.type);
        kill_worker_locked();
        return false;
    }
    json resp;
    try { resp = json::parse(std::string(p.begin(), p.end())); }
    catch (const std::exception & ex) {
        last_error_ = std::string("SAVE_WARMUP_RESP parse: ") + ex.what();
        return false;
    }
    bool ok = resp.value("ok", false);
    if (!ok) last_error_ = resp.value("error", std::string{"(no msg)"});
    return ok;
}

bool WorkerSession::align_words(const std::vector<std::string> & words,
                                std::vector<AlignedWord>        & out_words,
                                AlignProfile                    & out_profile) {
    out_words.clear();
    out_profile = AlignProfile{};
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) {
        last_error_ = "worker not ready (call ensure_loaded first)";
        return false;
    }
    if (loaded_cfg_.aligner_model.empty()) {
        last_error_ = "aligner_model not configured at load time";
        return false;
    }
    if (words.empty()) {
        last_error_ = "align_words: empty word list";
        return false;
    }
    json req = { {"words", words} };
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame(fd_, WorkerFrame::ALIGN_REQ, req_id, req.dump());
    if (e != IpcError::OK) {
        last_error_ = std::string("ALIGN_REQ send: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    FrameHeader hdr{};
    std::vector<uint8_t> p;
    e = recv_frame(fd_, &hdr, &p);
    if (e != IpcError::OK) {
        last_error_ = std::string("ALIGN_RESP recv: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    if (hdr.type != static_cast<uint32_t>(WorkerFrame::ALIGN_RESP)) {
        last_error_ = std::string("expected ALIGN_RESP, got 0x")
                    + std::to_string(hdr.type);
        kill_worker_locked();
        return false;
    }
    json resp;
    try { resp = json::parse(std::string(p.begin(), p.end())); }
    catch (const std::exception & ex) {
        last_error_ = std::string("ALIGN_RESP parse: ") + ex.what();
        return false;
    }
    if (!resp.value("ok", false)) {
        last_error_ = std::string("worker align failed: ")
                    + resp.value("error", std::string{"(no msg)"});
        return false;
    }
    if (resp.contains("words") && resp["words"].is_array()) {
        out_words.reserve(resp["words"].size());
        for (const auto & w : resp["words"]) {
            AlignedWord aw;
            aw.text  = w.value("text",  std::string{});
            aw.t0_ms = w.value("t0_ms", (int64_t) 0);
            aw.t1_ms = w.value("t1_ms", (int64_t) 0);
            out_words.push_back(std::move(aw));
        }
    }
    if (resp.contains("profile") && resp["profile"].is_object()) {
        const auto & pf = resp["profile"];
        out_profile.t_load_ms     = pf.value("t_load_ms",     (int64_t) 0);
        out_profile.t_resample_ms = pf.value("t_resample_ms", (int64_t) 0);
        out_profile.t_mel_ms      = pf.value("t_mel_ms",      (int64_t) 0);
        out_profile.t_encoder_ms  = pf.value("t_encoder_ms",  (int64_t) 0);
        out_profile.t_aligner_ms  = pf.value("t_aligner_ms",  (int64_t) 0);
        out_profile.t_total_ms    = pf.value("t_total_ms",    (int64_t) 0);
        out_profile.n_enc         = pf.value("n_enc",    0);
        out_profile.n_prompt      = pf.value("n_prompt", 0);
        out_profile.n_words       = pf.value("n_words",  0);
    }
    return true;
}

// ─────────────────── Streaming alignment (parent side) ──────────────────
//
// Three-step protocol against an aligner-only sibling worker:
//   begin_streaming_align(words, pcm_sr) → reset accumulator, stash words
//   push_partial_pcm(pcm, n, audio_seen_ms) → send delta, fire-and-forget
//   drain_partial_alignments(cb) → consume ALIGN_PARTIAL_RESP pending on fd
//   finalize_streaming_align(tail_pcm, ...) → ALIGN_FINAL_REQ + wait for resp
//
// Single in-flight stream per session. The reset is signalled to the
// worker via {"reset": true} on the first PARTIAL_REQ.

bool WorkerSession::begin_streaming_align(const std::vector<std::string> & words,
                                          int pcm_sample_rate) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (stream_align_active_) {
        last_error_ = "begin_streaming_align: previous stream not finalized";
        return false;
    }
    if (words.empty()) {
        last_error_ = "begin_streaming_align: empty word list";
        return false;
    }
    if (pcm_sample_rate <= 0) {
        last_error_ = "begin_streaming_align: bad pcm_sample_rate";
        return false;
    }
    stream_align_words_  = words;
    stream_align_pcm_sr_ = pcm_sample_rate;
    stream_align_active_ = true;
    return true;
}

bool WorkerSession::push_partial_pcm(const float * pcm, size_t n_samples,
                                     int64_t audio_seen_ms) {
    if (!pcm && n_samples > 0) {
        last_error_ = "push_partial_pcm: null pcm with n_samples>0";
        return false;
    }
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (!stream_align_active_) {
        last_error_ = "push_partial_pcm: no active stream (call begin_streaming_align first)";
        return false;
    }
    if (fd_ < 0) {
        last_error_ = "push_partial_pcm: worker not running";
        return false;
    }

    static thread_local bool sent_first_req = false;
    // The "reset" flag flips on the very first PARTIAL_REQ of a stream so
    // the worker drops any stale accumulator from a prior paragraph. We
    // approximate "first call" with a per-stream req_id parity — req_id_low
    // grows monotonically; first push sees req_id_low == 0 mod something.
    // Simpler: store a flag in session state. Done via stream_align_active_
    // + a per-session "sent_any_partial_yet" bool — let's just track it.
    (void) sent_first_req;
    // Use stream_align_pcm_sr_==0 sentinel? No, we set it in begin. Add a
    // dedicated flag below the active flag.
    json meta = {
        {"words",             stream_align_words_},
        {"pcm_sample_rate",   stream_align_pcm_sr_},
        {"audio_seen_ms",     audio_seen_ms},
        {"reset",             false},  // overwritten below if first call
    };
    // First call after begin: reset accumulator on the worker side. We
    // detect this by inspecting an internal counter on the session (the
    // begin_streaming_align flips the flag; first push resets it).
    // For now use stream_align_words_'s last-known req_id_baseline. Cheap:
    // we set reset=true if next_req_id_ wasn't bumped since begin → tracked
    // via an inline flag.
    // (No-op simplification: ALWAYS send reset=false here; rely on a
    // separate explicit reset path through begin_streaming_align if
    // strict-reset behaviour is needed. Worker treats consecutive
    // streams as a single append-only buffer otherwise.)
    if (!stream_align_has_sent_any_) {
        meta["reset"] = true;
        stream_align_has_sent_any_ = true;
    }

    std::vector<uint8_t> payload = pack_audio_payload(meta.dump(), pcm, n_samples);
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame(fd_, WorkerFrame::ALIGN_PARTIAL_REQ, req_id,
                            payload.data(), payload.size());
    if (e != IpcError::OK) {
        last_error_ = std::string("ALIGN_PARTIAL_REQ send: ") + ipc_error_str(e);
        return false;
    }
    return true;
}

// Non-blocking drain. Reads all currently-pending ALIGN_PARTIAL_RESP frames
// on fd_ and invokes cb for each. Stops as soon as poll() reports no more
// data. Holds io_mutex_ for the duration — kept short by the poll(0) bound.
bool WorkerSession::drain_partial_alignments(const PartialAlignCallback & cb) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (fd_ < 0) {
        // Not an error per se — caller may drain after worker is gone.
        return true;
    }
    for (;;) {
        struct pollfd pfd { fd_, POLLIN, 0 };
        int pr = ::poll(&pfd, 1, 0); // 0 ms — strictly non-blocking
        if (pr <= 0) return true;
        if (!(pfd.revents & POLLIN)) return true;

        FrameHeader hdr{};
        std::vector<uint8_t> p;
        IpcError e = recv_frame(fd_, &hdr, &p);
        if (e == IpcError::EofClean || e == IpcError::EofMidFrame) {
            last_error_ = "aligner worker EOF mid-stream";
            return false;
        }
        if (e != IpcError::OK) {
            last_error_ = std::string("ALIGN_PARTIAL_RESP recv: ") + ipc_error_str(e);
            return false;
        }
        // Tolerate non-PARTIAL frames (PONG could in principle arrive
        // out-of-band) — only consume ALIGN_PARTIAL_RESP.
        if (hdr.type != static_cast<uint32_t>(WorkerFrame::ALIGN_PARTIAL_RESP)) {
            // Anything else this side of the protocol is unexpected; drop.
            continue;
        }
        try {
            json r = json::parse(std::string(p.begin(), p.end()));
            if (!r.value("ok", false)) {
                // Worker reported a per-call error; surface in last_error_
                // but keep the stream open — caller decides whether to
                // continue or finalize.
                last_error_ = r.value("error", std::string("PARTIAL_RESP error"));
                continue;
            }
            std::vector<AlignedWord> words;
            for (const auto & w : r["words"]) {
                AlignedWord aw;
                aw.text  = w.value("text",  std::string{});
                aw.t0_ms = w.value("t0_ms", (int64_t) 0);
                aw.t1_ms = w.value("t1_ms", (int64_t) 0);
                words.push_back(std::move(aw));
            }
            const int64_t audio_seen_ms = r.value("audio_seen_ms", (int64_t) 0);
            // Release the mutex around the user callback so a slow client
            // can't backpressure other WorkerSession operations.
            io_mutex_.unlock();
            cb(audio_seen_ms, words);
            io_mutex_.lock();
        } catch (const std::exception & ex) {
            last_error_ = std::string("ALIGN_PARTIAL_RESP parse: ") + ex.what();
            return false;
        }
    }
}

bool WorkerSession::finalize_streaming_align(const float * tail_pcm,
                                             size_t n_tail_samples,
                                             int64_t audio_total_ms,
                                             std::vector<AlignedWord> & out_words,
                                             AlignProfile & out_profile) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    // Reset session state unconditionally on exit so a failed finalize
    // (socket error, parse error, worker died) doesn't strand
    // stream_align_active_=true and break the next paragraph's
    // begin_streaming_align. State is request-local; the caller already
    // sees the failure via the return value.
    struct ResetOnExit {
        WorkerSession * s;
        ~ResetOnExit() {
            s->stream_align_active_       = false;
            s->stream_align_has_sent_any_ = false;
            s->stream_align_words_.clear();
            s->stream_align_pcm_sr_       = 0;
        }
    } reset_on_exit{this};
    if (!stream_align_active_) {
        last_error_ = "finalize_streaming_align: no active stream";
        return false;
    }
    if (fd_ < 0) {
        last_error_ = "finalize_streaming_align: worker not running";
        return false;
    }

    // Drain any stragglers FIRST so the worker's accumulator is fully
    // up-to-date before we issue FINAL. We don't surface them through cb
    // here — caller's reader thread (if any) should have already seen
    // them. Whatever remains is silently dropped.
    for (;;) {
        struct pollfd pfd { fd_, POLLIN, 0 };
        int pr = ::poll(&pfd, 1, 0);
        if (pr <= 0) break;
        if (!(pfd.revents & POLLIN)) break;
        FrameHeader h{}; std::vector<uint8_t> p;
        IpcError e = recv_frame(fd_, &h, &p);
        if (e != IpcError::OK) break;
        if (h.type != static_cast<uint32_t>(WorkerFrame::ALIGN_PARTIAL_RESP)) {
            // Could be an old PONG; ignore.
            continue;
        }
    }

    json meta = {
        {"words",           stream_align_words_},
        {"pcm_sample_rate", stream_align_pcm_sr_},
        {"audio_total_ms",  audio_total_ms},
        {"reset",           false},
    };
    std::vector<uint8_t> payload = pack_audio_payload(meta.dump(), tail_pcm, n_tail_samples);
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame(fd_, WorkerFrame::ALIGN_FINAL_REQ, req_id,
                            payload.data(), payload.size());
    if (e != IpcError::OK) {
        last_error_ = std::string("ALIGN_FINAL_REQ send: ") + ipc_error_str(e);
        return false;
    }

    // Block for the matching FINAL_RESP. Tolerate stragglers in between
    // (any pending PARTIAL_RESP gets discarded — caller had its chance).
    for (;;) {
        FrameHeader h{}; std::vector<uint8_t> p;
        IpcError re = recv_frame(fd_, &h, &p);
        if (re != IpcError::OK) {
            last_error_ = std::string("ALIGN_FINAL_RESP recv: ") + ipc_error_str(re);
            return false;
        }
        if (h.type == static_cast<uint32_t>(WorkerFrame::ALIGN_PARTIAL_RESP)) {
            continue; // drop stragglers
        }
        if (h.type != static_cast<uint32_t>(WorkerFrame::ALIGN_FINAL_RESP)) {
            last_error_ = std::string("expected ALIGN_FINAL_RESP, got 0x")
                        + std::to_string(h.type);
            return false;
        }
        try {
            json r = json::parse(std::string(p.begin(), p.end()));
            if (!r.value("ok", false)) {
                last_error_ = r.value("error", std::string("FINAL_RESP error"));
                return false;
            }
            out_words.clear();
            for (const auto & w : r["words"]) {
                AlignedWord aw;
                aw.text  = w.value("text",  std::string{});
                aw.t0_ms = w.value("t0_ms", (int64_t) 0);
                aw.t1_ms = w.value("t1_ms", (int64_t) 0);
                out_words.push_back(std::move(aw));
            }
            if (r.contains("profile") && r["profile"].is_object()) {
                const auto & pf = r["profile"];
                out_profile.t_load_ms     = pf.value("t_load_ms",     (int64_t) 0);
                out_profile.t_resample_ms = pf.value("t_resample_ms", (int64_t) 0);
                out_profile.t_mel_ms      = pf.value("t_mel_ms",      (int64_t) 0);
                out_profile.t_encoder_ms  = pf.value("t_encoder_ms",  (int64_t) 0);
                out_profile.t_aligner_ms  = pf.value("t_aligner_ms",  (int64_t) 0);
                out_profile.t_total_ms    = pf.value("t_total_ms",    (int64_t) 0);
                out_profile.n_enc         = pf.value("n_enc",    0);
                out_profile.n_prompt      = pf.value("n_prompt", 0);
                out_profile.n_words       = pf.value("n_words",  0);
            }
        } catch (const std::exception & ex) {
            last_error_ = std::string("ALIGN_FINAL_RESP parse: ") + ex.what();
            return false;
        }
        break;
    }

    // Successful path: ResetOnExit will run the same clear sequence
    // below, but only after we've signalled success to the caller.
    return true;
}

// ───────────────────────── run_worker_loop (child) ─────────────────────

int run_worker_loop(int fd) {
    // Same unbuffered-stderr discipline as main() for crash handler logs.
    setvbuf(stderr, nullptr, _IONBF, 0);

    // Linux PR_SET_PDEATHSIG: ask the kernel to send SIGTERM if our parent
    // dies. Belt-and-braces against an orphaned worker holding 2.6 GiB of
    // VRAM after the parent process exits abnormally. (We can't follow up
    // with the usual `if (getppid() == 1) bail` race-mitigation: in a
    // container the parent IS pid 1, so that check would fire on every
    // boot.) If parent really did die in the fork/exec gap, the next
    // recv_frame returns EofClean and we exit normally.
#if defined(__linux__)
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0) {
        fprintf(stderr, "worker: prctl(PR_SET_PDEATHSIG) failed: %s (continuing)\n",
                strerror(errno));
    }
#endif

    fprintf(stderr, "worker[%d]: alive on fd=%d ppid=%d\n",
            (int) getpid(), fd, (int) getppid());

    // HELLO with our pid so the parent has a sanity-check.
    json hello = {
        {"pid",   (int) getpid()},
        {"role",  "qwen3-tts-worker"},
    };
    if (send_frame(fd, WorkerFrame::HELLO, 0, hello.dump()) != IpcError::OK) {
        fprintf(stderr, "worker: HELLO send failed; bailing\n");
        return 2;
    }

    Qwen3TTS tts;

    // ─── Forced-alignment session state ─────────────────────────────────
    // last_pcm / last_sr cache the most recent synth's PCM so ALIGN_REQ
    // can run without the parent re-sending audio. Captured both for
    // non-streaming (from result.audio) and streaming (accumulated in
    // sopts.on_pcm) synth paths.
    //
    // fa_model_path is set from LOAD_REQ; aligner context (fa_ctx) is
    // lazy-loaded on first ALIGN_REQ to keep startup VRAM at zero for
    // sessions that never request alignment. The cost is ~700-1000ms
    // on first align request (siglip2-style cold load) and stays in
    // VRAM until the worker dies (idle-unload SIGKILL reclaims all
    // VRAM in one shot — no per-call unload needed for v1).
    std::vector<float> last_pcm;
    int                last_sr = 0;
    std::string        fa_model_path;
#if QWEN3_TTS_HAS_FORCED_ALIGNMENT
    qwen3_asr_context * fa_ctx = nullptr;
#endif

    while (true) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        IpcError e = recv_frame(fd, &hdr, &payload);
        if (e == IpcError::EofClean) {
            fprintf(stderr, "worker: parent EOF, exiting cleanly\n");
            return 0;
        }
        if (e != IpcError::OK) {
            fprintf(stderr, "worker: recv_frame failed: %s\n", ipc_error_str(e));
            return 3;
        }

        switch (static_cast<WorkerFrame>(hdr.type)) {
            case WorkerFrame::SHUTDOWN: {
                fprintf(stderr, "worker: SHUTDOWN, exiting\n");
                return 0;
            }
            case WorkerFrame::PING: {
                send_frame(fd, WorkerFrame::PONG, hdr.req_id, payload);
                break;
            }
            case WorkerFrame::LOAD_REQ: {
                json req;
                std::string err_msg;
                bool ok = false;
                std::string voice_archive_dir;
                std::string model_id;
                try {
                    req = json::parse(std::string(payload.begin(), payload.end()));
                    std::string model     = req.value("model",             std::string{});
                    std::string vocoder   = req.value("vocoder",           std::string{});
                    std::string spk_enc   = req.value("speaker_encoder",   std::string{});
                    voice_archive_dir     = req.value("voice_archive_dir", std::string{});
                    model_id              = req.value("model_id",          std::string{});
                    fa_model_path         = req.value("aligner_model",     std::string{});
                    bool        lazy_load = req.value("lazy_load",         false);

                    if (lazy_load) {
                        tts.set_model_paths(model, vocoder, spk_enc);
                        ok = true;
                    } else {
                        ok = tts.load_model_files(model, vocoder, spk_enc);
                        if (!ok) err_msg = tts.get_error();
                    }
                } catch (const std::exception & ex) {
                    err_msg = std::string("LOAD_REQ parse failed: ") + ex.what();
                }

                // After successful load, scan voice_archive_dir for *.warmup
                // and inject into the model's prefill cache. This is what
                // gives us cross-restart warmup hits in worker mode.
                int loaded_warmups = 0;
                if (ok && tts.is_loaded() && !voice_archive_dir.empty()) {
                    try {
                        for (const auto & entry : std::filesystem::directory_iterator(voice_archive_dir)) {
                            if (!entry.is_directory()) continue;
                            const auto wp = entry.path() / "voice.warmup";
                            if (!std::filesystem::exists(wp)) continue;
                            if (tts.load_voice_warmup(wp.string(), model_id)) {
                                loaded_warmups++;
                            }
                        }
                    } catch (const std::exception & ex) {
                        fprintf(stderr, "worker: voice-warmup scan: %s\n", ex.what());
                    }
                    fprintf(stderr, "worker: loaded %d voice.warmup blob(s) from %s\n",
                            loaded_warmups, voice_archive_dir.c_str());
                }

                int sr = (ok && tts.is_loaded()) ? tts.get_sample_rate() : 0;
                json resp = {
                    {"ok",             ok},
                    {"error",          err_msg},
                    {"sample_rate",    sr},
                    {"loaded_warmups", loaded_warmups},
                };
                if (send_frame(fd, WorkerFrame::LOAD_RESP, hdr.req_id,
                               resp.dump()) != IpcError::OK) {
                    fprintf(stderr, "worker: LOAD_RESP send failed\n");
                    return 4;
                }
                break;
            }
            case WorkerFrame::SYNTH_REQ: {
                std::string text;
                std::vector<float>   embedding;
                std::vector<int32_t> ref_codes;
                int32_t n_ref_frames           = 0;
                int32_t stream_batch_size      = 0;
                int32_t stream_first_batch_size = 0;
                tts_params params;
                std::string err;

                if (!unpack_synth_payload(payload, &text, &embedding, &ref_codes,
                                          &n_ref_frames, &stream_batch_size,
                                          &stream_first_batch_size,
                                          &params, &err)) {
                    json e = { {"error", err} };
                    send_frame(fd, WorkerFrame::SYNTH_ERR, hdr.req_id, e.dump());
                    break;
                }

                if (!tts.is_loaded()) {
                    if (!tts.reload_model()) {
                        json e = { {"error",
                                    std::string("worker reload_model failed: ")
                                    + tts.get_error()} };
                        send_frame(fd, WorkerFrame::SYNTH_ERR, hdr.req_id, e.dump());
                        break;
                    }
                }

                // Reset the FA scratch buffer for each new synth so an
                // ALIGN_REQ after a failed/aborted synth can't accidentally
                // align against stale audio from two synths ago.
                last_pcm.clear();
                last_sr = 0;

                const bool streaming = (stream_batch_size > 0);
                streaming_opts sopts;
                std::atomic<bool> ipc_ok{ true };
                if (streaming) {
                    sopts.batch_size       = stream_batch_size;
                    sopts.first_batch_size = stream_first_batch_size;
                    const uint32_t cb_req_id = hdr.req_id;
                    sopts.on_pcm = [fd, cb_req_id, &ipc_ok, &last_pcm]
                                   (const float * pcm, size_t n) -> bool {
                        // Accumulate into FA scratch BEFORE IPC send so
                        // even a half-streamed synth (IPC failure mid-way)
                        // leaves usable audio for diagnostics. Resampling
                        // to 16k happens lazily in the ALIGN_REQ handler.
                        last_pcm.insert(last_pcm.end(), pcm, pcm + n);
                        if (!ipc_ok.load()) return false;
                        // AUDIO_FRAME payload uses pack_audio_payload's
                        // [u32 json_len][json][f32 samples] format. Empty json
                        // — chunk metadata is positional (req_id in header,
                        // sample count derivable from byte length).
                        auto buf = pack_audio_payload(std::string{},
                                                     pcm, n);
                        IpcError e2 = send_frame(fd, WorkerFrame::AUDIO_FRAME,
                                                 cb_req_id, buf);
                        if (e2 != IpcError::OK) {
                            fprintf(stderr, "worker: AUDIO_FRAME send failed: %s\n",
                                    ipc_error_str(e2));
                            ipc_ok.store(false);
                            return false;
                        }
                        return true;
                    };
                }

                tts_result result;
                if (!ref_codes.empty()) {
                    result = tts.synthesize_with_embedding(
                        text, embedding.data(), (int32_t) embedding.size(),
                        params, ref_codes.data(), n_ref_frames,
                        streaming ? &sopts : nullptr);
                } else if (!embedding.empty()) {
                    result = tts.synthesize_with_embedding(
                        text, embedding.data(), (int32_t) embedding.size(),
                        params, nullptr, 0,
                        streaming ? &sopts : nullptr);
                } else {
                    result = tts.synthesize(text, params,
                                            streaming ? &sopts : nullptr);
                }

                // Non-streaming path: result.audio holds the full PCM.
                // Streaming path: last_pcm was populated by on_pcm above.
                // Either way, stash the sample rate so ALIGN_REQ can
                // resample to 16k correctly.
                if (result.success) {
                    last_sr = result.sample_rate;
                    if (!streaming && !result.audio.empty()) {
                        last_pcm = result.audio;
                    }
                }

                if (!streaming) {
                    auto resp = pack_synth_resp(result);
                    if (send_frame(fd, WorkerFrame::SYNTH_RESP, hdr.req_id, resp)
                        != IpcError::OK) {
                        fprintf(stderr, "worker: SYNTH_RESP send failed\n");
                        return 5;
                    }
                } else {
                    // Streaming: send SYNTH_DONE with metadata only (no audio
                    // bytes — those streamed already as AUDIO_FRAMEs).
                    if (!ipc_ok.load() && !result.success) {
                        // both IPC and synth failed — best-effort error frame
                        json e = { {"error",
                                    std::string("worker synth failed: ")
                                    + result.error_msg} };
                        send_frame(fd, WorkerFrame::SYNTH_ERR, hdr.req_id, e.dump());
                        break;
                    }
                    json meta = result_metadata_to_json(result);
                    if (send_frame(fd, WorkerFrame::SYNTH_DONE, hdr.req_id,
                                   meta.dump()) != IpcError::OK) {
                        fprintf(stderr, "worker: SYNTH_DONE send failed\n");
                        return 5;
                    }
                }
                break;
            }
            case WorkerFrame::EXTRACT_EMBED_REQ: {
                std::string fp;
                std::vector<float> embedding;
                std::string err_msg;
                bool ok = false;
                try {
                    json req = json::parse(std::string(payload.begin(), payload.end()));
                    fp = req.value("filepath", std::string{});
                } catch (const std::exception & ex) {
                    err_msg = std::string("EXTRACT_EMBED_REQ parse: ") + ex.what();
                }
                if (err_msg.empty()) {
                    if (!tts.is_loaded() && !tts.reload_model()) {
                        err_msg = std::string("worker reload_model: ") + tts.get_error();
                    } else {
                        ok = tts.extract_speaker_embedding(fp, embedding);
                        if (!ok) err_msg = tts.get_error();
                    }
                }
                json meta = { {"ok", ok}, {"error", err_msg},
                              {"n_floats", (int) embedding.size()} };
                auto resp = pack_audio_payload(meta.dump(),
                                               embedding.data(), embedding.size());
                if (send_frame(fd, WorkerFrame::EXTRACT_EMBED_RESP, hdr.req_id, resp)
                    != IpcError::OK) {
                    fprintf(stderr, "worker: EXTRACT_EMBED_RESP send failed\n");
                    return 6;
                }
                break;
            }
            case WorkerFrame::ENCODE_CODES_REQ: {
                std::string err_msg;
                bool ok = false;
                std::vector<int32_t> codes;
                int32_t n_frames = 0;
                std::string in_meta_str;
                std::vector<float> samples;
                if (!unpack_audio_payload(payload, &in_meta_str, &samples)) {
                    err_msg = "ENCODE_CODES_REQ unpack failed";
                } else if (!tts.is_loaded() && !tts.reload_model()) {
                    err_msg = std::string("worker reload_model: ") + tts.get_error();
                } else {
                    ok = tts.encode_speech_codes(samples.data(),
                                                 (int32_t) samples.size(),
                                                 codes, n_frames);
                    if (!ok) err_msg = tts.get_error();
                }
                json meta = { {"ok", ok}, {"error", err_msg},
                              {"n_frames", n_frames} };
                std::string ms = meta.dump();
                uint32_t mlen = (uint32_t) ms.size();
                size_t code_bytes = codes.size() * sizeof(int32_t);
                std::vector<uint8_t> resp(sizeof(mlen) + ms.size() + code_bytes);
                std::memcpy(resp.data(), &mlen, sizeof(mlen));
                std::memcpy(resp.data() + sizeof(mlen), ms.data(), ms.size());
                if (code_bytes) {
                    std::memcpy(resp.data() + sizeof(mlen) + ms.size(),
                                codes.data(), code_bytes);
                }
                if (send_frame(fd, WorkerFrame::ENCODE_CODES_RESP, hdr.req_id, resp)
                    != IpcError::OK) {
                    fprintf(stderr, "worker: ENCODE_CODES_RESP send failed\n");
                    return 7;
                }
                break;
            }
            case WorkerFrame::SAVE_WARMUP_REQ: {
                std::string err_msg;
                bool ok = false;
                try {
                    json req = json::parse(std::string(payload.begin(), payload.end()));
                    std::string voice_id     = req.value("voice_id",    std::string{});
                    uint64_t prefill_key     = req.value("prefill_key", (uint64_t) 0);
                    uint64_t ref_hash        = req.value("ref_hash",    (uint64_t) 0);
                    std::string path         = req.value("path",        std::string{});
                    std::string model_id     = req.value("model_id",    std::string{});
                    if (path.empty() || prefill_key == 0) {
                        err_msg = "SAVE_WARMUP_REQ missing path or prefill_key";
                    } else {
                        ok = tts.save_voice_warmup(voice_id, prefill_key, ref_hash,
                                                   path, model_id);
                        if (!ok) err_msg = tts.get_error();
                    }
                } catch (const std::exception & ex) {
                    err_msg = std::string("SAVE_WARMUP_REQ parse: ") + ex.what();
                }
                json resp = { {"ok", ok}, {"error", err_msg} };
                if (send_frame(fd, WorkerFrame::SAVE_WARMUP_RESP, hdr.req_id, resp.dump())
                    != IpcError::OK) {
                    fprintf(stderr, "worker: SAVE_WARMUP_RESP send failed\n");
                    return 8;
                }
                break;
            }
#if QWEN3_TTS_HAS_FORCED_ALIGNMENT
            case WorkerFrame::ALIGN_REQ: {
                using clk = std::chrono::steady_clock;
                const auto t_start = clk::now();
                json req;
                std::vector<std::string> words;
                std::string err_msg;
                bool ok = false;
                std::vector<int64_t> out_t0_ms, out_t1_ms;
                int64_t t_load_ms = 0, t_resample_ms = 0, t_align_ms = 0;
                std::vector<float> samples_16k;
                try {
                    req = json::parse(std::string(payload.begin(), payload.end()));
                    if (req.contains("words") && req["words"].is_array()) {
                        for (const auto & w : req["words"]) {
                            if (w.is_string()) words.push_back(w.get<std::string>());
                        }
                    }
                } catch (const std::exception & ex) {
                    err_msg = std::string("ALIGN_REQ parse: ") + ex.what();
                }
                if (err_msg.empty() && words.empty()) {
                    err_msg = "ALIGN_REQ words list empty";
                }
                if (err_msg.empty() && fa_model_path.empty()) {
                    err_msg = "aligner_model not configured (set --hf-repo-fa)";
                }
                if (err_msg.empty() && last_pcm.empty()) {
                    err_msg = "no synth audio cached — call /v1/audio/speech first";
                }
                if (err_msg.empty() && last_sr <= 0) {
                    err_msg = "last synth sample_rate unknown";
                }

                // Lazy-load aligner GGUF on first request. Subsequent
                // requests reuse the same context; teardown happens
                // either on worker exit (idle-unload → SIGKILL) or
                // on a configured aligner-model swap (P2).
                if (err_msg.empty() && fa_ctx == nullptr) {
                    const bool probe_v = []() {
                        const char* e = std::getenv("QWEN3_FA_PROFILE_VRAM");
                        return e && *e && std::atoi(e) > 0;
                    }();
                    auto v_used = [&]() {
                        size_t f=0,t=0; auto g=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
                        if (g) ggml_backend_dev_memory(g,&f,&t);
                        return (double)(t-f)/1048576.0;
                    };
                    if (probe_v) fprintf(stderr, "  [fa-vram %-14s] gpu_used=%7.1f MiB\n",
                                          "pre-load", v_used());
                    const auto t0 = clk::now();
                    qwen3_asr_context_params p = qwen3_asr_context_default_params();
                    fa_ctx = qwen3_asr_init_from_file(fa_model_path.c_str(), p);
                    if (probe_v) fprintf(stderr, "  [fa-vram %-14s] gpu_used=%7.1f MiB\n",
                                          "post-load", v_used());
                    if (!fa_ctx) {
                        err_msg = std::string("qwen3_asr_init_from_file failed: ")
                                + fa_model_path;
                    } else {
                        const int H = qwen3_asr_lm_head_dim(fa_ctx);
                        if (H <= 0) {
                            err_msg = "aligner has zero lm_head_dim";
                            qwen3_asr_free(fa_ctx);
                            fa_ctx = nullptr;
                        } else if (H > 8192) {
                            // Sanity: FA variant has H≈5000; full-vocab ASR
                            // variants are 152064/151936. Refuse to use an
                            // ASR-only model here — the output won't be
                            // timestamps.
                            fprintf(stderr, "worker: aligner lm_head_dim=%d looks like an "
                                            "ASR model, not a forced aligner — refusing\n", H);
                            err_msg = "aligner GGUF appears to be an ASR model (lm_head too wide)";
                            qwen3_asr_free(fa_ctx);
                            fa_ctx = nullptr;
                        }
                    }
                    t_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    clk::now() - t0).count();
                    if (fa_ctx) {
                        fprintf(stderr, "worker: aligner loaded (lm_head_dim=%d) in %lld ms\n",
                                qwen3_asr_lm_head_dim(fa_ctx), (long long) t_load_ms);
                    }
                }

                // Resample last_pcm to 16 kHz mono float. last_pcm is
                // already mono float at last_sr (24 kHz V1 or 48 kHz V2).
                // Linear interp is good enough at 80 ms aligner resolution
                // — kaiser/sinc would be overkill for word timings.
                if (err_msg.empty()) {
                    const auto t0 = clk::now();
                    const int target_sr = 16000;
                    if (last_sr == target_sr) {
                        samples_16k = last_pcm;
                    } else {
                        const double ratio = (double) last_sr / (double) target_sr;
                        const int n_in  = (int) last_pcm.size();
                        const int n_out = (int) ((double) n_in / ratio);
                        samples_16k.resize((size_t) n_out);
                        for (int i = 0; i < n_out; i++) {
                            const double src = i * ratio;
                            const int idx0 = (int) src;
                            const int idx1 = idx0 + 1;
                            const double frac = src - idx0;
                            const float a = last_pcm[idx0];
                            const float b = (idx1 < n_in) ? last_pcm[idx1] : a;
                            samples_16k[i] = (float) ((1.0 - frac) * a + frac * b);
                        }
                    }
                    t_resample_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        clk::now() - t0).count();
                }

                // Run the high-level FA entry point. v1 measures total
                // align time only; sub-stage profiling (mel/encoder/
                // aligner-forward) is a v2 ask once we know overall RTF.
                if (err_msg.empty()) {
                    out_t0_ms.assign(words.size(), 0);
                    out_t1_ms.assign(words.size(), 0);
                    std::vector<const char *> cstr_words;
                    cstr_words.reserve(words.size());
                    for (const auto & w : words) cstr_words.push_back(w.c_str());

                    const auto t0 = clk::now();
                    const int rc = qwen3_asr_align_words(
                            fa_ctx,
                            samples_16k.data(), (int) samples_16k.size(),
                            cstr_words.data(),  (int) cstr_words.size(),
                            out_t0_ms.data(), out_t1_ms.data());
                    t_align_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     clk::now() - t0).count();
                    if (rc != 0) {
                        err_msg = std::string("qwen3_asr_align_words rc=")
                                + std::to_string(rc);
                    } else {
                        ok = true;
                    }
                }

                json resp;
                resp["ok"]    = ok;
                resp["error"] = err_msg;
                if (ok) {
                    json wj = json::array();
                    for (size_t i = 0; i < words.size(); i++) {
                        wj.push_back({
                            {"text",  words[i]},
                            {"t0_ms", out_t0_ms[i]},
                            {"t1_ms", out_t1_ms[i]},
                        });
                    }
                    resp["words"] = std::move(wj);
                    // Duration of the resampled 16k buffer, not last_pcm,
                    // so it's exactly what the aligner saw.
                    const int64_t dur_ms = samples_16k.empty()
                            ? 0
                            : (int64_t) ((samples_16k.size() * 1000ll) / 16000);
                    resp["duration_ms"] = dur_ms;
                }
                const int64_t t_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                               clk::now() - t_start).count();
                resp["profile"] = {
                    {"t_load_ms",     t_load_ms},
                    {"t_resample_ms", t_resample_ms},
                    {"t_aligner_ms",  t_align_ms}, // includes mel + encoder + body
                    {"t_total_ms",    t_total_ms},
                    {"n_words",       (int) words.size()},
                    {"n_samples_16k", (int) samples_16k.size()},
                    {"src_sample_rate", last_sr},
                };
                if (send_frame(fd, WorkerFrame::ALIGN_RESP, hdr.req_id, resp.dump())
                    != IpcError::OK) {
                    fprintf(stderr, "worker: ALIGN_RESP send failed\n");
                    return 9;
                }
                break;
            }
#endif // QWEN3_TTS_HAS_FORCED_ALIGNMENT
            default:
                fprintf(stderr, "worker: unexpected frame type=0x%x len=%u\n",
                        hdr.type, hdr.len);
                break;
        }
    }
    // Unreachable: loop above only exits via `return`. fa_ctx (if loaded)
    // is reclaimed when the worker process is SIGKILL'd by idle-unload
    // or parent shutdown — same pattern Qwen3TTS uses for its own ggml/
    // CUDA state.
}

// ───────────────────── run_aligner_worker_loop (child) ────────────────
//
// Aligner-only sibling subprocess (invoked via --worker-aligner <fd>).
// Loads the FA GGUF on first ALIGN_PARTIAL_REQ; keeps a PCM accumulator
// that grows as the parent sends deltas. Re-runs the full FA encode +
// LLM body per pass so each partial response reflects the most up-to-
// date audio. No talker, no vocoder, no spk-encoder — the only model in
// VRAM is the aligner.
//
// Phase-2 architectural rationale: a sibling subprocess gives us a
// separate CUDA context, so the synth worker's GPU work and this
// worker's encode+align passes can run in true parallel (no shared
// stream contention). The trade is ~700 MiB VRAM ceiling per aligner
// load + IPC overhead for PCM streaming (~96 KB/sec at 24 kHz F32).

int run_aligner_worker_loop(int fd) {
#if !QWEN3_TTS_HAS_FORCED_ALIGNMENT
    fprintf(stderr, "aligner-worker: built without forced-alignment support\n");
    json resp = { {"ok", false}, {"error", "no forced-alignment support"} };
    send_frame(fd, WorkerFrame::HELLO, 0, json{{"pid",(int)getpid()},{"role","qwen3-tts-aligner"}}.dump());
    (void) resp;
    return 1;
#else
    setvbuf(stderr, nullptr, _IONBF, 0);
    // NOTE: intentionally NOT calling prctl(PR_SET_PDEATHSIG) here.
    //
    // Linux's PR_SET_PDEATHSIG triggers when the *thread that created
    // this process via fork* terminates — not when the parent process
    // dies. With the eager-aligner-spawn pattern in server.cpp, the
    // aligner subprocess is forked from a short-lived std::thread that
    // hands off after ensure_loaded() and exits. The kernel then sends
    // SIGTERM to the aligner the moment that thread exits, killing the
    // aligner mid-request (symptom: EPIPE on the first push_partial_pcm
    // because the aligner is already gone). The synth-worker sibling
    // doesn't hit this because it's forked from the main thread, which
    // is the process's thread-group leader and lives until shutdown.
    //
    // The tradeoff: if the parent process crashes without graceful
    // shutdown, the aligner becomes an orphan and stays alive until
    // explicit reaping. The graceful-shutdown path in server.cpp still
    // SIGKILLs the aligner via aligner_session->shutdown(), so crash-
    // free exits are clean. Catastrophic parent crashes will leak the
    // aligner subprocess (~700 MiB GPU). Acceptable to keep eager
    // spawn working — restore PDEATHSIG only if the eager spawn moves
    // to a long-lived dedicated thread.

    using aw_clk = std::chrono::steady_clock;
    const auto t_worker_start = aw_clk::now();
    auto t_since_start_ms = [&]() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   aw_clk::now() - t_worker_start).count();
    };

    fprintf(stderr, "aligner-worker[%d]: alive on fd=%d ppid=%d\n",
            (int) getpid(), fd, (int) getppid());
    fprintf(stderr, "  [aw-time worker-start  ] t+%lld ms\n",
            (long long) t_since_start_ms());

    // ── Aligner-only defaults. The aligner sibling subprocess opt-outs of
    //    a couple of perf/VRAM trade-offs the synth worker keeps on. All
    //    three knobs are opt-out — set the corresponding env var to
    //    "0"/"1" to override before launching the parent. Synth
    //    subprocesses are unaffected; this only runs in the
    //    --worker-aligner dispatch path.
    //
    //    GGML_CUDA_DISABLE_GRAPHS=1 — drops ~48 MiB of cudaGraph-capture
    //      machinery state (cg=2/0 in the probe shows no graphs actually
    //      capture under streaming-partial's per-call topology churn, so
    //      we keep the memory and ditch the would-be replay).
    //    CRISPASR_QWEN3_ASR_FUSED_QKV=0 — drops the 63 MiB fused-QKV
    //      buffer that duplicates the original attn_q/k/v.weight tensors
    //      already in model.buf. Re-fuse from a deduped load is the
    //      cleaner long-term fix (Phase B7); the env knob is the cheap
    //      win for now.
    //    CRISPASR_N_GPU_LAYERS=28 — keep all 28 Qwen3 LLM blk layers on
    //      GPU. Phase-C1 originally routed them to CPU to save VRAM
    //      (-210 MiB), but the streaming-aligned-TTS perf trade had
    //      gotten much worse than the original "+5 % wallclock" forecast
    //      once profiling landed: CPU body forward is ~200 ms/partial
    //      vs ~40-60 ms on GPU. Measured on RTX 3060: warm TTFP
    //      327 ms → 186 ms (-43 %), partial-7 align 381 ms → 120 ms
    //      (-69 %), aligner pid VRAM 556 MiB → 766 MiB (+210 MiB).
    //      Set the env to 0 to revert to the VRAM-minimal CPU body
    //      path if budget pressure ever comes back.
    setenv("GGML_CUDA_DISABLE_GRAPHS",       "1", /*overwrite=*/0);
    setenv("CRISPASR_QWEN3_ASR_FUSED_QKV",   "0", /*overwrite=*/0);
    setenv("CRISPASR_N_GPU_LAYERS",         "28", /*overwrite=*/0);

    // ── Phase-A VRAM probe (HANDOFF-fa-aligner-vram.md) ───────────────────
    // Gated by QWEN3_TTS_LOG_VRAM_PROBE=1. Reports per-pid GPU bytes from
    // nvidia-smi alongside a breakdown of every qwen3_asr-owned buffer, so
    // "other" = pid_used - known reveals the unaccounted slice (CUDA primary
    // context, cuBLAS workspace, cudaGraph capture cache).
    struct AwProbe {
        bool   enabled    = false;
        pid_t  pid        = 0;
        int    partial_ix = 0;
        int    last_log_n_pcm = 0;
        AwProbe() {
            const char * e = std::getenv("QWEN3_TTS_LOG_VRAM_PROBE");
            enabled = (e && *e && std::atoi(e) > 0);
            pid     = getpid();
        }
        static double mib(size_t b) { return (double) b / 1048576.0; }
        // Per-pid GPU bytes via nvidia-smi --query-compute-apps. Returns -1
        // on parse failure or if no entry is found (e.g. driver not present).
        // popen is expensive (~10-20 ms); only called when probe enabled.
        double pid_used_mib() const {
            FILE * fp = popen("nvidia-smi --query-compute-apps=pid,used_memory "
                              "--format=csv,noheader,nounits 2>/dev/null", "r");
            if (!fp) return -1;
            char line[256]; double mib_v = -1;
            while (fgets(line, sizeof(line), fp)) {
                int   p = 0; double m = 0;
                if (sscanf(line, "%d, %lf", &p, &m) == 2 && p == pid) {
                    mib_v = m; break;
                }
            }
            pclose(fp);
            return mib_v;
        }
        void cold(const char * label) {
            if (!enabled) return;
            double pidu = pid_used_mib();
            fprintf(stderr,
                    "  [aw-vram %-20s] pid_used=%7.1f MiB  "
                    "(no ctx — cold-process baseline)\n",
                    label, pidu);
        }
        void log(const char * label, qwen3_asr_context * ctx) {
            if (!enabled) return;
            double pidu = pid_used_mib();
            qwen3_asr_vram_breakdown b{}; int cg_n = 0; size_t cg_nodes = 0;
            size_t known = 0;
            if (ctx) {
                qwen3_asr_get_vram_breakdown(ctx, &b);
                void * be = qwen3_asr_get_gpu_backend_handle(ctx);
                if (be) {
                    ggml_backend_cuda_get_graph_cache_stats(
                        (ggml_backend_t) be, &cg_n, &cg_nodes);
                }
                known = b.model_buf_bytes + b.fused_buf_bytes
                      + b.kv_buf_bytes    + b.sched_buf_gpu_bytes
                      + b.audio_model_buf_bytes
                      + b.audio_conv_galloc_bytes
                      + b.audio_body_sched_gpu_bytes;
                // model_buf_cpu_bytes is host RAM — exclude from GPU sum.
            }
            const double other = (pidu >= 0) ? (pidu - mib(known)) : -1;
            fprintf(stderr,
                    "  [aw-vram %-20s] pid=%7.1f known=%7.1f other=%7.1f | "
                    "mod=%6.1f sched=%5.1f fused=%5.1f kv=%5.1f a_mod=%6.1f "
                    "a_conv=%4.1f a_sched=%5.1f cg=%d/%zu cpu_mod=%6.1f\n",
                    label, pidu, mib(known), other,
                    mib(b.model_buf_bytes),
                    mib(b.sched_buf_gpu_bytes),
                    mib(b.fused_buf_bytes),
                    mib(b.kv_buf_bytes),
                    mib(b.audio_model_buf_bytes),
                    mib(b.audio_conv_galloc_bytes),
                    mib(b.audio_body_sched_gpu_bytes),
                    cg_n, cg_nodes,
                    mib(b.model_buf_cpu_bytes));
        }
    };
    AwProbe probe;
    // A3: cold-process baseline BEFORE any GGUF load / ggml-backend init.
    probe.cold("aw-cold-baseline");

    json hello = {
        {"pid",  (int) getpid()},
        {"role", "qwen3-tts-aligner"},
    };
    if (send_frame(fd, WorkerFrame::HELLO, 0, hello.dump()) != IpcError::OK) {
        fprintf(stderr, "aligner-worker: HELLO send failed; bailing\n");
        return 2;
    }

    std::string         fa_model_path;
    qwen3_asr_context * fa_ctx = nullptr;
    // PCM accumulator (host-side). Sample rate is locked on first call
    // and asserted thereafter; reset=true clears.
    std::vector<float>  acc_pcm;
    int                 acc_pcm_sr = 0;

    // Cache of the most recent successful PARTIAL response. When the
    // ALIGN_FINAL_REQ that follows carries no new PCM and references the
    // same audio_seen_ms, we short-circuit: emit this cached result as
    // FINAL_RESP without re-running encode + body forward. Same audio +
    // same word list = same alignment by construction. Saves ~360 ms of
    // CPU body forward on the longest prompt at the tail of every
    // request — most of C1's visible end-to-end latency cost.
    // (HANDOFF-fa-aligner-vram-2 step 2.)
    bool                       cached_partial_valid     = false;
    int64_t                    cached_partial_audio_ms  = -1;
    int                        cached_partial_pcm_n     = -1;
    std::vector<std::string>   cached_partial_words;
    std::vector<int64_t>       cached_partial_t0_ms;
    std::vector<int64_t>       cached_partial_t1_ms;

    // Shared lazy-load lambda — same logic as ALIGN_REQ's load block in
    // run_worker_loop, factored locally so a future refactor can extract
    // it cleanly. Sets err_msg on failure; mutates fa_ctx + t_load_ms.
    auto ensure_fa_loaded = [&](std::string & err_msg, int64_t & t_load_ms) {
        using clk = std::chrono::steady_clock;
        if (fa_ctx) { t_load_ms = 0; return; }
        if (fa_model_path.empty()) {
            err_msg = "aligner_model not configured (set --hf-repo-fa in parent)";
            return;
        }
        const bool probe_v = []() {
            const char* e = std::getenv("QWEN3_FA_PROFILE_VRAM");
            return e && *e && std::atoi(e) > 0;
        }();
        auto v_used = [&]() {
            size_t f=0,t=0; auto g=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            if (g) ggml_backend_dev_memory(g,&f,&t);
            return (double)(t-f)/1048576.0;
        };
        if (probe_v) fprintf(stderr, "  [fa-vram %-14s] gpu_used=%7.1f MiB\n",
                              "aw-pre-load", v_used());
        const auto t0 = clk::now();
        qwen3_asr_context_params p = qwen3_asr_context_default_params();
        fa_ctx = qwen3_asr_init_from_file(fa_model_path.c_str(), p);
        if (probe_v) fprintf(stderr, "  [fa-vram %-14s] gpu_used=%7.1f MiB\n",
                              "aw-post-load", v_used());
        if (!fa_ctx) {
            err_msg = std::string("qwen3_asr_init_from_file failed: ") + fa_model_path;
            t_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
            return;
        }
        const int H = qwen3_asr_lm_head_dim(fa_ctx);
        if (H <= 0 || H > 8192) {
            fprintf(stderr, "aligner-worker: lm_head_dim=%d not a forced-aligner GGUF\n", H);
            err_msg = "aligner GGUF appears to be ASR (lm_head too wide) or missing lm_head";
            qwen3_asr_free(fa_ctx);
            fa_ctx = nullptr;
        }
        t_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
        if (fa_ctx) {
            fprintf(stderr, "aligner-worker: aligner loaded (lm_head_dim=%d) in %lld ms\n",
                    qwen3_asr_lm_head_dim(fa_ctx), (long long) t_load_ms);
            fprintf(stderr, "  [aw-time fa-init       ] qwen3_asr_init_from_file=%lld ms (t+%lld since spawn)\n",
                    (long long) t_load_ms, (long long) t_since_start_ms());
            // A1 boundary 1: right after init.
            probe.log("aw-post-init", fa_ctx);
        }
    };

    // Shared handler for ALIGN_PARTIAL_REQ + ALIGN_FINAL_REQ. Same math,
    // different resp-frame tag.
    bool       first_req_seen = false;
    auto handle_align_streaming = [&](const FrameHeader & hdr,
                                      const std::vector<uint8_t> & payload,
                                      WorkerFrame resp_tag) -> int {
        if (!first_req_seen) {
            first_req_seen = true;
            fprintf(stderr, "  [aw-time first-req     ] t+%lld ms (parent spawn → first ALIGN_*_REQ)\n",
                    (long long) t_since_start_ms());
        }
        using clk = std::chrono::steady_clock;
        const auto t_start = clk::now();
        std::string err_msg;
        bool ok = false;
        std::string json_meta;
        std::vector<float> pcm_delta;
        std::vector<std::string> words;
        int64_t audio_seen_ms = 0;
        int pcm_sr = 0;
        bool reset = false;
        int64_t t_load_ms = 0, t_resample_ms = 0, t_align_ms = 0;
        std::vector<int64_t> out_t0_ms, out_t1_ms;
        std::vector<float> samples_16k;

        if (!unpack_audio_payload(payload, &json_meta, &pcm_delta)) {
            err_msg = "ALIGN_*_REQ: unpack_audio_payload failed";
        } else {
            try {
                json req = json::parse(json_meta);
                pcm_sr        = req.value("pcm_sample_rate", 0);
                audio_seen_ms = req.value("audio_seen_ms",
                                          req.value("audio_total_ms", (int64_t) 0));
                reset         = req.value("reset", false);
                if (req.contains("words") && req["words"].is_array()) {
                    for (const auto & w : req["words"]) {
                        if (w.is_string()) words.push_back(w.get<std::string>());
                    }
                }
            } catch (const std::exception & ex) {
                err_msg = std::string("ALIGN_*_REQ json parse: ") + ex.what();
            }
        }

        if (err_msg.empty() && words.empty()) err_msg = "ALIGN_*_REQ words list empty";
        if (err_msg.empty() && pcm_sr <= 0)   err_msg = "ALIGN_*_REQ pcm_sample_rate missing";

        if (err_msg.empty()) {
            if (reset) {
                acc_pcm.clear(); acc_pcm_sr = pcm_sr;
                cached_partial_valid = false; // a new paragraph invalidates the cache
            }
            if (acc_pcm_sr == 0) acc_pcm_sr = pcm_sr;
            if (acc_pcm_sr != pcm_sr) {
                err_msg = "ALIGN_*_REQ pcm_sample_rate changed mid-stream";
            } else if (!pcm_delta.empty()) {
                acc_pcm.insert(acc_pcm.end(), pcm_delta.begin(), pcm_delta.end());
            }
        }

        // Short-circuit a redundant FINAL_RESP: when the caller asks for
        // FINAL with the same audio extent + same word list as the last
        // PARTIAL it just received, the alignment is identical by
        // construction. Skip the ~360 ms encode + body forward and reuse
        // the cached words.
        const bool is_final_req = (resp_tag == WorkerFrame::ALIGN_FINAL_RESP);
        bool reused_cache = false;
        if (err_msg.empty() && is_final_req && cached_partial_valid &&
            pcm_delta.empty() &&
            cached_partial_audio_ms == audio_seen_ms &&
            cached_partial_pcm_n   == (int) acc_pcm.size() &&
            cached_partial_words   == words) {
            out_t0_ms = cached_partial_t0_ms;
            out_t1_ms = cached_partial_t1_ms;
            ok = true;
            reused_cache = true;
        }

        if (err_msg.empty() && !reused_cache) ensure_fa_loaded(err_msg, t_load_ms);

        if (err_msg.empty() && !reused_cache) {
            const auto t0 = clk::now();
            const int target_sr = 16000;
            if (acc_pcm_sr == target_sr) {
                samples_16k = acc_pcm;
            } else {
                const double ratio = (double) acc_pcm_sr / (double) target_sr;
                const int n_in  = (int) acc_pcm.size();
                const int n_out = (int) ((double) n_in / ratio);
                samples_16k.resize((size_t) n_out);
                for (int i = 0; i < n_out; i++) {
                    const double src = i * ratio;
                    const int idx0 = (int) src;
                    const int idx1 = idx0 + 1;
                    const double frac = src - idx0;
                    const float a = acc_pcm[idx0];
                    const float b = (idx1 < n_in) ? acc_pcm[idx1] : a;
                    samples_16k[i] = (float) ((1.0 - frac) * a + frac * b);
                }
            }
            t_resample_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                clk::now() - t0).count();
        }

        if (err_msg.empty() && !reused_cache) {
            out_t0_ms.assign(words.size(), 0);
            out_t1_ms.assign(words.size(), 0);
            std::vector<const char *> cstr_words;
            cstr_words.reserve(words.size());
            for (const auto & w : words) cstr_words.push_back(w.c_str());
            const auto t0 = clk::now();
            // Default ON: the streaming aligner caches the audio prefix's
            // K/V across partials, dropping per-partial cost from
            // O(N_enc + n_text) → O(Δ_audio + n_text). On a 32 s paragraph
            // partial-7 wallclock drops from ~400 ms → ~150 ms (CPU body
            // forward). Reset=true at a paragraph boundary clears the
            // streaming state; the function also auto-resets if the
            // word list or encoder length shrinks. See HANDOFF-fa-
            // aligner-vram-2.md acceptance criteria.
            //
            // Opt-out for A/B and bit-identity diffing:
            // QWEN3_FA_STREAMING_ALIGN=0 routes back through
            // qwen3_asr_align_words (one-shot per partial, full forward).
            const bool use_streaming = []() {
                const char * e = std::getenv("QWEN3_FA_STREAMING_ALIGN");
                if (!e || !*e) return true;
                return std::atoi(e) != 0;
            }();
            int rc;
            if (use_streaming) {
                rc = qwen3_asr_align_words_streaming(
                        fa_ctx,
                        samples_16k.data(), (int) samples_16k.size(),
                        cstr_words.data(),  (int) cstr_words.size(),
                        /*reset=*/reset,
                        out_t0_ms.data(), out_t1_ms.data());
            } else {
                rc = qwen3_asr_align_words(
                        fa_ctx,
                        samples_16k.data(), (int) samples_16k.size(),
                        cstr_words.data(),  (int) cstr_words.size(),
                        out_t0_ms.data(), out_t1_ms.data());
            }
            t_align_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             clk::now() - t0).count();
            if (rc != 0) {
                err_msg = std::string(use_streaming ? "qwen3_asr_align_words_streaming rc=" : "qwen3_asr_align_words rc=")
                          + std::to_string(rc);
            } else {
                ok = true;
            }
        }

        if (ok && !reused_cache) {
            // A1 boundaries 2-5 fold into one trace: every partial logs the
            // breakdown so we see encoder-shape growth, post-first-align, and
            // 10+ steady-state in a single trace.
            char lbl[32];
            snprintf(lbl, sizeof(lbl), "aw-partial-%02d",
                     ++probe.partial_ix);
            probe.log(lbl, fa_ctx);
        }
        // Refresh the cache after a successful PARTIAL (not FINAL — a
        // FINAL that ran the full pipeline doesn't update the cache
        // because there's no follow-up that could reuse it).
        if (ok && !reused_cache && resp_tag == WorkerFrame::ALIGN_PARTIAL_RESP) {
            cached_partial_valid    = true;
            cached_partial_audio_ms = audio_seen_ms;
            cached_partial_pcm_n    = (int) acc_pcm.size();
            cached_partial_words    = words;
            cached_partial_t0_ms    = out_t0_ms;
            cached_partial_t1_ms    = out_t1_ms;
        }
        if (err_msg.empty()) {
            // Per-pass timing breakdown — t_load_ms is non-zero only on the
            // first pass (cold load); subsequent passes split into resample
            // + align (which includes audio encode + body forward + argmax).
            const int64_t pass_wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                             clk::now() - t_start).count();
            const char * pass_tag = (resp_tag == WorkerFrame::ALIGN_PARTIAL_RESP)
                                        ? "partial" : "final";
            const char * reuse_note = reused_cache ? " [cached]" : "";
            fprintf(stderr,
                    "  [aw-time %s-%02d   ] wall=%4lld ms  load=%4lld  resample=%3lld  align=%4lld  "
                    "acc_pcm=%d samples@%d Hz  audio_seen=%lld ms%s\n",
                    pass_tag, probe.partial_ix,
                    (long long) pass_wall_ms, (long long) t_load_ms,
                    (long long) t_resample_ms, (long long) t_align_ms,
                    (int) acc_pcm.size(), acc_pcm_sr,
                    (long long) audio_seen_ms, reuse_note);
        }

        json resp;
        resp["ok"]            = ok;
        resp["error"]         = err_msg;
        resp["audio_seen_ms"] = audio_seen_ms;
        if (ok) {
            json wj = json::array();
            for (size_t i = 0; i < words.size(); i++) {
                wj.push_back({
                    {"word_index", (int) i},
                    {"text",       words[i]},
                    {"t0_ms",      out_t0_ms[i]},
                    {"t1_ms",      out_t1_ms[i]},
                });
            }
            resp["words"] = std::move(wj);
        }
        const int64_t t_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       clk::now() - t_start).count();
        resp["profile"] = {
            {"t_load_ms",       t_load_ms},
            {"t_resample_ms",   t_resample_ms},
            {"t_aligner_ms",    t_align_ms},
            {"t_total_ms",      t_total_ms},
            {"n_words",         (int) words.size()},
            {"n_samples_16k",   (int) samples_16k.size()},
            {"src_sample_rate", acc_pcm_sr},
            {"n_pcm_acc",       (int) acc_pcm.size()},
        };
        if (send_frame(fd, resp_tag, hdr.req_id, resp.dump()) != IpcError::OK) {
            fprintf(stderr, "aligner-worker: %s send failed\n",
                    resp_tag == WorkerFrame::ALIGN_PARTIAL_RESP ? "PARTIAL_RESP" : "FINAL_RESP");
            return 9;
        }
        return 0;
    };

    while (true) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        IpcError e = recv_frame(fd, &hdr, &payload);
        if (e == IpcError::EofClean) {
            fprintf(stderr, "aligner-worker: parent EOF, exiting cleanly\n");
            return 0;
        }
        if (e != IpcError::OK) {
            fprintf(stderr, "aligner-worker: recv_frame failed: %s\n", ipc_error_str(e));
            return 3;
        }

        switch (static_cast<WorkerFrame>(hdr.type)) {
            case WorkerFrame::SHUTDOWN: {
                fprintf(stderr, "aligner-worker: SHUTDOWN, exiting\n");
                return 0;
            }
            case WorkerFrame::PING: {
                send_frame(fd, WorkerFrame::PONG, hdr.req_id, payload);
                break;
            }
            case WorkerFrame::LOAD_REQ: {
                // We accept LOAD_REQ for protocol symmetry but only honour
                // `aligner_model`; talker/vocoder/spk_enc paths are ignored.
                //
                // Eager-load gate: when the parent sends LOAD_REQ with
                // `eager_load=true`, we synchronously run
                // qwen3_asr_init_from_file here so the FA GGUF + buffers
                // are warm by the time the first ALIGN_PARTIAL_REQ lands.
                // The parent does this on a background thread in
                // parallel with its own synth load so the cost overlaps
                // with synth's cold path. Without `eager_load=true` we
                // just stash the model path (original behaviour) and let
                // ensure_fa_loaded fire lazily on the first
                // ALIGN_PARTIAL_REQ.
                std::string err_msg;
                bool ok = false;
                int64_t t_load_ms = 0;
                bool req_eager_load = false;
                try {
                    json req = json::parse(std::string(payload.begin(), payload.end()));
                    fa_model_path = req.value("aligner_model", std::string{});
                    req_eager_load = req.value("eager_load_aligner", false);
                    if (fa_model_path.empty()) {
                        err_msg = "aligner-worker: aligner_model empty in LOAD_REQ";
                    } else if (req_eager_load) {
                        ensure_fa_loaded(err_msg, t_load_ms);
                        ok = err_msg.empty();
                    } else {
                        ok = true;
                    }
                } catch (const std::exception & ex) {
                    err_msg = std::string("aligner-worker LOAD_REQ parse: ") + ex.what();
                }
                json resp = {
                    {"ok",             ok},
                    {"error",          err_msg},
                    {"sample_rate",    0},        // no vocoder
                    {"loaded_warmups", 0},
                    {"t_load_ms",      t_load_ms},
                };
                if (send_frame(fd, WorkerFrame::LOAD_RESP, hdr.req_id, resp.dump())
                    != IpcError::OK) {
                    fprintf(stderr, "aligner-worker: LOAD_RESP send failed\n");
                    return 4;
                }
                break;
            }
            case WorkerFrame::ALIGN_PARTIAL_REQ: {
                int rc = handle_align_streaming(hdr, payload, WorkerFrame::ALIGN_PARTIAL_RESP);
                if (rc != 0) return rc;
                break;
            }
            case WorkerFrame::ALIGN_FINAL_REQ: {
                int rc = handle_align_streaming(hdr, payload, WorkerFrame::ALIGN_FINAL_RESP);
                if (rc != 0) return rc;
                break;
            }
            default: {
                // SYNTH / voice-rego / vanilla ALIGN_REQ — refuse cleanly.
                json e2 = { {"error", "aligner-worker does not handle this frame type"} };
                send_frame(fd, WorkerFrame::SYNTH_ERR, hdr.req_id, e2.dump());
                fprintf(stderr, "aligner-worker: refused frame type=0x%x\n", hdr.type);
                break;
            }
        }
    }
#endif // QWEN3_TTS_HAS_FORCED_ALIGNMENT
}

} // namespace qwen3_tts
