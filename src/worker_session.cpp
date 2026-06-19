// worker_session.cpp — see worker_session.h for design.

#include "worker_session.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <nlohmann/json.hpp>
#include <poll.h>
#include <sys/wait.h>
#include <thread>
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
                             std::vector<std::string> extra_argv)
    : argv0_(argv0 ? argv0 : ""), extra_argv_(std::move(extra_argv)) {}

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

void WorkerSession::cancel_in_flight() {
    // Snapshot req_id; if no synth is in flight, drop.
    uint32_t req_id = current_synth_req_id_.load(std::memory_order_acquire);
    if (req_id == 0) {
        return;
    }
    // Snapshot fd_ under a quick check — worker may have been killed
    // between our load above and the send below. send_mutex_ protects
    // the wire send from interleaving with other concurrent senders;
    // we deliberately do NOT take io_mutex_ here because the synth
    // call we're trying to cancel is the one holding it.
    int fd = fd_;
    if (fd < 0) {
        return;
    }
    std::lock_guard<std::mutex> slk(send_mutex_);
    IpcError e = send_frame(fd, WorkerFrame::CANCEL_REQ, req_id, nullptr, 0);
    if (e != IpcError::OK) {
        fprintf(stderr, "worker-session: CANCEL_REQ send failed (req_id=%u): %s\n",
                req_id, ipc_error_str(e));
        return;
    }
    fprintf(stderr, "worker-session: CANCEL_REQ sent (req_id=%u)\n", req_id);
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
    const std::string want_gpu = next_gpu_.empty() ? default_gpu_ : next_gpu_;
    if (pid_ > 0 && loaded_ok_
        && worker_gpu_ == want_gpu
        && loaded_cfg_.model == cfg.model
        && loaded_cfg_.vocoder == cfg.vocoder
        && loaded_cfg_.speaker_encoder == cfg.speaker_encoder
        && loaded_cfg_.aligner_model == cfg.aligner_model) {
        return true;
    }

    // Different config/GPU (or first load) → kill any existing worker, respawn.
    if (pid_ > 0) {
        if (worker_gpu_ != want_gpu)
            fprintf(stderr, "worker-session[synth]: relocating worker '%s' -> '%s'\n",
                    worker_gpu_.c_str(), want_gpu.c_str());
        kill_worker_locked();
    }

    const int64_t t_spawn_start = ggml_time_ms();
    pid_t child = spawn_worker(argv0_.c_str(), extra_argv_, &fd_, "--worker", want_gpu);
    worker_gpu_ = want_gpu;
    if (child < 0) {
        last_error_ = "spawn_worker failed";
        return false;
    }
    pid_ = child;
    fprintf(stderr, "worker-session[synth]: spawn_worker took %lld ms (pid=%d)\n",
            (long long)(ggml_time_ms() - t_spawn_start), (int)child);

    // Expect HELLO before LOAD_REQ.
    const int64_t t_hello_start = ggml_time_ms();
    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    IpcError e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK || hdr.type != static_cast<uint32_t>(WorkerFrame::HELLO)) {
        last_error_ = std::string("worker HELLO failed: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    fprintf(stderr, "worker-session[synth]: HELLO recv took %lld ms (pid=%d): %.*s\n",
            (long long)(ggml_time_ms() - t_hello_start),
            (int) pid_, (int) payload.size(), (const char *) payload.data());

    const int64_t t_load_req_start = ggml_time_ms();
    if (!send_load_req_locked(cfg)) {
        kill_worker_locked();
        return false;
    }
    fprintf(stderr, "worker-session[synth]: send_load_req+recv took %lld ms\n",
            (long long)(ggml_time_ms() - t_load_req_start));
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
    {
        std::lock_guard<std::mutex> slk(send_mutex_);
        IpcError e0 = send_frame(fd_, WorkerFrame::SYNTH_REQ, req_id, payload);
        if (e0 != IpcError::OK) {
            fail.error_msg = std::string("SYNTH_REQ send failed: ") + ipc_error_str(e0);
            kill_worker_locked();
            return fail;
        }
    }
    // Publish req_id AFTER the wire send so any concurrent
    // cancel_in_flight() that fires before the worker has even seen
    // SYNTH_REQ would be a no-op (current_synth_req_id_=0). Clear on
    // every return path so a late cancel after we already returned
    // doesn't get sent.
    current_synth_req_id_.store(req_id, std::memory_order_release);
    struct CurrentSynthGuard {
        std::atomic<uint32_t> & cur;
        ~CurrentSynthGuard() { cur.store(0, std::memory_order_release); }
    } current_synth_guard{current_synth_req_id_};
    IpcError e = IpcError::OK;

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
            aw.text       = w.value("text",       std::string{});
            aw.t0_ms      = w.value("t0_ms",      (int64_t) 0);
            aw.t1_ms      = w.value("t1_ms",      (int64_t) 0);
            aw.confidence = w.value("confidence", -1.0f);
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

// ───────────────────────── run_worker_loop (child) ─────────────────────

int run_worker_loop(int fd) {
    // Same unbuffered-stderr discipline as main() for crash handler logs.
    setvbuf(stderr, nullptr, _IONBF, 0);

    // TTFA decomposition timeline (parallels [aw-time] on the aligner side).
    const int64_t t_worker_start = ggml_time_ms();
    auto t_since_start = [t_worker_start]() {
        return (long long)(ggml_time_ms() - t_worker_start);
    };
    fprintf(stderr, "  [synth-time worker-start ] t+0 ms\n");

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

    // ─── Reader thread (control-frame multiplexer) ──────────────────────
    // The main thread spends most of its time in synth() and can't poll
    // the socket while doing so. A reader thread owns recv_frame()
    // exclusively: control frames (CANCEL_REQ, SHUTDOWN) are dispatched
    // immediately, everything else queues for the main thread.
    //
    // Sends stay on the main thread → no send_mutex needed. The reader
    // thread never writes to the socket; main thread never reads.
    struct WorkerCtrl {
        std::deque<std::pair<FrameHeader, std::vector<uint8_t>>> work_queue;
        std::mutex                queue_mutex;
        std::condition_variable   queue_cv;
        // Reader thread terminal states. main loop exits when reader_done
        // is true AND work_queue is drained.
        std::atomic<bool>     reader_done{false};       // EOF or SHUTDOWN seen
        int                   reader_exit_code = 0;     // 0 = clean
        // Cancel coordination. Main thread publishes the in-flight
        // synth's req_id BEFORE entering synth() and clears AFTER. The
        // reader compares incoming CANCEL_REQ.req_id against this; a
        // mismatch (or 0 = no synth in flight) is logged-and-dropped.
        // Release/acquire on active_synth_req_id pairs with main thread's
        // tts.clear_cancel() so a cancel reaching the reader after the
        // store is guaranteed to see the cleared flag.
        std::atomic<uint32_t> active_synth_req_id{0};
    };
    WorkerCtrl ctrl;

    std::thread reader_thread([fd, &ctrl, &tts]() {
        while (true) {
            FrameHeader hdr{};
            std::vector<uint8_t> payload;
            IpcError e = recv_frame(fd, &hdr, &payload);
            if (e == IpcError::EofClean) {
                fprintf(stderr, "worker-reader: parent EOF, exiting\n");
                ctrl.reader_done.store(true);
                ctrl.queue_cv.notify_all();
                return;
            }
            if (e != IpcError::OK) {
                fprintf(stderr, "worker-reader: recv_frame failed: %s\n",
                        ipc_error_str(e));
                ctrl.reader_exit_code = 3;
                ctrl.reader_done.store(true);
                ctrl.queue_cv.notify_all();
                return;
            }
            WorkerFrame ft = static_cast<WorkerFrame>(hdr.type);
            if (ft == WorkerFrame::CANCEL_REQ) {
                uint32_t active = ctrl.active_synth_req_id.load(std::memory_order_acquire);
                if (active != 0 && active == hdr.req_id) {
                    tts.request_cancel();
                    fprintf(stderr, "worker-reader: CANCEL_REQ req_id=%u accepted (active=%u)\n",
                            hdr.req_id, active);
                } else {
                    fprintf(stderr, "worker-reader: CANCEL_REQ req_id=%u dropped (active=%u)\n",
                            hdr.req_id, active);
                }
                continue;
            }
            if (ft == WorkerFrame::SHUTDOWN) {
                // SHUTDOWN goes through the queue so the main thread can
                // emit its log line and return cleanly without racing the
                // reader's exit. Reader marks done so main wakes if idle.
                std::lock_guard<std::mutex> lk(ctrl.queue_mutex);
                ctrl.work_queue.emplace_back(hdr, std::move(payload));
                ctrl.reader_done.store(true);
                ctrl.queue_cv.notify_all();
                return;
            }
            {
                std::lock_guard<std::mutex> lk(ctrl.queue_mutex);
                ctrl.work_queue.emplace_back(hdr, std::move(payload));
            }
            ctrl.queue_cv.notify_all();
        }
    });

    // Reader thread is joined on every return path via this guard.
    struct ReaderJoinGuard {
        std::thread & t;
        ~ReaderJoinGuard() { if (t.joinable()) t.join(); }
    } reader_join{reader_thread};

    while (true) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        {
            std::unique_lock<std::mutex> lk(ctrl.queue_mutex);
            ctrl.queue_cv.wait(lk, [&] {
                return !ctrl.work_queue.empty() || ctrl.reader_done.load();
            });
            if (ctrl.work_queue.empty()) {
                // Reader done + nothing left → exit.
                return ctrl.reader_exit_code;
            }
            hdr     = ctrl.work_queue.front().first;
            payload = std::move(ctrl.work_queue.front().second);
            ctrl.work_queue.pop_front();
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
                fprintf(stderr, "  [synth-time load-req-in   ] t+%lld ms\n", t_since_start());
                const int64_t t_load_in = ggml_time_ms();
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
                const int64_t t_after_models = ggml_time_ms();
                fprintf(stderr, "  [synth-time models-loaded ] t+%lld ms  (models %lld ms)\n",
                        t_since_start(), (long long)(t_after_models - t_load_in));

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
                fprintf(stderr, "  [synth-time warmup-scan   ] t+%lld ms  (scan %lld ms, %d blobs)\n",
                        t_since_start(),
                        (long long)(ggml_time_ms() - t_after_models),
                        loaded_warmups);

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
                fprintf(stderr, "  [synth-time synth-req-in  ] t+%lld ms\n", t_since_start());
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

                // Cancel coordination: clear stale cancel state from any
                // previous request BEFORE publishing this request's
                // req_id. The release-store on active_synth_req_id pairs
                // with the reader thread's acquire-load — a CANCEL_REQ
                // arriving after this store is guaranteed to see the
                // cleared flag, so it can flip request_cancel() without
                // racing our clear. RAII guard ensures we unpublish on
                // every return path (including from inside this case).
                tts.clear_cancel();
                ctrl.active_synth_req_id.store(hdr.req_id, std::memory_order_release);
                struct SynthActiveGuard {
                    std::atomic<uint32_t> & active;
                    ~SynthActiveGuard() { active.store(0, std::memory_order_release); }
                } synth_active_guard{ctrl.active_synth_req_id};

                // Reset the FA scratch buffer for each new synth so an
                // ALIGN_REQ after a failed/aborted synth can't accidentally
                // align against stale audio from two synths ago.
                last_pcm.clear();
                last_sr = 0;

                const bool streaming = (stream_batch_size > 0);
                streaming_opts sopts;
                std::atomic<bool> ipc_ok{ true };
                std::atomic<bool> first_audio_logged{ false };
                if (streaming) {
                    sopts.batch_size       = stream_batch_size;
                    sopts.first_batch_size = stream_first_batch_size;
                    const uint32_t cb_req_id = hdr.req_id;
                    sopts.on_pcm = [fd, cb_req_id, &ipc_ok, &last_pcm,
                                    &first_audio_logged, &t_since_start]
                                   (const float * pcm, size_t n) -> bool {
                        if (!first_audio_logged.exchange(true)) {
                            fprintf(stderr, "  [synth-time first-audio   ] t+%lld ms  (%zu samples in first batch)\n",
                                    t_since_start(), n);
                        }
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
                std::vector<float>   out_conf;
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
                    out_conf.assign(words.size(), 0.0f);
                    std::vector<const char *> cstr_words;
                    cstr_words.reserve(words.size());
                    for (const auto & w : words) cstr_words.push_back(w.c_str());

                    const auto t0 = clk::now();
                    const int rc = qwen3_asr_align_words(
                            fa_ctx,
                            samples_16k.data(), (int) samples_16k.size(),
                            cstr_words.data(),  (int) cstr_words.size(),
                            out_t0_ms.data(), out_t1_ms.data(),
                            out_conf.data());
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
                            {"text",       words[i]},
                            {"t0_ms",      out_t0_ms[i]},
                            {"t1_ms",      out_t1_ms[i]},
                            {"confidence", out_conf[i]},
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

} // namespace qwen3_tts
