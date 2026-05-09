// worker_session.cpp — see worker_session.h for design.

#include "worker_session.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <nlohmann/json.hpp>
#include <sys/wait.h>
#include <unistd.h>

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

bool WorkerSession::send_load_req_locked(const WorkerLoadConfig & cfg) {
    json req = {
        {"model",           cfg.model},
        {"vocoder",         cfg.vocoder},
        {"speaker_encoder", cfg.speaker_encoder},
        {"lazy_load",       cfg.lazy_load},
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

    // If config matches and worker is alive, we're good.
    if (pid_ > 0 && loaded_ok_
        && loaded_cfg_.model == cfg.model
        && loaded_cfg_.vocoder == cfg.vocoder
        && loaded_cfg_.speaker_encoder == cfg.speaker_encoder) {
        return true;
    }

    // Different config (or first load) → kill any existing worker, respawn.
    if (pid_ > 0) kill_worker_locked();

    pid_t child = spawn_worker(argv0_.c_str(), extra_argv_, &fd_);
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

// ───────────────────────── run_worker_loop (child) ─────────────────────

int run_worker_loop(int fd) {
    // Same unbuffered-stderr discipline as main() for crash handler logs.
    setvbuf(stderr, nullptr, _IONBF, 0);

    fprintf(stderr, "worker[%d]: alive on fd=%d\n", (int) getpid(), fd);

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
                try {
                    req = json::parse(std::string(payload.begin(), payload.end()));
                    std::string model      = req.value("model",           std::string{});
                    std::string vocoder    = req.value("vocoder",         std::string{});
                    std::string spk_enc    = req.value("speaker_encoder", std::string{});
                    bool        lazy_load  = req.value("lazy_load",       false);

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
                int sr = (ok && tts.is_loaded()) ? tts.get_sample_rate() : 0;
                json resp = {
                    {"ok",          ok},
                    {"error",       err_msg},
                    {"sample_rate", sr},
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

                const bool streaming = (stream_batch_size > 0);
                streaming_opts sopts;
                std::atomic<bool> ipc_ok{ true };
                if (streaming) {
                    sopts.batch_size       = stream_batch_size;
                    sopts.first_batch_size = stream_first_batch_size;
                    const uint32_t cb_req_id = hdr.req_id;
                    sopts.on_pcm = [fd, cb_req_id, &ipc_ok]
                                   (const float * pcm, size_t n) -> bool {
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
            default:
                fprintf(stderr, "worker: unexpected frame type=0x%x len=%u\n",
                        hdr.type, hdr.len);
                break;
        }
    }
}

} // namespace qwen3_tts
