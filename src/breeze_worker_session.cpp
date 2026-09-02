// breeze_worker_session.cpp — see breeze_worker_session.h.
//
// SPEECH_REQ wire contract, pack_codes_payload(json, int32 codes):
//   json = { "mode": "plain"|"stream",
//            "input": str,
//            "gp": {...},                       // gen_params, verbatim
//            "ref_T": int, "ref_text": str,     // present iff a clone voice
//            "chunk_frames": int }              // stream only
//   blob = the reference voice's codes [ref_T * n_codebooks], resolved by the
//          parent from its filesystem VoiceStore (the worker never touches it).

#include "breeze_worker_session.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <thread>

#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

using json = nlohmann::json;
using higgs::WFrame;
using higgs::FrameHeader;
using higgs::IpcError;
using higgs::send_frame;
using higgs::recv_frame;
using higgs::pack_audio_payload;
using higgs::pack_codes_payload;
using higgs::unpack_payload;
using higgs::ipc_error_str;
using higgs::spawn_worker;

namespace breeze {

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

static json gp_to_json(const gen_params & gp) {
    return json{
        {"temperature", gp.temperature}, {"top_k", gp.top_k}, {"top_p", gp.top_p},
        {"repetition_penalty", gp.repetition_penalty},
        {"depth_temperature", gp.depth_temperature}, {"depth_top_k", gp.depth_top_k},
        {"depth_top_p", gp.depth_top_p},
        {"max_new_frames", gp.max_new_frames}, {"seed", gp.seed},
        {"speaker", gp.speaker}, {"instruction", gp.instruction},
    };
}
static gen_params gp_from_json(const json & j) {
    gen_params gp;
    gp.temperature        = j.value("temperature", 0.9f);
    gp.top_k              = j.value("top_k", 50);
    gp.top_p              = j.value("top_p", 1.0f);
    gp.repetition_penalty = j.value("repetition_penalty", 1.1f);
    gp.depth_temperature  = j.value("depth_temperature", 0.9f);
    gp.depth_top_k        = j.value("depth_top_k", 50);
    gp.depth_top_p        = j.value("depth_top_p", 1.0f);
    gp.max_new_frames     = j.value("max_new_frames", 750);
    gp.seed               = j.value("seed", 0u);
    gp.speaker            = j.value("speaker", std::string{"S0"});
    gp.instruction        = j.value("instruction", std::string{});
    return gp;
}
static json meta_to_json(const gen_result & r, bool cancelled) {
    return json{{"T", r.T}, {"n_prompt_tokens", r.n_prompt_tokens},
                {"text_enc_ms", r.text_enc_ms}, {"prefill_ms", r.prefill_ms},
                {"decode_ms", r.decode_ms}, {"codec_ms", r.codec_ms},
                {"ttfa_ms", r.ttfa_ms}, {"cancelled", cancelled}};
}
static void meta_from_json(const json & j, gen_result & o) {
    o.T               = j.value("T", 0);
    o.n_prompt_tokens = j.value("n_prompt_tokens", 0);
    o.text_enc_ms     = j.value("text_enc_ms", 0.0);
    o.prefill_ms      = j.value("prefill_ms", 0.0);
    o.decode_ms       = j.value("decode_ms", 0.0);
    o.codec_ms        = j.value("codec_ms", 0.0);
    o.ttfa_ms         = j.value("ttfa_ms", 0.0);
}

// ───────────────────────────── parent side ─────────────────────────────

WorkerSession::WorkerSession(const char * argv0, std::vector<std::string> extra)
    : argv0_(argv0 ? argv0 : ""), extra_argv_(std::move(extra)) {}

WorkerSession::~WorkerSession() { shutdown(); }

void WorkerSession::kill_worker_locked() {
    if (pid_ > 0) {
        ::kill(pid_, SIGKILL);
        int wstat = 0;
        ::waitpid(pid_, &wstat, 0);
        fprintf(stderr, "breeze-session: killed worker pid=%d (wstat=0x%x)\n", (int) pid_, wstat);
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    pid_ = -1;
    loaded_ok_ = false;
    loaded_cfg_ = {};
}

void WorkerSession::shutdown() {
    std::lock_guard<std::mutex> lk(io_mutex_);
    kill_worker_locked();
}

void WorkerSession::request_cancel() {
    const uint32_t req_id = current_synth_req_id_.load(std::memory_order_acquire);
    if (req_id == 0) return;
    const int fd = fd_;
    if (fd < 0) return;
    std::lock_guard<std::mutex> slk(send_mutex_);
    send_frame(fd, WFrame::CANCEL_REQ, req_id, nullptr, 0);
}

bool WorkerSession::send_load_req_locked(const WorkerConfig & cfg) {
    json req = {{"model", cfg.model}, {"n_ctx", cfg.n_ctx}};
    if (send_frame(fd_, WFrame::LOAD_REQ, 0, req.dump()) != IpcError::OK) {
        last_error_ = "LOAD_REQ send failed"; return false;
    }
    FrameHeader hdr{}; std::vector<uint8_t> p;
    IpcError e = recv_frame(fd_, &hdr, &p);
    if (e != IpcError::OK) { last_error_ = std::string("LOAD_RESP recv: ") + ipc_error_str(e); return false; }
    if (hdr.type != (uint32_t) WFrame::LOAD_RESP) { last_error_ = "expected LOAD_RESP"; return false; }
    json resp;
    try { resp = json::parse(std::string(p.begin(), p.end())); }
    catch (const std::exception & ex) { last_error_ = std::string("LOAD_RESP parse: ") + ex.what(); return false; }
    if (!resp.value("ok", false)) {
        last_error_ = "worker load failed: " + resp.value("error", std::string{"(no msg)"});
        return false;
    }
    sample_rate_ = resp.value("sample_rate", 24000);
    return true;
}

bool WorkerSession::ensure_loaded(const WorkerConfig & cfg) {
    std::lock_guard<std::mutex> lk(io_mutex_);
    const std::string want_gpu = next_gpu_.empty() ? default_gpu_ : next_gpu_;
    if (pid_ > 0 && loaded_ok_ && worker_gpu_ == want_gpu &&
        loaded_cfg_.model == cfg.model && loaded_cfg_.n_ctx == cfg.n_ctx) {
        return true;
    }
    if (pid_ > 0) {
        if (worker_gpu_ != want_gpu)
            fprintf(stderr, "breeze-session: relocating worker '%s' -> '%s'\n",
                    worker_gpu_.c_str(), want_gpu.c_str());
        kill_worker_locked();
    }

    const int64_t t0 = now_ms();
    pid_t child = spawn_worker(argv0_.c_str(), extra_argv_, &fd_, "--breeze-worker", want_gpu);
    worker_gpu_ = want_gpu;
    if (child < 0) { last_error_ = "spawn_worker failed"; return false; }
    pid_ = child;

    FrameHeader hdr{}; std::vector<uint8_t> p;
    IpcError e = recv_frame(fd_, &hdr, &p);
    if (e != IpcError::OK || hdr.type != (uint32_t) WFrame::HELLO) {
        last_error_ = std::string("worker HELLO failed: ") + ipc_error_str(e);
        kill_worker_locked(); return false;
    }
    if (!send_load_req_locked(cfg)) {
        fprintf(stderr, "breeze-session: load failed: %s\n", last_error_.c_str());
        kill_worker_locked(); return false;
    }
    fprintf(stderr, "breeze-session: worker pid=%d loaded in %lld ms\n",
            (int) pid_, (long long) (now_ms() - t0));
    loaded_cfg_ = cfg;
    loaded_ok_  = true;
    return true;
}

bool WorkerSession::send_speech_locked(const std::string & meta_json,
                                       const std::vector<int32_t> & codes_blob,
                                       bool streaming, const BreezeTTS::pcm_cb & on_chunk,
                                       gen_result & out) {
    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) { last_error_ = "worker not ready"; return false; }
    auto payload = pack_codes_payload(meta_json, codes_blob.data(), codes_blob.size());
    const uint32_t req_id = next_req_id_.fetch_add(1);
    {
        std::lock_guard<std::mutex> slk(send_mutex_);
        if (send_frame(fd_, WFrame::SPEECH_REQ, req_id, payload) != IpcError::OK) {
            last_error_ = "SPEECH_REQ send failed"; kill_worker_locked(); return false;
        }
    }
    current_synth_req_id_.store(req_id, std::memory_order_release);
    struct Guard { std::atomic<uint32_t> & c; ~Guard(){ c.store(0, std::memory_order_release); } }
        guard{current_synth_req_id_};

    while (true) {
        FrameHeader hdr{}; std::vector<uint8_t> p;
        IpcError e = recv_frame(fd_, &hdr, &p);
        if (e != IpcError::OK) {
            last_error_ = std::string("speech recv: ") + ipc_error_str(e);
            kill_worker_locked(); return false;
        }
        switch (static_cast<WFrame>(hdr.type)) {
            case WFrame::AUDIO_FRAME: {
                std::string m; const uint8_t * blob = nullptr; size_t nb = 0;
                if (!unpack_payload(p, &m, &blob, &nb)) {
                    last_error_ = "AUDIO_FRAME unpack"; kill_worker_locked(); return false;
                }
                const int n = (int) (nb / sizeof(float));
                if (on_chunk && n > 0) on_chunk(reinterpret_cast<const float *>(blob), n, false);
                break;
            }
            case WFrame::SPEECH_RESP: {
                std::string m; const uint8_t * blob = nullptr; size_t nb = 0;
                if (!unpack_payload(p, &m, &blob, &nb)) {
                    last_error_ = "SPEECH_RESP unpack"; kill_worker_locked(); return false;
                }
                try { meta_from_json(json::parse(m), out); } catch (...) {}
                out.pcm.assign(reinterpret_cast<const float *>(blob),
                               reinterpret_cast<const float *>(blob) + nb / sizeof(float));
                return true;
            }
            case WFrame::SPEECH_DONE:
                try { meta_from_json(json::parse(std::string(p.begin(), p.end())), out); } catch (...) {}
                return true;
            case WFrame::SPEECH_ERR:
                try { last_error_ = json::parse(std::string(p.begin(), p.end()))
                                        .value("error", std::string{"worker error"}); }
                catch (...) { last_error_ = "worker error (unparseable)"; }
                return false;
            default:
                fprintf(stderr, "breeze-session: unexpected frame 0x%x\n", hdr.type);
                break;
        }
        (void) streaming;
    }
}

bool WorkerSession::synthesize(const std::string & text, const gen_params & gp,
                               const ref_voice * ref, gen_result & out) {
    std::lock_guard<std::mutex> lk(io_mutex_);
    json meta = {{"mode", "plain"}, {"input", text}, {"gp", gp_to_json(gp)}};
    std::vector<int32_t> blob;
    if (ref && ref->T > 0) {
        blob = ref->codes;
        meta["ref_T"] = ref->T;
        meta["ref_text"] = ref->ref_text;
    }
    return send_speech_locked(meta.dump(), blob, false, {}, out);
}

bool WorkerSession::synthesize_stream(const std::string & text, const gen_params & gp,
                                      const ref_voice * ref, int chunk_frames,
                                      const BreezeTTS::pcm_cb & on_chunk, gen_result & out) {
    std::lock_guard<std::mutex> lk(io_mutex_);
    json meta = {{"mode", "stream"}, {"input", text}, {"gp", gp_to_json(gp)},
                 {"chunk_frames", chunk_frames}};
    std::vector<int32_t> blob;
    if (ref && ref->T > 0) {
        blob = ref->codes;
        meta["ref_T"] = ref->T;
        meta["ref_text"] = ref->ref_text;
    }
    return send_speech_locked(meta.dump(), blob, true, on_chunk, out);
}

bool WorkerSession::encode_voice(const float * pcm, int n_samples,
                                 std::vector<int32_t> & out_codes, int & out_T) {
    std::lock_guard<std::mutex> lk(io_mutex_);
    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) { last_error_ = "worker not ready"; return false; }
    auto payload = pack_audio_payload(json({{"sample_rate", 24000}}).dump(), pcm, (size_t) n_samples);
    const uint32_t req_id = next_req_id_.fetch_add(1);
    {
        std::lock_guard<std::mutex> slk(send_mutex_);
        if (send_frame(fd_, WFrame::ENCODE_REQ, req_id, payload) != IpcError::OK) {
            last_error_ = "ENCODE_REQ send failed"; kill_worker_locked(); return false;
        }
    }
    FrameHeader hdr{}; std::vector<uint8_t> p;
    IpcError e = recv_frame(fd_, &hdr, &p);
    if (e != IpcError::OK) { last_error_ = std::string("ENCODE_RESP recv: ") + ipc_error_str(e); kill_worker_locked(); return false; }
    if (hdr.type != (uint32_t) WFrame::ENCODE_RESP) { last_error_ = "expected ENCODE_RESP"; kill_worker_locked(); return false; }
    std::string m; const uint8_t * blob = nullptr; size_t nb = 0;
    if (!unpack_payload(p, &m, &blob, &nb)) { last_error_ = "ENCODE_RESP unpack"; return false; }
    json j;
    try { j = json::parse(m); } catch (...) { last_error_ = "ENCODE_RESP meta parse"; return false; }
    if (!j.value("ok", false)) { last_error_ = j.value("error", std::string{"encode failed"}); return false; }
    out_T = j.value("T", 0);
    out_codes.assign(reinterpret_cast<const int32_t *>(blob),
                     reinterpret_cast<const int32_t *>(blob) + nb / sizeof(int32_t));
    return true;
}

// ───────────────────────────── child side ──────────────────────────────

int run_breeze_worker_loop(int fd) {
    setvbuf(stderr, nullptr, _IONBF, 0);
#if defined(__linux__)
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0)
        fprintf(stderr, "breeze-worker: prctl(PDEATHSIG) failed: %s (continuing)\n", strerror(errno));
#endif
    fprintf(stderr, "breeze-worker[%d]: alive on fd=%d ppid=%d\n", (int) getpid(), fd, (int) getppid());
    if (send_frame(fd, WFrame::HELLO, 0,
                   json({{"pid", (int) getpid()}, {"role", "breeze-worker"}}).dump()) != IpcError::OK) {
        fprintf(stderr, "breeze-worker: HELLO send failed\n"); return 2;
    }

    BreezeTTS tts;

    struct Ctrl {
        std::deque<std::pair<FrameHeader, std::vector<uint8_t>>> q;
        std::mutex m; std::condition_variable cv;
        std::atomic<bool> reader_done{false};
        int reader_exit = 0;
        std::atomic<uint32_t> active_req{0};
    } ctrl;

    std::thread reader([fd, &ctrl, &tts]() {
        while (true) {
            FrameHeader hdr{}; std::vector<uint8_t> payload;
            IpcError e = recv_frame(fd, &hdr, &payload);
            if (e == IpcError::EofClean) { ctrl.reader_done.store(true); ctrl.cv.notify_all(); return; }
            if (e != IpcError::OK) {
                fprintf(stderr, "breeze-worker-reader: recv failed: %s\n", ipc_error_str(e));
                ctrl.reader_exit = 3; ctrl.reader_done.store(true); ctrl.cv.notify_all(); return;
            }
            const WFrame ft = static_cast<WFrame>(hdr.type);
            if (ft == WFrame::CANCEL_REQ) {
                const uint32_t active = ctrl.active_req.load(std::memory_order_acquire);
                // req_id match so a stale cancel cannot kill the NEXT request.
                if (active != 0 && active == hdr.req_id) tts.request_cancel();
                continue;
            }
            if (ft == WFrame::SHUTDOWN) {
                std::lock_guard<std::mutex> lk(ctrl.m);
                ctrl.q.emplace_back(hdr, std::move(payload));
                ctrl.reader_done.store(true); ctrl.cv.notify_all(); return;
            }
            { std::lock_guard<std::mutex> lk(ctrl.m); ctrl.q.emplace_back(hdr, std::move(payload)); }
            ctrl.cv.notify_all();
        }
    });
    struct Join { std::thread & t; ~Join(){ if (t.joinable()) t.join(); } } join_guard{reader};

    while (true) {
        FrameHeader hdr{}; std::vector<uint8_t> payload;
        {
            std::unique_lock<std::mutex> lk(ctrl.m);
            ctrl.cv.wait(lk, [&]{ return !ctrl.q.empty() || ctrl.reader_done.load(); });
            if (ctrl.q.empty()) return ctrl.reader_exit;
            hdr = ctrl.q.front().first;
            payload = std::move(ctrl.q.front().second);
            ctrl.q.pop_front();
        }

        switch (static_cast<WFrame>(hdr.type)) {
            case WFrame::SHUTDOWN:
                fprintf(stderr, "breeze-worker: SHUTDOWN\n"); return 0;
            case WFrame::PING:
                send_frame(fd, WFrame::PONG, hdr.req_id, payload); break;

            case WFrame::LOAD_REQ: {
                bool ok = false; std::string err;
                int sr = 24000;
                try {
                    json req = json::parse(std::string(payload.begin(), payload.end()));
                    const std::string model = req.value("model", std::string{});
                    const int n_ctx = req.value("n_ctx", 2048);
                    fprintf(stderr, "breeze-worker: LOAD_REQ model=%s n_ctx=%d\n", model.c_str(), n_ctx);
                    ok = tts.load(model, n_ctx);
                    if (!ok) err = tts.get_error();
                    else { sr = tts.sample_rate(); tts.log_vram("worker-ready"); }
                } catch (const std::exception & ex) { err = std::string("LOAD_REQ parse: ") + ex.what(); }
                json resp = {{"ok", ok}, {"error", err}, {"sample_rate", sr}};
                if (send_frame(fd, WFrame::LOAD_RESP, hdr.req_id, resp.dump()) != IpcError::OK) return 4;
                break;
            }

            case WFrame::ENCODE_REQ: {
                std::string m; const uint8_t * blob = nullptr; size_t nb = 0;
                unpack_payload(payload, &m, &blob, &nb);
                std::vector<int32_t> codes; int T = 0;
                const bool ok = tts.encode_voice(reinterpret_cast<const float *>(blob),
                                                 (int) (nb / sizeof(float)), codes, T);
                json meta = {{"ok", ok}, {"error", ok ? std::string{} : tts.get_error()},
                             {"T", T}, {"N", tts.cfg().bb.n_codebooks}};
                send_frame(fd, WFrame::ENCODE_RESP, hdr.req_id,
                           pack_codes_payload(meta.dump(), ok ? codes.data() : nullptr,
                                              ok ? codes.size() : 0));
                break;
            }

            case WFrame::SPEECH_REQ: {
                std::string meta_str; const uint8_t * blob = nullptr; size_t nb = 0;
                if (!unpack_payload(payload, &meta_str, &blob, &nb)) {
                    send_frame(fd, WFrame::SPEECH_ERR, hdr.req_id,
                               json({{"error", "SPEECH_REQ unpack"}}).dump());
                    break;
                }
                json req;
                try { req = json::parse(meta_str); }
                catch (const std::exception & ex) {
                    send_frame(fd, WFrame::SPEECH_ERR, hdr.req_id,
                               json({{"error", std::string("SPEECH_REQ meta parse: ") + ex.what()}}).dump());
                    break;
                }
                const std::string mode = req.value("mode", std::string{"plain"});
                const std::string input = req.value("input", std::string{});
                const gen_params gp = gp_from_json(req.value("gp", json::object()));
                const bool streaming = (mode == "stream");

                ref_voice ref;
                if (req.contains("ref_T")) {
                    ref.T = req.value("ref_T", 0);
                    ref.ref_text = req.value("ref_text", std::string{});
                    ref.codes.assign(reinterpret_cast<const int32_t *>(blob),
                                     reinterpret_cast<const int32_t *>(blob) + nb / sizeof(int32_t));
                }

                tts.clear_cancel();
                ctrl.active_req.store(hdr.req_id, std::memory_order_release);
                struct AG { std::atomic<uint32_t> & a; ~AG(){ a.store(0, std::memory_order_release); } }
                    ag{ctrl.active_req};

                std::atomic<bool> ipc_ok{true};
                const uint32_t cb_req = hdr.req_id;
                BreezeTTS::pcm_cb on_chunk = [fd, cb_req, &ipc_ok](const float * pcm, int n, bool) {
                    if (!ipc_ok.load() || n <= 0) return;
                    if (send_frame(fd, WFrame::AUDIO_FRAME, cb_req,
                                   pack_audio_payload(std::string{}, pcm, (size_t) n)) != IpcError::OK)
                        ipc_ok.store(false);
                };

                gen_result r;
                bool ok;
                if (streaming) {
                    ok = tts.synthesize_stream(input, gp, ref.T ? &ref : nullptr,
                                               req.value("chunk_frames", 6), on_chunk, r);
                } else {
                    ok = tts.synthesize(input, gp, ref.T ? &ref : nullptr, r);
                }

                const bool cancelled = tts.cancelled();
                if (!ok && !cancelled) {
                    send_frame(fd, WFrame::SPEECH_ERR, hdr.req_id,
                               json({{"error", tts.get_error()}}).dump());
                    break;
                }
                if (streaming) {
                    send_frame(fd, WFrame::SPEECH_DONE, hdr.req_id, meta_to_json(r, cancelled).dump());
                } else {
                    send_frame(fd, WFrame::SPEECH_RESP, hdr.req_id,
                               pack_audio_payload(meta_to_json(r, cancelled).dump(),
                                                  r.pcm.data(), r.pcm.size()));
                }
                break;
            }

            default:
                fprintf(stderr, "breeze-worker: unexpected frame 0x%x\n", hdr.type);
                break;
        }
    }
}

} // namespace breeze
