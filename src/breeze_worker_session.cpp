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
//
// Several SPEECH_REQs may be in flight at once; every reply frame
// (AUDIO_FRAME / SPEECH_DONE / SPEECH_RESP / SPEECH_ERR / ENCODE_RESP) carries
// the req_id of the request it answers, and CANCEL_REQ names its target.

#include "breeze_worker_session.h"
#include "breeze_sched.h"

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
        {"cfg_scale", gp.cfg_scale},
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
    gp.cfg_scale          = j.value("cfg_scale", 1.0f);
    return gp;
}
static json meta_to_json(const gen_result & r, bool cancelled) {
    return json{{"T", r.T}, {"n_prompt_tokens", r.n_prompt_tokens},
                {"text_enc_ms", r.text_enc_ms}, {"prefill_ms", r.prefill_ms},
                {"decode_ms", r.decode_ms}, {"codec_ms", r.codec_ms},
                {"ttfa_ms", r.ttfa_ms}, {"truncated", r.truncated},
                {"cancelled", cancelled}};
}
static void meta_from_json(const json & j, gen_result & o) {
    o.T               = j.value("T", 0);
    o.n_prompt_tokens = j.value("n_prompt_tokens", 0);
    o.text_enc_ms     = j.value("text_enc_ms", 0.0);
    o.prefill_ms      = j.value("prefill_ms", 0.0);
    o.decode_ms       = j.value("decode_ms", 0.0);
    o.codec_ms        = j.value("codec_ms", 0.0);
    o.ttfa_ms         = j.value("ttfa_ms", 0.0);
    // Worker isolation runs the whole synth in the forked child, so anything the
    // caller needs must cross this boundary explicitly. Omitting `truncated`
    // meant the parent always reported "not truncated" -- the flag worked in
    // process and was silently dropped in the only configuration we deploy.
    o.truncated       = j.value("truncated", false);
}

// ───────────────────────────── parent side ─────────────────────────────

namespace { thread_local std::string tl_error; }

WorkerSession::WorkerSession(const char * argv0, std::vector<std::string> extra)
    : argv0_(argv0 ? argv0 : ""), extra_argv_(std::move(extra)) {}

WorkerSession::~WorkerSession() { shutdown(); }

const std::string & WorkerSession::get_error() const { return tl_error; }
void WorkerSession::set_error(const std::string & e) {
    tl_error = e;
    fprintf(stderr, "breeze-session: %s\n", e.c_str());
}

void WorkerSession::kill_worker_locked() {
    if (pid_ > 0) {
        ::kill(pid_, SIGKILL);
        int wstat = 0;
        ::waitpid(pid_, &wstat, 0);
        fprintf(stderr, "breeze-session: killed worker pid=%d (wstat=0x%x)\n", (int) pid_, wstat);
    }
    // The child is gone, so the reader's recv has hit EOF; join before the fd
    // is closed (closing under a blocked recv does not wake it on Linux).
    if (reader_.joinable()) reader_.join();
    {
        std::lock_guard<std::mutex> slk(send_mutex_);
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }
    pid_ = -1;
    loaded_ok_ = false;
    loaded_cfg_ = {};
    reader_dead_.store(false);
}

void WorkerSession::shutdown() {
    std::lock_guard<std::mutex> lk(io_mutex_);
    kill_worker_locked();
}

void WorkerSession::reader_loop(int fd) {
    for (;;) {
        FrameHeader hdr{}; std::vector<uint8_t> p;
        const IpcError e = recv_frame(fd, &hdr, &p);
        if (e != IpcError::OK) {
            if (e != IpcError::EofClean)
                fprintf(stderr, "breeze-session: reader: %s\n", ipc_error_str(e));
            break;
        }
        std::shared_ptr<Mailbox> mb;
        {
            std::lock_guard<std::mutex> lk(mb_mutex_);
            auto it = mailboxes_.find(hdr.req_id);
            if (it != mailboxes_.end()) mb = it->second;
        }
        if (!mb) continue;   // PONG, or a call that already gave up
        { std::lock_guard<std::mutex> lk(mb->m); mb->q.emplace_back(hdr, std::move(p)); }
        mb->cv.notify_one();
    }
    reader_dead_.store(true);
    std::lock_guard<std::mutex> lk(mb_mutex_);
    for (auto & kv : mailboxes_) {
        { std::lock_guard<std::mutex> l2(kv.second->m); kv.second->dead = true; }
        kv.second->cv.notify_all();
    }
}

void WorkerSession::request_cancel(CallHandle & h) {
    h.cancel.store(true, std::memory_order_relaxed);
    const uint32_t req_id = h.req_id.load(std::memory_order_acquire);
    if (req_id == 0) return;   // not sent yet; call() sends the cancel right after
    std::lock_guard<std::mutex> slk(send_mutex_);
    if (fd_ >= 0) send_frame(fd_, WFrame::CANCEL_REQ, req_id, nullptr, 0);
}

bool WorkerSession::send_load_req_locked(const WorkerConfig & cfg) {
    json req = {{"model", cfg.model}, {"n_ctx", cfg.n_ctx}};
    if (send_frame(fd_, WFrame::LOAD_REQ, 0, req.dump()) != IpcError::OK) {
        set_error("LOAD_REQ send failed"); return false;
    }
    FrameHeader hdr{}; std::vector<uint8_t> p;
    IpcError e = recv_frame(fd_, &hdr, &p);
    if (e != IpcError::OK) { set_error(std::string("LOAD_RESP recv: ") + ipc_error_str(e)); return false; }
    if (hdr.type != (uint32_t) WFrame::LOAD_RESP) { set_error("expected LOAD_RESP"); return false; }
    json resp;
    try { resp = json::parse(std::string(p.begin(), p.end())); }
    catch (const std::exception & ex) { set_error(std::string("LOAD_RESP parse: ") + ex.what()); return false; }
    if (!resp.value("ok", false)) {
        set_error("worker load failed: " + resp.value("error", std::string{"(no msg)"}));
        return false;
    }
    sample_rate_ = resp.value("sample_rate", 24000);
    return true;
}

bool WorkerSession::ensure_loaded(const WorkerConfig & cfg) {
    std::lock_guard<std::mutex> lk(io_mutex_);
    const std::string want_gpu = next_gpu_.empty() ? default_gpu_ : next_gpu_;
    const bool healthy = pid_ > 0 && loaded_ok_ && !reader_dead_.load();
    if (healthy && loaded_cfg_.model == cfg.model && loaded_cfg_.n_ctx == cfg.n_ctx) {
        if (worker_gpu_ == want_gpu) return true;
        // Relocating would SIGKILL the requests already rendering on it. The
        // new one joins them on the current card instead.
        if (active_calls_.load() > 0) {
            fprintf(stderr, "breeze-session: wanted '%s' but %d call(s) are rendering on '%s'; "
                            "staying\n", want_gpu.c_str(), active_calls_.load(), worker_gpu_.c_str());
            return true;
        }
    }
    if (pid_ > 0) {
        if (healthy && worker_gpu_ != want_gpu)
            fprintf(stderr, "breeze-session: relocating worker '%s' -> '%s'\n",
                    worker_gpu_.c_str(), want_gpu.c_str());
        kill_worker_locked();
    }

    const int64_t t0 = now_ms();
    pid_t child = spawn_worker(argv0_.c_str(), extra_argv_, &fd_, "--breeze-worker", want_gpu);
    worker_gpu_ = want_gpu;
    if (child < 0) { set_error("spawn_worker failed"); return false; }
    pid_ = child;

    FrameHeader hdr{}; std::vector<uint8_t> p;
    IpcError e = recv_frame(fd_, &hdr, &p);
    if (e != IpcError::OK || hdr.type != (uint32_t) WFrame::HELLO) {
        set_error(std::string("worker HELLO failed: ") + ipc_error_str(e));
        kill_worker_locked(); return false;
    }
    if (!send_load_req_locked(cfg)) {
        kill_worker_locked(); return false;
    }
    fprintf(stderr, "breeze-session: worker pid=%d loaded in %lld ms\n",
            (int) pid_, (long long) (now_ms() - t0));
    loaded_cfg_ = cfg;
    loaded_ok_  = true;
    reader_dead_.store(false);
    reader_ = std::thread([this, fd = fd_] { reader_loop(fd); });
    return true;
}

bool WorkerSession::call(WFrame type, const std::vector<uint8_t> & payload,
                         std::shared_ptr<Mailbox> & mb, uint32_t & req_id, CallHandle * h) {
    mb = std::make_shared<Mailbox>();
    req_id = next_req_id_.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(io_mutex_);
        if (pid_ <= 0 || fd_ < 0 || !loaded_ok_ || reader_dead_.load()) {
            set_error("worker not ready"); return false;
        }
        { std::lock_guard<std::mutex> ml(mb_mutex_); mailboxes_[req_id] = mb; }
        std::lock_guard<std::mutex> slk(send_mutex_);
        if (send_frame(fd_, type, req_id, payload) != IpcError::OK) {
            set_error("request send failed");
            drop_mailbox(req_id);
            return false;
        }
    }
    if (h) {
        h->req_id.store(req_id, std::memory_order_release);
        // A cancel that arrived before the id existed.
        if (h->cancel.load()) request_cancel(*h);
    }
    return true;
}

void WorkerSession::drop_mailbox(uint32_t req_id) {
    std::lock_guard<std::mutex> lk(mb_mutex_);
    mailboxes_.erase(req_id);
}

bool WorkerSession::send_speech(const std::string & meta_json,
                                const std::vector<int32_t> & codes_blob,
                                const BreezeTTS::pcm_cb & on_chunk, gen_result & out,
                                const BreezeTTS::chunk_cb & on_chunk_start, CallHandle * h) {
    struct Active { std::atomic<int> & n; Active(std::atomic<int> & a) : n(a) { n++; } ~Active() { n--; } }
        active{active_calls_};
    std::shared_ptr<Mailbox> mb; uint32_t req_id = 0;
    if (!call(WFrame::SPEECH_REQ, pack_codes_payload(meta_json, codes_blob.data(), codes_blob.size()),
              mb, req_id, h))
        return false;
    struct Drop { WorkerSession * s; uint32_t id; ~Drop() { s->drop_mailbox(id); } } drop{this, req_id};

    for (;;) {
        std::pair<FrameHeader, std::vector<uint8_t>> f;
        {
            std::unique_lock<std::mutex> lk(mb->m);
            mb->cv.wait(lk, [&] { return !mb->q.empty() || mb->dead; });
            if (mb->q.empty()) { set_error("worker died mid-request"); return false; }
            f = std::move(mb->q.front());
            mb->q.pop_front();
        }
        const FrameHeader & hdr = f.first;
        const std::vector<uint8_t> & p = f.second;
        switch (static_cast<WFrame>(hdr.type)) {
            case WFrame::AUDIO_FRAME: {
                std::string m; const uint8_t * blob = nullptr; size_t nb = 0;
                if (!unpack_payload(p, &m, &blob, &nb)) { set_error("AUDIO_FRAME unpack"); return false; }
                const int n = (int) (nb / sizeof(float));
                // A frame that opens a long-form chunk carries the chunk's
                // metadata; dispatched BEFORE its audio so the position the
                // caller reads is the one this chunk starts at.
                if (!m.empty() && on_chunk_start) {
                    try {
                        const json j = json::parse(m);
                        on_chunk_start((size_t) j.value("chunk", 0),
                                       (size_t) j.value("chunks_total", 0),
                                       j.value("chunk_text", std::string{}));
                    } catch (...) {}
                }
                if (on_chunk && n > 0) on_chunk(reinterpret_cast<const float *>(blob), n, false);
                break;
            }
            case WFrame::SPEECH_RESP: {
                std::string m; const uint8_t * blob = nullptr; size_t nb = 0;
                if (!unpack_payload(p, &m, &blob, &nb)) { set_error("SPEECH_RESP unpack"); return false; }
                try { meta_from_json(json::parse(m), out); } catch (...) {}
                out.pcm.assign(reinterpret_cast<const float *>(blob),
                               reinterpret_cast<const float *>(blob) + nb / sizeof(float));
                return true;
            }
            case WFrame::SPEECH_DONE:
                try { meta_from_json(json::parse(std::string(p.begin(), p.end())), out); } catch (...) {}
                return true;
            case WFrame::SPEECH_ERR:
                try { set_error(json::parse(std::string(p.begin(), p.end()))
                                    .value("error", std::string{"worker error"})); }
                catch (...) { set_error("worker error (unparseable)"); }
                return false;
            default:
                fprintf(stderr, "breeze-session: unexpected frame 0x%x\n", hdr.type);
                break;
        }
    }
}

bool WorkerSession::synthesize_long(const std::string & text, const gen_params & gp,
                                    const ref_voice * ref, int chunk_words, int ref_max_frames,
                                    int stream_chunk_frames, int gap_ms,
                                    const BreezeTTS::pcm_cb & on_chunk, gen_result & out,
                                    const BreezeTTS::chunk_cb & on_chunk_start,
                                    int first_chunk_words, CallHandle * h) {
    const bool streaming = stream_chunk_frames > 0 && on_chunk != nullptr;
    json meta = {{"mode", streaming ? "stream" : "plain"}, {"input", text},
                 {"gp", gp_to_json(gp)}, {"long", true},
                 {"chunk_words", chunk_words}, {"first_chunk_words", first_chunk_words},
                 {"long_ref_frames", ref_max_frames},
                 {"gap_ms", gap_ms},
                 {"chunk_frames", stream_chunk_frames > 0 ? stream_chunk_frames : 6}};
    std::vector<int32_t> blob;
    if (ref && ref->T > 0) {
        blob = ref->codes;
        meta["ref_T"] = ref->T;
        meta["ref_text"] = ref->ref_text;
    }
    return send_speech(meta.dump(), blob, on_chunk, out, on_chunk_start, h);
}

bool WorkerSession::synthesize(const std::string & text, const gen_params & gp,
                               const ref_voice * ref, gen_result & out, CallHandle * h) {
    json meta = {{"mode", "plain"}, {"input", text}, {"gp", gp_to_json(gp)}};
    std::vector<int32_t> blob;
    if (ref && ref->T > 0) {
        blob = ref->codes;
        meta["ref_T"] = ref->T;
        meta["ref_text"] = ref->ref_text;
    }
    return send_speech(meta.dump(), blob, {}, out, nullptr, h);
}

bool WorkerSession::synthesize_stream(const std::string & text, const gen_params & gp,
                                      const ref_voice * ref, int chunk_frames,
                                      const BreezeTTS::pcm_cb & on_chunk, gen_result & out,
                                      CallHandle * h) {
    json meta = {{"mode", "stream"}, {"input", text}, {"gp", gp_to_json(gp)},
                 {"chunk_frames", chunk_frames}};
    std::vector<int32_t> blob;
    if (ref && ref->T > 0) {
        blob = ref->codes;
        meta["ref_T"] = ref->T;
        meta["ref_text"] = ref->ref_text;
    }
    return send_speech(meta.dump(), blob, on_chunk, out, nullptr, h);
}

bool WorkerSession::encode_voice(const float * pcm, int n_samples,
                                 std::vector<int32_t> & out_codes, int & out_T) {
    struct Active { std::atomic<int> & n; Active(std::atomic<int> & a) : n(a) { n++; } ~Active() { n--; } }
        active{active_calls_};
    std::shared_ptr<Mailbox> mb; uint32_t req_id = 0;
    if (!call(WFrame::ENCODE_REQ,
              pack_audio_payload(json({{"sample_rate", 24000}}).dump(), pcm, (size_t) n_samples),
              mb, req_id, nullptr))
        return false;
    struct Drop { WorkerSession * s; uint32_t id; ~Drop() { s->drop_mailbox(id); } } drop{this, req_id};
    std::pair<FrameHeader, std::vector<uint8_t>> f;
    {
        std::unique_lock<std::mutex> lk(mb->m);
        mb->cv.wait(lk, [&] { return !mb->q.empty() || mb->dead; });
        if (mb->q.empty()) { set_error("worker died mid-encode"); return false; }
        f = std::move(mb->q.front());
    }
    if (f.first.type != (uint32_t) WFrame::ENCODE_RESP) { set_error("expected ENCODE_RESP"); return false; }
    std::string m; const uint8_t * blob = nullptr; size_t nb = 0;
    if (!unpack_payload(f.second, &m, &blob, &nb)) { set_error("ENCODE_RESP unpack"); return false; }
    json j;
    try { j = json::parse(m); } catch (...) { set_error("ENCODE_RESP meta parse"); return false; }
    if (!j.value("ok", false)) { set_error(j.value("error", std::string{"encode failed"})); return false; }
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

    // The scheduler thread and this one both write frames.
    std::mutex send_mtx;
    auto send = [&](WFrame t, uint32_t id, const std::vector<uint8_t> & p) {
        std::lock_guard<std::mutex> lk(send_mtx);
        return send_frame(fd, t, id, p);
    };
    auto send_s = [&](WFrame t, uint32_t id, const std::string & s) {
        return send(t, id, std::vector<uint8_t>(s.begin(), s.end()));
    };

    int max_streams = 2;
    if (const char * e = std::getenv("BREEZE_MAX_STREAMS")) max_streams = std::max(1, atoi(e));
    BreezeTTS tts;
    BreezeScheduler sched(tts, max_streams);
    fprintf(stderr, "breeze-worker: up to %d concurrent stream(s)\n", sched.max_lanes());

    struct Ctrl {
        std::deque<std::pair<FrameHeader, std::vector<uint8_t>>> q;
        std::mutex m; std::condition_variable cv;
        std::atomic<bool> reader_done{false};
        int reader_exit = 0;
        // Per-request cancel flags, created by the reader as the SPEECH_REQ
        // arrives so a CANCEL_REQ right behind it (handled before the request
        // is even dequeued) still lands.
        std::mutex fm;
        std::map<uint32_t, std::shared_ptr<std::atomic<bool>>> flags;
    } ctrl;

    std::thread reader([fd, &ctrl]() {
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
                std::lock_guard<std::mutex> lk(ctrl.fm);
                auto it = ctrl.flags.find(hdr.req_id);
                if (it != ctrl.flags.end()) it->second->store(true);   // stale ids are no-ops
                continue;
            }
            if (ft == WFrame::SPEECH_REQ) {
                std::lock_guard<std::mutex> lk(ctrl.fm);
                ctrl.flags[hdr.req_id] = std::make_shared<std::atomic<bool>>(false);
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
            if (ctrl.q.empty()) {
                // Parent gone: no one is reading the frames, so do not wait
                // for in-flight renders.
                _exit(ctrl.reader_exit);
            }
            hdr = ctrl.q.front().first;
            payload = std::move(ctrl.q.front().second);
            ctrl.q.pop_front();
        }

        switch (static_cast<WFrame>(hdr.type)) {
            case WFrame::SHUTDOWN:
                fprintf(stderr, "breeze-worker: SHUTDOWN\n"); _exit(0);
            case WFrame::PING:
                send(WFrame::PONG, hdr.req_id, payload); break;

            case WFrame::LOAD_REQ: {
                bool ok = false; std::string err;
                int sr = 24000;
                try {
                    json req = json::parse(std::string(payload.begin(), payload.end()));
                    const std::string model = req.value("model", std::string{});
                    const int n_ctx = req.value("n_ctx", 2048);
                    fprintf(stderr, "breeze-worker: LOAD_REQ model=%s n_ctx=%d\n", model.c_str(), n_ctx);
                    sched.run_sync([&] {
                        ok = tts.load(model, n_ctx);
                        if (!ok) err = tts.get_error();
                        else { sr = tts.sample_rate(); tts.log_vram("worker-ready"); }
                    });
                } catch (const std::exception & ex) { err = std::string("LOAD_REQ parse: ") + ex.what(); }
                json resp = {{"ok", ok}, {"error", err}, {"sample_rate", sr}};
                if (send_s(WFrame::LOAD_RESP, hdr.req_id, resp.dump()) != IpcError::OK) _exit(4);
                break;
            }

            case WFrame::ENCODE_REQ: {
                std::string m; const uint8_t * blob = nullptr; size_t nb = 0;
                unpack_payload(payload, &m, &blob, &nb);
                std::vector<int32_t> codes; int T = 0; bool ok = false; std::string err;
                sched.run_sync([&] {
                    ok = tts.encode_voice(reinterpret_cast<const float *>(blob),
                                          (int) (nb / sizeof(float)), codes, T);
                    if (!ok) err = tts.get_error();
                });
                json meta = {{"ok", ok}, {"error", err}, {"T", T}, {"N", tts.cfg().bb.n_codebooks}};
                send(WFrame::ENCODE_RESP, hdr.req_id,
                     pack_codes_payload(meta.dump(), ok ? codes.data() : nullptr, ok ? codes.size() : 0));
                break;
            }

            case WFrame::SPEECH_REQ: {
                const uint32_t req_id = hdr.req_id;
                std::shared_ptr<std::atomic<bool>> flag;
                {
                    std::lock_guard<std::mutex> lk(ctrl.fm);
                    auto & f = ctrl.flags[req_id];
                    if (!f) f = std::make_shared<std::atomic<bool>>(false);
                    flag = f;
                }
                auto forget = [&ctrl, req_id] {
                    std::lock_guard<std::mutex> lk(ctrl.fm);
                    ctrl.flags.erase(req_id);
                };
                std::string meta_str; const uint8_t * blob = nullptr; size_t nb = 0;
                if (!unpack_payload(payload, &meta_str, &blob, &nb)) {
                    send_s(WFrame::SPEECH_ERR, req_id, json({{"error", "SPEECH_REQ unpack"}}).dump());
                    forget(); break;
                }
                json req;
                try { req = json::parse(meta_str); }
                catch (const std::exception & ex) {
                    send_s(WFrame::SPEECH_ERR, req_id,
                           json({{"error", std::string("SPEECH_REQ meta parse: ") + ex.what()}}).dump());
                    forget(); break;
                }
                const std::string mode = req.value("mode", std::string{"plain"});
                const bool streaming = (mode == "stream");

                LaneRequest L;
                L.text = req.value("input", std::string{});
                L.gp   = gp_from_json(req.value("gp", json::object()));
                if (req.contains("ref_T")) {
                    L.ref.T = req.value("ref_T", 0);
                    L.ref.ref_text = req.value("ref_text", std::string{});
                    L.ref.codes.assign(reinterpret_cast<const int32_t *>(blob),
                                       reinterpret_cast<const int32_t *>(blob) + nb / sizeof(int32_t));
                }
                // Long-form runs INSIDE the worker: it feeds each chunk's
                // generated codes back in as the next chunk's reference, and
                // SPEECH_RESP only carries PCM.
                L.long_form         = req.value("long", false);
                L.chunk_words       = req.value("chunk_words", 120);
                L.ref_max_frames    = req.value("long_ref_frames", 250);
                L.gap_ms            = req.value("gap_ms", 180);
                L.first_chunk_words = req.value("first_chunk_words", 0);
                L.stream_chunk_frames = streaming ? req.value("chunk_frames", 6) : 0;
                L.cancel = flag;

                // A chunk boundary rides out on the frame that opens the chunk:
                // set by on_chunk_start, consumed by the next audio frame. Both
                // run on the scheduler thread.
                auto pending = std::make_shared<std::string>();
                auto ipc_ok  = std::make_shared<std::atomic<bool>>(true);
                if (streaming) {
                    L.on_chunk_start = [pending](size_t i, size_t n, const std::string & t) {
                        *pending = json{{"chunk", (int) i}, {"chunks_total", (int) n},
                                        {"chunk_text", t}}.dump();
                    };
                    L.on_pcm = [send, req_id, pending, ipc_ok](const float * pcm, int n, bool) {
                        if (!ipc_ok->load() || n <= 0) return;
                        std::string meta;
                        meta.swap(*pending);
                        if (send(WFrame::AUDIO_FRAME, req_id,
                                 pack_audio_payload(meta, pcm, (size_t) n)) != IpcError::OK)
                            ipc_ok->store(false);
                    };
                }
                L.on_done = [send, send_s, req_id, streaming, flag, forget](
                                bool ok, gen_result & r, const std::string & err) {
                    const bool cancelled = flag->load();
                    forget();
                    if (!ok && !cancelled) {
                        send_s(WFrame::SPEECH_ERR, req_id, json({{"error", err}}).dump());
                        return;
                    }
                    if (streaming)
                        send_s(WFrame::SPEECH_DONE, req_id, meta_to_json(r, cancelled).dump());
                    else
                        send(WFrame::SPEECH_RESP, req_id,
                             pack_audio_payload(meta_to_json(r, cancelled).dump(),
                                                r.pcm.data(), r.pcm.size()));
                };
                sched.submit(std::move(L));
                break;
            }

            default:
                fprintf(stderr, "breeze-worker: unexpected frame 0x%x\n", hdr.type);
                break;
        }
    }
}

} // namespace breeze
