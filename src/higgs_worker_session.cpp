// higgs_worker_session.cpp — see higgs_worker_session.h.
//
// Wire contract for SPEECH_REQ (pack_payload(json, codes_i32)):
//   json = { "mode": "plain|clone|long|multispeaker|stream",
//            "input": str,
//            "gp": {temperature,top_k,top_p,seed,max_new,ras_win_len,ras_max_repeat},
//            // clone:        "ref_T":int, "ref_text":str, "ref_cnt":int   (codes = blob[0..ref_cnt])
//            // long:         "buffer":int, "chunk_words":int
//            // multispeaker: "gap_ms":int, "rolling":bool,
//            //               "speakers":[{"key":str,"T":int,"ref_text":str,"off":int,"cnt":int}]
//            // stream:       "chunk_frames":int }
//   blob = concatenated int32 codes referenced by the descriptors (off/cnt/ref_cnt in int32 units).
//
// SPEECH_RESP / AUDIO_FRAME carry pack_audio_payload(meta_json, pcm_f32).
// SPEECH_DONE carries pure JSON meta. SPEECH_ERR carries {"error":str}.

#include "higgs_worker_session.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <thread>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

using json = nlohmann::json;

namespace higgs {

static constexpr int HIGGS_SAMPLE_RATE = 24000;

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

static json gp_to_json(const gen_params & gp) {
    return json{
        {"temperature",    gp.temperature},
        {"top_k",          gp.top_k},
        {"top_p",          gp.top_p},
        {"seed",           gp.seed},
        {"max_new",        gp.max_new},
        {"ras_win_len",    gp.ras_win_len},
        {"ras_max_repeat", gp.ras_max_repeat},
    };
}
static gen_params gp_from_json(const json & j) {
    gen_params gp;
    gp.temperature    = j.value("temperature", 0.7f);
    gp.top_k          = j.value("top_k", 50);
    gp.top_p          = j.value("top_p", 0.95f);
    gp.seed           = j.value("seed", 0u);
    gp.max_new        = j.value("max_new", 1024);
    gp.ras_win_len    = j.value("ras_win_len", 0);
    gp.ras_max_repeat = j.value("ras_max_repeat", 2);
    return gp;
}
static json result_meta_to_json(const gen_result & r, bool cancelled) {
    return json{
        {"T",          r.T},
        {"steps",      r.steps},
        {"prefill_ms", r.prefill_ms},
        {"decode_ms",  r.decode_ms},
        {"codec_ms",   r.codec_ms},
        {"cancelled",  cancelled},
    };
}
static void result_meta_from_json(const json & j, gen_result & out) {
    out.T          = j.value("T", 0);
    out.steps      = j.value("steps", 0);
    out.prefill_ms = j.value("prefill_ms", 0.0);
    out.decode_ms  = j.value("decode_ms", 0.0);
    out.codec_ms   = j.value("codec_ms", 0.0);
}

// ───────────────────────────── parent side ─────────────────────────────

HiggsWorkerSession::HiggsWorkerSession(const char * argv0, std::vector<std::string> extra_argv)
    : argv0_(argv0 ? argv0 : ""), extra_argv_(std::move(extra_argv)) {}

HiggsWorkerSession::~HiggsWorkerSession() { shutdown(); }

void HiggsWorkerSession::kill_worker_locked() {
    if (pid_ > 0) {
        ::kill(pid_, SIGKILL);
        int wstat = 0;
        ::waitpid(pid_, &wstat, 0);
        fprintf(stderr, "higgs-session: killed worker pid=%d (wstat=0x%x)\n", (int)pid_, wstat);
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    pid_ = -1;
    loaded_ok_ = false;
    loaded_cfg_ = {};
}

void HiggsWorkerSession::shutdown() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    kill_worker_locked();
}

void HiggsWorkerSession::request_cancel() {
    uint32_t req_id = current_synth_req_id_.load(std::memory_order_acquire);
    if (req_id == 0) return;
    int fd = fd_;
    if (fd < 0) return;
    std::lock_guard<std::mutex> slk(send_mutex_);
    IpcError e = send_frame(fd, WFrame::CANCEL_REQ, req_id, nullptr, 0);
    if (e != IpcError::OK)
        fprintf(stderr, "higgs-session: CANCEL_REQ send failed (req_id=%u): %s\n", req_id, ipc_error_str(e));
    else
        fprintf(stderr, "higgs-session: CANCEL_REQ sent (req_id=%u)\n", req_id);
}

bool HiggsWorkerSession::send_load_req_locked(const HiggsWorkerConfig & cfg) {
    json req = {{"backbone", cfg.backbone}, {"aux", cfg.aux}, {"n_ctx", cfg.n_ctx},
                {"kv", cfg.kv}, {"aux_enc", cfg.aux_enc}};
    if (send_frame(fd_, WFrame::LOAD_REQ, 0, req.dump()) != IpcError::OK) {
        last_error_ = "LOAD_REQ send failed"; return false;
    }
    FrameHeader hdr{}; std::vector<uint8_t> payload;
    IpcError e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK) { last_error_ = std::string("LOAD_RESP recv failed: ") + ipc_error_str(e); return false; }
    if (hdr.type != static_cast<uint32_t>(WFrame::LOAD_RESP)) { last_error_ = "expected LOAD_RESP"; return false; }
    json resp;
    try { resp = json::parse(std::string(payload.begin(), payload.end())); }
    catch (const std::exception & ex) { last_error_ = std::string("LOAD_RESP parse: ") + ex.what(); return false; }
    if (!resp.value("ok", false)) {
        last_error_ = std::string("worker load failed: ") + resp.value("error", std::string{"(no msg)"});
        return false;
    }
    sample_rate_ = resp.value("sample_rate", HIGGS_SAMPLE_RATE);
    return true;
}

bool HiggsWorkerSession::ensure_loaded(const HiggsWorkerConfig & cfg) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    const std::string want_gpu = next_gpu_.empty() ? default_gpu_ : next_gpu_;
    if (pid_ > 0 && loaded_ok_
        && worker_gpu_ == want_gpu
        && loaded_cfg_.backbone == cfg.backbone && loaded_cfg_.aux == cfg.aux
        && loaded_cfg_.n_ctx == cfg.n_ctx && loaded_cfg_.kv == cfg.kv
        && loaded_cfg_.aux_enc == cfg.aux_enc) {
        return true;
    }
    if (pid_ > 0) {
        if (worker_gpu_ != want_gpu)
            fprintf(stderr, "higgs-session: relocating worker '%s' -> '%s'\n",
                    worker_gpu_.c_str(), want_gpu.c_str());
        kill_worker_locked();
    }

    const int64_t t0 = now_ms();
    pid_t child = spawn_worker(argv0_.c_str(), extra_argv_, &fd_, "--higgs-worker", want_gpu);
    worker_gpu_ = want_gpu;
    if (child < 0) { last_error_ = "spawn_worker failed"; return false; }
    pid_ = child;

    FrameHeader hdr{}; std::vector<uint8_t> payload;
    IpcError e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK || hdr.type != static_cast<uint32_t>(WFrame::HELLO)) {
        last_error_ = std::string("worker HELLO failed: ") + ipc_error_str(e);
        kill_worker_locked(); return false;
    }
    fprintf(stderr, "higgs-session: worker pid=%d HELLO in %lld ms: %.*s\n",
            (int)pid_, (long long)(now_ms() - t0), (int)payload.size(), (const char *)payload.data());

    if (!send_load_req_locked(cfg)) {
        fprintf(stderr, "higgs-session: load failed: %s\n", last_error_.c_str());
        kill_worker_locked(); return false;
    }
    fprintf(stderr, "higgs-session: worker loaded in %lld ms total\n", (long long)(now_ms() - t0));
    loaded_cfg_ = cfg;
    loaded_ok_  = true;
    return true;
}

bool HiggsWorkerSession::send_speech_locked(const std::string & meta_json,
                                            const std::vector<int32_t> & codes_blob,
                                            bool streaming, const HiggsTTS::pcm_cb & on_chunk,
                                            gen_result & out) {
    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) { last_error_ = "worker not ready"; return false; }

    auto payload = pack_codes_payload(meta_json, codes_blob.data(), codes_blob.size());
    uint32_t req_id = next_req_id_.fetch_add(1);
    {
        std::lock_guard<std::mutex> slk(send_mutex_);
        if (send_frame(fd_, WFrame::SPEECH_REQ, req_id, payload) != IpcError::OK) {
            last_error_ = "SPEECH_REQ send failed"; kill_worker_locked(); return false;
        }
    }
    current_synth_req_id_.store(req_id, std::memory_order_release);
    struct Guard { std::atomic<uint32_t> & c; ~Guard(){ c.store(0, std::memory_order_release); } } guard{current_synth_req_id_};

    if (!streaming) {
        FrameHeader hdr{}; std::vector<uint8_t> p;
        IpcError e = recv_frame(fd_, &hdr, &p);
        if (e != IpcError::OK) { last_error_ = std::string("SPEECH_RESP recv: ") + ipc_error_str(e); kill_worker_locked(); return false; }
        if (hdr.type == static_cast<uint32_t>(WFrame::SPEECH_ERR)) {
            try { last_error_ = json::parse(std::string(p.begin(), p.end())).value("error", std::string{"worker error"}); }
            catch (...) { last_error_ = "worker error (unparseable)"; }
            return false;
        }
        if (hdr.type != static_cast<uint32_t>(WFrame::SPEECH_RESP)) { last_error_ = "expected SPEECH_RESP"; kill_worker_locked(); return false; }
        std::string meta; const uint8_t * blob = nullptr; size_t blob_bytes = 0;
        if (!unpack_payload(p, &meta, &blob, &blob_bytes)) { last_error_ = "SPEECH_RESP unpack"; kill_worker_locked(); return false; }
        try { result_meta_from_json(json::parse(meta), out); } catch (...) {}
        out.pcm.assign(reinterpret_cast<const float *>(blob),
                       reinterpret_cast<const float *>(blob) + blob_bytes / sizeof(float));
        return true;
    }

    // Streaming: AUDIO_FRAME* → on_chunk; SPEECH_DONE → return; SPEECH_ERR → fail.
    while (true) {
        FrameHeader hdr{}; std::vector<uint8_t> p;
        IpcError e = recv_frame(fd_, &hdr, &p);
        if (e != IpcError::OK) { last_error_ = std::string("AUDIO_FRAME recv: ") + ipc_error_str(e); kill_worker_locked(); return false; }
        if (hdr.type == static_cast<uint32_t>(WFrame::AUDIO_FRAME)) {
            std::string meta; const uint8_t * blob = nullptr; size_t blob_bytes = 0;
            if (!unpack_payload(p, &meta, &blob, &blob_bytes)) { last_error_ = "AUDIO_FRAME unpack"; kill_worker_locked(); return false; }
            int n = (int)(blob_bytes / sizeof(float));
            if (on_chunk && n > 0) on_chunk(reinterpret_cast<const float *>(blob), n, false);
            continue;
        }
        if (hdr.type == static_cast<uint32_t>(WFrame::SPEECH_DONE)) {
            try { result_meta_from_json(json::parse(std::string(p.begin(), p.end())), out); } catch (...) {}
            return true;
        }
        if (hdr.type == static_cast<uint32_t>(WFrame::SPEECH_ERR)) {
            try { last_error_ = json::parse(std::string(p.begin(), p.end())).value("error", std::string{"worker error"}); }
            catch (...) { last_error_ = "worker error (unparseable)"; }
            return false;
        }
        fprintf(stderr, "higgs-session: unexpected frame 0x%x during stream\n", hdr.type);
    }
}

bool HiggsWorkerSession::synthesize(const std::string & text, const gen_params & gp, gen_result & out) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    json meta = {{"mode", "plain"}, {"input", text}, {"gp", gp_to_json(gp)}};
    return send_speech_locked(meta.dump(), {}, false, {}, out);
}

bool HiggsWorkerSession::synthesize_with_ref(const std::string & text,
                                             const int32_t * ref_codes_TN, int ref_T,
                                             const std::string & ref_text,
                                             const gen_params & gp, gen_result & out, bool /*decode_audio*/) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (!ref_codes_TN || ref_T <= 0) {
        json meta = {{"mode", "plain"}, {"input", text}, {"gp", gp_to_json(gp)}};
        return send_speech_locked(meta.dump(), {}, false, {}, out);
    }
    const int N = 8;  // higgs n_codebooks (fixed); child re-derives N = ref_cnt/ref_T
    std::vector<int32_t> codes(ref_codes_TN, ref_codes_TN + (size_t)ref_T * N);
    json meta = {{"mode", "clone"}, {"input", text}, {"gp", gp_to_json(gp)},
                 {"ref_T", ref_T}, {"ref_text", ref_text}, {"ref_cnt", (int)codes.size()}};
    return send_speech_locked(meta.dump(), codes, false, {}, out);
}

bool HiggsWorkerSession::synthesize_long(const std::string & text, const gen_params & gp,
                                         int buffer, int chunk_words,
                                         const HiggsTTS::pcm_cb & /*on_chunk*/, gen_result & out) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    json meta = {{"mode", "long"}, {"input", text}, {"gp", gp_to_json(gp)},
                 {"buffer", buffer}, {"chunk_words", chunk_words}};
    return send_speech_locked(meta.dump(), {}, false, {}, out);
}

bool HiggsWorkerSession::synthesize_multispeaker(const std::string & text,
                                                 const std::map<std::string, HiggsTTS::named_voice> & voices,
                                                 const gen_params & gp, int gap_ms, bool rolling,
                                                 const HiggsTTS::pcm_cb & /*on_chunk*/, gen_result & out) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    std::vector<int32_t> blob;
    json speakers = json::array();
    for (const auto & kv : voices) {
        const auto & nv = kv.second;
        int off = (int)blob.size();
        blob.insert(blob.end(), nv.codes_TN.begin(), nv.codes_TN.end());
        speakers.push_back({{"key", kv.first}, {"T", nv.T}, {"ref_text", nv.ref_text},
                            {"off", off}, {"cnt", (int)nv.codes_TN.size()}});
    }
    json meta = {{"mode", "multispeaker"}, {"input", text}, {"gp", gp_to_json(gp)},
                 {"gap_ms", gap_ms}, {"rolling", rolling}, {"speakers", speakers}};
    return send_speech_locked(meta.dump(), blob, false, {}, out);
}

bool HiggsWorkerSession::synthesize_stream(const std::string & text, const gen_params & gp,
                                           int chunk_frames, const HiggsTTS::pcm_cb & on_chunk, gen_result & out) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    json meta = {{"mode", "stream"}, {"input", text}, {"gp", gp_to_json(gp)}, {"chunk_frames", chunk_frames}};
    return send_speech_locked(meta.dump(), {}, true, on_chunk, out);
}

bool HiggsWorkerSession::synthesize_stream_with_ref(const std::string & text,
                                                    const int32_t * ref_codes_TN, int ref_T,
                                                    const std::string & ref_text,
                                                    const gen_params & gp, int chunk_frames,
                                                    const HiggsTTS::pcm_cb & on_chunk, gen_result & out) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (!ref_codes_TN || ref_T <= 0) {
        json meta = {{"mode", "stream"}, {"input", text}, {"gp", gp_to_json(gp)}, {"chunk_frames", chunk_frames}};
        return send_speech_locked(meta.dump(), {}, true, on_chunk, out);
    }
    const int N = 8;  // higgs n_codebooks (fixed); child re-derives N = ref_cnt/ref_T
    std::vector<int32_t> codes(ref_codes_TN, ref_codes_TN + (size_t)ref_T * N);
    // mode=clone + stream=true → child runs synthesize_stream_with_ref (progressive
    // cloned-voice read-along). Same blob/meta as the buffered clone path + stream.
    json meta = {{"mode", "clone"}, {"stream", true}, {"input", text}, {"gp", gp_to_json(gp)},
                 {"chunk_frames", chunk_frames}, {"ref_T", ref_T}, {"ref_text", ref_text},
                 {"ref_cnt", (int)codes.size()}};
    return send_speech_locked(meta.dump(), codes, true, on_chunk, out);
}

bool HiggsWorkerSession::trial_codes(const std::string & text, const gen_params & gp,
                                     std::vector<int32_t> & out_codes, int & out_T, int & out_N) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) { last_error_ = "worker not ready"; return false; }
    json req = {{"input", text}, {"gp", gp_to_json(gp)}};
    uint32_t req_id = next_req_id_.fetch_add(1);
    {
        std::lock_guard<std::mutex> slk(send_mutex_);
        if (send_frame(fd_, WFrame::TRIAL_REQ, req_id, req.dump()) != IpcError::OK) {
            last_error_ = "TRIAL_REQ send failed"; kill_worker_locked(); return false;
        }
    }
    FrameHeader hdr{}; std::vector<uint8_t> p;
    IpcError e = recv_frame(fd_, &hdr, &p);
    if (e != IpcError::OK) { last_error_ = std::string("TRIAL_RESP recv: ") + ipc_error_str(e); kill_worker_locked(); return false; }
    if (hdr.type != static_cast<uint32_t>(WFrame::TRIAL_RESP)) { last_error_ = "expected TRIAL_RESP"; kill_worker_locked(); return false; }
    std::string meta; const uint8_t * blob = nullptr; size_t blob_bytes = 0;
    if (!unpack_payload(p, &meta, &blob, &blob_bytes)) { last_error_ = "TRIAL_RESP unpack"; kill_worker_locked(); return false; }
    json m;
    try { m = json::parse(meta); } catch (...) { last_error_ = "TRIAL_RESP meta parse"; return false; }
    if (!m.value("ok", false)) { last_error_ = m.value("error", std::string{"trial failed"}); return false; }
    out_T = m.value("T", 0);
    out_N = m.value("N", 8);
    out_codes.assign(reinterpret_cast<const int32_t *>(blob),
                     reinterpret_cast<const int32_t *>(blob) + blob_bytes / sizeof(int32_t));
    return true;
}

bool HiggsWorkerSession::encode_voice(const float * wav, int n_samples, int sr,
                                      std::vector<int32_t> & out_codes, int & out_T, int & out_N) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) { last_error_ = "worker not ready"; return false; }
    json meta = {{"sample_rate", sr}};
    auto payload = pack_audio_payload(meta.dump(), wav, (size_t)n_samples);
    uint32_t req_id = next_req_id_.fetch_add(1);
    {
        std::lock_guard<std::mutex> slk(send_mutex_);
        if (send_frame(fd_, WFrame::ENCODE_REQ, req_id, payload) != IpcError::OK) {
            last_error_ = "ENCODE_REQ send failed"; kill_worker_locked(); return false;
        }
    }
    FrameHeader hdr{}; std::vector<uint8_t> p;
    IpcError e = recv_frame(fd_, &hdr, &p);
    if (e != IpcError::OK) { last_error_ = std::string("ENCODE_RESP recv: ") + ipc_error_str(e); kill_worker_locked(); return false; }
    if (hdr.type != static_cast<uint32_t>(WFrame::ENCODE_RESP)) { last_error_ = "expected ENCODE_RESP"; kill_worker_locked(); return false; }
    std::string m; const uint8_t * blob = nullptr; size_t blob_bytes = 0;
    if (!unpack_payload(p, &m, &blob, &blob_bytes)) { last_error_ = "ENCODE_RESP unpack"; kill_worker_locked(); return false; }
    json j;
    try { j = json::parse(m); } catch (...) { last_error_ = "ENCODE_RESP meta parse"; return false; }
    if (!j.value("ok", false)) { last_error_ = j.value("error", std::string{"encode failed"}); return false; }
    out_T = j.value("T", 0);
    out_N = j.value("N", 8);
    out_codes.assign(reinterpret_cast<const int32_t *>(blob),
                     reinterpret_cast<const int32_t *>(blob) + blob_bytes / sizeof(int32_t));
    return true;
}

// ───────────────────────────── child side ──────────────────────────────

int run_higgs_worker_loop(int fd) {
    setvbuf(stderr, nullptr, _IONBF, 0);
#if defined(__linux__)
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0)
        fprintf(stderr, "higgs-worker: prctl(PR_SET_PDEATHSIG) failed: %s (continuing)\n", strerror(errno));
#endif
    fprintf(stderr, "higgs-worker[%d]: alive on fd=%d ppid=%d\n", (int)getpid(), fd, (int)getppid());

    json hello = {{"pid", (int)getpid()}, {"role", "higgs-worker"}};
    if (send_frame(fd, WFrame::HELLO, 0, hello.dump()) != IpcError::OK) {
        fprintf(stderr, "higgs-worker: HELLO send failed; bailing\n"); return 2;
    }

    HiggsTTS tts;

    // Reader thread multiplexes control frames (CANCEL_REQ / SHUTDOWN) so the
    // main thread can sit inside a synth call without polling the socket.
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
                fprintf(stderr, "higgs-worker-reader: recv failed: %s\n", ipc_error_str(e));
                ctrl.reader_exit = 3; ctrl.reader_done.store(true); ctrl.cv.notify_all(); return;
            }
            WFrame ft = static_cast<WFrame>(hdr.type);
            if (ft == WFrame::CANCEL_REQ) {
                uint32_t active = ctrl.active_req.load(std::memory_order_acquire);
                if (active != 0 && active == hdr.req_id) {
                    tts.request_cancel();
                    fprintf(stderr, "higgs-worker-reader: CANCEL_REQ req_id=%u accepted\n", hdr.req_id);
                } else {
                    fprintf(stderr, "higgs-worker-reader: CANCEL_REQ req_id=%u dropped (active=%u)\n", hdr.req_id, active);
                }
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
                fprintf(stderr, "higgs-worker: SHUTDOWN\n"); return 0;
            case WFrame::PING:
                send_frame(fd, WFrame::PONG, hdr.req_id, payload); break;

            case WFrame::LOAD_REQ: {
                bool ok = false; std::string err;
                try {
                    json req = json::parse(std::string(payload.begin(), payload.end()));
                    std::string backbone = req.value("backbone", std::string{});
                    std::string aux      = req.value("aux", std::string{});
                    int         n_ctx    = req.value("n_ctx", 8192);
                    std::string kv       = req.value("kv", std::string{"q8"});
                    std::string aux_enc  = req.value("aux_enc", std::string{});
                    fprintf(stderr, "higgs-worker: LOAD_REQ backbone=%s aux=%s n_ctx=%d kv=%s aux_enc=%s\n",
                            backbone.c_str(), aux.c_str(), n_ctx, kv.c_str(),
                            aux_enc.empty() ? "(none)" : aux_enc.c_str());
                    setenv("HIGGS_LM_KV", kv.c_str(), 1);
                    ok = tts.load(backbone, aux, n_ctx);
                    if (ok && !aux_enc.empty()) tts.set_aux_enc(aux_enc);
                    if (!ok) err = tts.get_error();
                    else { tts.lm().log_vram("worker-ready"); }
                } catch (const std::exception & ex) { err = std::string("LOAD_REQ parse: ") + ex.what(); }
                json resp = {{"ok", ok}, {"error", err}, {"sample_rate", HIGGS_SAMPLE_RATE}};
                if (send_frame(fd, WFrame::LOAD_RESP, hdr.req_id, resp.dump()) != IpcError::OK) {
                    fprintf(stderr, "higgs-worker: LOAD_RESP send failed\n"); return 4;
                }
                break;
            }

            case WFrame::TRIAL_REQ: {
                std::string text; gen_params gp;
                try {
                    json req = json::parse(std::string(payload.begin(), payload.end()));
                    text = req.value("input", std::string{});
                    gp = gp_from_json(req.value("gp", json::object()));
                } catch (const std::exception & ex) {
                    json e = {{"ok", false}, {"error", std::string("TRIAL_REQ parse: ") + ex.what()}};
                    send_frame(fd, WFrame::TRIAL_RESP, hdr.req_id, pack_codes_payload(e.dump(), nullptr, 0));
                    break;
                }
                tts.clear_cancel();
                gen_result r;
                bool ok = tts.synthesize(text, gp, r);
                if (ok && (r.codes_TN.empty() || r.T <= 0)) { ok = false; }
                int N = (ok && r.T > 0) ? (int)(r.codes_TN.size() / r.T) : 8;
                json meta = {{"ok", ok}, {"error", ok ? std::string{} : tts.get_error()},
                             {"T", r.T}, {"N", N}, {"seed", gp.seed}};
                auto buf = pack_codes_payload(meta.dump(), ok ? r.codes_TN.data() : nullptr,
                                              ok ? r.codes_TN.size() : 0);
                send_frame(fd, WFrame::TRIAL_RESP, hdr.req_id, buf);
                break;
            }

            case WFrame::ENCODE_REQ: {
                std::string meta_str; const uint8_t * blob = nullptr; size_t blob_bytes = 0;
                int sr = 24000;
                if (unpack_payload(payload, &meta_str, &blob, &blob_bytes)) {
                    try { sr = json::parse(meta_str).value("sample_rate", 24000); } catch (...) {}
                }
                const float * wav = reinterpret_cast<const float *>(blob);
                int n = (int)(blob_bytes / sizeof(float));
                std::vector<int32_t> codes; int T = 0, N = 8;
                bool ok = tts.encode_voice(wav, n, sr, codes, T, N);
                json meta = {{"ok", ok}, {"error", ok ? std::string{} : tts.get_error()}, {"T", T}, {"N", N}};
                auto buf = pack_codes_payload(meta.dump(), ok ? codes.data() : nullptr, ok ? codes.size() : 0);
                send_frame(fd, WFrame::ENCODE_RESP, hdr.req_id, buf);
                break;
            }

            case WFrame::SPEECH_REQ: {
                std::string meta_str; const uint8_t * blob = nullptr; size_t blob_bytes = 0;
                if (!unpack_payload(payload, &meta_str, &blob, &blob_bytes)) {
                    json e = {{"error", "SPEECH_REQ unpack"}};
                    send_frame(fd, WFrame::SPEECH_ERR, hdr.req_id, e.dump()); break;
                }
                const int32_t * codes = reinterpret_cast<const int32_t *>(blob);
                size_t n_codes = blob_bytes / sizeof(int32_t);

                json req;
                try { req = json::parse(meta_str); }
                catch (const std::exception & ex) {
                    json e = {{"error", std::string("SPEECH_REQ meta parse: ") + ex.what()}};
                    send_frame(fd, WFrame::SPEECH_ERR, hdr.req_id, e.dump()); break;
                }
                std::string mode  = req.value("mode", std::string{"plain"});
                std::string input = req.value("input", std::string{});
                gen_params gp = gp_from_json(req.value("gp", json::object()));
                // streaming = the zero-shot stream mode OR a clone with stream:true
                // (the read-along cloned-voice path) — both relay AUDIO_FRAMEs.
                const bool streaming = (mode == "stream") || req.value("stream", false);

                tts.clear_cancel();
                ctrl.active_req.store(hdr.req_id, std::memory_order_release);
                struct ActiveGuard { std::atomic<uint32_t> & a; ~ActiveGuard(){ a.store(0, std::memory_order_release); } } ag{ctrl.active_req};

                std::atomic<bool> ipc_ok{true};
                uint32_t cb_req = hdr.req_id;
                HiggsTTS::pcm_cb on_chunk = [fd, cb_req, &ipc_ok](const float * pcm, int n, bool) {
                    if (!ipc_ok.load()) return;
                    auto buf = pack_audio_payload(std::string{}, pcm, (size_t)n);
                    if (send_frame(fd, WFrame::AUDIO_FRAME, cb_req, buf) != IpcError::OK) ipc_ok.store(false);
                };

                gen_result r;
                bool ok = false;
                if (mode == "plain") {
                    ok = tts.synthesize(input, gp, r);
                } else if (mode == "clone") {
                    int ref_T = req.value("ref_T", 0);
                    std::string ref_text = req.value("ref_text", std::string{});
                    if (req.value("stream", false)) {
                        int chunk_frames = req.value("chunk_frames", 25);
                        ok = tts.synthesize_stream_with_ref(input, codes, ref_T, ref_text, gp, chunk_frames, on_chunk, r);
                    } else {
                        ok = tts.synthesize_with_ref(input, codes, ref_T, ref_text, gp, r, true);
                    }
                } else if (mode == "long") {
                    int buffer = req.value("buffer", 2);
                    int chunk_words = req.value("chunk_words", 100);
                    ok = tts.synthesize_long(input, gp, buffer, chunk_words, nullptr, r);
                } else if (mode == "multispeaker") {
                    int gap_ms = req.value("gap_ms", 250);
                    bool rolling = req.value("rolling", true);
                    std::map<std::string, HiggsTTS::named_voice> vmap;
                    for (const auto & s : req.value("speakers", json::array())) {
                        HiggsTTS::named_voice nv;
                        nv.T = s.value("T", 0);
                        nv.ref_text = s.value("ref_text", std::string{});
                        int off = s.value("off", 0), cnt = s.value("cnt", 0);
                        if (off >= 0 && cnt >= 0 && (size_t)(off + cnt) <= n_codes)
                            nv.codes_TN.assign(codes + off, codes + off + cnt);
                        vmap[s.value("key", std::string{})] = std::move(nv);
                    }
                    ok = tts.synthesize_multispeaker(input, vmap, gp, gap_ms, rolling, nullptr, r);
                } else if (mode == "stream") {
                    int chunk_frames = req.value("chunk_frames", 25);
                    ok = tts.synthesize_stream(input, gp, chunk_frames, on_chunk, r);
                } else {
                    json e = {{"error", "unknown mode: " + mode}};
                    send_frame(fd, WFrame::SPEECH_ERR, hdr.req_id, e.dump()); break;
                }

                const bool cancelled = tts.is_cancel_requested();
                if (!ok && !cancelled) {
                    json e = {{"error", tts.get_error()}};
                    send_frame(fd, WFrame::SPEECH_ERR, hdr.req_id, e.dump());
                    break;
                }
                // ok==false but cancelled → treat as a clean cancel (partial/empty audio).
                if (streaming) {
                    send_frame(fd, WFrame::SPEECH_DONE, hdr.req_id, result_meta_to_json(r, cancelled).dump());
                } else {
                    auto buf = pack_audio_payload(result_meta_to_json(r, cancelled).dump(), r.pcm.data(), r.pcm.size());
                    send_frame(fd, WFrame::SPEECH_RESP, hdr.req_id, buf);
                }
                break;
            }

            default:
                fprintf(stderr, "higgs-worker: unexpected frame 0x%x\n", hdr.type);
                break;
        }
    }
}

} // namespace higgs
