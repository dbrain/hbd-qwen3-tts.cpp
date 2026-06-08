// higgs_fa_session.cpp — see header. Parent-side handle + aligner-only child
// dispatch loop. The alignment math lives entirely in qwen3_fa (shared,
// engine-agnostic), so word timings are bit-identical to the qwen3-tts aligner.

#include "higgs_fa_session.h"

#include "qwen3_fa/qwen3_asr.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::json;

namespace higgs {

// unpack a pack_audio_payload frame into (meta json, f32 PCM). higgs IPC only
// exposes the zero-copy blob view; copy it out as floats here.
static bool unpack_audio(const std::vector<uint8_t> & payload,
                         std::string * out_meta, std::vector<float> * out_pcm) {
    const uint8_t * blob = nullptr;
    size_t          nbytes = 0;
    if (!unpack_payload(payload, out_meta, &blob, &nbytes)) return false;
    const size_t n = nbytes / sizeof(float);
    out_pcm->resize(n);
    if (n && blob) std::memcpy(out_pcm->data(), blob, n * sizeof(float));
    return true;
}

// ───────────────────────── HiggsAlignerSession (parent) ─────────────────────

HiggsAlignerSession::HiggsAlignerSession(const char * argv0,
                                         std::vector<std::string> extra_argv)
    : argv0_(argv0 ? argv0 : ""), extra_argv_(std::move(extra_argv)) {}

HiggsAlignerSession::~HiggsAlignerSession() { shutdown(); }

void HiggsAlignerSession::kill_worker_locked() {
    if (pid_ > 0) {
        ::kill(pid_, SIGKILL);
        int wstat = 0;
        ::waitpid(pid_, &wstat, 0);
        fprintf(stderr, "higgs-aligner-session: killed aligner pid=%d (wstat=0x%x)\n",
                (int) pid_, wstat);
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    pid_ = -1;
    loaded_ok_ = false;
    loaded_model_.clear();
    stream_align_active_       = false;
    stream_align_has_sent_any_ = false;
}

void HiggsAlignerSession::shutdown() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    kill_worker_locked();
}

bool HiggsAlignerSession::ensure_loaded(const std::string & aligner_model) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (pid_ > 0 && loaded_ok_ && loaded_model_ == aligner_model) return true;
    if (aligner_model.empty()) {
        last_error_ = "aligner_model not configured";
        return false;
    }
    if (pid_ > 0) kill_worker_locked();

    pid_t child = spawn_worker(argv0_.c_str(), extra_argv_, &fd_, "--higgs-aligner");
    if (child < 0) { last_error_ = "spawn_worker (aligner) failed"; return false; }
    pid_ = child;

    // Expect HELLO before LOAD_REQ.
    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    IpcError e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK || hdr.type != static_cast<uint32_t>(WFrame::HELLO)) {
        last_error_ = std::string("aligner HELLO failed: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    fprintf(stderr, "higgs-aligner-session: HELLO (pid=%d): %.*s\n",
            (int) pid_, (int) payload.size(), (const char *) payload.data());

    json req = { {"aligner_model", aligner_model}, {"eager_load_aligner", true} };
    e = send_frame(fd_, WFrame::LOAD_REQ, 0, req.dump());
    if (e != IpcError::OK) {
        last_error_ = std::string("aligner LOAD_REQ send: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK || hdr.type != static_cast<uint32_t>(WFrame::LOAD_RESP)) {
        last_error_ = std::string("aligner LOAD_RESP recv: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    try {
        json resp = json::parse(std::string(payload.begin(), payload.end()));
        if (!resp.value("ok", false)) {
            last_error_ = std::string("aligner load failed: ")
                        + resp.value("error", std::string{"(no msg)"});
            kill_worker_locked();
            return false;
        }
    } catch (const std::exception & ex) {
        last_error_ = std::string("aligner LOAD_RESP parse: ") + ex.what();
        kill_worker_locked();
        return false;
    }
    loaded_model_ = aligner_model;
    loaded_ok_    = true;
    return true;
}

bool HiggsAlignerSession::begin_streaming_align(const std::vector<std::string> & words,
                                                int pcm_sample_rate) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (stream_align_active_) {
        last_error_ = "begin_streaming_align: previous stream not finalized";
        return false;
    }
    if (words.empty())          { last_error_ = "begin_streaming_align: empty word list"; return false; }
    if (pcm_sample_rate <= 0)   { last_error_ = "begin_streaming_align: bad pcm_sample_rate"; return false; }
    stream_align_words_        = words;
    stream_align_pcm_sr_       = pcm_sample_rate;
    stream_align_active_       = true;
    stream_align_has_sent_any_ = false;
    return true;
}

bool HiggsAlignerSession::push_partial_pcm(const float * pcm, size_t n_samples,
                                           int64_t audio_seen_ms) {
    if (!pcm && n_samples > 0) { last_error_ = "push_partial_pcm: null pcm"; return false; }
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (!stream_align_active_) { last_error_ = "push_partial_pcm: no active stream"; return false; }
    if (fd_ < 0)               { last_error_ = "push_partial_pcm: worker not running"; return false; }

    json meta = {
        {"words",           stream_align_words_},
        {"pcm_sample_rate", stream_align_pcm_sr_},
        {"audio_seen_ms",   audio_seen_ms},
        {"reset",           false},
    };
    if (!stream_align_has_sent_any_) {
        meta["reset"] = true;          // first push of a paragraph clears the accumulator
        stream_align_has_sent_any_ = true;
    }
    std::vector<uint8_t> payload = pack_audio_payload(meta.dump(), pcm, n_samples);
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame(fd_, WFrame::ALIGN_PARTIAL_REQ, req_id, payload);
    if (e != IpcError::OK) {
        last_error_ = std::string("ALIGN_PARTIAL_REQ send: ") + ipc_error_str(e);
        return false;
    }
    return true;
}

bool HiggsAlignerSession::drain_partial_alignments(const PartialAlignCallback & cb) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (fd_ < 0) return true;
    for (;;) {
        struct pollfd pfd { fd_, POLLIN, 0 };
        int pr = ::poll(&pfd, 1, 0);
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
        if (hdr.type != static_cast<uint32_t>(WFrame::ALIGN_PARTIAL_RESP)) continue;
        try {
            json r = json::parse(std::string(p.begin(), p.end()));
            if (!r.value("ok", false)) {
                last_error_ = r.value("error", std::string("PARTIAL_RESP error"));
                continue;
            }
            std::vector<AlignedWord> words;
            for (const auto & w : r["words"]) {
                AlignedWord aw;
                aw.text       = w.value("text",       std::string{});
                aw.t0_ms      = w.value("t0_ms",      (int64_t) 0);
                aw.t1_ms      = w.value("t1_ms",      (int64_t) 0);
                aw.confidence = w.value("confidence", -1.0f);
                words.push_back(std::move(aw));
            }
            const int64_t audio_seen_ms = r.value("audio_seen_ms", (int64_t) 0);
            io_mutex_.unlock();
            cb(audio_seen_ms, words);
            io_mutex_.lock();
        } catch (const std::exception & ex) {
            last_error_ = std::string("ALIGN_PARTIAL_RESP parse: ") + ex.what();
            return false;
        }
    }
}

bool HiggsAlignerSession::finalize_streaming_align(const float * tail_pcm,
                                                   size_t n_tail_samples,
                                                   int64_t audio_total_ms,
                                                   std::vector<AlignedWord> & out_words,
                                                   AlignProfile & out_profile) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    struct ResetOnExit {
        HiggsAlignerSession * s;
        ~ResetOnExit() {
            s->stream_align_active_       = false;
            s->stream_align_has_sent_any_ = false;
            s->stream_align_words_.clear();
            s->stream_align_pcm_sr_       = 0;
        }
    } reset_on_exit{this};
    if (!stream_align_active_) { last_error_ = "finalize_streaming_align: no active stream"; return false; }
    if (fd_ < 0)               { last_error_ = "finalize_streaming_align: worker not running"; return false; }

    // Drain stragglers so the accumulator is fully up-to-date before FINAL.
    for (;;) {
        struct pollfd pfd { fd_, POLLIN, 0 };
        int pr = ::poll(&pfd, 1, 0);
        if (pr <= 0) break;
        if (!(pfd.revents & POLLIN)) break;
        FrameHeader h{}; std::vector<uint8_t> p;
        if (recv_frame(fd_, &h, &p) != IpcError::OK) break;
    }

    json meta = {
        {"words",           stream_align_words_},
        {"pcm_sample_rate", stream_align_pcm_sr_},
        {"audio_total_ms",  audio_total_ms},
        {"reset",           false},
    };
    std::vector<uint8_t> payload = pack_audio_payload(meta.dump(), tail_pcm, n_tail_samples);
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame(fd_, WFrame::ALIGN_FINAL_REQ, req_id, payload);
    if (e != IpcError::OK) {
        last_error_ = std::string("ALIGN_FINAL_REQ send: ") + ipc_error_str(e);
        return false;
    }

    for (;;) {
        FrameHeader h{}; std::vector<uint8_t> p;
        IpcError re = recv_frame(fd_, &h, &p);
        if (re != IpcError::OK) {
            last_error_ = std::string("ALIGN_FINAL_RESP recv: ") + ipc_error_str(re);
            return false;
        }
        if (h.type == static_cast<uint32_t>(WFrame::ALIGN_PARTIAL_RESP)) continue; // drop stragglers
        if (h.type != static_cast<uint32_t>(WFrame::ALIGN_FINAL_RESP)) {
            last_error_ = std::string("expected ALIGN_FINAL_RESP, got 0x") + std::to_string(h.type);
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
                aw.text       = w.value("text",       std::string{});
                aw.t0_ms      = w.value("t0_ms",      (int64_t) 0);
                aw.t1_ms      = w.value("t1_ms",      (int64_t) 0);
                aw.confidence = w.value("confidence", -1.0f);
                out_words.push_back(std::move(aw));
            }
            if (r.contains("profile") && r["profile"].is_object()) {
                const auto & pf = r["profile"];
                out_profile.t_load_ms     = pf.value("t_load_ms",     (int64_t) 0);
                out_profile.t_resample_ms = pf.value("t_resample_ms", (int64_t) 0);
                out_profile.t_aligner_ms  = pf.value("t_aligner_ms",  (int64_t) 0);
                out_profile.t_total_ms    = pf.value("t_total_ms",    (int64_t) 0);
                out_profile.n_words       = pf.value("n_words",  0);
            }
        } catch (const std::exception & ex) {
            last_error_ = std::string("ALIGN_FINAL_RESP parse: ") + ex.what();
            return false;
        }
        break;
    }
    return true;
}

// ───────────────────── run_higgs_aligner_worker_loop (child) ────────────────

int run_higgs_aligner_worker_loop(int fd) {
    setvbuf(stderr, nullptr, _IONBF, 0);
    // NOTE: deliberately NOT calling prctl(PR_SET_PDEATHSIG) — see the qwen3-tts
    // aligner-worker comment: with the eager-spawn-from-a-thread pattern the
    // kernel would SIGTERM us when that thread exits. Graceful shutdown SIGKILLs
    // us via the session; only a catastrophic parent crash leaks the process.

    fprintf(stderr, "higgs-aligner[%d]: alive on fd=%d ppid=%d\n",
            (int) getpid(), fd, (int) getppid());

    // Aligner-only VRAM/perf defaults (all opt-out via env). The task wants the
    // aligner lean (≤~1.1 GB) so the LLM body stays on CPU by default; the
    // graphs/fused-QKV knobs trim a few tens of MiB more. Set the env to
    // override before launching the parent.
    setenv("GGML_CUDA_DISABLE_GRAPHS",     "1", /*overwrite=*/0);
    setenv("CRISPASR_QWEN3_ASR_FUSED_QKV", "0", /*overwrite=*/0);
    setenv("CRISPASR_N_GPU_LAYERS",        "0", /*overwrite=*/0); // LLM blk on CPU → lean VRAM

    json hello = { {"pid", (int) getpid()}, {"role", "higgs-aligner"} };
    if (send_frame(fd, WFrame::HELLO, 0, hello.dump()) != IpcError::OK) {
        fprintf(stderr, "higgs-aligner: HELLO send failed; bailing\n");
        return 2;
    }

    std::string         fa_model_path;
    qwen3_asr_context * fa_ctx = nullptr;
    std::vector<float>  acc_pcm;
    int                 acc_pcm_sr = 0;

    // Cache of the last successful PARTIAL so a redundant FINAL (no new PCM,
    // same audio extent + words) short-circuits the encode + body forward.
    bool                     cached_valid    = false;
    int64_t                  cached_audio_ms = -1;
    int                      cached_pcm_n    = -1;
    std::vector<std::string> cached_words;
    std::vector<int64_t>     cached_t0, cached_t1;
    std::vector<float>       cached_conf;

    auto ensure_fa_loaded = [&](std::string & err, int64_t & t_load_ms) {
        using clk = std::chrono::steady_clock;
        if (fa_ctx) { t_load_ms = 0; return; }
        if (fa_model_path.empty()) { err = "aligner_model not configured"; return; }
        const auto t0 = clk::now();
        qwen3_asr_context_params p = qwen3_asr_context_default_params();
        fa_ctx = qwen3_asr_init_from_file(fa_model_path.c_str(), p);
        t_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
        if (!fa_ctx) { err = std::string("qwen3_asr_init_from_file failed: ") + fa_model_path; return; }
        const int H = qwen3_asr_lm_head_dim(fa_ctx);
        if (H <= 0 || H > 8192) {
            fprintf(stderr, "higgs-aligner: lm_head_dim=%d not a forced-aligner GGUF\n", H);
            err = "aligner GGUF appears to be ASR (lm_head too wide)";
            qwen3_asr_free(fa_ctx); fa_ctx = nullptr;
            return;
        }
        fprintf(stderr, "higgs-aligner: aligner loaded (lm_head_dim=%d) in %lld ms\n",
                H, (long long) t_load_ms);
    };

    auto handle_align = [&](const FrameHeader & hdr,
                            const std::vector<uint8_t> & payload,
                            WFrame resp_tag) -> int {
        using clk = std::chrono::steady_clock;
        const auto t_start = clk::now();
        std::string err, json_meta;
        std::vector<float> pcm_delta, samples_16k;
        std::vector<std::string> words;
        int64_t audio_seen_ms = 0, t_load_ms = 0, t_resample_ms = 0, t_align_ms = 0;
        int pcm_sr = 0;
        bool reset = false, ok = false;
        std::vector<int64_t> out_t0, out_t1;
        std::vector<float>   out_conf;

        if (!unpack_audio(payload, &json_meta, &pcm_delta)) {
            err = "ALIGN_*_REQ: unpack_payload failed";
        } else {
            try {
                json req = json::parse(json_meta);
                pcm_sr        = req.value("pcm_sample_rate", 0);
                audio_seen_ms = req.value("audio_seen_ms", req.value("audio_total_ms", (int64_t) 0));
                reset         = req.value("reset", false);
                if (req.contains("words") && req["words"].is_array())
                    for (const auto & w : req["words"]) if (w.is_string()) words.push_back(w.get<std::string>());
            } catch (const std::exception & ex) {
                err = std::string("ALIGN_*_REQ json parse: ") + ex.what();
            }
        }
        if (err.empty() && words.empty()) err = "ALIGN_*_REQ words list empty";
        if (err.empty() && pcm_sr <= 0)   err = "ALIGN_*_REQ pcm_sample_rate missing";

        if (err.empty()) {
            if (reset) { acc_pcm.clear(); acc_pcm_sr = pcm_sr; cached_valid = false; }
            if (acc_pcm_sr == 0) acc_pcm_sr = pcm_sr;
            if (acc_pcm_sr != pcm_sr) err = "ALIGN_*_REQ pcm_sample_rate changed mid-stream";
            else if (!pcm_delta.empty()) acc_pcm.insert(acc_pcm.end(), pcm_delta.begin(), pcm_delta.end());
        }

        const bool is_final = (resp_tag == WFrame::ALIGN_FINAL_RESP);
        bool reused = false;
        if (err.empty() && is_final && cached_valid && pcm_delta.empty() &&
            cached_audio_ms == audio_seen_ms && cached_pcm_n == (int) acc_pcm.size() &&
            cached_words == words) {
            out_t0 = cached_t0; out_t1 = cached_t1; out_conf = cached_conf;
            ok = true; reused = true;
        }

        if (err.empty() && !reused) ensure_fa_loaded(err, t_load_ms);

        if (err.empty() && !reused) {
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
            t_resample_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
        }

        if (err.empty() && !reused) {
            out_t0.assign(words.size(), 0);
            out_t1.assign(words.size(), 0);
            out_conf.assign(words.size(), 0.0f);
            std::vector<const char *> cw; cw.reserve(words.size());
            for (const auto & w : words) cw.push_back(w.c_str());
            const auto t0 = clk::now();
            // Default: streaming aligner (KV-cached audio prefix); bit-identical
            // word timings with QWEN3_FA_STREAMING_KV=0 (default). Opt out via
            // QWEN3_FA_STREAMING_ALIGN=0 (one-shot full forward per partial).
            const bool use_streaming = []() {
                const char * e = std::getenv("QWEN3_FA_STREAMING_ALIGN");
                return (!e || !*e) ? true : (std::atoi(e) != 0);
            }();
            int rc = use_streaming
                ? qwen3_asr_align_words_streaming(fa_ctx, samples_16k.data(), (int) samples_16k.size(),
                                                  cw.data(), (int) cw.size(), /*reset=*/reset,
                                                  out_t0.data(), out_t1.data(), out_conf.data())
                : qwen3_asr_align_words(fa_ctx, samples_16k.data(), (int) samples_16k.size(),
                                        cw.data(), (int) cw.size(),
                                        out_t0.data(), out_t1.data(), out_conf.data());
            t_align_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
            if (rc != 0) err = std::string("qwen3_asr_align rc=") + std::to_string(rc);
            else ok = true;
        }

        if (ok && !reused && resp_tag == WFrame::ALIGN_PARTIAL_RESP) {
            cached_valid = true; cached_audio_ms = audio_seen_ms;
            cached_pcm_n = (int) acc_pcm.size(); cached_words = words;
            cached_t0 = out_t0; cached_t1 = out_t1; cached_conf = out_conf;
        }

        json resp;
        resp["ok"] = ok; resp["error"] = err; resp["audio_seen_ms"] = audio_seen_ms;
        if (ok) {
            json wj = json::array();
            for (size_t i = 0; i < words.size(); i++)
                wj.push_back({{"word_index",(int)i},{"text",words[i]},
                              {"t0_ms",out_t0[i]},{"t1_ms",out_t1[i]},{"confidence",out_conf[i]}});
            resp["words"] = std::move(wj);
        }
        const int64_t t_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       clk::now() - t_start).count();
        resp["profile"] = {{"t_load_ms",t_load_ms},{"t_resample_ms",t_resample_ms},
                           {"t_aligner_ms",t_align_ms},{"t_total_ms",t_total_ms},
                           {"n_words",(int)words.size()}};
        fprintf(stderr, "  [hi-aligner %s] wall=%lldms load=%lld resample=%lld align=%lld "
                "acc=%d@%dHz seen=%lldms%s\n",
                is_final ? "final  " : "partial",
                (long long) t_total_ms, (long long) t_load_ms, (long long) t_resample_ms,
                (long long) t_align_ms, (int) acc_pcm.size(), acc_pcm_sr,
                (long long) audio_seen_ms, reused ? " [cached]" : "");
        if (send_frame(fd, resp_tag, hdr.req_id, resp.dump()) != IpcError::OK) {
            fprintf(stderr, "higgs-aligner: %s send failed\n", is_final ? "FINAL_RESP" : "PARTIAL_RESP");
            return 9;
        }
        return 0;
    };

    while (true) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        IpcError e = recv_frame(fd, &hdr, &payload);
        if (e == IpcError::EofClean) { fprintf(stderr, "higgs-aligner: parent EOF, exiting\n"); return 0; }
        if (e != IpcError::OK) { fprintf(stderr, "higgs-aligner: recv_frame: %s\n", ipc_error_str(e)); return 3; }

        switch (static_cast<WFrame>(hdr.type)) {
            case WFrame::SHUTDOWN:
                fprintf(stderr, "higgs-aligner: SHUTDOWN\n");
                if (fa_ctx) qwen3_asr_free(fa_ctx);
                return 0;
            case WFrame::PING:
                send_frame(fd, WFrame::PONG, hdr.req_id, payload);
                break;
            case WFrame::LOAD_REQ: {
                std::string err; bool ok = false; int64_t t_load_ms = 0;
                try {
                    json req = json::parse(std::string(payload.begin(), payload.end()));
                    fa_model_path = req.value("aligner_model", std::string{});
                    bool eager = req.value("eager_load_aligner", false);
                    if (fa_model_path.empty()) err = "aligner_model empty in LOAD_REQ";
                    else if (eager) { ensure_fa_loaded(err, t_load_ms); ok = err.empty(); }
                    else ok = true;
                } catch (const std::exception & ex) {
                    err = std::string("aligner LOAD_REQ parse: ") + ex.what();
                }
                json resp = {{"ok",ok},{"error",err},{"sample_rate",0},{"t_load_ms",t_load_ms}};
                if (send_frame(fd, WFrame::LOAD_RESP, hdr.req_id, resp.dump()) != IpcError::OK) {
                    fprintf(stderr, "higgs-aligner: LOAD_RESP send failed\n"); return 4;
                }
                break;
            }
            case WFrame::ALIGN_PARTIAL_REQ: {
                int rc = handle_align(hdr, payload, WFrame::ALIGN_PARTIAL_RESP);
                if (rc != 0) return rc;
                break;
            }
            case WFrame::ALIGN_FINAL_REQ: {
                int rc = handle_align(hdr, payload, WFrame::ALIGN_FINAL_RESP);
                if (rc != 0) return rc;
                break;
            }
            default: {
                json e2 = {{"error","higgs-aligner does not handle this frame type"}};
                send_frame(fd, WFrame::SPEECH_ERR, hdr.req_id, e2.dump());
                fprintf(stderr, "higgs-aligner: refused frame type=0x%x\n", hdr.type);
                break;
            }
        }
    }
}

} // namespace higgs
