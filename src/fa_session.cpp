// fa_session.cpp — see fa_session.h. Self-contained aligner sibling shared by
// qwen3-tts-server and higgs-server. Three parts: (1) a small AF_UNIX framing
// transport in `namespace fa`, (2) the parent-side AlignerSession (with the
// bounded-in-flight backpressure that prevents the long-render deadlock), and
// (3) the aligner-only child dispatch loop. The alignment math is entirely in
// qwen3_fa (shared, engine-agnostic), so timings are identical across engines.

#include "fa_session.h"

#include "qwen3_fa/qwen3_asr.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::json;

namespace fa {

// ───────────────────────────── transport ────────────────────────────────────

const char * ipc_error_str(IpcError e) {
    switch (e) {
        case IpcError::OK:           return "ok";
        case IpcError::EofClean:     return "peer closed cleanly";
        case IpcError::EofMidFrame:  return "peer closed mid-frame";
        case IpcError::SocketError:  return "socket error";
        case IpcError::PayloadTooBig:return "payload too big";
    }
    return "unknown";
}

static IpcError read_exact(int fd, void * buf, size_t len) {
    char * p = static_cast<char *>(buf);
    size_t got = 0;
    while (got < len) {
        ssize_t r = ::read(fd, p + got, len - got);
        if (r > 0) { got += (size_t) r; continue; }
        if (r == 0) return got == 0 ? IpcError::EofClean : IpcError::EofMidFrame;
        if (errno == EINTR) continue;
        return IpcError::SocketError;
    }
    return IpcError::OK;
}

IpcError send_frame(int fd, Frame type, uint32_t req_id, const void * payload, size_t len) {
    if (len > MAX_FRAME_PAYLOAD) return IpcError::PayloadTooBig;
    FrameHeader hdr{ (uint32_t) type, (uint32_t) len, req_id };
    if (len == 0) {
        size_t sent = 0;
        while (sent < sizeof(hdr)) {
            ssize_t w = ::write(fd, (const char *) &hdr + sent, sizeof(hdr) - sent);
            if (w > 0) { sent += (size_t) w; continue; }
            if (w < 0 && errno == EINTR) continue;
            if (w < 0) return IpcError::SocketError;
        }
        return IpcError::OK;
    }
    iovec iov[2];
    iov[0].iov_base = &hdr;                       iov[0].iov_len = sizeof(hdr);
    iov[1].iov_base = const_cast<void *>(payload); iov[1].iov_len = len;
    size_t total = sizeof(hdr) + len, sent = 0;
    while (sent < total) {
        iovec * cur = iov; int n_iov = 2; size_t skip = sent;
        if (skip >= sizeof(hdr)) {
            cur = &iov[1]; n_iov = 1; skip -= sizeof(hdr);
            cur[0].iov_base = (char *) iov[1].iov_base + skip;
            cur[0].iov_len  = len - skip;
        } else if (skip > 0) {
            iov[0].iov_base = (char *) &hdr + skip;
            iov[0].iov_len  = sizeof(hdr) - skip;
        }
        ssize_t w = ::writev(fd, cur, n_iov);
        if (w > 0) { sent += (size_t) w; continue; }
        if (w < 0 && errno == EINTR) continue;
        if (w < 0) return IpcError::SocketError;
    }
    return IpcError::OK;
}

IpcError send_frame(int fd, Frame type, uint32_t req_id, const std::string & j) {
    return send_frame(fd, type, req_id, j.data(), j.size());
}

IpcError recv_frame(int fd, FrameHeader * out_hdr, std::vector<uint8_t> * out_payload) {
    if (!out_hdr) return IpcError::SocketError;
    IpcError e = read_exact(fd, out_hdr, sizeof(*out_hdr));
    if (e != IpcError::OK) return e;
    if (out_hdr->len > MAX_FRAME_PAYLOAD) return IpcError::PayloadTooBig;
    out_payload->resize(out_hdr->len);
    if (out_hdr->len == 0) return IpcError::OK;
    return read_exact(fd, out_payload->data(), out_hdr->len);
}

std::vector<uint8_t> pack_audio_payload(const std::string & meta, const float * samples, size_t n) {
    const size_t jlen = meta.size(), bytes = n * sizeof(float);
    std::vector<uint8_t> out(sizeof(uint32_t) + jlen + bytes);
    uint32_t j = (uint32_t) jlen;
    std::memcpy(out.data(), &j, sizeof(j));
    if (jlen)  std::memcpy(out.data() + sizeof(j), meta.data(), jlen);
    if (bytes) std::memcpy(out.data() + sizeof(j) + jlen, samples, bytes);
    return out;
}

bool unpack_audio_payload(const std::vector<uint8_t> & payload, std::string * out_meta,
                          std::vector<float> * out_pcm) {
    if (payload.size() < sizeof(uint32_t)) return false;
    uint32_t jlen = 0;
    std::memcpy(&jlen, payload.data(), sizeof(jlen));
    if (sizeof(jlen) + jlen > payload.size()) return false;
    if (out_meta) out_meta->assign((const char *) payload.data() + sizeof(jlen), jlen);
    size_t off = sizeof(jlen) + jlen, bytes = payload.size() - off;
    if (bytes % sizeof(float) != 0) return false;
    if (out_pcm) {
        out_pcm->resize(bytes / sizeof(float));
        if (bytes) std::memcpy(out_pcm->data(), payload.data() + off, bytes);
    }
    return true;
}

pid_t spawn_aligner(const char * self_argv0, const std::vector<std::string> & extra_argv,
                    int * out_parent_fd) {
    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        fprintf(stderr, "fa::spawn_aligner: socketpair failed: %s\n", strerror(errno));
        return -1;
    }
    int parent_fd = sv[0], child_fd = sv[1];
    int flags = ::fcntl(parent_fd, F_GETFD);
    if (flags >= 0) ::fcntl(parent_fd, F_SETFD, flags | FD_CLOEXEC);

    pid_t pid = ::fork();
    if (pid < 0) {
        fprintf(stderr, "fa::spawn_aligner: fork failed: %s\n", strerror(errno));
        ::close(sv[0]); ::close(sv[1]); return -1;
    }
    if (pid == 0) {
        ::close(parent_fd);
        char fd_buf[16]; std::snprintf(fd_buf, sizeof(fd_buf), "%d", child_fd);
        std::vector<std::string> owned;
        owned.emplace_back(self_argv0);
        owned.emplace_back("--fa-aligner");
        owned.emplace_back(fd_buf);
        for (auto & a : extra_argv) owned.push_back(a);
        std::vector<char *> argv_p; argv_p.reserve(owned.size() + 1);
        for (auto & s : owned) argv_p.push_back(s.data());
        argv_p.push_back(nullptr);
        ::execv(self_argv0, argv_p.data());
        std::fprintf(stderr, "fa::spawn_aligner child: execv(%s) failed: %s\n",
                     self_argv0, strerror(errno));
        ::_exit(127);
    }
    ::close(child_fd);
    if (out_parent_fd) *out_parent_fd = parent_fd;
    return pid;
}

// ───────────────────────── AlignerSession (parent) ──────────────────────────

AlignerSession::AlignerSession(const char * argv0, std::vector<std::string> extra_argv)
    : argv0_(argv0 ? argv0 : ""), extra_argv_(std::move(extra_argv)) {}

AlignerSession::~AlignerSession() { shutdown(); }

void AlignerSession::kill_worker_locked() {
    if (pid_ > 0) {
        ::kill(pid_, SIGKILL);
        int wstat = 0; ::waitpid(pid_, &wstat, 0);
        fprintf(stderr, "fa-aligner-session: killed aligner pid=%d (wstat=0x%x)\n", (int) pid_, wstat);
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    pid_ = -1;
    loaded_ok_ = false;
    loaded_model_.clear();
    stream_align_active_       = false;
    stream_align_has_sent_any_ = false;
    pending_pcm_.clear();
    pending_seen_ms_ = 0;
    inflight_        = 0;
}

void AlignerSession::shutdown() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    kill_worker_locked();
}

bool AlignerSession::ensure_loaded(const std::string & aligner_model) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (pid_ > 0 && loaded_ok_ && loaded_model_ == aligner_model) return true;
    if (aligner_model.empty()) { last_error_ = "aligner_model not configured"; return false; }
    if (pid_ > 0) kill_worker_locked();

    pid_t child = spawn_aligner(argv0_.c_str(), extra_argv_, &fd_);
    if (child < 0) { last_error_ = "spawn_aligner failed"; return false; }
    pid_ = child;

    FrameHeader hdr{}; std::vector<uint8_t> payload;
    IpcError e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK || hdr.type != (uint32_t) Frame::HELLO) {
        last_error_ = std::string("aligner HELLO failed: ") + ipc_error_str(e);
        kill_worker_locked(); return false;
    }
    fprintf(stderr, "fa-aligner-session: HELLO (pid=%d): %.*s\n",
            (int) pid_, (int) payload.size(), (const char *) payload.data());

    json req = { {"aligner_model", aligner_model}, {"eager_load_aligner", true} };
    e = send_frame(fd_, Frame::LOAD_REQ, 0, req.dump());
    if (e != IpcError::OK) {
        last_error_ = std::string("aligner LOAD_REQ send: ") + ipc_error_str(e);
        kill_worker_locked(); return false;
    }
    e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK || hdr.type != (uint32_t) Frame::LOAD_RESP) {
        last_error_ = std::string("aligner LOAD_RESP recv: ") + ipc_error_str(e);
        kill_worker_locked(); return false;
    }
    try {
        json resp = json::parse(std::string(payload.begin(), payload.end()));
        if (!resp.value("ok", false)) {
            last_error_ = std::string("aligner load failed: ") + resp.value("error", std::string{"(no msg)"});
            kill_worker_locked(); return false;
        }
    } catch (const std::exception & ex) {
        last_error_ = std::string("aligner LOAD_RESP parse: ") + ex.what();
        kill_worker_locked(); return false;
    }
    loaded_model_ = aligner_model;
    loaded_ok_    = true;
    return true;
}

bool AlignerSession::begin_streaming_align(const std::vector<std::string> & words, int pcm_sample_rate) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (stream_align_active_) { last_error_ = "begin_streaming_align: previous stream not finalized"; return false; }
    if (words.empty())        { last_error_ = "begin_streaming_align: empty word list"; return false; }
    if (pcm_sample_rate <= 0) { last_error_ = "begin_streaming_align: bad pcm_sample_rate"; return false; }
    stream_align_words_        = words;
    stream_align_pcm_sr_       = pcm_sample_rate;
    stream_align_active_       = true;
    stream_align_has_sent_any_ = false;
    pending_pcm_.clear();
    pending_seen_ms_           = 0;
    inflight_                  = 0;
    return true;
}

// Sends pending_pcm_ as one PARTIAL_REQ iff a slot is free. Caller holds
// io_mutex_. Returns false only on a hard send error; a full queue or empty
// buffer is a no-op success.
bool AlignerSession::flush_pending_locked() {
    if (fd_ < 0)                   { last_error_ = "flush_pending: worker not running"; return false; }
    if (pending_pcm_.empty())      return true;
    if (inflight_ >= kMaxInflight) return true;   // slot busy; coalesce further

    json meta = {
        {"words",           stream_align_words_},
        {"pcm_sample_rate", stream_align_pcm_sr_},
        {"audio_seen_ms",   pending_seen_ms_},
        {"reset",           !stream_align_has_sent_any_},  // first SEND clears the accumulator
    };
    std::vector<uint8_t> payload = pack_audio_payload(meta.dump(), pending_pcm_.data(), pending_pcm_.size());
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame(fd_, Frame::PARTIAL_REQ, req_id, payload.data(), payload.size());
    if (e != IpcError::OK) { last_error_ = std::string("PARTIAL_REQ send: ") + ipc_error_str(e); return false; }
    stream_align_has_sent_any_ = true;
    pending_pcm_.clear();
    inflight_++;
    return true;
}

bool AlignerSession::push_partial_pcm(const float * pcm, size_t n_samples, int64_t audio_seen_ms) {
    if (!pcm && n_samples > 0) { last_error_ = "push_partial_pcm: null pcm"; return false; }
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (!stream_align_active_) { last_error_ = "push_partial_pcm: no active stream"; return false; }
    if (fd_ < 0)               { last_error_ = "push_partial_pcm: worker not running"; return false; }
    // Coalesce into the pending buffer (never blocks the synth thread). Flushes
    // immediately if a slot is free; otherwise drain_partial_alignments() sends
    // it when the aligner frees a slot.
    pending_pcm_.insert(pending_pcm_.end(), pcm, pcm + n_samples);
    pending_seen_ms_ = audio_seen_ms;
    return flush_pending_locked();
}

bool AlignerSession::drain_partial_alignments(const PartialAlignCallback & cb) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (fd_ < 0) return true;
    for (;;) {
        struct pollfd pfd { fd_, POLLIN, 0 };
        int pr = ::poll(&pfd, 1, 0);
        // No more responses queued: a slot has likely freed → push coalesced audio.
        if (pr <= 0 || !(pfd.revents & POLLIN)) return flush_pending_locked();

        FrameHeader hdr{}; std::vector<uint8_t> p;
        IpcError e = recv_frame(fd_, &hdr, &p);
        if (e == IpcError::EofClean || e == IpcError::EofMidFrame) {
            last_error_ = "aligner worker EOF mid-stream"; return false;
        }
        if (e != IpcError::OK) { last_error_ = std::string("PARTIAL_RESP recv: ") + ipc_error_str(e); return false; }
        if (hdr.type != (uint32_t) Frame::PARTIAL_RESP) continue;
        if (inflight_ > 0) inflight_--;   // this partial's slot is now free
        try {
            json r = json::parse(std::string(p.begin(), p.end()));
            if (!r.value("ok", false)) { last_error_ = r.value("error", std::string("PARTIAL_RESP error")); continue; }
            std::vector<AlignedWord> words;
            for (const auto & w : r["words"]) {
                AlignedWord aw;
                aw.text       = w.value("text", std::string{});
                aw.t0_ms      = w.value("t0_ms", (int64_t) 0);
                aw.t1_ms      = w.value("t1_ms", (int64_t) 0);
                aw.confidence = w.value("confidence", -1.0f);
                words.push_back(std::move(aw));
            }
            const int64_t audio_seen_ms = r.value("audio_seen_ms", (int64_t) 0);
            io_mutex_.unlock();
            cb(audio_seen_ms, words);
            io_mutex_.lock();
        } catch (const std::exception & ex) {
            last_error_ = std::string("PARTIAL_RESP parse: ") + ex.what(); return false;
        }
    }
}

bool AlignerSession::finalize_streaming_align(const float * tail_pcm, size_t n_tail_samples,
                                              int64_t audio_total_ms, std::vector<AlignedWord> & out_words,
                                              AlignProfile & out_profile) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    struct ResetOnExit {
        AlignerSession * s;
        ~ResetOnExit() {
            s->stream_align_active_       = false;
            s->stream_align_has_sent_any_ = false;
            s->stream_align_words_.clear();
            s->stream_align_pcm_sr_       = 0;
            s->pending_pcm_.clear();
            s->pending_seen_ms_           = 0;
            s->inflight_                  = 0;
        }
    } reset_on_exit{this};
    if (!stream_align_active_) { last_error_ = "finalize_streaming_align: no active stream"; return false; }
    if (fd_ < 0)               { last_error_ = "finalize_streaming_align: worker not running"; return false; }

    // Drain stragglers so the accumulator is fully up-to-date before FINAL.
    for (;;) {
        struct pollfd pfd { fd_, POLLIN, 0 };
        int pr = ::poll(&pfd, 1, 0);
        if (pr <= 0 || !(pfd.revents & POLLIN)) break;
        FrameHeader h{}; std::vector<uint8_t> p;
        if (recv_frame(fd_, &h, &p) != IpcError::OK) break;
    }

    // Any audio coalesced but never sent (slot was busy at EOS) must reach the
    // aligner's accumulator now, or the final timings miss the tail.
    std::vector<float> final_tail = std::move(pending_pcm_);
    pending_pcm_.clear();
    if (tail_pcm && n_tail_samples > 0)
        final_tail.insert(final_tail.end(), tail_pcm, tail_pcm + n_tail_samples);

    json meta = {
        {"words",           stream_align_words_},
        {"pcm_sample_rate", stream_align_pcm_sr_},
        {"audio_total_ms",  audio_total_ms},
        {"reset",           !stream_align_has_sent_any_},
    };
    std::vector<uint8_t> payload = pack_audio_payload(meta.dump(), final_tail.data(), final_tail.size());
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame(fd_, Frame::FINAL_REQ, req_id, payload.data(), payload.size());
    if (e != IpcError::OK) { last_error_ = std::string("FINAL_REQ send: ") + ipc_error_str(e); return false; }

    for (;;) {
        FrameHeader h{}; std::vector<uint8_t> p;
        IpcError re = recv_frame(fd_, &h, &p);
        if (re != IpcError::OK) { last_error_ = std::string("FINAL_RESP recv: ") + ipc_error_str(re); return false; }
        if (h.type == (uint32_t) Frame::PARTIAL_RESP) continue; // drop stragglers
        if (h.type != (uint32_t) Frame::FINAL_RESP) {
            last_error_ = std::string("expected FINAL_RESP, got 0x") + std::to_string(h.type); return false;
        }
        try {
            json r = json::parse(std::string(p.begin(), p.end()));
            if (!r.value("ok", false)) { last_error_ = r.value("error", std::string("FINAL_RESP error")); return false; }
            out_words.clear();
            for (const auto & w : r["words"]) {
                AlignedWord aw;
                aw.text       = w.value("text", std::string{});
                aw.t0_ms      = w.value("t0_ms", (int64_t) 0);
                aw.t1_ms      = w.value("t1_ms", (int64_t) 0);
                aw.confidence = w.value("confidence", -1.0f);
                out_words.push_back(std::move(aw));
            }
            if (r.contains("profile") && r["profile"].is_object()) {
                const auto & pf = r["profile"];
                out_profile.t_load_ms     = pf.value("t_load_ms",     (int64_t) 0);
                out_profile.t_resample_ms = pf.value("t_resample_ms", (int64_t) 0);
                out_profile.t_aligner_ms  = pf.value("t_aligner_ms",  (int64_t) 0);
                out_profile.t_total_ms    = pf.value("t_total_ms",    (int64_t) 0);
                out_profile.n_words       = pf.value("n_words",       0);
            }
        } catch (const std::exception & ex) {
            last_error_ = std::string("FINAL_RESP parse: ") + ex.what(); return false;
        }
        break;
    }
    return true;
}

// ─────────────────────────── run_aligner_loop (child) ───────────────────────

int run_aligner_loop(int fd) {
    setvbuf(stderr, nullptr, _IONBF, 0);
    // No prctl(PR_SET_PDEATHSIG): the eager-spawn-from-a-thread pattern would
    // SIGTERM us when that thread exits. Graceful shutdown SIGKILLs us via the
    // session; only a catastrophic parent crash leaks the process.
    fprintf(stderr, "fa-aligner[%d]: alive on fd=%d ppid=%d\n", (int) getpid(), fd, (int) getppid());

    // Lean VRAM defaults (all opt-out via env): LLM body on CPU, graphs off,
    // drop the fused-QKV dup. Set the env before launching the parent to override.
    setenv("GGML_CUDA_DISABLE_GRAPHS",     "1", 0);
    setenv("CRISPASR_QWEN3_ASR_FUSED_QKV", "0", 0);
    setenv("CRISPASR_N_GPU_LAYERS",        "0", 0);

    json hello = { {"pid", (int) getpid()}, {"role", "fa-aligner"} };
    if (send_frame(fd, Frame::HELLO, 0, hello.dump()) != IpcError::OK) {
        fprintf(stderr, "fa-aligner: HELLO send failed; bailing\n");
        return 2;
    }

    std::string         fa_model_path;
    qwen3_asr_context * fa_ctx = nullptr;
    std::vector<float>  acc_pcm;
    int                 acc_pcm_sr = 0;

    // Cache of the last PARTIAL so a redundant FINAL (no new PCM, same extent +
    // words) short-circuits the encode + body forward.
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
            fprintf(stderr, "fa-aligner: lm_head_dim=%d not a forced-aligner GGUF\n", H);
            err = "aligner GGUF appears to be ASR (lm_head too wide)";
            qwen3_asr_free(fa_ctx); fa_ctx = nullptr; return;
        }
        fprintf(stderr, "fa-aligner: aligner loaded (lm_head_dim=%d) in %lld ms\n", H, (long long) t_load_ms);
    };

    auto handle_align = [&](const FrameHeader & hdr, const std::vector<uint8_t> & payload,
                            Frame resp_tag) -> int {
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

        if (!unpack_audio_payload(payload, &json_meta, &pcm_delta)) {
            err = "ALIGN_*_REQ: unpack failed";
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

        const bool is_final = (resp_tag == Frame::FINAL_RESP);
        bool reused = false;
        if (err.empty() && is_final && cached_valid && pcm_delta.empty() &&
            cached_audio_ms == audio_seen_ms && cached_pcm_n == (int) acc_pcm.size() &&
            cached_words == words) {
            out_t0 = cached_t0; out_t1 = cached_t1; out_conf = cached_conf; ok = true; reused = true;
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
                    const int idx0 = (int) src, idx1 = idx0 + 1;
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

        if (ok && !reused && resp_tag == Frame::PARTIAL_RESP) {
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
        const int64_t t_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t_start).count();
        resp["profile"] = {{"t_load_ms",t_load_ms},{"t_resample_ms",t_resample_ms},
                           {"t_aligner_ms",t_align_ms},{"t_total_ms",t_total_ms},{"n_words",(int)words.size()}};
        fprintf(stderr, "  [fa-aligner %s] wall=%lldms load=%lld resample=%lld align=%lld acc=%d@%dHz seen=%lldms%s\n",
                is_final ? "final  " : "partial", (long long) t_total_ms, (long long) t_load_ms,
                (long long) t_resample_ms, (long long) t_align_ms, (int) acc_pcm.size(), acc_pcm_sr,
                (long long) audio_seen_ms, reused ? " [cached]" : "");
        if (send_frame(fd, resp_tag, hdr.req_id, resp.dump()) != IpcError::OK) {
            fprintf(stderr, "fa-aligner: %s send failed\n", is_final ? "FINAL_RESP" : "PARTIAL_RESP");
            return 9;
        }
        return 0;
    };

    while (true) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        IpcError e = recv_frame(fd, &hdr, &payload);
        if (e == IpcError::EofClean) { fprintf(stderr, "fa-aligner: parent EOF, exiting\n"); break; }
        if (e != IpcError::OK) { fprintf(stderr, "fa-aligner: recv_frame: %s\n", ipc_error_str(e)); if (fa_ctx) qwen3_asr_free(fa_ctx); return 3; }

        switch (static_cast<Frame>(hdr.type)) {
            case Frame::SHUTDOWN:
                fprintf(stderr, "fa-aligner: SHUTDOWN\n");
                if (fa_ctx) qwen3_asr_free(fa_ctx);
                return 0;
            case Frame::PING:
                send_frame(fd, Frame::PONG, hdr.req_id, payload.data(), payload.size());
                break;
            case Frame::LOAD_REQ: {
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
                json resp = {{"ok",ok},{"error",err},{"t_load_ms",t_load_ms}};
                if (send_frame(fd, Frame::LOAD_RESP, hdr.req_id, resp.dump()) != IpcError::OK) {
                    fprintf(stderr, "fa-aligner: LOAD_RESP send failed\n"); if (fa_ctx) qwen3_asr_free(fa_ctx); return 4;
                }
                break;
            }
            case Frame::PARTIAL_REQ: {
                int rc = handle_align(hdr, payload, Frame::PARTIAL_RESP);
                if (rc != 0) { if (fa_ctx) qwen3_asr_free(fa_ctx); return rc; }
                break;
            }
            case Frame::FINAL_REQ: {
                int rc = handle_align(hdr, payload, Frame::FINAL_RESP);
                if (rc != 0) { if (fa_ctx) qwen3_asr_free(fa_ctx); return rc; }
                break;
            }
            default:
                fprintf(stderr, "fa-aligner: unexpected frame 0x%x\n", hdr.type);
                break;
        }
    }
    if (fa_ctx) qwen3_asr_free(fa_ctx);
    return 0;
}

} // namespace fa
