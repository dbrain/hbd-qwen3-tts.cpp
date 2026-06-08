// higgs_worker_ipc.cpp — see higgs_worker_ipc.h. Mirrors the qwen3-tts
// worker_ipc framing (SOCK_STREAM + writev coalescing + EINTR-looping
// read/write), in namespace higgs with Higgs-specific frames.

#include "higgs_worker_ipc.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

namespace higgs {

const char * ipc_error_str(IpcError e) {
    switch (e) {
        case IpcError::OK:            return "OK";
        case IpcError::EofClean:      return "peer closed cleanly";
        case IpcError::EofMidFrame:   return "peer closed mid-frame";
        case IpcError::SocketError:   return "socket error";
        case IpcError::ProtocolError: return "protocol error";
        case IpcError::PayloadTooBig: return "payload too big";
    }
    return "unknown";
}

IpcError read_exact(int fd, void * buf, size_t len) {
    char * p = static_cast<char *>(buf);
    size_t got = 0;
    while (got < len) {
        ssize_t r = ::read(fd, p + got, len - got);
        if (r > 0) { got += static_cast<size_t>(r); continue; }
        if (r == 0) return got == 0 ? IpcError::EofClean : IpcError::EofMidFrame;
        if (errno == EINTR) continue;
        return IpcError::SocketError;
    }
    return IpcError::OK;
}

IpcError write_exact(int fd, const void * buf, size_t len) {
    const char * p = static_cast<const char *>(buf);
    size_t sent = 0;
    while (sent < len) {
        ssize_t w = ::write(fd, p + sent, len - sent);
        if (w > 0) { sent += static_cast<size_t>(w); continue; }
        if (w < 0) {
            if (errno == EINTR) continue;
            return IpcError::SocketError;
        }
    }
    return IpcError::OK;
}

IpcError send_frame(int fd, WFrame type, uint32_t req_id,
                    const void * payload, size_t payload_len) {
    if (payload_len > MAX_FRAME_PAYLOAD) return IpcError::PayloadTooBig;
    FrameHeader hdr{ static_cast<uint32_t>(type), static_cast<uint32_t>(payload_len), req_id };
    if (payload_len == 0) return write_exact(fd, &hdr, sizeof(hdr));

    iovec iov[2];
    iov[0].iov_base = &hdr;                          iov[0].iov_len = sizeof(hdr);
    iov[1].iov_base = const_cast<void *>(payload);   iov[1].iov_len = payload_len;

    size_t total = sizeof(hdr) + payload_len, sent = 0;
    while (sent < total) {
        iovec * cur_iov = iov; int n_iov = 2; size_t to_skip = sent;
        if (to_skip >= sizeof(hdr)) {
            cur_iov = &iov[1]; n_iov = 1; to_skip -= sizeof(hdr);
            cur_iov[0].iov_base = static_cast<char *>(iov[1].iov_base) + to_skip;
            cur_iov[0].iov_len  = payload_len - to_skip;
        } else if (to_skip > 0) {
            iov[0].iov_base = reinterpret_cast<char *>(&hdr) + to_skip;
            iov[0].iov_len  = sizeof(hdr) - to_skip;
        }
        ssize_t w = ::writev(fd, cur_iov, n_iov);
        if (w > 0) { sent += static_cast<size_t>(w); continue; }
        if (w < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "higgs send_frame writev failed: fd=%d errno=%d (%s)\n",
                    fd, errno, strerror(errno));
            return IpcError::SocketError;
        }
    }
    return IpcError::OK;
}

IpcError send_frame(int fd, WFrame type, uint32_t req_id, const std::string & json) {
    return send_frame(fd, type, req_id, json.data(), json.size());
}
IpcError send_frame(int fd, WFrame type, uint32_t req_id, const std::vector<uint8_t> & payload) {
    return send_frame(fd, type, req_id, payload.data(), payload.size());
}

IpcError recv_frame(int fd, FrameHeader * out_hdr, std::vector<uint8_t> * out_payload) {
    if (!out_hdr) return IpcError::ProtocolError;
    IpcError e = read_exact(fd, out_hdr, sizeof(*out_hdr));
    if (e != IpcError::OK) return e;
    if (out_hdr->len > MAX_FRAME_PAYLOAD) return IpcError::PayloadTooBig;
    out_payload->resize(out_hdr->len);
    if (out_hdr->len == 0) return IpcError::OK;
    return read_exact(fd, out_payload->data(), out_hdr->len);
}

std::vector<uint8_t> pack_payload(const std::string & meta, const void * blob, size_t blob_bytes) {
    const size_t jlen = meta.size();
    std::vector<uint8_t> out(sizeof(uint32_t) + jlen + blob_bytes);
    uint32_t j = static_cast<uint32_t>(jlen);
    std::memcpy(out.data(), &j, sizeof(j));
    if (jlen)       std::memcpy(out.data() + sizeof(j), meta.data(), jlen);
    if (blob_bytes) std::memcpy(out.data() + sizeof(j) + jlen, blob, blob_bytes);
    return out;
}

std::vector<uint8_t> pack_audio_payload(const std::string & meta, const float * samples, size_t n_samples) {
    return pack_payload(meta, samples, n_samples * sizeof(float));
}
std::vector<uint8_t> pack_codes_payload(const std::string & meta, const int32_t * codes, size_t n_codes) {
    return pack_payload(meta, codes, n_codes * sizeof(int32_t));
}

bool unpack_payload(const std::vector<uint8_t> & payload,
                    std::string * out_meta,
                    const uint8_t ** out_blob, size_t * out_blob_bytes) {
    if (payload.size() < sizeof(uint32_t)) return false;
    uint32_t jlen = 0;
    std::memcpy(&jlen, payload.data(), sizeof(jlen));
    if (sizeof(jlen) + (size_t)jlen > payload.size()) return false;
    if (out_meta) out_meta->assign(reinterpret_cast<const char *>(payload.data() + sizeof(jlen)), jlen);
    size_t off = sizeof(jlen) + jlen;
    if (out_blob)       *out_blob = payload.data() + off;
    if (out_blob_bytes) *out_blob_bytes = payload.size() - off;
    return true;
}

pid_t spawn_worker(const char * self_argv0,
                   const std::vector<std::string> & extra_argv,
                   int * out_parent_fd,
                   const char * role_flag) {
    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        fprintf(stderr, "higgs spawn_worker: socketpair failed: %s\n", strerror(errno));
        return -1;
    }
    int parent_fd = sv[0], worker_fd = sv[1];
    int flags = ::fcntl(parent_fd, F_GETFD);
    if (flags >= 0) ::fcntl(parent_fd, F_SETFD, flags | FD_CLOEXEC);

    pid_t pid = ::fork();
    if (pid < 0) {
        fprintf(stderr, "higgs spawn_worker: fork failed: %s\n", strerror(errno));
        ::close(sv[0]); ::close(sv[1]);
        return -1;
    }
    if (pid == 0) {
        ::close(parent_fd);
        char fd_buf[16];
        std::snprintf(fd_buf, sizeof(fd_buf), "%d", worker_fd);
        std::vector<std::string> owned;
        owned.emplace_back(self_argv0);
        owned.emplace_back(role_flag ? role_flag : "--higgs-worker");
        owned.emplace_back(fd_buf);
        for (auto & a : extra_argv) owned.push_back(a);
        std::vector<char *> argv_p;
        argv_p.reserve(owned.size() + 1);
        for (auto & s : owned) argv_p.push_back(s.data());
        argv_p.push_back(nullptr);
        ::execv(self_argv0, argv_p.data());
        std::fprintf(stderr, "higgs spawn_worker child: execv(%s) failed: %s\n",
                     self_argv0, strerror(errno));
        ::_exit(127);
    }
    ::close(worker_fd);
    if (out_parent_fd) *out_parent_fd = parent_fd;
    return pid;
}

} // namespace higgs
