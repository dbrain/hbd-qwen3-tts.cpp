// In-process mp3 + ogg-opus encoders backed by libav* (ffmpeg 7.1).
//
// Linked from the qwen3-tts-server target so HTTP responses can compress
// the vocoder's float32 mono output for chunked streaming or one-shot
// delivery. No subprocess shell-out — encoder/muxer state is per-request
// RAII so cancel-mid-stream is clean.

#include "ffmpeg_encode.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec.h>
#include <libavcodec/codec_id.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
}

namespace qwen3_tts_audio {

namespace {

constexpr int kAvioBufferSize = 4096;

// Floor on bitrate for the encoder's own opening — a request bitrate that
// got past server-side validation is already >= 32 kbps; this is just the
// encoder's "don't pass 0" floor.
constexpr int kMinBitrateKbps = 8;

void log_av_error(const char * where, int err) {
    char msg[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(err, msg, sizeof(msg));
    fprintf(stderr, "ffmpeg_encode: %s failed: %s\n", where, msg);
}

void float_to_s16p_mono(const float * src, int16_t * dst, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        float s = src[i];
        if (s > 1.0f)  s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        dst[i] = (int16_t)(s * 32767.0f);
    }
}

// Custom AVIO write callback for the ogg muxer. Appends pages into the
// std::vector<uint8_t> passed via opaque. Returns bytes written (matching
// the libavformat contract — negative on error).
//
// `buf` is non-const here to stay ABI-compatible with ffmpeg 6.x (ubuntu
// 24.04 apt). ffmpeg 7.x switched the parameter to `const uint8_t *`,
// which is implicitly compatible with the non-const declaration.
int avio_write_to_vector(void * opaque, uint8_t * buf, int buf_size) {
    auto * sink = static_cast<std::vector<uint8_t> *>(opaque);
    sink->insert(sink->end(), buf, buf + buf_size);
    return buf_size;
}

} // namespace

const char * content_type_for(Codec codec) {
    switch (codec) {
        case Codec::Mp3:     return "audio/mpeg";
        case Codec::OggOpus: return "audio/ogg";
        case Codec::Aac:     return "audio/aac";
    }
    return "application/octet-stream";
}

namespace {
// Codec needs the libavformat-side muxer (so its packets get wrapped into
// pages/ADTS frames before hitting the wire). Mp3 packets are
// self-syncing MPEG audio frames straight from the encoder — no muxer.
bool codec_needs_muxer(Codec c) {
    return c == Codec::OggOpus || c == Codec::Aac;
}

// Muxer name for the libavformat side. Must match an avformat_alloc_output_context2
// recognised muxer.
const char * codec_muxer(Codec c) {
    switch (c) {
        case Codec::OggOpus: return "ogg";
        case Codec::Aac:     return "adts";
        default:             return nullptr;
    }
}

// Codec id for avcodec_find_encoder.
AVCodecID codec_av_id(Codec c) {
    switch (c) {
        case Codec::Mp3:     return AV_CODEC_ID_MP3;
        case Codec::OggOpus: return AV_CODEC_ID_OPUS;
        case Codec::Aac:     return AV_CODEC_ID_AAC;
    }
    return AV_CODEC_ID_NONE;
}

const char * codec_pretty(Codec c) {
    switch (c) {
        case Codec::Mp3:     return "mp3";
        case Codec::OggOpus: return "ogg-opus";
        case Codec::Aac:     return "aac";
    }
    return "?";
}
} // namespace

struct StreamingEncoder::Impl {
    Codec            codec;
    int              sample_rate     = 0;
    int              bitrate_kbps    = 0;

    AVCodecContext * cctx            = nullptr;
    AVFrame *        frame           = nullptr;
    AVPacket *       pkt             = nullptr;
    AVAudioFifo *    fifo            = nullptr;

    // Ogg muxer (OggOpus only).
    AVFormatContext * fmt_ctx        = nullptr;
    AVIOContext *     avio           = nullptr;
    uint8_t *         avio_buffer    = nullptr;
    std::vector<uint8_t> mux_sink;
    AVStream *        stream         = nullptr;

    // Running sample count, used as PTS for muxer-bound packets.
    int64_t           samples_in     = 0;

    bool              valid          = false;
    bool              finished       = false;
    bool              header_written = false;

    Impl(Codec c, int sr, int kbps)
        : codec(c), sample_rate(sr), bitrate_kbps(std::max(kbps, kMinBitrateKbps)) {}
    ~Impl() { close(); }

    bool open();
    bool encode_one_frame(int nb_samples_in_frame, std::vector<uint8_t> & out);
    bool drain_packets(std::vector<uint8_t> & out);
    void drain_mux_sink(std::vector<uint8_t> & out);
    void close();

    AVSampleFormat sample_fmt() const {
        // libmp3lame wants planar int16; libopus wants interleaved float;
        // ffmpeg's native AAC encoder wants planar float (FLTP).
        switch (codec) {
            case Codec::Mp3:     return AV_SAMPLE_FMT_S16P;
            case Codec::OggOpus: return AV_SAMPLE_FMT_FLT;
            case Codec::Aac:     return AV_SAMPLE_FMT_FLTP;
        }
        return AV_SAMPLE_FMT_NONE;
    }
};

bool StreamingEncoder::Impl::open() {
    const AVCodec * encoder = avcodec_find_encoder(codec_av_id(codec));
    if (!encoder) {
        fprintf(stderr, "ffmpeg_encode: no encoder for %s\n", codec_pretty(codec));
        return false;
    }

    cctx = avcodec_alloc_context3(encoder);
    if (!cctx) return false;
    cctx->bit_rate    = (int64_t) bitrate_kbps * 1000;
    cctx->sample_rate = sample_rate;
    cctx->sample_fmt  = sample_fmt();
    av_channel_layout_default(&cctx->ch_layout, 1);

    int err = avcodec_open2(cctx, encoder, nullptr);
    if (err < 0) { log_av_error("avcodec_open2", err); return false; }

    pkt = av_packet_alloc();
    if (!pkt) return false;

    frame = av_frame_alloc();
    if (!frame) return false;
    frame->nb_samples = cctx->frame_size > 0 ? cctx->frame_size : 1024;
    frame->format     = cctx->sample_fmt;
    if (av_channel_layout_copy(&frame->ch_layout, &cctx->ch_layout) < 0) return false;
    err = av_frame_get_buffer(frame, 0);
    if (err < 0) { log_av_error("av_frame_get_buffer", err); return false; }

    fifo = av_audio_fifo_alloc(cctx->sample_fmt, 1, frame->nb_samples * 4);
    if (!fifo) return false;

    if (codec_needs_muxer(codec)) {
        err = avformat_alloc_output_context2(&fmt_ctx, nullptr, codec_muxer(codec), nullptr);
        if (err < 0 || !fmt_ctx) { log_av_error("avformat_alloc_output_context2", err); return false; }

        avio_buffer = (uint8_t *) av_malloc(kAvioBufferSize);
        if (!avio_buffer) return false;
        mux_sink.reserve(8192);
        avio = avio_alloc_context(avio_buffer, kAvioBufferSize, /*write_flag=*/1,
                                   /*opaque=*/&mux_sink,
                                   /*read_packet=*/nullptr,
                                   /*write_packet=*/&avio_write_to_vector,
                                   /*seek=*/nullptr);
        if (!avio) { av_free(avio_buffer); avio_buffer = nullptr; return false; }
        // ownership of avio_buffer transferred to AVIOContext; cleared in close()
        avio_buffer = nullptr;
        fmt_ctx->pb = avio;

        stream = avformat_new_stream(fmt_ctx, nullptr);
        if (!stream) return false;
        if (avcodec_parameters_from_context(stream->codecpar, cctx) < 0) return false;
        stream->time_base = AVRational{1, sample_rate};

        err = avformat_write_header(fmt_ctx, nullptr);
        if (err < 0) { log_av_error("avformat_write_header", err); return false; }
        header_written = true;
    }

    valid = true;
    return true;
}

bool StreamingEncoder::Impl::encode_one_frame(int nb_samples_in_frame,
                                              std::vector<uint8_t> & out) {
    int err = av_frame_make_writable(frame);
    if (err < 0) { log_av_error("av_frame_make_writable", err); return false; }

    // FIFO -> frame->data[0]. Zero-pad short tail frames if the FIFO has
    // fewer than nb_samples available.
    int fifo_have = av_audio_fifo_size(fifo);
    int n = std::min(nb_samples_in_frame, fifo_have);
    if (n > 0) {
        if (av_audio_fifo_read(fifo, (void **) frame->data, n) < n) {
            fprintf(stderr, "ffmpeg_encode: fifo underflow (had %d, wanted %d)\n",
                    fifo_have, n);
            return false;
        }
    }
    if (n < nb_samples_in_frame) {
        // Zero-pad. Plane size in bytes per channel for our 1-ch layout:
        const int bytes_per_sample = av_get_bytes_per_sample(cctx->sample_fmt);
        const int pad_bytes = (nb_samples_in_frame - n) * bytes_per_sample;
        std::memset(frame->data[0] + (n * bytes_per_sample), 0, pad_bytes);
    }
    frame->nb_samples = nb_samples_in_frame;
    frame->pts        = samples_in;
    samples_in       += nb_samples_in_frame;

    err = avcodec_send_frame(cctx, frame);
    if (err < 0) { log_av_error("avcodec_send_frame", err); return false; }
    return drain_packets(out);
}

bool StreamingEncoder::Impl::drain_packets(std::vector<uint8_t> & out) {
    while (true) {
        int err = avcodec_receive_packet(cctx, pkt);
        if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return true;
        if (err < 0) { log_av_error("avcodec_receive_packet", err); return false; }

        if (codec_needs_muxer(codec)) {
            // OggOpus + Aac: hand to the muxer. It wraps each packet in
            // its container framing (ogg pages / 7-byte ADTS header) and
            // calls our AVIO write_packet callback to spool the muxed
            // bytes into mux_sink, which we then drain.
            pkt->stream_index = stream->index;
            av_packet_rescale_ts(pkt, cctx->time_base, stream->time_base);
            int werr = av_write_frame(fmt_ctx, pkt);
            if (werr < 0) {
                log_av_error("av_write_frame", werr);
                av_packet_unref(pkt);
                return false;
            }
            drain_mux_sink(out);
        } else {
            // Mp3: each packet IS one self-syncing MPEG audio frame.
            // Append straight to the wire — no muxer.
            out.insert(out.end(), pkt->data, pkt->data + pkt->size);
        }
        av_packet_unref(pkt);
    }
}

void StreamingEncoder::Impl::drain_mux_sink(std::vector<uint8_t> & out) {
    if (mux_sink.empty()) return;
    out.insert(out.end(), mux_sink.begin(), mux_sink.end());
    mux_sink.clear();
}

void StreamingEncoder::Impl::close() {
    if (fmt_ctx) {
        if (header_written && fmt_ctx->pb) {
            // Best-effort flush; ignore errors.
            avio_flush(fmt_ctx->pb);
        }
        avformat_free_context(fmt_ctx);
        fmt_ctx = nullptr;
    }
    if (avio) {
        if (avio->buffer) av_free(avio->buffer);
        avio_context_free(&avio);
    }
    if (avio_buffer) {
        av_free(avio_buffer);
        avio_buffer = nullptr;
    }
    stream = nullptr;
    if (fifo) {
        av_audio_fifo_free(fifo);
        fifo = nullptr;
    }
    if (frame) av_frame_free(&frame);
    if (pkt)   av_packet_free(&pkt);
    if (cctx)  avcodec_free_context(&cctx);
    valid = false;
}

// --- StreamingEncoder public surface ---------------------------------------

StreamingEncoder::StreamingEncoder(Codec codec, int sample_rate, int bitrate_kbps)
    : impl_(std::make_unique<Impl>(codec, sample_rate, bitrate_kbps)) {
    if (!impl_->open()) impl_->close();
}

StreamingEncoder::~StreamingEncoder() = default;

bool StreamingEncoder::valid() const {
    return impl_ && impl_->valid;
}

bool StreamingEncoder::push_pcm(const float * pcm_mono, std::size_t n,
                                std::vector<uint8_t> & out) {
    if (!valid() || impl_->finished) return false;

    Impl & I = *impl_;
    // Drain anything the muxer has already buffered (OpusHead + OpusTags
    // land in mux_sink during avformat_write_header at construction time —
    // we want those bytes on the wire on the first push_pcm so TTFA isn't
    // artificially delayed by the encoder's first-frame warmup).
    I.drain_mux_sink(out);

    if (n == 0) return true;
    const int frame_size = I.cctx->frame_size > 0 ? I.cctx->frame_size : 1024;

    // Push the new samples into the FIFO in the encoder's preferred format.
    if (I.codec == Codec::Mp3) {
        // float -> S16P mono. Use a temporary stack-ish buffer in chunks
        // so we don't allocate-per-sample. 4096 samples = 8 KiB.
        constexpr std::size_t kChunk = 4096;
        int16_t tmp[kChunk];
        std::size_t off = 0;
        while (off < n) {
            std::size_t take = std::min(kChunk, n - off);
            float_to_s16p_mono(pcm_mono + off, tmp, take);
            void * data_ptr[1] = { (void *) tmp };
            int wrote = av_audio_fifo_write(I.fifo, data_ptr, (int) take);
            if (wrote < (int) take) {
                fprintf(stderr, "ffmpeg_encode: av_audio_fifo_write short (%d < %zu)\n",
                        wrote, take);
                I.valid = false;
                return false;
            }
            off += take;
        }
    } else {
        // Opus consumes interleaved float natively; mono = no interleave step.
        void * data_ptr[1] = { (void *) pcm_mono };
        int wrote = av_audio_fifo_write(I.fifo, data_ptr, (int) n);
        if (wrote < (int) n) {
            fprintf(stderr, "ffmpeg_encode: av_audio_fifo_write short (%d < %zu)\n",
                    wrote, n);
            I.valid = false;
            return false;
        }
    }

    // Drain full encoder frames from the FIFO.
    while (av_audio_fifo_size(I.fifo) >= frame_size) {
        if (!I.encode_one_frame(frame_size, out)) {
            I.valid = false;
            return false;
        }
    }
    return true;
}

bool StreamingEncoder::finish(std::vector<uint8_t> & out) {
    if (!impl_) return false;
    Impl & I = *impl_;
    if (I.finished) return true;
    I.finished = true;
    if (!I.valid) return false;

    const int frame_size = I.cctx->frame_size > 0 ? I.cctx->frame_size : 1024;
    int leftover = av_audio_fifo_size(I.fifo);
    if (leftover > 0) {
        if (!I.encode_one_frame(frame_size, out)) return false;
    }

    // Flush: send a NULL frame to signal end-of-stream, then drain.
    int err = avcodec_send_frame(I.cctx, nullptr);
    if (err < 0 && err != AVERROR_EOF) {
        log_av_error("avcodec_send_frame(NULL)", err);
        return false;
    }
    if (!I.drain_packets(out)) return false;

    if (codec_needs_muxer(I.codec) && I.fmt_ctx) {
        int werr = av_write_trailer(I.fmt_ctx);
        if (werr < 0) {
            log_av_error("av_write_trailer", werr);
            return false;
        }
        I.drain_mux_sink(out);
    }
    return true;
}

// --- one-shot encoder ------------------------------------------------------

std::vector<uint8_t> encode_one_shot(
    Codec codec, int sample_rate, int bitrate_kbps,
    const float * pcm_mono, std::size_t n_samples) {
    std::vector<uint8_t> out;
    if (n_samples == 0) return out;
    StreamingEncoder enc(codec, sample_rate, bitrate_kbps);
    if (!enc.valid()) return {};
    if (!enc.push_pcm(pcm_mono, n_samples, out)) return {};
    if (!enc.finish(out)) return {};
    return out;
}

} // namespace qwen3_tts_audio
