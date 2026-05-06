// In-process audio encoder helpers for the OpenAI-compat HTTP server.
//
// Two surfaces:
//   - encode_one_shot(): full PCM buffer in, encoded byte buffer out.
//     Used by the legacy non-stream /v1/audio/speech path that calls
//     res.set_content() with a complete payload.
//   - StreamingEncoder: stateful RAII wrapper that lives for the
//     lifetime of a chunked response. Fed PCM in batches via push_pcm()
//     as the vocoder produces them; flushed with finish() at the end.
//     Destructor cleans up codec/format contexts so a connection-cancel
//     mid-stream doesn't leak.
//
// All inputs are float32 mono in [-1, 1]. Outputs:
//   - Mp3:     a stream of self-syncing libmp3lame frames (no Xing/LAME
//              header — that requires knowing total length up front,
//              which kills streaming). Content-Type: audio/mpeg.
//   - OggOpus: an ogg-muxed opus stream. avformat_write_header() emits
//              OpusHead + OpusTags up front so streaming clients can
//              join on the next page boundary. Content-Type: audio/ogg.
//   - Aac:     ffmpeg's native AAC-LC encoder wrapped in ADTS framing
//              via libavformat's "adts" muxer. Each ADTS frame is
//              self-syncing (12-bit 0xFFF syncword + 7-byte header),
//              so streaming clients can join on any frame boundary.
//              Content-Type: audio/aac.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace qwen3_tts_audio {

enum class Codec {
    Mp3,
    OggOpus,
    Aac,
};

const char * content_type_for(Codec codec);

// Encode a complete float32-mono PCM buffer. Returns encoded bytes.
// Empty return on encoder failure (caller should respond 500).
std::vector<uint8_t> encode_one_shot(
    Codec codec, int sample_rate, int bitrate_kbps,
    const float * pcm_mono, std::size_t n_samples);

class StreamingEncoder {
public:
    StreamingEncoder(Codec codec, int sample_rate, int bitrate_kbps);
    ~StreamingEncoder();
    StreamingEncoder(const StreamingEncoder &) = delete;
    StreamingEncoder & operator=(const StreamingEncoder &) = delete;

    // Whether the encoder is in a usable state. Constructor failure or
    // a fatal codec error sets this to false.
    bool valid() const;

    // Push a batch of PCM samples through the encoder. Encoded bytes
    // produced during this call are appended to `out`. May produce zero
    // bytes on early calls before the first full encoder frame is ready.
    bool push_pcm(const float * pcm_mono, std::size_t n,
                  std::vector<uint8_t> & out);

    // Drain any tail samples (zero-padded if short of frame size),
    // flush the encoder, and (for ogg-opus) write the muxer trailer.
    // Append remaining bytes to `out`. Idempotent.
    bool finish(std::vector<uint8_t> & out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qwen3_tts_audio
