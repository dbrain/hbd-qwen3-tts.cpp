#pragma once

// Mimi codec, both directions, for Breeze-TTS-2.
//
// DECODE (the synthesis path):
//   codes[T,16] -> split RVQ decode -> upsample (depthwise ConvTranspose1d,
//   groups=512, x2 -> 25 Hz) -> causal decoder_transformer (8 layers, LayerNorm
//   + LayerScale, RoPE 1e4) -> SEANet decoder (ConvTranspose1d [8,6,5,4] with
//   ELU + resnet blocks) -> 24 kHz mono. 1920 samples per codec frame.
//
// ENCODE (voice cloning only): SEANet encoder -> encoder_transformer ->
//   downsample -> split RVQ nearest-neighbour search. Mirrors the Mimi encoder
//   already proven in audio_codec_encoder.cpp, against Breeze's tensor names.
//
// NOTE this is NOT the qwen3-tts vocoder in audio_tokenizer_decoder.h. That one
// is DAC-shaped (Snake + ConvNeXt upsample blocks); Mimi's decoder is SEANet
// with ELU and no Snake anywhere. The encoder halves DO match, architecturally.

#include "breeze_weights.h"

#include <string>
#include <vector>

namespace breeze {

class MimiCodec {
public:
    ~MimiCodec() { free_state(); }

    bool init(BreezeWeights * w);

    // codes: [n_frames * n_codebooks] row-major (codebook fastest).
    bool decode(const int32_t * codes, int n_frames, std::vector<float> & pcm);

    // 24 kHz mono in -> codes [n_frames * n_codebooks].
    bool encode(const float * pcm, int n_samples,
                std::vector<int32_t> & codes, int & n_frames);

    size_t sched_bytes() const {
        return (sched_ && w_) ? ggml_backend_sched_get_buffer_size(sched_, w_->backend()) : 0;
    }
    const std::string & get_error() const { return error_msg_; }

private:
    struct tfm_layer {
        struct ggml_tensor * attn_norm_w = nullptr, * attn_norm_b = nullptr;
        struct ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
        struct ggml_tensor * attn_scale = nullptr;
        struct ggml_tensor * ffn_norm_w = nullptr, * ffn_norm_b = nullptr;
        struct ggml_tensor * fc1 = nullptr, * fc2 = nullptr;
        struct ggml_tensor * ffn_scale = nullptr;
    };
    struct resnet {                      // block.1 (k=3) + block.3 (k=1)
        struct ggml_tensor * c1w = nullptr, * c1b = nullptr;
        struct ggml_tensor * c2w = nullptr, * c2b = nullptr;
        int dilation = 1;
    };

    void free_state();
    bool load_side(bool decoder);
    // shared graph pieces
    struct ggml_tensor * tfm_stack(struct ggml_context * c, struct ggml_tensor * x,
                                   const std::vector<tfm_layer> & layers,
                                   int seq, struct ggml_tensor * pos);
    struct ggml_tensor * resnet_block(struct ggml_context * c, struct ggml_tensor * x,
                                      const resnet & r);
    struct ggml_cgraph * build_decode_graph(struct ggml_context * c, int n_frames);
    struct ggml_cgraph * build_encode_graph(struct ggml_context * c, int n_samples);

    BreezeWeights * w_ = nullptr;
    codec_config cfg_;

    // quantizer
    struct ggml_tensor * vq_sem_embed_ = nullptr, * vq_sem_usage_ = nullptr;
    struct ggml_tensor * vq_sem_in_ = nullptr, * vq_sem_out_ = nullptr;
    std::vector<struct ggml_tensor *> vq_ac_embed_, vq_ac_usage_;
    struct ggml_tensor * vq_ac_in_ = nullptr, * vq_ac_out_ = nullptr;

    // decoder side
    std::vector<struct ggml_tensor *> up_tap_;    // 4 depthwise taps, [1, hidden]
    std::vector<tfm_layer> dec_tfm_;
    struct ggml_tensor * dec_in_w_ = nullptr, * dec_in_b_ = nullptr;   // layers.0 conv k=7
    struct ggml_tensor * dec_out_w_ = nullptr, * dec_out_b_ = nullptr; // layers.14 conv k=3
    std::vector<struct ggml_tensor *> dec_up_w_, dec_up_b_;            // ConvTranspose1d
    // Polyphase repack of each ConvTranspose1d: dec_up_poly_[i][p] is the
    // kernel-2 Conv1d for output phase p. See the comment in init().
    std::vector<std::vector<struct ggml_tensor *>> dec_up_poly_;
    std::vector<resnet> dec_res_;

    // encoder side
    std::vector<tfm_layer> enc_tfm_;
    struct ggml_tensor * enc_in_w_ = nullptr, * enc_in_b_ = nullptr;
    struct ggml_tensor * enc_out_w_ = nullptr, * enc_out_b_ = nullptr;
    std::vector<struct ggml_tensor *> enc_ds_w_, enc_ds_b_;
    std::vector<resnet> enc_res_;
    struct ggml_tensor * downsample_w_ = nullptr;

    // host-side copies of the RVQ codebooks (normalised) for the encode search
    std::vector<std::vector<float>> cb_host_;   // [n_quantizers][2048*256]

    // taps + anything we synthesise at load time live in their own tiny buffer
    struct ggml_context * aux_ctx_ = nullptr;
    ggml_backend_buffer_t aux_buf_ = nullptr;

    ggml_backend_sched_t sched_ = nullptr;
    std::vector<uint8_t> compute_meta_;
    std::string error_msg_;
};

} // namespace breeze
