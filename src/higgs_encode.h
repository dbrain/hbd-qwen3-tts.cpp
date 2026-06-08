#pragma once

// Higgs-Audio-v3 XCodec2 ENCODE path (24 kHz wav -> [T, 8] discrete codes).
//
// This is the inverse front-half of the already-ported DECODE codec
// (higgs_codec.{h,cpp}); it is the voice-clone path. Pipeline (mirrors
// HiggsAudioV2TokenizerModel.encode; see
// kobbler/docker/higgs-audio-dev/ENCODE-PORT-SPEC.md):
//
//   wav(24k,[L]) ->
//     _extract_semantic_features:
//        resample 24k->16k (Kaiser-windowed sinc, ratio 2/3),
//        pad(160,160), HuBERT (7-conv feature extractor w/ GroupNorm on layer 0,
//        feature_projection, weight-normed pos_conv add, encoder.layer_norm,
//        12 post-norm bidirectional transformer layers), mean-of-13-hidden-states,
//        stride-2 subsample                                          -> [T, 768]
//     encoder_semantic (conv + 2 conv_blocks)                        -> [768, T]
//     acoustic_encoder (DAC on raw 24k wav, conditional pad(480,480))-> [256, T]
//     cat([acoustic(256), semantic(768)])                            -> [1024, T]
//     fc (1024->1024)                                                -> [1024, T]
//     RVQ encode (8 stages: project_in 1024->64, nearest-L2 over 1024
//        codebook vecs, residual in 1024-dim via project_out)        -> codes[T,8]
//
// Weights come from the encode-augmented aux sidecar GGUF
// (higgs-codec-aux-enc-f16.gguf) produced by convert_aux_sidecar.py --encode.

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>

namespace higgs {

struct higgs_enc_config {
    int sample_rate          = 24000;
    int semantic_sample_rate = 16000;
    int hop_length           = 960;
    int downsample_factor    = 320;
    int semantic_downsample_factor = 2;
    int model_pad            = 480;
    int n_codebooks          = 8;
    int codebook_size        = 1024;
    int codebook_dim         = 64;
    int quantizer_dim        = 1024;
    int acoustic_dim         = 256;
    int semantic_dim         = 768;
    int hubert_hidden        = 512;
    int n_semantic_layers    = 12;
    int semantic_heads       = 12;
    int pos_conv_groups      = 16;
    int pos_conv_kernel      = 128;
};

// One HuBERT transformer layer (post-norm, bidirectional MHA).
struct hubert_layer {
    struct ggml_tensor * q_w = nullptr, * q_b = nullptr;
    struct ggml_tensor * k_w = nullptr, * k_b = nullptr;
    struct ggml_tensor * v_w = nullptr, * v_b = nullptr;
    struct ggml_tensor * o_w = nullptr, * o_b = nullptr;
    struct ggml_tensor * ln_w = nullptr, * ln_b = nullptr;       // post-attn LN
    struct ggml_tensor * ff1_w = nullptr, * ff1_b = nullptr;     // 768->3072
    struct ggml_tensor * ff2_w = nullptr, * ff2_b = nullptr;     // 3072->768
    struct ggml_tensor * fln_w = nullptr, * fln_b = nullptr;     // post-ffn LN
};

// One encoder_semantic residual unit: act(ELU) -> conv1(k3) -> act -> conv2(k1).
struct sem_res_unit {
    struct ggml_tensor * conv1_w = nullptr;   // [3,768,768]  bias-less
    struct ggml_tensor * conv2_w = nullptr;   // [1,768,768]  bias-less
};
struct sem_enc_block {
    sem_res_unit res[2];
    struct ggml_tensor * conv_w = nullptr;    // [3,768,768]
    struct ggml_tensor * conv_b = nullptr;    // [768]
};

// DAC encoder residual unit (snake1 -> conv1(k7,dil) -> snake2 -> conv2(k1)).
struct dac_enc_res_unit {
    int dilation = 1;
    struct ggml_tensor * snake1_alpha = nullptr, * snake1_inv = nullptr;
    struct ggml_tensor * snake2_alpha = nullptr, * snake2_inv = nullptr;
    struct ggml_tensor * conv1_w = nullptr;   // [7,C,C]
    struct ggml_tensor * conv1_b = nullptr;   // [C]
    struct ggml_tensor * conv2_w = nullptr;   // [1,C,C]
    struct ggml_tensor * conv2_b = nullptr;   // [C]
};
// DAC encoder block (res1 -> res2 -> res3 -> snake_pre -> strided conv1).
struct dac_enc_block {
    int stride = 1;
    dac_enc_res_unit res[3];
    struct ggml_tensor * snake_pre_alpha = nullptr, * snake_pre_inv = nullptr;
    struct ggml_tensor * conv1_w = nullptr;   // [2*stride, Cin, Cout]
    struct ggml_tensor * conv1_b = nullptr;   // [Cout]
};

struct higgs_enc_model {
    higgs_enc_config config;

    // --- HuBERT feature extractor ---
    struct ggml_tensor * feat_conv[7] = {nullptr};   // [k,Cin,Cout]
    struct ggml_tensor * gn0_w = nullptr, * gn0_b = nullptr;     // GroupNorm(512,512)
    // --- feature projection ---
    struct ggml_tensor * fproj_ln_w = nullptr, * fproj_ln_b = nullptr;   // LayerNorm(512)
    struct ggml_tensor * fproj_w = nullptr, * fproj_b = nullptr;         // 512->768
    // --- encoder front matter ---
    struct ggml_tensor * pos_conv_w = nullptr, * pos_conv_b = nullptr;   // grouped conv (reconstructed)
    struct ggml_tensor * enc_ln_w = nullptr, * enc_ln_b = nullptr;       // LayerNorm(768) pre-layers
    std::vector<hubert_layer> layers;

    // --- encoder_semantic ---
    struct ggml_tensor * sem_conv_w = nullptr;     // [3,768,768] bias-less
    std::vector<sem_enc_block> sem_blocks;

    // --- acoustic_encoder (DAC) ---
    struct ggml_tensor * aenc_conv1_w = nullptr, * aenc_conv1_b = nullptr; // [7,1,64]
    std::vector<dac_enc_block> aenc_blocks;
    struct ggml_tensor * aenc_snake_post_alpha = nullptr, * aenc_snake_post_inv = nullptr; // [1,2048,1]
    struct ggml_tensor * aenc_conv2_w = nullptr, * aenc_conv2_b = nullptr; // [3,2048,256]

    // --- projection + RVQ ---
    struct ggml_tensor * fc_w = nullptr, * fc_b = nullptr;       // 1024->1024
    std::vector<struct ggml_tensor *> vq_proj_in_w;   // [1024,64]  (ggml: ne0=1024)
    std::vector<struct ggml_tensor *> vq_proj_in_b;   // [64]
    std::vector<struct ggml_tensor *> vq_embed;       // codebook [64,1024]  (ggml: ne0=64)
    std::vector<struct ggml_tensor *> vq_proj_out_w;  // [64,1024]
    std::vector<struct ggml_tensor *> vq_proj_out_b;  // [1024]

    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    // host-precomputed F32 snake α + 1/(α+eps) (raw α stored F16 in gguf)
    struct ggml_context * snake_ctx = nullptr;
    ggml_backend_buffer_t snake_buffer = nullptr;

    std::map<std::string, struct ggml_tensor *> tensors;
};

class HiggsCodecEncoder {
public:
    HiggsCodecEncoder() = default;
    ~HiggsCodecEncoder();

    bool load_model(const std::string & gguf_path);
    void unload_model();

    // Encode a mono 24 kHz waveform to discrete codes.
    //   wav24k     : n_samples f32 PCM at 24000 Hz
    //   codes_TN   : row-major [T, n_codebooks] int32 output
    //   T          : number of frames
    bool encode(const float * wav24k, int n_samples,
                std::vector<int32_t> & codes_TN, int & T);

    // Same, but for an arbitrary input sample rate: resamples in_sr→24 kHz
    // (the encoder's own Kaiser-windowed sinc) before encoding. Use for
    // browser-recorded / uploaded clips at 44.1/48 kHz.
    bool encode(const float * wav, int n_samples, int in_sr,
                std::vector<int32_t> & codes_TN, int & T);

    // Debug variant: also returns each stage tensor in oracle layout
    //   semfeat  [T, 768]      (raw _extract_semantic_features)
    //   semantic [768, T]      (encoder_semantic output, channel-major idx=c*T+t)
    //   acoustic [256, T]
    //   prefc    [1024, T]
    //   postfc   [1024, T]
    bool encode_debug(const float * wav24k, int n_samples,
                      std::vector<int32_t> & codes_TN, int & T,
                      std::vector<float> & semfeat,
                      std::vector<float> & semantic,
                      std::vector<float> & acoustic,
                      std::vector<float> & prefc,
                      std::vector<float> & postfc);

    const higgs_enc_config & get_config() const { return model_.config; }
    const std::string & get_error() const { return error_msg_; }

private:
    // CPU-side band-limited sinc (Kaiser) resample, matching torchaudio defaults.
    static std::vector<float> resample(const std::vector<float> & in, int orig_sr, int new_sr);

    // HuBERT forward on a [L16] float buffer -> semfeat [T,768] (channel-minor: idx=t*768+d).
    bool run_semantic(const std::vector<float> & wav16, std::vector<float> & semfeat, int & T_sem);
    // encoder_semantic + acoustic + fc + RVQ given semfeat[T,768] & raw 24k wav.
    bool run_codes(const std::vector<float> & semfeat, int T_sem,
                   const float * wav24k, int n_samples,
                   std::vector<int32_t> & codes_TN, int & T,
                   std::vector<float> & semantic, std::vector<float> & acoustic,
                   std::vector<float> & prefc, std::vector<float> & postfc);

    struct ggml_tensor * apply_snake(struct ggml_context * ctx, struct ggml_tensor * x,
                                     struct ggml_tensor * alpha, struct ggml_tensor * inv);

    higgs_enc_model model_;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t backend_cpu_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    std::vector<uint8_t> compute_meta_;
    std::string error_msg_;
};

} // namespace higgs
