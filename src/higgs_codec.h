#pragma once

// Higgs-Audio-v3 XCodec2 decode path (codes -> 24 kHz PCM).
//
// This is the DECODE-only half of the bundled XCodec2 codec used by
// bosonai/higgs-audio-v3-tts-4b. The semantic branch + acoustic encoder are
// intentionally dropped (encode-only, used for cloning). See
// kobbler/docker/higgs-audio-dev/PORT-SPEC.md §7 and the reference
// implementation in higgs-audio-ref (HiggsAudioV2TokenizerModel.decode):
//
//   codes[N=8, T] -> RVQ decode:  per cb  q_i = project_out_i(embed_i[codes_i])
//                                 quantized = Σ_i q_i              [1024, T]
//                 -> fc2:         acoustic = fc2(quantizedᵀ)ᵀ      [256,  T]
//                 -> acoustic_decoder (HF DAC): conv1 256->1024,
//                    5 DecoderBlocks (snake -> convT(stride s_i)
//                    -> 3 ResidualUnits{snake,conv k7 dil,snake,conv k1}),
//                    final snake -> conv2 ->1 ; NO tanh (_adjust_dac_decoder).
//                    strides {8,5,4,2,3} → ×960 → 24000/960 = 25 fps.
//
// The aux sidecar GGUF (higgs-codec-aux-f16.gguf) is produced by
// kobbler/docker/higgs-audio-dev/convert_aux_sidecar.py.

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>

namespace higgs {

// One DAC ResidualUnit: snake1 -> conv1(k7,dilation) -> snake2 -> conv2(k1).
struct dac_res_unit {
    int dilation = 1;
    struct ggml_tensor * snake1_alpha = nullptr;   // [1,C,1]  raw α (host-precomputed F32 copies below)
    struct ggml_tensor * snake1_inv   = nullptr;   // [1,C,1]  1/(α+1e-9)
    struct ggml_tensor * snake2_alpha = nullptr;
    struct ggml_tensor * snake2_inv   = nullptr;
    struct ggml_tensor * conv1_w = nullptr;        // [7, C, C]
    struct ggml_tensor * conv1_b = nullptr;        // [C]
    struct ggml_tensor * conv2_w = nullptr;        // [1, C, C]
    struct ggml_tensor * conv2_b = nullptr;        // [C]
};

// One DAC DecoderBlock: snake_in -> conv_t(stride) -> res1 -> res2 -> res3.
struct dac_dec_block {
    int stride = 1;
    struct ggml_tensor * snake_in_alpha = nullptr; // [1,Cin,1]
    struct ggml_tensor * snake_in_inv   = nullptr;
    struct ggml_tensor * conv_t_w = nullptr;       // [2*stride, Cout, Cin]
    struct ggml_tensor * conv_t_b = nullptr;       // [Cout]
    dac_res_unit res[3];
};

struct higgs_codec_config {
    int n_codebooks   = 8;
    int codebook_size = 1024;
    int codebook_dim  = 64;
    int quantizer_dim = 1024;   // RVQ latent
    int acoustic_dim  = 256;    // after fc2
    int decoder_chan  = 1024;   // conv1 out
    int sample_rate   = 24000;
    int n_blocks      = 5;
    std::vector<int> strides;   // {8,5,4,2,3}
};

struct higgs_codec_model {
    higgs_codec_config config;

    // tied audio embedding [8208, 2560] — packed in the same sidecar but used
    // by the LM engine, not the codec. Loaded only if present; ignored here.
    struct ggml_tensor * audio_embd = nullptr;

    // RVQ decode tensors (per codebook)
    std::vector<struct ggml_tensor *> vq_embed;        // [codebook_dim, codebook_size]
    std::vector<struct ggml_tensor *> vq_proj_out_w;   // [codebook_dim, quantizer_dim]
    std::vector<struct ggml_tensor *> vq_proj_out_b;   // [quantizer_dim]

    struct ggml_tensor * fc2_w = nullptr;              // [quantizer_dim, acoustic_dim]
    struct ggml_tensor * fc2_b = nullptr;              // [acoustic_dim]

    struct ggml_tensor * conv1_w = nullptr;            // [7, acoustic_dim, decoder_chan]
    struct ggml_tensor * conv1_b = nullptr;            // [decoder_chan]
    struct ggml_tensor * conv2_w = nullptr;            // [7, last_chan, 1]
    struct ggml_tensor * conv2_b = nullptr;            // [1]
    struct ggml_tensor * snake_final_alpha = nullptr;  // [1, last_chan, 1]
    struct ggml_tensor * snake_final_inv   = nullptr;

    std::vector<dac_dec_block> blocks;

    struct ggml_context * ctx = nullptr;        // weight tensor metadata
    ggml_backend_buffer_t buffer = nullptr;     // weight data (device)

    // Host-precomputed F32 snake α / 1-over-(α+eps) tensors live in a side ctx
    // because they're synthesised post-load (the GGUF stores raw α as F16).
    struct ggml_context * snake_ctx = nullptr;
    ggml_backend_buffer_t snake_buffer = nullptr;

    std::map<std::string, struct ggml_tensor *> tensors;
};

class HiggsCodecDecoder {
public:
    HiggsCodecDecoder() = default;
    ~HiggsCodecDecoder();

    bool load_model(const std::string & gguf_path);
    void unload_model();

    // Decode codes (row-major [n_frames, n_codebooks], int32) to 24 kHz PCM.
    bool decode(const int32_t * codes, int32_t n_frames, std::vector<float> & samples);

    // Debug variant: also returns the post-RVQ [quantizer_dim, T] and post-fc2
    // [acoustic_dim, T] intermediates (ggml channel-major layout: idx = t*C + c)
    // for tensor-level validation against the PyTorch oracle.
    bool decode_debug(const int32_t * codes, int32_t n_frames,
                      std::vector<float> & samples,
                      std::vector<float> & post_rvq,
                      std::vector<float> & post_fc2);

    const higgs_codec_config & get_config() const { return model_.config; }
    const std::string & get_error() const { return error_msg_; }

private:
    struct ggml_cgraph * build_graph(struct ggml_context * ctx0, int32_t n_frames,
                                     struct ggml_tensor ** out_rvq,
                                     struct ggml_tensor ** out_fc2);
    struct ggml_tensor * apply_snake(struct ggml_context * ctx, struct ggml_tensor * x,
                                     struct ggml_tensor * alpha, struct ggml_tensor * inv);
    struct ggml_tensor * apply_res_unit(struct ggml_context * ctx, struct ggml_tensor * x,
                                        const dac_res_unit & ru);

    higgs_codec_model model_;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t backend_cpu_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    std::vector<uint8_t> compute_meta_;
    std::string error_msg_;
};

} // namespace higgs
