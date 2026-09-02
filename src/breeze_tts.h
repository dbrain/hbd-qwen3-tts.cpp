#pragma once

// Breeze-TTS-2 engine: prompt assembly + the two-level AR loop + Mimi decode.
//
// Prompt layout follows breeze_infer/templates.py exactly. Every text segment
// is tokenised with add_special_tokens=True (one <bos> each) and encoded by the
// text encoder INDEPENDENTLY, so segments never attend across each other;
// reference audio contributes one <|AUDIO|> id per codec frame plus a trailing
// <|audio_eos|>.
//
//   plain            [S0]{text}
//   instruction      [S0]<ins_bos>{instruction}<ins_eos>{text}
//   clone            [S0]{ref_text} | <audio> | [S0]{text}
//   clone+instruction[S0]{ref_text} | <audio> | [S0]<ins_bos>{ins}<ins_eos>{text}
//
// CFG is deliberately NOT implemented: the reference gates it on
// `cfg_scale != 1.0 and negative_prompt_ids is not None` and calls the
// alternative "a wasteful dual pass". Single pass is the default and the
// entire reason this port is fast.

#include "breeze_weights.h"
#include "breeze_text_enc.h"
#include "breeze_lm.h"
#include "breeze_depth.h"
#include "breeze_codec.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace breeze {

// Dumps and resets the per-stage BREEZE_PROF accumulators (no-op when unset).
void prof_dump_all();

struct gen_params {
    float temperature = 0.9f;
    int   top_k = 50;
    float top_p = 1.0f;
    float repetition_penalty = 1.1f;
    float depth_temperature = 0.9f;
    int   depth_top_k = 50;
    float depth_top_p = 1.0f;
    int   max_new_frames = 750;
    uint32_t seed = 0;
    std::string speaker = "S0";
    std::string instruction;      // empty => plain template
};

struct ref_voice {
    std::vector<int32_t> codes;   // [T * n_codebooks]
    int T = 0;
    std::string ref_text;
};

struct gen_result {
    std::vector<int32_t> codes;   // [T * n_codebooks]
    int T = 0;
    std::vector<float> pcm;       // 24 kHz mono
    int n_prompt_tokens = 0;
    double prefill_ms = 0, decode_ms = 0, codec_ms = 0, text_enc_ms = 0, ttfa_ms = 0;
};

class BreezeTTS {
public:
    bool load(const std::string & gguf_path, int n_ctx = 2048);
    bool loaded() const { return loaded_; }
    void unload();

    const breeze_config & cfg() const { return w_.cfg(); }
    int sample_rate() const { return w_.cfg().cc.sample_rate; }
    const std::string & get_error() const { return error_msg_; }

    // (pcm, n_samples, is_final)
    using pcm_cb = std::function<void(const float *, int, bool)>;

    bool synthesize(const std::string & text, const gen_params & gp,
                    const ref_voice * ref, gen_result & out);
    bool synthesize_stream(const std::string & text, const gen_params & gp,
                           const ref_voice * ref, int chunk_frames,
                           const pcm_cb & on_chunk, gen_result & out);

    // 24 kHz mono in -> Mimi codes, for voice cloning.
    bool encode_voice(const float * pcm, int n_samples, std::vector<int32_t> & codes, int & T);

    void request_cancel() { cancel_.store(true, std::memory_order_relaxed); }
    void clear_cancel()   { cancel_.store(false, std::memory_order_relaxed); }
    bool cancelled() const { return cancel_.load(std::memory_order_relaxed); }

    // VRAM inventory (weights + KV + scheduler reservations), MiB.
    void log_vram(const char * label) const;

private:
    // Builds the token ids and the matching prompt embedding block.
    bool build_prompt(const std::string & text, const gen_params & gp,
                      const ref_voice * ref,
                      std::vector<int32_t> & ids, std::vector<float> & embeds,
                      double & text_enc_ms);
    bool run_ar(const gen_params & gp, int chunk_frames, const pcm_cb * on_chunk,
                gen_result & out);
    std::string speaker_tag(const std::string & spk) const;

    // Carried between prefill and the AR loop so run_ar() has one entry shape.
    std::vector<float> prefill_hidden_, prefill_logits_;
    std::chrono::steady_clock::time_point t_ar_start_;

    BreezeWeights w_;
    TextEncoder   te_;
    Backbone      bb_;
    DepthDecoder  dd_;
    MimiCodec     cc_;
    bool loaded_ = false;
    std::atomic<bool> cancel_{false};
    std::string error_msg_;
};

} // namespace breeze
