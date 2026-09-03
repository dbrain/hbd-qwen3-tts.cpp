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
// The codec is the Qwen3-TTS 12.5 Hz audio tokenizer, NOT the Mimi that ships
// inside the Breeze checkpoint under `codec_model.*`. That Mimi is a decoy: the
// backbone emits codes in the tokenizer's space, and decoding them with Mimi
// gives fluent, correctly-paced speech that says the wrong words. Both halves
// already existed in this repo for qwen3-tts, so we reuse them verbatim --
// including the decoder's true streaming state (KV cache + causal-conv tails),
// which replaces the old prefix-re-decode loop.
#include "audio_tokenizer_decoder.h"
#include "audio_codec_encoder.h"

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

    // Rolling-context long-form. Chunk `text` at ~chunk_words words and render
    // the chunks in sequence, feeding the PREVIOUS chunk's generated codes (and
    // its text) back in as the reference for the next one. The prompt shape is
    // the clone template -- [S0]{prev_text}<audio prev>[S0]{next_text} -- which
    // is exactly a continuation, so the model carries intonation and pacing
    // across the seam, not just speaker identity.
    //
    // Chunking is unavoidable: max_new_frames defaults to 750 and the model
    // stops itself around 890-1100 frames whatever you set, so a book has to be
    // split. Text cannot be streamed into an in-flight generation instead --
    // the text sits BEFORE the audio frames in the sequence, so once frames are
    // being emitted there is nowhere to put more text.
    //
    // ref_max_frames caps how much history is carried: prompt tokens + ref
    // frames + new frames all share n_ctx (2048), so an unbounded reference
    // eats the budget it is trying to protect. ~250 frames (20 s) is plenty to
    // hold a voice.
    // stream_chunk_frames > 0 streams each text chunk's codec blocks through
    // on_chunk as they are produced, so TTFA stays at the first block rather
    // than the first whole chunk (~45 s in).
    // gap_ms inserts silence at each seam. The chunk boundary is a sentence
    // end, but the model cannot see the next chunk and so gives it no
    // sentence-final pause -- the last word of one chunk runs straight into the
    // first of the next, which is audible as a rush even when the voice matches.
    bool synthesize_long(const std::string & text, const gen_params & gp,
                         int chunk_words, int ref_max_frames,
                         int stream_chunk_frames, int gap_ms,
                         const pcm_cb & on_chunk, gen_result & out);

    // Split text into ~chunk_words chunks on sentence boundaries. Exposed so a
    // caller can align its own word offsets with what synthesize_long renders.
    static std::vector<std::string> chunk_text(const std::string & text, int chunk_words);

    // 24 kHz mono in -> Mimi codes, for voice cloning.
    bool encode_voice(const float * pcm, int n_samples, std::vector<int32_t> & codes, int & T);
    // Mimi codes -> 24 kHz mono. Exposed so a caller can round-trip known-good
    // audio through the codec alone and isolate it from the AR loop.
    bool decode_codes(const int32_t * codes, int T, std::vector<float> & pcm);

    // Diagnostic: prefill `text`'s prompt, then walk the given TRUE code frames
    // through the backbone, reporting for each frame the rank and log-prob our
    // lm_head assigns to the true codebook-0 code. Localises "the model is not
    // conditioned on the text" versus "the sampler or the loop is wrong".
    bool teacher_force(const std::string & text, const gen_params & gp,
                       const int32_t * codes, int T,
                       std::vector<int> & ranks, std::vector<float> & logprobs);

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
    qwen3_tts::AudioTokenizerDecoder cc_;
    qwen3_tts::AudioCodecEncoder      ce_;
    bool loaded_ = false;
    std::atomic<bool> cancel_{false};
    std::string error_msg_;
};

} // namespace breeze
