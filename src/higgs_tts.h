#pragma once

// Higgs-Audio-v3 TTS engine: LM backbone + 8-codebook delay AR loop + XCodec2
// decode. Mirrors higgs-audio-ref/ref_full.py end to end.
//
//   prompt ids -> prefill -> [audio_logits -> sampler.step (delay/EOC FSM)
//   -> feedback embed -> decode_step]* -> reverse delay -> codec -> 24kHz PCM
//
// Sampling FSM ported from sglang sampler.step (greedy + temp/top_k/top_p).

#include "higgs_lm.h"
#include "higgs_codec.h"
#include "text_tokenizer.h"

#include <atomic>
#include <map>
#include <string>
#include <vector>
#include <random>
#include <functional>

namespace higgs {

// Codec-vocab specials (inside each codebook's 1026-wide space).
constexpr int BOC_ID = 1024;
constexpr int EOC_ID = 1025;

// Higgs text-vocab special token IDs (Qwen3 tokenizer added tokens; PORT-SPEC §5).
struct higgs_specials {
    int tts = 151667, audio = 151670, text = 151672;
    int ref_text = 151680, ref_audio = 151679, eoc = 151674;
};

struct sampler_state {
    int n;
    int delay_count = 0;
    int eoc_countdown = -1;   // -1 == None
    bool done = false;
};

struct gen_params {
    float temperature = 0.0f;   // 0 => greedy
    int   top_k = 0;            // 0 => off
    float top_p = 1.0f;        // 1 => off
    int   max_new = 1024;
    uint32_t seed = 0;
    // Repetition-aware sampling (RAS, sglang/boson). 0 => off. When a per-codebook
    // token repeats >= ras_max_repeat times within the last ras_win_len sampled
    // frames, that codebook is resampled from the raw-logit softmax (no temperature).
    // Stops audio-token loops on long generations. Off by default (preserves the
    // validated greedy path); long-form enables it (win_len 7, max_repeat 2).
    int   ras_win_len = 0;
    int   ras_max_repeat = 2;
};

struct gen_result {
    std::vector<int32_t> codes_TN;   // reverted, clipped [T, N] row-major
    int T = 0;
    std::vector<float> pcm;          // 24 kHz mono
    double prefill_ms = 0, decode_ms = 0, codec_ms = 0;
    int steps = 0;
    std::vector<int32_t> delayed_flat;  // [L, N] raw delayed grid (debug)
    int L = 0;
};

class HiggsTTS {
public:
    HiggsTTS() = default;

    bool load(const std::string & backbone_gguf, const std::string & aux_gguf, int n_ctx = 4096);
    const std::string & get_error() const { return error_msg_; }

    HiggsLM & lm() { return lm_; }
    HiggsCodecDecoder & codec() { return codec_; }
    bool tokenizer_loaded() const { return tok_loaded_; }

    // Voice-clone from a recorded/uploaded waveform (the F32-encoder path).
    // set_aux_enc() points at the encode-augmented aux GGUF; encode_voice()
    // loads it on demand, encodes arbitrary-rate mono PCM → codes [T,N], then
    // FREES the encoder (keeps idle VRAM flat — uploads are rare). Returns false
    // if no encoder configured. N is the codebook count (8).
    void set_aux_enc(const std::string & path) { aux_enc_path_ = path; }
    bool has_encoder() const { return !aux_enc_path_.empty(); }
    bool encode_voice(const float * wav, int n_samples, int sr,
                      std::vector<int32_t> & codes_TN, int & T, int & N);

    // ---- cooperative cancellation (mirrors qwen3-tts) ----
    // request_cancel() from another thread flips an atomic the AR loops poll
    // between steps (and the long/multispeaker loops poll between turns). The
    // synth bails within ~one decode step (~tens of ms) and returns whatever
    // audio it had finalised. The CALLER must call clear_cancel() once before
    // each top-level synth (the engine never auto-clears, so a cancel raised
    // mid-long-form survives across its per-chunk primitives). is_cancel_requested()
    // lets the caller distinguish a cancelled run from a real failure.
    void request_cancel()  { cancel_requested_.store(true,  std::memory_order_relaxed); }
    void clear_cancel()    { cancel_requested_.store(false, std::memory_order_relaxed); }
    bool is_cancel_requested() const { return cancel_requested_.load(std::memory_order_relaxed); }

    // Per-PCM-block streaming callback (24 kHz mono): (pcm, n_samples, is_final).
    using pcm_cb = std::function<void(const float * pcm, int n, bool is_final)>;

    // Tokenise text into a Higgs zero-shot prompt:
    //   <|tts|> {control} <|text|> tok(text) <|audio|>
    // Control-token strings (e.g. "<|emotion:happy|>") in `text` are mapped to
    // their vocab IDs; the rest is BPE-encoded.
    std::vector<int32_t> build_prompt(const std::string & text) const;

    // MusicGen delay pattern: clean codes [T][N] (0..1023, row-major) -> delayed
    // [T+N-1][N] grid (cb i delayed i steps; BOC(1024) before a codebook's start,
    // EOC(1025) after its end). Inverse of the reverse-delay used after generation.
    static std::vector<int32_t> apply_delay_pattern(const int32_t * codes_TN, int T, int N);

    // Split text into ~max_words-word chunks (English): split paragraphs on
    // "\n\n", then group words; the last chunk of each paragraph keeps its "\n\n".
    // Mirrors boson prepare_chunk_text(method="word"). (Chinese jieba path skipped.)
    static std::vector<std::string> prepare_chunk_text(const std::string & text, int max_words = 100);

    // Tokenise an arbitrary string, mapping any "<|...|>" vocab tokens to their
    // single ids and BPE-encoding the rest. Lets callers build experimental prompt
    // layouts (e.g. <|system|>{desc}<|tts|><|text|>{t}<|audio|>) by hand.
    std::vector<int32_t> tokenize_raw(const std::string & s) const { return encode_with_specials(s); }

    // Full synth from text (tokenises + AR + codec).
    bool synthesize(const std::string & text, const gen_params & gp, gen_result & out);

    // Full synth from an already-tokenised prompt (text + control tokens, ending
    // in <|audio|>). Returns codes + PCM. `decode_audio=false` skips the codec
    // (codes-only, for fast AR validation).
    bool synthesize_from_ids(const std::vector<int32_t> & ids, const gen_params & gp,
                             gen_result & out, bool decode_audio = true);

    // Synthesize `text` conditioned on reference audio given as CLEAN codes
    // [ref_T][N] (0..1023, row-major). Builds the v3 voice-clone prompt and
    // injects the delayed ref codes in the <|ref_audio|> slot via the audio-embed
    // path (NO encoder):
    //   <|tts|> [<|ref_text|> tok(ref_text)] <|ref_audio|> [delayed ref] <|text|> tok(text) <|audio|>
    // ref_T==0 (or ref_codes_TN==nullptr) falls back to the zero-shot prompt.
    // ref_text="" omits the transcript turn (voice-only conditioning). This is the
    // shared primitive for rolling-context long-form (#8) and voice clone (#7).
    bool synthesize_with_ref(const std::string & text,
                             const int32_t * ref_codes_TN, int ref_T,
                             const std::string & ref_text,
                             const gen_params & gp, gen_result & out, bool decode_audio = true);

    // Rolling-context long-form: chunk `text` (~chunk_words words), render chunks
    // sequentially feeding the previous `buffer` chunks' GENERATED codes as the
    // <|ref_audio|> context so the voice carries across the whole document. Streams
    // each chunk's PCM via on_chunk (may be null); the full result is in `out`.
    // Uses gp as-is per chunk (caller sets temp ~0.3 + RAS for long-form).
    bool synthesize_long(const std::string & text, const gen_params & gp,
                         int buffer, int chunk_words,
                         const pcm_cb & on_chunk, gen_result & out);

    // Streaming synth: interleaves the AR loop with block codec-decode and
    // invokes `on_chunk(pcm, n_samples, is_final)` with each new PCM block
    // (24 kHz mono). Decodes all codes-so-far each block (codec is cheap, no
    // boundary artifacts) and emits only the newly-finalised tail. The full
    // result is also returned in `out`.
    bool synthesize_stream(const std::string & text, const gen_params & gp,
                           int chunk_frames, const pcm_cb & on_chunk, gen_result & out);

    // ---- multi-speaker dialogue (NOT model-native; a clone+stitch wrapper) ----
    struct dialogue_turn { std::string speaker; std::string text; };
    // Parse "[Speaker_N]:" tagged text into ordered turns. Leading untagged text
    // becomes a turn with an empty speaker. Tags tokenize as plain text in v3
    // (verified) so they must be stripped and dispatched here, not fed to the LM.
    static std::vector<dialogue_turn> parse_dialogue(const std::string & text);

    // A speaker's voice for multi-speaker synth: clean codes [T,N] + optional ref
    // transcript (usually empty — the transcript lever is marginal on v3).
    struct named_voice { std::vector<int32_t> codes_TN; int T = 0; std::string ref_text; };

    // Render a "[Speaker_N]:" dialogue: each turn is cloned from its speaker's
    // voice (voices[speaker]) via synthesize_with_ref and the per-turn PCM is
    // concatenated with `gap_ms` of silence between turns. When `rolling`, a
    // speaker's previously-generated codes are appended to its voice codes as
    // extra <|ref_audio|> context for prosodic continuity across that speaker's
    // turns. Turns whose speaker has no mapped voice are skipped. Streams each
    // turn's PCM via on_chunk (may be null). Full audio + summed timings in `out`.
    bool synthesize_multispeaker(const std::string & text,
                                 const std::map<std::string, named_voice> & voices,
                                 const gen_params & gp, int gap_ms, bool rolling,
                                 const pcm_cb & on_chunk, gen_result & out);

private:
    // BPE-encode `text`, but emit any "<|...|>" substrings that exist in the
    // vocab as their single token id (added-token handling for control tokens).
    std::vector<int32_t> encode_with_specials(const std::string & text) const;
    // Run the AR loop from the first-step `logits` (after prefill): sampler.step
    // FSM + decode_step feedback, then reverse delay + (optional) codec. Fills the
    // codes/T/pcm/decode_ms/codec_ms fields of `out` (caller sets prefill_ms).
    bool run_ar(std::vector<float> & logits, const gen_params & gp,
                gen_result & out, bool decode_audio);
    // One AR step of the multi-codebook delay/EOC sampler. Mutates `st`. Writes
    // the N sampled codes into `codes` (incl. BOC/EOC specials). `hist` is the
    // delayed grid generated so far (for RAS look-back; pass {} to disable).
    void sampler_step(const std::vector<float> & logits, sampler_state & st,
                      const gen_params & gp, std::vector<int32_t> & codes,
                      const std::vector<std::vector<int32_t>> & hist);
    int sample_one(const float * logits, int V, const gen_params & gp);
    // Sample from the raw-logit softmax (no temperature) — RAS resample path.
    int sample_softmax_raw(const float * logits, int V);

    HiggsLM lm_;
    HiggsCodecDecoder codec_;
    qwen3_tts::TextTokenizer tok_;
    bool tok_loaded_ = false;
    higgs_specials sp_;
    std::mt19937 rng_;
    std::string error_msg_;
    std::string aux_enc_path_;   // F32 voice-clone encoder GGUF (load-on-demand)
    std::atomic<bool> cancel_requested_{false};
};

} // namespace higgs
