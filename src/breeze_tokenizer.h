#pragma once

// Gemma BPE tokenizer for Breeze-TTS-2, read from our GGUF's tokenizer.ggml.*
// arrays. This is NOT the Qwen2 pre-tokenizer in text_tokenizer.h -- it is
// SentencePiece-derived BPE:
//
//   1. split the RAW text on the 6428 `added_tokens` (leftmost-longest, matched
//      before any normalisation; the set includes runs of \n, \t and U+2581)
//   2. normalise each remaining chunk: ' ' -> U+2581 (metaspace). There is NO
//      pre-tokenizer regex -- the whole chunk is one BPE word.
//   3. BPE over unicode characters, byte_fallback to <0xXX> for anything the
//      vocab does not carry, merged by rank with a lowest-rank-first heap.
//
// `add_special` prepends <bos> exactly once (TemplateProcessing "single").
// The reference prepends it per *segment* and then re-encodes the joined string
// with add_special=false, which nets out to the same single leading <bos> --
// getting that wrong is the double-BOS trap in PORT-SPEC §4.

#include "gguf.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace breeze {

class Tokenizer {
public:
    bool load_from_gguf(struct gguf_context * ctx);

    std::vector<int32_t> encode(const std::string & text, bool add_special) const;
    std::string decode(const std::vector<int32_t> & ids, bool skip_special = false) const;
    std::string token_text(int32_t id) const;

    int32_t token_to_id(const std::string & tok) const {
        auto it = vocab_.find(tok);
        return it == vocab_.end() ? -1 : it->second;
    }
    int32_t bos_id() const { return bos_; }
    int32_t eos_id() const { return eos_; }
    int32_t n_vocab() const { return (int32_t) tokens_.size(); }
    bool is_added(int32_t id) const {
        return id >= 0 && id < (int32_t) is_added_.size() && is_added_[id];
    }
    bool loaded() const { return !tokens_.empty(); }
    const std::string & get_error() const { return error_msg_; }

private:
    // BPE one already-normalised chunk into `out`.
    void bpe_chunk(const std::string & chunk, std::vector<int32_t> & out) const;

    std::vector<std::string> tokens_;                    // id -> text
    std::vector<uint8_t>     is_added_;                  // id -> matched literally
    std::unordered_map<std::string, int32_t> vocab_;     // text -> id
    std::unordered_map<std::string, int32_t> merges_;    // "A B" -> rank
    std::unordered_map<std::string, int32_t> added_;     // literal -> id
    int32_t byte_tok_[256];                              // <0xXX> ids, -1 if absent
    size_t  added_max_len_ = 0;
    int32_t bos_ = 2, eos_ = 1, pad_ = 0, unk_ = 3;
    std::string error_msg_;
};

} // namespace breeze
