#include "breeze_tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <queue>

namespace breeze {

namespace {

const char * METASPACE = "\xe2\x96\x81";   // U+2581 LOWER ONE EIGHTH BLOCK

inline size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x06) return 2;
    if ((c >> 4) == 0x0e) return 3;
    if ((c >> 3) == 0x1e) return 4;
    return 1;   // invalid lead byte: consume one
}

} // namespace

bool Tokenizer::load_from_gguf(gguf_context * ctx) {
    for (int i = 0; i < 256; ++i) byte_tok_[i] = -1;

    const int64_t ti = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (ti < 0) { error_msg_ = "tokenizer.ggml.tokens missing"; return false; }
    const size_t nv = gguf_get_arr_n(ctx, ti);
    tokens_.resize(nv);
    is_added_.assign(nv, 0);
    vocab_.reserve(nv * 2);
    for (size_t i = 0; i < nv; ++i) {
        tokens_[i] = gguf_get_arr_str(ctx, ti, i);
        // First writer wins: ids are unique per string in this vocab, but a
        // defensive insert keeps the lowest id if that ever stops holding.
        vocab_.emplace(tokens_[i], (int32_t) i);
    }
    // Byte-fallback table: "<0xAB>" -> id.
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        snprintf(buf, sizeof(buf), "<0x%02X>", b);
        auto it = vocab_.find(buf);
        if (it != vocab_.end()) byte_tok_[b] = it->second;
    }

    const int64_t mi = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (mi < 0) { error_msg_ = "tokenizer.ggml.merges missing"; return false; }
    const size_t nm = gguf_get_arr_n(ctx, mi);
    merges_.reserve(nm * 2);
    for (size_t i = 0; i < nm; ++i) {
        merges_.emplace(gguf_get_arr_str(ctx, mi, i), (int32_t) i);
    }

    const int64_t ai = gguf_find_key(ctx, "breeze-tts.tokenizer.added_tokens");
    const int64_t ii = gguf_find_key(ctx, "breeze-tts.tokenizer.added_ids");
    if (ai >= 0 && ii >= 0) {
        const size_t na = gguf_get_arr_n(ctx, ai);
        const void * ids = gguf_get_arr_data(ctx, ii);
        const enum gguf_type it_ = gguf_get_arr_type(ctx, ii);
        for (size_t i = 0; i < na; ++i) {
            const char * s = gguf_get_arr_str(ctx, ai, i);
            int32_t id = 0;
            switch (it_) {
                case GGUF_TYPE_UINT32: id = (int32_t) ((const uint32_t *) ids)[i]; break;
                case GGUF_TYPE_INT32:  id =           ((const int32_t  *) ids)[i]; break;
                default: continue;
            }
            added_.emplace(s, id);
            added_max_len_ = std::max(added_max_len_, std::strlen(s));
            if (id >= 0 && id < (int32_t) is_added_.size()) is_added_[id] = 1;
        }
    } else {
        // Older converter output: fall back to token_type == CONTROL.
        const int64_t tt = gguf_find_key(ctx, "tokenizer.ggml.token_type");
        if (tt >= 0) {
            const int32_t * ty = (const int32_t *) gguf_get_arr_data(ctx, tt);
            for (size_t i = 0; i < nv && i < gguf_get_arr_n(ctx, tt); ++i) {
                if (ty[i] == 3) {
                    added_.emplace(tokens_[i], (int32_t) i);
                    added_max_len_ = std::max(added_max_len_, tokens_[i].size());
                    is_added_[i] = 1;
                }
            }
        }
    }

    auto gid = [&](const char * key, int32_t def) -> int32_t {
        const int64_t i = gguf_find_key(ctx, key);
        if (i < 0) return def;
        if (gguf_get_kv_type(ctx, i) == GGUF_TYPE_UINT32) return (int32_t) gguf_get_val_u32(ctx, i);
        if (gguf_get_kv_type(ctx, i) == GGUF_TYPE_INT32)  return gguf_get_val_i32(ctx, i);
        return def;
    };
    bos_ = gid("tokenizer.ggml.bos_token_id", bos_);
    eos_ = gid("tokenizer.ggml.eos_token_id", eos_);
    pad_ = gid("tokenizer.ggml.padding_token_id", pad_);
    unk_ = gid("tokenizer.ggml.unknown_token_id", unk_);
    return true;
}

// Lowest-rank-first merge over a doubly linked list of symbols. Same shape as
// llama.cpp's BPE and HF `tokenizers`' binary-heap merge_word, so the merge
// ORDER matches: ties break on left-most position.
void Tokenizer::bpe_chunk(const std::string & chunk, std::vector<int32_t> & out) const {
    if (chunk.empty()) return;

    struct sym { int prev, next; std::string text; };
    std::vector<sym> syms;
    syms.reserve(chunk.size());

    for (size_t i = 0; i < chunk.size();) {
        const size_t l = std::min(utf8_len((unsigned char) chunk[i]), chunk.size() - i);
        std::string ch = chunk.substr(i, l);
        i += l;
        if (vocab_.count(ch)) {
            syms.push_back({(int) syms.size() - 1, (int) syms.size() + 1, std::move(ch)});
        } else {
            // byte_fallback: one <0xXX> symbol per UTF-8 byte
            for (unsigned char b : ch) {
                char buf[8];
                snprintf(buf, sizeof(buf), "<0x%02X>", b);
                syms.push_back({(int) syms.size() - 1, (int) syms.size() + 1, buf});
            }
        }
    }
    if (syms.empty()) return;
    syms.back().next = -1;

    struct cand {
        int left, right, rank; size_t len;
        bool operator<(const cand & o) const {
            return rank != o.rank ? rank > o.rank : left > o.left;  // min-heap
        }
    };
    std::priority_queue<cand> work;
    auto try_pair = [&](int l, int r) {
        if (l < 0 || r < 0) return;
        auto it = merges_.find(syms[l].text + " " + syms[r].text);
        if (it == merges_.end()) return;
        work.push({l, r, it->second, syms[l].text.size() + syms[r].text.size()});
    };
    for (int i = 0; i + 1 < (int) syms.size(); ++i) try_pair(i, i + 1);

    while (!work.empty()) {
        const cand c = work.top(); work.pop();
        sym & L = syms[c.left];
        sym & R = syms[c.right];
        // Stale entry: one side was already consumed or re-merged.
        if (L.text.empty() || R.text.empty()) continue;
        if (L.next != c.right) continue;
        if (L.text.size() + R.text.size() != c.len) continue;

        L.text += R.text;
        R.text.clear();
        L.next = R.next;
        if (R.next >= 0) syms[R.next].prev = c.left;
        try_pair(L.prev, c.left);
        try_pair(c.left, L.next);
    }

    for (int i = 0; i >= 0 && i < (int) syms.size(); i = syms[i].next) {
        if (syms[i].text.empty()) continue;
        auto it = vocab_.find(syms[i].text);
        out.push_back(it != vocab_.end() ? it->second : unk_);
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string & text, bool add_special) const {
    std::vector<int32_t> out;
    if (add_special) out.push_back(bos_);

    std::string pending;   // raw (un-normalised) run between added tokens
    auto flush = [&]() {
        if (pending.empty()) return;
        // Normaliser: Replace(" " -> U+2581). Applied per non-added chunk.
        std::string norm;
        norm.reserve(pending.size() + pending.size() / 4);
        for (char c : pending) {
            if (c == ' ') norm += METASPACE; else norm += c;
        }
        bpe_chunk(norm, out);
        pending.clear();
    };

    for (size_t i = 0; i < text.size();) {
        // Leftmost-longest added-token match on the RAW bytes.
        int32_t hit_id = -1;
        size_t  hit_len = 0;
        const size_t maxl = std::min(added_max_len_, text.size() - i);
        for (size_t l = maxl; l >= 1; --l) {
            auto it = added_.find(text.substr(i, l));
            if (it != added_.end()) { hit_id = it->second; hit_len = l; break; }
        }
        if (hit_id >= 0) {
            flush();
            out.push_back(hit_id);
            i += hit_len;
        } else {
            pending += text[i];
            ++i;
        }
    }
    flush();
    return out;
}

std::string Tokenizer::token_text(int32_t id) const {
    if (id < 0 || id >= (int32_t) tokens_.size()) return {};
    return tokens_[id];
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids, bool skip_special) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || id >= (int32_t) tokens_.size()) continue;
        if (skip_special && is_added_[id]) continue;
        const std::string & t = tokens_[id];
        // ByteFallback decoder: <0xXX> -> the raw byte.
        if (t.size() == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x' && t[5] == '>') {
            const auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                return -1;
            };
            const int hi = hex(t[3]), lo = hex(t[4]);
            if (hi >= 0 && lo >= 0) { out += (char) (hi * 16 + lo); continue; }
        }
        // Replace decoder: U+2581 -> ' '.
        for (size_t i = 0; i < t.size();) {
            if (t.compare(i, 3, METASPACE) == 0) { out += ' '; i += 3; }
            else { out += t[i]; ++i; }
        }
    }
    return out;
}

} // namespace breeze
