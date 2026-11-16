#include "text.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

#include "unicode-tables.h"

namespace fusealign {

namespace {

// Strict UTF-8 (what Python's decoder accepts): no overlong forms, no surrogates, nothing above U+10FFFF.
// Decodes the code point at s[i] and advances i; throws std::invalid_argument on malformed input.
char32_t next_code_point(const std::string & s, size_t & i) {
    const auto byte = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char b0 = byte(i);
    if (b0 < 0x80) {
        ++i;
        return b0;
    }
    size_t n;
    char32_t cp;
    unsigned char lo = 0x80, hi = 0xBF;  // allowed range of the second byte
    if (b0 >= 0xC2 && b0 <= 0xDF) {
        n = 1;
        cp = b0 & 0x1F;
    } else if (b0 >= 0xE0 && b0 <= 0xEF) {
        n = 2;
        cp = b0 & 0x0F;
        if (b0 == 0xE0) lo = 0xA0;  // overlong
        if (b0 == 0xED) hi = 0x9F;  // surrogates
    } else if (b0 >= 0xF0 && b0 <= 0xF4) {
        n = 3;
        cp = b0 & 0x07;
        if (b0 == 0xF0) lo = 0x90;  // overlong
        if (b0 == 0xF4) hi = 0x8F;  // above U+10FFFF
    } else {
        throw std::invalid_argument("invalid UTF-8: bad lead byte at offset " + std::to_string(i));
    }
    for (size_t k = 1; k <= n; ++k) {
        if (i + k >= s.size()) {
            throw std::invalid_argument("invalid UTF-8: truncated sequence at offset " + std::to_string(i));
        }
        const unsigned char b = byte(i + k);
        const unsigned char min = k == 1 ? lo : 0x80, max = k == 1 ? hi : 0xBF;
        if (b < min || b > max) {
            throw std::invalid_argument("invalid UTF-8: bad continuation byte at offset " + std::to_string(i + k));
        }
        cp = (cp << 6) | (b & 0x3F);
    }
    i += n + 1;
    return cp;
}

template <size_t N>
bool in_ranges(const unicode::CodepointRange (&table)[N], char32_t cp) {
    const auto it = std::upper_bound(std::begin(table), std::end(table), static_cast<uint32_t>(cp),
                                     [](uint32_t value, const unicode::CodepointRange & r) { return value < r.lo; });
    return it != std::begin(table) && cp <= std::prev(it)->hi;
}

bool is_space(char32_t cp) { return in_ranges(unicode::SPACE_RANGES, cp); }

}  // namespace

std::u32string decode_utf8(const std::string & utf8) {
    std::u32string out;
    out.reserve(utf8.size());
    for (size_t i = 0; i < utf8.size();) {
        out.push_back(next_code_point(utf8, i));
    }
    return out;
}

void check_utf8(const std::string & utf8) {
    for (size_t i = 0; i < utf8.size();) {
        next_code_point(utf8, i);
    }
}

std::vector<std::string> split_whitespace(const std::string & utf8) {
    std::vector<std::string> out;
    size_t piece = std::string::npos;  // start of the current piece
    for (size_t i = 0; i < utf8.size();) {
        const size_t at = i;
        if (is_space(next_code_point(utf8, i))) {
            if (piece != std::string::npos) {
                out.push_back(utf8.substr(piece, at - piece));
                piece = std::string::npos;
            }
        } else if (piece == std::string::npos) {
            piece = at;
        }
    }
    if (piece != std::string::npos) {
        out.push_back(utf8.substr(piece));
    }
    return out;
}

bool has_whitespace(const std::string & utf8) {
    bool found = false;
    for (size_t i = 0; i < utf8.size();) {
        found = is_space(next_code_point(utf8, i)) || found;
    }
    return found;
}

bool is_wordy(const std::string & utf8) {
    bool found = false;
    for (size_t i = 0; i < utf8.size();) {
        found = in_ranges(unicode::WORDY_RANGES, next_code_point(utf8, i)) || found;
    }
    return found;
}

char ascii_key(char32_t cp) {
    const auto it = std::lower_bound(std::begin(unicode::ASCII_KEYS), std::end(unicode::ASCII_KEYS),
                                     static_cast<uint32_t>(cp),
                                     [](const unicode::AsciiKey & k, uint32_t value) { return k.cp < value; });
    return it != std::end(unicode::ASCII_KEYS) && it->cp == cp ? it->key : '\0';
}

}  // namespace fusealign
