// UTF-8 token helpers matching the Python package.
#pragma once

#include <string>
#include <vector>

namespace fusealign {

// Python str.split() with no argument: split on runs of Unicode whitespace (str.isspace), drop empty pieces.
std::vector<std::string> split_whitespace(const std::string & utf8);

// True if any code point of the token is Unicode whitespace (Python str.isspace).
bool has_whitespace(const std::string & utf8);

// fusealign_hub.aligner._is_wordy: any code point whose Unicode general category starts with L or N
// (Unicode 15.0, Python 3.12's unicodedata).
bool is_wordy(const std::string & utf8);

// Throws std::invalid_argument if the string is not valid UTF-8.
void check_utf8(const std::string & utf8);

// The code points of a UTF-8 string; throws std::invalid_argument as check_utf8.
std::u32string decode_utf8(const std::string & utf8);

// The [0-9a-z] character re.sub(r"[^0-9a-z]", "", chr(cp).lower()) keeps for one code point, or '\0'.
char ascii_key(char32_t cp);

}  // namespace fusealign
