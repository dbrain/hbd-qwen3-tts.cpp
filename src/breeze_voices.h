#pragma once

// Breeze voice archive: the on-disk library behind /v1/audio/voices.
//
// A voice is the Mimi codes [T,16] that condition the clone path, stored as a
// plain float32 C-order .npy (so numpy can read them) plus optional
// `<id>.reftext` and `<id>.wav` sidecars. Deliberately NOT higgs::VoiceStore:
// that one clamps codes to 0..1023 for XCodec2's 8 codebooks, which would
// silently truncate half of Mimi's 0..2047 range.
//
// The store is pure filesystem — no model, no GPU — so list/delete/from-codes
// keep working while the worker is idle-unloaded.

#include <cstdint>
#include <string>
#include <vector>

namespace breeze {

struct VoiceInfo {
    std::string id;
    int  T = 0;
    int  N = 0;
    bool has_ref_text = false;
    bool has_sample = false;
};

class VoiceStore {
public:
    explicit VoiceStore(const std::string & dir);
    const std::string & dir() const { return dir_; }

    std::vector<VoiceInfo> list() const;
    bool exists(const std::string & id) const;

    bool save(const std::string & id, const int32_t * codes_TN, int T, int N,
              const std::string & ref_text, int code_max, std::string & err);
    bool load(const std::string & id, std::vector<int32_t> & codes_TN,
              int & T, int & N, std::string & ref_text) const;
    bool remove(const std::string & id, std::string & err);

    bool save_wav(const std::string & id, const std::string & wav_bytes, std::string & err);
    bool load_wav(const std::string & id, std::string & wav_bytes) const;
    bool load_ref_text(const std::string & id, std::string & ref_text) const;

    static std::string sanitize(const std::string & name);

private:
    std::string dir_;
    std::string npy_path(const std::string & id) const;
    std::string txt_path(const std::string & id) const;
    std::string wav_path(const std::string & id) const;
};

} // namespace breeze
