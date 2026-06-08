#pragma once

// Higgs-Audio-v3 voice store: a persistent library of "voices" on disk. A voice
// is the codes [T,N] (0..1023) that condition the clone path — saved as a
// standard float32 .npy (interoperable with higgs_synth --save-codes/--codes and
// numpy) plus an optional reference-transcript sidecar. This is the engine-side
// backing for the server's /v1/audio/voices API (list/save/delete) and the
// `voice` param on /v1/audio/speech. No model or encoder needed to manage voices
// — a voice baked from a zero-shot trial (save its OUTPUT codes) needs nothing
// but this store.

#include <string>
#include <vector>
#include <cstdint>

namespace higgs {

struct VoiceInfo {
    std::string id;
    int  T = 0;            // frames
    int  N = 0;            // codebooks
    bool has_ref_text = false;
};

class VoiceStore {
public:
    explicit VoiceStore(const std::string & dir);
    const std::string & dir() const { return dir_; }

    std::vector<VoiceInfo> list() const;          // sorted by id
    bool exists(const std::string & id) const;

    // Persist codes [T*N] (row-major, c-fastest) as voice `id`. Values are clipped
    // to 0..1023 on write. ref_text (may be empty) is stored alongside. Returns
    // false + err on I/O failure.
    bool save(const std::string & id, const int32_t * codes_TN, int T, int N,
              const std::string & ref_text, std::string & err);

    // Load voice `id` -> codes_TN [T*N] + T/N + ref_text. False if missing/bad.
    bool load(const std::string & id, std::vector<int32_t> & codes_TN,
              int & T, int & N, std::string & ref_text) const;

    bool remove(const std::string & id, std::string & err);

    // Optional original-clip sidecar (`<id>.wav`), stored on wav-upload so the
    // server can serve a reference sample GPU-free (parity with qwen3's
    // /voices/{id}/sample.wav). Code-only voices (trial-and-save / from-codes)
    // have none → load_wav returns false (404).
    bool save_wav(const std::string & id, const std::string & wav_bytes, std::string & err);
    bool load_wav(const std::string & id, std::string & wav_bytes) const;

    // Read the ref_text sidecar (`<id>.reftext`). False if none.
    bool load_ref_text(const std::string & id, std::string & ref_text) const;

    // Map an arbitrary display name to a filesystem-safe voice id ([A-Za-z0-9._-],
    // others -> '_'; collapses repeats; trims). Empty -> "voice".
    static std::string sanitize(const std::string & name);

private:
    std::string dir_;
    std::string npy_path(const std::string & id) const;
    std::string txt_path(const std::string & id) const;
    std::string wav_path(const std::string & id) const;
};

} // namespace higgs
