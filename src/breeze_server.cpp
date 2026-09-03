// OpenAI-ish HTTP server for the Breeze-TTS-2 C++ engine.
//
// Deliberately the SAME wire shape as qwen3-tts-server / higgs-server so
// kobbler's TtsBackend needs no new client: same routes, same SSE event names,
// same voice-archive semantics, same /v1/admin/{load,unload} + /v1/gpu/status
// contract for koblem's GPU gate.
//
//   breeze-server --model breeze.gguf [-H 0.0.0.0] [-p 8203] [--n-ctx 2048]
//                 [--voices-dir DIR] [--aligner-model FA.gguf]
//
// Worker isolation (BREEZE_WORKER_ISOLATION=1): the parent process is CUDA-free
// (HTTP + filesystem VoiceStore only) and forks a child that owns the engine.
// The child is SIGKILLed after BREEZE_IDLE_UNLOAD_SECONDS of inactivity → VRAM
// true-0; the next request respawns and reloads.
//
// Endpoints
//   GET    /health                          -> {"status":"ok","model":..,"loaded":bool}
//   GET    /v1/gpu/status                   -> {"loaded":bool,"gpu":uuid|null}
//   GET    /v1/models                       -> model list
//   GET    /v1/audio/voices                 -> {"model_id":..,"voices":[..]}
//   POST   /v1/audio/voices                 -> clone from wav (multipart) | from codes
//   GET    /v1/audio/voices/<id>/sample.wav -> the reference clip
//   GET    /v1/audio/voices/<id>/ref_text   -> the reference transcript
//   DELETE /v1/audio/voices/<id>            -> {}
//   POST   /v1/audio/speech                 -> wav|pcm|mp3 bytes, or SSE stream
//   POST   /v1/admin/load | /v1/admin/unload

#include "audio/ffmpeg_encode.h"
#include "breeze_tts.h"
#include "breeze_voices.h"
#include "breeze_worker_session.h"
#include "fa_session.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
using namespace std::chrono_literals;

static const char * argval(int argc, char ** argv, const char * key, const char * def) {
    for (int i = 1; i < argc - 1; ++i) if (!strcmp(argv[i], key)) return argv[i + 1];
    return def;
}

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string base64_encode(const void * data, size_t len) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const uint8_t * p = (const uint8_t *) data;
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = (uint32_t) p[i] << 16;
        if (i + 1 < len) n |= (uint32_t) p[i + 1] << 8;
        if (i + 2 < len) n |= p[i + 2];
        out.push_back(tbl[(n >> 18) & 63]);
        out.push_back(tbl[(n >> 12) & 63]);
        out.push_back((i + 1 < len) ? tbl[(n >> 6) & 63] : '=');
        out.push_back((i + 2 < len) ? tbl[n & 63] : '=');
    }
    return out;
}

static std::string pcm_f32_to_s16le(const float * pcm, size_t n) {
    std::string b; b.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        int v = (int) lrintf(pcm[i] * 32767.0f);
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        const int16_t x = (int16_t) v;
        b.append((const char *) &x, 2);
    }
    return b;
}

// Correct RIFF/data sizes. qwen3-tts-server writes 0xFFFFFFFF placeholders here
// and strict parsers (python `wave`) then report ~89478 s of audio; do not
// reproduce that bug.
static std::string wav_bytes(const std::vector<float> & pcm, int sr) {
    std::string b;
    auto put = [&](const void * p, size_t n) { b.append((const char *) p, n); };
    auto u32 = [&](uint32_t v) { put(&v, 4); };
    auto u16 = [&](uint16_t v) { put(&v, 2); };
    const uint32_t n = (uint32_t) pcm.size(), db = n * 2;
    put("RIFF", 4); u32(36 + db); put("WAVE", 4);
    put("fmt ", 4); u32(16); u16(1); u16(1); u32(sr); u32(sr * 2); u16(2); u16(16);
    put("data", 4); u32(db);
    b.reserve(b.size() + db);
    b += pcm_f32_to_s16le(pcm.data(), pcm.size());
    return b;
}

static bool wav_decode(const std::string & b, std::vector<float> & out, int & sr) {
    auto rd_u32 = [&](size_t o) -> uint32_t {
        return (uint8_t) b[o] | ((uint32_t)(uint8_t) b[o+1] << 8) |
               ((uint32_t)(uint8_t) b[o+2] << 16) | ((uint32_t)(uint8_t) b[o+3] << 24); };
    auto rd_u16 = [&](size_t o) -> uint16_t {
        return (uint16_t) ((uint8_t) b[o] | ((uint8_t) b[o+1] << 8)); };
    if (b.size() < 44 || b.compare(0, 4, "RIFF") || b.compare(8, 4, "WAVE")) return false;
    uint16_t fmt = 1, ch = 1, bits = 16; uint32_t rate = 24000;
    size_t data_off = 0, data_len = 0, p = 12;
    while (p + 8 <= b.size()) {
        const std::string cid = b.substr(p, 4);
        const uint32_t csz = rd_u32(p + 4);
        const size_t body = p + 8;
        if (cid == "fmt " && body + 16 <= b.size()) {
            fmt = rd_u16(body); ch = rd_u16(body + 2); rate = rd_u32(body + 4); bits = rd_u16(body + 14);
        } else if (cid == "data") {
            data_off = body; data_len = std::min((size_t) csz, b.size() - body); break;
        }
        p = body + csz + (csz & 1);
    }
    if (!data_off || ch == 0) return false;
    sr = (int) rate;
    const char * d = b.data() + data_off;
    if (fmt == 1 && bits == 16) {
        const size_t frames = (data_len / 2) / ch;
        out.resize(frames);
        for (size_t f = 0; f < frames; ++f) {
            int acc = 0;
            for (int c = 0; c < ch; ++c) {
                const int16_t s = (int16_t) ((uint8_t) d[(f*ch+c)*2] | ((uint8_t) d[(f*ch+c)*2+1] << 8));
                acc += s;
            }
            out[f] = (float) acc / ch / 32768.0f;
        }
        return true;
    }
    if ((fmt == 3 || fmt == 0xFFFE) && bits == 32) {
        const size_t frames = (data_len / 4) / ch;
        out.resize(frames);
        for (size_t f = 0; f < frames; ++f) {
            float acc = 0;
            for (int c = 0; c < ch; ++c) { float s; std::memcpy(&s, d + (f*ch+c)*4, 4); acc += s; }
            out[f] = acc / ch;
        }
        return true;
    }
    return false;
}

// Linear resample to 24 kHz. Uploaded clips are whatever the browser recorded.
static void resample_24k(std::vector<float> & pcm, int sr) {
    if (sr == 24000 || pcm.empty()) return;
    const double ratio = 24000.0 / sr;
    std::vector<float> out((size_t) (pcm.size() * ratio));
    for (size_t i = 0; i < out.size(); ++i) {
        const double s = i / ratio;
        const size_t i0 = (size_t) s, i1 = std::min(i0 + 1, pcm.size() - 1);
        const double t = s - i0;
        out[i] = (float) (pcm[i0] * (1 - t) + pcm[i1] * t);
    }
    pcm.swap(out);
}

// Mirrors kobbler's str::split_whitespace(), which drives the read-along
// word_index mapping: split on ASCII whitespace, punctuation stays glued.
static std::vector<std::string> whitespace_split_for_align(const std::string & s) {
    std::vector<std::string> out;
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        while (i < n && (unsigned char) s[i] <= ' ') i++;
        const size_t start = i;
        while (i < n && (unsigned char) s[i] > ' ') i++;
        if (i > start) out.push_back(s.substr(start, i - start));
    }
    return out;
}

static void fill_params(const json & body, breeze::gen_params & gp) {
    gp.temperature        = body.value("temperature", 0.9f);
    gp.top_k              = body.value("top_k", 50);
    gp.top_p              = body.value("top_p", 1.0f);
    gp.repetition_penalty = body.value("repetition_penalty", 1.1f);
    gp.depth_temperature  = body.value("depth_temperature", gp.temperature);
    gp.depth_top_k        = body.value("depth_top_k", gp.top_k);
    gp.depth_top_p        = body.value("depth_top_p", gp.top_p);
    gp.seed               = body.value("seed", 0u);
    // Breeze counts codec FRAMES (12.5 Hz), not tokens; accept the qwen3-side
    // knob names kobbler already sends and convert.
    gp.max_new_frames     = body.value("max_new_frames",
                              body.value("max_new_tokens", body.value("max_audio_tokens", 750)));
    gp.speaker            = body.value("speaker", std::string{"S0"});
    // kobbler and koblem both send "instructions" (plural) -- qwen3-tts's
    // spelling. Accepting only the singular silently dropped every voice-design
    // string with a 200 OK, which is the worst kind of incompatibility.
    gp.instruction        = body.value("instruction",
                              body.value("instructions",
                                body.value("voice_description", std::string{})));
}

struct ServerCtx {
    std::mutex               mtx;
    breeze::VoiceStore *     voices = nullptr;
    const char *             model_id = "breeze-tts-2";
    int                      code_max = 2047;
    std::atomic<int64_t>     last_activity_ms{0};
    std::atomic<int>         inflight{0};
    std::function<bool()>    ensure;
    std::function<bool()>    is_loaded;
    std::function<std::string()> worker_gpu;
    std::function<void()>    unload;
    std::function<void(const std::string &)> set_gpu;

    fa::AlignerSession *     aligner = nullptr;
    std::string              aligner_model;
    std::mutex               aligner_mtx;
    std::atomic<int64_t>     aligner_last_activity_ms{0};
    std::atomic<int>         aligner_inflight{0};
};

// Bumps inflight around a GPU request so the idle-unload watchdog can never
// SIGKILL the worker mid-synth.
struct Activity {
    ServerCtx & cx;
    explicit Activity(ServerCtx & c) : cx(c) { cx.inflight.fetch_add(1); cx.last_activity_ms.store(now_ms()); }
    ~Activity() { cx.last_activity_ms.store(now_ms()); cx.inflight.fetch_sub(1); }
};

template <class ENG>
static void install_routes(httplib::Server & srv, ENG * eng, ServerCtx & cx) {
    auto err_json = [](httplib::Response & res, int code, const std::string & msg) {
        res.status = code;
        res.set_content(json({{"error", msg}}).dump(), "application/json");
    };

    srv.Get("/health", [&cx](const httplib::Request &, httplib::Response & res) {
        res.set_content(json({{"status","ok"},{"model",cx.model_id},
                              {"loaded", cx.is_loaded()}}).dump(), "application/json");
    });
    srv.Get("/v1/gpu/status", [&cx](const httplib::Request &, httplib::Response & res) {
        const bool loaded = cx.is_loaded();
        const std::string gpu = (loaded && cx.worker_gpu) ? cx.worker_gpu() : std::string();
        json body = {{"loaded", loaded}};
        if (!gpu.empty()) body["gpu"] = gpu; else body["gpu"] = nullptr;
        res.set_content(body.dump(), "application/json");
    });
    srv.Get("/v1/models", [&cx](const httplib::Request &, httplib::Response & res) {
        res.set_content(json({{"object","list"},
                              {"data", {{{"id", cx.model_id},{"object","model"}}}}}).dump(),
                        "application/json");
    });

    // GET /v1/audio/languages — qwen3-tts parity. Breeze is bilingual by
    // design ("Generates natural English and Chinese speech with a single
    // model"), and unlike qwen3-tts it takes no language_id: the id column is
    // reported as null so a client cannot mistake it for a selectable knob.
    // The language is inferred from the text, and the vocal-event syntax
    // differs per language — (laugh) in English, [笑] in Chinese.
    srv.Get("/v1/audio/languages", [](const httplib::Request &, httplib::Response & res) {
        json langs = json::array();
        for (const auto & [code, name] : std::vector<std::pair<std::string, std::string>>{
                 {"en", "English"}, {"zh", "Chinese"}}) {
            langs.push_back({{"code", code}, {"id", nullptr}, {"name", name}});
        }
        res.set_content(json({{"languages", langs}}).dump(), "application/json");
    });

    // ---- voice library (filesystem; GPU-free — works while the worker is unloaded) ----
    srv.Get("/v1/audio/voices", [&cx](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(cx.mtx);
        json arr = json::array();
        for (const auto & v : cx.voices->list())
            arr.push_back({{"id",v.id},{"frames",v.T},{"codebooks",v.N},
                           {"has_ref_text",v.has_ref_text},{"has_sample",v.has_sample}});
        // Two shapes in one body. qwen3-tts answers `{"<model_id>": ["default",
        // "alice", ...]}` and kobbler's parse_voices_body expects exactly that,
        // so emit it under the model-id key for drop-in compatibility while
        // keeping our richer {id, frames, codebooks, ...} objects under
        // "voices". A client that understands either shape works unchanged.
        json flat = json::array({"default"});
        for (const auto & v : cx.voices->list()) flat.push_back(v.id);
        json body_out = {{"model_id", cx.model_id}, {"voices", arr}};
        body_out[cx.model_id] = flat;
        res.set_content(body_out.dump(), "application/json");
    });

    srv.Post("/v1/audio/voices", [&cx, eng, err_json](const httplib::Request & req, httplib::Response & res) {
        if (req.is_multipart_form_data() || req.has_file("audio_sample") || req.has_file("voice_file")) {
            std::string name = req.has_file("name") ? req.get_file_value("name").content
                             : req.has_file("voice_name") ? req.get_file_value("voice_name").content : "";
            if (name.empty()) return err_json(res, 400, "missing name");
            std::string wav = req.has_file("audio_sample") ? req.get_file_value("audio_sample").content
                            : req.has_file("voice_file") ? req.get_file_value("voice_file").content : "";
            if (wav.empty()) return err_json(res, 400, "missing audio_sample");
            const std::string ref_text = req.has_param("ref_text") ? req.get_param_value("ref_text")
                                       : req.has_file("ref_text") ? req.get_file_value("ref_text").content : "";
            const std::string id = breeze::VoiceStore::sanitize(name);

            std::vector<float> samples; int sr = 24000;
            if (!wav_decode(wav, samples, sr) || samples.empty())
                return err_json(res, 400, "could not decode WAV (need PCM s16/f32)");
            resample_24k(samples, sr);

            Activity act(cx);
            if (!cx.ensure()) return err_json(res, 503, "engine load failed");
            std::vector<int32_t> codes; int T = 0; std::string e; bool ok;
            {
                std::lock_guard<std::mutex> lk(cx.mtx);
                ok = eng->encode_voice(samples.data(), (int) samples.size(), codes, T);
                if (ok) {
                    const int N = T > 0 ? (int) (codes.size() / T) : 0;
                    ok = cx.voices->save(id, codes.data(), T, N, ref_text, cx.code_max, e);
                    if (ok) cx.voices->save_wav(id, wav, e);
                }
            }
            if (!ok) return err_json(res, 500, e.empty() ? eng->get_error() : e);
            res.set_content(json({{"id",id},{"name",id},{"frames",T},
                                  {"codebooks", T > 0 ? (int)(codes.size()/T) : 0},
                                  {"mode","clone"},{"ref_frames",T}}).dump(), "application/json");
            return;
        }

        json body;
        try { body = json::parse(req.body); } catch (...) { return err_json(res, 400, "bad json"); }
        const std::string name = body.value("name", std::string());
        if (name.empty()) return err_json(res, 400, "missing name");
        const std::string id = breeze::VoiceStore::sanitize(name);
        const std::string ref_text = body.value("ref_text", std::string());
        if (!body.contains("codes_flat")) return err_json(res, 400, "need codes_flat or a wav upload");
        auto flat = body["codes_flat"].get<std::vector<int>>();
        const int N = body.value("n_codebooks", 16);
        if (N <= 0 || (int) flat.size() % N != 0)
            return err_json(res, 400, "codes_flat not divisible by n_codebooks");
        const int T = (int) flat.size() / N;
        std::vector<int32_t> codes(flat.begin(), flat.end());
        std::lock_guard<std::mutex> lk(cx.mtx);
        std::string e;
        if (!cx.voices->save(id, codes.data(), T, N, ref_text, cx.code_max, e))
            return err_json(res, 500, e);
        res.set_content(json({{"id",id},{"frames",T},{"codebooks",N}}).dump(), "application/json");
    });

    srv.Get(R"(/v1/audio/voices/([^/]+)/sample\.wav)",
            [&cx, err_json](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        std::lock_guard<std::mutex> lk(cx.mtx);
        std::string wav;
        if (!cx.voices->load_wav(id, wav)) return err_json(res, 404, "no sample for voice: " + id);
        res.set_content(wav, "audio/wav");
    });
    srv.Get(R"(/v1/audio/voices/([^/]+)/ref_text)",
            [&cx, err_json](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        std::lock_guard<std::mutex> lk(cx.mtx);
        std::string rt;
        if (!cx.voices->load_ref_text(id, rt)) return err_json(res, 404, "no ref_text for voice: " + id);
        res.set_content(rt, "text/plain; charset=utf-8");
    });
    srv.Delete(R"(/v1/audio/voices/([^/]+))",
               [&cx, err_json](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        std::lock_guard<std::mutex> lk(cx.mtx);
        std::string e;
        if (!cx.voices->remove(id, e)) return err_json(res, 404, e);
        res.set_content("{}", "application/json");
    });

    // ---- standalone forced alignment ----
    // Align an EXISTING clip against known text, with no synthesis. The SSE
    // speech path can only align audio it just generated, and the TTS sampler
    // is not bit-reproducible across runs, so aligner A/B work there compares
    // two different waveforms. This route pins the audio so aligner changes are
    // the only variable; it is also what a reader client wants when it already
    // has the audio cached. Multipart: file=<wav>, text=<transcript>.
    srv.Post("/v1/audio/alignment", [&cx, err_json](const httplib::Request & req, httplib::Response & res) {
        if (!cx.aligner || cx.aligner_model.empty())
            return err_json(res, 400, "no aligner: server started without --aligner-model");
        const std::string wav = req.has_file("file") ? req.get_file_value("file").content : std::string();
        const std::string text = req.has_file("text") ? req.get_file_value("text").content
                               : req.has_param("text") ? req.get_param_value("text") : std::string();
        if (wav.empty()) return err_json(res, 400, "missing file (wav)");
        if (text.empty()) return err_json(res, 400, "missing text");
        std::vector<float> pcm; int sr = 24000;
        if (!wav_decode(wav, pcm, sr) || pcm.empty())
            return err_json(res, 400, "could not decode WAV (need PCM s16/f32)");
        const std::vector<std::string> words = whitespace_split_for_align(text);
        if (words.empty()) return err_json(res, 400, "no words in text");

        std::lock_guard<std::mutex> lk(cx.aligner_mtx);
        cx.aligner_inflight.fetch_add(1);
        cx.aligner_last_activity_ms.store(now_ms());
        std::vector<fa::AlignedWord> aligned;
        fa::AlignProfile prof;
        bool ok = cx.aligner->ensure_loaded(cx.aligner_model)
               && cx.aligner->begin_streaming_align(words, sr)
               && cx.aligner->finalize_streaming_align(pcm.data(), pcm.size(),
                                                       (int64_t) pcm.size() * 1000 / (sr > 0 ? sr : 1),
                                                       aligned, prof);
        const std::string err = ok ? std::string() : cx.aligner->last_error();
        cx.aligner_inflight.fetch_sub(1);
        cx.aligner_last_activity_ms.store(now_ms());
        if (!ok) return err_json(res, 500, err.empty() ? "alignment failed" : err);
        json wj = json::array();
        for (size_t i = 0; i < aligned.size(); i++)
            wj.push_back({{"word_index",(int)i},{"text",aligned[i].text},
                          {"t0_ms",aligned[i].t0_ms},{"t1_ms",aligned[i].t1_ms},
                          {"confidence",aligned[i].confidence}});
        res.set_content(json({{"audio_total_ms", (int64_t) pcm.size() * 1000 / (sr > 0 ? sr : 1)},
                              {"sample_rate", sr},
                              {"words", std::move(wj)},
                              {"profile", {{"t_load_ms",prof.t_load_ms},{"t_resample_ms",prof.t_resample_ms},
                                           {"t_aligner_ms",prof.t_aligner_ms},{"t_total_ms",prof.t_total_ms},
                                           {"n_words",prof.n_words}}}}).dump(), "application/json");
    });

    srv.Post("/v1/admin/unload", [&cx](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(cx.mtx);
        const bool was = cx.is_loaded();
        cx.unload();
        res.set_content(json({{"unloaded", was},{"model_loaded", cx.is_loaded()}}).dump(), "application/json");
    });
    srv.Post("/v1/admin/load", [&cx](const httplib::Request &, httplib::Response & res) {
        Activity act(cx);
        const bool ok = cx.ensure();
        res.set_content(json({{"model_loaded", ok && cx.is_loaded()}}).dump(), "application/json");
    });

    // ---- speech ----
    srv.Post("/v1/audio/speech", [&cx, eng, err_json](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { return err_json(res, 400, "bad json"); }
        const std::string input = body.value("input", body.value("text", std::string()));
        if (input.empty()) return err_json(res, 400, "empty input");

        breeze::gen_params gp;
        fill_params(body, gp);
        if (cx.set_gpu) cx.set_gpu(body.value("gpu", std::string()));
        Activity act(cx);
        if (!cx.ensure()) return err_json(res, 503, "engine load failed");

        // Resolve an optional clone voice from the filesystem store (GPU-free).
        breeze::ref_voice ref;
        {
            const std::string voice = body.value("voice", std::string());
            if (!voice.empty() && voice != "default") {
                int N = 0;
                if (!cx.voices->load(voice, ref.codes, ref.T, N, ref.ref_text))
                    return err_json(res, 400, "unknown voice: " + voice);
                ref.ref_text = body.value("ref_text", ref.ref_text);
            }
        }
        const bool have_voice = ref.T > 0;

        // Alignment is only produced on the SSE path, interleaved with the audio
        // deltas — a buffered response has nowhere to put it. Silently dropping
        // the flag made a caller think alignment was on when no aligner ever ran,
        // so refuse instead.
        if (body.value("align", false)
            && body.value("stream_format", std::string()) != "sse") {
            return err_json(res, 400,
                            "align requires stream_format=\"sse\"; word timings are delivered as "
                            "interleaved speech.audio.alignment.* events, not in a buffered body");
        }

        // ---- SSE streaming + optional interleaved forced alignment ----
        // The path kobbler's BookReader takes: stream_format="sse",
        // response_format="pcm", align=true, align_stream="partial".
        if (body.value("stream_format", std::string()) == "sse") {
            const bool do_align      = body.value("align", false);
            const std::string a_mode = body.value("align_stream", std::string("final-only"));
            // align_stream only chooses whether the INCREMENTAL events reach the
            // client; it used to gate the aligner itself, so the documented
            // default ("final-only") ran no aligner and emitted nothing at all.
            // The internal partial pass still runs in final-only mode — it is
            // how PCM reaches the aligner subprocess, and an undrained
            // PARTIAL_RESP left in the socket would desync the FINAL handshake.
            const bool do_align_run  = do_align && cx.aligner && !cx.aligner_model.empty();
            const bool emit_partials = do_align_run && a_mode == "partial";
            // kobbler sends qwen3-tts's spelling, `stream_first_batch_size: 1`,
            // to ask for the smallest possible first chunk (i.e. lowest TTFA).
            // Our knob is `chunk_frames` -- same meaning, since the emit size
            // doubles from there up to a cap -- so accept both names rather than
            // silently ignoring the caller's TTFA tuning.
            const int chunk_frames   = body.value("chunk_frames",
                                         body.value("stream_first_batch_size",
                                           body.value("stream_batch_size", 6)));
            std::vector<std::string> words = whitespace_split_for_align(input);

            res.set_header("Content-Type", "text/event-stream");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider("text/event-stream",
                [eng, &cx, input, gp, do_align_run, emit_partials, have_voice, chunk_frames,
                 ref = std::move(ref), words = std::move(words)]
                (size_t, httplib::DataSink & sink) mutable -> bool {
                    Activity act_stream(cx);
                    std::mutex sink_mtx;
                    auto emit_event = [&](const char * ev, const json & j) {
                        const std::string s = std::string("event: ") + ev + "\ndata: " + j.dump() + "\n\n";
                        std::lock_guard<std::mutex> lk(sink_mtx);
                        sink.write(s.data(), s.size());
                    };

                    bool partial_active = false;
                    std::unique_lock<std::mutex> aligner_lock;
                    std::thread reader_thread;
                    std::atomic<bool> reader_stop{false};
                    std::atomic<int64_t> audio_offset_ms{0};
                    if (do_align_run) {
                        aligner_lock = std::unique_lock<std::mutex>(cx.aligner_mtx);
                        cx.aligner_inflight.fetch_add(1);
                        cx.aligner_last_activity_ms.store(now_ms());
                        if (!cx.aligner->ensure_loaded(cx.aligner_model)) {
                            emit_event("speech.audio.alignment.error",
                                       {{"type","speech.audio.alignment.error"},
                                        {"error", std::string("aligner load failed: ") + cx.aligner->last_error()}});
                            cx.aligner_inflight.fetch_sub(1);
                            aligner_lock.unlock();
                        } else if (!cx.aligner->begin_streaming_align(words, 24000)) {
                            emit_event("speech.audio.alignment.error",
                                       {{"type","speech.audio.alignment.error"},
                                        {"error", std::string("begin_streaming_align failed: ") + cx.aligner->last_error()}});
                            cx.aligner_inflight.fetch_sub(1);
                            aligner_lock.unlock();
                        } else {
                            partial_active = true;
                            reader_thread = std::thread([&]() {
                                while (!reader_stop.load(std::memory_order_relaxed)) {
                                    cx.aligner->drain_partial_alignments(
                                        [&](int64_t seen, const std::vector<fa::AlignedWord> & ws) {
                                            if (!emit_partials) return;  // final-only: drain, don't publish
                                            json wj = json::array();
                                            for (size_t i = 0; i < ws.size(); i++)
                                                wj.push_back({{"word_index",(int)i},{"text",ws[i].text},
                                                              {"t0_ms",ws[i].t0_ms},{"t1_ms",ws[i].t1_ms},
                                                              {"confidence",ws[i].confidence}});
                                            emit_event("speech.audio.alignment.partial",
                                                       {{"type","speech.audio.alignment.partial"},
                                                        {"audio_seen_ms",seen},{"words",std::move(wj)}});
                                        });
                                    std::this_thread::sleep_for(20ms);
                                }
                            });
                        }
                    }

                    // Disconnect watchdog: the worker stays warm (499-style).
                    std::atomic<bool> wd_stop{false};
                    std::thread wd([&]() {
                        while (!wd_stop.load()) {
                            if (sink.is_writable && !sink.is_writable()) { eng->request_cancel(); break; }
                            std::this_thread::sleep_for(50ms);
                        }
                    });

                    auto emit_audio = [&](const float * pcm, int n) {
                        cx.last_activity_ms.store(now_ms());
                        const std::string s16 = pcm_f32_to_s16le(pcm, (size_t) n);
                        emit_event("speech.audio.delta",
                                   {{"type","speech.audio.delta"},
                                    {"audio", base64_encode(s16.data(), s16.size())}});
                        if (partial_active) {
                            const int64_t chunk_ms = (int64_t) n * 1000 / 24000;
                            const int64_t total = audio_offset_ms.fetch_add(chunk_ms) + chunk_ms;
                            cx.aligner_last_activity_ms.store(now_ms());
                            cx.aligner->push_partial_pcm(pcm, (size_t) n, total);
                        }
                    };

                    breeze::gen_result r;
                    bool ok;
                    {
                        std::lock_guard<std::mutex> lk(cx.mtx);
                        eng->clear_cancel();
                        ok = eng->synthesize_stream(input, gp, have_voice ? &ref : nullptr, chunk_frames,
                                                    [&](const float * pcm, int n, bool) {
                                                        if (n > 0) emit_audio(pcm, n);
                                                    }, r);
                    }

                    if (partial_active) {
                        reader_stop.store(true);
                        if (reader_thread.joinable()) reader_thread.join();
                    }
                    wd_stop.store(true);
                    if (wd.joinable()) wd.join();

                    if (partial_active) {
                        const int64_t total_ms = audio_offset_ms.load();
                        std::vector<fa::AlignedWord> aligned;
                        fa::AlignProfile prof;
                        if (cx.aligner->finalize_streaming_align(nullptr, 0, total_ms, aligned, prof)) {
                            json wj = json::array();
                            for (size_t i = 0; i < aligned.size(); i++)
                                wj.push_back({{"word_index",(int)i},{"text",aligned[i].text},
                                              {"t0_ms",aligned[i].t0_ms},{"t1_ms",aligned[i].t1_ms},
                                              {"confidence",aligned[i].confidence}});
                            emit_event("speech.audio.alignment.final",
                                       {{"type","speech.audio.alignment.final"},
                                        {"audio_total_ms",total_ms},{"words",std::move(wj)}});
                        } else {
                            emit_event("speech.audio.alignment.error",
                                       {{"type","speech.audio.alignment.error"},
                                        {"error", cx.aligner->last_error()}});
                        }
                        cx.aligner_inflight.fetch_sub(1);
                        if (aligner_lock.owns_lock()) aligner_lock.unlock();
                    }

                    emit_event("speech.audio.done",
                               {{"type","speech.audio.done"},{"frames",r.T},
                                {"ttfa_ms",r.ttfa_ms},{"ok",ok}});
                    sink.done();
                    return true;
                });
            return;
        }

        // ---- chunked-WAV streaming ----
        if (body.value("stream", false)) {
            res.set_header("Content-Type", "audio/wav");
            const int chunk_frames = body.value("chunk_frames", 6);
            res.set_chunked_content_provider("audio/wav",
                [eng, &cx, input, gp, have_voice, chunk_frames, ref = std::move(ref)]
                (size_t, httplib::DataSink & sink) mutable {
                    // Streaming WAV cannot know its length up front; 0xFFFFFFFF is
                    // the conventional placeholder here (unlike the buffered path,
                    // which writes real sizes).
                    std::string h;
                    auto u32 = [&](uint32_t v) { h.append((const char *) &v, 4); };
                    auto u16 = [&](uint16_t v) { h.append((const char *) &v, 2); };
                    h.append("RIFF", 4); u32(0xFFFFFFFF); h.append("WAVE", 4);
                    h.append("fmt ", 4); u32(16); u16(1); u16(1); u32(24000); u32(48000); u16(2); u16(16);
                    h.append("data", 4); u32(0xFFFFFFFF);
                    sink.write(h.data(), h.size());

                    std::atomic<bool> wd_stop{false};
                    std::thread wd([&] {
                        while (!wd_stop.load()) {
                            if (sink.is_writable && !sink.is_writable()) { eng->request_cancel(); break; }
                            std::this_thread::sleep_for(50ms);
                        }
                    });
                    breeze::gen_result r;
                    bool ok;
                    {
                        std::lock_guard<std::mutex> lk(cx.mtx);
                        eng->clear_cancel();
                        ok = eng->synthesize_stream(input, gp, have_voice ? &ref : nullptr, chunk_frames,
                            [&](const float * pcm, int n, bool) {
                                if (n <= 0) return;
                                const std::string b = pcm_f32_to_s16le(pcm, (size_t) n);
                                sink.write(b.data(), b.size());
                            }, r);
                    }
                    wd_stop.store(true);
                    if (wd.joinable()) wd.join();
                    sink.done();
                    return ok;
                });
            return;
        }

        // ---- buffered ----
        breeze::gen_result r;
        bool ok;
        {
            std::lock_guard<std::mutex> lk(cx.mtx);
            eng->clear_cancel();
            ok = eng->synthesize(input, gp, have_voice ? &ref : nullptr, r);
        }
        if (!ok) return err_json(res, 500, eng->get_error());
        // A worker abort surfaces as an empty result, and an HTTP 200 with a
        // 44-byte WAV passes a status-code smoke test. Fail loudly instead.
        if (r.pcm.empty()) return err_json(res, 500, "engine produced no audio");

        const double secs = (double) r.pcm.size() / 24000.0;
        res.set_header("X-Audio-Seconds", std::to_string(secs));
        res.set_header("X-Frames", std::to_string(r.T));
        const double busy = (r.text_enc_ms + r.prefill_ms + r.decode_ms + r.codec_ms) / 1000.0;
        if (secs > 0) res.set_header("X-RTF", std::to_string(busy / secs));

        // kobbler's preview endpoint defaults to response_format=mp3 and then
        // labels the body audio/mpeg. Falling through to WAV shipped WAV bytes
        // under an mp3 content-type -- silent, and browsers mostly cope, which
        // is what makes it dangerous. Encode for real.
        const std::string fmt = body.value("response_format", std::string("wav"));
        if (fmt == "pcm") {
            res.set_content(pcm_f32_to_s16le(r.pcm.data(), r.pcm.size()), "audio/pcm");
        } else if (fmt == "mp3") {
            const int kbps = body.value("bitrate_kbps", 64);
            auto enc = qwen3_tts_audio::encode_one_shot(
                qwen3_tts_audio::Codec::Mp3, 24000, kbps, r.pcm.data(), r.pcm.size());
            if (enc.empty()) return err_json(res, 500, "mp3 encode failed");
            res.set_content(std::string(enc.begin(), enc.end()),
                            qwen3_tts_audio::content_type_for(qwen3_tts_audio::Codec::Mp3));
        } else {
            res.set_content(wav_bytes(r.pcm, 24000), "audio/wav");
        }
    });
}

int main(int argc, char ** argv) {
    // Worker / aligner child roles never start the HTTP server.
    for (int i = 1; i < argc - 1; ++i) {
        if (!strcmp(argv[i], "--breeze-worker")) return breeze::run_breeze_worker_loop(atoi(argv[i+1]));
        if (!strcmp(argv[i], "--fa-aligner"))    return fa::run_aligner_loop(atoi(argv[i+1]));
    }

    const char * model_env = std::getenv("BREEZE_MODEL");
    const std::string model = argval(argc, argv, "--model", model_env ? model_env : "");
    const std::string host  = argval(argc, argv, "-H", "0.0.0.0");
    const int port  = atoi(argval(argc, argv, "-p", "8203"));
    const int n_ctx = atoi(argval(argc, argv, "--n-ctx", "2048"));
    const char * vdir_env = std::getenv("BREEZE_VOICES_DIR");
    const std::string voices_dir = argval(argc, argv, "--voices-dir",
                                          vdir_env ? vdir_env : "/app/voices");
    if (model.empty()) { fprintf(stderr, "need --model breeze.gguf\n"); return 2; }

    const bool isolation = [] {
        const char * e = std::getenv("BREEZE_WORKER_ISOLATION");
        return e && e[0] && e[0] != '0';
    }();
    int idle_unload_seconds = 0;
    if (const char * e = std::getenv("BREEZE_IDLE_UNLOAD_SECONDS")) {
        idle_unload_seconds = atoi(e);
        if (idle_unload_seconds < 0) idle_unload_seconds = 0;
    }

    const char * fa_env = std::getenv("BREEZE_FA_MODEL");
    const std::string fa_model = argval(argc, argv, "--aligner-model", fa_env ? fa_env : "");
    int aligner_idle_seconds = idle_unload_seconds;
    if (const char * e = std::getenv("BREEZE_ALIGNER_IDLE_UNLOAD_SECONDS")) {
        aligner_idle_seconds = atoi(e);
        if (aligner_idle_seconds < 0) aligner_idle_seconds = 0;
    }

    breeze::VoiceStore voices(voices_dir);
    fprintf(stderr, "voice library: %s (%zu voices)\n", voices_dir.c_str(), voices.list().size());

    ServerCtx cx;
    cx.voices = &voices;
    cx.last_activity_ms.store(now_ms());
    cx.aligner_last_activity_ms.store(now_ms());

    std::unique_ptr<fa::AlignerSession> aligner;
    if (!fa_model.empty()) {
        aligner = std::make_unique<fa::AlignerSession>(argv[0]);
        cx.aligner = aligner.get();
        cx.aligner_model = fa_model;
        fprintf(stderr, "breeze-server: forced-alignment enabled (fa=%s, idle-unload %ds)\n",
                fa_model.c_str(), aligner_idle_seconds);
    } else {
        fprintf(stderr, "breeze-server: forced-alignment DISABLED (no --aligner-model)\n");
    }

    httplib::Server srv;
    std::unique_ptr<breeze::BreezeTTS>     inproc;
    std::unique_ptr<breeze::WorkerSession> session;
    breeze::WorkerConfig wcfg{model, n_ctx};

    if (isolation) {
        session = std::make_unique<breeze::WorkerSession>(argv[0]);
        if (const char * g = std::getenv("WORKER_DEFAULT_GPU")) {
            session->set_default_gpu(g);
            fprintf(stderr, "breeze worker-isolation: default GPU = %s\n", g);
        }
        cx.ensure     = [&] { return session->ensure_loaded(wcfg); };
        cx.set_gpu    = [&](const std::string & g) { session->set_next_gpu(g); };
        cx.is_loaded  = [&] { return session->is_alive(); };
        cx.worker_gpu = [&] { return session->is_alive() ? session->worker_gpu() : std::string(); };
        cx.unload     = [&] { session->shutdown(); if (cx.aligner) cx.aligner->shutdown(); };
        fprintf(stderr, "breeze-server: WORKER ISOLATION on (idle-unload %ds; lazy load)\n",
                idle_unload_seconds);
        install_routes(srv, session.get(), cx);

        if (idle_unload_seconds > 0) {
            std::thread([&cx, &session, idle_unload_seconds] {
                const int64_t threshold = (int64_t) idle_unload_seconds * 1000;
                const int check_s = std::max(1, idle_unload_seconds / 5);
                for (;;) {
                    std::this_thread::sleep_for(std::chrono::seconds(check_s));
                    if (cx.inflight.load() > 0) continue;
                    if (!session->is_alive()) continue;
                    if (now_ms() - cx.last_activity_ms.load() < threshold) continue;
                    std::unique_lock<std::mutex> lk(cx.mtx, std::try_to_lock);
                    if (!lk.owns_lock()) continue;
                    if (cx.inflight.load() > 0) continue;
                    fprintf(stderr, "breeze idle-unload: killing worker pid=%d\n", session->pid());
                    session->shutdown();
                }
            }).detach();
        }
    } else {
        inproc = std::make_unique<breeze::BreezeTTS>();
        fprintf(stderr, "loading breeze engine (in-process)...\n");
        if (!inproc->load(model, n_ctx)) {
            fprintf(stderr, "load failed: %s\n", inproc->get_error().c_str());
            return 1;
        }
        inproc->log_vram("ready");
        cx.code_max   = inproc->cfg().cc.codebook_size - 1;
        cx.ensure     = [] { return true; };
        cx.is_loaded  = [] { return true; };
        cx.worker_gpu = [] { return std::string(); };
        cx.unload     = [&] { if (cx.aligner) cx.aligner->shutdown(); };
        install_routes(srv, inproc.get(), cx);
    }

    if (aligner && aligner_idle_seconds > 0) {
        std::thread([&cx, &aligner, aligner_idle_seconds] {
            const int64_t threshold = (int64_t) aligner_idle_seconds * 1000;
            const int check_s = std::max(1, aligner_idle_seconds / 5);
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(check_s));
                if (cx.aligner_inflight.load() > 0) continue;
                if (!aligner->is_alive()) continue;
                if (now_ms() - cx.aligner_last_activity_ms.load() < threshold) continue;
                std::unique_lock<std::mutex> lk(cx.aligner_mtx, std::try_to_lock);
                if (!lk.owns_lock()) continue;
                if (cx.aligner_inflight.load() > 0) continue;
                fprintf(stderr, "breeze aligner idle-unload: killing aligner pid=%d\n", aligner->pid());
                aligner->shutdown();
            }
        }).detach();
    }

    fprintf(stderr, "breeze-server listening on %s:%d\n", host.c_str(), port);
    if (!srv.listen(host.c_str(), port)) { fprintf(stderr, "listen failed\n"); return 1; }
    return 0;
}
