// OpenAI-ish HTTP server for the Higgs-Audio-v3 C++ TTS engine, with a voice
// library (clone/catalog) + multi-speaker dialogue, and optional fork/IPC
// worker isolation (idle-VRAM→0 + cooperative cancellation).
//
//   higgs-server --backbone B.gguf --aux A.gguf [-H 0.0.0.0] [-p 8200]
//                [--n-ctx 8192] [--kv f16|q8] [--voices-dir DIR]
//
// Worker isolation (HIGGS_WORKER_ISOLATION=1): the parent process is CUDA-free
// (HTTP + filesystem VoiceStore only) and forks a child that owns the engine.
// The child is SIGKILLed after HIGGS_IDLE_UNLOAD_SECONDS of inactivity →
// VRAM true-0; the next request respawns + reloads. /v1/admin/{load,unload}
// drive load state explicitly (the koblem GPU gate unloads the peer TTS engine
// via /v1/admin/unload). Default (env unset) = single in-process engine,
// model resident, no idle-unload (the original behaviour).
//
// Endpoints
//   GET    /health                       -> {"status":"ok","model":..,"loaded":bool}
//   GET    /v1/models                    -> model list
//   GET    /v1/audio/voices              -> {"model_id":..,"voices":[..]}
//   POST   /v1/audio/voices              -> save a voice (trial-and-save | from-codes)
//   DELETE /v1/audio/voices/<id>         -> {} (404 if missing)
//   POST   /v1/audio/speech              -> WAV bytes (audio/wav)
//   POST   /v1/admin/load                -> {"model_loaded":bool}
//   POST   /v1/admin/unload              -> {"unloaded":bool,"model_loaded":bool}

#include "higgs_tts.h"
#include "higgs_voices.h"
#include "higgs_worker_session.h"
#include "fa_session.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
using namespace std::chrono_literals;

static const char * argval(int argc, char ** argv, const char * key, const char * def) {
    for (int i = 1; i < argc - 1; ++i) if (!strcmp(argv[i], key)) return argv[i+1];
    return def;
}

// Standard base64 (for SSE speech.audio.delta payloads — raw s16le PCM b64).
static std::string base64_encode(const void * data, size_t len) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const uint8_t * p = (const uint8_t *) data;
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = p[i] << 16;
        if (i + 1 < len) n |= p[i+1] << 8;
        if (i + 2 < len) n |= p[i+2];
        out.push_back(tbl[(n >> 18) & 63]);
        out.push_back(tbl[(n >> 12) & 63]);
        out.push_back((i + 1 < len) ? tbl[(n >> 6) & 63] : '=');
        out.push_back((i + 2 < len) ? tbl[n & 63] : '=');
    }
    return out;
}

// Convert float PCM [-1,1] → little-endian s16 bytes (clamped). Matches the
// 24 kHz s16le mono shape kobbler's reader decodes from speech.audio.delta.
static std::string pcm_f32_to_s16le(const float * pcm, size_t n) {
    std::string b; b.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        int v = (int) lrintf(pcm[i] * 32767.0f);
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        int16_t x = (int16_t) v; b.append((const char *) &x, 2);
    }
    return b;
}

// Whitespace-split a UTF-8 string into word tokens for forced alignment. Mirrors
// kobbler's `str::split_whitespace()` (which drives the read-along word_index
// mapping): split on ASCII whitespace, drop empty runs, keep punctuation glued
// to the word. The aligner's BPE tokenizer is byte-level so non-ASCII passes
// through unchanged.
static std::vector<std::string> whitespace_split_for_align(const std::string & s) {
    std::vector<std::string> out;
    size_t i = 0, n = s.size();
    while (i < n) {
        while (i < n && (unsigned char) s[i] <= ' ') i++;
        size_t start = i;
        while (i < n && (unsigned char) s[i] > ' ') i++;
        if (i > start) out.push_back(s.substr(start, i - start));
    }
    return out;
}

static std::string wav_bytes(const std::vector<float> & pcm, int sr) {
    std::string b;
    auto put = [&](const void * p, size_t n){ b.append((const char*)p, n); };
    auto u32 = [&](uint32_t v){ put(&v,4); }; auto u16 = [&](uint16_t v){ put(&v,2); };
    uint32_t n = (uint32_t)pcm.size(), db = n*2;
    put("RIFF",4); u32(36+db); put("WAVE",4);
    put("fmt ",4); u32(16); u16(1); u16(1); u32(sr); u32(sr*2); u16(2); u16(16);
    put("data",4); u32(db);
    b.reserve(b.size()+db);
    for (float s : pcm) { int v=(int)lrintf(s*32767.0f); if(v>32767)v=32767; if(v<-32768)v=-32768; int16_t x=(int16_t)v; put(&x,2); }
    return b;
}

// Decode a WAV blob (PCM s16 or f32, any channel count) → mono float + sample
// rate. Scans chunks so non-canonical headers still parse. The koblem clone UI
// always uploads mono 16-bit (encodeWavMono16); this also tolerates stereo /
// f32 dropped files. Returns false on a non-PCM / malformed blob.
static bool wav_decode(const std::string & b, std::vector<float> & out, int & sr) {
    auto rd_u32 = [&](size_t o)->uint32_t { return (uint8_t)b[o] | ((uint8_t)b[o+1]<<8) | ((uint8_t)b[o+2]<<16) | ((uint32_t)(uint8_t)b[o+3]<<24); };
    auto rd_u16 = [&](size_t o)->uint16_t { return (uint8_t)b[o] | ((uint8_t)b[o+1]<<8); };
    if (b.size() < 44 || b.compare(0,4,"RIFF") || b.compare(8,4,"WAVE")) return false;
    uint16_t fmt = 1, ch = 1, bits = 16; uint32_t rate = 24000;
    size_t data_off = 0, data_len = 0;
    size_t p = 12;
    while (p + 8 <= b.size()) {
        std::string cid = b.substr(p, 4);
        uint32_t csz = rd_u32(p + 4);
        size_t body = p + 8;
        if (cid == "fmt " && body + 16 <= b.size()) {
            fmt = rd_u16(body); ch = rd_u16(body+2); rate = rd_u32(body+4); bits = rd_u16(body+14);
        } else if (cid == "data") {
            data_off = body; data_len = std::min((size_t)csz, b.size() - body); break;
        }
        p = body + csz + (csz & 1);   // chunks are word-aligned
    }
    if (!data_off || ch == 0) return false;
    sr = (int)rate;
    if (fmt == 1 && bits == 16) {
        size_t n = data_len / 2; size_t frames = n / ch;
        out.resize(frames);
        const char * d = b.data() + data_off;
        for (size_t f = 0; f < frames; ++f) {
            int acc = 0;
            for (int c = 0; c < ch; ++c) { int16_t s = (int16_t)((uint8_t)d[(f*ch+c)*2] | ((uint8_t)d[(f*ch+c)*2+1]<<8)); acc += s; }
            out[f] = (float)acc / ch / 32768.0f;
        }
        return true;
    }
    if ((fmt == 3 || fmt == 0xFFFE) && bits == 32) {
        size_t n = data_len / 4; size_t frames = n / ch;
        out.resize(frames);
        const char * d = b.data() + data_off;
        for (size_t f = 0; f < frames; ++f) {
            float acc = 0; for (int c = 0; c < ch; ++c) { float s; std::memcpy(&s, d + (f*ch+c)*4, 4); acc += s; }
            out[f] = acc / ch;
        }
        return true;
    }
    return false;   // unsupported (e.g. compressed / 24-bit)
}

static void fill_params(const json & body, higgs::gen_params & gp, bool long_form) {
    gp.temperature = body.value("temperature", long_form ? 0.3f : 0.7f);
    gp.top_k = body.value("top_k", 50);
    gp.top_p = body.value("top_p", 0.95f);
    gp.seed  = body.value("seed", 0u);
    gp.max_new = body.value("max_new_tokens", 1024);
    gp.ras_win_len = body.value("ras_win_len", long_form ? 7 : 0);
    gp.ras_max_repeat = body.value("ras_max_repeat", 2);
}

// ── shared server state + the engine-agnostic speech/voice handlers ──
// ENG is either higgs::HiggsTTS (in-process) or higgs::HiggsWorkerSession
// (isolated); both expose the same synth surface + clear_cancel/request_cancel.
struct ServerCtx {
    std::mutex                mtx;        // serialises the single GPU engine + voice store
    higgs::VoiceStore *       voices = nullptr;
    const char *              model_id = "higgs-audio-v3-tts-4b";
    std::atomic<int64_t>      last_activity_ms{0};
    std::atomic<int>          inflight{0};
    std::function<bool()>     ensure;     // load engine if needed (spawn in isolation); true=ready
    std::function<bool()>     is_loaded;  // engine currently resident?
    std::function<void()>     unload;     // release the engine's GPU (SIGKILL child in isolation)
    std::function<void(const std::string&)> set_gpu;  // per-request GPU target (UUID) for placement

    // ── forced-alignment (word-highlight read-along) sibling ──
    // The aligner is a SEPARATE subprocess (own CUDA context) lazy-spawned on the
    // first aligned-stream request and SIGKILLed on idle / unload (VRAM true-0).
    // `aligner_model` empty ⇒ FA disabled (align requests fall back to plain SSE
    // audio with no highlight). Held across a whole request via `aligner_mtx`.
    fa::AlignerSession * aligner = nullptr;
    std::string                  aligner_model;
    std::mutex                   aligner_mtx;
    std::atomic<int64_t>         aligner_last_activity_ms{0};
    std::atomic<int>             aligner_inflight{0};
};

// trial_voice: render a zero-shot sample + recover its OUTPUT codes (the
// "audition then keep" path). Backend-specific overloads, declared up front so
// the templated install_routes() can call them. Defined below.
static bool trial_voice(higgs::HiggsTTS * eng, const std::string & text,
                        const higgs::gen_params & gp, std::vector<int32_t> & codes, int & T, int & N);
static bool trial_voice(higgs::HiggsWorkerSession * eng, const std::string & text,
                        const higgs::gen_params & gp, std::vector<int32_t> & codes, int & T, int & N);

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// RAII activity marker: bumps inflight around a GPU request so the idle-unload
// watchdog never kills the worker mid-synth, and refreshes last_activity_ms.
struct Activity {
    ServerCtx & cx;
    explicit Activity(ServerCtx & c) : cx(c) { cx.inflight.fetch_add(1); cx.last_activity_ms.store(now_ms()); }
    ~Activity() { cx.last_activity_ms.store(now_ms()); cx.inflight.fetch_sub(1); }
};

template <class ENG>
static void install_routes(httplib::Server & srv, ENG * eng, ServerCtx & cx) {
    auto err_json = [](httplib::Response & res, int code, const std::string & msg){
        res.status = code; res.set_content(json({{"error",msg}}).dump(), "application/json");
    };

    srv.Get("/health", [&cx](const httplib::Request&, httplib::Response& res){
        res.set_content(json({{"status","ok"},{"model",cx.model_id},
                              {"loaded", cx.is_loaded()}}).dump(), "application/json");
    });
    srv.Get("/v1/models", [&cx](const httplib::Request&, httplib::Response& res){
        res.set_content(json({{"object","list"},{"data",{{{"id",cx.model_id},{"object","model"}}}}}).dump(), "application/json");
    });

    // ---- voice library (filesystem; GPU-free — works while worker unloaded) ----
    srv.Get("/v1/audio/voices", [&cx](const httplib::Request&, httplib::Response& res){
        std::lock_guard<std::mutex> lk(cx.mtx);
        json arr = json::array();
        for (const auto & v : cx.voices->list())
            arr.push_back({{"id",v.id},{"frames",v.T},{"codebooks",v.N},{"has_ref_text",v.has_ref_text}});
        res.set_content(json({{"model_id",cx.model_id},{"voices",arr}}).dump(), "application/json");
    });

    srv.Post("/v1/audio/voices", [&cx, eng, err_json](const httplib::Request& req, httplib::Response& res){
        // ---- multipart wav upload: encode a recorded/uploaded clip (clone) ----
        if (req.is_multipart_form_data() || req.has_file("audio_sample") || req.has_file("voice_file")) {
            std::string name = req.has_file("name") ? req.get_file_value("name").content
                             : req.has_file("voice_name") ? req.get_file_value("voice_name").content : "";
            if (name.empty()) return err_json(res,400,"missing name");
            std::string wav = req.has_file("audio_sample") ? req.get_file_value("audio_sample").content
                            : req.has_file("voice_file") ? req.get_file_value("voice_file").content : "";
            if (wav.empty()) return err_json(res,400,"missing audio_sample");
            std::string ref_text = req.has_param("ref_text") ? req.get_param_value("ref_text")
                                 : req.has_file("ref_text") ? req.get_file_value("ref_text").content : "";
            std::string id = higgs::VoiceStore::sanitize(name);

            std::vector<float> samples; int sr = 24000;
            if (!wav_decode(wav, samples, sr) || samples.empty())
                return err_json(res,400,"could not decode WAV (need PCM s16/f32)");
            if (!cx.ensure()) return err_json(res,503,"engine load failed");
            Activity act(cx);
            std::vector<int32_t> codes; int T=0, N=8; std::string e; bool ok;
            {
                std::lock_guard<std::mutex> lk(cx.mtx);
                ok = eng->encode_voice(samples.data(), (int)samples.size(), sr, codes, T, N);
                if (ok) ok = cx.voices->save(id, codes.data(), T, N, ref_text, e);
                if (ok) cx.voices->save_wav(id, wav, e);   // best-effort original clip for sample.wav
            }
            if (!ok) return err_json(res,500, e.empty()? eng->get_error() : e);
            res.set_content(json({{"id",id},{"name",id},{"frames",T},{"codebooks",N},
                                  {"mode","clone"},{"ref_frames",T}}).dump(), "application/json");
            return;
        }

        json body;
        try { body = json::parse(req.body); } catch (...) { return err_json(res,400,"bad json"); }
        std::string name = body.value("name", std::string());
        if (name.empty()) return err_json(res,400,"missing name");
        std::string id = higgs::VoiceStore::sanitize(name);
        std::string ref_text = body.value("ref_text", std::string());

        // from-codes: store client-supplied codes directly (no GPU).
        if (body.contains("codes_flat")) {
            auto flat = body["codes_flat"].get<std::vector<int>>();
            int N = body.value("n_codebooks", 8);
            if (N <= 0 || (int)flat.size() % N != 0) return err_json(res,400,"codes_flat not divisible by n_codebooks");
            int T = (int)flat.size() / N;
            std::vector<int32_t> codes(flat.begin(), flat.end());
            std::lock_guard<std::mutex> lk(cx.mtx); std::string e;
            if (!cx.voices->save(id, codes.data(), T, N, ref_text, e)) return err_json(res,500,e);
            res.set_content(json({{"id",id},{"frames",T},{"codebooks",N}}).dump(), "application/json");
            return;
        }

        // trial-and-save: render a zero-shot sample at `seed` and keep its codes.
        std::string text = body.value("text", body.value("input", std::string()));
        if (text.empty()) return err_json(res,400,"need 'text' (trial line) or 'codes_flat'");
        higgs::gen_params gp; fill_params(body, gp, false);
        // Activity BEFORE ensure(): bumps inflight so the idle-unload watchdog
        // can't SIGKILL the worker mid-load on a cold-start request (stale
        // last_activity + inflight==0 + cx.mtx still free during the ~7.5s load).
        Activity act(cx);
        if (!cx.ensure()) return err_json(res,503,"engine load failed");
        std::vector<int32_t> codes; int T=0, N=8; std::string e; bool ok;
        {
            std::lock_guard<std::mutex> lk(cx.mtx);
            ok = trial_voice(eng, text, gp, codes, T, N);
            if (ok) ok = cx.voices->save(id, codes.data(), T, N, ref_text, e);
        }
        if (!ok) return err_json(res,500, e.empty()? eng->get_error() : e);
        res.set_content(json({{"id",id},{"frames",T},{"codebooks",N},{"seed",gp.seed}}).dump(), "application/json");
    });

    // Reference-clip download (qwen3 parity). GPU-free — served from the
    // VoiceStore wav sidecar; 404 for code-only voices (trial-and-save).
    srv.Get(R"(/v1/audio/voices/([^/]+)/sample\.wav)", [&cx, err_json](const httplib::Request& req, httplib::Response& res){
        std::string id = req.matches[1];
        std::lock_guard<std::mutex> lk(cx.mtx); std::string wav;
        if (!cx.voices->load_wav(id, wav)) return err_json(res,404,"no sample for voice: "+id);
        res.set_content(wav, "audio/wav");
    });
    srv.Get(R"(/v1/audio/voices/([^/]+)/ref_text)", [&cx, err_json](const httplib::Request& req, httplib::Response& res){
        std::string id = req.matches[1];
        std::lock_guard<std::mutex> lk(cx.mtx); std::string rt;
        if (!cx.voices->load_ref_text(id, rt)) return err_json(res,404,"no ref_text for voice: "+id);
        res.set_content(rt, "text/plain; charset=utf-8");
    });

    srv.Delete(R"(/v1/audio/voices/([^/]+))", [&cx, err_json](const httplib::Request& req, httplib::Response& res){
        std::string id = req.matches[1];
        std::lock_guard<std::mutex> lk(cx.mtx); std::string e;
        if (!cx.voices->remove(id, e)) return err_json(res,404,e);
        res.set_content("{}", "application/json");
    });

    // ---- admin: explicit load / unload (idle-VRAM control for the gate) ----
    srv.Post("/v1/admin/unload", [&cx](const httplib::Request&, httplib::Response& res){
        std::lock_guard<std::mutex> lk(cx.mtx);
        bool was = cx.is_loaded();
        cx.unload();
        res.set_content(json({{"unloaded", was},{"model_loaded", cx.is_loaded()}}).dump(), "application/json");
    });
    srv.Post("/v1/admin/load", [&cx](const httplib::Request&, httplib::Response& res){
        Activity act(cx);   // guard the explicit load against the idle watchdog
        bool ok = cx.ensure();
        res.set_content(json({{"model_loaded", ok && cx.is_loaded()}}).dump(), "application/json");
    });

    // ---- speech ----
    srv.Post("/v1/audio/speech", [&cx, eng, err_json](const httplib::Request& req, httplib::Response& res){
        json body;
        try { body = json::parse(req.body); } catch (...) { return err_json(res,400,"bad json"); }
        std::string input = body.value("input", body.value("text", std::string()));
        if (input.empty()) return err_json(res,400,"empty input");

        const bool long_form = body.value("long", false);
        higgs::gen_params gp; fill_params(body, gp, long_form);
        // Per-request GPU target (gate placement) before ensure() relocates.
        if (cx.set_gpu) cx.set_gpu(body.value("gpu", std::string()));
        // Activity BEFORE ensure() — see trial-save note (cold-start load race).
        Activity act(cx);
        if (!cx.ensure()) return err_json(res,503,"engine load failed");

        // ---- SSE streaming + forced-alignment (word-highlight read-along) ----
        // Path taken by kobbler's BookReader via koblem: stream_format="sse",
        // response_format="pcm", align=true, align_stream="partial". Emits
        // speech.audio.delta (b64 24 kHz s16le PCM) + interleaved
        // speech.audio.alignment.{partial,final} + speech.audio.done — the exact
        // event shape kobbler already parses for qwen3. The aligner is a sibling
        // subprocess (24 kHz PCM → per-word t0/t1); falls back to plain SSE audio
        // (no highlight) when no aligner model is configured.
        if (body.value("stream_format", std::string()) == "sse") {
            const bool do_align       = body.value("align", false);
            const std::string a_mode  = body.value("align_stream", std::string("final-only"));
            const bool do_partial     = do_align && a_mode == "partial"
                                        && cx.aligner && !cx.aligner_model.empty();
            // Reader paragraphs can be long; don't truncate. max_audio_tokens is
            // the qwen3-side knob name kobbler sends; accept both.
            gp.max_new = (int) body.value("max_new_tokens", body.value("max_audio_tokens", 2048));

            // Resolve an optional clone voice (filesystem VoiceStore, GPU-free).
            std::string voice = body.value("voice", std::string());
            const bool have_voice = !voice.empty() && voice != "default";
            higgs::HiggsTTS::named_voice nv; int nvN = 0;
            std::string ref_text;
            if (have_voice) {
                if (!cx.voices->load(voice, nv.codes_TN, nv.T, nvN, nv.ref_text))
                    return err_json(res, 400, "unknown voice: " + voice);
                ref_text = body.value("ref_text", nv.ref_text);
            }
            std::vector<std::string> words = whitespace_split_for_align(input);

            res.set_header("Content-Type", "text/event-stream");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider("text/event-stream",
                [eng, &cx, input, gp, do_partial, have_voice,
                 nv = std::move(nv), ref_text, words = std::move(words)]
                (size_t, httplib::DataSink & sink) mutable -> bool {
                    Activity act_stream(cx);
                    std::mutex sink_write_mutex;
                    auto write_sse = [&](const std::string & s) -> bool {
                        std::lock_guard<std::mutex> lk(sink_write_mutex);
                        return sink.write(s.data(), s.size());
                    };
                    auto emit_event = [&](const char * ev, const json & j) {
                        write_sse(std::string("event: ") + ev + "\ndata: " + j.dump() + "\n\n");
                    };

                    // ── aligner sibling: ensure + begin + reader thread ──
                    bool partial_active = false;
                    std::unique_lock<std::mutex> aligner_lock;
                    std::thread reader_thread;
                    std::atomic<bool> reader_stop{false};
                    std::atomic<int64_t> audio_offset_ms{0};
                    if (do_partial) {
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

                    // ── disconnect watchdog: worker stays warm (499-style) ──
                    std::atomic<bool> wd_stop{false};
                    std::thread wd([&]() {
                        while (!wd_stop.load()) {
                            if (sink.is_writable && !sink.is_writable()) { eng->request_cancel(); break; }
                            std::this_thread::sleep_for(50ms);
                        }
                    });

                    auto emit_audio = [&](const float * pcm, int n) {
                        cx.last_activity_ms.store(now_ms());
                        std::string s16 = pcm_f32_to_s16le(pcm, (size_t) n);
                        emit_event("speech.audio.delta",
                                   {{"type","speech.audio.delta"},
                                    {"audio", base64_encode(s16.data(), s16.size())}});
                        if (partial_active) {
                            const int64_t chunk_ms = (int64_t)((int64_t)n * 1000 / 24000);
                            const int64_t total = audio_offset_ms.fetch_add(chunk_ms) + chunk_ms;
                            cx.aligner_last_activity_ms.store(now_ms());
                            cx.aligner->push_partial_pcm(pcm, (size_t) n, total);
                        }
                    };

                    higgs::gen_result r; bool ok = false;
                    {
                        std::lock_guard<std::mutex> lk(cx.mtx);
                        eng->clear_cancel();
                        if (have_voice) {
                            // Progressive cloned-voice read-along: real TTFA +
                            // progressive partial alignment (not buffered).
                            ok = eng->synthesize_stream_with_ref(input, nv.codes_TN.data(), nv.T, ref_text, gp, 25,
                                [&](const float * pcm, int n, bool) { emit_audio(pcm, n); }, r);
                        } else {
                            ok = eng->synthesize_stream(input, gp, 25,
                                [&](const float * pcm, int n, bool) { emit_audio(pcm, n); }, r);
                        }
                    }

                    if (partial_active) {
                        reader_stop.store(true);
                        if (reader_thread.joinable()) reader_thread.join();
                    }
                    wd_stop.store(true);
                    if (wd.joinable()) wd.join();

                    // ── finalize alignment ──
                    if (partial_active) {
                        const int64_t total_ms = audio_offset_ms.load();
                        std::vector<fa::AlignedWord> aligned; fa::AlignProfile prof;
                        const bool fok = cx.aligner->finalize_streaming_align(nullptr, 0, total_ms, aligned, prof);
                        if (fok) {
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

                    emit_event("speech.audio.done", {{"type","speech.audio.done"}});
                    sink.done();
                    return true;
                });
            return;
        }

        // ---- multi-speaker dialogue (speakers map present) ----
        if (body.contains("speakers") && body["speakers"].is_object()) {
            std::map<std::string, higgs::HiggsTTS::named_voice> vmap;
            for (auto it = body["speakers"].begin(); it != body["speakers"].end(); ++it) {
                std::string vid = it.value().get<std::string>();
                higgs::HiggsTTS::named_voice nv; int N = 0;
                if (!cx.voices->load(vid, nv.codes_TN, nv.T, N, nv.ref_text))
                    return err_json(res,400,"unknown voice for "+it.key()+": "+vid);
                std::string spk = it.key(); for (char & c : spk) if (c==' ') c='_';
                vmap[spk] = std::move(nv);
            }
            const int gap_ms = body.value("gap_ms", 250);
            const bool rolling = body.value("rolling", true);
            higgs::gen_result r; bool ok;
            { std::lock_guard<std::mutex> lk(cx.mtx);
              eng->clear_cancel();
              ok = eng->synthesize_multispeaker(input, vmap, gp, gap_ms, rolling, nullptr, r); }
            if (!ok) return err_json(res,500,eng->get_error());
            res.set_header("X-Audio-Seconds", std::to_string(r.T/25.0));
            res.set_content(wav_bytes(r.pcm, 24000), "audio/wav");
            return;
        }

        // ---- clone from a stored voice (voice param) ----
        std::string voice = body.value("voice", std::string());
        if (!voice.empty() && voice != "default") {
            higgs::HiggsTTS::named_voice nv; int N = 0;
            if (!cx.voices->load(voice, nv.codes_TN, nv.T, N, nv.ref_text))
                return err_json(res,400,"unknown voice: "+voice);
            std::string rt = body.value("ref_text", nv.ref_text);
            higgs::gen_result r; bool ok;
            { std::lock_guard<std::mutex> lk(cx.mtx);
              eng->clear_cancel();
              ok = eng->synthesize_with_ref(input, nv.codes_TN.data(), nv.T, rt, gp, r, true); }
            if (!ok) return err_json(res,500,eng->get_error());
            res.set_header("X-Audio-Seconds", std::to_string(r.T/25.0));
            res.set_content(wav_bytes(r.pcm, 24000), "audio/wav");
            return;
        }

        // ---- long-form ----
        if (long_form) {
            const int buffer = body.value("buffer", 2);
            const int chunk_words = body.value("chunk_words", 100);
            higgs::gen_result r; bool ok;
            { std::lock_guard<std::mutex> lk(cx.mtx);
              eng->clear_cancel();
              ok = eng->synthesize_long(input, gp, buffer, chunk_words, nullptr, r); }
            if (!ok) return err_json(res,500,eng->get_error());
            res.set_header("X-Audio-Seconds", std::to_string(r.T/25.0));
            res.set_content(wav_bytes(r.pcm, 24000), "audio/wav");
            return;
        }

        // ---- streaming (chunked WAV; client-disconnect → cooperative cancel) ----
        if (body.value("stream", false)) {
            res.set_header("Content-Type", "audio/wav");
            res.set_chunked_content_provider("audio/wav",
                [eng, &cx, input, gp](size_t, httplib::DataSink & sink) {
                    auto put16 = [&](const float * pcm, int n){
                        std::string b; b.reserve((size_t)n*2);
                        for (int i=0;i<n;++i){ int v=(int)lrintf(pcm[i]*32767.0f); if(v>32767)v=32767; if(v<-32768)v=-32768; int16_t x=(int16_t)v; b.append((const char*)&x,2);}
                        sink.write(b.data(), b.size());
                    };
                    std::string h; auto u32=[&](uint32_t v){h.append((const char*)&v,4);}; auto u16=[&](uint16_t v){h.append((const char*)&v,2);};
                    h.append("RIFF",4); u32(0xFFFFFFFF); h.append("WAVE",4);
                    h.append("fmt ",4); u32(16); u16(1); u16(1); u32(24000); u32(48000); u16(2); u16(16);
                    h.append("data",4); u32(0xFFFFFFFF);
                    sink.write(h.data(), h.size());

                    // Disconnect watchdog: poll sink.is_writable; on drop, ask the
                    // engine to bail (worker stays warm — 499-style, not SIGKILL).
                    std::atomic<bool> wd_stop{false};
                    std::thread wd([&]{
                        while (!wd_stop.load()) {
                            if (sink.is_writable && !sink.is_writable()) { eng->request_cancel(); break; }
                            std::this_thread::sleep_for(50ms);
                        }
                    });
                    higgs::gen_result r;
                    bool ok;
                    {
                        std::lock_guard<std::mutex> lk(cx.mtx);
                        eng->clear_cancel();
                        ok = eng->synthesize_stream(input, gp, 25,
                            [&](const float * pcm, int n, bool){ put16(pcm, n); }, r);
                    }
                    wd_stop.store(true); wd.join();
                    sink.done();
                    return ok;
                });
            return;
        }

        // ---- plain zero-shot ----
        higgs::gen_result r; bool ok;
        { std::lock_guard<std::mutex> lk(cx.mtx); eng->clear_cancel(); ok = eng->synthesize(input, gp, r); }
        if (!ok) return err_json(res,500,eng->get_error());
        res.set_header("X-Audio-Seconds", std::to_string(r.T/25.0));
        res.set_header("X-RTF", std::to_string((r.T/25.0)/((r.prefill_ms+r.decode_ms+r.codec_ms)/1000.0)));
        res.set_content(wav_bytes(r.pcm, 24000), "audio/wav");
    });
}

// trial_voice: render a zero-shot sample and recover its OUTPUT codes. The two
// engine backends expose this differently (in-process via synthesize; session
// via a dedicated TRIAL frame), so it's a small overload set.
static bool trial_voice(higgs::HiggsTTS * eng, const std::string & text,
                        const higgs::gen_params & gp, std::vector<int32_t> & codes, int & T, int & N) {
    higgs::gen_result r;
    eng->clear_cancel();
    if (!eng->synthesize(text, gp, r)) return false;
    if (r.codes_TN.empty() || r.T <= 0) return false;
    codes = r.codes_TN; T = r.T; N = (int)(r.codes_TN.size() / r.T);
    return true;
}
static bool trial_voice(higgs::HiggsWorkerSession * eng, const std::string & text,
                        const higgs::gen_params & gp, std::vector<int32_t> & codes, int & T, int & N) {
    return eng->trial_codes(text, gp, codes, T, N);
}

int main(int argc, char ** argv) {
    // Worker-isolation child: when "--higgs-worker <fd>" is passed we ARE the
    // GPU synth subprocess. "--fa-aligner <fd>" → the shared FA aligner sibling.
    // Either way we run a dispatch loop and never start the HTTP server.
    for (int i = 1; i < argc - 1; ++i) {
        if (!strcmp(argv[i], "--higgs-worker"))
            return higgs::run_higgs_worker_loop(atoi(argv[i+1]));
        if (!strcmp(argv[i], "--fa-aligner"))
            return fa::run_aligner_loop(atoi(argv[i+1]));
    }

    std::string bb = argval(argc,argv,"--backbone","");
    std::string aux = argval(argc,argv,"--aux","");
    const char * auxenc_env = std::getenv("HIGGS_AUX_ENC");
    std::string aux_enc = argval(argc,argv,"--aux-enc", auxenc_env ? auxenc_env : "");
    std::string host = argval(argc,argv,"-H","0.0.0.0");
    int port = atoi(argval(argc,argv,"-p","8200"));
    int n_ctx = atoi(argval(argc,argv,"--n-ctx","8192"));
    const char * vdir_env = std::getenv("HIGGS_VOICES_DIR");
    std::string voices_dir = argval(argc,argv,"--voices-dir", vdir_env ? vdir_env : "/app/voices");
    if (!std::getenv("HIGGS_LM_KV")) setenv("HIGGS_LM_KV", "q8", 1);
    if (const char * kv = argval(argc,argv,"--kv",nullptr)) setenv("HIGGS_LM_KV", kv, 1);
    std::string kv = std::getenv("HIGGS_LM_KV");
    if (bb.empty() || aux.empty()) { fprintf(stderr,"need --backbone and --aux\n"); return 2; }

    const bool isolation = [](){ const char * e = std::getenv("HIGGS_WORKER_ISOLATION"); return e && e[0] && e[0] != '0'; }();
    int idle_unload_seconds = 0;
    if (const char * e = std::getenv("HIGGS_IDLE_UNLOAD_SECONDS")) { idle_unload_seconds = atoi(e); if (idle_unload_seconds < 0) idle_unload_seconds = 0; }

    // Forced-alignment (word-highlight read-along) GGUF. Local path (eval/local
    // only): --worker-aligner-fa / --aligner-model, or HIGGS_FA_MODEL. Empty ⇒
    // FA disabled (aligned-stream requests degrade to plain SSE audio). The
    // aligner is its own sibling subprocess; isolation must be on (it forks off
    // the same binary via --higgs-aligner).
    const char * fa_env = std::getenv("HIGGS_FA_MODEL");
    std::string fa_model = argval(argc, argv, "--worker-aligner-fa",
                            argval(argc, argv, "--aligner-model", fa_env ? fa_env : ""));
    int aligner_idle_seconds = idle_unload_seconds > 0 ? idle_unload_seconds : 0;
    if (const char * e = std::getenv("HIGGS_ALIGNER_IDLE_UNLOAD_SECONDS")) {
        aligner_idle_seconds = atoi(e); if (aligner_idle_seconds < 0) aligner_idle_seconds = 0;
    }

    higgs::VoiceStore voices(voices_dir);
    fprintf(stderr, "voice library: %s (%zu voices)\n", voices_dir.c_str(), voices.list().size());

    ServerCtx cx;
    cx.voices = &voices;
    cx.last_activity_ms.store(now_ms());
    cx.aligner_last_activity_ms.store(now_ms());

    // Forced-alignment sibling (its own subprocess + CUDA context). Lazy-spawned
    // on the first aligned-stream request; SIGKILLed on idle / unload. Works in
    // both isolation and in-process modes (it never shares the synth context).
    std::unique_ptr<fa::AlignerSession> aligner;
    if (!fa_model.empty()) {
        aligner = std::make_unique<fa::AlignerSession>(argv[0]);
        cx.aligner = aligner.get();
        cx.aligner_model = fa_model;
        fprintf(stderr, "higgs-server: forced-alignment enabled (fa=%s, aligner idle-unload %ds)\n",
                fa_model.c_str(), aligner_idle_seconds);
    } else {
        fprintf(stderr, "higgs-server: forced-alignment DISABLED (no --worker-aligner-fa / HIGGS_FA_MODEL)\n");
    }

    httplib::Server srv;

    // Two backends share install_routes() + ServerCtx; only the engine object,
    // ensure/is_loaded/unload closures, and lifecycle differ.
    std::unique_ptr<higgs::HiggsTTS>           inproc;
    std::unique_ptr<higgs::HiggsWorkerSession> session;
    // wcfg must outlive the request handlers (the ensure lambda captures it by
    // ref), so it lives at main scope — NOT inside the `if (isolation)` block.
    higgs::HiggsWorkerConfig wcfg{bb, aux, n_ctx, kv, aux_enc};
    if (!aux_enc.empty()) fprintf(stderr, "voice-clone encoder: %s\n", aux_enc.c_str());

    if (isolation) {
        session = std::make_unique<higgs::HiggsWorkerSession>(argv[0]);
        if (const char * g = std::getenv("WORKER_DEFAULT_GPU")) {
            session->set_default_gpu(g);
            fprintf(stderr, "higgs worker-isolation: default GPU = %s\n", g);
        }
        cx.ensure     = [&]{ return session->ensure_loaded(wcfg); };
        cx.set_gpu    = [&](const std::string & g){ session->set_next_gpu(g); };
        cx.is_loaded  = [&]{ return session->is_alive(); };
        // unload also drops the aligner sibling (the gate swaps the whole TTS
        // engine out; the aligner's ~0.6-1.1 GB shouldn't linger). io_mutex_-
        // level SIGKILL — no deadlock with the request-level aligner_mtx.
        cx.unload     = [&]{ session->shutdown(); if (cx.aligner) cx.aligner->shutdown(); };
        fprintf(stderr, "higgs-server: WORKER ISOLATION on (idle-unload %ds; lazy load on first request)\n", idle_unload_seconds);
        install_routes(srv, session.get(), cx);

        if (idle_unload_seconds > 0) {
            std::thread([&cx, &session, idle_unload_seconds]{
                const int64_t threshold = (int64_t)idle_unload_seconds * 1000;
                const int check_s = std::max(1, idle_unload_seconds / 5);
                for (;;) {
                    std::this_thread::sleep_for(std::chrono::seconds(check_s));
                    if (cx.inflight.load() > 0) continue;
                    if (!session->is_alive()) continue;
                    if (now_ms() - cx.last_activity_ms.load() < threshold) continue;
                    std::unique_lock<std::mutex> lk(cx.mtx, std::try_to_lock);
                    if (!lk.owns_lock()) continue;           // a synth is mid-flight
                    if (cx.inflight.load() > 0) continue;
                    fprintf(stderr, "higgs idle-unload: %lld s idle, killing worker pid=%d\n",
                            (long long)((now_ms() - cx.last_activity_ms.load())/1000), session->pid());
                    session->shutdown();
                }
            }).detach();
        }
    } else {
        inproc = std::make_unique<higgs::HiggsTTS>();
        fprintf(stderr, "loading higgs engine (in-process)...\n");
        if (!inproc->load(bb, aux, n_ctx)) { fprintf(stderr,"load failed: %s\n", inproc->get_error().c_str()); return 1; }
        if (!aux_enc.empty()) inproc->set_aux_enc(aux_enc);
        if (!inproc->tokenizer_loaded()) fprintf(stderr,"tokenizer not loaded — /v1/audio/speech will 500\n");
        inproc->lm().log_vram("ready");
        cx.ensure    = []{ return true; };
        cx.is_loaded = []{ return true; };
        // in-process: synth model stays resident, but the aligner sibling can
        // still be reclaimed (own subprocess).
        cx.unload    = [&]{ if (cx.aligner) cx.aligner->shutdown(); };
        install_routes(srv, inproc.get(), cx);
    }

    // Dedicated aligner idle-unload watchdog — drops the aligner's VRAM fast
    // when the reader pauses between paragraphs, independent of the synth
    // worker's idle window. SIGKILL → VRAM true-0; next paragraph respawns it.
    if (aligner && aligner_idle_seconds > 0) {
        std::thread([&cx, &aligner, aligner_idle_seconds]{
            const int64_t threshold = (int64_t) aligner_idle_seconds * 1000;
            const int check_s = std::max(1, aligner_idle_seconds / 5);
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(check_s));
                if (cx.aligner_inflight.load() > 0) continue;
                if (!aligner->is_alive()) continue;
                if (now_ms() - cx.aligner_last_activity_ms.load() < threshold) continue;
                std::unique_lock<std::mutex> lk(cx.aligner_mtx, std::try_to_lock);
                if (!lk.owns_lock()) continue;            // an align stream is mid-flight
                if (cx.aligner_inflight.load() > 0) continue;
                fprintf(stderr, "higgs aligner idle-unload: %lld s idle, killing aligner pid=%d\n",
                        (long long)((now_ms() - cx.aligner_last_activity_ms.load())/1000), aligner->pid());
                aligner->shutdown();
            }
        }).detach();
    }

    fprintf(stderr, "higgs-server listening on %s:%d\n", host.c_str(), port);
    if (!srv.listen(host.c_str(), port)) { fprintf(stderr,"listen failed\n"); return 1; }
    return 0;
}
