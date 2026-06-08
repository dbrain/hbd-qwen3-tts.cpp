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
        // Activity BEFORE ensure() — see trial-save note (cold-start load race).
        Activity act(cx);
        if (!cx.ensure()) return err_json(res,503,"engine load failed");

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
    // GPU subprocess; run the dispatch loop and never start the HTTP server.
    for (int i = 1; i < argc - 1; ++i) {
        if (!strcmp(argv[i], "--higgs-worker")) {
            return higgs::run_higgs_worker_loop(atoi(argv[i+1]));
        }
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

    higgs::VoiceStore voices(voices_dir);
    fprintf(stderr, "voice library: %s (%zu voices)\n", voices_dir.c_str(), voices.list().size());

    ServerCtx cx;
    cx.voices = &voices;
    cx.last_activity_ms.store(now_ms());

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
        cx.ensure     = [&]{ return session->ensure_loaded(wcfg); };
        cx.is_loaded  = [&]{ return session->is_alive(); };
        cx.unload     = [&]{ session->shutdown(); };
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
        cx.unload    = []{};   // in-process: model stays resident (no SIGKILL reclaim)
        install_routes(srv, inproc.get(), cx);
    }

    fprintf(stderr, "higgs-server listening on %s:%d\n", host.c_str(), port);
    if (!srv.listen(host.c_str(), port)) { fprintf(stderr,"listen failed\n"); return 1; }
    return 0;
}
