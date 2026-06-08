// Higgs TTS standalone synth CLI: prompt-ids (.npy int) -> 24 kHz wav.
//
//   higgs_synth --backbone B.gguf --aux A.gguf --ids ids.npy --out out.wav \
//               [--temp 0.0] [--top-k 0] [--top-p 1.0] [--seed 0] [--max-new 1024] [--kv f16|q8]
//
// Tokenisation is done upstream (python ref adapter) and passed as ids.npy so
// the ear-test path doesn't depend on the in-C++ tokenizer (server work).

#include "higgs_tts.h"
#include "higgs_encode.h"
#include "npy.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void wav_write(const std::string & path, const std::vector<float> & pcm, int sr) {
    FILE * f = fopen(path.c_str(), "wb"); if (!f) return;
    auto u32=[&](uint32_t v){fwrite(&v,4,1,f);}; auto u16=[&](uint16_t v){fwrite(&v,2,1,f);};
    uint32_t n=(uint32_t)pcm.size(), db=n*2;
    fwrite("RIFF",1,4,f); u32(36+db); fwrite("WAVE",1,4,f);
    fwrite("fmt ",1,4,f); u32(16); u16(1); u16(1); u32(sr); u32(sr*2); u16(2); u16(16);
    fwrite("data",1,4,f); u32(db);
    for (float s: pcm){ int v=(int)lrintf(s*32767.0f); if(v>32767)v=32767; if(v<-32768)v=-32768; int16_t x=(int16_t)v; fwrite(&x,2,1,f);}
    fclose(f);
}

static const char * argval(int argc, char**argv, const char*key, const char*def){
    for(int i=1;i<argc-1;++i) if(!strcmp(argv[i],key)) return argv[i+1];
    return def;
}

// Minimal 16-bit PCM mono WAV reader -> float [-1,1] at the file's sample rate.
static std::vector<float> wav_read(const std::string & path, int & sr) {
    std::vector<float> out; sr = 0;
    FILE * f = fopen(path.c_str(), "rb"); if (!f) return out;
    char id[4]; uint32_t sz; uint16_t fmt=1, ch=1, bits=16; uint32_t rate=24000;
    fread(id,1,4,f); fread(&sz,4,1,f); fread(id,1,4,f); // RIFF .... WAVE
    while (fread(id,1,4,f)==4) {
        uint32_t csz; fread(&csz,4,1,f);
        if (!memcmp(id,"fmt ",4)) {
            fread(&fmt,2,1,f); fread(&ch,2,1,f); fread(&rate,4,1,f);
            uint32_t br; uint16_t ba; fread(&br,4,1,f); fread(&ba,2,1,f); fread(&bits,2,1,f);
            if (csz>16) fseek(f, csz-16, SEEK_CUR);
        } else if (!memcmp(id,"data",4)) {
            size_t n = csz/2; std::vector<int16_t> pcm(n); fread(pcm.data(),2,n,f);
            out.resize(n/ch);
            for (size_t i=0;i<out.size();++i) out[i] = pcm[i*ch] / 32768.0f;  // ch0 only
            break;
        } else fseek(f, csz, SEEK_CUR);
    }
    fclose(f); sr = (int)rate; return out;
}

int main(int argc, char ** argv) {
    std::string bb = argval(argc,argv,"--backbone","");
    std::string aux= argval(argc,argv,"--aux","");
    std::string ids_f = argval(argc,argv,"--ids","");
    std::string text = argval(argc,argv,"--text","");
    std::string out = argval(argc,argv,"--out","out.wav");
    // encode-only "bake a voice" mode: --ref-wav + --save-ref-codes, no text/ids.
    bool bake_only = argval(argc,argv,"--ref-wav",nullptr) && argval(argc,argv,"--save-ref-codes",nullptr) && text.empty() && ids_f.empty();
    bool has_raw = argval(argc,argv,"--raw-prompt",nullptr) != nullptr;
    if (bb.empty()||aux.empty()||(ids_f.empty()&&text.empty()&&!bake_only&&!has_raw)) { fprintf(stderr,"need --backbone --aux and (--ids or --text)\n"); return 2; }

    higgs::gen_params gp;
    gp.temperature = atof(argval(argc,argv,"--temp","0.0"));
    gp.top_k = atoi(argval(argc,argv,"--top-k","0"));
    gp.top_p = atof(argval(argc,argv,"--top-p","1.0"));
    gp.seed  = (uint32_t)strtoul(argval(argc,argv,"--seed","0"),nullptr,10);
    gp.max_new = atoi(argval(argc,argv,"--max-new","1024"));
    gp.ras_win_len = atoi(argval(argc,argv,"--ras-win","0"));     // 0 => RAS off
    gp.ras_max_repeat = atoi(argval(argc,argv,"--ras-rep","2"));
    int n_ctx = atoi(argval(argc,argv,"--n-ctx","8192"));   // dynamic KV: cap only; allocates on demand
    bool do_long = argval(argc,argv,"--long",nullptr) != nullptr;
    int buffer = atoi(argval(argc,argv,"--buffer","2"));
    int chunk_words = atoi(argval(argc,argv,"--chunk-words","100"));
    std::string ref_wav = argval(argc,argv,"--ref-wav","");
    std::string ref_text = argval(argc,argv,"--ref-text","");
    std::string aux_enc = argval(argc,argv,"--aux-enc","");
    std::string codes_in = argval(argc,argv,"--codes","");        // voice = saved codes (no encoder/wav)
    std::string codes_out = argval(argc,argv,"--save-codes","");  // save this run's OUTPUT codes as a voice
    std::string ref_codes_out = argval(argc,argv,"--save-ref-codes",""); // bake ref-wav encode -> voice.npy (one-time)
    std::string raw_prompt = argval(argc,argv,"--raw-prompt","");  // build the prompt by hand (experimental token layouts)
    if (const char* kv = argval(argc,argv,"--kv",nullptr)) setenv("HIGGS_LM_KV", kv, 1);

    // Voice clone: encode the ref wav -> codes BEFORE loading the TTS backbone, so
    // the (F32) encoder never coexists with the 3.4 GB backbone — keeps the true
    // VRAM peak == zero-shot (encoder freed before the backbone loads).
    std::vector<int32_t> rcodes; int rT = 0;
    if (!ref_wav.empty()) {
        if (aux_enc.empty()) { fprintf(stderr,"--ref-wav needs --aux-enc <encode gguf>\n"); return 2; }
        higgs::HiggsCodecEncoder enc;
        if (!enc.load_model(aux_enc)) { fprintf(stderr,"enc load: %s\n",enc.get_error().c_str()); return 1; }
        int rsr=0; auto rwav = wav_read(ref_wav, rsr);
        if (rwav.empty()) { fprintf(stderr,"ref wav read failed: %s\n",ref_wav.c_str()); return 1; }
        fprintf(stderr,"ref wav: %zu samples @ %d Hz (%.2fs)\n", rwav.size(), rsr, rwav.size()/(double)rsr);
        if (!enc.encode(rwav.data(), (int)rwav.size(), rcodes, rT)) { fprintf(stderr,"encode: %s\n",enc.get_error().c_str()); return 1; }
        enc.unload_model();   // free encoder VRAM before the backbone loads
        // Bake the encoded ref into a reusable voice file (encode wav ONCE, then
        // clone forever with --codes — no encoder/wav needed again).
        if (!ref_codes_out.empty()) {
            std::vector<float> cf(rcodes.begin(), rcodes.end());
            npy::save_f32(ref_codes_out, cf, {(size_t)rT, (size_t)(rcodes.size()/std::max(1,rT))});
            fprintf(stderr,"saved REF voice codes [%d,8] -> %s\n", rT, ref_codes_out.c_str());
        }
        if (text.empty()) { fprintf(stderr,"encode-only (no --text): done.\n"); return 0; }  // bake-a-voice mode
        fprintf(stderr,"ref codes: T=%d (%.2fs); cloning with%s transcript\n", rT, rT/25.0, ref_text.empty()?"out":"");
    } else if (!codes_in.empty() && !text.empty()) {
        // Voice catalog: ref codes come straight from a saved .npy [T,N] "voice"
        // (produced by --save-codes on an earlier zero-shot/clone run). No encoder,
        // no ref wav, no qwen3-tts — clone from your own generated voice.
        auto cf = npy::load(codes_in); auto fv = cf.as_f32();
        rT = (int)cf.shape[0]; int cn = cf.shape.size()>1 ? (int)cf.shape[1] : 8;
        rcodes.resize(fv.size());
        for (size_t i=0;i<fv.size();++i){ int v=(int)lrintf(fv[i]); rcodes[i] = v<0?0:(v>1023?1023:v); }
        fprintf(stderr,"voice codes loaded: T=%d N=%d from %s\n", rT, cn, codes_in.c_str());
    }

    higgs::HiggsTTS tts;
    if (!tts.load(bb, aux, n_ctx)) { fprintf(stderr,"load: %s\n",tts.get_error().c_str()); return 1; }

    ggml_backend_dev_t dev = ggml_backend_dev_get(0);
    size_t free_b=0, total_b=0;
    if (dev) ggml_backend_dev_memory(dev, &free_b, &total_b);
    double load_mib = total_b ? (total_b - free_b)/(1024.0*1024.0) : 0;

    bool do_stream = argval(argc,argv,"--stream",nullptr) != nullptr;
    higgs::gen_result r;
    if (!raw_prompt.empty()) {
        auto ids = tts.tokenize_raw(raw_prompt);
        printf("raw prompt ids (%zu):", ids.size());
        for (int id : ids) printf(" %d", id);
        printf("\n");
        if (!tts.synthesize_from_ids(ids, gp, r, true)) { fprintf(stderr,"synth: %s\n",tts.get_error().c_str()); return 1; }
    } else if (rT > 0 && !text.empty()) {
        // ref codes were encoded above (before the backbone loaded); now clone.
        if (!tts.synthesize_with_ref(text, rcodes.data(), rT, ref_text, gp, r, true)) {
            fprintf(stderr,"clone: %s\n",tts.get_error().c_str()); return 1; }
    } else if (do_long && !text.empty()) {
        auto chunks = higgs::HiggsTTS::prepare_chunk_text(text, chunk_words);
        fprintf(stderr, "long-form: %zu chunks (buffer=%d, temp=%.2f, ras_win=%d)\n",
                chunks.size(), buffer, gp.temperature, gp.ras_win_len);
        int nchunks = 0;
        if (!tts.synthesize_long(text, gp, buffer, chunk_words,
              [&](const float*, int n, bool fin){ ++nchunks;
                  fprintf(stderr, "  emitted chunk %d: %d samples%s\n", nchunks, n, fin?" [final]":""); },
              r)) { fprintf(stderr,"long: %s\n",tts.get_error().c_str()); return 1; }
    } else if (do_stream && !text.empty()) {
        int nchunks = 0; size_t tot = 0;
        if (!tts.synthesize_stream(text, gp, 25,
              [&](const float*, int n, bool fin){ ++nchunks; tot += n;
                  fprintf(stderr, "  chunk %d: %d samples (%.2fs)%s\n", nchunks, n, tot/24000.0, fin?" [final]":""); },
              r)) { fprintf(stderr,"stream: %s\n",tts.get_error().c_str()); return 1; }
        printf("streamed in %d chunks, %zu samples\n", nchunks, tot);
    } else if (!text.empty()) {
        auto pids = tts.build_prompt(text);
        printf("prompt ids (%zu):", pids.size());
        for (int id : pids) printf(" %d", id);
        printf("\n");
        if (!tts.synthesize(text, gp, r)) { fprintf(stderr,"synth: %s\n",tts.get_error().c_str()); return 1; }
    } else {
        auto ids = npy::load(ids_f).as_i32();
        if (!tts.synthesize_from_ids(std::vector<int32_t>(ids.begin(),ids.end()), gp, r, true)) {
            fprintf(stderr,"synth: %s\n",tts.get_error().c_str()); return 1;
        }
    }
    wav_write(out, r.pcm, 24000);

    // Save this run's codes as a reusable "voice" (load later with --codes).
    if (!codes_out.empty() && r.T > 0 && !r.codes_TN.empty()) {
        int N = (int)(r.codes_TN.size() / r.T);
        std::vector<float> cf(r.codes_TN.begin(), r.codes_TN.end());
        npy::save_f32(codes_out, cf, {(size_t)r.T, (size_t)N});
        fprintf(stderr, "saved voice codes [%d,%d] -> %s\n", r.T, N, codes_out.c_str());
    }

    double audio_sec = r.T / 25.0;
    double wall_sec = (r.prefill_ms + r.decode_ms + r.codec_ms) / 1000.0;
    double rtf = wall_sec > 0 ? audio_sec / wall_sec : 0;
    double ttfa_ms = r.prefill_ms + (r.steps>0 ? r.decode_ms / r.steps : 0);
    size_t free2=0,total2=0; if (dev) ggml_backend_dev_memory(dev,&free2,&total2);
    double peak_mib = total2 ? (total2 - free2)/(1024.0*1024.0) : 0;
    printf("OUT=%s steps=%d T=%d audio_sec=%.2f wall_sec=%.2f rtf=%.2f ttfa_ms=%.0f vram_load_mib=%.0f vram_peak_mib=%.0f\n",
           out.c_str(), r.steps, r.T, audio_sec, wall_sec, rtf, ttfa_ms, load_mib, peak_mib);
    return 0;
}
