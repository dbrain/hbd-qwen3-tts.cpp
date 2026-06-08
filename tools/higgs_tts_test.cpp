// Validate the full Higgs AR loop (LM + delay/EOC sampler + codec) vs the
// PyTorch oracle. Greedy decode must reproduce full_codes.npy exactly.
//
//   higgs_tts_test <backbone.gguf> <aux.gguf> <ref/out dir>

#include "higgs_tts.h"
#include "npy.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
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

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr,"usage: %s <backbone.gguf> <aux.gguf> <ref-out-dir>\n",argv[0]); return 2; }
    std::string bb=argv[1], aux=argv[2], dir=argv[3];
    auto P=[&](const char* f){ return dir+"/"+f; };

    higgs::HiggsTTS tts;
    if (!tts.load(bb, aux, 4096)) { fprintf(stderr,"load: %s\n",tts.get_error().c_str()); return 1; }
    tts.lm().log_vram("loaded");

    auto ids = npy::load(P("full_prompt_ids.npy")).as_i32();
    const int N = tts.lm().get_config().n_codebooks, V = tts.lm().get_config().cb_vocab;

    // ---- TEACHER-FORCED validation (isolates decode path from greedy drift) ----
    // Feed the oracle's delayed codes each step; check my per-step argmax of the
    // REAL codebooks (oracle code < 1024) matches the oracle's next-frame codes.
    {
        auto od = npy::load(P("full_codes_delayed.npy")); auto teach = od.as_i32();
        int oL = (int)od.shape[0];
        tts.lm().reset();
        std::vector<float> logits;
        if (!tts.lm().prefill(ids.data(), (int)ids.size(), logits)) { fprintf(stderr,"tf prefill\n"); return 1; }
        int real=0, real_match=0, cb0_real=0, cb0_match=0;
        for (int s = 0; s < oL; ++s) {
            for (int c = 0; c < N; ++c) {
                int oc = teach[(size_t)s*N+c];
                if (oc >= 1024) continue;             // BOC/EOC forced, not predicted
                const float * lg = &logits[(size_t)c*V];
                int am=0; float bv=lg[0];
                for (int v=1; v<V; ++v) if (lg[v]>bv){bv=lg[v];am=v;}
                ++real; real_match += (am==oc);
                if (c==0){ ++cb0_real; cb0_match += (am==oc); }
            }
            if (s+1 < oL) {
                std::vector<int32_t> fc(teach.begin()+(size_t)s*N, teach.begin()+(size_t)(s+1)*N);
                if (!tts.lm().decode_step(fc.data(), logits)) { fprintf(stderr,"tf step %d\n",s); return 1; }
            }
        }
        printf("TEACHER-FORCED argmax match: all-real %d/%d (%.1f%%)  cb0 %d/%d (%.1f%%)\n",
               real_match, real, 100.0*real_match/real, cb0_match, cb0_real, 100.0*cb0_match/cb0_real);
    }

    printf("\n---- free-running greedy ----\n");
    higgs::gen_params gp; gp.temperature = 0.0f; gp.max_new = 1024;
    higgs::gen_result r;
    if (!tts.synthesize_from_ids(std::vector<int32_t>(ids.begin(),ids.end()), gp, r, true)) {
        fprintf(stderr,"synth: %s\n",tts.get_error().c_str()); return 1;
    }
    printf("steps=%d  T=%d  (%.2fs audio)  prefill=%.0fms decode=%.0fms codec=%.0fms\n",
           r.steps, r.T, r.T/25.0, r.prefill_ms, r.decode_ms, r.codec_ms);

    // dump first delayed rows vs oracle full_codes_delayed.npy
    {
        auto od = npy::load(P("full_codes_delayed.npy")); auto odc = od.as_i32();
        int oL = (int)od.shape[0];
        printf("delayed grid (mine L=%d, oracle L=%d):\n", r.L, oL);
        int rows = std::min({r.L, oL, 12});
        for (int t=0;t<rows;++t){
            printf("  t%2d mine[",t); for(int c=0;c<N;++c) printf("%4d ", r.delayed_flat[(size_t)t*N+c]);
            printf("] ora["); for(int c=0;c<N;++c) printf("%4d ", odc[(size_t)t*N+c]); printf("]\n");
        }
        // full delayed-grid diff
        int dmis=0, first=-1, cmpL=std::min(r.L,oL);
        for (int t=0;t<cmpL;++t) for(int c=0;c<N;++c)
            if (r.delayed_flat[(size_t)t*N+c]!=odc[(size_t)t*N+c]){ ++dmis; if(first<0)first=t; }
        printf("  delayed-grid mismatches: %d / %d  (first at t=%d)\n", dmis, cmpL*N, first);
    }
    // compare codes vs oracle full_codes.npy [T,N]
    auto ref = npy::load(P("full_codes.npy"));
    int rT = (int)ref.shape[0], rN = (int)ref.shape[1];
    auto refc = ref.as_i32();
    printf("oracle codes [%d,%d]\n", rT, rN);
    bool codes_pass = true;
    if (rT != r.T || rN != N) { printf("  shape mismatch (mine T=%d N=%d)\n", r.T, N); codes_pass = false; }
    int mism = 0, cmpT = std::min(rT, r.T);
    for (int t=0;t<cmpT;++t) for (int c=0;c<N;++c)
        if (r.codes_TN[(size_t)t*N+c] != refc[(size_t)t*rN+c]) ++mism;
    if (mism) codes_pass = false;
    printf("  code mismatches: %d / %d  %s\n", mism, cmpT*N, codes_pass?"PASS":"FAIL");

    // audio: compare to golden codes decoded by our (validated) codec — i.e.
    // primary gate is codes; audio is a sanity rms/peak print.
    double rms=0, pk=0; for (float s: r.pcm){ rms+=(double)s*s; pk=std::max(pk,(double)std::fabs(s)); }
    rms = std::sqrt(rms/(r.pcm.size()?r.pcm.size():1));
    printf("  audio: %zu samples rms=%.4f peak=%.4f\n", r.pcm.size(), rms, pk);

    wav_write(P("higgs_tts_cpp.wav"), r.pcm, 24000);
    printf("wrote %s\n", P("higgs_tts_cpp.wav").c_str());
    printf("\n==== %s ====\n", codes_pass ? "PASS" : "FAIL");
    return codes_pass ? 0 : 1;
}
