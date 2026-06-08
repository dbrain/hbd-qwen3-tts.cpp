// Validate the C++ Higgs codec decoder against the PyTorch oracle.
//
//   higgs_codec_test <aux.gguf> <ref/out dir>
//
// Loads codec_codes.npy [T,8], decodes via HiggsCodecDecoder, and diffs the
// result against codec_audio.npy / codec_postrvq.npy / codec_postfc2.npy.
// Writes higgs_codec_cpp.wav + .npy next to the oracles for ear/eye checks.

#include "higgs_codec.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static void wav_write(const std::string & path, const std::vector<float> & pcm, int sr) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return;
    auto u32 = [&](uint32_t v){ fwrite(&v,4,1,f); };
    auto u16 = [&](uint16_t v){ fwrite(&v,2,1,f); };
    uint32_t n = (uint32_t)pcm.size();
    uint32_t data_bytes = n * 2;
    fwrite("RIFF",1,4,f); u32(36 + data_bytes); fwrite("WAVE",1,4,f);
    fwrite("fmt ",1,4,f); u32(16); u16(1); u16(1); u32(sr); u32(sr*2); u16(2); u16(16);
    fwrite("data",1,4,f); u32(data_bytes);
    for (float s : pcm) {
        int v = (int)lrintf(s * 32767.0f);
        if (v > 32767) v = 32767; if (v < -32768) v = -32768;
        int16_t s16 = (int16_t)v; fwrite(&s16,2,1,f);
    }
    fclose(f);
}

struct Diff { double max_abs, mean_abs, cosine, rms_ref, rms_got; };

static Diff diff(const std::vector<float> & got, const std::vector<float> & ref) {
    Diff d{0,0,0,0,0};
    size_t n = std::min(got.size(), ref.size());
    double s_ab=0, dot=0, ng=0, nr=0;
    for (size_t i=0;i<n;++i){
        double e=std::fabs((double)got[i]-(double)ref[i]);
        d.max_abs=std::max(d.max_abs,e); s_ab+=e;
        dot+=(double)got[i]*ref[i]; ng+=(double)got[i]*got[i]; nr+=(double)ref[i]*ref[i];
    }
    d.mean_abs=s_ab/(n?n:1);
    d.cosine=(ng>0&&nr>0)?dot/std::sqrt(ng*nr):0;
    d.rms_ref=std::sqrt(nr/(n?n:1)); d.rms_got=std::sqrt(ng/(n?n:1));
    return d;
}

// transpose ggml channel-major [C,T] (idx=t*C+c) -> oracle [C,T] (idx=c*T+t)
static std::vector<float> to_ct(const std::vector<float> & ggml_ct, int C, int T) {
    std::vector<float> out((size_t)C*T);
    for (int t=0;t<T;++t) for (int c=0;c<C;++c) out[(size_t)c*T+t]=ggml_ct[(size_t)t*C+c];
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <aux.gguf> <ref-out-dir>\n", argv[0]); return 2; }
    std::string gguf = argv[1];
    std::string dir  = argv[2];
    auto P = [&](const char * f){ return dir + "/" + f; };

    higgs::HiggsCodecDecoder dec;
    if (!dec.load_model(gguf)) { fprintf(stderr, "load failed: %s\n", dec.get_error().c_str()); return 1; }
    const auto & cfg = dec.get_config();

    npy::Array codes_npy = npy::load(P("codec_codes.npy"));   // [T, 8] int64
    int T = (int)codes_npy.shape[0];
    int N = (int)codes_npy.shape[1];
    printf("codes: [%d, %d]  (expect N=%d)\n", T, N, cfg.n_codebooks);
    std::vector<int32_t> codes = codes_npy.as_i32();          // row-major [T,N]

    std::vector<float> samples, post_rvq, post_fc2;
    if (!dec.decode_debug(codes.data(), T, samples, post_rvq, post_fc2)) {
        fprintf(stderr, "decode failed: %s\n", dec.get_error().c_str()); return 1;
    }
    printf("decoded: %zu samples (L/T=%.1f, expect 960)\n", samples.size(), (double)samples.size()/T);

    bool pass = true;

    // audio
    {
        auto ref = npy::load(P("codec_audio.npy")).as_f32();
        Diff d = diff(samples, ref);
        bool ok = d.cosine > 0.999 && d.max_abs < 0.05;
        pass &= ok;
        printf("[audio]   n=%zu  max_abs=%.6f mean_abs=%.6f cosine=%.6f  rms ref=%.4f got=%.4f  %s\n",
               ref.size(), d.max_abs, d.mean_abs, d.cosine, d.rms_ref, d.rms_got, ok?"PASS":"FAIL");
    }
    // intermediates: try both layout interpretations of the ggml dump and
    // accept the better-matching one (settles channel-major vs time-major).
    auto check_inter = [&](const char * tag, const char * file,
                           const std::vector<float> & raw, int C) {
        auto ref = npy::load(P(file)).as_f32();
        Diff d_ct = diff(to_ct(raw, C, T), ref);   // assume ggml idx = t*C+c
        Diff d_id = diff(raw, ref);                 // assume already c*T+t
        const Diff & d = d_ct.cosine >= d_id.cosine ? d_ct : d_id;
        const char * which = d_ct.cosine >= d_id.cosine ? "ct" : "id";
        bool ok = d.cosine > 0.999;
        pass &= ok;
        printf("[%-7s] n=%zu  max_abs=%.6f mean_abs=%.6f cosine=%.6f (%s)  %s\n",
               tag, ref.size(), d.max_abs, d.mean_abs, d.cosine, which, ok?"PASS":"FAIL");
    };
    check_inter("postrvq", "codec_postrvq.npy", post_rvq, cfg.quantizer_dim);
    check_inter("postfc2", "codec_postfc2.npy", post_fc2, cfg.acoustic_dim);

    wav_write(P("higgs_codec_cpp.wav"), samples, cfg.sample_rate);
    npy::save_f32(P("higgs_codec_cpp_audio.npy"), samples, {(int64_t)samples.size()});
    printf("wrote %s\n", P("higgs_codec_cpp.wav").c_str());

    printf("\n==== %s ====\n", pass ? "ALL PASS" : "FAIL");
    return pass ? 0 : 1;
}
