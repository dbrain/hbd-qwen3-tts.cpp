// Validate the C++ Higgs codec ENCODER against the PyTorch oracle.
//
//   higgs_encode_test <aux-enc.gguf> <ref/out dir>
//
// Loads enc_input.npy [1,1,L] (preprocessed 24 kHz wav), runs HiggsCodecEncoder,
// and diffs every stage against the enc_*.npy oracles (cosine + max abs diff),
// plus exact code-match % vs enc_codes.npy.
//
// Mirrors tools/higgs_codec_test.cpp.

#include "higgs_encode.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

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

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <aux-enc.gguf> <ref-out-dir>\n", argv[0]); return 2; }
    std::string gguf = argv[1];
    std::string dir  = argv[2];
    auto P = [&](const char * f){ return dir + "/" + f; };

    higgs::HiggsCodecEncoder enc;
    if (!enc.load_model(gguf)) { fprintf(stderr, "load failed: %s\n", enc.get_error().c_str()); return 1; }
    const auto & cfg = enc.get_config();

    // enc_input.npy [1,1,L] @ 24 kHz
    npy::Array in_npy = npy::load(P("enc_input.npy"));
    std::vector<float> wav = in_npy.as_f32();
    int L = (int)wav.size();
    printf("enc_input: L=%d @ %d Hz\n", L, cfg.sample_rate);

    std::vector<int32_t> codes; int T = 0;
    std::vector<float> semfeat, semantic, acoustic, prefc, postfc;
    if (!enc.encode_debug(wav.data(), L, codes, T, semfeat, semantic, acoustic, prefc, postfc)) {
        fprintf(stderr, "encode failed: %s\n", enc.get_error().c_str()); return 1;
    }
    printf("T=%d  L/T=%.2f (expect ~%d)\n", T, (double)L/std::max(1,T), cfg.hop_length);

    bool pass = true;

    auto check = [&](const char * tag, const char * file, const std::vector<float> & got) {
        auto ref = npy::load(P(file)).as_f32();
        Diff d = diff(got, ref);
        bool ok = d.cosine > 0.99 && got.size() == ref.size();
        pass &= ok;
        printf("[%-9s] n_got=%zu n_ref=%zu  max_abs=%.5f mean_abs=%.5f cosine=%.6f  rms ref=%.4f got=%.4f  %s\n",
               tag, got.size(), ref.size(), d.max_abs, d.mean_abs, d.cosine, d.rms_ref, d.rms_got,
               ok?"PASS":"FAIL");
    };

    // semfeat is row-major [T_sem,768]; oracle enc_semfeat.npy is [1,T_sem,768] -> same flat order.
    check("semfeat",  "enc_semfeat.npy",  semfeat);
    // semantic/acoustic/prefc/postfc are channel-major [C,T] (idx=c*T+t) == oracle [1,C,T].
    check("semantic", "enc_semantic.npy", semantic);
    check("acoustic", "enc_acoustic.npy", acoustic);
    check("prefc",    "enc_prefc.npy",    prefc);
    check("postfc",   "enc_postfc.npy",   postfc);
    npy::save_f32(P("enc_prefc_cpp.npy"),  prefc,  {(size_t)1024, (size_t)T});
    npy::save_f32(P("enc_postfc_cpp.npy"), postfc, {(size_t)1024, (size_t)T});
    { std::vector<float> cf(codes.begin(), codes.end());
      npy::save_f32(P("enc_codes_cpp.npy"), cf, {(size_t)T, (size_t)8}); }

    // codes: oracle enc_codes.npy [T,8] int64 (row-major). ours codes [T,8] int32.
    {
        npy::Array codes_npy = npy::load(P("enc_codes.npy"));
        std::vector<int32_t> ref = codes_npy.as_i32();
        int Tref = (int)codes_npy.shape[0];
        int N = (int)codes_npy.shape[1];
        size_t n = std::min(codes.size(), ref.size());
        size_t match = 0;
        for (size_t i = 0; i < n; ++i) if (codes[i] == ref[i]) ++match;
        double pct = n ? 100.0 * (double)match / (double)n : 0.0;
        bool ok = (Tref == T) && (N == cfg.n_codebooks) && pct >= 99.0;
        pass &= ok;
        printf("[codes    ] T=%d/%d N=%d  match=%zu/%zu = %.2f%%  %s\n",
               T, Tref, N, match, n, pct, ok?"PASS":"FAIL");
        // dump a few mismatches for debugging
        int shown = 0;
        for (size_t i = 0; i < n && shown < 8; ++i) {
            if (codes[i] != ref[i]) {
                printf("    mismatch [t=%zu cb=%zu] got=%d ref=%d\n",
                       i/cfg.n_codebooks, i%cfg.n_codebooks, codes[i], ref[i]);
                ++shown;
            }
        }
    }

    printf("\n==== %s ====\n", pass ? "ALL PASS" : "FAIL");
    return pass ? 0 : 1;
}
