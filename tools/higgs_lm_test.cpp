// Validate the C++ Higgs LM backbone step-0 audio logits vs the PyTorch oracle.
//
//   higgs_lm_test <backbone.gguf> <aux.gguf> <ref/out dir>
//
// Loads full_prompt_ids.npy, prefills, computes audio logits [8,1026], and
// compares per-codebook argmax + cosine against full_logits0.npy.

#include "higgs_lm.h"
#include "npy.h"
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <backbone.gguf> <aux.gguf> <ref-out-dir>\n", argv[0]); return 2; }
    std::string bb = argv[1], aux = argv[2], dir = argv[3];
    auto P = [&](const char * f){ return dir + "/" + f; };

    higgs::HiggsLM lm;
    if (!lm.load_model(bb, aux, 4096)) { fprintf(stderr, "load: %s\n", lm.get_error().c_str()); return 1; }
    lm.log_vram("loaded");
    const auto & cfg = lm.get_config();
    const int N = cfg.n_codebooks, V = cfg.cb_vocab;

    auto ids = npy::load(P("full_prompt_ids.npy")).as_i32();
    printf("prompt: %zu tokens\n", ids.size());

    std::vector<float> logits;
    if (!lm.prefill(ids.data(), (int)ids.size(), logits)) { fprintf(stderr, "prefill: %s\n", lm.get_error().c_str()); return 1; }
    printf("logits: %zu (expect %d)\n", logits.size(), N*V);

    auto oracle = npy::load(P("full_logits0.npy")).as_f32();  // [8,1026]

    int argmax_match = 0;
    double mean_cos = 0;
    printf("cb |  my_argmax  ref_argmax  cosine\n");
    for (int cb = 0; cb < N; ++cb) {
        const float * mine = &logits[(size_t)cb*V];
        const float * ref  = &oracle[(size_t)cb*V];
        int ai=0, ar=0; float bm=-1e30f, br=-1e30f;
        double dot=0, nm=0, nr=0;
        for (int v=0; v<V; ++v) {
            if (mine[v] > bm) { bm=mine[v]; ai=v; }
            if (ref[v]  > br) { br=ref[v];  ar=v; }
            dot += (double)mine[v]*ref[v]; nm += (double)mine[v]*mine[v]; nr += (double)ref[v]*ref[v];
        }
        double cos = (nm>0&&nr>0) ? dot/std::sqrt(nm*nr) : 0;
        mean_cos += cos;
        bool ok = (ai == ar);
        argmax_match += ok;
        printf("%2d |  %8d   %8d   %.4f  %s\n", cb, ai, ar, cos, ok?"":" <-- MISMATCH");
    }
    mean_cos /= N;
    bool pass = (argmax_match == N);
    printf("\nargmax match %d/%d  mean cosine %.4f  ==== %s ====\n",
           argmax_match, N, mean_cos, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
