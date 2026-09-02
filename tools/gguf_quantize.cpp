// gguf_quantize.cpp : GGUF -> GGUF requantizer for this repo's flat layouts.
//
// Ported from acestep.cpp/tools/quantize.cpp (same author, same ggml). Two
// things changed: the imatrix path is dropped (we have no importance matrix for
// these models), and the tensor-name policy speaks our flat naming --
// `<subsystem>.blk.N.<role>.weight` -- instead of HF's `model.layers.N.*`.
//
// It exists because gguf-py cannot WRITE K-quants (only Q4_0/Q4_1/Q5_0/Q5_1/
// Q8_0), which is how a legacy Q4_0 -- the weakest 4-bit format there is,
// weight relerr 0.086 against Q4_K's ~0.03 -- once shipped as a default.
// House standard is Q8_0 or a K-quant, never a legacy qN_0.
//
// The INPUT MUST BE F16/BF16/F32: ggml_quantize_chunk reads float rows, so a
// q8_0 GGUF cannot be requantized. Convert once at --type f16, keep that as the
// master, and cut every shipping variant from it.
//
// Usage: gguf_quantize <input.gguf> <output.gguf> <type> [--set <prefix>=<type> ...]
// Types: Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0
//
// --set overrides the variant for one tensor-name prefix, so the subsystems can
// be dialled independently:
//   --set depth.=Q4_K_M --set backbone.=Q5_K_M
// The depth decoder runs 15x per frame and is ~79% of AR time, so it is the one
// worth spending bits differently from the rest.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ggml.h"
#include "gguf.h"

// Quant variant: base type + optional bump rules for important tensors.
struct QuantVariant {
    const char *   name;
    enum ggml_type base;
    enum ggml_type bump;   // type for "important" tensors (or COUNT = no bump)
    enum ggml_type embed;  // type for embeddings / output heads
    // bump_mode: 0=none, 1=first N layers, 2=first+last+every 3rd, 3=all important
    int            bump_mode;
    int            bump_n;
};

static const QuantVariant VARIANTS[] = {
    // name       base            bump             embed           mode  n
    { "Q2_K",   GGML_TYPE_Q2_K, GGML_TYPE_Q4_K,  GGML_TYPE_Q6_K, 1, 4 },
    { "Q3_K_S", GGML_TYPE_Q3_K, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0 },
    { "Q3_K_M", GGML_TYPE_Q3_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, 2, 0 },
    { "Q3_K_L", GGML_TYPE_Q3_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, 3, 0 },
    { "Q4_K_S", GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, 1, 4 },
    { "Q4_K_M", GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,  GGML_TYPE_Q6_K, 2, 0 },
    { "Q5_K_S", GGML_TYPE_Q5_K, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0 },
    { "Q5_K_M", GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,  GGML_TYPE_Q6_K, 2, 0 },
    { "Q6_K",   GGML_TYPE_Q6_K, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0 },
    { "Q8_0",   GGML_TYPE_Q8_0, GGML_TYPE_COUNT, GGML_TYPE_Q8_0, 0, 0 },
};

static const QuantVariant * find_variant(const char * s) {
    for (const auto & v : VARIANTS) {
        if (strcasecmp(s, v.name) == 0) return &v;
    }
    return nullptr;
}

// Per-prefix variant overrides, longest prefix wins so "depth.blk" can be more
// specific than "depth.".
struct PrefixRule { std::string prefix; const QuantVariant * v; };
static std::vector<PrefixRule> g_rules;

static const QuantVariant & variant_for(const char * name, const QuantVariant & base) {
    const QuantVariant * best = &base;
    size_t best_len = 0;
    for (const auto & r : g_rules) {
        if (r.prefix.size() > best_len && strncmp(name, r.prefix.c_str(), r.prefix.size()) == 0) {
            best = r.v; best_len = r.prefix.size();
        }
    }
    return *best;
}

static bool has(const char * name, const char * needle) {
    return strstr(name, needle) != nullptr;
}

static bool strends(const char * s, const char * suffix) {
    const size_t n = strlen(s), m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

// `<subsystem>.blk.N.` / `<subsystem>.pre_tfm.blk.N.` -> N, else -1.
static int extract_layer(const char * name) {
    const char * p = strstr(name, "blk.");
    return p ? atoi(p + 4) : -1;
}

// llama-quantize's rule, translated: value projections and the FFN down
// projection carry more than their share and get bumped a rung in S/M.
static bool is_important_sm(const char * name) {
    return has(name, "attn_v.weight") || has(name, "ffn_down.weight");
}
static bool is_important_l(const char * name) {
    return is_important_sm(name) || has(name, "attn_output.weight");
}

// Embeddings and output heads: one rung better than the body.
static bool is_embed_or_head(const char * name) {
    return has(name, "token_embd.weight") || has(name, "lm_head.weight") ||
           has(name, "codebooks_head.weight");
}

// Tensors that must never be touched.
static bool should_quantize(const char * name, int n_dims) {
    if (n_dims < 2) return false;                 // norms, biases, alphas
    // The vocoder and codec encoder own their own precision policy: conv kernels
    // must stay F16 (conv-1d-direct.cu asserts it), codebooks and the Snake /
    // ConvNeXt / LayerScale per-channel params are tiny and quantisation-
    // sensitive, and the decoder already quantizes its own mat-muls at load.
    if (!strncmp(name, "tok_dec.", 8) || !strncmp(name, "tok_enc.", 8)) return false;
    // Note the trailing dot on ".codebook.": without it this also matches
    // "depth.codebooks_head.weight", a 31.5M-parameter output head that would
    // silently stay F16 in every variant.
    if (has(name, ".conv") || has(name, ".codebook.") || strends(name, ".codebook") ||
        has(name, ".usage")) return false;
    if (has(name, "norm") || has(name, "_scale") || has(name, ".alpha") ||
        has(name, ".beta") || has(name, ".gamma")) return false;
    // eoi_embd is a single 1152-vector read directly, not a mat-mul weight.
    if (has(name, "eoi_embd")) return false;
    return true;
}

static enum ggml_type pick_type(const char * name, int n_dims,
                                const QuantVariant & v, int n_layers) {
    if (!should_quantize(name, n_dims)) return GGML_TYPE_COUNT;

    if (is_embed_or_head(name)) {
        return (v.embed != GGML_TYPE_COUNT) ? v.embed : v.base;
    }

    const bool important = (v.bump_mode == 3) ? is_important_l(name) : is_important_sm(name);
    if (important && v.bump != GGML_TYPE_COUNT) {
        const int layer = extract_layer(name);
        bool bumped = false;
        switch (v.bump_mode) {
            case 1: bumped = (layer >= 0 && layer < v.bump_n); break;
            case 2: {
                const int ql = n_layers > 0 ? n_layers : 28;
                bumped = (layer >= 0) &&
                         (layer < ql / 9 || layer >= ql - ql / 7 || layer % 3 == 0);
                break;
            }
            case 3: bumped = true; break;
        }
        if (bumped) return v.bump;
    }
    return v.base;
}

static bool to_f32(const void * src, float * dst, int64_t n, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_BF16: ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, n); return true;
        case GGML_TYPE_F16:  ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n); return true;
        case GGML_TYPE_F32:  memcpy(dst, src, (size_t) n * sizeof(float));             return true;
        default: return false;
    }
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <type> [--set <prefix>=<type> ...]\n", argv[0]);
        fprintf(stderr, "Types:");
        for (const auto & v : VARIANTS) fprintf(stderr, " %s", v.name);
        fprintf(stderr, "\nInput must be F16/BF16/F32 -- a quantized GGUF cannot be requantized.\n");
        return 1;
    }

    const char * inp_path = argv[1];
    const char * out_path = argv[2];
    const QuantVariant * variant = find_variant(argv[3]);
    if (!variant) { fprintf(stderr, "[quantize] unknown type: %s\n", argv[3]); return 1; }

    for (int i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "--set") != 0 || i + 1 >= argc) {
            fprintf(stderr, "[quantize] unexpected argument: %s\n", argv[i]); return 1;
        }
        const char * spec = argv[++i];
        const char * eq = strchr(spec, '=');
        if (!eq) { fprintf(stderr, "[quantize] --set wants <prefix>=<type>, got %s\n", spec); return 1; }
        const QuantVariant * pv = find_variant(eq + 1);
        if (!pv) { fprintf(stderr, "[quantize] unknown type in --set: %s\n", eq + 1); return 1; }
        g_rules.push_back({ std::string(spec, (size_t)(eq - spec)), pv });
        fprintf(stderr, "[quantize]   override %.*s* -> %s\n", (int)(eq - spec), spec, pv->name);
    }

    int fd = open(inp_path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    const size_t file_size = (size_t) st.st_size;
    void * mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    struct ggml_context * meta = nullptr;
    struct gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/&meta };
    struct gguf_context * inp = gguf_init_from_file(inp_path, params);
    if (!inp) {
        fprintf(stderr, "[quantize] failed to read %s\n", inp_path);
        munmap(mapping, file_size); close(fd);
        return 1;
    }

    const size_t data_off  = gguf_get_data_offset(inp);
    const int    n_tensors = (int) gguf_get_n_tensors(inp);

    char arch[64] = "unknown";
    if (int64_t idx = gguf_find_key(inp, "general.architecture"); idx >= 0) {
        snprintf(arch, sizeof(arch), "%s", gguf_get_val_str(inp, (int) idx));
    }
    int n_layers = 0;
    {
        char key[128];
        snprintf(key, sizeof(key), "%s.backbone.block_count", arch);
        int64_t idx = gguf_find_key(inp, key);
        if (idx < 0) {
            snprintf(key, sizeof(key), "%s.block_count", arch);
            idx = gguf_find_key(inp, key);
        }
        if (idx >= 0) n_layers = (int) gguf_get_val_u32(inp, (int) idx);
    }
    fprintf(stderr, "[quantize] %s -> %s (%s)  arch=%s layers=%d\n",
            inp_path, out_path, variant->name, arch, n_layers);

    struct gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, inp);
    gguf_set_val_u32(out, "general.quantization_version", 2);
    {
        std::string ft = variant->name;
        for (const auto & r : g_rules) ft += "+" + r.prefix + r.v->name;
        gguf_set_val_str(out, "general.file_type", ft.c_str());
    }

    struct TensorPlan { enum ggml_type target; bool quantize; bool promote; };
    std::vector<TensorPlan> plans((size_t) n_tensors);
    int n_fallback = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(inp, i);
        struct ggml_tensor * t = ggml_get_tensor(meta, name);
        const int n_dims = ggml_n_dims(t);

        gguf_add_tensor(out, t);
        plans[(size_t) i] = { GGML_TYPE_COUNT, false, false };

        const enum ggml_type target = pick_type(name, n_dims, variant_for(name, *variant), n_layers);

        if (target == GGML_TYPE_COUNT) {
            // Norms and biases go to F32; they cost almost nothing and the
            // graphs read them at full precision anyway.
            if (n_dims < 2 && (t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_F16)) {
                gguf_set_tensor_type(out, name, GGML_TYPE_F32);
                plans[(size_t) i] = { GGML_TYPE_F32, false, true };
            }
            continue;
        }

        const bool can_convert = (t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_F16 ||
                                  t->type == GGML_TYPE_F32);
        // K-quants are 256-wide, so a row length like the text encoder's 1152
        // (or its 262158-entry embedding) cannot take one. There is no 4- or
        // 5-bit K-quant with a 32-wide block, and legacy qN_0 is not an option
        // we accept, so those tensors fall back to Q8_0 rather than F16 --
        // half the size, and the only sanctioned non-K format.
        enum ggml_type eff = target;
        if (can_convert && t->ne[0] % ggml_blck_size(eff) != 0 &&
            t->ne[0] % ggml_blck_size(GGML_TYPE_Q8_0) == 0) {
            eff = GGML_TYPE_Q8_0;
            n_fallback++;
        }
        const bool aligned = (t->ne[0] % ggml_blck_size(eff) == 0);
        if (can_convert && aligned) {
            gguf_set_tensor_type(out, name, eff);
            plans[(size_t) i] = { eff, true, false };
        } else if (!can_convert) {
            fprintf(stderr, "[quantize] WARNING %s is %s, not a float type -- passed through. "
                    "Requantize from an f16 master, not a quantized GGUF.\n",
                    name, ggml_type_name(t->type));
        }
    }

    if (!gguf_write_to_file(out, out_path, true)) {
        fprintf(stderr, "[quantize] failed to write metadata %s\n", out_path);
        return 1;
    }
    FILE * fout = fopen(out_path, "ab");
    if (!fout) { fprintf(stderr, "[quantize] cannot append to %s\n", out_path); return 1; }

    const size_t alignment = gguf_get_alignment(out);
    int n_quantized = 0, n_promoted = 0;
    int64_t bytes_in = 0, bytes_out = 0;
    size_t data_pos = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(inp, i);
        struct ggml_tensor * t = ggml_get_tensor(meta, name);
        const int64_t nel = ggml_nelements(t);
        const size_t src_size = ggml_nbytes(t);
        const void * src = (const uint8_t *) mapping + data_off + gguf_get_tensor_offset(inp, i);
        bytes_in += (int64_t) src_size;

        const size_t pad = (alignment - (data_pos % alignment)) % alignment;
        if (pad > 0) {
            uint8_t zeros[64] = {};
            fwrite(zeros, 1, pad, fout);
            data_pos += pad;
        }

        const TensorPlan & plan = plans[(size_t) i];
        if (plan.promote) {
            std::vector<float> f32((size_t) nel);
            to_f32(src, f32.data(), nel, t->type);
            const size_t n = (size_t) nel * sizeof(float);
            fwrite(f32.data(), 1, n, fout);
            data_pos += n; bytes_out += (int64_t) n; n_promoted++;
        } else if (plan.quantize) {
            std::vector<float> f32((size_t) nel);
            to_f32(src, f32.data(), nel, t->type);
            const int64_t n_per_row = t->ne[0];
            const int64_t nrows = nel / n_per_row;
            const size_t qsize = ggml_row_size(plan.target, n_per_row) * (size_t) nrows;
            std::vector<uint8_t> qbuf(qsize);
            ggml_quantize_chunk(plan.target, f32.data(), qbuf.data(), 0, nrows, n_per_row, nullptr);
            fwrite(qbuf.data(), 1, qsize, fout);
            data_pos += qsize; bytes_out += (int64_t) qsize; n_quantized++;
        } else {
            fwrite(src, 1, src_size, fout);
            data_pos += src_size; bytes_out += (int64_t) src_size;
        }
    }
    fclose(fout);

    fprintf(stderr, "[quantize] quantized %d/%d tensors, promoted %d to F32, "
            "%d fell back to Q8_0 (row not a multiple of 256)\n",
            n_quantized, n_tensors, n_promoted, n_fallback);
    fprintf(stderr, "[quantize] %.2f GB -> %.2f GB (%.2fx)\n",
            (double) bytes_in / 1e9, (double) bytes_out / 1e9,
            bytes_out > 0 ? (double) bytes_in / (double) bytes_out : 0.0);

    gguf_free(out); gguf_free(inp); ggml_free(meta);
    munmap(mapping, file_size); close(fd);
    return 0;
}
