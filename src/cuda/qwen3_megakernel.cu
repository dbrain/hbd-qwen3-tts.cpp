// Qwen3-TTS megakernel — Phase A specialized MMVQ.
// See kobbler/docker/tts-qwen3-dev/HANDOFF-megakernel-v0.md for the full plan.
//
// What lives here:
//   - F32 → Q8_1 row quantizer (templated on K)
//   - Q8_0 × Q8_1 specialized MMVQ for M=1 (templated on K, N)
//   - Hook function that ggml-cuda calls for every mul_mat. Inspects shape
//     and types; if it's one of the 7 specialised (K, N) Q8_0 patterns at
//     M=1, dispatches to our kernel and returns true. Otherwise returns
//     false and ggml's generic path runs.
//
// Phase A v0 only specialises the (K=1024, N=2048) shape end-to-end. The
// remaining 6 shapes will be filled in once the wire-up + kernel pattern
// are validated by microbench + breakit. All shape decisions are made in
// `qwen3_mul_mat_hook`, so adding a shape is one line + one instantiation.

#include "qwen3_megakernel.cuh"
#include "qwen3_megakernel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// We talk to ggml only via the hook signature — we never include ggml-cuda
// internal headers. We DO need to read a few fields off ggml_tensor (type,
// ne[], data) for the shape match, so we declare the minimum subset we use
// here. This couples us to ggml's tensor layout but not to its internals.

extern "C" {
// Mirror of the leading members of struct ggml_tensor that we touch. Layout
// must match ggml.h. If ggml ever reorders these, the hook breaks at compile
// time via the static_assert below.
//
// We use a public ggml.h include instead of mirroring once the build wires
// up — see `extern "C"` block at file scope below for the actual include.
}

#include "ggml.h"

namespace qwen3_megakernel {

// ----------------------------------------------------------------------------
// F32 → Q8_1 quantize (templated on K)
// ----------------------------------------------------------------------------
//
// One block per K/32 Q8_1 block. 32 threads cooperatively scan one block:
// find absmax, derive scale, write int8 quantized values + d (fp16 scale)
// + s (fp16 sum-of-quants * scale).
//
// This mirrors ggml's quantize_row_q8_1 but with K compile-time so the
// block count is a constant.

template <int K>
__global__ void quantize_x_q8_1_kernel(
    const float * __restrict__ x,      // [K]
    block_q8_1 *  __restrict__ y) {    // [K/32]
    constexpr int K_BLOCKS = K / 32;
    static_assert(K % 32 == 0, "K must be a multiple of 32");

    const int kb = blockIdx.x;
    if (kb >= K_BLOCKS) return;

    const int lane = threadIdx.x;  // 0..31
    const float v = x[kb * 32 + lane];

    // Warp absmax reduction
    float amax = fabsf(v);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    }

    const float scale  = amax / 127.0f;
    const float iscale = scale > 0.0f ? 1.0f / scale : 0.0f;

    const int q = __float2int_rn(v * iscale);
    const int qclamped = q < -127 ? -127 : (q > 127 ? 127 : q);

    // Warp sum of quantized ints (for Q8_1's `s` field).
    float qsum = (float) qclamped;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_xor_sync(0xffffffff, qsum, offset);
    }

    y[kb].qs[lane] = (int8_t) qclamped;
    if (lane == 0) {
        y[kb].d = __float2half(scale);
        y[kb].s = __float2half(qsum * scale);
    }
}

template <int K>
void launch_quantize_x_q8_1(
    const float * x,
    block_q8_1 *  y,
    cudaStream_t  stream) {
    constexpr int K_BLOCKS = K / 32;
    quantize_x_q8_1_kernel<K><<<K_BLOCKS, 32, 0, stream>>>(x, y);
}

// Aligned int loads for Q8_0 / Q8_1 quantized blocks.
//
// block_q8_0 = {half d, int8_t qs[32]} = 34 B → block n starts at n*34. The
// `qs` field is at offset 2 inside the block, so its absolute address is
// only 2-byte aligned (alternates with 4-byte across n). Reading 4 bytes
// at a time via int* therefore traps on every other block — exactly the
// "misaligned address" CUDA fault we hit on first wire-up.
//
// ggml's fix is `get_int_b2`: read two u16s and pack into u32. We mirror
// that here. block_q8_1 = {half d, half s, int8_t qs[32]} = 36 B; qs at
// offset 4, always 4-byte aligned, so a plain int load is fine.
__device__ __forceinline__ int load_q8_0_int(const int8_t * qs, int i32) {
    const uint16_t * q16 = reinterpret_cast<const uint16_t *>(qs);
    int v  = q16[2 * i32 + 0] <<  0;
    v     |= q16[2 * i32 + 1] << 16;
    return v;
}

__device__ __forceinline__ int load_q8_1_int(const int8_t * qs, int i32) {
    return reinterpret_cast<const int *>(qs)[i32];
}

// ----------------------------------------------------------------------------
// Q8_0 × Q8_1 specialized MMVQ (M=1, K and N compile-time)
// ----------------------------------------------------------------------------
//
// One warp per output row. Each thread handles one or more K-blocks. Uses
// __dp4a for int8×int8 dot accumulation (matches ggml's pattern). Final
// per-row warp reduction via __shfl_xor_sync.
//
// Layout assumption: w is row-major [N, K/32] block_q8_0, i.e. row n's
// K-blocks are contiguous starting at w[n * K_BLOCKS]. This matches ggml's
// Q8_0 storage for src0 (where ne[0]=K, ne[1]=N).

template <int K, int N>
__global__ void mmvq_q8_0_q8_1_kernel(
    float *            __restrict__ y,        // [N]
    const block_q8_0 * __restrict__ w,        // [N, K/32]
    const block_q8_1 * __restrict__ x_q8_1) { // [K/32]
    constexpr int K_BLOCKS = K / 32;
    static_assert(K % 32 == 0,  "K must be a multiple of 32");
    static_assert(K_BLOCKS >= 32, "K must be >= 1024 (one warp covers 32 K-blocks min)");
    static_assert(K_BLOCKS % 32 == 0, "K_BLOCKS must be a multiple of warp size for v0 layout");

    const int n = blockIdx.x;
    if (n >= N) return;

    const int lane = threadIdx.x;  // 0..31

    // Each lane processes K_BLOCKS / 32 K-blocks.
    constexpr int KB_PER_LANE = K_BLOCKS / 32;

    float acc = 0.0f;

    #pragma unroll
    for (int kbi = 0; kbi < KB_PER_LANE; ++kbi) {
        const int kbx = lane + kbi * 32;
        const block_q8_0 & wb = w[n * K_BLOCKS + kbx];
        const block_q8_1 & xb = x_q8_1[kbx];

        // 8 int loads each side = 32 int8 elements per block. Q8_0's qs is
        // only 2-byte aligned across blocks, so use the b2 (paired-u16) load.
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int v = load_q8_0_int(wb.qs, j);
            const int u = load_q8_1_int(xb.qs, j);
            sumi = __dp4a(v, u, sumi);
        }

        const float d_w = __half2float(wb.d);
        const float d_x = __half2float(xb.d);
        acc += d_w * d_x * (float) sumi;
    }

    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        y[n] = acc;
    }
}

template <int K, int N>
void launch_mmvq_q8_0_q8_1(
    float *            y,
    const block_q8_0 * w,
    const block_q8_1 * x_q8_1,
    cudaStream_t       stream) {
    // One block per output row, 32 threads (one warp).
    mmvq_q8_0_q8_1_kernel<K, N><<<N, 32, 0, stream>>>(y, w, x_q8_1);
}

template <int K, int N>
void launch_mmvq_q8_0_f32(
    float *            y,
    const block_q8_0 * w,
    const float *      x,
    block_q8_1 *       x_q8_1_scratch,
    cudaStream_t       stream) {
    launch_quantize_x_q8_1<K>(x, x_q8_1_scratch, stream);
    launch_mmvq_q8_0_q8_1<K, N>(y, w, x_q8_1_scratch, stream);
}

// Explicit instantiations.
//
// Quantize-x: one per K. Three K values cover all known mul_mats (talker +
// code-pred, both AR and prefill steps).
template void launch_quantize_x_q8_1<1024>(const float *, block_q8_1 *, cudaStream_t);
template void launch_quantize_x_q8_1<2048>(const float *, block_q8_1 *, cudaStream_t);
template void launch_quantize_x_q8_1<3072>(const float *, block_q8_1 *, cudaStream_t);

// MMVQ: K x N grid. Code-pred uses (1024, *) and (*, 1024); talker uses
// (2048, *) and (*, 2048). intermediate=3072 brings in the wider FFN
// projections for both. Heads (codec_head, code_pred_head) are also
// covered when their (K, N) lands in this set — code_pred_head is
// (1024, 2048), already here; talker codec_head depends on the model's
// codec_vocab_size and may fall through if it's not 1024/2048/3072.
#define INSTANTIATE_MMVQ(K_, N_) \
    template void launch_mmvq_q8_0_q8_1<K_, N_>(float *, const block_q8_0 *, const block_q8_1 *, cudaStream_t); \
    template void launch_mmvq_q8_0_f32  <K_, N_>(float *, const block_q8_0 *, const float *, block_q8_1 *, cudaStream_t);

INSTANTIATE_MMVQ(1024, 1024)
INSTANTIATE_MMVQ(1024, 2048)
INSTANTIATE_MMVQ(1024, 3072)
INSTANTIATE_MMVQ(2048, 1024)
INSTANTIATE_MMVQ(2048, 2048)
INSTANTIATE_MMVQ(2048, 3072)
INSTANTIATE_MMVQ(3072, 1024)
INSTANTIATE_MMVQ(3072, 2048)

#undef INSTANTIATE_MMVQ

// ----------------------------------------------------------------------------
// Hook
// ----------------------------------------------------------------------------
//
// Called by ggml-cuda.cu for every mul_mat. We claim ops where:
//   - src0 is Q8_0 (the weight)
//   - src1 is F32 (the activation)
//   - dst  is F32
//   - M = 1 (single-token AR step)
//   - (K, N) matches one of our specialised shapes
// and dispatch to a specialised kernel. For everything else we return false
// and ggml's generic path runs.
//
// The Q8_1 staging buffer is a one-time-per-call allocation. We could share
// across the layer's mul_mats but for v0 we re-quantize per call to mirror
// ggml's behaviour (same launch count). Optimization: hoist the quantize
// up so all of attn_q/k/v share one quantize-x-q8_1 launch — that's a
// meaningful reduction in launches and a candidate for v0.1.

// Q8_1 staging buffer. Pre-allocated at max K=3072 (96 blocks × 36 B = 3.4 KiB)
// on first use; never reallocated. cudaFree-on-grow would trigger UB because
// async kernels queued on the compute stream still read from the old buffer
// when the realloc happens.
struct StagingBuffer {
    block_q8_1 * d_ptr   = nullptr;
    size_t       d_bytes = 0;

    static constexpr size_t MAX_BLOCKS = 3072 / 32;  // covers K up to 3072

    block_q8_1 * get() {
        if (d_ptr) return d_ptr;
        const size_t need = MAX_BLOCKS * sizeof(block_q8_1);
        if (cudaMalloc(&d_ptr, need) != cudaSuccess) {
            d_ptr = nullptr;
            d_bytes = 0;
            return nullptr;
        }
        d_bytes = need;
        return d_ptr;
    }
};

// Process-wide. Mul_mat dispatch is serialised on the cuda backend's compute
// stream so a single global is fine.
static StagingBuffer g_staging;

// Quantize-x cache. ggml's q/k/v projections all consume the same post-
// attn-norm tensor; ffn_gate/up consume the same post-ffn-norm tensor. We
// keep one Q8_1 staging buffer and skip the F32->Q8_1 quantize when the
// next mul_mat's src1 has the same {data ptr, K} as the previously
// quantized input. Cache invalidates at every graph-compute-begin
// (qwen3_graph_begin_hook) since pool allocators may reuse the same
// pointer for a different logical tensor across graph computes.
//
// Disable with QWEN3_TTS_SPECIALIZED_MMVQ_NO_CACHE=1 for A/B testing.
// Cache infrastructure kept for future revisit but DEFAULT OFF — naive
// "src1->data ptr match" within a single graph compute false-hits because
// ggml's graph allocator does live-range buffer reuse: a tensor's data
// pointer can be the same across different logical tensors within the
// same compute, and skipping the quantize there yields stale Q8_1.
// Re-enable via QWEN3_TTS_SPECIALIZED_MMVQ_CACHE=1 to debug further.
static const float * g_cached_x      = nullptr;
static int64_t       g_cached_x_K    = 0;
static bool          g_cache_enabled = false;
static uint64_t      g_cache_hits    = 0;
static uint64_t      g_cache_misses  = 0;

// Hit-counter for boot diagnostics. The first N hits log shape + outcome;
// after that we go silent to keep the log readable. Toggled by
// QWEN3_TTS_SPECIALIZED_MMVQ_VERBOSE=1.
static int s_log_budget = 0;

extern "C" bool qwen3_mul_mat_hook(
    ggml_backend_cuda_context * ctx,
    const ggml_tensor *         src0,
    const ggml_tensor *         src1,
    ggml_tensor *               dst,
    cudaStream_t                stream) {
    if (!src0 || !src1 || !dst) return false;
    if (src0->type != GGML_TYPE_Q8_0)  return false;
    if (src1->type != GGML_TYPE_F32)   return false;
    if (dst->type  != GGML_TYPE_F32)   return false;

    // M=1 only: src1 has shape [K, M, ...]; we want M==1.
    if (src1->ne[1] != 1) return false;
    // No batching for v0.
    if (src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (dst->ne[1]  != 1 || dst->ne[2]  != 1 || dst->ne[3] != 1) return false;

    // K = src0->ne[0] = src1->ne[0]. N = src0->ne[1] = dst->ne[0].
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    if (K != src1->ne[0]) return false;
    if (N != dst->ne[0])  return false;

    // Strides: we need contiguous source layout for the simple kernel.
    // src0 row stride must be K/32 q8_0 blocks = K/32 * 34 bytes.
    // For Phase A v0 we only handle the contiguous case.
    if (src0->nb[0] != sizeof(block_q8_0))  return false;  // 34
    if (src1->nb[0] != sizeof(float))       return false;  // 4
    if (dst->nb[0]  != sizeof(float))       return false;

    // Dispatch table: only the (K, N) pairs we have explicit instantiations
    // for are claimed. Everything else falls through to ggml. Operates on
    // pre-quantized Q8_1 input — caller (the hook below) handles caching
    // the quantize-x kernel across same-input mul_mats.
    auto dispatch = [&](float * y, const block_q8_0 * w,
                        const block_q8_1 * x_q8_1, cudaStream_t stream) -> bool {
#define TRY_SHAPE(K_, N_) \
        if (K == K_ && N == N_) { \
            launch_mmvq_q8_0_q8_1<K_, N_>(y, w, x_q8_1, stream); \
            return true; \
        }

        TRY_SHAPE(1024, 1024)
        TRY_SHAPE(1024, 2048)
        TRY_SHAPE(1024, 3072)
        TRY_SHAPE(2048, 1024)
        TRY_SHAPE(2048, 2048)
        TRY_SHAPE(2048, 3072)
        TRY_SHAPE(3072, 1024)
        TRY_SHAPE(3072, 2048)

#undef TRY_SHAPE
        return false;
    };

    // Probe whether the shape is in our table BEFORE allocating the staging
    // buffer, to avoid pool churn on misses.
    bool probe = false;
    {
#define PROBE_SHAPE(K_, N_) if (K == K_ && N == N_) probe = true;
        PROBE_SHAPE(1024, 1024) PROBE_SHAPE(1024, 2048) PROBE_SHAPE(1024, 3072)
        PROBE_SHAPE(2048, 1024) PROBE_SHAPE(2048, 2048) PROBE_SHAPE(2048, 3072)
        PROBE_SHAPE(3072, 1024) PROBE_SHAPE(3072, 2048)
#undef PROBE_SHAPE
    }

    if (!probe) {
        if (s_log_budget > 0) {
            std::fprintf(stderr, "[qwen3-megakernel] hook miss K=%lld N=%lld (skip)\n",
                         (long long) K, (long long) N);
            --s_log_budget;
        }
        return false;
    }

    if (s_log_budget > 0) {
        std::fprintf(stderr, "[qwen3-megakernel] hook hit  K=%lld N=%lld\n",
                     (long long) K, (long long) N);
        --s_log_budget;
    }

    (void) ctx;  // device already set by ggml-cuda dispatcher

    block_q8_1 * x_q8_1 = g_staging.get();
    if (!x_q8_1) {
        return false;  // alloc failure — graceful degrade to ggml
    }

    const float *      x_f32 = static_cast<const float *>(src1->data);
    const bool         can_reuse = g_cache_enabled
                                && (x_f32 == g_cached_x)
                                && (K      == g_cached_x_K);

    if (!can_reuse) {
        // Quantize x once; subsequent same-input mul_mats will skip this.
        switch (K) {
            case 1024: launch_quantize_x_q8_1<1024>(x_f32, x_q8_1, stream); break;
            case 2048: launch_quantize_x_q8_1<2048>(x_f32, x_q8_1, stream); break;
            case 3072: launch_quantize_x_q8_1<3072>(x_f32, x_q8_1, stream); break;
            default:   return false;  // shouldn't happen — probe gates K
        }
        g_cached_x   = x_f32;
        g_cached_x_K = K;
        ++g_cache_misses;
    } else {
        ++g_cache_hits;
    }

    return dispatch(
        static_cast<float *>(dst->data),
        static_cast<const block_q8_0 *>(src0->data),
        x_q8_1,
        stream);
}

// ----------------------------------------------------------------------------
// Graph-compute-begin hook
// ----------------------------------------------------------------------------
//
// ggml-cuda calls this once at the top of each ggml_backend_cuda_graph_compute.
// We invalidate the quantize-x cache here: across graph computes, the pool
// allocator may give the same data pointer to a different logical tensor,
// and reusing the stale Q8_1 staging buffer would give wrong outputs.

extern "C" void qwen3_graph_begin_hook(ggml_backend_cuda_context * /*ctx*/) {
    g_cached_x   = nullptr;
    g_cached_x_K = 0;
}

// ----------------------------------------------------------------------------
// install() / is_installed()
// ----------------------------------------------------------------------------

static bool s_installed = false;

bool is_installed() { return s_installed; }

bool install() {
    if (s_installed) return false;

    const char * env = std::getenv("QWEN3_TTS_SPECIALIZED_MMVQ");
    if (!env || env[0] == '\0' || std::strcmp(env, "0") == 0) {
        return false;
    }

    const char * cache_env = std::getenv("QWEN3_TTS_SPECIALIZED_MMVQ_CACHE");
    if (cache_env && cache_env[0] != '\0' && std::strcmp(cache_env, "0") != 0) {
        g_cache_enabled = true;  // experimental, see comment above g_cache_enabled
    }

    ggml_cuda_set_mul_mat_hook(qwen3_mul_mat_hook);
    ggml_cuda_set_graph_begin_hook(qwen3_graph_begin_hook);
    s_installed = true;

    const char * verbose = std::getenv("QWEN3_TTS_SPECIALIZED_MMVQ_VERBOSE");
    if (verbose && verbose[0] != '\0' && std::strcmp(verbose, "0") != 0) {
        s_log_budget = 200;  // log first ~200 hook calls
    }

    std::fprintf(stderr,
                 "[qwen3-megakernel] Phase A specialized MMVQ installed "
                 "(shapes 1024/2048/3072 x 1024/2048/3072%s%s)\n",
                 g_cache_enabled ? ", cache=experimental" : "",
                 s_log_budget > 0 ? ", verbose" : "");
    return true;
}

}  // namespace qwen3_megakernel
