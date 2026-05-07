// Qwen3-TTS megakernel — Phase A specialized MMVQ + hook-level QKV / gate-up fusion.
// See kobbler/docker/tts-qwen3-dev/HANDOFF-megakernel-v0.md for the full plan.
//
// What lives here:
//   - F32 → Q8_1 row quantizer (templated on K)
//   - Q8_0 × Q8_1 specialized MMVQ for M=1 (templated on K, N)
//   - Fused QKV / gate-up kernels: one launch produces multiple outputs
//     from a shared Q8_1 staging buffer, eliminating ~4 launches per
//     QKV triplet and ~2 per gate-up pair.
//   - Hook function ggml-cuda calls for every mul_mat. Inspects shape +
//     types; when src0 is Q8_0 + M=1 + (K, N) is one of the 8 specialised
//     pairs, dispatches to our kernel. When the dst belongs to a fused
//     QKV/gate-up plan, either launches the fused kernel (leader role) or
//     no-ops (follower — data already written).
//   - Graph-begin hook: receives cgraph, scans for Q→K→V triplets and
//     gate→up pairs (matched by tensor-name pattern + shared src1), builds
//     a per-graph fusion plan, and invalidates the quantize-x cache.

#include "qwen3_megakernel.cuh"
#include "qwen3_megakernel.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <unordered_map>
#include <vector>

namespace qwen3_megakernel {

// ----------------------------------------------------------------------------
// F32 → Q8_1 quantize (templated on K)
// ----------------------------------------------------------------------------
//
// One block per K/32 Q8_1 block. 32 threads cooperatively scan one block:
// find absmax, derive scale, write int8 quantized values + d (fp16 scale)
// + s (fp16 sum-of-quants * scale).

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

    float amax = fabsf(v);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    }

    const float scale  = amax / 127.0f;
    const float iscale = scale > 0.0f ? 1.0f / scale : 0.0f;

    const int q = __float2int_rn(v * iscale);
    const int qclamped = q < -127 ? -127 : (q > 127 ? 127 : q);

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
// at a time via int* therefore traps on every other block.
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
// Single-shape Q8_0 × Q8_1 MMVQ (M=1)
// ----------------------------------------------------------------------------

template <int K, int N>
__global__ void mmvq_q8_0_q8_1_kernel(
    float *            __restrict__ y,        // [N]
    const block_q8_0 * __restrict__ w,        // [N, K/32]
    const block_q8_1 * __restrict__ x_q8_1) { // [K/32]
    constexpr int K_BLOCKS = K / 32;
    static_assert(K % 32 == 0,  "K must be a multiple of 32");
    static_assert(K_BLOCKS >= 32, "K must be >= 1024");
    static_assert(K_BLOCKS % 32 == 0, "K_BLOCKS must be multiple of warp size");

    const int n = blockIdx.x;
    if (n >= N) return;

    const int lane = threadIdx.x;  // 0..31
    constexpr int KB_PER_LANE = K_BLOCKS / 32;

    float acc = 0.0f;

    #pragma unroll
    for (int kbi = 0; kbi < KB_PER_LANE; ++kbi) {
        const int kbx = lane + kbi * 32;
        const block_q8_0 & wb = w[n * K_BLOCKS + kbx];
        const block_q8_1 & xb = x_q8_1[kbx];

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

// ----------------------------------------------------------------------------
// Fused QKV / fused gate-up kernels
// ----------------------------------------------------------------------------
//
// One launch produces all outputs of a Q→K→V triplet (or gate→up pair) from
// a single shared Q8_1 staging buffer. Each block handles one output row;
// the block decides which weight matrix and dst it serves from blockIdx.x.
// Total grid = sum of N over all outputs.

template <int K, int N_Q, int N_K, int N_V>
__global__ void fused_qkv_q8_0_q8_1_kernel(
    float *            __restrict__ y_q,
    float *            __restrict__ y_k,
    float *            __restrict__ y_v,
    const block_q8_0 * __restrict__ w_q,
    const block_q8_0 * __restrict__ w_k,
    const block_q8_0 * __restrict__ w_v,
    const block_q8_1 * __restrict__ x_q8_1) {
    constexpr int K_BLOCKS    = K / 32;
    constexpr int KB_PER_LANE = K_BLOCKS / 32;
    constexpr int N_TOTAL     = N_Q + N_K + N_V;

    const int n = blockIdx.x;
    if (n >= N_TOTAL) return;

    const int lane = threadIdx.x;

    // Route this block to the right (W, y, n_local).
    const block_q8_0 * w;
    float *            y;
    int                n_local;
    if (n < N_Q) {
        w = w_q; y = y_q; n_local = n;
    } else if (n < N_Q + N_K) {
        w = w_k; y = y_k; n_local = n - N_Q;
    } else {
        w = w_v; y = y_v; n_local = n - N_Q - N_K;
    }

    float acc = 0.0f;

    #pragma unroll
    for (int kbi = 0; kbi < KB_PER_LANE; ++kbi) {
        const int kbx = lane + kbi * 32;
        const block_q8_0 & wb = w[n_local * K_BLOCKS + kbx];
        const block_q8_1 & xb = x_q8_1[kbx];

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

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        y[n_local] = acc;
    }
}

template <int K, int N_Q, int N_K, int N_V>
void launch_fused_qkv_q8_0_q8_1(
    float *            y_q,
    float *            y_k,
    float *            y_v,
    const block_q8_0 * w_q,
    const block_q8_0 * w_k,
    const block_q8_0 * w_v,
    const block_q8_1 * x_q8_1,
    cudaStream_t       stream) {
    constexpr int N_TOTAL = N_Q + N_K + N_V;
    fused_qkv_q8_0_q8_1_kernel<K, N_Q, N_K, N_V><<<N_TOTAL, 32, 0, stream>>>(
        y_q, y_k, y_v, w_q, w_k, w_v, x_q8_1);
}

template <int K, int N_GATE, int N_UP>
__global__ void fused_gate_up_q8_0_q8_1_kernel(
    float *            __restrict__ y_gate,
    float *            __restrict__ y_up,
    const block_q8_0 * __restrict__ w_gate,
    const block_q8_0 * __restrict__ w_up,
    const block_q8_1 * __restrict__ x_q8_1) {
    constexpr int K_BLOCKS    = K / 32;
    constexpr int KB_PER_LANE = K_BLOCKS / 32;
    constexpr int N_TOTAL     = N_GATE + N_UP;

    const int n = blockIdx.x;
    if (n >= N_TOTAL) return;

    const int lane = threadIdx.x;

    const block_q8_0 * w;
    float *            y;
    int                n_local;
    if (n < N_GATE) {
        w = w_gate; y = y_gate; n_local = n;
    } else {
        w = w_up;   y = y_up;   n_local = n - N_GATE;
    }

    float acc = 0.0f;

    #pragma unroll
    for (int kbi = 0; kbi < KB_PER_LANE; ++kbi) {
        const int kbx = lane + kbi * 32;
        const block_q8_0 & wb = w[n_local * K_BLOCKS + kbx];
        const block_q8_1 & xb = x_q8_1[kbx];

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

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        y[n_local] = acc;
    }
}

template <int K, int N_GATE, int N_UP>
void launch_fused_gate_up_q8_0_q8_1(
    float *            y_gate,
    float *            y_up,
    const block_q8_0 * w_gate,
    const block_q8_0 * w_up,
    const block_q8_1 * x_q8_1,
    cudaStream_t       stream) {
    constexpr int N_TOTAL = N_GATE + N_UP;
    fused_gate_up_q8_0_q8_1_kernel<K, N_GATE, N_UP><<<N_TOTAL, 32, 0, stream>>>(
        y_gate, y_up, w_gate, w_up, x_q8_1);
}

// ----------------------------------------------------------------------------
// Fused gate-up + SiLU + mul: one block per output column N_INTERMEDIATE,
// computes gate_dot AND up_dot into registers, then silu(gate)*up.
// ----------------------------------------------------------------------------

template <int K, int N_INTERMEDIATE>
__global__ void fused_gate_up_silu_q8_0_q8_1_kernel(
    float *            __restrict__ y_intermediate,
    const block_q8_0 * __restrict__ w_gate,
    const block_q8_0 * __restrict__ w_up,
    const block_q8_1 * __restrict__ x_q8_1) {
    constexpr int K_BLOCKS    = K / 32;
    constexpr int KB_PER_LANE = K_BLOCKS / 32;
    static_assert(K % 32 == 0, "K must be a multiple of 32");
    static_assert(K_BLOCKS >= 32, "K must be >= 1024");
    static_assert(K_BLOCKS % 32 == 0, "K_BLOCKS must be multiple of warp size");

    const int n = blockIdx.x;
    if (n >= N_INTERMEDIATE) return;

    const int lane = threadIdx.x;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    #pragma unroll
    for (int kbi = 0; kbi < KB_PER_LANE; ++kbi) {
        const int kbx = lane + kbi * 32;
        const block_q8_0 & wb_g = w_gate[n * K_BLOCKS + kbx];
        const block_q8_0 & wb_u = w_up  [n * K_BLOCKS + kbx];
        const block_q8_1 & xb   = x_q8_1[kbx];

        int sumi_g = 0;
        int sumi_u = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int v_g = load_q8_0_int(wb_g.qs, j);
            const int v_u = load_q8_0_int(wb_u.qs, j);
            const int u   = load_q8_1_int(xb.qs, j);
            sumi_g = __dp4a(v_g, u, sumi_g);
            sumi_u = __dp4a(v_u, u, sumi_u);
        }

        const float d_x = __half2float(xb.d);
        gate_acc += __half2float(wb_g.d) * d_x * (float) sumi_g;
        up_acc   += __half2float(wb_u.d) * d_x * (float) sumi_u;
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, o);
        up_acc   += __shfl_xor_sync(0xffffffff, up_acc,   o);
    }

    if (lane == 0) {
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        const float silu_g = gate_acc / (1.0f + expf(-gate_acc));
        y_intermediate[n] = silu_g * up_acc;
    }
}

template <int K, int N_INTERMEDIATE>
void launch_fused_gate_up_silu_q8_0_q8_1(
    float *            y_intermediate,
    const block_q8_0 * w_gate,
    const block_q8_0 * w_up,
    const block_q8_1 * x_q8_1,
    cudaStream_t       stream) {
    fused_gate_up_silu_q8_0_q8_1_kernel<K, N_INTERMEDIATE>
        <<<N_INTERMEDIATE, 32, 0, stream>>>(
            y_intermediate, w_gate, w_up, x_q8_1);
}



// Explicit instantiations for the 0.6B model. Talker and code-pred share
// hidden=1024, intermediate=3072, n_heads=16, n_kv_heads=8, head_dim=128, so
// QKV = (K=1024, N_Q=2048, N_K=1024, N_V=1024) and gate/up = (K=1024,
// N_GATE=3072, N_UP=3072). If a future model deviates we'll need new
// instantiations; the hook below falls through to per-op for unknown shapes.
template void launch_quantize_x_q8_1<1024>(const float *, block_q8_1 *, cudaStream_t);
template void launch_quantize_x_q8_1<2048>(const float *, block_q8_1 *, cudaStream_t);
template void launch_quantize_x_q8_1<3072>(const float *, block_q8_1 *, cudaStream_t);

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

template void launch_fused_qkv_q8_0_q8_1<1024, 2048, 1024, 1024>(
    float *, float *, float *,
    const block_q8_0 *, const block_q8_0 *, const block_q8_0 *,
    const block_q8_1 *, cudaStream_t);

template void launch_fused_gate_up_q8_0_q8_1<1024, 3072, 3072>(
    float *, float *,
    const block_q8_0 *, const block_q8_0 *,
    const block_q8_1 *, cudaStream_t);

template void launch_fused_gate_up_silu_q8_0_q8_1<1024, 3072>(
    float *, const block_q8_0 *, const block_q8_0 *,
    const block_q8_1 *, cudaStream_t);

// ----------------------------------------------------------------------------
// Q8_1 staging buffer
// ----------------------------------------------------------------------------
//
// Pre-allocated at max K=3072 on first use; never freed. Async kernels in
// flight on the compute stream still read this when subsequent mul_mats
// arrive, so realloc-on-grow would be UB.

struct StagingBuffer {
    block_q8_1 * d_ptr   = nullptr;
    static constexpr size_t MAX_BLOCKS = 3072 / 32;

    block_q8_1 * get() {
        if (d_ptr) return d_ptr;
        const size_t need = MAX_BLOCKS * sizeof(block_q8_1);
        if (cudaMalloc(&d_ptr, need) != cudaSuccess) {
            d_ptr = nullptr;
            return nullptr;
        }
        return d_ptr;
    }
};

static StagingBuffer g_staging;

// ----------------------------------------------------------------------------
// Quantize-x cache
// ----------------------------------------------------------------------------
//
// When two consecutive mul_mats consume the same logical input tensor, we
// can skip re-quantizing x. Key on the *ggml_tensor* pointer of src1, NOT
// on src1->data: ggml's graph allocator does live-range buffer reuse, so a
// fresh logical tensor can inherit a recently-freed data pointer within
// the same graph compute. ggml_tensor* is unique per logical tensor.
//
// Invalidated at every graph-compute-begin in qwen3_graph_begin_hook to
// be safe across graph rebuilds (where ggml_tensor* may be reused as
// well). Cache also bypassed when the fusion plan handles a triplet —
// fused dispatch hoists the quantize itself.

// Cache: when two consecutive mul_mats share the same logical input tensor
// (Q/K/V all consume post-attn-norm `cur`; gate/up share post-ffn-norm
// `cur`), skip the redundant quantize_x launch. Key on the *ggml_tensor*
// pointer of src1, not on src1->data — ggml's graph allocator does live-
// range buffer reuse, so a fresh logical tensor can inherit a recently-
// freed data ptr; ggml_tensor* is unique per logical tensor within a
// graph compute.
//
// Safe because cache hits keep the per-op kernel-launch order intact
// (only the quantize_x is skipped). The "QKV fusion" idea of launching
// all 3 mmvqs from one hook is NOT safe — ggml's allocator may alias
// K's and V's dsts on the assumption that K's downstream consumer
// (k_rms_norm) runs before V's mul_mat. See feedback in the megakernel
// handoff for full receipts.
static const ggml_tensor * g_cached_src1     = nullptr;
static int64_t             g_cached_src1_K   = 0;
static bool                g_cache_enabled   = true;
static uint64_t            g_cache_hits      = 0;
static uint64_t            g_cache_misses    = 0;

// Boot diagnostics budget.
static int s_log_budget = 0;

// ----------------------------------------------------------------------------
// Fusion plan
// ----------------------------------------------------------------------------
//
// Built per graph in qwen3_graph_begin_hook by walking cgraph->nodes. Each
// MUL_MAT node whose src0->name contains "attn_q.weight" / "attn_k.weight"
// / "attn_v.weight" (or "ffn_gate.weight" / "ffn_up.weight") is a fusion
// candidate. We group by (src1 ggml_tensor *) — Q/K/V all consume the
// post-attn-norm `cur` so their src1 matches; gate/up consume post-ffn-norm
// `cur`. The lowest-cgraph-index node in a complete group becomes the
// "leader" — the first to be dispatched, where the fused launch happens.
// The other nodes' dsts are marked as "satisfied" and short-circuit on
// hook entry.

enum FusedRole {
    ROLE_LEADER_QKV,   // launches fused_qkv kernel
    ROLE_LEADER_GATE_UP,
    ROLE_FOLLOWER,     // hook returns true with no kernel launch
};

struct FusedEntry {
    FusedRole role;
    // Shapes captured at plan time.
    int64_t K = 0;
    int64_t N_Q = 0, N_K = 0, N_V = 0;       // QKV leader fields (N_V unused for gate-up)
    int64_t N_GATE = 0, N_UP = 0;            // gate-up leader fields
    // Node pointers — ggml fills in ->data lazily during dispatch, so we
    // capture nodes at plan time and dereference ->data at hook-fire time.
    const ggml_tensor * src1   = nullptr;    // shared input
    const ggml_tensor * q_node = nullptr;    // Q (or gate) mul_mat node
    const ggml_tensor * k_node = nullptr;    // K (or up)   mul_mat node
    const ggml_tensor * v_node = nullptr;    // V (qkv only)
    // Gate-up + silu + mul fusion (optional — populated if a downstream
    // silu(gate)*up chain is detected). The leader writes
    // silu(gate)*up directly into post_silu_mul_node->data.
    const ggml_tensor * silu_node          = nullptr;
    const ggml_tensor * post_silu_mul_node = nullptr;
};

// Map dst tensor → fusion entry. Lifetime: one graph compute (cleared at
// graph-begin). Lookup by raw ggml_tensor pointer.
static std::unordered_map<const ggml_tensor *, FusedEntry> g_fusion_plan;
static bool      g_fusion_enabled = true;


// QKV fusion is ON. The graph-build path in tts_transformer.cpp explicitly
// expands Q/K/V mul_mats adjacently before any consumer is built, so
// ggml's graph allocator gives them distinct dst slots. Without that
// (older builds) the allocator aliased K's and V's dsts because K's
// rms_norm consumed K before V's mul_mat in topo order — breaking any
// fusion that reorders the K/V launches.
static bool      g_fusion_qkv_enabled     = true;
static bool      g_fusion_gate_up_enabled = true;
// Fold silu(gate) * up into the gate-up leader: each block writes
// silu(gate@x) * up@x for one output column directly to the post-mul
// intermediate tensor's slot. Saves silu and mul launches per layer.
static bool      g_fusion_silu_up_enabled = true;
// When false (default): a single multi-route kernel handles all outputs
// from one launch (1 quantize + 1 fused kernel = 2 launches per group),
// vs 4 launches in split mode. ~+0.04 RTF win measured on breakit.
// When true: the fused leader launches one mmvq per output, all reading
// the same hoisted Q8_1 staging buffer — bit-equivalent to the standalone
// path. Useful as a numerical-correctness escape hatch.
static bool      g_fusion_split_only = false;
static uint64_t  g_fusion_groups_built  = 0;
static uint64_t  g_fusion_leader_fires  = 0;
static uint64_t  g_fusion_follower_fires = 0;

// ----------------------------------------------------------------------------
// Pre-scan helpers
// ----------------------------------------------------------------------------

namespace {

// Returns one of {0=q, 1=k, 2=v, 3=gate, 4=up, -1=other} based on src0->name.
// Pattern match on the suffix; talker / code-pred share the same suffixes.
int classify_weight(const ggml_tensor * src0) {
    if (!src0 || !src0->name[0]) return -1;
    const char * n = src0->name;
    if (std::strstr(n, "attn_q.weight"))   return 0;
    if (std::strstr(n, "attn_k.weight"))   return 1;
    if (std::strstr(n, "attn_v.weight"))   return 2;
    if (std::strstr(n, "ffn_gate.weight")) return 3;
    if (std::strstr(n, "ffn_up.weight"))   return 4;
    return -1;
}

// Extract "talker.blk.<L>." or "code_pred.blk.<L>." prefix length so we can
// confirm two ops belong to the same layer block when grouping.
int layer_block_prefix_len(const ggml_tensor * src0) {
    if (!src0 || !src0->name[0]) return 0;
    const char * n = src0->name;
    const char * dot = std::strchr(n, '.');
    if (!dot) return 0;
    // Skip "talker." or "code_pred." prefix.
    const char * blk = std::strstr(n, "blk.");
    if (!blk) return 0;
    const char * after_idx = std::strchr(blk + 4, '.');
    if (!after_idx) return 0;
    return (int) (after_idx + 1 - n);  // length up to and including the '.'
}

bool is_q8_0_mul_mat_at_m1(const ggml_tensor * node) {
    if (!node) return false;
    if (node->op != GGML_OP_MUL_MAT) return false;
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (!src0 || !src1) return false;
    if (src0->type != GGML_TYPE_Q8_0) return false;
    if (src1->type != GGML_TYPE_F32)  return false;
    if (node->type != GGML_TYPE_F32)  return false;
    if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (node->ne[1] != 1 || node->ne[2] != 1 || node->ne[3] != 1) return false;
    return true;
}


}  // namespace

// ----------------------------------------------------------------------------
// Hook
// ----------------------------------------------------------------------------

extern "C" bool qwen3_mul_mat_hook(
    ggml_backend_cuda_context * ctx,
    const ggml_tensor *         src0,
    const ggml_tensor *         src1,
    ggml_tensor *               dst,
    cudaStream_t                stream) {
    if (!src0 || !src1 || !dst) return false;

    // Fast path: fusion plan lookup. If this dst is part of a fused group,
    // we either launch the fused kernel (leader) or no-op (follower).
    if (g_fusion_enabled) {
        auto it = g_fusion_plan.find(dst);
        if (it != g_fusion_plan.end()) {
            const FusedEntry & e = it->second;
            if (e.role == ROLE_FOLLOWER) {
                ++g_fusion_follower_fires;
                return true;
            }

            // Leader: hoist quantize-x once; launch fused kernel. Skip
            // quantize-x if the staging buffer is already populated for
            // this src1 — the per-op norm-quantize fusion ran its fused
            // launch on the same src1 just upstream and cached it.
            block_q8_1 * x_q8_1 = g_staging.get();
            if (!x_q8_1) return false;

            const bool already_cached = g_cache_enabled
                                     && (e.src1 == g_cached_src1)
                                     && ((int64_t) e.K == g_cached_src1_K);
            if (!already_cached) {
                const float * x_f32 = static_cast<const float *>(e.src1->data);
                switch (e.K) {
                    case 1024: launch_quantize_x_q8_1<1024>(x_f32, x_q8_1, stream); break;
                    case 2048: launch_quantize_x_q8_1<2048>(x_f32, x_q8_1, stream); break;
                    case 3072: launch_quantize_x_q8_1<3072>(x_f32, x_q8_1, stream); break;
                    default:   return false;
                }
                g_cached_src1   = e.src1;
                g_cached_src1_K = e.K;
                ++g_cache_misses;
            } else {
                ++g_cache_hits;
            }

            // Resolve data pointers lazily — ggml's allocator fills ->data
            // during dispatch, not at graph_begin time.
            void *       y_q = e.q_node ? e.q_node->data : nullptr;
            void *       y_k = e.k_node ? e.k_node->data : nullptr;
            void *       y_v = e.v_node ? e.v_node->data : nullptr;
            const void * w_q = (e.q_node && e.q_node->src[0]) ? e.q_node->src[0]->data : nullptr;
            const void * w_k = (e.k_node && e.k_node->src[0]) ? e.k_node->src[0]->data : nullptr;
            const void * w_v = (e.v_node && e.v_node->src[0]) ? e.v_node->src[0]->data : nullptr;
            const void * x_data = e.src1 ? e.src1->data : nullptr;

            ++g_fusion_leader_fires;

            if (s_log_budget > 0) {
                if (e.role == ROLE_LEADER_QKV) {
                    std::fprintf(stderr,
                        "[qwen3-megakernel] qkv-leader fire #%llu: y_q=%p y_k=%p y_v=%p (alias=%s)\n",
                        (unsigned long long) g_fusion_leader_fires,
                        y_q, y_k, y_v,
                        (y_q == y_k || y_q == y_v || y_k == y_v) ? "YES-BAD" : "no");
                } else {
                    std::fprintf(stderr,
                        "[qwen3-megakernel] gate-up-leader fire #%llu: y_g=%p y_u=%p (alias=%s)\n",
                        (unsigned long long) g_fusion_leader_fires,
                        y_q, y_k,
                        (y_q == y_k) ? "YES-BAD" : "no");
                }
                --s_log_budget;
            }

            // Bail to standalone if any required ptr isn't ready (shouldn't
            // happen if the leader is the lowest-cgraph-index of the group,
            // but defensive — and if a follower's data isn't ready when its
            // own kernel would have run, we'd wrongly skip work).
            if (!y_q || !x_data || !w_q) return false;
            if (e.role == ROLE_LEADER_QKV && (!y_k || !y_v || !w_k || !w_v)) {
                return false;
            }
            if (e.role == ROLE_LEADER_GATE_UP && (!y_k || !w_k)) {
                return false;
            }

            if (e.role == ROLE_LEADER_QKV) {
                if (e.K == 1024 && e.N_Q == 2048 && e.N_K == 1024 && e.N_V == 1024) {
                    if (g_fusion_split_only) {
                        launch_mmvq_q8_0_q8_1<1024, 2048>(
                            static_cast<float *>(y_q),
                            static_cast<const block_q8_0 *>(w_q),
                            x_q8_1, stream);
                        launch_mmvq_q8_0_q8_1<1024, 1024>(
                            static_cast<float *>(y_k),
                            static_cast<const block_q8_0 *>(w_k),
                            x_q8_1, stream);
                        launch_mmvq_q8_0_q8_1<1024, 1024>(
                            static_cast<float *>(y_v),
                            static_cast<const block_q8_0 *>(w_v),
                            x_q8_1, stream);
                    } else {
                        launch_fused_qkv_q8_0_q8_1<1024, 2048, 1024, 1024>(
                            static_cast<float *>(y_q),
                            static_cast<float *>(y_k),
                            static_cast<float *>(y_v),
                            static_cast<const block_q8_0 *>(w_q),
                            static_cast<const block_q8_0 *>(w_k),
                            static_cast<const block_q8_0 *>(w_v),
                            x_q8_1, stream);
                    }
                    return true;
                }
                return false;
            } else if (e.role == ROLE_LEADER_GATE_UP) {
                if (e.K == 1024 && e.N_GATE == 3072 && e.N_UP == 3072) {
                    // Fully fused gate+up+silu+mul: write the SwiGLU
                    // intermediate (silu(gate)*up) directly into the
                    // mul-node's allocated slot. silu and mul nodes
                    // are op-hook followers and will no-op.
                    if (!g_fusion_split_only && e.post_silu_mul_node) {
                        void * y_int = e.post_silu_mul_node->data;
                        if (!y_int) return false;
                        launch_fused_gate_up_silu_q8_0_q8_1<1024, 3072>(
                            static_cast<float *>(y_int),
                            static_cast<const block_q8_0 *>(w_q),
                            static_cast<const block_q8_0 *>(w_k),
                            x_q8_1, stream);
                        return true;
                    }
                    if (g_fusion_split_only) {
                        launch_mmvq_q8_0_q8_1<1024, 3072>(
                            static_cast<float *>(y_q),
                            static_cast<const block_q8_0 *>(w_q),
                            x_q8_1, stream);
                        launch_mmvq_q8_0_q8_1<1024, 3072>(
                            static_cast<float *>(y_k),
                            static_cast<const block_q8_0 *>(w_k),
                            x_q8_1, stream);
                    } else {
                        launch_fused_gate_up_q8_0_q8_1<1024, 3072, 3072>(
                            static_cast<float *>(y_q),
                            static_cast<float *>(y_k),
                            static_cast<const block_q8_0 *>(w_q),
                            static_cast<const block_q8_0 *>(w_k),
                            x_q8_1, stream);
                    }
                    return true;
                }
                return false;
            }
            return false;
        }
    }

    // Non-fused path: claim only Q8_0 × F32 → F32 at M=1 with one of our
    // specialized (K, N) shapes.
    if (src0->type != GGML_TYPE_Q8_0)  return false;
    if (src1->type != GGML_TYPE_F32)   return false;
    if (dst->type  != GGML_TYPE_F32)   return false;
    if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (dst->ne[1]  != 1 || dst->ne[2]  != 1 || dst->ne[3] != 1) return false;

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    if (K != src1->ne[0]) return false;
    if (N != dst->ne[0])  return false;
    if (src0->nb[0] != sizeof(block_q8_0))  return false;
    if (src1->nb[0] != sizeof(float))       return false;
    if (dst->nb[0]  != sizeof(float))       return false;

    auto dispatch = [&](float * y, const block_q8_0 * w,
                        const block_q8_1 * x_q8_1) -> bool {
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

    bool probe = false;
    {
#define PROBE_SHAPE(K_, N_) if (K == K_ && N == N_) probe = true;
        PROBE_SHAPE(1024, 1024) PROBE_SHAPE(1024, 2048) PROBE_SHAPE(1024, 3072)
        PROBE_SHAPE(2048, 1024) PROBE_SHAPE(2048, 2048) PROBE_SHAPE(2048, 3072)
        PROBE_SHAPE(3072, 1024) PROBE_SHAPE(3072, 2048)
#undef PROBE_SHAPE
    }
    if (!probe) return false;

    (void) ctx;

    block_q8_1 * x_q8_1 = g_staging.get();
    if (!x_q8_1) return false;

    const bool can_reuse = g_cache_enabled
                        && (src1 == g_cached_src1)
                        && (K    == g_cached_src1_K);

    if (!can_reuse) {
        const float * x_f32 = static_cast<const float *>(src1->data);
        switch (K) {
            case 1024: launch_quantize_x_q8_1<1024>(x_f32, x_q8_1, stream); break;
            case 2048: launch_quantize_x_q8_1<2048>(x_f32, x_q8_1, stream); break;
            case 3072: launch_quantize_x_q8_1<3072>(x_f32, x_q8_1, stream); break;
            default:   return false;
        }
        g_cached_src1   = src1;
        g_cached_src1_K = K;
        ++g_cache_misses;
    } else {
        ++g_cache_hits;
    }

    return dispatch(
        static_cast<float *>(dst->data),
        static_cast<const block_q8_0 *>(src0->data),
        x_q8_1);
}

// ----------------------------------------------------------------------------
// Graph-compute-begin hook
// ----------------------------------------------------------------------------

namespace {

// Build the fusion plan for one graph. Walks cgraph->nodes once; for each
// MUL_MAT node that's a Q/K/V or gate/up at our shape, group by (same
// layer-block prefix, same src1). Complete groups (3-of-3 for QKV, 2-of-2
// for gate-up) become entries; partial groups are dropped (leader pattern
// only fires when we can fully replace the per-op work).

void build_fusion_plan(const ggml_cgraph * cgraph) {
    g_fusion_plan.clear();
    if (!g_fusion_enabled || !cgraph) return;

    struct Slot {
        ggml_tensor * nodes[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};  // q,k,v,gate,up
        int           idx[5]   = {-1, -1, -1, -1, -1};
    };
    // Key = (layer-prefix string, src1 ptr). We intern the prefix into a
    // small per-call vector so we don't allocate strings repeatedly.
    struct Key {
        const char *        layer_name;  // points into a node name; lifetime = this call
        int                 layer_name_len;
        const ggml_tensor * src1;
        bool operator==(const Key & o) const {
            return src1 == o.src1
                && layer_name_len == o.layer_name_len
                && std::strncmp(layer_name, o.layer_name, layer_name_len) == 0;
        }
    };
    struct KeyHash {
        size_t operator()(const Key & k) const {
            // FNV-1a over the layer prefix mixed with the src1 ptr.
            uint64_t h = 1469598103934665603ull;
            for (int i = 0; i < k.layer_name_len; ++i) {
                h ^= (unsigned char) k.layer_name[i];
                h *= 1099511628211ull;
            }
            h ^= reinterpret_cast<uintptr_t>(k.src1);
            h *= 1099511628211ull;
            return (size_t) h;
        }
    };
    std::unordered_map<Key, Slot, KeyHash> groups;

    // ggml_cgraph is opaque from this TU; go through the public accessors.
    ggml_cgraph * cg_mut = const_cast<ggml_cgraph *>(cgraph);
    const int n_nodes = ggml_graph_n_nodes(cg_mut);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(cg_mut, i);
        if (!is_q8_0_mul_mat_at_m1(node)) continue;
        const int role = classify_weight(node->src[0]);
        if (role < 0) continue;

        const int prefix_len = layer_block_prefix_len(node->src[0]);
        if (prefix_len == 0) continue;

        Key k{node->src[0]->name, prefix_len, node->src[1]};
        Slot & s = groups[k];
        s.nodes[role] = node;
        s.idx[role]   = i;
    }

    for (auto & kv : groups) {
        Slot & s = kv.second;

        // Try QKV.
        if (g_fusion_qkv_enabled && s.nodes[0] && s.nodes[1] && s.nodes[2]) {
            ggml_tensor * q = s.nodes[0];
            ggml_tensor * k_n = s.nodes[1];
            ggml_tensor * v = s.nodes[2];

            // Validate shape match against our instantiation. If a future
            // model deviates, we silently skip and the per-op path runs.
            const int64_t K   = q->src[0]->ne[0];
            const int64_t N_Q = q->src[0]->ne[1];
            const int64_t N_K = k_n->src[0]->ne[1];
            const int64_t N_V = v->src[0]->ne[1];
            const bool shape_ok = (K == 1024 && N_Q == 2048 && N_K == 1024 && N_V == 1024);
            if (shape_ok && k_n->src[0]->ne[0] == K && v->src[0]->ne[0] == K) {
                int leader_role = 0;
                int leader_idx  = s.idx[0];
                if (s.idx[1] >= 0 && s.idx[1] < leader_idx) { leader_role = 1; leader_idx = s.idx[1]; }
                if (s.idx[2] >= 0 && s.idx[2] < leader_idx) { leader_role = 2; leader_idx = s.idx[2]; }
                ggml_tensor * leader = s.nodes[leader_role];

                FusedEntry le{};
                le.role   = ROLE_LEADER_QKV;
                le.K      = K;
                le.N_Q    = N_Q;
                le.N_K    = N_K;
                le.N_V    = N_V;
                le.src1   = q->src[1];
                le.q_node = q;
                le.k_node = k_n;
                le.v_node = v;
                g_fusion_plan[leader] = le;

                FusedEntry fe{};
                fe.role = ROLE_FOLLOWER;
                if (q     != leader) g_fusion_plan[q]   = fe;
                if (k_n   != leader) g_fusion_plan[k_n] = fe;
                if (v     != leader) g_fusion_plan[v]   = fe;
                ++g_fusion_groups_built;
            }
        }

        // Try gate/up.
        if (g_fusion_gate_up_enabled && s.nodes[3] && s.nodes[4]) {
            ggml_tensor * g_n = s.nodes[3];
            ggml_tensor * u_n = s.nodes[4];

            const int64_t K     = g_n->src[0]->ne[0];
            const int64_t N_G   = g_n->src[0]->ne[1];
            const int64_t N_U   = u_n->src[0]->ne[1];
            const bool shape_ok = (K == 1024 && N_G == 3072 && N_U == 3072);
            if (shape_ok && u_n->src[0]->ne[0] == K) {
                int leader_role = 3;
                int leader_idx  = s.idx[3];
                if (s.idx[4] >= 0 && s.idx[4] < leader_idx) { leader_role = 4; leader_idx = s.idx[4]; }
                ggml_tensor * leader = s.nodes[leader_role];

                FusedEntry le{};
                le.role   = ROLE_LEADER_GATE_UP;
                le.K      = K;
                le.N_GATE = N_G;
                le.N_UP   = N_U;
                le.src1   = g_n->src[1];
                le.q_node = g_n;   // gate node in q slot
                le.k_node = u_n;   // up   node in k slot

                // Look for the silu(gate) and mul(silu, up) downstream.
                // Pattern: silu->src[0] == g_n; mul->src[*] in {silu, u_n}.
                // ggml_mul is symmetric in its two operands, so accept
                // either order.
                if (g_fusion_silu_up_enabled) {
                    ggml_tensor * silu_node = nullptr;
                    ggml_tensor * mul_node  = nullptr;
                    for (int j = std::max(s.idx[3], s.idx[4]) + 1; j < n_nodes; ++j) {
                        ggml_tensor * cand = ggml_graph_node(cg_mut, j);
                        if (!cand) continue;
                        if (cand->op == GGML_OP_UNARY
                            && ggml_get_unary_op(cand) == GGML_UNARY_OP_SILU
                            && cand->src[0] == g_n) {
                            silu_node = cand;
                            continue;
                        }
                        if (silu_node && cand->op == GGML_OP_MUL) {
                            const ggml_tensor * a = cand->src[0];
                            const ggml_tensor * b = cand->src[1];
                            const bool match = (a == silu_node && b == u_n)
                                            || (a == u_n      && b == silu_node);
                            if (match) {
                                mul_node = cand;
                                break;
                            }
                        }
                    }
                    if (silu_node && mul_node) {
                        le.silu_node          = silu_node;
                        le.post_silu_mul_node = mul_node;
                    }
                }

                g_fusion_plan[leader] = le;

                FusedEntry fe{};
                fe.role = ROLE_FOLLOWER;
                if (g_n != leader) g_fusion_plan[g_n] = fe;
                if (u_n != leader) g_fusion_plan[u_n] = fe;
                if (le.silu_node)          g_fusion_plan[(ggml_tensor *) le.silu_node]          = fe;
                if (le.post_silu_mul_node) g_fusion_plan[(ggml_tensor *) le.post_silu_mul_node] = fe;
                ++g_fusion_groups_built;
            }
        }
    }
}

}  // namespace

extern "C" void qwen3_graph_begin_hook(
    ggml_backend_cuda_context * /*ctx*/,
    const ggml_cgraph *         cgraph) {
    g_cached_src1   = nullptr;
    g_cached_src1_K = 0;

    build_fusion_plan(cgraph);

    // Periodic counter dump (gated by QWEN3_TTS_SPECIALIZED_MMVQ_STATS=N —
    // dumps every N graph_begin calls; default 0 = off). Useful to verify the
    // QKV / gate-up multi-route leader/follower ratios in production, e.g.
    // after a new model variant whose graph topology might differ.
    static uint64_t s_gb_calls = 0;
    static uint64_t s_stats_period = []() -> uint64_t {
        const char * env = std::getenv("QWEN3_TTS_SPECIALIZED_MMVQ_STATS");
        if (!env || env[0] == '\0') return 0;
        long long v = std::atoll(env);
        return v > 0 ? (uint64_t) v : 0;
    }();
    ++s_gb_calls;
    if (s_stats_period > 0 && (s_gb_calls % s_stats_period) == 0) {
        std::fprintf(stderr,
            "[qwen3-megakernel:stats] gb=%llu  qkv/gateup{groups=%llu lead=%llu fol=%llu}\n",
            (unsigned long long) s_gb_calls,
            (unsigned long long) g_fusion_groups_built,
            (unsigned long long) g_fusion_leader_fires,
            (unsigned long long) g_fusion_follower_fires);
    }
}

// ----------------------------------------------------------------------------
// Generic per-op hook: dispatches sub-op-chain fusions
// ----------------------------------------------------------------------------

extern "C" bool qwen3_op_hook(
    ggml_backend_cuda_context * /*ctx*/,
    ggml_tensor *               dst,
    cudaStream_t                stream) {
    if (!dst) return false;



    // gate-up+silu+mul fusion — silu and mul nodes are followers in the
    // FUSION_PLAN map (added at graph_begin). The leader (gate-mm) writes
    // the post-silu-mul intermediate directly into mul_node->data, so
    // running the silu/mul ops a second time would corrupt it.
    if (g_fusion_enabled) {
        auto it = g_fusion_plan.find(dst);
        if (it != g_fusion_plan.end() && it->second.role == ROLE_FOLLOWER) {
            ++g_fusion_follower_fires;
            return true;
        }
    }

    return false;
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

    if (const char * f = std::getenv("QWEN3_TTS_SPECIALIZED_MMVQ_NO_FUSION")) {
        if (f[0] != '\0' && std::strcmp(f, "0") != 0) g_fusion_enabled = false;
    }
    // Force the safe split path (one mmvq per output, bit-equivalent to the
    // standalone path). Useful as a numerical-correctness escape hatch.
    if (const char * sp = std::getenv("QWEN3_TTS_SPECIALIZED_MMVQ_FUSION_SPLIT")) {
        if (sp[0] != '\0' && std::strcmp(sp, "0") != 0) g_fusion_split_only = true;
    }
    // Disable QKV fusion (the topo-order patch in tts_transformer.cpp gives
    // Q/K/V distinct dst slots; if you suspect that's not landed for some
    // reason, this is the kill switch.)
    if (const char * nq = std::getenv("QWEN3_TTS_SPECIALIZED_MMVQ_NO_FUSION_QKV")) {
        if (nq[0] != '\0' && std::strcmp(nq, "0") != 0) g_fusion_qkv_enabled = false;
    }
    if (const char * ng = std::getenv("QWEN3_TTS_SPECIALIZED_MMVQ_NO_FUSION_GATE_UP")) {
        if (ng[0] != '\0' && std::strcmp(ng, "0") != 0) g_fusion_gate_up_enabled = false;
    }
    // silu(gate) * up fusion (folded into gate-up leader). Default ON.
    if (const char * ns = std::getenv("QWEN3_TTS_SPECIALIZED_MMVQ_NO_FUSION_SILU_UP")) {
        if (ns[0] != '\0' && std::strcmp(ns, "0") != 0) g_fusion_silu_up_enabled = false;
    }
    // Cache disable (escape hatch).
    if (const char * nc = std::getenv("QWEN3_TTS_SPECIALIZED_MMVQ_NO_CACHE")) {
        if (nc[0] != '\0' && std::strcmp(nc, "0") != 0) g_cache_enabled = false;
    }

    ggml_cuda_set_mul_mat_hook(qwen3_mul_mat_hook);
    ggml_cuda_set_graph_begin_hook(qwen3_graph_begin_hook);
    ggml_cuda_set_op_hook(qwen3_op_hook);
    s_installed = true;

    if (const char * v = std::getenv("QWEN3_TTS_SPECIALIZED_MMVQ_VERBOSE")) {
        if (v[0] != '\0' && std::strcmp(v, "0") != 0) s_log_budget = 200;
    }

    char fusion_buf[96] = ", fusion=off";
    if (g_fusion_enabled) {
        std::snprintf(fusion_buf, sizeof(fusion_buf),
                      ", fusion=%s%s%s",
                      g_fusion_qkv_enabled     ? "qkv" : "",
                      (g_fusion_qkv_enabled && g_fusion_gate_up_enabled) ? "+" : "",
                      g_fusion_gate_up_enabled ? "gate-up" : "");
        if (!g_fusion_qkv_enabled && !g_fusion_gate_up_enabled) {
            std::snprintf(fusion_buf, sizeof(fusion_buf), ", fusion=on (no groups)");
        }
        if (!g_fusion_split_only) std::strncat(fusion_buf, " multi-route",
                                               sizeof(fusion_buf) - std::strlen(fusion_buf) - 1);
        if (g_fusion_silu_up_enabled && g_fusion_gate_up_enabled) {
            std::strncat(fusion_buf, "+silu*up",
                         sizeof(fusion_buf) - std::strlen(fusion_buf) - 1);
        }
    }
    std::fprintf(stderr,
                 "[qwen3-megakernel] installed: shapes 1024/2048/3072 x 1024/2048/3072%s%s%s\n",
                 g_cache_enabled  ? ", cache=on (src1-tensor key)" : ", cache=off",
                 fusion_buf,
                 s_log_budget > 0 ? ", verbose"                   : "");
    return true;
}

}  // namespace qwen3_megakernel
