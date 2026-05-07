// Microbench: INT8 tensor-core MMVQ vs __dp4a MMVQ on M=1 matvec.
//
// Decision input for the next megakernel lever (HANDOFF-megakernel-v0.md
// "Tensor-core MMVQ via INT8 mma.sync"). The handoff projected a ~16×
// compute-throughput advantage for tensor cores on Ampere — that's the
// datacenter A100 number. RTX 3060 (GA106) consumer Ampere is much closer
// to DP4A peak; with matvec's 1/8 utilization (M=1 wastes 7/8 of the
// m16n8k32 mma's M dim or N dim), the actual win could be smaller — or
// even negative. This bench is the load-bearing measurement.
//
// Build (inside the tts-qwen3-dev:builder image):
//   nvcc -arch=sm_86 -O3 -std=c++17 mmvq_int8_bench.cu -o mmvq_int8_bench
// Run:
//   ./mmvq_int8_bench
//
// Contestants:
//  A. dp4a baseline — port of mmvq_q8_0_q8_1_kernel from
//     src/cuda/qwen3_megakernel.cu. One block per output row, one warp,
//     each lane handles K/(32*32) q8_0 blocks.
//  B. mma.sync m16n8k32 INT8 — 16 output rows per block, broadcast X
//     across the n=0..7 mma N-dim (1/8 utilization).
//
// Sweep over the qwen3-tts compute-hot shapes:
//   (K=1024, N=2048) — attn_q
//   (K=1024, N=1024) — attn_k / attn_v
//   (K=2048, N=1024) — attn_output
//   (K=1024, N=3072) — ffn_gate / ffn_up (hottest — 2x of these per layer)
//   (K=3072, N=1024) — ffn_down (hottest single matmul shape)
//
// For each shape: run both contestants, time with cudaEvents, compare
// outputs against a reference (CPU computed from the same INT8 inputs).
//
// Decision rule: if mma is ≥ 1.3× faster on the hot wide-N shapes
// (1024×3072, 3072×1024), proceed to wire it into qwen3_megakernel.
// If < 1.0×, abort the lever.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do {                                                       \
    cudaError_t _err = (call);                                                      \
    if (_err != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                           \
                     __FILE__, __LINE__, cudaGetErrorString(_err));                 \
        std::exit(1);                                                               \
    }                                                                               \
} while (0)

// ----------------------------------------------------------------------------
// Q8_0 / Q8_1 layout (must match ggml's)
// ----------------------------------------------------------------------------

#define QK8_0 32
#define QK8_1 32

struct __align__(2) block_q8_0 {
    __half d;          // delta (scale)
    int8_t qs[QK8_0];  // quants
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 block size");

struct __align__(4) block_q8_1 {
    __half2 ds;            // d, s packed (s = sum-of-x*d)
    int8_t  qs[QK8_1];    // quants
};
static_assert(sizeof(block_q8_1) == 36, "q8_1 block size");

// ----------------------------------------------------------------------------
// Reference: dot(W[n], X) on CPU using the int8 representation directly
// ----------------------------------------------------------------------------

static float ref_dot(const block_q8_0 * w_row, const block_q8_1 * x, int K_BLOCKS) {
    float acc = 0.0f;
    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; ++j) {
            sumi += (int) w_row[kb].qs[j] * (int) x[kb].qs[j];
        }
        const float d_w = __half2float(w_row[kb].d);
        const __half2 ds_x = x[kb].ds;
        const float d_x = __half2float(__low2half(ds_x));
        acc += d_w * d_x * (float) sumi;
    }
    return acc;
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

__device__ __forceinline__ int load_q8_0_int(const int8_t * qs, int i32) {
    const uint16_t * qs16 = reinterpret_cast<const uint16_t *>(qs);
    const uint16_t lo = qs16[2 * i32 + 0];
    const uint16_t hi = qs16[2 * i32 + 1];
    return (int) lo | ((int) hi << 16);
}

__device__ __forceinline__ int load_q8_1_int(const int8_t * qs, int i32) {
    return reinterpret_cast<const int *>(qs)[i32];
}

// ----------------------------------------------------------------------------
// Contestant A: dp4a baseline (port of qwen3_megakernel.cu)
// ----------------------------------------------------------------------------

template <int K, int N>
__global__ void dp4a_mmvq_kernel(
    float *            __restrict__ y,
    const block_q8_0 * __restrict__ w,
    const block_q8_1 * __restrict__ x_q8_1) {
    constexpr int K_BLOCKS = K / 32;
    static_assert(K % 32 == 0,  "K must be a multiple of 32");
    static_assert(K_BLOCKS >= 32, "K must be >= 1024");
    static_assert(K_BLOCKS % 32 == 0, "K_BLOCKS must be multiple of warp size");

    const int n = blockIdx.x;
    if (n >= N) return;

    const int lane = threadIdx.x;
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
        const float d_x = __half2float(__low2half(xb.ds));
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

// ----------------------------------------------------------------------------
// Contestant B: mma.sync m16n8k32 INT8
// ----------------------------------------------------------------------------
//
// Strategy: 16 output rows per block, one warp per block.
//   - A[16][32] = W rows [n_base+0..n_base+15][k_off..k_off+31] (one q8_0 block per row)
//   - B[8][32]  = X[k_off..k_off+31] broadcast across n=0..7 (1/8 util)
//   - D[16][8]  = sum_k A[m][k] * B[n][k]; we use only D[m][0]
//
// A layout per Ampere PTX m16n8k32 INT8 (row-major):
//   thread t holds A.x[0..3] = 4 ints = 16 int8.
//   lane t in [0..31] maps to (row r, col c) where:
//     r = t / 4 + (idx_within_4 < 2 ? 0 : 8)  ... not quite — actually:
//   Per PTX docs for mma.m16n8k32 .row.col with .s8.s8:
//     A is row-major M=16 K=32. Each thread holds 4 .b32 (= 16 int8).
//     The 8 rows [0..7] are loaded by lanes 0..3, 4..7, 8..11, 12..15, ...
//     in groups of 4. Specifically: A.x[0] = A[r0][c0..c3], A.x[1] = A[r0][c16..c19],
//                                   A.x[2] = A[r0+8][c0..c3], A.x[3] = A[r0+8][c16..c19]
//     where r0 = t / 4, c0 = (t % 4) * 4.
//
// B layout:
//   B is column-major N=8 K=32, holds 2 .b32 per thread.
//   B.x[0] = B[c0..c3][r0], B.x[1] = B[c16..c19][r0]
//   where r0 = t % 8, c0 = (t / 8) * 4.
//
// D layout:
//   D is row-major M=16 N=8, 4 .b32 per thread (s32).
//   D.x[0..3]: D.x[0] = D[r0][c0], D.x[1] = D[r0][c0+1],
//              D.x[2] = D[r0+8][c0], D.x[3] = D[r0+8][c0+1]
//   where r0 = t / 4, c0 = (t % 4) * 2.
//
// For broadcast B (B[n][k] = X[k] regardless of n): all 32 lanes' B.x values
// are the same 4-byte pack of X[c0..c3] / X[c16..c19] (where c0 = (t/8)*4).
// Wait — that means lanes with the same (t/8) hold the same B values. Lanes
// 0..7 share, 8..15 share, etc. So we need 4 distinct .b32 packs of X for the
// 4 K-quartets of the lane group.

template <int K, int N>
__global__ void mma_mmvq_kernel(
    float *            __restrict__ y,
    const block_q8_0 * __restrict__ w,
    const block_q8_1 * __restrict__ x_q8_1) {
    constexpr int K_BLOCKS = K / 32;
    static_assert(K % 32 == 0,  "K must be a multiple of 32");
    static_assert(N % 16 == 0,  "N must be a multiple of 16");

    const int n_block = blockIdx.x;
    const int n_base  = n_block * 16;
    if (n_base >= N) return;

    const int lane = threadIdx.x;

    // Per-lane A registers: 4 ints (16 int8 each).
    int A0, A1, A2, A3;  // A.x[0..3]
    int B0, B1;          // B.x[0..1]

    int D0 = 0, D1 = 0, D2 = 0, D3 = 0;

    // Per-q8_0-block scaling. We accumulate the int32 dotproduct over each
    // 32-elt q8_0 block, then scale by d_w * d_x and add to a per-row float
    // accumulator. (We can't fold scales inside int32 without precision loss.)
    // Per-thread, two rows are owned: r_top = t/4 (0..7), r_bot = r_top + 8.
    // D.x[0,1] contribute to two cols of r_top; D.x[2,3] contribute to two cols
    // of r_bot. We sum across the 8 N-cols at the very end (since B is
    // broadcast all 8 cols are equivalent).

    const int r_top = lane / 4;
    const int r_bot = r_top + 8;
    const int c_q   = (lane % 4) * 4;          // 0,4,8,12 → A col within K-tile
    const int b_kq  = (lane / 8) * 4;          // 0,4,8,12 → B (K) col group

    float acc_top = 0.0f;
    float acc_bot = 0.0f;

    // Iterate over K_BLOCKS in steps of 1 (each q8_0 block = 32 elts = one K-tile of mma).
    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        const block_q8_1 * x_kb = x_q8_1 + kb;        // X for K-tile

        // Load A[16][32] for this K-tile and this 16-row block:
        //   row r_top, cols c_q..c_q+3  -> A0
        //   row r_top, cols c_q+16..c_q+19 -> A1
        //   row r_bot, cols c_q..c_q+3  -> A2
        //   row r_bot, cols c_q+16..c_q+19 -> A3
        // Layout: w[(n_base + row) * K_BLOCKS + kb].qs[col]
        const block_q8_0 & wb_top = w[(n_base + r_top) * K_BLOCKS + kb];
        const block_q8_0 & wb_bot = w[(n_base + r_bot) * K_BLOCKS + kb];

        A0 = load_q8_0_int(wb_top.qs, c_q / 4);          // qs[c_q..c_q+3]
        A1 = load_q8_0_int(wb_top.qs, (c_q + 16) / 4);
        A2 = load_q8_0_int(wb_bot.qs, c_q / 4);
        A3 = load_q8_0_int(wb_bot.qs, (c_q + 16) / 4);

        // Load B[8][32] for this K-tile (X broadcast to all 8 n-cols):
        //   B.x[0] = B[c0..c3][r_b] = X[c0..c3]  (b_kq = c0)
        //   B.x[1] = B[c16..c19][r_b] = X[c16..c19]
        // For broadcast, all r_b values get the same B. So we just load X.
        B0 = load_q8_1_int(x_kb->qs, b_kq / 4);
        B1 = load_q8_1_int(x_kb->qs, (b_kq + 16) / 4);

        // Reset D for this K-tile (we accumulate per q8_0 block in int32, then
        // scale and add to float per-row accumulator).
        D0 = 0; D1 = 0; D2 = 0; D3 = 0;

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(D0), "+r"(D1), "+r"(D2), "+r"(D3)
            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));
#endif
#endif

        // Apply per-q8_0-block scales.
        // D.x[0] holds D[r_top][c0] with c0 = (lane % 4) * 2 — col within N=8.
        // Since B is broadcast, D[*][0..7] are all equivalent. We sum D0..D3
        // across the 8 cols at the end via warp shuffle.
        //
        // For now: per-tile, accumulate the int32 sum into per-row float
        // accumulators, scaled by d_w * d_x.
        const float d_w_top = __half2float(wb_top.d);
        const float d_w_bot = __half2float(wb_bot.d);
        const float d_x     = __half2float(__low2half(x_kb->ds));

        // D0 + D1 = sum over both 4-col groups for r_top (this lane's 2 cols).
        // Multiply by 8 (since broadcast → 8 equivalent N-cols) ONLY if we
        // want the full mma "useful" result; but each lane only HOLDS 2 cols
        // (D0, D1), so we sum within those 2 and use the across-lane shuffle
        // at the end to reduce N-cols.
        acc_top += d_w_top * d_x * (float) (D0 + D1);
        acc_bot += d_w_bot * d_x * (float) (D2 + D3);
    }

    // At this point each lane holds:
    //   acc_top = scaled sum of W[n_base + r_top, *] * X[*] for 2/8 of N-cols
    //   acc_bot = same for r_bot
    // Each row's full result = sum of (acc_top across the 4 lanes that own
    // that row's N-col slices). Since B is broadcast, the 8 N-cols are
    // identical, but each lane only computed 2 of those 8 cols. The reduction
    // across lanes for the same row needs to NOT divide by 8 — we should sum
    // the 2 cols to get 2x the truth, divide by 8 at the very end? No.
    //
    // Actually: D[m][n] for fixed m gives the same value for all n. Each lane
    // has D0 = D[r_top][c0], D1 = D[r_top][c0+1] (two of 8 equivalent values).
    // So acc_top (for THIS lane) = scale * (D0 + D1) = scale * 2 * truth.
    // Across lanes 0..3 (with r_top=0), we have 4 lanes each holding 2 cols
    // (8 total, all equal). So sum across lanes = scale * 8 * truth.
    // We want truth, so divide by 8.
    //
    // Simpler reduction: each lane has its own (r_top, r_bot) pair. 4 lanes
    // share the same row index (lanes 0,1,2,3 -> r_top=0; 4,5,6,7 -> r_top=1; etc).
    // Reduce across these 4 lanes (within an 8-lane group), then write.

    // Reduce across the 4 lanes in the same row group.
    // Lane t has r_top = t/4. Lanes 0,1,2,3 share r_top=0; 4..7 share 1; etc.
    // shuffle xor with offset 1, 2 sums all 4 lanes' acc.
    acc_top += __shfl_xor_sync(0xffffffff, acc_top, 1);
    acc_top += __shfl_xor_sync(0xffffffff, acc_top, 2);
    acc_bot += __shfl_xor_sync(0xffffffff, acc_bot, 1);
    acc_bot += __shfl_xor_sync(0xffffffff, acc_bot, 2);

    // Now lanes 0,1,2,3 all hold acc_top = sum_{4 lanes}(scale * 2 * truth)
    //                                     = 8 * scale * truth (8 cols summed)
    // Divide by 8 to get truth. (Note: truth is already pre-multiplied by all
    // the per-q8_0-block scales, so the /8 applies cleanly.)
    acc_top *= 0.125f;
    acc_bot *= 0.125f;

    // Lane (lane % 4 == 0) writes its row.
    if ((lane % 4) == 0) {
        y[n_base + r_top] = acc_top;
        y[n_base + r_bot] = acc_bot;
    }
}

// ----------------------------------------------------------------------------
// Bench harness
// ----------------------------------------------------------------------------

template <int K, int N>
static void run_shape(std::mt19937 & rng, int n_iters) {
    constexpr int K_BLOCKS = K / 32;

    // Allocate host buffers.
    std::vector<block_q8_0> h_w(N * K_BLOCKS);
    std::vector<block_q8_1> h_x(K_BLOCKS);
    std::vector<float>      h_y(N, 0.0f);
    std::vector<float>      h_y_dp4a(N, 0.0f);
    std::vector<float>      h_y_mma (N, 0.0f);

    // Fill weights with random int8.
    std::uniform_int_distribution<int> qdist(-127, 127);
    std::uniform_real_distribution<float> ddist(0.005f, 0.05f);
    for (int n = 0; n < N; ++n) {
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            block_q8_0 & b = h_w[n * K_BLOCKS + kb];
            b.d = __float2half(ddist(rng));
            for (int j = 0; j < QK8_0; ++j) b.qs[j] = (int8_t) qdist(rng);
        }
    }
    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        block_q8_1 & b = h_x[kb];
        const float d  = ddist(rng);
        // Compute the "s" term properly (sum of qs * d).
        int sum = 0;
        for (int j = 0; j < QK8_1; ++j) {
            b.qs[j] = (int8_t) qdist(rng);
            sum    += (int) b.qs[j];
        }
        const float s = d * (float) sum;
        b.ds = __floats2half2_rn(d, s);
    }

    // Reference (CPU).
    for (int n = 0; n < N; ++n) {
        h_y[n] = ref_dot(&h_w[n * K_BLOCKS], h_x.data(), K_BLOCKS);
    }

    // Device buffers.
    block_q8_0 * d_w; block_q8_1 * d_x; float * d_y;
    CUDA_CHECK(cudaMalloc(&d_w, h_w.size() * sizeof(block_q8_0)));
    CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(block_q8_0), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // Bench dp4a.
    for (int i = 0; i < 3; ++i) {  // warmup
        dp4a_mmvq_kernel<K, N><<<N, 32>>>(d_y, d_w, d_x);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < n_iters; ++i) {
        dp4a_mmvq_kernel<K, N><<<N, 32>>>(d_y, d_w, d_x);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms_dp4a = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_dp4a, e0, e1));
    ms_dp4a /= (float) n_iters;
    CUDA_CHECK(cudaMemcpy(h_y_dp4a.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Bench mma.
    static_assert(N % 16 == 0, "N must be %16 for mma");
    constexpr int N_BLOCKS = N / 16;
    for (int i = 0; i < 3; ++i) {  // warmup
        mma_mmvq_kernel<K, N><<<N_BLOCKS, 32>>>(d_y, d_w, d_x);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < n_iters; ++i) {
        mma_mmvq_kernel<K, N><<<N_BLOCKS, 32>>>(d_y, d_w, d_x);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms_mma = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_mma, e0, e1));
    ms_mma /= (float) n_iters;
    CUDA_CHECK(cudaMemcpy(h_y_mma.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Numerical check.
    auto cosine = [&](const std::vector<float> & a, const std::vector<float> & b) {
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += (double) a[i] * b[i];
            na  += (double) a[i] * a[i];
            nb  += (double) b[i] * b[i];
        }
        return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    };
    auto max_abs_diff = [&](const std::vector<float> & a, const std::vector<float> & b) {
        float m = 0;
        for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
        return m;
    };

    const double cos_dp4a = cosine(h_y, h_y_dp4a);
    const double cos_mma  = cosine(h_y, h_y_mma);
    const float  mad_dp4a = max_abs_diff(h_y, h_y_dp4a);
    const float  mad_mma  = max_abs_diff(h_y, h_y_mma);

    // Bandwidth: weights = N*K_BLOCKS*34 bytes, x = K_BLOCKS*36 bytes.
    const double bytes = (double) N * K_BLOCKS * sizeof(block_q8_0)
                       + (double) K_BLOCKS * sizeof(block_q8_1);
    const double gbps_dp4a = bytes / (ms_dp4a * 1e-3) / 1e9;
    const double gbps_mma  = bytes / (ms_mma  * 1e-3) / 1e9;

    // Useful TOPS: 2*N*K MAC ops per iter. (Counting MAC = 2 ops.)
    const double useful_ops = 2.0 * (double) N * (double) K;
    const double tops_dp4a = useful_ops / (ms_dp4a * 1e-3) / 1e12;
    const double tops_mma  = useful_ops / (ms_mma  * 1e-3) / 1e12;

    std::printf(
        "%5d x %-5d   dp4a %7.3f us (%6.1f GB/s, %5.2f TOPS, cos=%.6f mad=%.4g)   "
        "mma %7.3f us (%6.1f GB/s, %5.2f TOPS, cos=%.6f mad=%.4g)   speedup %.2fx\n",
        K, N,
        ms_dp4a * 1000.0, gbps_dp4a, tops_dp4a, cos_dp4a, mad_dp4a,
        ms_mma  * 1000.0, gbps_mma,  tops_mma,  cos_mma,  mad_mma,
        ms_dp4a / ms_mma);

    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
}

int main() {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device: %s (cc=%d.%d, %d SMs, %.0f MHz, mem %.1f GB/s)\n\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.clockRate / 1000.0,
                2.0 * prop.memoryClockRate / 1.0e6 * (prop.memoryBusWidth / 8.0));

    std::printf("%-15s   %-50s   %-50s   %s\n",
                "shape", "dp4a", "mma m16n8k32 (broadcast B, 1/8 util)", "speedup");
    std::printf("%-15s   %-50s   %-50s   %s\n",
                "-------", "----", "------------------------------------",
                "-------");

    std::mt19937 rng(0xC0FFEE);
    constexpr int n_iters = 200;

    // Hot shapes from qwen3-tts (Q8_0 weights, M=1).
    run_shape<1024, 2048>(rng, n_iters);
    run_shape<1024, 1024>(rng, n_iters);
    run_shape<2048, 1024>(rng, n_iters);
    run_shape<1024, 3072>(rng, n_iters);
    run_shape<3072, 1024>(rng, n_iters);

    std::printf("\nDecision rule (HANDOFF-megakernel-v0.md):\n");
    std::printf("  speedup ≥ 1.3x on the wide-N shapes (1024×3072, 3072×1024) → wire it up\n");
    std::printf("  speedup 1.0-1.3x → marginal, decide based on engineering cost\n");
    std::printf("  speedup < 1.0x   → abort the lever\n");
    return 0;
}
