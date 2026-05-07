// Microbench: measure grid_group::sync() latency on the local GPU.
// Decision input for the cooperative-launch megakernel plan: if the per-sync
// cost is more than ~1-2 µs at our expected grid size, the cooperative path is
// net-negative vs the existing per-op kernel launches we'd be replacing.
//
// Build: nvcc -arch=sm_86 -O3 grid_sync_bench.cu -o grid_sync_bench
// Run:   ./grid_sync_bench   (no args)
//
// Methodology: launch a cooperative kernel that performs N grid-wide syncs
// in a loop, time the kernel end-to-end with cudaEvents, divide by N. Sweep
// (n_blocks, n_threads) pairs that bracket what a per-layer megakernel would
// use:
//   - 28 blocks × 256 threads — full-occupancy on RTX 3060 (28 SMs)
//   - 56 blocks × 128 threads — 2 blocks per SM
//   - 112 blocks × 64 threads — 4 blocks per SM
//   - 14 blocks × 256 threads — under-subscribed (one block per 2 SMs)
//
// The kernel does only sync work (no compute) so the result is the pure
// barrier overhead. Compute fused into the kernel would amortize this cost.

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) do {                                                       \
    cudaError_t _err = (call);                                                      \
    if (_err != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                           \
                     __FILE__, __LINE__, cudaGetErrorString(_err));                 \
        std::exit(1);                                                               \
    }                                                                               \
} while (0)

__global__ void grid_sync_loop_kernel(int n_syncs, volatile int * sink) {
    cg::grid_group grid = cg::this_grid();

    int local = 0;
    for (int s = 0; s < n_syncs; ++s) {
        // Tiny per-sync work so the compiler can't elide the loop body.
        local += s ^ (int) threadIdx.x;
        grid.sync();
    }

    // Single thread writes a sink so the loop work isn't dead-code-eliminated.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *sink = local;
    }
}

static double bench_one(int n_blocks, int n_threads, int n_syncs, int n_iters,
                        int * d_sink) {
    void * args[] = { (void *) &n_syncs, (void *) &d_sink };

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // Warmup
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void *) grid_sync_loop_kernel,
            dim3(n_blocks, 1, 1), dim3(n_threads, 1, 1),
            args, 0, nullptr));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < n_iters; ++i) {
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void *) grid_sync_loop_kernel,
            dim3(n_blocks, 1, 1), dim3(n_threads, 1, 1),
            args, 0, nullptr));
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, e0, e1));

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    // Total ms = n_iters launches × (launch overhead + n_syncs × sync_overhead).
    // We want sync_overhead. Assume launch overhead is constant; we run with
    // big n_syncs so the launch term is negligible.
    const double ms_per_launch = total_ms / (double) n_iters;
    const double us_per_sync   = (ms_per_launch * 1000.0) / (double) n_syncs;
    return us_per_sync;
}

int main() {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device: %s (cc=%d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::printf("cooperative-launch supported: %d\n", prop.cooperativeLaunch);

    if (!prop.cooperativeLaunch) {
        std::printf("ERROR: device does not support cooperative-launch.\n");
        return 1;
    }

    int * d_sink;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(int)));

    // (blocks, threads) sweep — second column is the "occupancy intent".
    struct Cfg { int blocks; int threads; const char * note; };
    Cfg cfgs[] = {
        { 14,  256, "under-subscribed (1 block per 2 SMs)"                       },
        { 28,  256, "full occupancy (1 block per SM, our megakernel target)"     },
        { 28,  128, "full occupancy, half block size"                            },
        { 56,  128, "2 blocks per SM"                                            },
        { 56,  256, "2 blocks per SM, full block size"                           },
        { 112,  64, "4 blocks per SM, small blocks"                              },
        { 224,  64, "8 blocks per SM (max-saturated)"                            },
    };
    constexpr int N_CFGS = sizeof(cfgs) / sizeof(cfgs[0]);

    constexpr int n_syncs = 1024;
    constexpr int n_iters = 50;

    std::printf("\n%-6s %-6s %-15s %s\n", "blocks", "thr", "us/sync", "note");
    std::printf("%-6s %-6s %-15s %s\n", "------", "-----", "-------", "----");
    for (int i = 0; i < N_CFGS; ++i) {
        const Cfg & c = cfgs[i];
        // 3 measurements for stability.
        double best = 1e9;
        for (int r = 0; r < 3; ++r) {
            const double us = bench_one(c.blocks, c.threads, n_syncs, n_iters, d_sink);
            if (us < best) best = us;
        }
        std::printf("%-6d %-6d %-15.3f %s\n", c.blocks, c.threads, best, c.note);
    }

    CUDA_CHECK(cudaFree(d_sink));

    std::printf("\nDecision rule from kobbler/docker/tts-qwen3-dev/HANDOFF-megakernel-v0.md:\n");
    std::printf("  ≤ 1.0 µs/sync  → green: cooperative-launch monolith viable\n");
    std::printf("  1.0-2.0 µs     → marginal: design doc but flag risk\n");
    std::printf("  > 2.0 µs       → abort: stay with split-megakernel ladder\n");
    return 0;
}
