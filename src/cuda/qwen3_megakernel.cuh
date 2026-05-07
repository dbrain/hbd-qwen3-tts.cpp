// Qwen3-TTS megakernel — Phase A specialized MMVQ + Phase B layer megakernel.
// See kobbler/docker/tts-qwen3-dev/HANDOFF-megakernel-v0.md for the full plan.
//
// Phase A: shape-specialized Q8_0 mul_mat_vec_q for the model's known (K, N)
// pairs. Hooks into ggml-cuda's mul_mat dispatcher so any matching mul_mat
// (talker layer or code-pred layer) routes to a compile-time-specialized
// kernel.
//
// Phase B: per-layer talker megakernel — not implemented. See handoff for
// design.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Forward declarations for the ggml types we touch — full definitions live
// in ggml-cuda. We only need the struct names to receive opaque pointers.
struct ggml_tensor;
struct ggml_backend_cuda_context;

extern "C" {

// Hook signature accepted by ggml-cuda.cu's dispatcher.
typedef bool (*ggml_cuda_mul_mat_hook_fn)(
    ggml_backend_cuda_context * ctx,
    const ggml_tensor *         src0,
    const ggml_tensor *         src1,
    ggml_tensor *               dst);

// Implemented in ggml-cuda.cu.
void ggml_cuda_set_mul_mat_hook(ggml_cuda_mul_mat_hook_fn fn);

}  // extern "C"

namespace qwen3_megakernel {

// ----------------------------------------------------------------------------
// Model shape constants (Qwen3-TTS 0.6B, talker)
// ----------------------------------------------------------------------------

constexpr int HIDDEN              = 1024;
constexpr int INTERMEDIATE        = 3072;
constexpr int N_HEADS             = 16;
constexpr int N_KV_HEADS          = 8;
constexpr int HEAD_DIM            = 128;
constexpr int N_ATTN_OUT          = N_HEADS * HEAD_DIM;       // 2048
constexpr int N_KV_OUT            = N_KV_HEADS * HEAD_DIM;    // 1024
constexpr int N_TALKER_LAYERS     = 28;
constexpr int N_CODE_PRED_LAYERS  = 5;
constexpr int CODE_PRED_VOCAB     = 2048;

// Q8_0 block: 32 int8 values + fp16 scale. Matches ggml's block_q8_0 layout.
struct block_q8_0 {
    __half d;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 layout mismatch with ggml");

// Q8_1 block: 32 int8 activations + (fp16 scale, fp16 sum-of-quants*scale).
// Matches ggml's block_q8_1 layout.
struct block_q8_1 {
    __half d;
    __half s;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 layout mismatch with ggml");

// install() / is_installed() are declared in qwen3_megakernel.h (C++-only).
// Defined in qwen3_megakernel.cu.

// ----------------------------------------------------------------------------
// Standalone launchers — exposed for the microbench harness so it can compare
// each specialization against ggml's MMVQ in isolation. Production traffic
// goes through install()/the hook, not these.
// ----------------------------------------------------------------------------

// Quantize a F32 vector x[K] into K/32 Q8_1 blocks. K is compile-time.
// One launch per call (32 threads, 1 block per K-block).
template <int K>
void launch_quantize_x_q8_1(
    const float *      x,           // [K]
    block_q8_1 *       x_q8_1,      // [K/32]
    cudaStream_t       stream);

// Specialized MMVQ: y = W * x, M=1, K and N compile-time. Q8_0 weights,
// Q8_1 staged activations, F32 output.
template <int K, int N>
void launch_mmvq_q8_0_q8_1(
    float *            y,           // [N]
    const block_q8_0 * w,           // [N, K/32] row-major
    const block_q8_1 * x_q8_1,      // [K/32]
    cudaStream_t       stream);

// Convenience: F32 input → Q8_1 staging → specialized MMVQ. Same launches as
// ggml's path; used by the hook and the microbench.
template <int K, int N>
void launch_mmvq_q8_0_f32(
    float *            y,           // [N]
    const block_q8_0 * w,           // [N, K/32]
    const float *      x,           // [K]
    block_q8_1 *       x_q8_1_scratch,  // [K/32], caller-provided
    cudaStream_t       stream);

}  // namespace qwen3_megakernel
