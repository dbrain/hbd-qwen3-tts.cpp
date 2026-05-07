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
struct ggml_cgraph;
struct ggml_backend_cuda_context;

extern "C" {

// Hook signature accepted by ggml-cuda.cu's dispatcher. The stream
// parameter is the ggml-cuda context's compute stream — must launch on
// it (NOT stream 0) since ggml creates per-ctx streams with
// cudaStreamNonBlocking, which don't synchronize with the legacy default.
typedef bool (*ggml_cuda_mul_mat_hook_fn)(
    ggml_backend_cuda_context * ctx,
    const ggml_tensor *         src0,
    const ggml_tensor *         src1,
    ggml_tensor *               dst,
    cudaStream_t                stream);

typedef void (*ggml_cuda_graph_begin_hook_fn)(
    ggml_backend_cuda_context * ctx,
    const ggml_cgraph *         cgraph);

// Generic per-op hook. Returns true if the hook fully handled the op
// (ggml-cuda dispatch is skipped). Used by sub-op-chain fusion: the
// anchor op runs a fused kernel and returns true; follower ops in the
// chain (rms_norm → mul → quantize_x; rope → set_rows; etc.) return true
// with no kernel launch.
typedef bool (*ggml_cuda_op_hook_fn)(
    ggml_backend_cuda_context * ctx,
    ggml_tensor *               dst,
    cudaStream_t                stream);

// Implemented in ggml-cuda.cu.
void ggml_cuda_set_mul_mat_hook(ggml_cuda_mul_mat_hook_fn fn);
void ggml_cuda_set_graph_begin_hook(ggml_cuda_graph_begin_hook_fn fn);
void ggml_cuda_set_op_hook(ggml_cuda_op_hook_fn fn);

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

// Fused QKV: one launch produces y_q [N_Q], y_k [N_K], y_v [N_V] from a
// shared K-dim Q8_1 staging buffer. Each block handles one output row
// across the concatenated logical [N_Q + N_K + N_V] output. K is shared.
//
// All shapes (K, N_Q, N_K, N_V) compile-time. Saves 2 mmvq launches (and
// 2 quantize_x launches, since the caller hoists x quantization) per
// triplet vs the unfused path.
template <int K, int N_Q, int N_K, int N_V>
void launch_fused_qkv_q8_0_q8_1(
    float *            y_q,
    float *            y_k,
    float *            y_v,
    const block_q8_0 * w_q,
    const block_q8_0 * w_k,
    const block_q8_0 * w_v,
    const block_q8_1 * x_q8_1,
    cudaStream_t       stream);

// Fused gate/up: same idea, two outputs. K is shared; N_GATE == N_UP for
// the qwen3-tts SwiGLU shape but kept distinct for clarity.
template <int K, int N_GATE, int N_UP>
void launch_fused_gate_up_q8_0_q8_1(
    float *            y_gate,
    float *            y_up,
    const block_q8_0 * w_gate,
    const block_q8_0 * w_up,
    const block_q8_1 * x_q8_1,
    cudaStream_t       stream);

// Fused gate-up + SiLU + mul: one launch produces the SwiGLU-activated
// intermediate (silu(gate@x) * up@x) in a single output buffer, replacing
// gate-mm + up-mm + silu + mul (4 launches → 1). Each block computes BOTH
// gate-row and up-row dot products into registers, then writes
// `silu(gate_row) * up_row` for one output column. Total work is the
// same as the split fused-gate-up + silu + mul; we just collapse them.
template <int K, int N_INTERMEDIATE>
void launch_fused_gate_up_silu_q8_0_q8_1(
    float *            y_intermediate,    // [N_INTERMEDIATE]
    const block_q8_0 * w_gate,
    const block_q8_0 * w_up,
    const block_q8_1 * x_q8_1,
    cudaStream_t       stream);

// Fused (rms_norm + mul-norm-weight + quantize-x → Q8_1) — replaces 3
// kernel launches with 1. Single block of K threads; warp-level reduction
// for the sum-of-squares, then per-warp Q8_1 quantization. F32 norm
// weight (the qwen3-tts norm weights load as F32; if a future GGUF stores
// F16 norms we'll need a separate instantiation).
//
// K must be ≤ 1024 (single block design). Used for the per-layer
// pre-attn-norm and pre-ffn-norm chains where the post-norm `cur`
// is consumed only by Q/K/V or gate/up mul_mats — i.e., the F32
// post-norm output is never read by anything else, so we can skip
// writing it and only emit the Q8_1 staging for the downstream mul_mat
// hook.
template <int K>
void launch_fused_rmsnorm_mul_quantize_q8_1(
    const float *      x,           // [K]
    const float *      norm_w,      // [K] — F32 norm weight
    float              eps,
    block_q8_1 *       y_q8_1,      // [K/32] — staging out
    cudaStream_t       stream);

}  // namespace qwen3_megakernel
