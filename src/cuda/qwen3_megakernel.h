// Qwen3-TTS megakernel — install API.
//
// Pure C++ header (no CUDA dependencies), safe to include from any TU. The
// kernels themselves live in qwen3_megakernel.cuh / .cu.

#pragma once

namespace qwen3_megakernel {

// Install the specialized-mul_mat hook in ggml-cuda. Idempotent — call once
// during process init when QWEN3_TTS_SPECIALIZED_MMVQ is set in the env.
// Returns true if the hook was registered, false if it was already installed
// or if the env gate disables it.
bool install();

// Did install() succeed in this process? Useful for boot logging.
bool is_installed();

}  // namespace qwen3_megakernel
