#!/usr/bin/env python3
"""Convert the Qwen3-TTS-Tokenizer F16 GGUF to a Q8-mixed variant.

Pre-quantizes the subset of vocoder mat-mul weights that
`src/audio_tokenizer_decoder.cpp::should_quantize_q8` would convert at load
time. Shipping this pre-quantized GGUF eliminates the ~180 ms F16→F32→Q8_0
CPU step that fires on every cold start, without changing what the runtime
actually uploads to the GPU (same Q8_0 bytes for the same tensors).

Bit-identical to the current runtime path (uses the same `ggml_quants`
math). Tensors that the runtime keeps F16 because `ne[0] % 32 != 0`
(the four vq_first/vq_rest projections) are passed through F16 here to
match. Everything else (conv kernels, biases, norms, codebooks) stays
exactly as in the source GGUF.

Usage:
    python scripts/convert_vocoder_to_q8_mixed.py <src_f16.gguf> <dst_q8mixed.gguf>

Requires: `pip install gguf numpy`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gguf
import numpy as np


# Mirror of audio_tokenizer_decoder.cpp::should_quantize_q8.
def _wants_q8(name: str) -> bool:
    if "tok_dec.pre_tfm.blk." in name:
        for sfx in (
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ):
            if name.endswith(sfx):
                return True
    if name in (
        "tok_dec.pre_tfm.input_proj.weight",
        "tok_dec.pre_tfm.output_proj.weight",
        "tok_dec.vq_first.input_proj.weight",
        "tok_dec.vq_first.output_proj.weight",
        "tok_dec.vq_rest.input_proj.weight",
        "tok_dec.vq_rest.output_proj.weight",
    ):
        return True
    if "tok_dec.upsample." in name and (
        ".pwconv1.weight" in name or ".pwconv2.weight" in name
    ):
        return True
    return False


def convert(src: Path, dst: Path) -> None:
    reader = gguf.GGUFReader(str(src))

    # Pull the architecture string out of the source so the writer copies
    # it verbatim (the constructor wants `arch` to seed general.architecture).
    arch = "qwen3-tts-tokenizer"
    arch_field = reader.fields.get("general.architecture")
    if arch_field is not None:
        try:
            arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
        except Exception:
            pass

    writer = gguf.GGUFWriter(str(dst), arch=arch)

    # Copy metadata. The writer sets general.architecture itself, and the
    # header/footer fields are auto-managed.
    skip = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "general.architecture",
    }
    for fname, field in reader.fields.items():
        if fname in skip:
            continue
        try:
            writer.add_key_value(fname, field.contents(), field.types[0])
        except Exception as e:
            print(f"WARN: skipped field {fname}: {e}", file=sys.stderr)

    n_q8 = 0
    n_passthru = 0
    bytes_in = 0
    bytes_out = 0
    for t in reader.tensors:
        name = t.name
        src_type = t.tensor_type
        data = np.asarray(t.data)
        bytes_in += t.n_bytes
        # data.shape is in numpy/ggml order: data.shape[-1] is the inner dim
        # (= ggml ne[0]), which is what the runtime checks for 32-alignment.
        inner = data.shape[-1] if data.ndim >= 1 else 0
        wants = (
            _wants_q8(name)
            and src_type == gguf.GGMLQuantizationType.F16
            and inner > 0
            and (inner % 32) == 0
        )
        if wants:
            f32 = data.astype(np.float16).astype(np.float32, copy=False)
            q8 = gguf.quants.quantize(f32, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, q8, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            bytes_out += q8.nbytes
            n_q8 += 1
        else:
            # Passthrough: keep on-disk dtype + bytes intact.
            logical = tuple(int(x) for x in t.shape[::-1])
            writer.add_tensor(name, data, raw_shape=logical, raw_dtype=src_type)
            bytes_out += t.n_bytes
            n_passthru += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(
        f"wrote {dst}\n"
        f"  quantized:   {n_q8} tensors  → Q8_0\n"
        f"  passthrough: {n_passthru} tensors  (unchanged dtype)\n"
        f"  file size:   {bytes_in/1048576:.1f} MiB → {bytes_out/1048576:.1f} MiB"
        f"  ({(bytes_out-bytes_in)/1048576:+.1f} MiB)"
    )


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
