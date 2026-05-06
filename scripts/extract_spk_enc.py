#!/usr/bin/env python3
"""
Extract a speaker-encoder-only GGUF sidecar from a Qwen3-TTS Base GGUF.

The Base GGUF (~2.4 GB at Q8_0) carries the talker, codec, tokenizer AND
speaker encoder. The C++ binary, when given `--speaker-encoder <file>`,
only reads tensors prefixed `spk_enc.` and metadata keys prefixed
`qwen3-tts.speaker_encoder.`. Everything else is dead weight on disk.

Output: a new GGUF containing only the spk_enc.* tensors and
qwen3-tts.speaker_encoder.* metadata, plus the minimum required GGUF
header keys. Expected size ~24 MB.

Note on naming: the spk_enc tensors come out at F16 (weights) + F32 (biases)
even when extracted from a "Q8_0"-tagged Base GGUF — Q8_0 quantization only
applies to large matrices and skips the speaker encoder's small / odd-shaped
convs. The encoder's bias-add op also requires F32 biases. So name the output
"...-F16.gguf" to reflect the actual storage precision.

Usage:
    python3 extract_spk_enc.py \
        --input  /path/to/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
        --output /path/to/Qwen3-TTS-12Hz-Speaker-Encoder-F16.gguf

Then verify by passing `--speaker-encoder /path/to/output.gguf` to
qwen3-tts-server in place of the Base GGUF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import gguf
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter

SPK_TENSOR_PREFIX = "spk_enc."
SPK_META_PREFIX = "qwen3-tts.speaker_encoder."

# Metadata keys we keep verbatim regardless of prefix.
# The C++ encoder only reads two SPK-specific keys (sample_rate,
# embedding_length) but we also surface a couple of provenance keys so the
# resulting file is self-describing in `gguf_dump.py`-style tooling.
PASSTHROUGH_KEYS = {
    "general.architecture",  # for any tooling that requires it
}


def _add_meta(writer: GGUFWriter, key: str, value, vtype: GGUFValueType, sub_type=None) -> None:
    if vtype == GGUFValueType.STRING:
        writer.add_string(key, value)
    elif vtype == GGUFValueType.UINT8:
        writer.add_uint8(key, int(value))
    elif vtype == GGUFValueType.INT8:
        writer.add_int8(key, int(value))
    elif vtype == GGUFValueType.UINT16:
        writer.add_uint16(key, int(value))
    elif vtype == GGUFValueType.INT16:
        writer.add_int16(key, int(value))
    elif vtype == GGUFValueType.UINT32:
        writer.add_uint32(key, int(value))
    elif vtype == GGUFValueType.INT32:
        writer.add_int32(key, int(value))
    elif vtype == GGUFValueType.UINT64:
        writer.add_uint64(key, int(value))
    elif vtype == GGUFValueType.INT64:
        writer.add_int64(key, int(value))
    elif vtype == GGUFValueType.FLOAT32:
        writer.add_float32(key, float(value))
    elif vtype == GGUFValueType.FLOAT64:
        writer.add_float64(key, float(value))
    elif vtype == GGUFValueType.BOOL:
        writer.add_bool(key, bool(value))
    elif vtype == GGUFValueType.ARRAY:
        writer.add_array(key, list(value))
    else:
        raise ValueError(f"unsupported metadata type {vtype} for key {key!r}")


def extract(input_path: Path, output_path: Path) -> None:
    print(f"reading: {input_path}", flush=True)
    reader = GGUFReader(str(input_path), "r")

    arch = "qwen3-tts-speaker-encoder"
    writer = GGUFWriter(str(output_path), arch)

    # Standard provenance keys we synthesize. These let `gguf_dump.py` show
    # something useful even if the Base GGUF didn't carry general.architecture.
    writer.add_name("Qwen3-TTS Speaker Encoder")
    writer.add_description(
        "Speaker-encoder-only sidecar (spk_enc.* tensors) extracted from a "
        "Qwen3-TTS Base GGUF. Pair with --speaker-encoder on qwen3-tts-server."
    )

    kept_meta = []
    skipped_meta = 0
    for key, field in reader.fields.items():
        # GGUF reader exposes metadata as Field objects with .types/.parts/.data
        if key.startswith(SPK_META_PREFIX) or key in PASSTHROUGH_KEYS:
            try:
                value = field.contents()
            except Exception:
                value = None
            # Inspect the canonical leaf type (last entry of .types) — for
            # arrays the first entry is ARRAY, second is element type.
            types = list(field.types)
            if not types:
                skipped_meta += 1
                continue
            if types[0] == GGUFValueType.ARRAY:
                # Skip array passthrough — we don't currently need any.
                skipped_meta += 1
                continue
            leaf = types[-1]
            try:
                _add_meta(writer, key, value, leaf)
                kept_meta.append((key, leaf.name, value))
            except Exception as e:
                print(f"  WARN: skipping metadata {key!r}: {e}", file=sys.stderr)
                skipped_meta += 1
        else:
            skipped_meta += 1

    if not any(k.startswith(SPK_META_PREFIX) for k, _, _ in kept_meta):
        print(
            "WARN: no qwen3-tts.speaker_encoder.* metadata keys found in input. "
            "The C++ encoder will fall back to defaults (sample_rate=24000, "
            "embedding_length=1024). If the source model used non-default "
            "values this will silently mis-decode.",
            file=sys.stderr,
        )

    # Tensors: keep only spk_enc.*
    kept_tensors = []
    skipped_tensors = 0
    total_bytes = 0
    for tensor in reader.tensors:
        if not tensor.name.startswith(SPK_TENSOR_PREFIX):
            skipped_tensors += 1
            continue

        # GGUFReader gives us a numpy view over the raw quantized bytes via
        # tensor.data (shape matches the dequantized tensor; dtype reflects
        # the storage). To preserve quantization losslessly, copy the raw
        # bytes via the .data buffer at the original tensor_type.
        raw = tensor.data
        # Reconstruct shape in GGML's reverse-row-major order.
        shape = tuple(int(d) for d in reversed(tensor.shape))
        writer.add_tensor(
            tensor.name,
            np.asarray(raw),
            raw_shape=shape,
            raw_dtype=tensor.tensor_type,
        )
        kept_tensors.append((tensor.name, tensor.tensor_type, shape, raw.nbytes))
        total_bytes += raw.nbytes

    print(f"  kept   : {len(kept_meta)} metadata keys, {len(kept_tensors)} tensors ({total_bytes/1e6:.1f} MB raw)")
    print(f"  dropped: {skipped_meta} metadata keys, {skipped_tensors} tensors")

    print(f"writing: {output_path}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = output_path.stat().st_size
    in_size = input_path.stat().st_size
    print(f"done. input={in_size/1e6:.1f} MB → output={out_size/1e6:.1f} MB ({100.0*out_size/in_size:.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="Source GGUF (e.g. Qwen3-TTS Base)")
    ap.add_argument("--output", required=True, type=Path, help="Destination sidecar GGUF")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    extract(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
