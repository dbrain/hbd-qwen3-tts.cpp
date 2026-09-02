#!/usr/bin/env python3
"""
Convert BreezeBlue/Breeze-TTS-2 (HuggingFace safetensors) to our flat GGUF layout.

Breeze is four models in one checkpoint:

  text_encoder      T5Gemma2 encoder, 26 layers, 1152 hidden, GQA 4/1, head_dim 256
  backbone_model    Qwen3ForCausalLM, 28 layers, 2048 hidden, GQA 16/8, QK-norm
  depth_decoder     12 layers, 1024 hidden, 16 codebooks, NO QK-norm
  codec_model       Mimi, 12.5 Hz / 24 kHz, 32 quantizers (16 valid)

Naming follows scripts/convert_tts_to_gguf.py: flat, llama.cpp-ish, one prefix per
subsystem -- text_enc.* / backbone.* / depth.* / codec.*.

Do NOT use the GGUF published by audio.cpp. That file is audio.cpp-native
(general.architecture = "audiocpp", logical names in an `audiocpp.tensor_names`
array, config/tokenizer JSON and even a logo PNG embedded as `_audiocpp.*`
tensors). Our gguf_loader is a plain gguf_init_from_file + name map.

Usage:
    python scripts/convert_breeze_to_gguf.py \
        --input  /home/dbrain/models/Breeze-TTS-2-hf \
        --output /home/dbrain/models/breeze-tts-2-ours-q8_0.gguf \
        --type   q8_0
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

# numpy / safetensors / tqdm / gguf are imported lazily inside main(), so that
# --dry-run (which validates the name mapping against the safetensors index and is
# the highest-risk part of this script) runs on a bare Python with no deps at all.

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ARCH = "breeze-tts"

# ---------------------------------------------------------------------------
# Tensors we deliberately do not emit.
#
# embed_text_tokens: [262158, 2048], ~570 MB at q8_0. The reference validates its
# shape and never reads it on any path -- see PORT-SPEC §1b. Dropping it is the
# single largest size win available.
DROP_EXACT = {
    "embed_text_tokens.weight",
}

# Mimi ships 32 quantizers (1 semantic + 31 acoustic) but only 16 are valid --
# n_valid_quantizers = 16 in src/audio_codec_encoder.h, and our decoder is
# vq_first (1 semantic) + vq_rest (15 acoustic). Acoustic layers 15.. are dead.
N_VALID_ACOUSTIC = 15


# ---------------------------------------------------------------------------
# Minimal safetensors reader.
#
# safetensors' own numpy backend raises `data type 'bfloat16' not understood`,
# and Breeze is stored bf16 throughout. Pulling in torch just to widen 16 bits is
# not worth it: the container format is an 8-byte little-endian header length, a
# JSON header, then a flat data block. bf16 -> f32 is a shift into the high half.
_ST_DTYPE = {
    "F64": ("<f8", None), "F32": ("<f4", None), "F16": ("<f2", None),
    "I64": ("<i8", None), "I32": ("<i4", None), "I16": ("<i2", None),
    "I8": ("|i1", None), "U8": ("|u1", None), "BOOL": ("|b1", None),
    "BF16": ("<u2", "bf16"),
}


class SafeTensorsShard:
    def __init__(self, path):
        import numpy as np
        self._np = np
        self._f = open(path, "rb")
        n = int.from_bytes(self._f.read(8), "little")
        self._hdr = json.loads(self._f.read(n))
        self._base = 8 + n

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def get(self, name):
        np = self._np
        e = self._hdr[name]
        dt, special = _ST_DTYPE[e["dtype"]]
        start, end = e["data_offsets"]
        self._f.seek(self._base + start)
        raw = self._f.read(end - start)
        arr = np.frombuffer(raw, dtype=np.dtype(dt))
        if special == "bf16":
            # bf16 occupies the high 16 bits of the f32 it truncates
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.reshape(e["shape"]) if e["shape"] else arr.reshape(())


def _drop(name: str) -> bool:
    if name in DROP_EXACT:
        return True
    m = re.match(r"codec_model\.quantizer\.acoustic_residual_vector_quantizer\.layers\.(\d+)\.", name)
    if m and int(m.group(1)) >= N_VALID_ACOUSTIC:
        return True
    # `initialized` is a bookkeeping flag from EMA training, not a weight.
    if name.endswith(".codebook.initialized"):
        return True
    return False


# ---------------------------------------------------------------------------
# name mapping.  (regex, replacement) applied in order; first match wins.
RULES: list[tuple[str, str]] = [
    # ---- T5Gemma2 text encoder ------------------------------------------
    (r"^text_encoder\.embed_tokens\.weight$",                    r"text_enc.token_embd.weight"),
    (r"^text_encoder\.embed_tokens\.eoi_embedding$",             r"text_enc.eoi_embd.weight"),
    (r"^text_encoder\.norm\.weight$",                            r"text_enc.output_norm.weight"),
    (r"^text_encoder_proj\.weight$",                             r"text_enc.proj.weight"),
    # Gemma sandwich norms -- four per layer, all distinct. Getting these
    # confused is a silent quality regression, so they keep explicit names.
    (r"^text_encoder\.layers\.(\d+)\.pre_self_attn_layernorm\.weight$",    r"text_enc.blk.\1.attn_norm.weight"),
    (r"^text_encoder\.layers\.(\d+)\.post_self_attn_layernorm\.weight$",   r"text_enc.blk.\1.attn_post_norm.weight"),
    (r"^text_encoder\.layers\.(\d+)\.pre_feedforward_layernorm\.weight$",  r"text_enc.blk.\1.ffn_norm.weight"),
    (r"^text_encoder\.layers\.(\d+)\.post_feedforward_layernorm\.weight$", r"text_enc.blk.\1.ffn_post_norm.weight"),
    (r"^text_encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight$",  r"text_enc.blk.\1.attn_q.weight"),
    (r"^text_encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight$",  r"text_enc.blk.\1.attn_k.weight"),
    (r"^text_encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight$",  r"text_enc.blk.\1.attn_v.weight"),
    (r"^text_encoder\.layers\.(\d+)\.self_attn\.o_proj\.weight$",  r"text_enc.blk.\1.attn_output.weight"),
    (r"^text_encoder\.layers\.(\d+)\.self_attn\.q_norm\.weight$",  r"text_enc.blk.\1.attn_q_norm.weight"),
    (r"^text_encoder\.layers\.(\d+)\.self_attn\.k_norm\.weight$",  r"text_enc.blk.\1.attn_k_norm.weight"),
    (r"^text_encoder\.layers\.(\d+)\.mlp\.gate_proj\.weight$",     r"text_enc.blk.\1.ffn_gate.weight"),
    (r"^text_encoder\.layers\.(\d+)\.mlp\.up_proj\.weight$",       r"text_enc.blk.\1.ffn_up.weight"),
    (r"^text_encoder\.layers\.(\d+)\.mlp\.down_proj\.weight$",     r"text_enc.blk.\1.ffn_down.weight"),

    # ---- Qwen3 backbone --------------------------------------------------
    (r"^backbone_model\.norm\.weight$",                          r"backbone.output_norm.weight"),
    (r"^lm_head\.weight$",                                       r"backbone.lm_head.weight"),
    (r"^backbone_model\.layers\.(\d+)\.input_layernorm\.weight$",          r"backbone.blk.\1.attn_norm.weight"),
    (r"^backbone_model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"backbone.blk.\1.ffn_norm.weight"),
    (r"^backbone_model\.layers\.(\d+)\.self_attn\.q_proj\.weight$",  r"backbone.blk.\1.attn_q.weight"),
    (r"^backbone_model\.layers\.(\d+)\.self_attn\.k_proj\.weight$",  r"backbone.blk.\1.attn_k.weight"),
    (r"^backbone_model\.layers\.(\d+)\.self_attn\.v_proj\.weight$",  r"backbone.blk.\1.attn_v.weight"),
    (r"^backbone_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$",  r"backbone.blk.\1.attn_output.weight"),
    (r"^backbone_model\.layers\.(\d+)\.self_attn\.q_norm\.weight$",  r"backbone.blk.\1.attn_q_norm.weight"),
    (r"^backbone_model\.layers\.(\d+)\.self_attn\.k_norm\.weight$",  r"backbone.blk.\1.attn_k_norm.weight"),
    (r"^backbone_model\.layers\.(\d+)\.mlp\.gate_proj\.weight$",     r"backbone.blk.\1.ffn_gate.weight"),
    (r"^backbone_model\.layers\.(\d+)\.mlp\.up_proj\.weight$",       r"backbone.blk.\1.ffn_up.weight"),
    (r"^backbone_model\.layers\.(\d+)\.mlp\.down_proj\.weight$",     r"backbone.blk.\1.ffn_down.weight"),

    # ---- depth decoder (NOTE: no q_norm/k_norm, unlike the backbone) -----
    (r"^depth_decoder\.codebooks_head\.weight$",                       r"depth.codebooks_head.weight"),
    (r"^depth_decoder\.model\.embed_tokens\.weight$",                  r"depth.token_embd.weight"),
    (r"^depth_decoder\.model\.inputs_embeds_projector\.weight$",       r"depth.in_proj.weight"),
    (r"^depth_decoder\.model\.norm\.weight$",                          r"depth.output_norm.weight"),
    (r"^depth_decoder\.model\.layers\.(\d+)\.input_layernorm\.weight$",          r"depth.blk.\1.attn_norm.weight"),
    (r"^depth_decoder\.model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"depth.blk.\1.ffn_norm.weight"),
    (r"^depth_decoder\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight$",  r"depth.blk.\1.attn_q.weight"),
    (r"^depth_decoder\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight$",  r"depth.blk.\1.attn_k.weight"),
    (r"^depth_decoder\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight$",  r"depth.blk.\1.attn_v.weight"),
    (r"^depth_decoder\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight$",  r"depth.blk.\1.attn_output.weight"),
    (r"^depth_decoder\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight$",     r"depth.blk.\1.ffn_gate.weight"),
    (r"^depth_decoder\.model\.layers\.(\d+)\.mlp\.up_proj\.weight$",       r"depth.blk.\1.ffn_up.weight"),
    (r"^depth_decoder\.model\.layers\.(\d+)\.mlp\.down_proj\.weight$",     r"depth.blk.\1.ffn_down.weight"),

    # ---- Mimi codec ------------------------------------------------------
    (r"^codec_model\.downsample\.conv\.weight$",  r"codec.downsample.conv.weight"),
    (r"^codec_model\.upsample\.conv\.weight$",    r"codec.upsample.conv.weight"),
    (r"^codec_model\.(encoder|decoder)\.layers\.(\d+)\.conv\.(weight|bias)$",
     r"codec.\1.layers.\2.conv.\3"),
    (r"^codec_model\.(encoder|decoder)\.layers\.(\d+)\.block\.(\d+)\.conv\.(weight|bias)$",
     r"codec.\1.layers.\2.block.\3.conv.\4"),
    (r"^codec_model\.(encoder|decoder)_transformer\.layers\.(\d+)\.(.+)$",
     r"codec.\1_tfm.blk.\2.\3"),
    (r"^codec_model\.quantizer\.semantic_residual_vector_quantizer\.(input_proj|output_proj)\.weight$",
     r"codec.vq_first.\1.weight"),
    (r"^codec_model\.quantizer\.semantic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.(embed_sum|cluster_usage)$",
     r"codec.vq_first.\1.\2"),
    (r"^codec_model\.quantizer\.acoustic_residual_vector_quantizer\.(input_proj|output_proj)\.weight$",
     r"codec.vq_rest.\1.weight"),
    (r"^codec_model\.quantizer\.acoustic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.(embed_sum|cluster_usage)$",
     r"codec.vq_rest.\1.\2"),
]


def map_name(hf: str) -> str | None:
    for pat, rep in RULES:
        if re.match(pat, hf):
            return re.sub(pat, rep, hf)
    return None


# Tensors that must stay full precision regardless of --type.
#   *.cluster_usage / *.embed_sum : the decoder computes
#       normalized_codebook = embed_sum / max(cluster_usage, 1e-5)
#   at load; quantizing either side wrecks the division.
#   conv weights: our ggml_conv_1d_direct CUDA kernel asserts F16 weights.
def pick_dtype(name, arr, out_type):
    import numpy as np
    import gguf
    if arr.ndim <= 1:
        return arr.astype(np.float32), gguf.GGMLQuantizationType.F32
    if name.endswith((".cluster_usage", ".embed_sum")):
        return arr.astype(np.float32), gguf.GGMLQuantizationType.F32
    if ".conv." in name or name.endswith(".conv.weight"):
        # conv-1d-direct.cu: GGML_ASSERT(a->type == GGML_TYPE_F16)
        return arr.astype(np.float16), gguf.GGMLQuantizationType.F16
    if out_type == "f32":
        return arr.astype(np.float32), gguf.GGMLQuantizationType.F32
    if out_type == "f16":
        return arr.astype(np.float16), gguf.GGMLQuantizationType.F16
    if out_type == "q8_0":
        try:
            return (gguf.quants.quantize(arr.astype(np.float32),
                                         gguf.GGMLQuantizationType.Q8_0),
                    gguf.GGMLQuantizationType.Q8_0)
        except Exception as e:  # noqa: BLE001
            log.warning("Q8_0 failed for %s (%s); keeping F16", name, e)
            return arr.astype(np.float16), gguf.GGMLQuantizationType.F16
    raise ValueError(f"unknown --type {out_type}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="HF model directory")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--type", default="q8_0", choices=["f32", "f16", "q8_0"])
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the name mapping without reading tensor data")
    args = ap.parse_args()

    if not args.dry_run:
        import numpy as np  # noqa: F401
        GGUF_PY_PATH = Path(__file__).resolve().parents[1] / "gguf-py"
        if GGUF_PY_PATH.exists():
            sys.path.insert(0, str(GGUF_PY_PATH))
        import gguf

    cfg = json.loads((args.input / "config.json").read_text())
    te = cfg.get("text_encoder_config", {})
    bb = cfg.get("backbone_config", {}) or cfg.get("backbone", {})
    dd = cfg.get("depth_decoder_config", {}) or cfg.get("depth_decoder", {})
    cc = cfg.get("codec_config", {}) or cfg.get("audio_tokenizer_config", {}) or {}

    index_path = args.input / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]

    # ---- validate the mapping BEFORE touching 7 GB of tensors -------------
    mapped, dropped, unmapped, collisions = {}, [], [], {}
    for hf in sorted(weight_map):
        if _drop(hf):
            dropped.append(hf)
            continue
        gg = map_name(hf)
        if gg is None:
            unmapped.append(hf)
            continue
        if gg in collisions:
            log.error("NAME COLLISION: %s and %s both -> %s", collisions[gg], hf, gg)
        collisions[gg] = hf
        mapped[hf] = gg

    log.info("mapped=%d dropped=%d unmapped=%d", len(mapped), len(dropped), len(unmapped))
    if unmapped:
        log.error("%d tensors have no mapping rule -- refusing to write a partial model:",
                  len(unmapped))
        for n in unmapped[:25]:
            log.error("    %s", n)
        if len(unmapped) > 25:
            log.error("    ... and %d more", len(unmapped) - 25)
        return 1
    log.info("dropped (dead weight): %d tensors, incl. %s",
             len(dropped), ", ".join(sorted(DROP_EXACT)))

    if args.dry_run:
        log.info("dry run OK -- every tensor maps to a unique target name")
        return 0

    writer = gguf.GGUFWriter(path=None, arch=ARCH)
    writer.add_name("Breeze-TTS-2")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type({"f32": gguf.LlamaFileType.ALL_F32,
                          "f16": gguf.LlamaFileType.MOSTLY_F16,
                          "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}[args.type])

    def u32(k, v):
        if v is not None:
            writer.add_uint32(f"{ARCH}.{k}", int(v))

    def f32(k, v):
        if v is not None:
            writer.add_float32(f"{ARCH}.{k}", float(v))

    # text encoder -- layer_types drives the per-layer RoPE theta split and is
    # the single easiest thing to get wrong, so it is carried explicitly.
    u32("text_enc.block_count",   te.get("num_hidden_layers"))
    u32("text_enc.embedding_len", te.get("hidden_size"))
    u32("text_enc.ffn_len",       te.get("intermediate_size"))
    u32("text_enc.head_count",    te.get("num_attention_heads"))
    u32("text_enc.head_count_kv", te.get("num_key_value_heads"))
    u32("text_enc.head_dim",      te.get("head_dim"))
    u32("text_enc.vocab_size",    te.get("vocab_size"))
    u32("text_enc.sliding_window", te.get("sliding_window"))
    u32("text_enc.query_pre_attn_scalar", te.get("query_pre_attn_scalar"))
    lt = te.get("layer_types") or []
    if lt:
        writer.add_array(f"{ARCH}.text_enc.layer_types", lt)
        writer.add_array(f"{ARCH}.text_enc.layer_is_full",
                         [1 if t == "full_attention" else 0 for t in lt])
    f32("text_enc.rope_theta_sliding", 10000.0)
    f32("text_enc.rope_theta_full",    1000000.0)

    for pfx, c in (("backbone", bb), ("depth", dd)):
        u32(f"{pfx}.block_count",   c.get("num_hidden_layers"))
        u32(f"{pfx}.embedding_len", c.get("hidden_size"))
        u32(f"{pfx}.ffn_len",       c.get("intermediate_size"))
        u32(f"{pfx}.head_count",    c.get("num_attention_heads"))
        u32(f"{pfx}.head_count_kv", c.get("num_key_value_heads"))
        u32(f"{pfx}.head_dim",      c.get("head_dim"))
        f32(f"{pfx}.rope_theta",    c.get("rope_theta"))
        f32(f"{pfx}.rms_norm_eps",  c.get("rms_norm_eps"))
    u32("depth.num_codebooks", dd.get("num_codebooks") or 16)

    # codec: upsample_rates is read by audio_tokenizer_decoder at load time, so
    # Breeze's [8,6,5,4] vs our vocoder's [8,5,4,3] is DATA, not code.
    ur = cc.get("upsampling_ratios") or [8, 6, 5, 4]
    writer.add_array(f"{ARCH}.codec.upsample_rates", [int(x) for x in ur])
    u32("codec.sample_rate",   cc.get("sampling_rate") or 24000)
    u32("codec.hidden_size",   cc.get("hidden_size") or 512)
    u32("codec.codebook_size", cc.get("codebook_size") or 2048)
    u32("codec.codebook_dim",  cc.get("codebook_dim") or 256)
    u32("codec.num_quantizers", N_VALID_ACOUSTIC + 1)
    f32("codec.frame_rate",    cc.get("frame_rate") or 12.5)

    shards = sorted({weight_map[h] for h in mapped})
    log.info("reading %d shard(s)", len(shards))
    n = 0
    for shard in shards:
        todo = [h for h in mapped if weight_map[h] == shard]
        with SafeTensorsShard(args.input / shard) as f:
            for i, hf in enumerate(todo, 1):
                arr = f.get(hf)
                if arr.dtype.name not in ("float32", "float16"):
                    arr = arr.astype("float32")
                data, dt = pick_dtype(mapped[hf], arr, args.type)
                writer.add_tensor(mapped[hf], data, raw_dtype=dt)
                n += 1
                if i % 100 == 0 or i == len(todo):
                    log.info("  %s  %d/%d", shard, i, len(todo))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log.info("writing %s (%d tensors)", args.output, n)
    writer.write_header_to_file(path=args.output)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    log.info("done: %.2f GB", args.output.stat().st_size / 1e9)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
