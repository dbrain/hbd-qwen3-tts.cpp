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
    # Re-emitted below as 15 transposed 2-D heads, so it must not also flow
    # through the generic name map.
    "depth_decoder.codebooks_head.weight",
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
# gguf-py can only *write* the legacy quants (no K-quants), which is fine: the
# depth decoder and backbone are bandwidth-bound at batch 1, so what matters is
# bytes-per-weight, and Q4_0/Q5_0 have well-tuned CUDA MMVQ kernels.
_QUANTS = {
    "f32":  None, "f16": None,
    "q8_0": "Q8_0", "q5_1": "Q5_1", "q5_0": "Q5_0", "q4_1": "Q4_1", "q4_0": "Q4_0",
}


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
    qname = _QUANTS.get(out_type)
    if qname is None:
        raise ValueError(f"unknown --type {out_type}")
    qt = getattr(gguf.GGMLQuantizationType, qname)
    try:
        return gguf.quants.quantize(arr.astype(np.float32), qt), qt
    except Exception as e:  # noqa: BLE001
        log.warning("%s failed for %s (%s); keeping F16", qname, name, e)
        return arr.astype(np.float16), gguf.GGMLQuantizationType.F16


# ---------------------------------------------------------------------------
# Gemma BPE tokenizer -> GGUF.
#
# tokenizer.json is a `tokenizers` BPE with byte_fallback, a Replace(" " ->
# "\u2581") normalizer and NO pre-tokenizer regex (the Split on " " is a no-op
# after normalisation). 262144 BPE entries + 14 added ids on top = 262158.
# `added_tokens` (6428 of them) are matched literally on the RAW text before
# normalisation, so they are carried in their own array rather than inferred.
def _emit_tokenizer(writer, model_dir, log):
    tj = json.loads((model_dir / "tokenizer.json").read_text())
    model = tj["model"]
    vocab = model["vocab"]
    added = tj["added_tokens"]

    n_vocab = max(max(vocab.values()), max(a["id"] for a in added)) + 1
    tokens = [""] * n_vocab
    ttype  = [1] * n_vocab          # 1 = normal, 3 = control/added, 6 = byte
    for tok, i in vocab.items():
        tokens[i] = tok
        if tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6:
            ttype[i] = 6
    for a in added:
        tokens[a["id"]] = a["content"]
        ttype[a["id"]] = 3
    missing = [i for i, t in enumerate(tokens) if t == ""]
    if missing:
        raise SystemExit(f"tokenizer: {len(missing)} unfilled ids, first {missing[:5]}")

    merges = ["%s %s" % (a, b) for a, b in model["merges"]]

    writer.add_tokenizer_model("bpe-gemma")
    writer.add_token_list(tokens)
    writer.add_token_types(ttype)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(tj["post_processor"]["special_tokens"]["<bos>"]["ids"][0])
    writer.add_eos_token_id(1)
    writer.add_pad_token_id(0)
    writer.add_unk_token_id(3)
    # The literal set the pre-normalisation splitter matches. Kept separate
    # from token_type so the C++ side never has to guess which ids are added.
    writer.add_array(f"{ARCH}.tokenizer.added_tokens",
                     [a["content"] for a in added])
    writer.add_array(f"{ARCH}.tokenizer.added_ids",
                     [int(a["id"]) for a in added])
    writer.add_string(f"{ARCH}.tokenizer.metaspace", "\u2581")
    writer.add_uint32(f"{ARCH}.tokenizer.byte_fallback", 1)
    log.info("tokenizer: %d tokens, %d merges, %d added",
             n_vocab, len(merges), len(added))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="HF model directory")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--type", default="q8_0", choices=list(_QUANTS))
    # Per-subsystem overrides. The depth decoder is re-read 15x per 80 ms frame
    # and the backbone once, so their bytes-per-weight dominates RTF; the text
    # encoder runs once per request and only costs VRAM.
    for sub in ("text-enc", "backbone", "depth", "codec"):
        ap.add_argument(f"--type-{sub}", default=None, choices=list(_QUANTS))
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the name mapping without reading tensor data")
    args = ap.parse_args()

    if not args.dry_run:
        import numpy as np
        GGUF_PY_PATH = Path(__file__).resolve().parents[1] / "gguf-py"
        if GGUF_PY_PATH.exists():
            sys.path.insert(0, str(GGUF_PY_PATH))
        import gguf
        globals()["np"] = np

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

    subs = {"text_enc.": args.type_text_enc, "backbone.": args.type_backbone,
            "depth.": args.type_depth, "codec.": args.type_codec}

    def sub_type(name):
        for pfx, t in subs.items():
            if t and name.startswith(pfx):
                return t
        return args.type

    if any(subs.values()):
        log.info("per-subsystem types: %s (default %s)",
                 {k: v for k, v in subs.items() if v}, args.type)

    writer = gguf.GGUFWriter(path=None, arch=ARCH)
    writer.add_name("Breeze-TTS-2")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type({"f32":  gguf.LlamaFileType.ALL_F32,
                          "f16":  gguf.LlamaFileType.MOSTLY_F16,
                          "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
                          "q5_1": gguf.LlamaFileType.MOSTLY_Q5_1,
                          "q5_0": gguf.LlamaFileType.MOSTLY_Q5_0,
                          "q4_1": gguf.LlamaFileType.MOSTLY_Q4_1,
                          "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0}[args.type])

    def u32(k, v):
        if v is not None:
            writer.add_uint32(f"{ARCH}.{k}", int(v))

    def f32(k, v):
        if v is not None:
            writer.add_float32(f"{ARCH}.{k}", float(v))

    def sstr(k, v):
        if v is not None:
            writer.add_string(f"{ARCH}.{k}", str(v))

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
    f32("text_enc.rms_norm_eps",  te.get("rms_norm_eps") or 1e-6)
    u32("text_enc.eoi_token_index", te.get("eoi_token_index") or 256000)
    lt = te.get("layer_types") or []
    if lt:
        writer.add_array(f"{ARCH}.text_enc.layer_types", lt)
        writer.add_array(f"{ARCH}.text_enc.layer_is_full",
                         [1 if t == "full_attention" else 0 for t in lt])
    # rope_parameters is per layer-type. full_attention here is rope_type
    # "linear" with factor 8 -- inv_freq /= 8. Missing that silently detunes
    # every 6th layer, so both thetas AND the scale factor are carried.
    rp = te.get("rope_parameters") or {}
    slid = rp.get("sliding_attention") or {}
    full = rp.get("full_attention") or {}
    f32("text_enc.rope_theta_sliding", slid.get("rope_theta") or 10000.0)
    f32("text_enc.rope_theta_full",    full.get("rope_theta") or 1000000.0)
    sstr("text_enc.rope_type_sliding", slid.get("rope_type") or "default")
    sstr("text_enc.rope_type_full",    full.get("rope_type") or "default")
    f32("text_enc.rope_factor_full",   full.get("factor") or 1.0)

    for pfx, c in (("backbone", bb), ("depth", dd)):
        u32(f"{pfx}.block_count",   c.get("num_hidden_layers"))
        u32(f"{pfx}.embedding_len", c.get("hidden_size"))
        u32(f"{pfx}.ffn_len",       c.get("intermediate_size"))
        u32(f"{pfx}.head_count",    c.get("num_attention_heads"))
        u32(f"{pfx}.head_count_kv", c.get("num_key_value_heads"))
        u32(f"{pfx}.head_dim",      c.get("head_dim"))
        f32(f"{pfx}.rope_theta",    c.get("rope_theta"))
        f32(f"{pfx}.rms_norm_eps",  c.get("rms_norm_eps"))
        # rope_scaling: the backbone's is null (plain NeoX); the depth
        # decoder's is llama3 with original_max_position_embeddings=16, which
        # rewrites 35 of its 64 inv_freq entries. Silently skipping it detunes
        # every codebook past the first few.
        rs = c.get("rope_scaling") or {}
        sstr(f"{pfx}.rope_scaling_type", rs.get("rope_type") or "none")
        if rs:
            f32(f"{pfx}.rope_scaling_factor",       rs.get("factor"))
            f32(f"{pfx}.rope_scaling_low_freq",     rs.get("low_freq_factor"))
            f32(f"{pfx}.rope_scaling_high_freq",    rs.get("high_freq_factor"))
            u32(f"{pfx}.rope_scaling_orig_ctx",     rs.get("original_max_position_embeddings"))
        u32(f"{pfx}.max_position", c.get("max_position_embeddings"))
    u32("depth.num_codebooks", dd.get("num_codebooks") or 16)
    u32("depth.vocab_size",    dd.get("vocab_size") or 2051)
    u32("backbone.audio_vocab_size", cfg.get("vocab_size") or 2051)
    u32("backbone.lm_head_size",     (cfg.get("vocab_size") or 2051) + 1)
    u32("backbone.num_codebooks",    cfg.get("num_codebooks") or 16)

    # codec: upsample_rates is read by the Mimi decoder at load time, so
    # Breeze's [8,6,5,4] vs our vocoder's [8,5,4,3] is DATA, not code.
    ur = cc.get("upsampling_ratios") or [8, 6, 5, 4]
    writer.add_array(f"{ARCH}.codec.upsample_rates", [int(x) for x in ur])
    u32("codec.sample_rate",   cc.get("sampling_rate") or 24000)
    u32("codec.hidden_size",   cc.get("hidden_size") or 512)
    u32("codec.codebook_size", cc.get("codebook_size") or 2048)
    u32("codec.codebook_dim",  cc.get("codebook_dim") or 256)
    u32("codec.num_quantizers", N_VALID_ACOUSTIC + 1)
    f32("codec.frame_rate",    cc.get("_frame_rate") or cc.get("frame_rate") or 12.5)
    u32("codec.num_tfm_layers", cc.get("num_hidden_layers") or 8)
    u32("codec.num_heads",     cc.get("num_attention_heads") or 8)
    u32("codec.head_dim",      cc.get("head_dim") or 64)
    u32("codec.ffn_dim",       cc.get("intermediate_size") or 2048)
    u32("codec.sliding_window", cc.get("sliding_window") or 250)
    f32("codec.norm_eps",      cc.get("norm_eps") or 1e-5)
    f32("codec.rope_theta",    cc.get("rope_theta") or 10000.0)
    u32("codec.num_filters",   cc.get("num_filters") or 64)
    u32("codec.kernel_size",   cc.get("kernel_size") or 7)
    u32("codec.last_kernel_size", cc.get("last_kernel_size") or 3)
    u32("codec.residual_kernel_size", cc.get("residual_kernel_size") or 3)
    u32("codec.dilation_growth_rate", cc.get("dilation_growth_rate") or 2)
    u32("codec.compress",      cc.get("compress") or 2)
    u32("codec.num_residual_layers", cc.get("num_residual_layers") or 1)
    u32("codec.upsample_groups", cc.get("upsample_groups") or 512)
    u32("codec.causal",        1 if cc.get("use_causal_conv", True) else 0)
    sstr("codec.pad_mode",     cc.get("pad_mode") or "constant")
    f32("codec.trim_right_ratio", cc.get("trim_right_ratio") or 1.0)
    f32("codec.layer_scale_initial", cc.get("layer_scale_initial_scale") or 0.01)

    # special token ids -- read from config, never hardcoded downstream
    stc = cfg.get("text_encoder_special_tokens_config", {}) or {}
    tid = stc.get("token_ids", {}) or {}
    u32("tokens.audio",         cfg.get("audio_token_id"))
    u32("tokens.audio_eos",     cfg.get("audio_eos_token_id"))
    u32("tokens.ins_bos",       tid.get("<ins_bos>"))
    u32("tokens.ins_eos",       tid.get("<ins_eos>"))
    u32("tokens.speaker_base",  tid.get("[S0]"))
    u32("tokens.n_speakers",    len(stc.get("speaker_tokens") or []) or 10)
    u32("tokens.bos",           cfg.get("bos_token_id"))
    u32("tokens.codebook_eos",  cfg.get("codebook_eos_token_id"))
    u32("tokens.codebook_pad",  cfg.get("codebook_pad_token_id"))
    u32("tokens.text_vocab_size", cfg.get("text_vocab_size"))

    _emit_tokenizer(writer, args.input, log)

    # depth_decoder.codebooks_head is [15, 1024, 2051] -- one head per codebook.
    # Emitted TRANSPOSED, as [15, 2051, 1024], for two reasons:
    #   * ggml_mul_mat wants [in, out] per head, and the transpose done here is
    #     free where doing it in the 15-step inner loop would be a 4 MB copy;
    #   * the original's last dim is 2051, not a multiple of the Q8_0 block, so
    #     it cannot be quantised at all. Transposed it ends in 1024 and can.
    # Kept as ONE 3-D tensor so the depth graph can evaluate all heads with a
    # single broadcast mul_mat and stay shape-static across the AR steps.
    head_hf = "depth_decoder.codebooks_head.weight"
    if head_hf not in weight_map:
        log.error("checkpoint has no %s", head_hf)
        return 1
    with SafeTensorsShard(args.input / weight_map[head_hf]) as f:
        head = f.get(head_hf)
    head_t = np.ascontiguousarray(head.transpose(0, 2, 1))   # [n_heads, vocab, hidden]
    log.info("codebooks_head %s -> %s (transposed)", head.shape, head_t.shape)
    data, dt = pick_dtype("depth.codebooks_head.weight", head_t, sub_type("depth.codebooks_head.weight"))
    writer.add_tensor("depth.codebooks_head.weight", data, raw_dtype=dt)

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
                data, dt = pick_dtype(mapped[hf], arr, sub_type(mapped[hf]))
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
