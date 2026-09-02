"""Dump per-stage fixtures from the Breeze reference for C++ parity checks.

float32 on CPU, greedy everywhere. Writes .bin (float32/int32 raw) + a manifest.
"""
from __future__ import annotations
import json, os, sys, hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.breeze import BreezeForConditionalGeneration          # noqa: E402
from breeze_infer.templates import get_template, prepare_inputs   # noqa: E402

CKPT = Path(os.environ.get("BREEZE_CKPT", "/models/Breeze-TTS-2-hf"))
OUT  = Path(os.environ.get("OUT", "/out"))
TEXT = os.environ.get("TEXT", "The quick brown fox jumps over the lazy dog.")
INSTR = os.environ.get("INSTR", "Speak clearly and naturally.")
NFRAMES = int(os.environ.get("NFRAMES", "24"))

OUT.mkdir(parents=True, exist_ok=True)
man = {}

def save(name, arr):
    a = np.ascontiguousarray(arr)
    p = OUT / f"{name}.bin"
    a.tofile(p)
    man[name] = {"shape": list(a.shape), "dtype": str(a.dtype),
                 "sha256": hashlib.sha256(a.tobytes()).hexdigest()[:16]}
    print(f"  saved {name} {a.shape} {a.dtype}", flush=True)

torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(CKPT)
print("loading model (fp32, cpu)...", flush=True)
from transformers import AutoConfig
_cfg = AutoConfig.from_pretrained(CKPT)
_cfg.text_encoder_config.preferred_attn_implementation = "eager"
_cfg.text_encoder_config._attn_implementation = "eager"
_cfg._attn_implementation = "eager"
model = BreezeForConditionalGeneration.from_pretrained(
    CKPT, config=_cfg, dtype=torch.float32, attn_implementation="eager")
model.text_encoder.config._attn_implementation = "eager"
model.text_encoder.config.preferred_attn_implementation = "eager"
model = model.float()
model.eval()
cfg = model.config
print("model loaded", flush=True)

# ---------------------------------------------------------------- tokenization
request = {"id": "fix", "text": TEXT, "instruction": INSTR, "speaker": "S0"}
inputs = prepare_inputs(tok, None, model, [request], get_template("tts_instruction"),
                        guidance_scale=1.0, guidance_scale_ref=None, guidance_scale_ins=None)
input_ids = inputs["input_ids"]
text_ids_mask = inputs["text_ids_mask"]
text_ids_len = inputs["text_ids_len"]
print("input_ids", input_ids.shape, input_ids[0].tolist(), flush=True)
save("input_ids", input_ids[0].numpy().astype(np.int32))
save("text_ids_mask", text_ids_mask[0].numpy().astype(np.int32))
save("text_ids_len", text_ids_len.numpy().astype(np.int32))
man["text"] = TEXT
man["instruction"] = INSTR
man["rendered"] = tok.decode(input_ids[0].tolist(), skip_special_tokens=False)

# ---------------------------------------------------------------- text encoder
seg = input_ids[0][text_ids_mask[0]]
te_out = model.text_encoder(input_ids=seg.unsqueeze(0),
                            attention_mask=torch.ones(1, seg.shape[0], dtype=torch.long),
                            position_ids=torch.arange(seg.shape[0]).unsqueeze(0),
                            output_hidden_states=True)
save("text_enc_hidden", te_out.last_hidden_state[0].float().numpy())
save("text_enc_layer0_in", te_out.hidden_states[0][0].float().numpy())   # post-embed
save("text_enc_layer1_out", te_out.hidden_states[1][0].float().numpy())
proj = model.text_encoder_proj(te_out.last_hidden_state)[0]
save("text_enc_proj", proj.float().numpy())

# ---------------------------------------------------------------- prompt embeds
merged = model._merge_input_ids_with_input_values(
    input_ids=input_ids, input_values=inputs.get("input_values"),
    text_ids_mask=text_ids_mask, text_ids_len=text_ids_len,
    attention_mask=inputs["attention_mask"])
inputs_embeds = merged["inputs_embeds"]
save("prompt_embeds", inputs_embeds[0].float().numpy())

# ---------------------------------------------------------------- backbone prefill
bb = model.backbone_model(inputs_embeds=inputs_embeds, use_cache=True)
h = bb.last_hidden_state
save("backbone_hidden", h[0].float().numpy())
logits0 = model.lm_head(h[:, -1, :])[0]
save("lm_logits_step0", logits0.float().numpy())
c0 = int(torch.argmax(logits0[: cfg.vocab_size + 1]).item())
man["greedy_c0_step0"] = c0
print("greedy c0 step0 =", c0, flush=True)

# ---------------------------------------------------------------- depth decoder, frame 0
bhs = h[:, -1, :]
dd_in = torch.tensor([[0, c0]], dtype=torch.long)
dd = model.depth_decoder(input_ids=dd_in, backbone_last_hidden_state=bhs,
                         use_cache=False, return_dict=True)
save("depth_logits_cb1", dd.logits[0, 0].float().numpy())

# full greedy depth for frame 0 (no cache, re-run each step like the CFG path)
seqs = dd_in.clone()
depth_all = []
for step in range(cfg.num_codebooks - 1):
    out = model.depth_decoder(input_ids=seqs, backbone_last_hidden_state=bhs,
                              use_cache=False, return_dict=True)
    lg = out.logits[:, -1, :].float()
    lg[..., cfg.codec_config.codebook_size : cfg.vocab_size] = float("-inf")
    depth_all.append(lg[0].numpy().copy())
    nxt = torch.argmax(lg, dim=-1, keepdim=True)
    seqs = torch.cat([seqs, nxt], dim=-1)
save("depth_logits_frame0_all", np.stack(depth_all))
frame0 = seqs[0, 1:].numpy().astype(np.int32)
save("frame0_codes", frame0)
print("frame0 codes", frame0.tolist(), flush=True)

# ---------------------------------------------------------------- greedy generation
print(f"greedy generating {NFRAMES} frames...", flush=True)
model.generation_config.do_sample = False
model.generation_config.temperature = 1.0
model.generation_config.top_p = 1.0
model.generation_config.top_k = 0
model.generation_config.max_new_tokens = NFRAMES
vars(model.depth_decoder.generation_config).update(
    {"_from_model_config": False, "do_sample": False, "temperature": 1.0,
     "top_p": 1.0, "top_k": 0})
gen = model.generate(**{k: v for k, v in inputs.items() if v is not None},
                     max_new_tokens=NFRAMES, do_sample=False)
codes = gen[0].numpy().astype(np.int32)   # [T, 16]
save("greedy_codes", codes)
print("codes", codes.shape, codes[:3].tolist(), flush=True)

# ---------------------------------------------------------------- codec decode
valid = codes[(codes < cfg.codec_config.codebook_size).all(axis=1)]
save("codec_input_codes", valid)
ct = torch.tensor(valid.T, dtype=torch.long).unsqueeze(0)   # [1, 16, T]
dec = model.codec_model.decode(ct)
wav = dec.audio_values[0, 0].float().numpy()
save("codec_wav", wav)

# intermediate: quantizer.decode + upsample + decoder_transformer
emb = model.codec_model.quantizer.decode(ct)
save("codec_quant_emb", emb[0].float().numpy())
up = model.codec_model.upsample(emb)
save("codec_upsample", up[0].float().numpy())
dt = model.codec_model.decoder_transformer(up.transpose(1, 2))
save("codec_dec_tfm", dt[0][0].float().numpy())

(OUT / "manifest.json").write_text(json.dumps(man, indent=2))
print("done", flush=True)
