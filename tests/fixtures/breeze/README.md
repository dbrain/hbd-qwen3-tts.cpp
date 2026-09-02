# Breeze-TTS-2 parity fixtures

Float32-on-CPU, greedy, dumped from the PyTorch reference
(`github.com/breezeblue-ai/breeze-tts`, Apache-2.0) by
`scripts/dump_breeze_fixtures.py`. `tools/breeze_parity.cpp` checks each port
stage against them as a cosine — bit-exact sampled audio is unattainable
because Breeze can reproduce torch's CUDA RNG and this port does not.

| file | shape | what it pins |
|---|---|---|
| `input_ids` / `text_ids_mask` / `text_ids_len` | [19] | Gemma BPE + prompt assembly (the double-BOS trap) |
| `text_enc_hidden` | [19,1152] | T5Gemma2 encoder, incl. the per-layer RoPE theta split |
| `text_enc_proj` / `prompt_embeds` | [19,2048] | `text_encoder_proj` into the backbone's space |
| `backbone_hidden` | [19,2048] | Qwen3 backbone prefill |
| `lm_logits_step0` | [2052] | the 2051-code + EOS head |
| `depth_logits_cb1` | [2051] | depth decoder, step 1 in isolation (deterministic) |
| `depth_logits_frame0_all` | [15,2051] | all 15 depth steps (chaotic: one flip changes the rest) |
| `frame0_codes` / `greedy_codes` | [16] / [24,16] | end-to-end greedy frame grid |
| `codec_*` | — | Mimi quantiser decode, upsample, transformer, waveform |
| `tokenizer_cases.json` | 24 strings | exact token ids, with and without `add_special_tokens` |

Regenerate (CPU torch container, ~40 s):

    docker build -t breeze-ref-cpu docker/breeze-ref     # torch cpu + transformers 4.57.3
    docker run --rm -v <breeze-ref>:/work:ro -v /path/to/models:/models:ro -v $PWD/tests/fixtures/breeze:/out \
      breeze-ref-cpu python /work/dump_fixtures.py
