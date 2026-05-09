# qwen3-tts.cpp

> ⚠️ PSAs/TLDRs from the potato that told Claude to make [khimaros/qwen3-tts.cpp](https://github.com/khimaros/qwen3-tts.cpp) usable on my hardware
> - LLM generated noise - I'm a software engineer but I won't pretend to know anything about this space, I just wanted my stuff to run quickerer and when the wins started rolling in framed the goal as "go insane" then "while true; echo 'NOT INSANE ENOUGH, I WANT RTF TO BE AT LEAST.... 3 TIMES AS BIG/SMALL AS THIS!'; done". Current master is the result of the Claudes starting to push back with evidence that we were at the cap. Will ask Mythos if it ever comes out instead of just being pre-shilled around the internet.
> - Entirely tested and targetted at my hardware (RTX 3060 12GB, AMD misc), may explode on anything else or run slower. Likely any CUDA device would benefit but I'm no nvidiaologist
> - RTF is backwards, 1/RTF to get the right number. I.E. 1/4.29 == 0.23. I.E. I can generate speech at 4x real time consistently, which seems decent looking at people whinging about how slow qwen3-tts is on much better hardware
> - I was legitimately getting 0.1 RTF on the upstream fork, i.e. "10" real RTF. I'm sure there was something stupid going on and RTF 0.1 isn't "standard", but the jump from high 1's to low to mid 4's was effort.
> - "V2 48 kHz vocoder" ended up being clippy - even using the python impl (i.e. not just this project), do not recommend unless you care enough to understand the "why" there.
> - Q4_K_M is another speed boost and VRAM savings, and TBH I can't decide if it's worse quality or if the numbers are clouding my brain. The pacing of the speech maybe "feels" weird? I guess a bit more uncanny valley than Q8.
> - TL;DR TL;DR "Works for me, too embarassing to upstream because I'm not going to be 'that guy' who PRs something he does not understand short of laughing at 'megakernel' sticking in claudes brain". Here be dragons.
>
> Unclear from claude noise below features: 
> - You can use VoiceDesign AND clone using an extracted Base speaker encoder with this, VoiceDesign is usually instruct only, Base is usually clone only. This has worked well for me, but I'm assuming there's a reason why Qwen3-TTS didn't make this default.
> - Instruct/cloned voices can be persisted in a way that is quick to load and potentially keeps the voice more "samey" across runs, note: if you switch quants the persisted versions need to be replaced.
> - Added noise I care about (unload to 0 VRAM/load) - I'm cheap and an idiot so balance services so my 12GB VRAM goes a long way
> - Streamining supports mp3/ogg/aac on top of wav - and .. I haven't tested it in a long time but streaming made RTF worse - so if you don't care you may see better numbers (you know, if it still works)

Self-hosted [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) inference server in C++ via GGML — fork of [khimaros/qwen3-tts.cpp](https://github.com/khimaros/qwen3-tts.cpp), itself forked from [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp). Tuned for low-VRAM single-GPU homelab hosting: voice cloning, streaming PCM, persistent voice cache, lazy-load + idle-unload, OpenAI-compatible HTTP API.

This fork pairs with [dbrain/ggml](https://github.com/dbrain/ggml) (submodule) for CUDA kernels and infrastructure that don't fit upstream as-is.

## At a glance

Single RTX 3060 12 GB / Ampere, calm GPU, 74-test breakit suite × 2 seeds, V1 24 kHz vocoder:

| Talker quant | RTF avg | RTF max | TTFA med | VRAM peak | Notes |
|---|---:|---:|---:|---:|---|
| **Q4_K_M** | **4.84×** | 5.03× | 43 ms | 2.5 GiB | smallest VRAM, fastest decode |
| **Q8_0** (default) | 4.29× | 4.41× | 47 ms | 3.5 GiB | quality default |
| **F16** | 2.86× | 2.95× | 66 ms | 5.1 GiB | reference precision |

> **RTF ≥ 1.0 = realtime.** All three ship faster than realtime on a 5-year-old consumer GPU. RTF here = audio-seconds produced per wall-second of compute (audio/wall convention; matches upstream `faster-qwen3-tts`'s bench).

Full table including V2 48 kHz vocoder + methodology in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Where we started → where we ended

Anecdotal starting points on the same RTX 3060 (per measurements at fork time, not re-bench-able now upstream has moved):

| Stack | RTF | This fork's multiplier |
|---|---:|---:|
| upstream khimaros, naive config | ~0.1 | **~43×** |
| upstream PyTorch + transformers (BF16) | ~0.9 | **~4.8×** |
| `qwen-research/faster-qwen3-tts` (C++) | ~1.8 | **~2.4×** |
| **this fork @ Q8_0, V1** | **4.29** | — |

## How we got here

**Vocoder side (early forks).** Tensor-core `wmma` kernels for `conv_1d_direct` and `conv_transpose_1d` replace the im2col + cuBLAS path; on-the-fly im2col index calculation, F16 weights only. Single biggest jump. Followed by F16 cascade activations (every dec_block intermediate runs F16 not F32 — halves the scheduler arena), and a fused `GGML_OP_SNAKE` op (replaces a `pow(sin(αx), 2)/α + βx` broadcast chain at every vocoder residual).

**Talker side (megakernel-v0).** Shape-specialized Q8/Q8 MMVQ kernels for the 8 hot `(K,N)` shapes in this model (eliminates ggml's generic dispatch overhead at M=1). On top: a 3-line topology fix in `tts_transformer.cpp` that lets ggml's allocator give Q/K/V distinct slots, unblocking single-launch QKV + gate-up fusion. Then the code-predictor's 14 sequential AR steps fold into one cgraph (`cudaGraphExecUpdate` reuses ~150 launches per frame); GPU-side Gumbel-max sampling replaces host roundtrips; the talker step graph is cgraph-cached too (single-entry, `kv_n_eff` bucket-keyed); async vocoder dispatch on a dedicated CUDA backend pipelines behind the talker AR loop.

**Streaming + voice persistence.** Persistent F16 KV slab + `flash_attn_ext` (was rebuilt via `ggml_concat` per call, growing unbounded). `voice.bundle` (speaker embedding + ref_codes + model_id stamp, atomically written) + `voice.warmup` (vocoder ICL warm-up snapshot, restored in 5–7 ms instead of a 700–1200 ms re-decode) make voice clone TTFA flat.

→ Lever-by-lever detail and the no-go list (things that *didn't* win) in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Hardware

- **Tested**: RTX 3060 12 GB (Ampere, sm_86), Linux + CUDA 12.x.
- **Should work**: any CUDA device with **sm_70+** (Volta and later — the `wmma` kernels need `__CUDA_ARCH__ >= 700`). Default build covers Turing → Hopper (`-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89;90`).
- **Trim per host**: `--build-arg QWEN3_TTS_CUDA_ARCHS=86` (or whichever single arch you target) drops the idle CUDA-context floor by ~15–20 MiB per arch.
- **VRAM**: 4 GiB fits Q4_K_M comfortably; 6 GiB fits Q8_0; 8 GiB fits F16.

## API

In-process HTTP server (`qwen3-tts-server`), OpenAI-compatible with this fork's extras layered on:

- `POST /v1/audio/speech` — main TTS endpoint. Body accepts `input` / `voice` / `instructions` (voice design) / `seed` / `temperature` / `top_k` / `repetition_penalty` / `language` / `max_audio_tokens` / `stream_batch_size`. Streaming via `stream_format=sse` + `response_format=pcm` for low TTFA.
- `POST /v1/audio/voices` — register a voice clone (multipart: `name` + `audio_sample` WAV + optional `ref_text` ICL transcript). Lazy loads encoders, persists `voice.bundle` + (after first synth) `voice.warmup` to the archive volume.
- `GET /v1/audio/voices` — list voices.
- `DELETE /v1/audio/voices/{name}` — remove from in-memory map. `voice.bundle` on disk persists by design (re-register or restart-rescan picks it back up).
- `POST /v1/admin/load` / `POST /v1/admin/unload` — explicit GPU lifecycle control.
- `GET /health`, `GET /v1/models`, `GET /v1/audio/languages`.

Lifecycle env vars worth knowing about:

- `QWEN3_TTS_LAZY_LOAD=1` — boot without touching GPU; first synth triggers materialization (~1 s extra cold-path).
- `QWEN3_TTS_IDLE_UNLOAD_SECONDS=N` — release model VRAM after `N` seconds of inactivity. Reload is automatic on next request (~30 s default in dev images). In single-process mode the ~370 MiB CUDA-context floor stays resident; pair with `QWEN3_TTS_WORKER_ISOLATION=1` to drop to zero.
- `QWEN3_TTS_WORKER_ISOLATION=1` — fork the GPU work into a subprocess child; `/v1/admin/unload` and the idle-unload watchdog `SIGKILL` the child so the CUDA primary context is torn down by the kernel and **all VRAM is reclaimed**. Parent process never touches CUDA. Per-call IPC overhead is sub-noise (p99 ~16 µs per AUDIO_FRAME). See [docs/ARCHITECTURE.md#worker-isolation](docs/ARCHITECTURE.md#worker-isolation).
- `QWEN3_TTS_VOICE_ARCHIVE_DIR=/path` — where bundles + warmup snapshots persist.
- `QWEN3_TTS_KV_Q8=1` (default) — Q8_0 KV cache for the talker, ~64 MiB peak savings, no measurable RTF cost.
- `QWEN3_TTS_MAX_INPUT_CHARS=6144` (default) — input text cap.

→ Full endpoint + parameter + env-var reference in [docs/API.md](docs/API.md).

## Build & run

### Docker (recommended)

```bash
docker build \
  --build-arg QWEN3_TTS_CUDA_ARCHS=86 \
  -t qwen3-tts.cpp:local .

docker run --gpus all -p 8000:8000 \
  -v qwen3-tts-models:/root/.cache/huggingface \
  -v qwen3-tts-voices:/app/voice-archive \
  -e HF_TOKEN="$HF_TOKEN" \
  qwen3-tts.cpp:local \
  --hf-repo    khimaros/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF:Q8_0 \
  --hf-repo-v  khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF:F16 \
  --hf-repo-se khimaros/Qwen3-TTS-12Hz-1.7B-Base-GGUF:Q8_0 \
  -H 0.0.0.0 -p 8000 -j 6
```

`--hf-repo-se` is the speaker encoder source. The Base GGUF works because the binary only reads its `spk_enc.*` tensors — alternatively use [`dbrains/Qwen3-TTS-12Hz-Speaker-Encoder-GGUF`](https://huggingface.co/dbrains/Qwen3-TTS-12Hz-Speaker-Encoder-GGUF) (~24 MB sidecar, F16) to skip the 2.4 GB Base download.

For 48 kHz output, swap `--hf-repo-v` to [`dbrains/Qwen3-TTS-Tokenizer-12Hz-48kHz-GGUF:F16`](https://huggingface.co/dbrains/Qwen3-TTS-Tokenizer-12Hz-48kHz-GGUF).

For Q4_K_M (~1 GB lighter VRAM, ~13% faster RTF on Ampere), use [`dbrains/Qwen3-TTS-12Hz-1.7B-VoiceDesign-Q4_K_M-GGUF:Q4_K_M`](https://huggingface.co/dbrains/Qwen3-TTS-12Hz-1.7B-VoiceDesign-Q4_K_M-GGUF) for `--hf-repo`.

### Native

```bash
git clone --recurse-submodules https://github.com/dbrain/qwen3-tts.cpp.git
cd qwen3-tts.cpp
cmake -S . -B build -G Ninja \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
ninja -C build qwen3-tts-server
./build/qwen3-tts-server --hf-repo khimaros/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF:Q8_0 -H 0.0.0.0 -p 8000
```

## Models

The server auto-downloads GGUFs from HF on first request. Three pieces:

- **Talker** (`--hf-repo`): the 1.7B transformer. Q8_0 (default), Q4_K_M (low-VRAM), or F16. Locally-converted Q4_K_M at [`dbrains/Qwen3-TTS-12Hz-1.7B-VoiceDesign-Q4_K_M-GGUF`](https://huggingface.co/dbrains/Qwen3-TTS-12Hz-1.7B-VoiceDesign-Q4_K_M-GGUF). F16 not published — convert with `scripts/convert_tts_to_gguf.py -t f16` from the HF safetensors.
- **Vocoder** (`--hf-repo-v`): tokenizer/decoder. F16 at 24 kHz from khimaros ([`khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF)) or 48 kHz from dbrains ([`dbrains/Qwen3-TTS-Tokenizer-12Hz-48kHz-GGUF`](https://huggingface.co/dbrains/Qwen3-TTS-Tokenizer-12Hz-48kHz-GGUF)). **Vocoder must be F16** — the wmma kernels demand it.
- **Speaker encoder** (`--hf-repo-se`): F16 ECAPA-TDNN. Either the full Base GGUF (~2.4 GB, only `spk_enc.*` is read) or [`dbrains/Qwen3-TTS-12Hz-Speaker-Encoder-GGUF`](https://huggingface.co/dbrains/Qwen3-TTS-12Hz-Speaker-Encoder-GGUF) (~24 MB sidecar).

## Benchmarks

Headline table above is Q8/Q4/F16 × V1 vocoder. Full Q8/Q4/F16 × {V1 24 kHz, V2 48 kHz} matrix, per-prompt distributions, methodology (incl. the read1() Python TTFA gotcha), and reproduce instructions: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## ggml fork

This fork's submodule pointer references [dbrain/ggml@master](https://github.com/dbrain/ggml), which carries:

- `GGML_OP_SNAKE` (CPU + CUDA) — fused `α·sin(βx)² + γx`, vocoder activation.
- `GGML_OP_CONV_1D_DIRECT` smem-tiled CUDA + tensor-core `wmma` variants for both conv_1d_direct and conv_transpose_1d (F16 weights).
- F16 in/out paths for conv_1d_direct, conv_transpose_1d, snake, acc + `ggml_*_to` dst-type API variants.
- I32 + F16 paths through `concat`.
- `mul_mat` dispatcher hook + per-op hook (lets the megakernel install shape-specialized Q8 MMVQ at runtime).
- Stream priority knob for multi-backend latency-sensitive layouts.
- `Q4_K` `get_rows` CUDA kernel (was missing — Q4 talker needs it for codec embed lookup).
- Sched fixes (hash_set sizing, null-checks).

These are mostly Qwen3-TTS-specific or invasive enough that they don't merge upstream as-is, hence the fork.

## Limitations / no-gos

- **Vocoder weights are F16-only on CUDA** — the wmma kernels demand it. Q4 vocoder would force the im2col + cuBLAS path back and tank RTF.
- **Speaker encoder weights are F16-only** — same reason. Bit-identical to upstream BF16 in practice (ECAPA-TDNN class).
- **CUDA-context floor (~370 MiB)** stays resident in single-process mode — `cuMemFree` releases the model but the cuBLAS / kernel-cache / driver state only goes away when the process exits. Set `QWEN3_TTS_WORKER_ISOLATION=1` and the unload-watchdog / `/v1/admin/unload` instead `SIGKILL` a forked child that owns the CUDA context, dropping post-unload VRAM to zero. (Cold respawn on next request adds ~1.2 s.)
- **Q5_K_M is slower than Q4_K_M on Ampere** via ggml MMQ — practical quant choice is Q4_K_M or Q8_0 only.
- See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#dead-ends) for things that were tried + measured + ripped out (INT8 mma at M=1, fused-AR, etc.).

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) by Alibaba Qwen.
- [`predict-woo/qwen3-tts.cpp`](https://github.com/predict-woo/qwen3-tts.cpp) — the original C++ port.
- [`khimaros/qwen3-tts.cpp`](https://github.com/khimaros/qwen3-tts.cpp) — 1.7B model support, ICL voice cloning, voice design, multi-language, OpenAI HTTP server, streaming vocoder, GGUF artifact publishing.
- [GGML](https://github.com/ggml-org/ggml) tensor library.
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) vocoder architecture.

## License

Inherits the upstream license (see commit history).
