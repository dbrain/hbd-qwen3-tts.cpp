# qwen3-tts.cpp

## Added In This Fork

This fork ([dbrain/qwen3-tts.cpp](https://github.com/dbrain/qwen3-tts.cpp), paired with [dbrain/ggml](https://github.com/dbrain/ggml)) layers the following on top of [khimaros/qwen3-tts.cpp](https://github.com/khimaros/qwen3-tts.cpp). Headline numbers on a single RTX 3060 / 12 GB / Ampere, Q8_0 talker, V1 24 kHz vocoder: **~3.4× realtime** (i.e. ~3.4 s of 24 kHz audio per wall-second of compute), default-budget paragraph **~3.05 GB peak VRAM**, max-budget cap-out (`max_audio_tokens=8192`) **~3.45 GB peak**. Compared to upstream PyTorch + transformers on the same model (BF16), that's roughly **+80 %** end-to-end at single-stream decode while sitting under 4 GB.

### Performance

- **wmma `conv_1d_direct` and `conv_transpose_1d` kernels** in dbrain/ggml — single biggest fork-arc jump (RTF 1.87 → 2.61 on conv_1d_direct, then 2.65 → 3.37 once the smem-tiled F16 `conv_transpose_1d` landed). Tensor-core `nvcuda::wmma` GEMM with on-the-fly im2col index calculation, replacing the im2col-temp + cuBLAS path. **F16 weights only** (vocoder cascade — do not quantise the vocoder beyond F16, that drops the kernel).
- **F16 cascade activations** — every cascade intermediate (4-5 dec_blocks × `conv_transpose_1d` + 3 residuals each) runs F16 instead of F32. The wmma accumulator and snake math stay F32 internally, only dst writes truncate. Halves the scheduler arena: vocoder `sched_cu` 144 → 94 MiB at chunk=30. `QWEN3_TTS_VOCODER_FP16_CASCADE=0` reverts.
- **Streaming batch=30 default** — `--streaming-batch-size 30`, `--streaming-first-batch-size 1`. Cuts cascade buffer size linearly vs the prior batch=60 default (each cascade intermediate is `batch × stride^cascade × channels`), saving another 144 MiB sched_cu at noise-level RTF cost. TTFB unchanged because `first_batch_size=1` is independent of steady-state batch. Per-request body wins; envs `QWEN3_TTS_DEFAULT_STREAM_*` override.
- **Chunked ICL warmup** — the voice-clone warmup decode used to feed the entire `n_ref_frames`-wide ref clip through `stream_decode` in one call, building a single oversized cascade graph that pinned the scheduler arena (e.g. 96-frame ref → 478 MiB sched_cu, frozen for the lifetime of the process). The warmup now chunks at the steady-state batch size, keeping the union flat at 144 MiB on cloned-voice paths.
- **Persistent GPU streaming KV slab + flash_attn_ext** — replaces the prior per-call `ggml_concat(past, current)` rebuild with one F16 slab written via `ggml_set_rows`, lazily grown in doubling steps to the synth's `max_audio_tokens` budget. Default-budget paragraphs cap at ~64 MiB slab, max-budget at 256 MiB, vs the prior unbounded growth that pushed the CUDA pool past 4 GB on long inputs. `QWEN3_TTS_STREAM_KV_KEEP=1` forces "always keep at high-water"; default shrinks between synths when next budget ≤ ½ current.
- **Q8_0 KV cache for the talker**, paired with the FA prefill path, default ON. Saves ~64 MiB peak VRAM at no measurable RTFA cost. `QWEN3_TTS_KV_Q8=0` reverts to F16 KV.
- **Lazy talker KV growth** — initial alloc = `prefill + 1024 + 8`, doubles in 2048-frame steps capped at the synth's `max_audio_tokens + 8`. Default-budget paragraphs sit at ~60 MiB talker KV vs the prior 120 MiB always-alloc-to-cap. `QWEN3_TTS_TALKER_INITIAL_AUDIO=N` overrides.
- **Fused `GGML_OP_SNAKE` op** (CPU + CUDA, F32/F16 paths) — replaces the `pow(sin(αx), 2) / α + β·x` broadcast chain used at every vocoder residual. Removes a ~10× tensor-broadcast overhead and a `powf` NaN edge case.
- **Talker prefill K/V view narrowed** to `round_up(n_past + n_tokens, 256)` per call (vs full `n_ctx`). +8 % realtime-speedup, +13 % t/s.
- **Voice-keyed prefill ICL cache** + per-voice prefill K/V state cache — cold-from-disk TTFA <300 ms.
- **`max_audio_tokens` env-tunable** (default 2048). Higher values cost linear t/s on every synth via FA's full-`n_ctx` scan; expose so ops can pick.
- **Q8_0 vocoder mat-mul weights** — load-time quant of pre_tfm attn/FFN/proj + upsample pwconv1/2 + VQ projections. Conv weights stay F16 (wmma kernels demand it). Saves 42 MiB on weights at every budget. `QWEN3_TTS_VOCODER_NO_Q8_LOAD=1` reverts.
- **CUDA graphs disabled by default** (`GGML_CUDA_DISABLE_GRAPHS=1`) — at batch=1 autoregressive decode they cost +~700 MiB peak with no measurable gain (talker-step cgraph rebuilds every frame, capture's per-key warmup never converges). `=0` to re-enable for workloads where capture pays off.

### Functionality

- **In-process model lifecycle**: `POST /v1/admin/unload` releases all GPU/CPU buffers; `POST /v1/admin/load` reloads from the captured paths. Idle-unload via `QWEN3_TTS_IDLE_UNLOAD_SECONDS=N` triggers automatic unload after N seconds of synth inactivity, with lazy reload on the next request. Lets a server share a GPU with intermittent workloads.
- **`QWEN3_TTS_LAZY_LOAD=1`** — skip startup model load, defer until first request. Voice-archive scan still runs at startup (host-only). Useful for fast-restart prod compose.
- **C++-owned voice archive** at `$QWEN3_TTS_VOICE_ARCHIVE_DIR`. Each voice is a directory with:
  - `voice.bundle` — primary persistent format (binary `QTVB` magic): speaker embedding + ref_codes + model_id stamp. Atomically written on register; re-validated against the loaded model on startup.
  - `ref_text.txt` — optional ICL transcript.
  - `description.txt` — optional VoiceDesign description.
  - `ref.wav` — optional, kept only for human replay (never read at synth time).
  The WAV is purely the ingest format. Embeds + codes are the durable representation, so a voice survives across model swaps as long as the model_id matches.
- **`voice.warmup`** — persistent on-disk snapshot of the vocoder's ICL warm-up state (causal-conv tails, conv_t overlap buffers, per-layer KV cache) keyed by ref_codes hash. Restored on first synth per voice — eliminates the ~700–1200 ms vocoder warm-up decode that would otherwise run on every cold-from-disk request.
- **Persistent voice caches** — derived voice artefacts (codec embeddings, speaker embeddings) are cached on disk per voice and re-used across restarts.
- **Speaker + codec encoders unloaded after voice register** — they're one-shot pieces, no need to keep their VRAM resident after a voice is registered. The next register lazily reloads them.
- **Speaker-encoder-only GGUF sidecar**: `--speaker-encoder <file>` accepts a tiny GGUF that contains *only* `spk_enc.*` tensors and `qwen3-tts.speaker_encoder.*` metadata. The full Base GGUF (~2.4 GB) carries the entire talker + codec + tokenizer + speaker-encoder, but the running binary only reads the spk_enc subset. The sidecar is **~24 MB** (F16 weights + F32 biases — Q8_0 quantization in the conventional Base GGUF skips the small/odd-shaped speaker-encoder convs anyway). Pair with a VoiceDesign GGUF for a "design + clone" stack at ~3.4 GB peak resident. Extraction script: `scripts/extract_spk_enc.py`. Companion `--hf-repo-se / --hf-file-se` flags auto-download from a HuggingFace repo.
- **Voice-archive C++ ownership** — the archive scanner is in-process (`server.cpp`), not a wrapper. Bundles with `model_id` mismatch are deferred (not deleted) so a model swap doesn't lose voices.
- **`--hf-repo-v` / `--hf-repo-se`** — separate HuggingFace repos for the vocoder and speaker-encoder GGUFs (paired with `--hf-file-v` / `--hf-file-se` overrides for the file within the repo). Lets the server pull a 1.7B-VoiceDesign talker + a 24 MB spk_enc sidecar + a tokenizer in three independent fetches.
- **Input character cap raised** to 6144 (env-tunable via `QWEN3_TTS_INPUT_CHAR_CAP`).
- **C API** (`qwen3tts_c_api.h`) for bindings into non-C++ hosts.
- **`/health` endpoint** + bench-friendly timing log (`-V` / `--print-timing`) splitting prefill, decode steps, and vocoder.

### Required GGML kernels (dbrain/ggml@master, 7 commits ahead of upstream)

| Commit | Op |
|---|---|
| `f9d89c05` | `GGML_OP_SNAKE` (CPU + CUDA), fused `α·sin(βx)² + γx` for the vocoder activation |
| `a8474481` | `GGML_OP_CONV_1D_DIRECT`, smem-tiled CUDA kernel |
| `68a08cbf` | Asymmetric padding API for `conv_1d_direct` + `im2col` `gridDim.y` chunking |
| `963780ac` | Snake powf NaN fix + `conv_1d_direct` early-return sync hazard |
| `963d6660` | tensor-core `wmma` kernel for `conv_1d_direct` (RTF 1.87 → 2.61) |
| `86d9a0fc` | smem-tiled F16 wmma `conv_transpose_1d` kernel — vocoder `sched_cpu` 39.6 → 3.6 MiB, RTF 2.65 → 3.37 |
| `7d41fb0f` | F16 in/out paths through `conv_1d_direct`, `conv_transpose_1d`, `snake`, `acc` + `ggml_*_to` dst-type API variants — keeps the cascade in F16 (sched_cu 144 → 94 MiB at chunk=30) |

The submodule pointer in `.gitmodules` references this fork. From upstream `ggml-org/ggml`, both ops and the wmma kernels are non-mergeable as-is (Qwen3-TTS-specific layouts), so they live in the fork.

### A note on "RTF"

This fork (and our up-fork PyTorch reference, [faster-qwen3-tts](https://github.com/qwen-research/faster-qwen3-tts)) reports **RTF as audio-seconds-per-wall-second** (higher is better — 3.4 means we produce ~3.4 s of audio per wall-second of compute). The strict definition (wall / audio, lower is better) is the inverse. We've kept the audio-per-wall convention so numbers compare apples-to-apples with faster-qwen3-tts's bench script.

### Quant ladder on Ampere (RTX 3060) — single-stream decode

| Quant | RTF (audio/wall) | weights VRAM | Notes |
|-------|------------------|---------------|-------|
| Q8_0  | **3.40** | 2316 MiB | fastest — best-tuned MMVQ for batch=1 decode, default |
| Q4_K_M | 2.45 | 1000 MiB | only useful when VRAM-bound or on pre-Volta GPUs (Q4_K MMQ is slow on cc=86) |
| F16   | 2.20 | ~3400 MiB | slower than Q8 *and* fattest; only use if you want unquantised weights for some external reason |

Don't promote F16 or Q4_K_M to default on Ampere. Q8_0 is the sweet spot.

### TTFA / streaming latency

Time-to-first-audio is captured by the SSE streaming path (request start → first PCM event). With `streaming-first-batch-size=1` the first emit is a single 12 Hz codec frame, so once prefill + 1 talker step + 1 vocoder decode complete, PCM is on the wire — independent of total synth length.

| Path                                     | Audio length | TTFA       |
|------------------------------------------|-------------:|-----------:|
| Default voice / VoiceDesign (no ICL)     |  0.6–21 s    | **35–42 ms** |
| ICL clone, `voice.warmup` cache HIT      |  5–21 s      | **51–70 ms** |
| ICL clone, cache MISS (first synth/voice)|  5–6 s       | 384–641 ms |

Hot-path TTFA stays flat across audio length — neither doubles for a 17 s synth nor halves for a 0.6 s one — because only the first codec frame has to be ready before bytes flow. The cache-HIT row is fast because the `voice.warmup` snapshot lets every cold-from-disk request restore the vocoder ICL state in 5–7 ms instead of re-running the warmup decode.

The ICL cache-MISS path pays the warmup decode once per voice on the first synth in a fresh process, then persists the snapshot to `voice.warmup` on disk so subsequent boots restore from disk in single-digit-ms. The decode itself is chunked at the steady-state streaming batch size, so it doesn't pin the scheduler arena (see "Performance" above).

### HTTP API

OpenAI-compatible endpoints with this fork's extras layered on top. Per-request body always overrides defaults; defaults are tuned for single-stream decode on a 12 GB Ampere.

**`POST /v1/audio/speech`**

| Parameter | Origin | Notes |
|---|---|---|
| `input` | OpenAI | required; default cap 6144 chars (env `QWEN3_TTS_MAX_INPUT_CHARS`) |
| `voice` | OpenAI | `"default"`, a built-in speaker, or a registered clone id; empty = `"default"` |
| `instructions` | OpenAI | VoiceDesign description; works alongside `voice="default"` (no clone needed) |
| `response_format` | OpenAI (subset) | this fork accepts `wav` and `pcm`; OpenAI's `mp3`/`opus`/`aac`/`flac` are not implemented |
| `stream_format` | OpenAI | `audio` (chunked binary) or `sse` (`speech.audio.delta` events with base64 audio); empty = one-shot |
| `model`, `speed` | OpenAI | accepted in the body but currently ignored |
| `language` | this fork | language code (`en`, `zh`, ...); see `GET /v1/audio/languages` |
| `temperature`, `top_k`, `top_p`, `seed`, `repetition_penalty` | this fork | sampling controls |
| `max_audio_tokens` | this fork | 1..8192, default 2048; KV `n_ctx` scales linearly (see Performance for the t/s tax) |
| `stream_batch_size` | this fork | vocoder chunk size, default 30; smaller = lower per-chunk sched_cu, similar TTFA |
| `stream_first_batch_size` | this fork | first chunk only, default 1 = lowest TTFA |

Streaming mode matrix (`response_format` × `stream_format`):

| `response_format` | `stream_format` | Wire format |
|---|---|---|
| `wav` | `audio` | `audio/wav`, chunked: 44-byte placeholder header (size fields = `0xFFFFFFFF`) emitted with first PCM batch, then 16-bit-LE PCM chunks. Most players accept the placeholder; `wave.open()` rejects it — fix by rewriting bytes 4 and 40 once the stream lands. |
| `pcm` | `audio` | `audio/pcm`, chunked: raw 16-bit-LE PCM, no header. Caller must know the sample rate (24 kHz V1, 48 kHz V2). |
| `wav` \| `pcm` | `sse` | `text/event-stream`: `speech.audio.delta` events with `{type, audio: <base64>}` data, then a closing `speech.audio.done` event carrying usage/timing for both OpenAI clients and llama-swap's `metrics_monitor`. |
| any | empty | one-shot: full WAV/PCM body in a single response, no chunked transfer encoding. |

Lifecycle and discovery:

- `GET /health` → `{"status":"ok","model_loaded":bool}`
- `GET /v1/models` / `GET /v1/audio/languages` / `GET /v1/audio/voices`
- `POST /v1/audio/voices` — register a clone (multipart: `name`, `audio_sample` file, optional `ref_text` for ICL). HITs the on-disk `voice.bundle` if one exists for that name + current model and skips the encoder forward passes.
- `DELETE /v1/audio/voices/<id>` — drop the in-memory voice; on-disk `voice.bundle` / `voice.warmup` persist (re-register or restart-rescan re-loads them).
- `POST /v1/admin/unload` / `POST /v1/admin/load` — explicit lifecycle for sharing the GPU; idle unload is automatic via `QWEN3_TTS_IDLE_UNLOAD_SECONDS`.

`docker/tts-qwen3-dev/cold-start.py` exercises every path above end-to-end (model reload, voice register HIT/MISS, post-reload TTFA, VoiceDesign, all four streaming-mode combinations) and reports TTFA / wall / framing checks. Run it after a deploy to validate that the table above and the TTFA numbers above still hold on your hardware.

### Current limitations

- **CUDA-context floor (~370 MiB)**: ggml's CUDA backend reserves a per-device pool that doesn't shrink mid-process. Below this floor needs a `cudaDeviceReset()` on idle-unload, which would risk re-init safety on the ggml CUDA statics. Not currently attempted.
- **Vocoder conv weights are F16-only on CUDA**: the wmma `conv_1d_direct` and `conv_transpose_1d` kernels demand F16 weights. Quantising the vocoder beyond F16 (e.g. Q4_K_M conv weights) would drop the wmma path back to im2col + cuBLAS and tank RTF. Per-channel snake α/β stay F32.
- **Speaker-encoder weights are F16-only**: the ECAPA-TDNN encoder dispatches conv layers through the F16-only `conv_1d_direct` path. F32 weights for the speaker encoder aren't supported, but in practice this isn't a quality limitation — the upstream BF16 weights downcast bit-identically to F16 for ECAPA-TDNN-class encoders, and the bias-add op already requires F32 biases (which is what GGUF stores them as).

---

## Added In Khimaros' Fork

The [khimaros/qwen3-tts.cpp](https://github.com/khimaros/qwen3-tts.cpp) fork layered the following on top of [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp):

- **1.7B model support** with MTP projection bridging the 2048-dim talker and 1024-dim code predictor, plus dynamic model detection
- **ICL voice cloning** via Mimi codec encoder — reference audio is encoded to discrete speech codes and combined with `ref_text` in prefill, as an alternative to x-vector speaker embeddings
- **Voice steering** via `--instructions` flag and server API (1.7B+ only)
- **Multi-language synthesis** via `-l/--language` (en, zh, ja, ko, ru, de, fr, es, it, pt)
- **Proper UTF-8 tokenization** via GPT-2 regex pre-tokenization (fixes tokenization of non-ASCII text)
- **WAVE_FORMAT_EXTENSIBLE** WAV header support (e.g. macOS screen recordings)
- **GPU-safe vocoder codebook normalization** (unbreaks Vulkan backend)
- **ICL encoder parity fix**: correct conv-layer bias loading (a subtle `sscanf` partial-match bug silently dropped input-conv and resunit biases), raising encoder/Python cosine similarity from ~0.989 to ~0.99999 per stage and eliminating the ~350ms start-of-audio noise in cloned voices
- **Performance optimizations**: flash attention for decode steps, static KV cache with `ggml_set_rows`, cached vocoder decoder graph
- **OpenAI-compatible HTTP server** (`qwen3-tts-server`) with `/v1/audio/speech` and `/v1/audio/voices` endpoints, voice cloning via multipart upload, and `--hf-repo` for auto-downloading GGUFs from the [Qwen3-TTS collection](https://huggingface.co/collections/khimaros/qwen3-tts)
- **Streaming audio output**: chunked vocoder decode with causal tails, per-layer KV rolling, and conv-transpose overlap state, so PCM is emitted frame-by-frame while the transformer is still generating. Exposed via the CLI `--streaming-batch-size N` flag and wired through the server for live HTTP responses; parity-tested against one-shot decode
- **Real token accounting** in `tts_result` (text / prefill / audio tokens, plus prefill time broken out of total generate time) for accurate OpenAI `usage` reporting
- **Multi-variant model support** (Base / CustomVoice / VoiceDesign) with speaker presets and language IDs stored in GGUF metadata
- **Batch model conversion** script that downloads and converts all Qwen3-TTS variants in one shot

### HuggingFace Models

Pre-converted GGUF artifacts are published in the [Qwen3-TTS collection](https://huggingface.co/collections/khimaros/qwen3-tts):

| Repo | Variant |
|------|---------|
| [`khimaros/Qwen3-TTS-12Hz-0.6B-Base-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-12Hz-0.6B-Base-GGUF) | 0.6B, ICL voice clone |
| [`khimaros/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF) | 0.6B, speaker presets |
| [`khimaros/Qwen3-TTS-12Hz-1.7B-Base-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-12Hz-1.7B-Base-GGUF) | 1.7B, ICL voice clone |
| [`khimaros/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF) | 1.7B, speaker presets |
| [`khimaros/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF) | 1.7B, `--instructions` steering |
| [`khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF) | shared vocoder |

Each repo ships F16 and Q8_0 quants. The server auto-downloads and caches them via `--hf-repo`:

```bash
./build/qwen3-tts-server \
    --hf-repo khimaros/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF:Q8_0 \
    --hf-repo-v khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF:F16
```

`--hf-repo <repo>[:<quant>]` defaults to `Q8_0`; override the exact GGUF filename with `--hf-file`.

The rest of this README is the original from upstream.

---

![PyTorch vs qwen3-tts.cpp benchmark](./docs/benchmark_pytorch_vs_cpp.png)

**Benchmark Snapshot (PyTorch vs qwen3-tts.cpp):** Basic 3.19x faster, Clone 4.07x faster. Peak RSS delta: Basic +19.0%, Clone +7.7%.

C++ inference for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) using the [GGML](https://github.com/ggml-org/ggml) tensor library.

Runs the full TTS pipeline in pure C++17, including text tokenization, speaker encoding, transformer code generation, and vocoder decoding, without Python or PyTorch at inference time.

## Features

- Full text-to-speech pipeline in C++17 with GGML backend
- Voice cloning from reference audio (ECAPA-TDNN x-vector extraction)
- Greedy and sampled decoding (temperature, top-k, repetition penalty)
- GGUF model format (F16 and Q8_0 quantization)
- Runtime backend selection with GPU/Metal preference and CPU fallback
- Deterministic reference tests comparing C++ output against Python
- Compile-time timing instrumentation with zero overhead in normal builds

## Prerequisites

- C++17 compiler (GCC 9+ or Clang 10+)
- CMake 3.14+
- [GGML](https://github.com/ggml-org/ggml) built from source
- Python 3.10+ with [uv](https://github.com/astral-sh/uv) (model conversion only)

## Quickstart (macOS, copy/paste)

Run these commands from a fresh clone:

```bash
git clone https://github.com/predict-woo/qwen3-tts.cpp.git
cd qwen3-tts.cpp
git submodule update --init --recursive

# 1) Build GGML with Metal
cmake -S ggml -B ggml/build -DGGML_METAL=ON
cmake --build ggml/build -j4

# 2) Build qwen3-tts.cpp
cmake -S . -B build
cmake --build build -j4

# 3) Create a uv Python environment for setup/conversion tools
uv venv .venv
source .venv/bin/activate

# 4) Install Python dependencies
uv pip install --upgrade pip
uv pip install huggingface_hub gguf torch safetensors numpy tqdm coremltools

# Optional if model access requires auth:
# hf auth login

# 5) Download and generate all runtime model artifacts
python scripts/setup_pipeline_models.py

# 6) Basic synthesis example
./build/qwen3-tts-cli \
  -m models \
  -t "Hello from qwen3-tts.cpp running on macOS with CoreML by default." \
  -o examples/readme_example_basic.wav

# 7) Voice-clone example using sample audio in this repo
./build/qwen3-tts-cli \
  -m models \
  -r examples/readme_clone_input.wav \
  -t "This is a voice cloning example generated from the sample audio file in this directory." \
  -o examples/readme_example_clone.wav
```

Expected model artifacts after step 5:

- `models/qwen3-tts-0.6b-f16.gguf`
- `models/qwen3-tts-tokenizer-f16.gguf`
- `models/coreml/code_predictor.mlpackage` (on macOS)

Expected audio outputs after steps 6-7:

- `examples/readme_example_basic.wav`
- `examples/readme_example_clone.wav`

Included voice-clone input/output pair (so users can compare directly):

- Input reference audio: `examples/readme_clone_input.wav`
- Generated output audio: `examples/readme_example_clone.wav`

Audio preview (inline):

<audio controls src="./examples/readme_clone_input.wav"></audio>
<br/>
<audio controls src="./examples/readme_example_clone.wav"></audio>

If your Markdown renderer does not show inline controls, use direct links:

- [Play input reference WAV](./examples/readme_clone_input.wav)
- [Play generated output WAV](./examples/readme_example_clone.wav)

## Build

```bash
git clone https://github.com/predict-woo/qwen3-tts.cpp.git
cd qwen3-tts.cpp
git submodule update --init --recursive

# Build GGML (vendored in ./ggml)
cmake -S ggml -B ggml/build -DGGML_METAL=ON
cmake --build ggml/build -j4

# Build qwen3-tts.cpp
cmake -S . -B build
cmake --build build -j4
```

> **Note:** The top-level CMake currently expects GGML in `./ggml` with libraries under `./ggml/build/src`.

## Model Setup (Recommended)

Use the one-shot setup script:

```bash
source .venv/bin/activate
python scripts/setup_pipeline_models.py
```

Useful flags:

- `--force` re-downloads and re-generates all artifacts.
- `--coreml auto|on|off` controls CoreML export behavior.
- `--skip-download` skips HF download and uses existing local model dirs.

## Manual Model Conversion (Advanced)

Convert HuggingFace models to GGUF format:

```bash
# Download the model
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --local-dir models/Qwen3-TTS-12Hz-0.6B-Base

# Convert TTS model (transformer + speaker encoder + tokenizer)
python scripts/convert_tts_to_gguf.py \
    models/Qwen3-TTS-12Hz-0.6B-Base \
    models/qwen3-tts-0.6b-f16.gguf

# Convert vocoder (audio decoder)
python scripts/convert_tokenizer_to_gguf.py \
    models/Qwen3-TTS-12Hz-0.6B-Base \
    models/qwen3-tts-tokenizer-f16.gguf
```

Place both `.gguf` files in a `models/` directory.

## Usage

```bash
# Basic synthesis
./build/qwen3-tts-cli -m models -t "Hello, world!" -o hello.wav

# Voice cloning from reference audio
./build/qwen3-tts-cli -m models -t "Hello! How are you?" -r reference.wav -o cloned.wav

# Greedy decoding with max length
./build/qwen3-tts-cli -m models -t "Hello!" -r ref.wav -o out.wav \
    --temperature 0 --max-tokens 2048
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model <dir>` | Model directory containing GGUF files | (required) |
| `-t, --text <text>` | Text to synthesize | (required) |
| `-o, --output <file>` | Output WAV file path | `output.wav` |
| `-r, --reference <file>` | Reference audio for voice cloning | (none) |
| `--temperature <val>` | Sampling temperature (0 = greedy) | 0.9 |
| `--top-k <n>` | Top-k sampling (0 = disabled) | 50 |
| `--top-p <val>` | Top-p sampling | 1.0 |
| `--max-tokens <n>` | Maximum audio frames to generate | 4096 |
| `--repetition-penalty <val>` | Repetition penalty on codebook-0 token generation | 1.05 |
| `-j, --threads <n>` | Number of compute threads | 4 |

`--top-p` is currently parsed by the CLI but not yet wired into transformer sampling.

On macOS, CoreML code predictor is enabled by default when `models/coreml/code_predictor.mlpackage` exists.
Set `QWEN3_TTS_USE_COREML=0` to disable it. Low-memory mode is opt-in via `QWEN3_TTS_LOW_MEM=1`.

### Backend Selection

At runtime, each component logs its selected backend (for example, `TTSTransformer backend: MTL0` or `BLAS`).

- Preferred order: `IGPU` -> `GPU` -> `ACCEL` -> `CPU`
- Encoder and transformer can run on Metal/other accelerators with CPU fallback in the scheduler
- Decoder now follows the same backend preference and will use Metal when available

## Architecture

```
Text ──► [Tokenizer] ──► token IDs
                              │
Reference Audio ──► [Speaker Encoder] ──► speaker embedding
                              │
token IDs + speaker embedding ──► [TTS Transformer] ──► speech codes (N frames x 16 codebooks)
                              │
speech codes ──► [Vocoder] ──► audio waveform (24kHz)
```

### Source Files

| File | Component | Description |
|------|-----------|-------------|
| `text_tokenizer.{h,cpp}` | Tokenizer | BPE text tokenizer from GGUF |
| `audio_tokenizer_encoder.{h,cpp}` | Speaker Encoder | ECAPA-TDNN x-vector extractor |
| `tts_transformer.{h,cpp}` | TTS Transformer | 28-layer Qwen2 talker + 5-layer code predictor |
| `audio_tokenizer_decoder.{h,cpp}` | Vocoder | WavTokenizer decoder (codes to waveform) |
| `qwen3_tts.{h,cpp}` | Pipeline | Full pipeline orchestration |
| `main.cpp` | CLI | Command-line interface |
| `gguf_loader.{h,cpp}` | GGUF | Model loading utilities |

### TTS Transformer Details

The transformer generates speech codes in two stages per frame:

1. **Talker** (28 layers, 16 heads, 1024 hidden) produces a hidden state and codebook-0 logits.
2. **Code Predictor** (5 layers) autoregressively generates codebooks 1-15 from that hidden state.

The prefill embedding mirrors the Python pipeline exactly:
- Positions 0-2: text-projected role tokens (`<|im_start|>`, `assistant`, `\n`)
- Positions 3-6: TTS pad + codec embeddings (think tokens, language ID)
- Position 7: TTS pad + speaker embedding
- Position 8: TTS BOS + codec pad embedding
- Position 9+: text-projected text tokens + codec BOS/embeddings

## Testing

```bash
# Run full test suite
bash scripts/run_all_tests.sh

# Individual component tests
./build/test_tokenizer --model models/qwen3-tts-0.6b-f16.gguf
./build/test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf \
    --audio clone.wav --reference reference/ref_audio_embedding.bin
./build/test_transformer --model models/qwen3-tts-0.6b-f16.gguf \
    --ref-dir reference/
./build/test_decoder --tokenizer models/qwen3-tts-tokenizer-f16.gguf \
    --codes reference/speech_codes.bin --reference reference/decoded_audio.bin

# End-to-end Python vs C++ comparison
uv run python scripts/compare_e2e.py

# Generate deterministic reference data from Python
uv run python scripts/generate_deterministic_reference.py
```

### Test Results (F16 model)

- Prefill logits: cosine similarity = 0.99999994 with Python reference
- Codebook 0 match rate: 81% (frame-level exact match)
- Codebooks 1-4: ~84% match rate
- Audio output is perceptually equivalent; low waveform correlation is expected due to autoregressive divergence from F16 precision

## Profiling

Build with compile-time timing instrumentation (zero overhead when disabled):

```bash
cmake .. -DQWEN3_TTS_TIMING=ON
make -j4
```

Example output (92 frames, 7.3s audio):

```
=== Detailed Generation Timing (92 frames) ===

  Prefill:
      Compute:           175.9 ms

  Talker forward_step:
      Graph build:        21.8 ms   (0.2 ms/frame)
      Graph alloc:        34.1 ms   (0.4 ms/frame)
      Compute:          7717.4 ms   (83.9 ms/frame)

  Code predictor:
      Init/KV/embed:       7.7 ms   (0.1 ms/frame)
      Prefill (2tok):   1393.2 ms   (15.1 ms/frame)
      Steps (14):      19531.7 ms   (212.3 ms/frame)
      Compute:         20702.6 ms   (225.0 ms/frame)

  Total generate:      28915.0 ms   (3.2 frames/s)
```

The code predictor (15 sequential forward passes per frame) accounts for ~71% of generation time.

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) by Alibaba Qwen team
- [GGML](https://github.com/ggml-org/ggml) tensor library by Georgi Gerganov
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) vocoder architecture
