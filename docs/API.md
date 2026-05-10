# HTTP API reference

OpenAI-compatible endpoints with this fork's extras layered on top. Per-request body always overrides defaults; defaults are tuned for single-stream decode on a 12 GB Ampere.

## `POST /v1/audio/speech`

Main TTS endpoint. Accepts JSON body, returns audio (one-shot) or a streaming response.

### Body

| Parameter | Origin | Notes |
|---|---|---|
| `input` | OpenAI | required; default cap 6144 chars (env `QWEN3_TTS_MAX_INPUT_CHARS`) |
| `voice` | OpenAI | `"default"`, a built-in speaker, or a registered clone id; empty = `"default"` |
| `instructions` | OpenAI | VoiceDesign description; works alongside `voice="default"` (no clone needed) |
| `response_format` | OpenAI (subset) | `wav`, `pcm`, `mp3`, `ogg` (Opus-in-Ogg), `aac` (ADTS). FLAC is not implemented. Compressed formats (mp3/ogg/aac) honour `bitrate_kbps` (32–192, default 64). |
| `stream_format` | OpenAI | `audio` (chunked binary) or `sse` (`speech.audio.delta` events with base64 audio); empty = one-shot |
| `model`, `speed` | OpenAI | accepted in the body but currently ignored |
| `language` | this fork | language code (`en`, `zh`, ...); see `GET /v1/audio/languages` |
| `temperature`, `top_k`, `top_p`, `seed`, `repetition_penalty` | this fork | sampling controls |
| `max_audio_tokens` | this fork | 1..8192, default 2048; KV `n_ctx` scales linearly (linear t/s tax) |
| `stream_batch_size` | this fork | vocoder chunk size, default 30; smaller = lower per-chunk sched_cu, similar TTFA |
| `stream_first_batch_size` | this fork | first chunk only, default 1 = lowest TTFA |

### Streaming-mode matrix (`response_format` × `stream_format`)

| `response_format` | `stream_format` | Wire format |
|---|---|---|
| `wav` | `audio` | `audio/wav`, chunked: 44-byte placeholder header (size fields = `0xFFFFFFFF`) emitted with first PCM batch, then 16-bit-LE PCM chunks. Most players accept the placeholder; `wave.open()` rejects it — fix by rewriting bytes 4 and 40 once the stream lands. |
| `pcm` | `audio` | `audio/pcm`, chunked: raw 16-bit-LE PCM, no header. Caller must know the sample rate (24 kHz V1, 48 kHz V2). |
| `mp3` | `audio` | `audio/mpeg`, chunked: self-syncing MP3 frames; no leading header needed. |
| `ogg` | `audio` | `audio/ogg`, chunked: standalone Ogg Opus pages. |
| `aac` | `audio` | `audio/aac`, chunked: ADTS-framed AAC. Default `stream_first_batch_size` for AAC is 4 (encoder buffers ≥1 frame internally; smaller values gain little TTFA). |
| `wav` \| `pcm` | `sse` | `text/event-stream`: `speech.audio.delta` events with `{type, audio: <base64>}` data, then a closing `speech.audio.done` event carrying usage/timing. |
| any | empty | one-shot: full body in a single response, no chunked transfer encoding. |

### TTFA-accurate Python clients

Python's `urllib` `HTTPResponse` uses an internal buffered reader. `read()` and `readline()` block waiting for the buffer to fill, which inflates TTFA by hundreds of ms over a streaming SSE response. **Use `read1(N)`** — returns whatever bytes are currently available without trying to top up the buffer — and parse SSE events from the raw stream by hand:

```python
read_chunk = getattr(resp, "read1", None) or (lambda n: resp.read(n))
buf = b""
while chunk := read_chunk(4096):
    buf += chunk
    while b"\n\n" in buf:
        evt_bytes, buf = buf.split(b"\n\n", 1)
        # parse "event: ..." and "data: ..." lines, decode base64 PCM
```

Reference implementation: [`docker/tts-qwen3-dev/breakit_seeds.py`](#) `post_speech_pcm()`. `requests` and `aiohttp` have analogous gotchas — verify your client doesn't line-buffer the SSE stream before trusting any TTFA measurement.

### Hot-path TTFA characteristics

With `stream_first_batch_size=1` the first emit is a single 12 Hz codec frame, so once prefill + 1 talker step + 1 vocoder decode complete, PCM is on the wire — independent of total synth length.

| Path | Audio length | TTFA |
|---|---:|---:|
| Default voice / VoiceDesign (no ICL) | 0.6–21 s | **35–47 ms** |
| ICL clone, `voice.warmup` cache HIT | 5–21 s | **51–70 ms** |
| ICL clone, cache MISS (first synth/voice) | 5–6 s | 384–641 ms |

Hot-path TTFA stays flat across audio length — neither doubles for a 17 s synth nor halves for a 0.6 s one — because only the first codec frame has to be ready before bytes flow.

The cache-MISS path pays the warmup decode once per voice on the first synth in a fresh process, then persists the snapshot to `voice.warmup` on disk so subsequent boots restore from disk in single-digit-ms.

## `GET /v1/audio/voices`

List available voices. Returns built-in speakers + every registered clone.

## `POST /v1/audio/voices`

Register a voice clone. Multipart:

| Field | Required | Notes |
|---|---|---|
| `name` | yes | the voice id; must pass `is_safe_voice_name` (alnum, dash, underscore) |
| `audio_sample` | yes | reference WAV (mono int16, any rate the speaker encoder accepts) |
| `ref_text` | optional | ICL transcript matching the audio; enables prompt-conditioned voice cloning |

On success: `voice.bundle` written to `$QWEN3_TTS_VOICE_ARCHIVE_DIR/<name>/`. Re-registering the same name updates the bundle in place. Speaker + codec encoders auto-unload after register (one-shot pieces, no need to keep VRAM resident).

If a `voice.bundle` already exists for this name + current model, the encoder forward passes are skipped — you can re-register a clone in <50 ms.

## `DELETE /v1/audio/voices/{name}`

Remove the voice from the in-memory map. **Disk artifacts persist** — `voice.bundle` and `voice.warmup` stay in the archive volume so a re-register or a process restart picks them back up. To truly purge, also `rm -rf` the directory under `$QWEN3_TTS_VOICE_ARCHIVE_DIR/<name>/`.

## Lifecycle endpoints

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | `{"status":"ok","model_loaded":bool}` |
| `GET` | `/v1/models` | OpenAI-compat models list |
| `GET` | `/v1/audio/languages` | language codes the model supports |
| `POST` | `/v1/admin/load` | re-materialize the model from captured paths (no-op if already loaded). In worker-isolation mode this respawns the worker subprocess. |
| `POST` | `/v1/admin/unload` | release all GPU/CPU buffers. **Single-process mode**: frees model weights + KV slabs but the ~370 MiB CUDA-context floor stays resident until process exit. **Worker-isolation mode** (`QWEN3_TTS_WORKER_ISOLATION=1`): SIGKILL + waitpid the worker subprocess; the response only returns once the kernel has reaped it, so VRAM is fully reclaimed by the time the HTTP response sends (~80–100 ms wall). |

`POST /v1/admin/{load,unload}` give you explicit control over GPU lifecycle, useful for sharing the GPU with intermittent peers. Otherwise idle-unload via `QWEN3_TTS_IDLE_UNLOAD_SECONDS` does it automatically.

## Voice archive layout

`$QWEN3_TTS_VOICE_ARCHIVE_DIR/<name>/` per voice:

| File | Required | Notes |
|---|---|---|
| `voice.bundle` | required | binary `QTVB` magic; speaker embedding + ref_codes + model_id stamp. Atomically written on register; re-validated against the loaded model on startup. |
| `ref_text.txt` | optional | ICL transcript |
| `description.txt` | optional | VoiceDesign description |
| `ref.wav` | optional | kept only for human replay; never read at synth time |
| `voice.warmup` | optional | persistent vocoder ICL warm-up snapshot (causal-conv tails, conv_t overlap buffers, per-layer KV cache) keyed by ref_codes hash. Written after the first synth per voice; restored on subsequent cold-from-disk synths in 5–7 ms instead of running the warmup decode (which would be 700–1200 ms). |

Voice survives across model swaps as long as `model_id` matches. Mismatched bundles are deferred (not deleted) so a model swap doesn't lose voices — when the original model returns, the bundle is picked back up.

## Environment variables

### Lifecycle

| Var | Default | Notes |
|---|---|---|
| `QWEN3_TTS_LAZY_LOAD` | unset | `=1` skip startup model load, defer to first request. Voice-archive scan still runs at startup. |
| `QWEN3_TTS_IDLE_UNLOAD_SECONDS` | unset (off in prod compose; 30 in dev) | release GPU after N seconds of inactivity, lazy-reload on next request |
| `QWEN3_TTS_WORKER_ISOLATION` | unset | `=1` runs the GPU work in a forked subprocess so `/v1/admin/unload` and the idle-unload watchdog can SIGKILL it and reclaim **all** VRAM (no CUDA-context floor residue). Per-call IPC overhead is sub-noise (~16 µs p99 per AUDIO_FRAME). See [ARCHITECTURE.md#worker-isolation](ARCHITECTURE.md#worker-isolation) for the protocol + cycle-test. |
| `QWEN3_TTS_VOICE_ARCHIVE_DIR` | `./voice-archive` | where bundles + warmup snapshots persist |

### Quality / sizing

| Var | Default | Notes |
|---|---|---|
| `QWEN3_TTS_KV_Q8` | `1` | Q8_0 KV cache for the talker, ~64 MiB peak savings, no measurable RTFA cost. `=0` reverts to F16 KV. |
| `QWEN3_TTS_TALKER_INITIAL_AUDIO` | `1024` | initial talker KV alloc (frames) |
| `QWEN3_TTS_STREAM_KV_KEEP` | unset | `=1` keep streaming KV slab at high-water across synths; default shrinks if next budget ≤ ½ current |
| `QWEN3_TTS_STREAM_KV_MAX_NPAST` | (model max) | cap on streaming KV slab growth |
| `QWEN3_TTS_MAX_INPUT_CHARS` | `6144` | input text cap |
| `QWEN3_TTS_DEFAULT_STREAM_BATCH_SIZE` | `30` | default vocoder chunk size |
| `QWEN3_TTS_DEFAULT_STREAM_FIRST_BATCH_SIZE` | `1` | first chunk is one codec frame (lowest TTFA) |
| `QWEN3_TTS_VOCODER_FP16_CASCADE` | `1` | F16 cascade activations; `=0` reverts to F32 (heavier sched_cu) |
| `QWEN3_TTS_VOCODER_NO_Q8_LOAD` | unset | `=1` keeps vocoder mat-mul weights F16 instead of Q8_0 |

### Performance opt-outs (debug only — defaults are correct)

| Var | Effect |
|---|---|
| `QWEN3_TTS_NO_FULL_AR_CACHE=1` | disable v9.4 full-AR cgraph cache |
| `QWEN3_TTS_NO_TALKER_GPU_SAMPLE=1` | disable v9.5 GPU-side talker sampling |
| `QWEN3_TTS_NO_TALKER_STEP_CACHE=1` | disable v9.7 talker step cgraph cache |
| `QWEN3_TTS_NO_FULL_AR_GRAPH=1` | disable v7-phase2 full-AR cgraph (per-step fallback) |
| `QWEN3_TTS_NO_ASYNC_VOCODER=1` | disable v9.2 async vocoder dispatch |
| `QWEN3_TTS_TALKER_PRIORITY=default` | disable v9.3 talker = HIGH stream priority (also `low` / `high` overrides) |
| `QWEN3_TTS_VOCODER_PRIORITY=default` | disable v9.3 vocoder = LOW (no-op on RTX 3060) |
| `QWEN3_TTS_SPECIALIZED_MMVQ=0` | disable shape-specialized Q8 MMVQ kernels (fall through to ggml stock) |
| `GGML_CUDA_DISABLE_GRAPHS=1` | disable CUDA-graph capture |

Set to default values in production — the opt-outs exist for diagnostic comparison, not steady-state operation.

### Boot-time CUDA arch trim

Build-arg `QWEN3_TTS_CUDA_ARCHS` controls which arches are baked into the image. Default `75;80;86;89;90` (Turing → Hopper); set to `86` (or whichever arch you target) to drop the per-arch ~15–20 MiB CUDA-context floor on a single-arch host.

## Companion tools

`docker/tts-qwen3-dev/cold-start.py` exercises every path above end-to-end (model reload, voice register HIT/MISS, post-reload TTFA, VoiceDesign, all four streaming-mode combinations) and reports TTFA / wall / framing checks. Run it after a deploy to validate.
