# Microphone Backend Implementation

Date: 2026-11-05. Scope: independent browser push-to-talk transcription backend.

## Changed Paths

- `src/streammuse/infrastructure/rap/utterance_input.py`
- `src/streammuse/presentation/rap_demo/microphone_api.py`
- `tests/unit/infrastructure/rap/test_utterance_input.py`
- `tests/unit/presentation/rap_demo/test_microphone_api.py`
- `docs/reports/2026-11-05-microphone-implementation.md`

No edits to server.py, cli.py, static assets, dependencies, audio/stress policy,
or pre-existing dirty files. No commits, service starts, GPU jobs, microphone
recording, or model downloads were performed by this implementation task.

## Integration

Parent owns server mounting and browser capture:

```python
from streammuse.presentation.rap_demo.microphone_api import create_microphone_router

app.include_router(create_microphone_router())
```

Mount once per FastAPI process. Construction/status checks do not import Whisper
or load its model. The first non-silent utterance lazily loads one persistent
`base.en` CPU/int8 model with two CPU threads and one inference worker.
`STREAMMUSE_ASR_MODEL=small.en` selects the larger English model; `base` and
`small` aliases are also accepted. Arbitrary model paths/IDs are not accepted.
`STREAMMUSE_MICROPHONE_ENABLED=0` disables transcription gracefully.

The parent installed faster-whisper 1.2.1 and provisioned the base.en cache in
the existing `.venv`; no repository dependency declaration was changed. For a
fresh environment, the optional environment-only installation is:

```sh
uv pip install --python .venv/bin/python 'faster-whisper>=1.1,<2'
```

Model provisioning belongs to the parent/deployment. The constructor uses
`local_files_only=True`; a missing/unloadable model produces a sanitized 503 and
an unavailable status. Restart/recreate the router after repairing an initial
model-load failure. A warm-up in another process populates the cache, but does
not eliminate model initialization in the running server process.

## HTTP Contract

`GET /api/microphone` returns `available`, `model`, `state`,
`transcriber_state`, `reason`, `busy`, input limits, and (with the built-in
transcriber) `device`, `compute_type`, and `language`. States are `disabled`,
`unavailable`, `unloaded`, `loading`, `ready`, and the API overlay `busy`.
`available=true` in `unloaded` means the dependency exists, not that the cached
model has already been verified. GET never triggers model initialization.

`POST /api/microphone/transcribe` takes the raw WAV body, not JSON, multipart,
or a file path. Content-Type is `audio/wav` (`audio/wave` and `audio/x-wav` are
also accepted). Browser capture must produce RIFF/WAVE, uncompressed PCM16,
one channel, 8,000-48,000 Hz, nonempty and at most 15 seconds. Maximum upload
size is 1,444,096 bytes, including headers/metadata. Actual streamed bytes are
bounded even when Content-Length is absent or misleading.

Send a fresh UUID in `X-Utterance-ID` for each push-to-talk release and reuse
that UUID only for retries of identical WAV bytes. Header syntax is 1-128 ASCII
letters/digits or `_.:-`. Without the header, a UUID is generated server-side.

Example response:

```json
{
  "event_id": "browser-utterance-uuid",
  "text": "Raindrops tap on glassy rooftops at sunrise.",
  "transcription_ms": 416.96,
  "duration_seconds": 2.72,
  "sample_rate": 24000,
  "silence": false,
  "model": "base.en"
}
```

The response is immediately usable as the visible transcript, with no
confirmation endpoint or pending-approval state. This API never mutates story,
planner, lyric, or playback state. Optional automatic story routing belongs to
the parent/UI and should deduplicate by `event_id`. Silence returns HTTP 200,
`text=""`, and `silence=true`; the UI must not submit empty story updates.

Error bodies use FastAPI's `detail` object, with `code` and usually `event_id`:

| Status | Condition |
| --- | --- |
| 400 | Malformed WAV, invalid ID/length, unsupported channels/rate/PCM format |
| 408 | Upload exceeded its 10-second deadline |
| 409 | Same ID is in progress, or a cached ID is reused with different bytes |
| 413 | Byte limit or 15-second duration exceeded |
| 415 | Content-Type is not an accepted WAV media type |
| 429 | Another utterance occupies the one processing slot |
| 500 | Sanitized inference failure |
| 503 | Disabled or unavailable dependency/model |
| 504 | Recognition exceeded its 30-second response deadline |

Busy/in-progress/recognition-timeout responses include `Retry-After: 1`.
The router factory accepts injected `transcriber`, `transcription_timeout_s`,
`upload_timeout_s`, `cache_size`, and `cache_ttl_s` for integration/tests.

## Concurrency, Retry, and Privacy

- One slot covers both upload and recognition; parallel work is rejected, not queued.
- Recognition runs in an executor thread, leaving FastAPI responsive.
- Timeout or request cancellation does not free the slot while native CPU work
  is still running. Its completed result remains available by retrying the same ID.
- Completed responses, including inference errors, are cached for five minutes,
  bounded to 64 entries. Same ID plus same body replays the exact response;
  same ID plus different body returns 409. Retrying with a different ID is new work.
- Cache entries retain a SHA-256 digest and response, not uploaded audio. The
  backend does not persist transcripts/audio or create temporary audio files.
  File-like input is closed after success or failure, including lazy segment
  iteration. Cache memory expires lazily on subsequent requests.
- Quiet PCM (normalized RMS below 0.0001) is returned as silence without loading
  Whisper; Whisper VAD additionally filters nonspeech.

## Verification

Tests were written before implementation; the initial run reported the missing
modules. The first implementation run passed 43 tests and exposed one inaccurate
test fixture: 0.25 seconds cannot have an integral frame count at 22,050 Hz.
Changing that fixture to exactly representable 0.5 seconds resolved it.

Focused command (53 passed):

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/unit/infrastructure/rap/test_utterance_input.py \
  tests/unit/presentation/rap_demo/test_microphone_api.py -q
```

Focused plus adjacent regression command (145 passed):

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/unit/infrastructure/rap/test_utterance_input.py \
  tests/unit/presentation/rap_demo/test_microphone_api.py \
  tests/unit/presentation/rap_demo/test_server.py \
  tests/unit/experiments/rap_audio_protocols/test_evaluation.py -q
```

Coverage includes inclusive duration/rate boundaries, malformed/truncated RIFF,
oversized declared and streamed bodies, stereo/non-PCM inputs, no file-path
input, silence, missing dependency/disabled state, lazy persistent CPU loading,
no temporary files and closed streams on success/failure, duplicate replay and
conflict, bounded/expiring cache, nonblocking status, busy rejection, upload
timeout, inference timeout, upload cancellation, and inference cancellation.

Real offline CPU smoke test used the existing generated file
`output/rap_alignment_20261028/glass_rooftops/glass_rooftops_bar_00_source.wav`
(2.72 seconds, mono PCM16, 24 kHz), with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. Both calls returned:

> Raindrops tap on glassy rooftops at sunrise.

Measured transcriber times: first call 2293.88 ms including lazy model load;
second call 416.96 ms with the same persistent model. This is a small smoke
test, not a latency or accuracy benchmark. `git diff --check` passed.

A second real offline smoke test mounted the router in FastAPI TestClient and
posted the same WAV twice with `X-Utterance-ID: offline-smoke-20261105`. Both
responses were HTTP 200 and exactly identical, including event ID and measured
transcription time (909.29 ms). GET then reported `state=ready` and `busy=false`.

## Known Limits

- Parent must mount the router and implement capture/release/transcript display.
  Browser end-to-end acceptance and automatic story submission are parent work.
- Single-process idempotency only; cache expiry/eviction/restart permits the
  same key to be processed again. Multiple server workers do not share a cache.
- A Python thread cannot safely terminate a stuck native inference call. A 504
  bounds HTTP waiting, not CPU execution; a stuck worker keeps the backend busy.
- Model-load failures remain unavailable until the transcriber/router is recreated.
- English-only bounded utterances, no full-duplex capture, streaming ASR, echo
  cancellation, speaker isolation, or voice cloning. Playback audio leaking into
  browser capture can be transcribed; no audio policy was changed to address it.
- The complete repository suite was not run; verification was focused and used
  the existing venv with `PYTHONPATH=src` to avoid its obsolete editable path.
