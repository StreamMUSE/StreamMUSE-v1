# Task 4 Implementation Report

## Delivered

Implemented the private two-bar rap render service and idempotent artifact
cache in `streammuse.presentation.rap_render_server`. The dependency-injected
FastAPI app exposes `GET /health` and `POST /v1/rap/chunks/render`; successful
responses are encoded only by Task 1's `encode_chunk_package()` and use
`application/vnd.streammuse.rap-chunk+zip`.

Fix round 1 adds bounded request streaming, lock-minimal idempotency,
failure-safe coalescing, exact MMS/source artifact retention, durable atomic
renames, finalized packaging timing, strict health allowlists, and lazy real
H200 composition with lifecycle ownership.

The fresh re-review correction packages the shared MOSS backend for installed
entrypoints, rolls back a failed final cache marker even after rename, makes
the production shared workspace durable, bounds health scalar values, and
accounts for the complete two-pass publication sequence in final timing.

## RED/GREEN Evidence

All RED failures below were observed before the corresponding production edit.

1. Request boundary and upload limit:
   - RED: the parser test reached a poisoned `from_dict`, oversized
     `Content-Length` returned 200, and streamed overflow returned 422
     (`3 failed`).
   - GREEN: direct `from_payload()` parsing and pre-buffer byte accounting
     passed all three cases (`3 passed`).
2. Idempotency lock and failed-owner completion:
   - RED: a blocked cache read for one ID held the global lock, and a failed
     diagnostic write replaced the owner's typed error while its waiter timed
     out (`2 failed`).
   - GREEN: unrelated cache I/O completed concurrently, and owner/waiter both
     received the original `PhraseRenderFailed` despite diagnostic I/O failure
     (`2 passed`).
3. Artifact durability and timing:
   - RED: the full MMS record was not retained separately and atomic replace
     did not fsync the containing directory (`2 failed`).
   - GREEN: exact full-record bytes are retained as `mms_alignment.json`,
     bounded manifest diagnostics are written to `alignment.json`, directory
     fsync follows replace, and decoded package timing is nonzero/final.
4. Health projection:
   - RED: compound and nested secret/path fields escaped the denylist. An
     intermediate implementation also emitted an absent `warmup` as null. A
     final sparse-health test showed absent top-level fields overwrote required
     protocol/schema defaults with null.
   - GREEN: the exact recursively allowlisted health payload passed, and the
     sentinel secret did not occur anywhere in the response. Presence checks
     retain required defaults without emitting absent optional summaries
     (`3 passed`).
5. Production composition and lifecycle:
   - RED: `main()` could not consume an owned composition object and the
     default production factory unconditionally raised.
   - GREEN: injected CLI lifecycle, shutdown-on-server-error, and complete
     resident component construction/warmup/close ownership passed
     (`3 passed`). A later RED test proved default-server import occurred after
     composition; resolving uvicorn first made all CLI lifecycle cases GREEN
     (`4 passed`) without exposing resident resources to that failure window.
6. Corrected Task 3 artifact contract:
   - RED: both reviewed legacy layouts (`moss-source.wav` or full
     `alignment.json`) were accepted (`2 failed`).
   - GREEN: only canonical `source.wav` and `mms_alignment.json` are accepted;
     either legacy name produces bounded 503 with no completion marker
     (`3 passed`, including canonical success).
7. Production vLLM default:
   - RED: the parser default omitted the OpenAI-compatible `/v1` prefix.
   - GREEN: the loopback CLI test now verifies
     `http://127.0.0.1:8000/v1` (`1 passed`).
8. Installed package and entrypoint:
   - RED: the wheel built from `50e6e8ff` contained the console-script metadata
     but no `scripts/rap_audio_backends/moss_backend.py`; the production MOSS
     loader could not resolve its absolute import outside the checkout.
   - GREEN: pruned multi-root discovery now includes only `streammuse*` and
     `scripts*`. The built wheel contains the backend and package marker. In a
     clean temporary venv, from `/tmp` with `PYTHONPATH` removed,
     `streammuse-rap-render-server --help` exited 0 and `find_spec()` resolved
     the backend beneath that venv's `site-packages`.
9. Final-marker rollback:
   - RED: adversarial `os.replace()` probes that renamed `response.zip` and
     then raised either `OSError` or a cancellation-like `BaseException` left
     the marker visible; both new cases failed (`2 failed`).
   - GREEN: owner and joined waiter receive the identical original failure,
     `response.zip` is removed followed by directory fsync, and an identical
     retry rerenders instead of hitting cache (`2 passed`).
10. Shared production workspace:
    - RED: when fake renderer and store used the same artifact root, only the
      workspace directory appeared in the fsync trace; none of the three Task
      3 artifact files did (`1 failed`).
    - GREEN: `source.wav`, `mms_alignment.json`, and `vocal.wav` are each file
      fsynced and their containing directory is fsynced before the response
      marker (`1 passed`), with exact file contents retained.
11. Bounded health identity:
    - RED: an absolute MOSS model path escaped unchanged, integer `ready` was
      accepted, overlong status was unbounded, and NaN/infinity survived the
      projection (`1 failed`).
    - GREEN: ready fields require real booleans; public text is capped at 128
      characters; non-finite/out-of-range numbers, URLs, secret assignments,
      and private paths are dropped; MOSS publishes only the snapshot basename
      (`1 passed`).
12. Complete publication timing:
    - RED: the clock-controlled test observed only the initial two clock reads,
      proving final manifest/package/metadata/marker work was absent
      (`1 failed`).
    - GREEN: a measured first pass mirrors the exact final publication
      sequence, cleanup is measured, and that measured publication duration is
      included again as the final-pass estimate. The controlled case records
      `packaging=9.0`, `total=24.0`, with byte package, persisted manifest,
      timing metadata, and `Server-Timing` in agreement (`1 passed`).

## Request And Idempotency Algorithm

- The endpoint rejects `Content-Length` above the explicit 64 KiB protocol
  limit before reading the body. It then consumes `Request.stream()` while
  checking cumulative bytes, so chunked overflow is rejected with stable HTTP
  413 before unbounded buffering.
- JSON decoding is followed directly and exclusively by
  `RemoteRapChunkRequest.from_payload()`. The server does not probe parser
  names or hand-validate contract fields.
- `Idempotency-Key` must equal the contract-validated `request_id`. Identity is
  compared using `request.canonical_json_bytes()`.
- Under one process-local lock, the store only inserts or joins
  `(canonical_body, Future)` for the request ID. Cache reads, request writes,
  rendering, artifact persistence, and package reads all happen outside that
  lock.
- Same-ID/same-canonical-body callers wait on the same Future. Same-ID/different
  canonical body fails with HTTP 409, whether work is active or complete.
- Every owner failure path attempts bounded diagnostics independently and then
  completes the Future exactly once with the original exception in a `finally`
  block. Diagnostic persistence failure is not allowed to strand waiters or
  replace the typed render failure.
- A matching `request.json` plus final `response.zip` is the only cache hit.
  Identical completed requests return the stored package bytes unchanged.
- Final response publication has a stricter wrapper than ordinary artifacts.
  Any `BaseException` from its atomic write triggers marker unlink and parent
  directory fsync before the original failure completes the shared Future.
  Rollback failures never replace the original owner/waiter exception.

## Atomic Artifacts And Timing

Each request workspace is `<artifact-root>/<request-id>/`. All JSON, WAV, copied
artifacts, timing metadata, and ZIP files use a temporary sibling, file flush +
fsync, `os.replace`, and containing-directory fsync. `response.zip` is replaced
last and is the sole success marker, so incomplete and failed workspaces cannot
be returned as completed cache entries. If final replace or directory fsync
raises after rename, the marker is removed and that removal is directory-fsynced
before the original error propagates.

The service requires and preserves the corrected Task 3 artifacts exactly:

- `source.wav`: raw connected MOSS phrase
- `mms_alignment.json`: complete MMS result, copied byte-for-byte
- `vocal.wav`: Task 3's final atomically published aligned phrase
- `render_failure.json`: Task 3's separate bounded failure evidence when render
  fails
- `alignment.json`: separate bounded Task 1 manifest alignment diagnostics
- `candidate_ledger.json`: complete H200-only candidate ledger
- `aligned.wav`: exact response audio
- renderer-supplied TextGrid files only; none are fabricated or required

When the Task 3 renderer and Task 4 store share the production workspace, the
three canonical renderer files are not copied onto themselves. Task 4 instead
opens and fsyncs each existing file, then fsyncs the workspace directory before
publishing the response marker. A separate renderer workspace still uses the
atomic durable copier.

Task 4 measures all artifact work and a first-pass publication that mirrors the
final sequence: manifest write, ZIP encode, timing-metadata write, and durable
response write. It also measures removal of the temporary measurement files.
Because exact measurement of the final ZIP would require mutating its own
manifest, the measured first-pass publication duration is included a second
time as the final-pass estimate. This explicitly accounts for both real passes
without an unrepresented encode/write. The nonzero result replaces provisional
`packaging=0`; total remains at least every stage and their sum. The final ZIP,
persisted manifest, timing JSON, and `Server-Timing` all use the same finalized
values, including cache responses.

## Errors And Health

Malformed input and idempotency-header mismatches return bounded HTTP 422
`invalid_request`; upload overflow is HTTP 413 `request_too_large`; canonical
identity conflicts are HTTP 409 `idempotency_conflict`. Typed orchestration
failures map to 422 `budget_exhausted`, 422 `no_valid_candidates`, or 503
`render_failed`. Unknown exceptions map to sanitized HTTP 500
`internal_error`. No response includes exception text, prompts, request bodies,
authorization data, tracebacks, or failure-diagnostic details.

Health is projected, not redacted. Top-level compatibility fields and only the
`vllm`, `moss`, `aligner`, `rubberband`, and `warmup` subsections are considered.
Recursive summaries permit only `ready`, `status`, `identity`, `version`,
`model`, `profile`, and nested `warmup` scalar values. URLs, API keys, tokens,
paths, cache locations, credentials, arbitrary keys, nested variants, and
non-scalar values cannot escape. Ready values require `bool`; strings are
trimmed, control-checked, and capped at 128 characters; numeric warmup summaries
must be finite and bounded. Absolute model paths become only a bounded basename,
while absolute paths under every other approved key are omitted. Production
composition likewise stores only that public MOSS model identity.

## Lazy Composition And Lifecycle

The server module remains Mac-importable: MOSS, MMS, aligned-renderer, Torch,
and related runtime imports occur only during app execution or in
`_compose_real_worker()` and its lazy helpers. CLI argument parsing does not
traverse orchestration/codec package initializers. Production composition:

1. constructs and probes `LocalChatModelClient` for the selected local vLLM;
2. constructs `IndependentChoiceCandidateGenerator`, `CmuProsodyAnalyzer`,
   default `ScoreWeights`, and `ChunkCandidatePlanner`;
3. loads one `PersistentMossSynthesizer` and one `MmsForcedAligner`;
4. executes real MOSS synthesis and MMS alignment warmup before readiness;
5. probes Rubber Band, constructs `MossAlignedPhraseRenderer`, then
   `RapChunkOrchestrator`; and
6. publishes allowlisted ready/model/version/profile/warmup health.

An `ExitStack` owns every public `close()` offered by resident components,
cleans partial startup in reverse order, and is closed exactly once when the
server exits or raises. The default server runner is resolved before resident
composition, so runner import failure cannot leak loaded resources. The CLI
remains loopback-only by default and requires `--allow-public-bind` for any
non-loopback host.

## Tests

Focused service, Task 1 package, Task 2 orchestration, Task 3 renderer/runtime,
candidate/planner dependency, local client, and shared render-contract run:

`uv run pytest tests/unit/presentation/test_rap_render_server.py tests/unit/application/rap/test_chunk_orchestration.py tests/unit/infrastructure/rap/test_chunk_package.py tests/unit/infrastructure/rap/test_moss_tts.py tests/unit/infrastructure/rap/test_mms_forced_alignment.py tests/unit/infrastructure/rap/test_moss_aligned_phrase.py tests/unit/infrastructure/rap/test_generators.py tests/unit/infrastructure/rap/test_prosody.py tests/unit/infrastructure/inference/test_local_chat_model_client.py tests/unit/experiments/rap_audio_protocols/test_contracts.py -q`

Result after the fresh re-review corrections: `195 passed in 3.08s`. The first
sandboxed run had `193 passed` plus the two expected loopback-bind denials; the
unchanged command passed with loopback permission.

Also passed:

- focused server suite: `30 passed in 0.98s`
- built-wheel smoke in a clean temporary venv, from `/tmp` with no repository
  `PYTHONPATH`: installed console entrypoint `--help` exited 0 and the packaged
  backend resolved under `site-packages/scripts/rap_audio_backends/moss_backend.py`
- model-free lazy composition/import tests; no real model was loaded
- `uv run ruff check` on the owned server and test: all checks passed
- `uv run ruff format --check` on the owned server and test: already formatted
- `git diff --check` on Task 4-owned files

## Integration State

Task 3 fix `cdc26f82` and shared request update `d7b1cdb3` are integrated. The
real renderer/orchestrator test crosses the exact Task 1 diagnostics schema at a
non-90 BPM tempo and retains canonical `source.wav`, `mms_alignment.json`, and
`vocal.wav`. Task 4 requires those corrected names and has no compatibility
fallback for `moss-source.wav` or the old full-record `alignment.json` layout.

The coordinator still owns the real H200 model/download, vLLM, Rubber Band,
MOSS/MMS warmup, and HTTP smoke. Unit and adjacent tests intentionally use
injected runtimes and do not load or download model weights.

The normal H200 quickstart no longer relies on the absent
`environment_manifest.json`, the nonexistent `$ASSET_ROOT/ted` reference, or a
checkout-root `PYTHONPATH`. It installs the checkout editable into the resident
MOSS environment and invokes the console entrypoint directly. The documented
verified assets are snapshot
`/data/home/Andrew.Yang/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-TTS-v1.5/snapshots/cdd3b911b1585e3f2dbc7775ef10f9926f58850a`
and reference WAV
`/data/home/Andrew.Yang/StreamMUSE/audio-protocol-downloads/TED-TTS/datasets/Ref/0011_000001.wav`.
