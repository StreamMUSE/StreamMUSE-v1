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

## Atomic Artifacts And Timing

Each request workspace is `<artifact-root>/<request-id>/`. All JSON, WAV, copied
artifacts, timing metadata, and ZIP files use a temporary sibling, file flush +
fsync, `os.replace`, and containing-directory fsync. `response.zip` is replaced
last and is the sole success marker, so incomplete and failed workspaces cannot
be returned as completed cache entries.

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

Task 4 measures artifact persistence plus a first package encode/durable write.
That nonzero first-pass measurement replaces the provisional `packaging=0`,
the final total is at least the prior orchestration total plus packaging and at
least the sum of component stages, and the provisional warning is removed.
The finalized immutable manifest is encoded again for `response.zip`.
`Server-Timing` is generated from that same final manifest total and persisted
for completed-cache responses.

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
non-scalar values cannot escape.

## Lazy Composition And Lifecycle

The server module remains Mac-importable: MOSS, MMS, aligned-renderer, Torch,
and related runtime imports occur only in `_compose_real_worker()` or its lazy
helpers. Production composition:

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

Result: `190 passed in 3.05s`.

Also passed:

- model-free lazy composition import smoke, including proof that loading the
  exact Task 3 classes does not import Torch or torchaudio
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
