# Task 4 Implementation Report

## Delivered

Implemented `streammuse.presentation.rap_render_server` and registered the
`streammuse-rap-render-server` CLI. The dependency-injected FastAPI app exposes
`GET /health` and `POST /v1/rap/chunks/render`; successful responses use the
Task 1 package codec and `application/vnd.streammuse.rap-chunk+zip` media type.

## RED/GREEN Evidence

1. RED: `uv run pytest tests/unit/presentation/test_rap_render_server.py -q`
   initially failed at collection because `rap_render_server` did not exist.
2. GREEN: initial implementation exposed a real artifact serialization failure:
   frozen `mappingproxy` candidate ledgers could not be JSON encoded. A direct
   store reproduction located the failure at the atomic JSON write boundary.
3. GREEN after recursive mapping normalization: server suite passed (9 tests),
   then expanded coverage passed (11 tests), and final focused coverage passed
   with the adjacent orchestrator tests (39 tests).

## Request And Idempotency Algorithm

- The endpoint decodes JSON and delegates all schema validation to
  `RemoteRapChunkRequest.from_dict()` when provided by the contract, otherwise
  the current Task 1 `from_payload()` compatibility name. It uses the request's
  `canonical_json_bytes()` for identity; it never manually reads request fields.
- `Idempotency-Key` must exactly equal the parsed `request_id`.
- A process-local lock protects only the request-ID map and completed-marker
  lookup. The map stores `(canonical_body, Future)` for each active request.
- Same canonical body joins the existing future; a different canonical body for
  the same ID returns HTTP 409. The owner invokes the blocking orchestrator in
  FastAPI's threadpool without holding the lock.
- Completed `response.zip` plus matching `request.json` returns the stored bytes
  unchanged. Failed work has no final package marker and is rendered again on a
  later identical request.

## Atomic Artifacts

Each request workspace is `<artifact-root>/<request-id>/`. JSON, copied
renderer artifacts, aligned WAV, and the package use a temporary sibling,
`fsync`, and `os.replace`. `response.zip` is written last and is the only cache
completion marker. The service persists canonical `request.json`, full
`candidate_ledger.json`, `alignment.json` from MMS diagnostics, `manifest.json`,
`aligned.wav`, `response.zip`, and bounded `failure.json` when applicable.
It copies `source.wav` and any renderer-supplied TextGrid; it neither requires
nor fabricates a TextGrid. A successful renderer artifact must provide
`source.wav`.

## Errors And Health

Malformed input and idempotency-header mismatches return bounded HTTP 422
`invalid_request`. Typed orchestration failures map to 422
`budget_exhausted`, 422 `no_valid_candidates`, or 503 `render_failed`; unknown
exceptions map to a sanitized 500 `internal_error`. Responses never echo
exception messages, prompt material, or authorization headers. Health exposes
only approved readiness/version/model fields and strips sensitive URL, path,
credential, cache, and reference-WAV keys.

## Lazy Composition And CLI

The module imports no MOSS/MMS runtime dependencies. `main()` supports an
injected composition factory and starts only after loopback/public-bind checks;
the default host is `127.0.0.1`, and non-loopback binding needs
`--allow-public-bind`. The default real-worker factory is intentionally a small
late-bound seam because Task 3 currently provides `PersistentMossSynthesizer`
and `MmsForcedAligner` but not `MossAlignedPhraseRenderer`. This keeps Mac
imports and unit tests model-free.

## Tests

`uv run pytest tests/unit/presentation/test_rap_render_server.py tests/unit/application/rap/test_chunk_orchestration.py -q`

Result: `39 passed in 0.78s`.

Also passed: `uv run python -m py_compile src/streammuse/presentation/rap_render_server.py`
and `git diff --check` on Task 4 source/test/package files.

## Integration Concerns

- Task 1 currently calls the constructor `from_payload`, while the controller
  amendment calls it `from_dict`; the server uses the contract-only compatibility
  dispatch so either stable public alias works.
- Task 3 integration must implement the default composition factory and make its
  phrase renderer place the source WAV in the request workspace as `source.wav`.
  Any active optional TextGrid can be emitted there and will be retained; MMS
  alignment JSON is persisted by this service.
- The coordinator should perform the stated real H200 MOSS/MMS warmup and smoke
  after the Task 3 phrase renderer is integrated.
