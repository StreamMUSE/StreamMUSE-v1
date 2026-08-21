# Task 7 Implementation Report

## Status

The rejected `0cecbfc5` implementation was repaired starting from integrated
revision `5bf9ff84cea9cf9814ee629fc0d704ea50de9082`, then verified with concurrent
Task 4 revision `05fa3d84e6244c3d24fc60c60cc88d962bf8c3de`. Production telemetry now starts
at the remote wire/renderer and Mac preparation boundaries, survives the real
controller and canonical publisher, and is ordered correctly across browser
resets, reconnect snapshots, coordinator epochs, and session restarts.
The fresh rereview corrections were developed against integrated revision
`d67360bae22465c422f2243ff32ac583634ae2fd`.

The permitted minimum telemetry plumbing changed the remote diagnostics
contract, MOSS result metadata, orchestrator, Mac preparation strategy,
controller projection, monitoring, terminal, and browser files. It did not
change accepted audio timing/playback behavior. Task 4 render-server source,
its tests, `pyproject.toml`, and unrelated offline experiment work were not
modified by this repair. Concurrent quickstart command edits were preserved and
the documentation additions were kept narrow.

## RED And GREEN Record

### Baselines

Before Task 7 edits, the focused monitoring, terminal, website, and server set
passed with `73 passed`:

```text
uv run pytest tests/unit/application/rap/test_monitoring_payloads.py tests/unit/application/rap/test_monitoring.py tests/unit/presentation/rap_demo/test_terminal.py tests/unit/presentation/rap_demo/test_terminal_state.py tests/unit/presentation/rap_demo/test_terminal_dashboard.py tests/unit/presentation/rap_demo/test_terminal_stream.py tests/unit/presentation/rap_demo/test_server.py tests/integration/test_rap_demo_browser_reducer.py -q
```

The existing eSpeak regression set passed with `66 passed`:

```text
uv run pytest tests/unit/infrastructure/rap/test_speech.py tests/unit/application/rap/test_audio_coordination.py tests/unit/presentation/rap_demo/test_cli.py tests/integration/test_realtime_rap_audio.py -q
```

### Cycle 1: Event And Projector Bounds

RED:

```text
uv run pytest tests/unit/application/rap/test_monitoring_payloads.py tests/unit/application/rap/test_monitoring.py -q
```

Collection failed because `bounded_chunk_event_payload` did not exist. The new
tests required bounded lines, flows, slot/stress schedules, candidate counts,
component scores, prompt/context, canonical stage timings, deadline evidence,
alignment summaries, warnings, hashes, artifact references, and failure state,
while rejecting WAV bodies, candidate ledgers, anchors, and character spans.

GREEN: the same command completed with `23 passed`.

### Cycle 2: Terminal Projection

RED:

```text
uv run pytest tests/unit/presentation/rap_demo/test_terminal_state.py tests/unit/presentation/rap_demo/test_terminal_dashboard.py tests/unit/presentation/rap_demo/test_terminal_stream.py -q
```

The run produced `4 failed, 21 passed`: terminal state had no remote chunk and
neither terminal renderer displayed the required evidence.

GREEN: the same command completed with `25 passed`. The remote section is
conditional, so existing eSpeak snapshots remain byte-for-byte unchanged.

### Cycle 3: Browser Projection And Layout Contract

RED:

```text
uv run pytest tests/integration/test_rap_demo_browser_reducer.py tests/unit/presentation/rap_demo/test_server.py -q
```

The run produced `3 failed, 16 passed`: the browser reducer had no bounded
chunk API and the static monitor had no remote audit panel.

GREEN: the same command completed with `19 passed`. The tests also prove that
the runtime control group still contains only Start, Stop, and Reset.

### Cycle 4: Requested-State Placeholders

RED:

```text
uv run pytest tests/integration/test_rap_demo_browser_reducer.py -q
```

The run produced `1 failed, 2 passed`: absent candidate counts rendered as
`0 / 1 / 2 / 3` because `Array.map` supplied indexes as fallback arguments.
An initial sandboxed invocation could not read the existing uv cache; rerunning
the unchanged command with approved cache access produced the behavioral RED.

GREEN: the same command completed with `3 passed`, with requested-state counts
rendered as `-- / -- / -- / --`.

### Review Repair Cycle 1: Versioned Wire And Production Evidence

RED:

```text
uv run pytest -q tests/unit/domain/rap/test_remote_chunk.py tests/unit/application/rap/test_chunk_orchestration.py tests/unit/infrastructure/rap/test_moss_aligned_phrase.py tests/unit/infrastructure/rap/test_chunk_package.py tests/unit/application/rap/test_chunk_audio.py tests/unit/application/rap/test_monitoring_payloads.py tests/unit/application/rap/test_chunk_realtime.py tests/unit/application/rap/test_monitoring.py tests/integration/test_rap_demo_browser_reducer.py
```

The first run stopped during collection because
`MAX_BOUNDED_CHUNK_EVENT_BYTES` did not exist. A second run excluding that
module produced `78 failed, 148 passed, 13 errors`; the new tests exposed the
missing strict monitoring schema, real `PreparedRapChunk` propagation, failure
reason, browser session/epoch ordering, and snapshot projection.

The contract/renderer/package GREEN slice:

```text
uv run pytest -q tests/unit/domain/rap/test_remote_chunk.py tests/unit/application/rap/test_chunk_orchestration.py tests/unit/infrastructure/rap/test_moss_aligned_phrase.py tests/unit/infrastructure/rap/test_chunk_package.py
```

Result: `151 passed`.

### Review Repair Cycle 2: Bounded Mac And Canonical Publication

RED after the first implementation pass:

```text
uv run pytest -q tests/unit/application/rap/test_chunk_audio.py tests/unit/application/rap/test_monitoring_payloads.py tests/unit/application/rap/test_chunk_realtime.py tests/unit/application/rap/test_monitoring.py
```

Result: `3 failed, 84 passed`. The failures caught immutable prepared values,
the actual coordinator epoch, and the existing 256-byte warning limit. The
follow-up idempotence test then caught selected schedules being lost during the
publisher/projector's repeated normalization. GREEN results are included in
Final Verification below.

### Review Repair Cycle 3: Terminal And Browser Ordering

Terminal RED:

```text
uv run pytest -q tests/unit/presentation/rap_demo/test_terminal_stream.py::test_stream_renders_complete_remote_chunk_diagnostics_and_espeak_fallback tests/unit/presentation/rap_demo/test_terminal_dashboard.py::test_dashboard_displays_complete_bounded_remote_chunk_research_evidence tests/unit/presentation/rap_demo/test_terminal_state.py::test_projector_exposes_only_the_latest_bounded_remote_chunk_diagnostics
```

Result: `2 failed, 1 passed`; target and selected schedules and truthful
generation-input labeling were absent. GREEN: `3 passed` as part of the
`25 passed` terminal suite.

Browser bound RED:

```text
node --check src/streammuse/presentation/rap_demo/static/js/rap-demo-state.js
node --check src/streammuse/presentation/rap_demo/static/js/rap-demo.js
uv run pytest -q tests/integration/test_rap_demo_browser_reducer.py tests/unit/presentation/rap_demo/test_server.py
```

Result: `2 failed, 21 passed`; the new hostile payload exceeded the byte
ceiling and exposed a missing reducer return. GREEN: `23 passed` after the
bounded compaction path was fixed.

### Review Repair Cycle 4: Published Artifact And Lazy Browser Bounds

RED:

```text
uv run pytest -q tests/unit/domain/rap/test_remote_chunk.py tests/integration/test_rap_demo_browser_reducer.py::test_browser_chunk_sanitizer_enforces_numeric_collection_and_byte_bounds
```

Result: `33 failed, 55 passed`. The strict contract rejected Task 4's retained
`vocal.wav`, and a proxied browser array proved that bar sanitization scanned
past the two-item prefix. GREEN: the same command completed with `88 passed`
after the exact artifact contract, docs, lazy slice, and normalized
stress/alignment bounds were corrected.

### Review Repair Cycle 5: Semantic Numeric Ranges

RED:

```text
uv run pytest -q tests/unit/application/rap/test_monitoring_payloads.py::test_chunk_event_payload_preserves_zero_budget_and_rejects_negative_durations tests/integration/test_rap_demo_browser_reducer.py::test_browser_chunk_projection_orders_by_session_epoch_and_strict_sequence tests/integration/test_rap_demo_browser_reducer.py::test_browser_chunk_sanitizer_enforces_numeric_collection_and_byte_bounds
```

Result: `3 failed`. A zero request budget was replaced by a manifest fallback,
negative duration/timing values were retained, and fractional browser event
sequences survived sanitization. GREEN: the same command completed with
`3 passed` after explicit nonnegative duration/timing ranges and safe-integer
event identity handling were added. Deadline slack remains signed by design.

### Fresh Rereview: Rejected Evidence, Reconnect, And Artifact IDs

RED commands:

```text
uv run pytest -q tests/unit/application/rap/test_chunk_audio.py::test_remote_chunk_rejects_selected_schedule_mismatch_as_preparation_failure tests/unit/application/rap/test_chunk_realtime.py::test_real_mac_rejection_retains_returned_evidence_through_fallback_recording tests/unit/application/rap/test_monitoring_payloads.py::test_chunk_event_payload_preserves_all_versioned_artifact_references
uv run pytest -q tests/integration/test_rap_demo_browser_reducer.py::test_stopped_disconnect_reconnects_and_accepts_later_restart_snapshot tests/integration/test_rap_demo_browser_reducer.py::test_browser_chunk_projection_and_event_cache_drop_unbounded_artifact_bodies
```

The Python RED runs exposed an untyped preparation error, no completed event
for a decoded-but-rejected response, and only eight retained artifact IDs. The
browser RED runs exposed a stopped snapshot suppressing reconnection and the
same eight-entry truncation of the versioned artifact map.

GREEN command:

```text
uv run pytest -q tests/unit/application/rap/test_chunk_audio.py::test_remote_chunk_rejects_selected_schedule_mismatch_as_preparation_failure tests/unit/application/rap/test_chunk_realtime.py::test_real_mac_rejection_retains_returned_evidence_through_fallback_recording tests/unit/application/rap/test_monitoring_payloads.py::test_chunk_event_payload_preserves_all_versioned_artifact_references tests/integration/test_rap_demo_browser_reducer.py::test_stopped_disconnect_reconnects_and_accepts_later_restart_snapshot tests/integration/test_rap_demo_browser_reducer.py::test_browser_chunk_projection_and_event_cache_drop_unbounded_artifact_bodies
```

Result: `5 passed in 0.83s`.

## Implementation

- `streammuse.rap_chunk_monitor.v1` is a strict nested wire schema for alignment
  method/confidence, source WAV hash, and exact request-relative artifact IDs.
- MOSS produces the source evidence; the orchestrator binds stable Task 4
  artifact names; Mac preparation combines the original request, returned
  manifest, transfer measurements, and Mac validation/mix timing.
- Controller chunk events pass that production summary through the canonical
  publisher and recorder. Rejections carry an explicit failure reason while
  retaining bounded returned manifest evidence through a typed rejection when
  package decoding succeeded; rejected audio still activates local fallback.
- Chunk publication and both state projectors apply one idempotent whitelist
  before data enters event, terminal, or browser history.
- Live state is limited to two selected lines, two flows, 32 slots per flow,
  four context lines, 16 component scores, and eight generic diagnostic/map
  entries. The explicit versioned artifact map separately retains all ten
  stable IDs and remains capped at exactly that contract size.
- Stage names are canonicalized to generation, evaluation, MOSS, aligner, R3,
  package, transfer, Mac, and end-to-end total. Manifest `warp` and `packaging`
  aliases are projected as R3 and package.
- Alignment projection retains method, confidence, and bounded fallback counts,
  but excludes source/target anchors and character spans.
- Numeric magnitudes, derived schedules, map/list scans, nesting, and strings
  are bounded, with a tested 24,000-byte serialized ceiling in Python and JS.
- Browser ordering uses session ID, coordinator epoch, and strict sequence;
  reset/session changes and authoritative null snapshots clear the panel, and
  websocket snapshots update it directly. A stopped runtime snapshot does not
  suppress transport reconnection, so a later restart snapshot can be applied.
- The terminal dashboard and stream name local eSpeak fallback commitments and
  display request/chunk lifecycle, target and selected schedules, all bounded
  research evidence, and failure.
- The website uses text-only DOM updates, a full-width audit band with a dense
  two-column inner grid, and the existing palette. It adds no controls.

## Operations And Acceptance Docs

The quickstart now documents:

- loopback H200 vLLM and render-service startup with explicit unused GPUs;
- normal Mac forwarding of port 8020 only;
- the Mac-local website, Start/Stop/Reset endpoints, and health endpoints;
- H200 and Mac artifact locations and the bounded-state boundary;
- ordered Mac, tunnel, render-service, and vLLM shutdown;
- the exact device-free local eSpeak legacy command;
- direct port-8001 forwarding only as optional vLLM diagnostics.

The acceptance record contains no claimed H200 performance evidence. GPU,
latency, candidate, alignment, exact-duration, fallback, underrun, artifact,
and listening fields are explicitly `TODO (Task 8)`.

## Visual Verification

The populated monitor was inspected in the in-app browser against a temporary
local fixture served through the real FastAPI monitor:

- Desktop at 1440 x 900: two 693.5 px diagnostic columns, no body overflow,
  and controls exactly `Start`, `Stop`, `Reset`.
- Mobile at 390 x 844: one 351 px diagnostic column, no audit-panel overflow,
  and the same three runtime controls.
- Browser console logs were empty during desktop inspection.

The fixture values were used only to exercise rendering and are not H200
latency or acceptance measurements.

## Final Verification

Focused Task 7 gate:

```text
uv run pytest -q tests/unit/domain/rap/test_remote_chunk.py tests/unit/application/rap/test_chunk_orchestration.py tests/unit/infrastructure/rap/test_moss_aligned_phrase.py tests/unit/infrastructure/rap/test_chunk_package.py tests/unit/application/rap/test_chunk_audio.py tests/unit/application/rap/test_chunk_realtime.py tests/unit/application/rap/test_monitoring_payloads.py tests/unit/application/rap/test_monitoring.py tests/unit/presentation/rap_demo/test_terminal.py tests/unit/presentation/rap_demo/test_terminal_state.py tests/unit/presentation/rap_demo/test_terminal_dashboard.py tests/unit/presentation/rap_demo/test_terminal_stream.py tests/unit/presentation/rap_demo/test_server.py tests/integration/test_rap_demo_browser_reducer.py tests/unit/presentation/test_rap_render_server.py
```

Result: `343 passed in 4.54s`.

Existing eSpeak snapshot and regression gate:

```text
uv run pytest -q tests/unit/infrastructure/rap/test_speech.py tests/unit/application/rap/test_audio_coordination.py tests/unit/presentation/rap_demo/test_cli.py tests/integration/test_realtime_rap_audio.py
```

Result: `66 passed in 1.91s`.

Static checks:

```text
uv run ruff check src/streammuse/application/rap/chunk_audio.py src/streammuse/application/rap/chunk_realtime.py src/streammuse/application/rap/monitoring_payloads.py tests/unit/application/rap/test_chunk_audio.py tests/unit/application/rap/test_chunk_realtime.py tests/unit/application/rap/test_monitoring_payloads.py tests/integration/test_rap_demo_browser_reducer.py
node --check src/streammuse/presentation/rap_demo/static/js/rap-demo-state.js
node --check src/streammuse/presentation/rap_demo/static/js/rap-demo.js
git diff --check
```

Results: `All checks passed!`; both JavaScript syntax checks and the diff check
exited 0.
