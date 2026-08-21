# Task 7 Implementation Report

## Status

Implemented bounded remote-chunk monitoring, terminal presentation, Mac website
projection, normal-runtime operations documentation, and the Task 8 acceptance
handoff. The implementation starts from accepted Task 6 revision
`0af9f57c8e3046326b6cd7aa47e65310fcd3d60c`.

Task 3 renderer files, Task 4 server files, Task 6 controller/events/CLI files,
research notes, and unrelated offline experiment work were not modified for
Task 7. Existing quickstart content was retained and extended in place.

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

## Implementation

- Chunk publication and both state projectors apply one fixed whitelist before
  data enters event, terminal, or browser history.
- Live state is limited to two selected lines, two flows, 32 slots per flow,
  four context lines, 16 component scores, and eight diagnostic/map entries.
- Stage names are canonicalized to generation, evaluation, MOSS, aligner, R3,
  package, transfer, Mac, and total. Manifest `warp` and `packaging` aliases are
  projected as R3 and package.
- Alignment projection retains method, confidence, and bounded fallback counts,
  but excludes source/target anchors and character spans.
- The terminal dashboard and stream name local eSpeak fallback commitments and
  display request/chunk lifecycle, all bounded research evidence, and failure.
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
uv run pytest tests/unit/application/rap/test_monitoring_payloads.py tests/unit/application/rap/test_monitoring.py tests/unit/presentation/rap_demo/test_terminal.py tests/unit/presentation/rap_demo/test_terminal_state.py tests/unit/presentation/rap_demo/test_terminal_dashboard.py tests/unit/presentation/rap_demo/test_terminal_stream.py tests/unit/presentation/rap_demo/test_server.py tests/integration/test_rap_demo_browser_reducer.py -q
```

Result: `82 passed in 1.35s`.

Existing eSpeak snapshot and regression gate:

```text
uv run pytest tests/unit/infrastructure/rap/test_speech.py tests/unit/application/rap/test_audio_coordination.py tests/unit/presentation/rap_demo/test_cli.py tests/integration/test_realtime_rap_audio.py -q
```

Result: `66 passed in 1.91s`.

Static checks:

```text
uv run ruff check src/streammuse/application/rap/monitoring_payloads.py src/streammuse/application/rap/monitoring.py src/streammuse/presentation/rap_demo/terminal_state.py src/streammuse/presentation/rap_demo/terminal_dashboard.py src/streammuse/presentation/rap_demo/terminal_stream.py tests/unit/application/rap/test_monitoring_payloads.py tests/unit/application/rap/test_monitoring.py tests/unit/presentation/rap_demo/test_terminal_state.py tests/unit/presentation/rap_demo/test_terminal_dashboard.py tests/unit/presentation/rap_demo/test_terminal_stream.py tests/unit/presentation/rap_demo/test_server.py tests/integration/test_rap_demo_browser_reducer.py
node --check src/streammuse/presentation/rap_demo/static/js/rap-demo-state.js
```

Results: `All checks passed!` and JavaScript syntax exit code 0.
