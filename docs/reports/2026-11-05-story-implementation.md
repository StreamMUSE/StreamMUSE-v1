# Directed Story Backend, 2026-11-05

## Ready for Parent Integration

Implemented a session-scoped, immutable, ordered direction ledger through the existing
v3 live-edit path. Directions are separate from topic and recent lyric context, survive
rescue waves, and remain present after subsequent topic/flow edits. No summarizer,
scheduling-policy change, audio/stress-policy change, deployment, commit, or push.
Existing dirty tempo changes were preserved. No GPU calls were made.

Parent mounts the optional router; this task did not edit demo server.py, cli.py,
or static files:

```python
from streammuse.presentation.rap_demo.story_api import create_story_router
app.include_router(create_story_router(runtime))
```

`runtime.controller` must be the v3 `BarStreamController`; it is resolved on each
request so runtime replacement/reset cannot leave the router holding an old controller.

## Interfaces

- `controller.submit_story_event(text, event_id=None, *, source="typed")`.
- `controller.story_snapshot()`.
- GET `/api/story`: `{session_id, enabled, version, events, limits, continuity}`.
- POST `/api/story/events`: input `{text, event_id?, source?, session_id?}`; output
  `{session_id, event, duplicate, effective_from_bar, status:"accepted"}`.
- Event: `{event_id, sequence, text, received_time, source}`. `received_time` is
  Unix seconds; sequence starts at one. Omitted event IDs are generated UUID hex.
- Same ID/text/source is idempotent and returns the original event; conflicting
  content returns HTTP 409. Invalid or over-budget input returns 422. Wrong runtime
  or unsupported live peer returns 409. Transport failures return 503.
- Clients should supply stable event IDs when retrying requests.
- `source` accepts exactly `"typed"` or `"microphone"`, defaulting to `"typed"`.
  The route passes it through to the immutable event; invalid values return 422.
  Reusing an event ID with a changed source returns 409, like changed text.
- Optional `session_id` is forwarded by the parent-owned API to controller
  `expected_session_id`; a mismatch raises `StoryConflict` before any mutation.
  Requests omitting this field retain legacy behavior and are not reset-fenced.

The first event enables story mode. The effective bar is the existing
`playhead + 1` editable boundary. Before Start, directions are queued into the start
payload. Reset discards the plan/ledger and advances the session epoch. Stop/resume
within a session retains the ledger. There is no standalone clear/delete operation.

## Protocol and Prompt

Optional `PlanEdit.story_context` serializes as
`{"version": N, "events": [event, ...]}` within the existing `update` message.
Start payloads retain story edits through the existing `edits` list. New capability:
`story_context_v1`. Deploy updated client and renderer together; a live controller
rejects story submission if its peer did not advertise this capability. Legacy
topic/flow edits omit the new field and keep their existing payload and prompt.

The writer propagates the immutable context through every candidate/rescue request.
Both generator prompt variants include the directions. Recent lyric context remains
replaceable and is labeled draft context in story mode, not confirmed playback.
Generated records expose `provenance.story_version` and `story_event_ids`; these
record prompt inclusion, not fulfillment. Stale in-flight results are discarded via
the existing plan sequence check. Superseded failures cannot consume new edit retries.
An exact repeated story edit is idempotent at the session boundary.

Bounds: 32 events, 64 characters per event ID, 512 characters per individual text,
and **512 total UTF-8 bytes of direction text** across the session. Overflow rejects
the new event without truncating earlier events. Metadata is not repeated in the
model prompt; only sequence numbers and text are included. This conservative byte
budget keeps the added text compact for the 2048-token deployment, but is not a
tokenizer-aware total-prompt guarantee for arbitrary topics, flows, or model settings.

## Verification

Tests were added first and initially failed on the missing story module. Additional
tests exposed and then verified before-start delivery, failed-send retry, missing
peer capability, and superseded writer-failure handling.

Final command (local CPU tests, using the requested import override):

```sh
env PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/domain/rap/test_story.py \
  tests/unit/application/rap/test_story.py \
  tests/unit/presentation/test_story_api.py \
  tests/unit/application/rap/test_lyric_writer.py \
  tests/unit/application/rap/test_lyric_session.py \
  tests/unit/application/rap/test_bar_stream.py \
  tests/unit/domain/rap/test_lyric_stream.py \
  tests/unit/infrastructure/rap/test_generators.py \
  tests/unit/presentation/test_rap_lyric_stream_server.py
```

Result: **187 passed in 14.14s**. `git diff --check` passed.
Tests cover ordering, immutable bounds, serialization, duplicate/conflicting IDs,
three incremental directions through serialized start, ordinary topic preservation,
rescue retention, editing during generation, old-result rejection, committed-bar
preservation, exact edit retry, failed transport retry, reset, and API validation.

## Limits

- Persistence is in-memory and session-scoped, not disk durability. Reconnect/start
  serialization carries the plan, but no new acknowledged transport/replay protocol
  was added. `accepted` means locally accepted, not generated, played, or fulfilled.
- Confirmed-playback continuity is deferred. Existing server history mixes speculative
  predecessors and committed lyrics. Neither is promoted into an established story
  ledger. `continuity` explicitly reports `directions_only`; no played-lines claim.
- Fallback catalogs and emergency audio are unchanged and do not promise plot adherence.
- The model may fail to honor a direction. No semantic evaluator or automatic summary
  was added, and a 12-bar live/GPU demo was not run by this task.
- Parent owns mounting, H200 deployment on the dedicated new port, and audible testing.

## Changed Paths

New:
- `src/streammuse/domain/rap/story.py`
- `src/streammuse/presentation/rap_demo/story_api.py`
- `tests/unit/domain/rap/test_story.py`
- `tests/unit/application/rap/test_story.py`
- `tests/unit/presentation/test_story_api.py`
- This report.

Modified:
- `src/streammuse/domain/rap/generation.py`
- `src/streammuse/domain/rap/lyric_stream.py`
- `src/streammuse/application/rap/live_plan.py` (required propagation through existing edits)
- `src/streammuse/application/rap/lyric_session.py`
- `src/streammuse/application/rap/lyric_writer.py`
- `src/streammuse/application/rap/bar_stream.py`
- `src/streammuse/infrastructure/rap/generators.py`

## Follow-up Acceptance Harness

Added `scripts/experiments/research_day/story_acceptance.py`. Not executed against
the real demo; the parent owns execution. From the worktree root:

```sh
env PYTHONPATH=src .venv/bin/python -m scripts.experiments.research_day.story_acceptance \
  --url http://127.0.0.1:8018 --output-dir /tmp/story-acceptance-20261105
```

Defaults: continuous mode, 60 BPM, 12 playback bars; submit concise directions when
playback callbacks for bars 0, 4, and 8 are observed. Requests stop-after-bar when bar
11 starts, then waits for stopped state. Startup/run deadline is 240 seconds.
`--bpm`, `--bars`, `--event-bars`, `--events`, `--timeout`, and `--poll-interval`
are configurable. Output must be empty; existing evidence is never overwritten.

The tool stops before configuring/resetting and confirms reset cleared the story.
It attempts stop again in `finally`, including failures and interruption. No audio
gain/stress-policy settings are changed, and no models/services are launched.

Artifacts: `http.jsonl`, `events.jsonl`, `states.jsonl`, `summary.json`,
`story_events.json`, `chosen_lyrics.json`. Summary records acceptance, effective bar,
HTTP timing, exact chosen lyric/version, and the first subsequent callback carrying
each direction's event ID. Missing provenance remains unverified. Event gaps and
incomplete observation are explicit. Status `observed` means software evidence
collected, never acoustic or semantic success.

Required client instrumentation: `bar_stream.py` now includes `story_version` and
`story_event_ids` from the exact lyric record in `bar_reserved` and `bar_frozen`.
The harness joins the frozen choice to a same-session, same-bar subsequent playback
callback; it never assigns the latest draft's version to an older chosen recording.
Deploy these metadata-only additions with the client before running the harness.

Offline tests are in `tests/unit/experiments/test_story_acceptance.py`, plus the
provenance propagation test in `tests/unit/application/rap/test_story.py`.
They cover the mocked 12-bar run, safe cleanup, output preservation, missing metadata,
older chosen versions, session isolation, parameter validation, and malformed ACKs.

Follow-up verification: **127 passed in 6.97s**, using
`env PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider` with
`test_story_acceptance.py`, `test_rap_live_edit_acceptance.py`, application
`test_story.py`, `test_bar_stream.py`, and presentation `test_story_api.py`.
`git diff --check` passed. No live acceptance run was performed.

## Final Integration Evidence

The parent ran the real demo after deploying its story-priority prompt improvement
and restarting the Mac client with the telemetry additions. This task inspected
`output/research_day_20261105/story_prioritized/summary.json` directly:

- URL `http://127.0.0.1:8019`, 60 BPM, 16 target/observed playback bars.
- Status `observed`; stopped/ready at completion; no event gaps.
- All 16 chosen bars: `remote_latest`, source `moss_lyric_stream`.
- All three events accepted, effective bars 1, 5, 9.
- First matching story-version callbacks: versions 1, 2, 3 at bars 1, 5, 9;
  request-to-callback delays **3849.600542, 3886.328542, 3953.344708 ms**.
- Selected first matching lyrics mention the missing map, stranger, and package.
  These are trace observations, not a semantic evaluation or acoustic measurement.

The earlier `story` run lacked telemetry because its Mac client started at
14:15:30 UTC, before the telemetry edit at 14:15:59 UTC. All 24 reserved and
13 frozen records lacked the keys entirely. Serialization and event wrappers
preserve provenance; the parent client restart resolved the stale loaded code.
The old run remains incomplete rather than receiving inferred provenance.

Source-field verification: 39 focused API/application/harness tests passed.
Parent-owned prompt changes in `generators.py` were left untouched by later work.
The browser microphone story run is parent-owned and was still in progress when
this report was updated; no result is asserted here.

## Per-run WAV Preservation

The harness now copies the stopped run's
`session.metadata.audio.artifact_paths.wav` to `output_dir/story_mix.wav`
immediately after confirming stopped state in `finally`. It streams the copy and
records SHA-256 under `summary.audio_artifact`:

```json
{"status": "copied", "source_path": ".../mixed.wav", "path": ".../story_mix.wav",
 "sha256": "...", "warning": null, "is_acoustic_capture": false}
```

Missing metadata/file, unconfirmed stop, or copy errors produce `status: unavailable`
and a diagnostic warning without changing acceptance criteria. Existing destination
files are not overwritten. This preserves future runs before runtime reset can
replace `mixed.wav`; no attempt was made to copy potentially overwritten earlier
recordings. The parent owns the final live run with this updated harness.

Verification: **57 passed in 1.11s** with
`env PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider`
on `tests/unit/experiments/test_story_acceptance.py` and
`tests/unit/experiments/test_rap_live_edit_acceptance.py`; `git diff --check` passed.
The last code change is harness-only; backend/telemetry/API are stable.

## Phase-challenged Acceptance

`story_acceptance.py` accepts `--edit-phase` in `[0,1)`, default zero. For example,
`--bpm 90 --edit-phase .75` targets tick 12: the trigger bar's playback callback
timestamp plus `.75 * (240 / 90)` seconds. Submissions wait for that boundary.
If the trigger bar has passed, the request is recorded as `late_not_submitted`
and the run fails explicitly instead of silently using a later bar.

Each event records requested phase, trigger callback, scheduled submission,
actual request offset/phase, phase error, and available milliseconds until the
next nominal bar boundary. A boundary crossing during HTTP setup is labeled
`late_submitted` and fails the run. These remain software callback timing proxies.
Default phase-zero behavior and WAV preservation are unchanged.

Phase verification: 64 harness tests passed in 1.32s. No application code was
changed for this extension. Parent owns the planned 90-BPM phase-.75 live run.

## Review Fixes: Delivery and Reset Races

Addressed the controller/transport portions of
`docs/reports/2026-11-05-interaction-review.md`:

- **P1 playback lock:** local mutation and acknowledgement capture occur under
  `_lock`; socket delivery occurs after releasing it. A separate `_plan_edit_lock`
  serializes topic/story control ordering without being acquired by ticks, Stop,
  Reset, or snapshots. Both initial and duplicate-retry sends follow this path.
- **P1 stale retry:** when a failed story edit has been overtaken by another plan
  edit, retry creates a fresh edit sequence carrying the complete current story
  ledger at the current editable boundary. Original event IDs/sequences/content
  remain unchanged and the response is still `duplicate: true`. Successful full
  snapshot delivery clears superseded pending story sends. No outbound queue or
  renderer protocol change was introduced.
- **P2 controller guard:** optional `expected_session_id` is checked under the
  mutation lock. The acknowledgement retains the original captured session even
  if Reset runs while delivery blocks. Pending-state completion is epoch-guarded.
  Parent owns the API and GUI session binding.
- **Cross-reset socket race:** `LyricStreamConnection.send_for_session(session_id,
  message)` verifies the expected session while capturing the socket under the
  transport lock, then releases that lock before socket I/O. A reset/open between
  controller checking and transport capture cannot redirect the edit to the new
  connection. Ordinary `send()` calls, including commit/tempo/stop messages, retain
  their existing interface and behavior.

Changed production paths for these review fixes only:
`application/rap/bar_stream.py`, `infrastructure/rap/lyric_stream_client.py`.
The test `_Stream` fixture gained the new session-bound method. New regressions:
`tests/unit/application/rap/test_story_delivery.py` and
`tests/unit/infrastructure/rap/test_story_transport.py`.

Verified blocked initial/retry delivery with concurrent snapshot/tick/Stop/Reset,
frozen acknowledgement identity, concurrent topic ordering, one/multiple failed
directions with intervening topic/flow edits, idempotent recovery, stale session
rejection, and the exact reset-between-epoch-check-and-socket-capture race.
Parent's `test_story_session_guard.py` also passes.

Result: **110 passed, 1 deselected in 5.89s** using the existing venv and
`PYTHONPATH=src`, across the new tests plus story, bar-stream, API, and lyric-stream
timing suites. The deselected existing test
`test_audio_get_recovers_real_http_disconnect_then_decodes_valid_audio` hit
`PermissionError: Operation not permitted` when binding its local HTTP fixture in
the sandbox; the preceding run had 109 passing tests and that one environmental
failure. `git diff --check` passed.

Ready for parent integration. The Mac client must load both changed modules;
no renderer change or restart is required for these fixes. No live services,
prompts, audio policies, or GPU jobs were changed by this task.
