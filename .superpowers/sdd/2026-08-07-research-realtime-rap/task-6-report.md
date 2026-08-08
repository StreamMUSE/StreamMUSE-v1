# Task 6 Implementation Report: Canonical Rap Events, Recorder, and Metrics

## Status

Implemented the canonical event dispatch, presentation-agnostic state projection,
crash-tolerant session recorder, deterministic artifact derivations, and focused
behavior coverage. Existing `RollingRapController` and terminal-demo behavior
remain unchanged.

## Changed Files

- `src/streammuse/application/rap/__init__.py`
- `src/streammuse/application/rap/monitoring.py`
- `src/streammuse/infrastructure/rap/__init__.py`
- `src/streammuse/infrastructure/rap/recorder.py`
- `tests/unit/application/rap/test_monitoring.py`
- `tests/unit/infrastructure/rap/test_recorder.py`
- `.superpowers/sdd/2026-08-07-research-realtime-rap/task-6-report.md`

`src/streammuse/domain/rap/events.py` already contained the complete required
canonical `RapEventType` vocabulary and frozen `RapEvent` record, so it was
intentionally retained rather than duplicated or renamed.

## Implementation

- `RapEventPublisher.emit()` now assigns sequence, timestamps, and performs the
  queue put under the same lock. This makes FIFO queue order identical to
  canonical sequence order across concurrent producers.
- `RapEventDispatcher` continues to run all sinks off the musical clock on one
  worker thread. It disables a sink after an exception and emits one canonical
  `presentation_error` event to the remaining sinks without recursive failure
  publication. Close drains those failure events even when they were appended
  behind the close sentinel.
- `RapStateProjector` is a callable sink with locked `apply()` and deep-copying
  `snapshot()`. It contains only JSON-compatible primitive state: tick,
  segment, pending request, candidate table, frozen bars, emitted syllables,
  latency aggregates, and fallback counters.
- `RapSessionRecorder` creates an exclusive session directory, writes a
  recursively redacted deterministic manifest, streams sorted JSONL records,
  ignores incomplete/corrupt JSONL lines during recovery, and generates
  `summary.json` and `bars.csv` after close.
- Redaction normalizes each manifest key and replaces the associated value when
  the normalized name contains `key`, `token`, `secret`, or `authorization`.
- Bar rows are sorted numerically by bar and use the larger of the batch's
  declared candidate count and its observed evaluation count, avoiding double
  counting when both evidence sources are present.

## Metric Definitions

All ratios contain `numerator`, `denominator`, and `rate`; a zero denominator
produces `null` rate.

- Candidate validity: valid `candidate_evaluated` events / all evaluated
  candidates.
- Fallback: frozen bars whose `fallback` is true / all frozen bars.
- Deadline miss: frozen bars with `fallback_reason == "deadline_miss"` / all
  frozen bars.
- Generator error: candidate-batch events with a nonempty `error_type` / all
  candidate-batch events.
- Pronunciation fallback: word-analysis entries not sourced from
  `cmudict_first_pronunciation` / all word-analysis entries.
- Repetition: candidate evaluations rejected for
  `duplicate_normalized_text` / all evaluated candidates.
- Generation latency and deadline slack are taken from candidate-batch payloads;
  emission jitter is taken from syllable-emitted payloads. Each records count,
  linear-interpolated p50/p95, and max, or `null` percentiles/max with no
  observations.

## TDD Evidence

Initial RED run:

```text
uv run pytest tests/unit/application/rap/test_monitoring.py \
  tests/unit/infrastructure/rap/test_recorder.py -v
# collection failed as intended: RapStateProjector and recorder module missing
```

The atomic ordering test uses a deliberately blocking first queue put. The
previous sequence-then-unlocked-put implementation permits sequence two to
overtake sequence one; the lock now covers both operations.

Review-regression RED run:

```text
uv run pytest tests/unit/infrastructure/rap/test_recorder.py::\
test_bar_rows_do_not_double_count_declared_and_evaluated_candidates -v
# failed as intended: candidate_count was 2, expected 1
```

After separating declared and observed counts, the focused GREEN run was:

```text
uv run pytest tests/unit/application/rap/test_monitoring.py \
  tests/unit/infrastructure/rap/test_recorder.py -v
# 8 passed
```

Affected RAP regression verification:

```text
uv run pytest tests/unit/domain/rap tests/unit/application/rap \
  tests/unit/infrastructure/rap tests/unit/presentation/rap \
  tests/unit/presentation/rap_demo -q --tb=no
# 187 passed; one existing pretty_midi/pkg_resources deprecation warning
```

## Self-Review

- Event vocabulary remains singular in `domain.rap.events`; no parallel record
  type or enum was introduced.
- The dispatch worker owns sink execution, so recorder and presentation I/O do
  not run on the controller's musical-clock path.
- Sink failure disables the failed sink and does not recursively invoke it with
  the generated error event.
- Snapshots cannot alter projected state because the returned structure is a
  deep copy under the projector lock.
- Recorder outputs are derived solely from canonical events, with stable JSON
  key ordering and sorted bar rows.
- Missing metric observations remain `null`, never misleading zero values.

## Concerns

- The running controller does not yet construct or wire a full required
  session manifest into the recorder; this foundation accepts and redacts the
  manifest supplied by a later integration task.
- The affected suite has one pre-existing `pretty_midi` warning about
  `pkg_resources` deprecation.

## Fix Round 1/5

### Changed Files

- `src/streammuse/application/rap/__init__.py`
- `src/streammuse/application/rap/monitoring.py`
- `src/streammuse/application/rap/realtime.py`
- `src/streammuse/infrastructure/rap/__init__.py`
- `src/streammuse/infrastructure/rap/recorder.py`
- `tests/unit/application/rap/test_monitoring.py`
- `tests/unit/application/rap/test_realtime.py`
- `tests/unit/infrastructure/rap/test_recorder.py`
- `.superpowers/sdd/2026-08-07-research-realtime-rap/task-6-report.md`

### Findings Addressed

1. `flush_and_close()` now closes publication under the publisher lock. An
   emitter that acquired the lock first queues before the sentinel; emitters
   arriving later receive a closed-publisher error. Dispatcher-owned
   presentation errors use a private canonical path after close and remain
   non-recursive.
2. Summary deadline-miss and generator-error denominators are distinct planning
   request IDs. Late batches and generator failures are deduplicated by request
   ID, so a batch error plus `generation_failed` pair counts once.
3. Candidate validity uses each distinct batch's parsed `candidate_count`; when
   that value is absent, distinct `(request_id, candidate_id)` evaluations are
   the fallback denominator. Bar rows apply the same evidence identity rules.
4. Repetition is normalized bigrams repeated against the prior configured
   frozen-bar window, divided by all frozen-bar bigrams. The event-stream
   `session_started.repetition_window_bars` controls it; legacy streams use the
   documented default of four bars.
5. Added `RapSessionManifest`, `build_session_manifest`, and
   `validate_session_manifest`. The recorder rejects incomplete manifests and
   requires scenario/seed/tempo/templates with provenance/generator and model
   configuration/weights/threshold/timeout/lookahead/runtime versions/package
   version/git revision/dirty state/window metadata.
   Validation occurs before the session directory is created, so a rejected
   manifest leaves no partial artifact directory.
6. Manifest and event payload recording use one recursive key-name redaction
   boundary for `key`, `token`, `secret`, and `authorization` names.
7. JSONL recovery accepts only an actually truncated final JSON container.
   Interior corruption, blank records, malformed complete final records
   (including without a trailing newline), mixed sessions, and non-contiguous
   sequences now raise `ValueError` with line context.
8. `bars.csv` derivation now returns rows only for bars with `bar_frozen`
   evidence.
9. `RapStateProjector` accepts positive live-view limits, keeps bounded current
   candidates/recent bars/recent syllables, retains cumulative aggregates,
   clears candidates for each new request, clears pending work after failures,
   and only changes current segment when a tick reaches that bar.
10. Publisher payload copies are deep, so post-emit nested caller mutations
    cannot alter queued evidence.
11. `RollingRapController` captures worker completion monotonic time and emits
    deadline slack as planned target-bar start minus that completion time. Batch
    `late` is derived from nonpositive slack or an already-frozen safety check,
    rather than the last observed tick alone.
12. Application and infrastructure `__all__` lists restore all prior public
    names and append the monitoring/recorder APIs.

### TDD Evidence

Initial fix-round RED:

```text
uv run pytest tests/unit/application/rap/test_monitoring.py \
  tests/unit/infrastructure/rap/test_recorder.py \
  tests/unit/application/rap/test_realtime.py -v
# collection failed as intended: complete manifest APIs missing
```

The first implementation pass then exposed two stale assertions that still
expected non-frozen CSV rows and pre-repetition fixture text; those were
corrected to the approved artifact contract. A focused run then passed 29
tests.

Additional clarification REDs:

```text
test_state_projector_replaces_candidates_when_a_new_request_starts
# failed: ['old', 'new'] retained across requests

test_read_events_rejects_interior_corruption_complete_final_corruption_and_mixed_sessions
# failed: a complete '{bad}' final line without newline was silently ignored
```

The final focused verification was:

```text
uv run pytest tests/unit/application/rap/test_monitoring.py \
  tests/unit/infrastructure/rap/test_recorder.py \
  tests/unit/application/rap/test_realtime.py -v
# 30 passed
```

Affected compatibility verification was:

```text
uv run pytest tests/unit/domain/rap tests/unit/application/rap \
  tests/unit/infrastructure/rap tests/unit/presentation/rap \
  tests/unit/presentation/rap_demo -q --tb=no
# 200 passed; one existing pretty_midi/pkg_resources deprecation warning
```

### Fix-Round Self-Review

- Close ordering is tested with a blocked timestamp callback, which makes the
  former sentinel race deterministic.
- Strict persisted-stream checks and defensive in-memory derivation deduplication
  are both present; valid evidence cannot be inflated by repeated request or
  candidate records.
- Every persisted value crossing the recorder boundary is recursively redacted
  before JSON encoding.
- Deadline slack and late use the same captured completion value, and summary
  derivation consumes that event evidence directly.
- No terminal/controller behavior was changed beyond the new event payload
  timing evidence; the complete affected RAP/terminal suite remains green.
