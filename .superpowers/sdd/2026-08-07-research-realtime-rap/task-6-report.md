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
