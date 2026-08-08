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

## Fix Round 2/5

### Changed Files

- `src/streammuse/application/rap/__init__.py`
- `src/streammuse/application/rap/monitoring.py`
- `src/streammuse/application/rap/realtime.py`
- `src/streammuse/application/rap/runtime.py`
- `src/streammuse/infrastructure/rap/__init__.py`
- `src/streammuse/infrastructure/rap/recorder.py`
- `tests/unit/application/rap/test_monitoring.py`
- `tests/unit/application/rap/test_realtime.py`
- `tests/unit/application/rap/test_runtime.py`
- `tests/unit/infrastructure/rap/test_recorder.py`
- `.superpowers/sdd/2026-08-07-research-realtime-rap/task-6-report.md`

### Findings Addressed

1. EOF recovery now accepts only parser-confirmed EOF truncation or a genuine
   unterminated string. Invalid pre-EOF tokens such as `NOT_JSON` raise rather
   than being hidden as crash tails.
2. The reproducibility manifest now contains full scenario ID/loop/tempo and
   contiguous segment schedule with topic/template/fallback lines. Tempo,
   template slots/provenance, identities, finite weights, threshold, timeout,
   and dimensions are deeply validated. Strict `allow_nan=False` JSON encoding
   is verified before directory creation.
3. `RapDemoDependencies` carries a validated positive
   `repetition_window_bars` value into `session_started`; recorder close passes
   the manifest window to derivation, which rejects event/manifest disagreement.
4. Worker results capture response completion immediately after generation and
   decision completion after analysis/ranking. Batch `deadline_slack_ms` and
   `late` are response metrics; separate decision completion/slack/late fields
   represent readiness. Replacement is blocked by `decision_late`, preserving
   the fallback when scoring misses the deadline.
5. Candidate counts must be nonnegative integers. Distinct evaluations beyond
   a declared batch count raise a clear contradiction error before summary or
   CSV output; bar rows also expose generator failures with no batch evidence.
6. The projector retains the latest request ID independently of pending status
   and ignores evaluations without that exact request ID.
7. Segment metadata retains the current and all future reserved bars; pruning
   happens only after a tick passes a bar, so large lookahead cannot evict the
   active segment.
8. Both package initializers now have one docstring, one import block, and one
   complete export list.

### TDD Evidence

The initial round-2 red suite produced the expected failures for stale
candidates, manifest depth, invalid EOF tokens, missing manifest-window
comparison, invalid/contradictory candidate counts, missing failure-only CSV
errors, and post-ranking timing fields.

```text
uv run pytest tests/unit/application/rap/test_monitoring.py \
  tests/unit/infrastructure/rap/test_recorder.py \
  tests/unit/application/rap/test_realtime.py -v
# 11 failures before implementation
```

The follow-up decision-gate and runtime-constructor RED run failed because a
decision-late result replaced the fallback and zero repetition windows were
accepted. Both regressions passed after using `decision_late` for replacement
and validating the dependency field in `__post_init__`.

Final verification:

```text
uv run pytest tests/unit/application/rap/test_monitoring.py \
  tests/unit/infrastructure/rap/test_recorder.py \
  tests/unit/application/rap/test_realtime.py \
  tests/unit/application/rap/test_runtime.py -v
# 40 passed

uv run pytest tests/unit/domain/rap tests/unit/application/rap \
  tests/unit/infrastructure/rap tests/unit/presentation/rap \
  tests/unit/presentation/rap_demo -q --tb=no
# 208 passed; one existing pretty_midi/pkg_resources deprecation warning
```

### Fix-Round Self-Review

- Manifest serialization and schema validation complete before any session
  directory mutation, preventing partial artifacts on bad research metadata.
- The response/decision split preserves accurate generator metrics while the
  decision deadline controls musical replacement safety.
- Repetition-window evidence is canonical at session start and checked against
  the manifest during recorder derivation and future regeneration.
- Current-request filtering and past-only segment pruning keep the bounded
  presentation state both truthful and usable with large lookahead values.

## Fix Round 3/5

### Changed Files

- `src/streammuse/infrastructure/rap/recorder.py`
- `tests/unit/infrastructure/rap/test_recorder.py`
- `.superpowers/sdd/2026-08-07-research-realtime-rap/task-6-report.md`

### Findings Addressed

1. Manifest validation now verifies the schedule as a cross-object contract:
   every segment template reference resolves to one unique recorded template,
   and each template meter exactly matches the session tempo.
2. Template records now require a nonempty display name; each slot must carry
   tick, duration, stress, boundary strength (integer 0 through 5), and a null
   or nonempty rhyme group. Provenance requires nonempty kind/source, a null or
   nonempty source hash, and finite nonnegative quantization error.
3. Score weights must contain exactly the six canonical `ScoreWeights`
   components, use finite nonnegative values, and sum to one within `1e-9`.
4. EOF recovery now forms candidate completions for true/false/null prefixes
   and incomplete decimal or exponent forms, appends outstanding container
   closers, and accepts a tail only after `json.loads` succeeds. Invalid token
   near-matches remain persisted-stream corruption and raise `ValueError`.

### TDD Evidence

The initial recorder-only RED run failed the seven valid scalar tail cases and
the eight new manifest cross-object/schema cases before implementation.

```text
uv run pytest tests/unit/infrastructure/rap/test_recorder.py -v
# 15 failures before implementation
```

Final verification:

```text
uv run pytest tests/unit/infrastructure/rap/test_recorder.py -v
# 36 passed

uv run pytest tests/unit/application/rap/test_monitoring.py \
  tests/unit/infrastructure/rap/test_recorder.py \
  tests/unit/application/rap/test_realtime.py \
  tests/unit/application/rap/test_runtime.py -v
# 61 passed

uv run pytest tests/unit/domain/rap tests/unit/application/rap \
  tests/unit/infrastructure/rap tests/unit/presentation/rap \
  tests/unit/presentation/rap_demo -q --tb=no
# 229 passed; one existing pretty_midi/pkg_resources deprecation warning
```

### Fix-Round Self-Review

- The manifest fixture is complete and cross-object negative tests cover
  unresolved template references, meter mismatch, missing/invalid slot and
  provenance fields, and both illegal score-weight names and totals.
- EOF recovery is deliberately parser-confirmed: an incomplete lexical token
  is recoverable only when completing it and all still-open containers yields
  valid JSON. This retains crash recovery without treating invalid token text
  as a recoverable final line.

## Fix Round 4/5

### Changed Files

- `src/streammuse/infrastructure/rap/recorder.py`
- `tests/unit/infrastructure/rap/test_recorder.py`
- `.superpowers/sdd/2026-08-07-research-realtime-rap/task-6-report.md`

### Finding Addressed

The completion-suffix heuristic was replaced by a recursive-descent JSON-prefix
recognizer. It follows the standard JSON grammar for nested objects and arrays,
whitespace, key and value strings, every simple escape, partial Unicode escapes,
literals, and every number state. EOF is recoverable only from a grammar state
that can still be extended into one valid document. Illegal escapes and control
characters, malformed structure and delimiters, literal near-matches,
leading-zero and other invalid number forms, non-ASCII number digits, and
trailing garbage are rejected.

`read_events()` still calls `json.loads()` first for every complete record. The
prefix recognizer is consulted only for a JSON decoding failure on the final
physical line when that line has no newline. Interior corruption, complete
final corruption, sequence checks, and session checks are unchanged. The
physical line is no longer stripped before either parser runs, preventing an
invalid partial token followed by whitespace from being reclassified as a
recoverable prefix.

### TDD Evidence

The initial focused RED failed during collection because `_is_json_prefix` did
not exist. After the parser implementation, a separate mixed-digit regression
proved that `str.isdigit()` incorrectly accepted `1\u0662`; replacing all number
states with ASCII digit checks made that test pass. A second RED showed that
trailing whitespace was stripped from invalid `tru ` and `1e ` tails; preserving
the complete physical line made both reject without changing empty-line checks.

The canonical-prefix test builds one actual recorder line with
`json.dumps(event_to_dict(event), sort_keys=True)`. Its payload includes escaped
quotes and backslashes, a Unicode escape, a negative exponent number, and a
boolean. Every nonempty strict prefix of that line is recognized as recoverable.
Focused cases cover all simple escapes, dangling escapes, partial `\\uXXXX`,
container and delimiter states, literal prefixes, bare minus, fraction and
exponent states, and representative invalid structural/string/number inputs.

### Verification

```text
uv run pytest tests/unit/infrastructure/rap/test_recorder.py -v
# 94 passed

uv run pytest tests/unit/application/rap/test_monitoring.py \
  tests/unit/infrastructure/rap/test_recorder.py \
  tests/unit/application/rap/test_realtime.py \
  tests/unit/application/rap/test_runtime.py -v
# 119 passed

uv run pytest tests/unit/domain/rap tests/unit/application/rap \
  tests/unit/infrastructure/rap tests/unit/presentation/rap \
  tests/unit/presentation/rap_demo -q --tb=no
# 287 passed; one existing pretty_midi/pkg_resources deprecation warning
```

### Fix-Round Self-Review

- Recovery scope remains limited to a non-newline final line that failed
  `json.loads()`; no interior or canonical event validation path was relaxed.
- The parser consumes the entire supplied prefix and has no completion guesses
  that can skip an invalid token or trailing character.
- This round changes no manifest, metric, controller, or runtime behavior.
