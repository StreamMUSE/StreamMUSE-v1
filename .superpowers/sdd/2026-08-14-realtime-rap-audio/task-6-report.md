# Task 6 Report: Rap Playback State Machine

## Scope

Implemented `RapPlaybackService` and its application export only. The service
owns playback lifecycle and performs all event publication and rolling tick
observation from a 5 ms daemon observer or explicit `poll()`, never from the
audio callback.

## Delivered

- Lifecycle: `prime`, `start`, `enqueue`, `request_stop`, `reset`, `wait`, and
  idempotent `close`.
- Canonical sink-notice mapping for bar start/completion, underrun, and device
  failure.
- Sample-clock tick catch-up through `on_tick` only; the service never
  publishes `TICK` events.
- Exact absolute sample metadata for `SYLLABLE_EMITTED`, including separate
  `observation_delay_ms` and `software_error_samples=0`.
- Bar-quantized stop delegation, stopped-only reset, epoch-fenced polling, and
  observer cancellation/join behavior across reset, close, and a concurrent
  start/close race.

## Tests

Baseline before Task 6:

```text
uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
410 passed in 1.95s
```

TDD red evidence:

```text
uv run pytest tests/unit/application/rap/test_playback.py -v
ModuleNotFoundError: streammuse.application.rap.playback

uv run pytest tests/unit/application/rap/test_playback.py::test_completed_bar_keeps_crossed_syllable_metadata_until_after_observation -v
FAILED: no syllable event after BAR_COMPLETED removed its prepared-bar metadata

uv run pytest tests/unit/application/rap/test_playback.py::test_close_during_start_does_not_join_an_unstarted_observer -v
FAILED: RuntimeError: cannot join thread before it is started
```

Final verification:

```text
uv run pytest tests/unit/application/rap/test_playback.py -q
11 passed in 0.31s

uv run ruff check src/streammuse/application/rap/playback.py tests/unit/application/rap/test_playback.py
All checks passed!

uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
421 passed in 1.93s
```

## Commits

- Base: `62706060` (`docs: record detached stream cleanup`)
- Implementation: `fecb91fd` (`feat: drive rap playback from the audio sample clock`)
- Review fixes: `eeef26f4` (`fix: harden rap playback stop and polling`)

## Review Remediation

The Task 6 review found four lifecycle and ordering gaps. The fixes remain
within playback; no coordinator, runtime, CLI, or UI wiring was added.

- After a bar-quantized stop, `prime` and `enqueue` accept only the completed
  bar's successor. The service resets the stopped Task 5 sink to discard its
  retained future queue, then offsets local sink samples to the successor's
  global musical bar. Public `reset()` clears that continuation state and again
  requires bar 0.
- A completion notice now raises the observation frame to the completed
  prepared bar's exact end even if the separately-read snapshot is older. The
  service observes final ticks and diagnostics before discarding metadata.
- `SESSION_STOPPED` is queued only after final tick and syllable effects.
- A dedicated poll mutex serializes sink reads, effect derivation, and callback
  execution without holding the lifecycle lock across callbacks.

Review TDD red evidence:

```text
test_restart_after_stop_discards_stale_future_bars_and_starts_next_complete_bar
FAILED: prime requires prepared bar 0

test_completion_notice_advances_observation_past_an_older_snapshot_before_metadata_removal
FAILED: no SYLLABLE_EMITTED event after a newer BAR_COMPLETED notice

test_interleaved_polls_do_not_derive_or_execute_effects_concurrently
FAILED: the second poll read its snapshot while the first poll callback blocked

test_enqueue_after_stop_discards_stale_future_bars_and_primes_the_next_bar
FAILED: the first prepared bar must be bar 0

test_reset_after_stop_requires_a_fresh_bar_zero_session
FAILED: DID NOT RAISE ValueError
```

Review-fix verification:

```text
uv run pytest tests/unit/application/rap/test_playback.py -q
16 passed in 0.50s

uv run ruff check src/streammuse/application/rap/playback.py tests/unit/application/rap/test_playback.py
All checks passed!

uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
426 passed in 2.10s
```

## Review Round Two

Round two keeps the poll mutex across effect delivery so tick callbacks and
publishers remain strictly ordered and non-concurrent. A close invoked from
that dispatch owner now closes the sink but defers observer joining until the
poll releases the mutex; a normal external close still joins the observer
before returning.

- Stop during `PRIMING` now resets the sink and clears prepared/session timing
  state before returning `STOPPED`, so a fresh bar 0 can be primed without a
  stale queued bar.
- Notice payload frames capture the continuation sample origin during effect
  derivation, so bar start/completion and underrun records use the same global
  coordinate as ticks and syllables.
- The existing stop-ordering regression remains in the focused suite.

Round-two TDD red evidence:

```text
test_stop_during_priming_discards_queued_bar_and_allows_fresh_prime
FAILED: expected sink.reset_calls == 1, got 0

test_close_from_tick_callback_defers_observer_join_until_poll_releases_its_mutex
FAILED: observer join was attempted while the callback owned dispatch

test_continuation_notice_payloads_use_global_absolute_samples
FAILED: notice payload frames were [96, 96, 96], not [192096, 192096, 192096]
```

Round-two verification:

```text
uv run pytest tests/unit/application/rap/test_playback.py -q
19 passed in 0.52s

uv run ruff check src/streammuse/application/rap/playback.py tests/unit/application/rap/test_playback.py
All checks passed!

uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
429 passed in 2.00s
```

## Concerns

Task 6 intentionally does not wire the service into the coordinator, runtime,
CLI, or UI; those remain Tasks 7-9. The service delegates physical
bar-quantized completion and sample advancement to the Task 5 sink contract.
