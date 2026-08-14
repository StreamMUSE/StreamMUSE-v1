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

## Concerns

Task 6 intentionally does not wire the service into the coordinator, runtime,
CLI, or UI; those remain Tasks 7-9. The service delegates physical
bar-quantized completion and sample advancement to the Task 5 sink contract.
