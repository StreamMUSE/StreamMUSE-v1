# Task 9 Report: Audio Controls, Telemetry, And Warnings

## Delivered

- Lifespan starts the broadcaster for every runtime, but only starts runtimes
  with `autostart is True`. Text mode explicitly remains automatic and audio
  mode waits for Start.
- Added restart-safe Start, Stop, and Reset endpoints with `202`, `202`, and
  `200` responses, invalid-transition `409`s, and unsupported-control `404`s.
  Each restart receives a new runtime thread; permanent close remains limited
  to lifespan shutdown.
- Extended monitor and recorder projections with additive audio state, bounded
  warning records, and synthesis, bar-render, and commit-slack distributions.
- Added dense terminal AUDIO, WARN, and PLAY evidence plus dashboard playback
  and warning sections without removing prompt, candidate, score, flow,
  fallback, or syllable detail.
- Added only Start, Stop, and Reset runtime buttons to the monitor header,
  plus audio evidence and the compact warning table. Browser code only sends
  control requests; it does not schedule or render audio.

## Verification

```text
uv run pytest tests/unit/application/rap/test_monitoring.py \
  tests/unit/presentation/rap_demo -q
87 passed in 1.84s

uv run pytest tests/unit/domain/rap tests/unit/application/rap \
  tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
490 passed in 3.02s

uv run pytest tests/ -q --tb=no
Completed without test failures (4 environment skips)

uv run ruff check [Task 9 changed source and test paths]
All checks passed

git diff --check
no output
```

## Visual Notes

- Captured the current-worktree static monitor at 1440x900 and 390x844.
  Both checks had no page or header horizontal overflow; the mobile header
  retained all three runtime buttons without overlap.
- The normal monitor ports 8012 and 8013 were already occupied by unrelated
  local processes, so API behavior was verified through FastAPI tests and the
  visual layout was served temporarily from this worktree's static files.

## Commit

`feat: monitor and control realtime rap audio`

## Concerns

- No physical audio device, browser audio, H200, or SSH workflow was used.
- Static visual verification intentionally had no API backend; live endpoint
  behavior is covered by the focused server tests.

## Review Fix Round 1

- `SESSION_RESET` now clears the projected runtime/research epoch, including
  bars, candidates, ticks, syllables, errors, fallbacks, warnings, latency
  aggregates, and cumulative metrics. Static session configuration and audio
  device/recording configuration are retained.
- Bar render latency is sampled only from `AUDIO_RENDER_COMPLETED`; commit
  slack remains sampled only from `BAR_AUDIO_COMMITTED`.
- The renderer records per-syllable synthesis latency, renderer phonemes, and
  target samples in prepared diagnostics. The coordinator publishes those
  values in completed-render and warning payloads, including pronunciation,
  timing-pressure, forced-bar-fit, and synthesis-failed warnings.
- Playback notices now publish sink-derived `queue_depth` and
  `buffered_seconds`; the audio runtime publishes configured device and
  recording fields before playback starts. The monitor, terminal state, and
  dashboard tests cover non-placeholder producer-path values.
- Added concurrent HTTP duplicate Start/Stop coverage and a FastAPI lifecycle
  test against `RapAudioDemoDependencies`; it asserts one accepted transition,
  one `409`, no double start, restart, reset, and lifespan close behavior.
- The website files were not changed in this round. It still has exactly the
  existing compact Start, Stop, and Reset controls and the previous responsive
  layout verification remains applicable.

### Review Verification

```text
uv run pytest tests/unit/application/rap/test_monitoring.py \
  tests/unit/application/rap/test_playback.py \
  tests/unit/application/rap/test_audio_coordination.py \
  tests/unit/presentation/rap_demo/test_server.py \
  tests/unit/presentation/rap_demo/test_terminal_state.py \
  tests/unit/presentation/rap_demo/test_terminal_stream.py \
  tests/unit/presentation/rap_demo/test_terminal_dashboard.py -q
176 passed in 2.81s

uv run pytest tests/unit/domain/rap tests/unit/application/rap \
  tests/unit/infrastructure/rap tests/unit/presentation/rap_demo \
  tests/integration/test_realtime_rap_audio.py -q
499 passed in 3.86s

uv run pytest tests/ -q --tb=no
Completed without failures (4 environment skips)

uv run ruff check [review-fix source and test paths]
All checks passed

git diff --check
no output
```

### Review Commit

`fix: complete realtime rap audio telemetry`
