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
