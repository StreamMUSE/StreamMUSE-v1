# Task 6 Implementation And Review-Fix Report

## Status

Task 6 is implemented and the findings against rejected commit `0af9f57c` are
fixed. The Mac side now supports the existing eSpeak controller or a rolling
two-bar remote MOSS controller selected by CLI. Remote mode always prepares a
complete local eSpeak pair before authorizing remote work, commits at pair
granularity while enqueuing `PreparedRapBar` values individually, and preserves
Start, Stop, and Reset as the only controls.

The existing `RollingRapController`, eSpeak defaults, playback/audio path,
server code, monitoring/projection code, and visible web controls were not
changed by the review fix.

## Owned Scope

The original Task 6 commit owns:

- `src/streammuse/application/rap/audio_service.py`
- `src/streammuse/application/rap/chunk_realtime.py`
- `src/streammuse/application/rap/runtime.py`
- `src/streammuse/domain/rap/events.py`
- `src/streammuse/presentation/rap_demo/cli.py`
- `tests/unit/application/rap/test_chunk_realtime.py`
- `tests/unit/application/rap/test_runtime.py`
- Task 6 hunks in `tests/unit/presentation/rap_demo/test_cli.py`
- this report

The review-fix commit changes only the chunk controller, shared audio runtime,
their two test files, and this report. Concurrent Task 3/7 work and unrelated
dirty files were left untouched.

## RED/GREEN Record

### Original vertical slice

1. Baseline runtime/playback/coordinator/CLI tests: `81 passed`.
2. Controller RED: missing module, followed by 12 behavioral failures against
   a deliberate stub.
3. Controller GREEN: startup, rolling commitment, deadlines, fallback,
   context, finite/looping, stop, reset, close, and events: `12 passed`.
4. CLI/runtime RED: 9 new failures with 40 existing passes for the missing
   flags, validation, health gate, optional coordinator, and remote graph.
5. CLI/runtime GREEN: `49 passed`.
6. Nonblocking local preparation RED/GREEN: a manual preparation executor
   proved tick-time fallback rendering, then asynchronous staging produced
   `13 passed`.
7. Initial stale-preparation RED/GREEN: queued old fallback work emitted eight
   stale events after reset, then epoch checks produced `14 passed`.

### Rejected-commit corrections

All review regressions were added before the corresponding production edits.
The corrected RED run was `17 failed, 30 passed` and reproduced the reviewed
behavior deterministically:

- Local pair 4-5 completed after guard tick 63, but tick 64 left the queue at
  bars 0-3 and stranded the pending pair.
- A blocking `strategy.abort()` kept the tick observer blocked for 250 ms and
  bars 2-3 were not enqueued until abort returned.
- Completing a queued old remote future after reset, stop, or close increased
  strategy calls from one to two.
- Reset returned while request construction was deliberately blocked after the
  old epoch check; the old request then submitted and published afterward.
- `object()`, `None`, and a mapping returned by the strategy raised
  `AttributeError` in the tick path before fallback selection.
- Stop during pair 4-5 local staging discarded the successor and made
  `resume_audio(4)` raise.
- Pair 2-3 rendered synchronously in `start()` and captured its deadline before
  the simulated enqueue handoff.
- Looping and shorter-than-scenario finite caps left
  `restart_requires_reset == False`.
- 100 KB selected lines, warnings, timing keys, and exception messages escaped
  dedicated event bounds.

The first GREEN run was `47 passed`. A second lock-exit barrier test then
isolated the final check-to-call window between remote authorization and
`strategy.prepare()`; it failed because reset returned before that worker
handoff. A second execution-boundary check plus cancellation fencing made that
test green. Final focused result: `48 passed`.

The tests use a manual monotonic clock, manual remote/local/cancellation
executors, and explicit barriers. The only short real-time wait is used to
prove that a lifecycle transition cannot return through a deliberately paused
execution handoff.

## Controller State Machine

Only one future pair exists. It moves through these states:

```text
IDLE
  Start
    -> STARTUP_LOCAL_READY (render bars 0-1 synchronously)
    -> STARTUP_REMOTE_PENDING (wait only up to startup timeout)
    -> COMMITTED (remote or both local fallbacks)
    -> enqueue bars 0-1
    -> establish playback clock origin
    -> LOCAL_STAGING for bars 2-3

RUNNING / LOCAL_STAGING
  both fallbacks ready before immutable deadline and abort barrier clear
    -> atomically authorize and submit REMOTE_PENDING
  fallbacks ready at/after deadline
    -> retain ready pair without remote submission
  observed tick >= pair guard
    -> commit retained fallbacks immediately; never strand staging

RUNNING / REMOTE_PENDING
  valid completion strictly before deadline
    -> REMOTE_READY
  failure, malformed value, invalid chunk, or late completion
    -> FALLBACK_READY
  observed tick >= pair guard while still pending
    -> revoke remote authorization and select fallback without waiting
  pair decision
    -> atomically commit both bars, enqueue each bar, stage one successor pair

CANCELLING
  fallback is committed and delivered first
  abort runs on the cancellation executor
  successor local staging may run, but successor remote authorization waits for
  the abort barrier so an old global abort cannot cancel a new request

STOPPING
  revoke/cancel remote waiting
  preserve completed successor, commit pending successor fallback, or allow the
  exact successor local staging task to finish and retain its fallback pair
  Resume -> enqueue retained successor/trailing pair bar and rebase clock
  terminal successor None -> Reset required

RESET/CLOSE
  increment epoch and revoke authorization under the state lock
  old queued workers fail at the actual remote execution boundary
  active workers are aborted and fenced before reset returns or close finishes
```

Request construction, executor submission, pending-state installation, and the
`chunk_request_submitted` event share one locked handoff. A remote worker checks
epoch, pending identity, authorization, stop, and close state twice immediately
before calling the strategy. Reset/stop cancellation does not return while a
worker that crossed the first check can still begin later. Old workers cannot
submit, call `strategy.prepare()`, publish, occupy the new session, commit, or
change context after the lifecycle transition returns.

## Deadline And Clock Math

Startup captures one immutable deadline before synchronous fallback work:

```text
startup_deadline = startup_started_monotonic + startup_timeout_seconds
```

The startup remote wait is bounded by the remaining duration to that deadline.
Regardless of outcome, both startup bars are committed and enqueued before the
rolling musical origin is recorded.

Every later pair beginning at `start_bar` captures:

```text
guard_tick = start_bar * ticks_per_bar - 1
playback_guard = clock_origin + tempo.tick_to_seconds(guard_tick)
timeout_guard = staging_started_monotonic + rolling_timeout_seconds
immutable_deadline = min(playback_guard, timeout_guard)
```

At 120 BPM, pair 2-3 has guard tick 31, 3.875 seconds after the playback
handoff. The enqueue-handoff regression advances the manual clock by one second
while bars 0-1 are delivered and proves pair 2-3 receives deadline `4.875` and
an initial `3,875 ms` request budget. Pair 2 local rendering is asynchronous,
so it consumes playing-pair lookahead rather than delaying playback startup.

Completion eligibility is strict:

```text
completed_monotonic < immutable_deadline
```

Completion at the deadline is late. Tick commitment uses `tick >= guard_tick`,
so fallback staging completed exactly at the guard commits on that observation,
and staging completed afterward commits on the next observed tick.

## Fallback, Queue, And Context Rules

- Every request owns exactly two consecutive absolute bars.
- Both fallback plans use the actual scenario topic and flow template, and both
  local eSpeak bars finish before remote submission.
- Strategy return type is checked before any renderer, bars, or diagnostics
  field access. Arbitrary values reject to the ready fallback pair.
- Remote identity, chunk index, renderer, bars, and provenance are validated
  before acceptance. Task 5 remains authoritative for package, WAV, schedule,
  resampling, and Mac mixing validation.
- Pair selection and context mutation are atomic. Enqueue remains bar-granular
  and ordered, preserving stop-after-current-bar behavior.
- Context advances only from committed full-text bars. Late, failed, invalid,
  canceled, and stale remote text never reaches later request context.
- Stop never discards local staging needed by its actual successor. A resume can
  wait for that local task and uses the retained immutable bar.

## Finite Runtime

`RollingRapChunkController.terminal_bar_limit` exposes the resolved finite
remote planning cap to `RapAudioDemoDependencies`. Reaching that cap passes
`successor_bar=None`, marks `restart_requires_reset`, and makes Start reject
until Reset. This applies to looping scenarios and caps shorter than a
non-looping scenario. Controllers without this remote property retain the
existing eSpeak runtime behavior.

## CLI Composition And Validation

The public switch remains:

```text
--rap-audio-renderer espeak|moss_aligned_remote  (default: espeak)
--rap-render-url http://127.0.0.1:8020
--rap-render-profile realtime
--rap-render-startup-timeout 120
--rap-render-rolling-timeout 5.0
```

Remote mode requires audio output, 48 kHz stereo float32 audio, local eSpeak
fallback availability in the real factory path, two-bar lookahead, positive
finite timeouts, a nonempty URL, successful schema-compatible health, and zero
or an even finite `max-bars`. It creates no `LocalChatModelClient`, including
when the legacy generator flag says `local_chat`.

The eSpeak graph is unchanged. The remote graph is:

```text
DeterministicRapBarRenderer (local eSpeak fallback)
RemoteChunkClient -> RemoteMossChunkPreparationStrategy
RollingRapChunkController -> RapPlaybackService
```

## Event Schema And Bounds

Dedicated event types remain:

- `chunk_request_submitted`
- `chunk_remote_completed`
- `chunk_remote_rejected`
- `chunk_committed`
- `chunk_fallback_activated`

Their payload has exactly the bounded summary fields: state, renderer decision,
chunk index, two bar numbers, up to two selected lines, two flow summaries,
stage timings, deadline slack, and warnings. Bounds are:

- selected line: 512 UTF-8 bytes
- warning/error: 256 UTF-8 bytes
- state/timing key: 64 UTF-8 bytes
- warnings and finite timing entries: at most 8 each
- flow slots: at most 32 per flow
- nested maps: fixed keys, no candidate ledger or arbitrary diagnostics

Tests also enforce at most 16 mapping entries, at most 32 sequence entries,
maximum depth 6, no bytes, and a compact serialized payload for adversarial
100 KB input. Existing bar, playback, tick, and syllable events remain active.

## Verification

Binding affected command:

```text
uv run pytest \
  tests/unit/application/rap/test_chunk_realtime.py \
  tests/unit/application/rap/test_runtime.py \
  tests/unit/application/rap/test_playback.py \
  tests/unit/application/rap/test_audio_coordination.py \
  tests/unit/presentation/rap_demo/test_cli.py -q
```

Final review-fix result: `120 passed in 2.24s`.

Adjacent existing eSpeak/controller regressions:

```text
uv run pytest \
  tests/unit/application/rap/test_realtime.py \
  tests/integration/test_realtime_rap_audio.py \
  tests/unit/presentation/rap_demo/test_cli.py -q
```

Final review-fix result: `70 passed in 2.40s`.

Focused controller/runtime result: `48 passed in 0.98s`.

Full Task 6 file-surface Ruff: `All checks passed!`. Owned production, test,
and report `git diff --check` completed without errors.

## Fake-Service WAV Smoke

The local six-bar smoke used real CLI assembly, local eSpeak fallback rendering,
Task 5 Mac validation/mixing, timed playback, float32 WAV recording, event
recording, and runtime lifecycle. Only the remote transport was injected with
success/failure/success outcomes.

Verified at 90 BPM:

- WAV: 48 kHz, stereo float32, 768,000 frames, exactly 16.0 seconds.
- Renderer decisions: `moss_aligned_remote`, `prevalidated_fallback`,
  `moss_aligned_remote`.
- Scripted transport: success, failure, success.
- Completed bars: 0 through 5.
- Playback underruns: 0.
- Remote client closed: true.

Retained evidence:

- Artifact root:
  `/var/folders/kz/s5k2yj7911104rs_x8scxtb40000gn/T/streammuse-task6-fix-smoke-k0pzb4k2`
- Summary: `smoke-summary.json`
- WAV: `six-bars.wav`
- Canonical event ledger:
  `logs/rap-20260821T082250Z-b5d9186f/events.jsonl`
- Session manifest, summary, and bar CSV are in the same session directory.

## Assumptions And Remaining Concerns

- Deterministic eSpeak is the final local fallback. A local renderer failure is
  bounded and reported as `fallback_preparation_error`; Task 6 has no lower
  quality audio renderer.
- The strategy contract requires `abort()` to terminate an active useful wait.
  Cancellation occurs away from tick observation and is fenced before a new
  remote request can start.
- `realtime` remains the only exposed remote policy profile.
- The fake-service smoke validates the complete local boundary. Real H200
  latency, availability, and perceptual quality remain Task 8 acceptance work.
