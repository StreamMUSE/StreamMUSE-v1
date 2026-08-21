# Task 6 Implementation Report

## Status

Implemented the Mac-side rolling two-bar controller, deterministic local eSpeak
fallback commitment, runtime controller protocol, dedicated chunk events, and
the `espeak|moss_aligned_remote` CLI switch. The existing `RollingRapController`
implementation and the default eSpeak composition remain unchanged.

## Owned Files

- Created `src/streammuse/application/rap/chunk_realtime.py`.
- Created `tests/unit/application/rap/test_chunk_realtime.py`.
- Modified `src/streammuse/application/rap/audio_service.py`.
- Modified `src/streammuse/application/rap/runtime.py`.
- Modified `src/streammuse/domain/rap/events.py`.
- Modified `src/streammuse/presentation/rap_demo/cli.py`.
- Modified `tests/unit/application/rap/test_runtime.py`.
- Modified Task 6-owned hunks in `tests/unit/presentation/rap_demo/test_cli.py`.
- Created this report.

Concurrent Task 4 server files, offline scripts, plans/specs, monitoring/UI
files, and unrelated dirty changes were not modified for Task 6.

## RED/GREEN Record

1. Baseline: the existing runtime, playback, coordinator, and CLI set passed
   with `81 passed` before Task 6 edits.
2. Controller RED: the new suite first failed collection because
   `chunk_realtime` did not exist, then produced 12 behavioral failures against
   a deliberate `NotImplementedError` stub.
3. Controller GREEN: startup, rolling commitment, deadlines, fallback,
   context, finite/looping, stop, reset, close, and event tests passed with
   `12 passed`.
4. CLI/runtime RED: 9 new failures with 40 existing passes proved the missing
   flags, validation, health gate, optional coordinator, and remote assembly.
5. CLI/runtime GREEN: `49 passed` after adding the mode-specific graph.
6. Nonblocking RED/GREEN: a manual preparation executor proved that a running
   tick had synchronously rendered the next fallback pair. Dedicated fallback
   staging changed the result to `13 passed` without inline tick rendering.
7. Deadline RED/GREEN: the test first observed a 4.000 second playback-boundary
   deadline instead of the existing guard tick. The corrected 120 BPM deadline
   is 3.875 seconds, and the controller suite passed.
8. Stale-preparation RED/GREEN: reset followed by execution of queued old-epoch
   fallback work initially emitted 8 stale events. Epoch checks now reject that
   work before publication or remote submission; `14 passed`.

All tests use a manual monotonic clock and manually completed futures for the
deadline and lifecycle branches. No sleep-based controller timing assertions
were added.

## Controller State Machine

The controller owns one useful remote pair and one optional local-preparation
staging future:

```text
IDLE
  start -> STARTUP_PREPARING -> pair 0 commit -> RUNNING

RUNNING
  poll only from on_tick
  final guard tick of current pair -> atomic next-pair decision/commit
  schedule local fallback staging -> submit one remote pair after both locals exist
  request_stop -> STOPPING

STOPPING
  cancel useful staging/remote wait
  retain an already committed successor, or atomically retain pending pair fallback
  resume_audio + resume_after_stop -> RUNNING
  terminal successor None -> reset required

RESET
  increment epoch first -> cancel/abort old work -> clear committed context -> IDLE

CLOSE
  increment epoch -> cancel -> abort -> stop fallback executor -> close strategy
  -> stop remote executor -> CLOSED
```

`start()` may wait on chunk 0 up to `rap_render_startup_timeout`. Running tick
callbacks call only `done()`/nonblocking `result()` on completed work and submit
fallback staging to its executor. They never wait on network, model, or eSpeak
work.

## Deadline Math

Startup chunk 0 captures one immutable deadline:

```text
startup_deadline = startup_started_monotonic + startup_timeout_seconds
```

Every later pair beginning at `start_bar` captures:

```text
commit_tick = start_bar * ticks_per_bar - 1
playback_guard_deadline = clock_origin + tempo.tick_to_seconds(commit_tick)
rolling_timeout_deadline = submission_monotonic + rolling_timeout_seconds
deadline = min(playback_guard_deadline, rolling_timeout_deadline)
```

At 120 BPM, the pair beginning at bar 2 commits on tick 31, so its playback
guard deadline is 3.875 seconds after the origin. At 90 BPM, the same guard is
about 5.167 seconds, so the default 5.0 second rolling timeout wins and leaves
about 0.167 seconds before the guard.

The worker records `completed_monotonic` immediately after strategy return.
Eligibility is strict: `completed_monotonic < deadline`. Completion exactly at
the deadline is late. The deadline is never recomputed. A pending worker at the
commit tick is rejected without waiting.

## Pair, Fallback, And Context Rules

- Every request contains exactly two consecutive `RemoteRapBarRequest` values.
- Finite nonzero limits must be even. Non-looping runs cannot exceed the
  scenario. Looping runs resolve each absolute bar through `segment_for_bar`.
- Both local fallback plans use the actual absolute bar topic and flow template.
- Both local eSpeak bars are fully rendered before the remote strategy is
  submitted.
- Remote identity, chunk index, renderer, bar numbers, and per-bar provenance
  are checked before acceptance. Task 5 remains authoritative for package,
  schedule, WAV, and Mac mixing validation.
- Failure, cancellation, invalid output, pending-at-guard, and late completion
  all select the already-ready local pair. Selection is atomic, but the two
  immutable `PreparedRapBar` values are enqueued in order so bar-quantized stop
  behavior remains intact.
- Context advances only after pair commitment and contains the last four
  committed lines. Rejected remote text is never appended. The next request is
  built only from that committed snapshot.
- Reset advances the epoch before clearing state. Old remote and local-staging
  results cannot emit, commit, enqueue, or seed later context.

## CLI Composition And Validation

Added:

```text
--rap-audio-renderer espeak|moss_aligned_remote  (default: espeak)
--rap-render-url http://127.0.0.1:8020
--rap-render-profile realtime
--rap-render-startup-timeout 120
--rap-render-rolling-timeout 5.0
```

Remote mode requires audio output, 48 kHz stereo float32 playback, local
`espeak-ng` fallback availability in the real factory path, `lookahead-bars=2`,
positive finite timeouts, a nonempty URL, and zero-or-even finite `max-bars`.
The `RemoteChunkClient.health()` schema check must succeed and report `ready`
before assembly returns.

The remote graph is:

```text
DeterministicRapBarRenderer (local fallback)
RemoteChunkClient -> RemoteMossChunkPreparationStrategy
RollingRapChunkController -> RapPlaybackService
```

It creates no `LocalChatModelClient`, including when the legacy `--generator`
flag is `local_chat`. The eSpeak graph still uses `RollingRapController`,
`BarAudioCoordinator`, the selected existing generator, and all existing audio
sink choices. `RapAudioDemoDependencies` is typed against `RapAudioController`
and accepts `coordinator=None` only for controller-owned remote strategy mode.

## Event Schema

Added bounded event types:

- `chunk_request_submitted`
- `chunk_remote_completed`
- `chunk_remote_rejected`
- `chunk_committed`
- `chunk_fallback_activated`

Chunk payloads contain state, renderer decision, chunk index, the two bar
numbers, at most two selected lines, two flow summaries, at most eight finite
stage timings, deadline slack, and at most eight warnings. They never contain
WAV bytes or candidate ledgers. Existing bar reserved/frozen/fallback,
audio-render/ready/committed, playback, tick, and playback-owned syllable events
remain active.

## Verification

Focused affected command:

```text
uv run pytest \
  tests/unit/application/rap/test_chunk_realtime.py \
  tests/unit/application/rap/test_runtime.py \
  tests/unit/application/rap/test_playback.py \
  tests/unit/application/rap/test_audio_coordination.py \
  tests/unit/presentation/rap_demo/test_cli.py -q
```

Final result after all controller lifecycle cases: `103 passed`.

Broader existing eSpeak/controller regressions:

```text
uv run pytest \
  tests/unit/application/rap/test_realtime.py \
  tests/integration/test_realtime_rap_audio.py \
  tests/unit/presentation/rap_demo/test_cli.py -q
```

Result: `70 passed`.

Final focused Ruff result: `All checks passed!`. Final `git diff --check`
completed with no errors.

## Fake-Service WAV Smoke

Ran the real CLI assembly, local eSpeak fallback renderer, Task 5 Mac validator
and mixer, timed playback sink, WAV recorder, event recorder, and lifecycle for
six bars at 90 BPM with `--audio-output wav --no-web`. A dependency-injected
fake chunk client scripted chunk decisions as remote success, remote failure,
remote success.

Verified:

- WAV: 768,000 frames at 48,000 Hz, exactly 16.0 seconds.
- Decisions: `moss_aligned_remote`, `prevalidated_fallback`,
  `moss_aligned_remote`.
- Playback underruns: 0.
- Bar completion: all six bars.
- Remote client closed: true.
- Artifact root:
  `/var/folders/kz/s5k2yj7911104rs_x8scxtb40000gn/T/streammuse-task6-smoke-e93oi55e`.

The first smoke execution completed all six bars but its verifier used Python's
standard `wave` reader, which cannot read IEEE float WAV format tag 3. Replacing
only the temporary verifier with `scipy.io.wavfile.read` produced the successful
result above; no production change was needed.

## Assumptions And Remaining Scope

- The existing deterministic eSpeak renderer remains capable of preparing a
  future pair within the currently playing pair. A local renderer exception is
  reported as `fallback_preparation_error`; there is no lower-quality fallback
  below eSpeak in Task 6.
- `realtime` is the only exposed remote profile in this vertical slice and maps
  to `RemoteCandidatePolicy.realtime_default()`.
- Task 7 remains responsible for dedicated chunk projection/UI presentation;
  Task 6 adds no visible controls and does not modify monitoring/UI files.
- The smoke validates the local boundary with a fake remote service. Real H200
  latency and perceptual quality remain Task 8 acceptance work.
