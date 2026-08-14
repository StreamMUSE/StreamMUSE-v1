# Task 5 Report: Realtime Rap Audio Output Adapters

## Scope

Implemented the Task 5 audio-output adapters only. No runtime, CLI, web, or
default-setting wiring was added.

## Changed Files

- `src/streammuse/infrastructure/rap/audio_output.py`
  - Added `SoundDeviceAudioSink` with injected stream construction, callback
    queue/offset playback, silence fill, status underrun notices, and
    stop-after-current-bar behavior.
  - Added `Float32WavAudioSink`, which writes a 44-byte little-endian IEEE
    float WAV header and records only bars confirmed as completed.
  - Added `CompositeAudioSink`, which commits completed live bars to the WAV
    recorder from playback notices.
  - Added `NullAudioSink` for device-free deterministic tests.
  - Added `TimedAudioSink`, the device-free realtime primary sink required for
    future standalone WAV output. It advances the absolute sample snapshot
    from a monotonic clock and supports start, stop-after-current-bar, reset,
    and close without PortAudio.
- `src/streammuse/infrastructure/rap/__init__.py`
  - Exported the five audio sinks.
- `tests/unit/infrastructure/rap/test_audio_output.py`
  - Added focused sink coverage for arbitrary callback block boundaries,
    silence/status behavior, bar-quantized stop, IEEE float WAV output,
    composite completion commit, null playback, and timed clock/stop/reset.
- `pyproject.toml` and `uv.lock`
  - Added and locked `sounddevice>=0.5.2` (`0.5.5` resolved).

## Design Choices

- `sounddevice` is imported only inside the default stream factory. Importing
  the adapter module and all text-only paths remains device independent.
- The PortAudio callback copies PCM without holding queue or state locks. It
  takes a short queue lock only when acquiring the next immutable bar and uses
  a `SimpleQueue` for notices.
- WAV bytes are appended only after `BAR_COMPLETED`; queued or unplayed bars
  cannot enter the finalized artifact.
- The timed primary sink exists solely for the ledger ruling on future
  `--audio-output wav` assembly. This task intentionally does not wire that
  choice into the CLI or runtime.

## Verification

```text
uv run pytest tests/unit/infrastructure/rap/test_audio_output.py -q
7 passed in 0.43s

uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
395 passed in 2.01s

git diff --check
(no output)
```

## Commits

- `561a2ecb feat: add realtime rap audio outputs`

## Concerns

- The injected fake stream intentionally avoids opening a physical audio
  device. A later runtime/acceptance task should exercise a real CoreAudio
  device and observe device-failure reporting; physical speaker latency remains
  outside the software sample-placement guarantee.

## Review Fix Round

The Task 5 review findings were addressed without beginning Task 6.

- Callback stop now raises an injected callback-termination exception. The
  default factory lazily creates `sounddevice.CallbackStop`; the callback no
  longer invokes `stream.stop()` or other device control operations.
- Prepared-bar acquisition now holds the short state and queue critical
  sections through active-bar assignment and `BAR_STARTED` publication, so a
  concurrent Stop observes either the active bar or the untouched queue.
- Empty prepared queues emit one `UNDERRUN` notice and increment the count per
  starvation episode. Acquiring a bar clears the episode, allowing a later
  shortage to be observed without one notice per callback block.
- `CompositeAudioSink` retains primary notices while it commits completed bars
  to the WAV recorder, including when close performs the final drain.
- Null and timed primary sinks now stop immediately when no bar is active and
  leave queued bars untouched.
- Stream-factory and stream-start exceptions convert to `DEVICE_FAILED` notices
  with a `STOPPED` snapshot instead of escaping from `start()`.
- `uv.lock` was restored to lock revision 3. Relative to the Task 4 base it
  contains only the 18 required `sounddevice` entries; `uv lock --locked`
  validates the result.

## Fix Round Verification

```text
uv lock --locked
Resolved 195 packages in 2ms

uv run pytest tests/unit/infrastructure/rap/test_audio_output.py -q
15 passed in 0.52s

uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
403 passed in 1.97s

git diff --check
(no output)
```

## Fix Round Commit

- `e3724294 fix: harden realtime rap audio sinks`

## Fix Round 2

The final two Task 5 lifecycle defects were addressed without beginning Task 6.

- `SoundDeviceAudioSink.start()` now validates `CLOSED` before entering device
  failure handling. A closed sink remains terminal: both `start()` and
  `enqueue()` raise without publishing `DEVICE_FAILED` or changing the
  snapshot state.
- `SoundDeviceAudioSink` now increments an epoch on reset and close. Each
  callback captures that epoch, and all subsequent frame-offset commits and
  notice publication verify it. PCM assignment remains lock-free; a callback
  that started before reset can finish its assignment but cannot mutate the
  reset state or add stale notices.
- Reviewed sibling sink terminal behavior: Float32 WAV rejects operations after
  close, and Null/Timed preserve `CLOSED` across reset while rejecting start or
  enqueue. Composite delegates lifecycle operations to those terminal sinks.

## Fix Round 2 Verification

```text
uv run pytest tests/unit/infrastructure/rap/test_audio_output.py -q
17 passed in 0.51s

uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
405 passed in 1.85s

git diff --check
(no output)
```

## Fix Round 2 Commit

- `12bd2046 fix: guard rap audio sink lifecycle`

## Fix Round 3

The final `start()` transition race was addressed without beginning Task 6.

- `SoundDeviceAudioSink.start()` captures the lifecycle epoch before stream
  construction and checks it again after `stream.start()` returns. If reset,
  close, or a stop transition canceled that generation while start was in
  flight, the sink immediately stops and closes the physical stream without
  changing the newer logical state.
- Device start failures still publish `DEVICE_FAILED` for the active lifecycle,
  including after a completed reset. Failures belonging to a canceled reset are
  suppressed, and a concurrent close remains terminal rather than being
  overwritten.
- Deterministic fake streams block inside `start()` while close or reset
  completes. Tests verify that releasing the start leaves the stream inactive
  and closed with the expected `CLOSED` or `STOPPED` snapshot.

## Fix Round 3 Verification

```text
uv run pytest tests/unit/infrastructure/rap/test_audio_output.py -q
20 passed in 0.49s

uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
408 passed in 1.98s

git diff --check
(no output)
```

## Fix Round 3 Commit

- `81fd1a4c fix: cancel stale rap audio stream starts`

## Fix Round 4

The remaining reset/start ABA race was addressed without beginning Task 6.

- `SoundDeviceAudioSink.reset()` now atomically detaches `_stream` before it
  advances the lifecycle epoch and stops the old device. The following start
  must construct a distinct stream instance.
- A stale start retains only its local stream reference. Its cancellation
  cleanup can stop and close that detached stream but cannot touch a later
  stream owned by the current lifecycle.
- The deterministic three-phase regression blocks start A, completes reset,
  starts B, then releases A. It verifies B remains physically active and the
  sink remains logically `RUNNING`. A close variant is not applicable because
  `CLOSED` has no valid restart path.

## Fix Round 4 Verification

```text
uv run pytest tests/unit/infrastructure/rap/test_audio_output.py -q
21 passed in 0.81s

uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
409 passed in 3.34s

git diff --check
(no output)
```

## Fix Round 4 Commit

- `97d90a14 fix: detach stale rap audio streams on reset`
