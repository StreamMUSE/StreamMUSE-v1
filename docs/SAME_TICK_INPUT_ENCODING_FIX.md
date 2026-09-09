# Same-tick input encoding fix

Branch: `codex/fix-same-tick-melody-encoding`.
Parent: `03228353` (test-only acceptance work on trusted system base `2303f341`).

## Scope

- `MidiConverter.events_to_pianoroll`: sort chronologically but preserve input
  order at equal ticks. An observed ON then OFF at one tick leaves neither
  sustain nor onset. OFF then ON still retriggers the pitch.
- `LekaiPromptEngine._event_sort_key`: apply the same order when computing
  active pitches carried into a Prompt window.
- `LekaiHttpBackend._event_sort_key`: apply the same order when computing
  carry state and trimming histories for continuation encoding.

Original event lists and logs are not modified. Positive one-tick notes,
including those crossing beat boundaries, remain valid. This does not lengthen
short notes, change quantization, reorder playback, change scheduler timing,
modify sampling/weights, or change offline MIDI parsing or MIDI export.

## Verification

- New focused tests before the fix: 12 failed, 3 passed.
- After the fix: all 15 focused tests pass.
- Inference unit tests plus Prompt-continuation service tests: 429 passed.
- MIDI output sink tests also pass as part of the separate 208-test targeted run.
- Captured automated device A / MIDI-file B input from 2026-09-09:
  CPU reconstruction now yields identical Prompt input tokens; these also match
  the previous B capture. Pitch 70 at tick 5 no longer sustains through tick 31.
- On full recorded histories, all 40 four-tick Melody encoding windows and
  outgoing active-pitch states match between A/B after this fix.

The recorded-input reconstruction is not a new live model inference test:
it uses known histories and tick windows, not asynchronous request arrival.
No post-fix GPU run or physical MIDI-device acceptance is claimed here.
If an OFF has not yet reached a request snapshot, the encoder must not use it
early; this commit does not change that request-visibility issue or playback
deadline variation. It cannot repair old logs whose ordering was already lost.

The old acceptance analyzer must be run from its pinned pre-fix checkout to
reproduce the original mismatch. Do not reinterpret old outputs as fixed runs.

## Post-fix inference acceptance

The isolated harness accepts `--system-ref 99ddaa68` on both server and client
to verify system source against the fix commit. Use `--seed 1051154023138951872`
on the client to retain the pre-fix test's seed; B/C still copy the seed record
actually returned for A. This changes only harness configuration, not runtime
quantization, scheduler or model behavior. Results must be saved to a new folder.
