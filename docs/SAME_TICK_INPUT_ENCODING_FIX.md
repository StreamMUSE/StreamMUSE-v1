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
Those CPU assertions alone are not GPU inference or physical MIDI-device acceptance.
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

Completed on H200 GPU0 with a virtual MIDI client on Spark, using checkout
`0cb42f90` (system source identical to fix commit `99ddaa68`). Results are in
`F:/repos/StreamMUSE-v1/remote_results/device_midi_replay_same_tick_fix_20260909`.

- A/B Prompt input and selected output: exact.
- A has 32 continuation calls; B/C have 33. B/C additionally generated at tick
  160, the exclusive run stop. This extra call is not hidden or treated as equal.
- On the 32 shared generation ticks, A/B Melody tokens match on 29, full model
  input tokens on 12, raw output tokens and decoded events on 25.
- A/B Melody differences occur at ticks 52, 84, 88. Each is reconstructed from
  protocol history with a matching captured pianoroll SHA256. At generation 84,
  A has not received OFF@81 but B has; at generation 88, A has OFF@86 but B does
  not. At generation 52, A sees an as-yet-unclosed ON@49 absent from B's MIDI.
- Completed A/B Melody histories encode identically; online visible snapshots
  do not. Common-call RNG states match throughout, so the observed output
  differences are not evidence of a seed reset failure.
- B/C: all 33 calls and all 224 continuation sampling tokens/logits/RNG states
  match. Nevertheless, B/C playback MIDI has 78/82 accompaniment notes.

Device-to-MIDI end-to-end exact replay still fails. The same-tick ordering fix
is retained; no subsequent timing or scheduler changes were made. The dedicated
backend was stopped after the test. No physical keyboard/audio was tested, and
the sampler observer makes these runs unsuitable as a latency benchmark.

`scripts/analyze_device_replay_acceptance.py` reports unmatched calls explicitly.
`scripts/check_device_replay_input_snapshots.py` reproduces the remaining three
Melody-input differences using captured request prefixes, without a model load.
