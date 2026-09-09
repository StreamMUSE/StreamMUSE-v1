# Virtual MIDI device versus exported-MIDI replay

This is an automated software-path acceptance test, not a new human performance,
not a music-quality experiment, and not a latency benchmark. User10 results are
not reused. The source is repository `prompts/old_input/mel/001.mid`.

## Frozen system

- Trusted 9/7 base: `2303f341d13a9a6b415335afc69c839d33a8f5f9`.
- No changes to `src/` or bundled `transformers/`.
- Spark: private virtual MIDI sender and existing CLI, CPU only, session output.
- H200 GPU0: one dedicated backend used by all three runs.
- Rule If-Else N=10, tonal constraint and empty-token guard enabled.
- Prompt/continuation: temperature 1.1, top-p .95, top-k 50, penalty 1.
- Playback/model BPM: 90/80, 4 steps/beat, 4/4, count-in 4 beats.
- Prompt 8 beats, context 32 beats, request interval 2 ticks.
- Stop at tick 160, source covers first 24 beats after its first note-on.
- Device input uses existing snap-forward .4; MIDI-file effective snap remains
  zero. These existing semantics are not changed to obtain a passing result.
- No physical output device or Spark model/GPU is used.

## Three runs

1. A: send MIDI messages through a private ALSA port into `MidiDeviceInput`.
2. B: replay A's exported `prompt_continuation_replay_melody.mid` through the
   existing MIDI-file CLI, using A's recorded seed for both models.
3. C: repeat B unchanged to distinguish path differences from replay variation.

A invokes the same session initialization API as Web Start, without supplying
a seed, and adopts the returned session using the existing CLI mechanism.
B and C initialize via the existing API with the recorded seed, then use the existing CLI session
adoption mechanism. No response or token is substituted.

The first harness attempt omitted initialization for bare device CLI and did
not produce a session-seed artifact. It stopped before B/C and is not an
acceptance result. The corrected harness supplies initialization for all runs;
no system code changed.

## Evidence and boundaries

Save original session traces, commands, code identities, sender timestamps,
seed records and the existing strict comparator results. A sampler observer
passes through the existing sampler unchanged and records pre-sampling logits,
RNG states and selected tokens. It does not reseed or alter logits. It does
introduce synchronization/logging overhead, so delivery timing is diagnostic.

Check Prompt selection, every common continuation input/output, raw history,
and exported playback separately. Do not promote equal common prefixes or
incomplete traces into a full-session exact-match claim. A passing automated
case does not guarantee all human performances or cross-GPU exactness.
