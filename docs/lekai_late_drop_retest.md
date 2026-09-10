# Pure Lekai Late-Onset Drop Retest

Only `lekai_no_prompt` is rerun. Reuse P+C and Legacy from their existing
validated batches; do not overwrite or relabel their results.

- Frozen source inputs/settings: `formal_user10_test40_3system_3seed_20260909/plan.json`.
- user10: 10 complete trimmed Melody MIDIs, seeds 0/1/2, playback/model BPM
  90/80, temperature 1.1, top-p .95, top-k 50, count-in 4 beats.
- test40: the same 40 trimmed test MIDIs, seeds 0/1/2, playback/model BPM
  120/120, temperature 1.05, top-p .98, top-k 0, stop tick 128, no count-in.
- Both cohorts: four steps/beat, 4/4, repetition penalty 1, no Prompt Model,
  no continuation constraints, existing checkpoint and normal MIDI-file CLI.
- New behavior: late note-ons are dropped rather than shifted to the current
  tick. Active-note off cleanup stays enabled. Model and input code unchanged.
- 150 formal trials total. GPU 0/1/2 only; each GPU runs its matching seed.
- Existing native input/raw/event/schedule/system logs and combined MIDI are
  preserved. Empty model outputs remain in the batch; no quality filtering.
- Results go to a new root; stop on any validation failure, no automatic retries.

`scripts/run_lekai_late_drop_retest.sh` wraps existing matched-runner APIs.
It does not implement a new inference path. `prepare` pins hashes and settings;
`worker --smoke` runs two repeats each for one user10 and one test40 case per
GPU, capped at 128 ticks. The smoke includes both previously shifted-onset
test40 cases. `accept` requires equal input/raw repeats and equality against
the corresponding old baseline request prefix. Only then may formal workers
run. Smoke does not establish full-cohort event-vs-MIDI acceptance.

Formal playback is published under `listening_by_cohort/user10` and `test40`.
Raw remains in per-session `inferences.json`; `accompaniment_history.json`
alone is a rolling buffer and must not be treated as the whole-song raw output.
ISR_f/music metrics must be recalculated for the new formal playback; old
metric tables are not silently reused for this condition.
