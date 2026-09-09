# P+C Beat-Tail Delivery Retest

Branch: `codex/fix-pc-beat-tail-delivery`, based on the formal runner at
`69ff9fa0` (production baseline `ec843a79`).

## Production Changes

- Client: restore the existing tick loop's beat-tail submission. With four
  steps per beat, the request for tick 44 is submitted after the input buffer
  at tick 43 (nominally 43.1), not after tick 44. The eight-beat opening Prompt
  still waits for completion of the observation window.
- Backend: retain event admission by request boundary. An old-tick event that
  arrives after one snapshot remains eligible for the next request. Already
  delivered events are not duplicated. A slow worker cannot retroactively use
  events from later requests in an earlier generation.
- Unchanged: quantization and snap-forward, model/checkpoints, sampling,
  Rule If-Else and constraints, generation target/chunk size, playback late
  policy, and generic Late Recovery (off).

The client path is shared by live MIDI-device input and MIDI-file input.
This restores inference lead time for both. Live events arriving after a
snapshot may enter the next request, even when their quantized tick belongs
to the previous beat. They keep their original quantized tick. This is not a
claim that an arbitrary live recording can be reproduced from MIDI alone:
that requires matching recorded per-request event visibility as well.

## Frozen Evaluation

`scripts/run_pc_beat_tail_retest.py` reuses the existing matched evaluation
server and normal MIDI-file CLI. It does not hook model inputs or replace
playback with raw reconstruction. The prior formal plan supplies immutable
Melody MIDI paths, SHA-256 hashes, stop ticks, checkpoints, and parameters.

- P+C Rule If-Else N=10 with constraints only; seeds 0, 1, 2.
- user10: full songs, playback/model BPM 90/80; T=1.1, p=0.95, k=50,
  repetition penalty 1.0, four-beat count-in.
- test40: original 128-tick stop, BPM 120/120; T=1.05, p=0.98, k=0,
  repetition penalty 1.0, no count-in.
- Prompt and Continuation keep identical sampling settings within each cohort.
- Only H200 GPU 0/1/2. No Spark or local GPU use.
- 150 formal trials. Prior outputs, pure Lekai, and Legacy are not overwritten.

Before formal execution, smoke workers each run one user10 and one test40
piece twice with seed 0, stopping at tick 128. Inspect native model input/raw
logs and deadline delivery, then write a commit-linked acceptance record.
Smoke outputs are diagnostic only and excluded from formal results.

`combined.mid` is actual playback. Native raw and Prompt histories remain
in each session directory for separate reconstruction and diagnosis.
An error stops that worker and retains its logs; there are no automatic
retries or replacement trials.
