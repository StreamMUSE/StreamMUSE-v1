# Causal offline paired history-repair experiment

## Question

Does changing one completed accompaniment bar in conditioning history improve
subsequent generation, without the confounding effect of missed playback deadlines?

This is a mechanism/quality pilot, not a realtime throughput experiment and not
an offline oracle given the full future melody.

## Frozen design

- Source: `formal_pc_beattail_stopfix_user10_test40_3seed_20260910/plan.json`, user10.
- Ten full songs, seeds 0/1/2, baseline and partial repair: 30 pairs / 60 outputs.
- Generate one Rule If-Else N=10 Prompt per pair, then inject exactly that Prompt
  into both conditions. Prompt and continuation sampling: temperature 1.1,
  top-p 0.95, top-k 50, repetition penalty 1. Model BPM 80; MIDI playback BPM 90.
- Same continuation checkpoint, 32-beat context, tonal and EMPTY guards enabled.
- One beat per synchronous request, no wall-clock sleep/deadline/drop. Inject
  Melody `[0,32)` initially, then submit only newly available events with ticks
  strictly below each target beat. Never re-quantize the frozen input MIDI.
- After generating `[80,84)`, regenerate historical bar `[64,80)` using a private
  worker, seed 101, and the already received Melody for that historical bar.
- Wait for the worker offline, apply existing `preserve_cross_boundary` projection
  and strict transaction validation, then resume generation at tick 84.
- The candidate is a partial replacement: original cross-boundary notes remain;
  only compatible complete interior notes from the candidate replace interiors.
- No quality gate, repeated resampling, best-candidate search or per-piece tuning.
- Repair changes future conditioning only. Already emitted raw events are never
  rewritten in the listening MIDI. Invalid repairs retain original history and
  remain in the experiment denominator.
- Boundary generation is synchronous in both conditions. This experiment uses
  the existing continuation engine, not the realtime scheduler/HTTP playback loop.

## Integrity checks

Checkpoint and input hashes checked against the frozen plan. Each run saves its
source hashes, configuration, shared Prompt, full per-call tokens and raw events.
Melody-token and pianoroll digests must match throughout each pair. Full input
and raw-token digests must match before tick 84. If repair is rejected, complete
generation must remain identical. Any violated invariant stops the batch.

`generation.json` is the authoritative per-condition trace. Backend debug logs
use a tick-based filename and may be overwritten across conditions; do not use
that flat debug directory as the paired evidence.

Exports close still-active notes at the fixed song end, separately recorded in
`export_audit.json`. Original raw JSON events are unchanged. Both conditions use
the same export routine and endpoint.

## Metrics

Primary window: `[84, song_end)` system ticks (four ticks/beat). Secondary window:
`[32, song_end)`. The modified historical bar itself is not the primary endpoint.

CR/UR call the existing evaluator's `_compute_harmonic_consonance` directly.
Their semantics remain whole-Melody-note duration weighting with any overlapping
consonant pitch in interval classes `{0,3,4,7,8,9}`. The existing function returns
null for empty Melody/accompaniment; retain those rows and show defined counts.
Do not silently impute a different CR/UR metric.

Additional descriptive measures: onset density per beat, union-of-sounding-time
coverage, and exact adjacent nonempty beat onset-pattern repetition. The latter
compares pitch, local onset and duration; it is not presented as an established
StreamMUSE metric. Empty outputs remain in coverage/nonempty denominators.

Aggregate seeds within each piece, then pieces. Save paired per-piece deltas
with valid seed-pair counts. No verified aligned accompaniment GT is available
for this user10 pilot, so do not compute FMD/JSD using baseline output as GT.
Metric changes alone are not evidence of perceptual improvement; paired listening
is still required. NLL and system ISR are not part of this experiment.

## Run

Experimental branch: `codex/bar-quality-history-experiment`, uncommitted pilot.
Remote checkout: `/data/home/yuanxin/StreamMUSE-bar-quality-history-experiment`.
Launcher: `experiments/bar_quality_history/run_offline_paired.sh`.
GPU: H200 GPU 2, verified idle before launch. Spark remains untouched.

First launch: tmux `bar_history_offline_user10_20261007`.
Output: `/data/home/yuanxin/experiments/bar_history_offline_user10_20261007/run_IkyAZT`.
`exit_code.txt` and `results/complete.json` distinguish completion from progress.
The launcher runs regression tests, all inference, then music-metric analysis.

## Completed result

Exit code 0. All 30 pairs / 60 full-song outputs completed. 25 repairs applied;
two candidates had no compatible interior notes and three projections made no
change. All rejected pairs retained identical full-song output events.

All 3,837 paired calls have equal Melody token/roll digests. All complete input
tokens and emitted events before tick 84 match. Application audits preserve the
live EMPTY guard and restore the candidate's recorded bar-start constraint state.

Primary window macro means, baseline -> repair: CR 0.732759 -> 0.719768;
UR 0.041162 -> 0.060757; sounding coverage 0.927670 -> 0.906727. All 30 trials
per condition have nonempty onsets and defined CR/UR. No overall improvement is
demonstrated by this fixed, ungated historical repair. Exact adjacent nonempty
onset-pattern repetition is zero in both groups and is not discriminative here.

Local package:
`F:/repos/StreamMUSE-v1/remote_results/00_bar_history_probe_20261007/offline_user10`.
`listening_by_piece` contains the 60 unchanged exported MIDI files, named by song
and paired by seed. `integrity_and_dispersion.json` independently checks complete
input/event equality before intervention and reports across-piece paired-delta
sample standard deviations (not confidence intervals).

170 tests passed before launch; five additional metric/export boundary tests
passed afterward. H200 GPU 2 was released after completion. No production branch
was changed or pushed.
