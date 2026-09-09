# Formal user10 + test40, three systems, three seeds

450 new runs: (10 user pieces + 40 held-out pieces) x 3 systems x seeds 0/1/2.
Only H200 GPUs 0/1/2, one sequential worker per GPU/seed. Spark is not used.
Six separate smoke runs (one piece per cohort/system, seed 0) must complete
before the formal batch can start. Failures stop a worker and remain recorded.

## Systems

- Legacy M2A: original MIDI Melody durations plus absolute monotonic pacing,
  I=4 ticks, GL=7 interleaved frames, original M2A checkpoint. No Prompt Model.
- Pure Lekai: normal standard service, no Prompt Model, no tonal constraint,
  no empty-token guard. This is not P+C single N=1.
- P+C: Rule If-Else N=10, tonal constraint and empty-token guard enabled.

Modern production `src` and `transformers` are unchanged from accepted
`ec843a79`. Only experiment orchestration/configuration is changed. No fixed
event adapter or additional model observer is loaded in these formal runs.
All systems use their ordinary MIDI-file input and realtime playback clock.
Raw output equality does not imply lossless or identical audible delivery.

## Frozen cohort settings

| Setting | user10 | test40 |
| --- | --- | --- |
| Playback / model BPM (Lekai) | 90 / 80 | 120 / 120 |
| Sampling T / top-p / top-k (both Lekai stages) | 1.1 / .95 / 50 | 1.05 / .98 / 0 |
| Repetition penalty | 1 | 1 |
| Modern count-in | 4 beats | 0 beats |
| Extent | full song, retained per-piece stop tick | ticks [0,128) |
| Test40 metric window | not applicable | ticks [32,128), beats [8,32) |

Both cohorts: 4 steps/beat, 4/4. Modern model time-signature index is explicitly
4, matching the accepted current system (the old test40 runner defaulted to 0).
Modern I=4 ticks, GL=4 ticks, Prompt=32 ticks, context=32 beats / 128 ticks.
Use the accepted asynchronous boundary generation order, not the old runner's
forced synchronous setting. Generic late recovery and rehydration remain off.
Continuation and Prompt seeds are explicitly reset for each trial.

## Inputs and outputs

Reuse the existing trimmed user10 inputs from the Legacy duration batch and
the frozen trimmed test40 cohort. Copy each MIDI once, without changing bytes;
all three systems read that exact copy. Check hashes before every run. Test40
GT MIDI is retained for metrics, not supplied as a generation Prompt. No NPZ
is read by inference. Preserve source manifests and checkpoint hashes.

`listening_by_cohort/<cohort>/<piece>/` contains short filenames for actual
`combined.mid` playback exports, plus input Melody (and test40 GT). These
copies do not insert a generated Prompt into playback retroactively.

`runs/` retains full native session artifacts, requests/responses and model
traces. P+C stores raw history and Prompt history separately. For plain Lekai,
full-song raw outputs come from all per-generation logs, not only the retained
rolling history. Backend logs live under `_servers/`, launch/provenance under
`_metadata/`, and progress under `status/gpu0..2.json`.

Music metrics: FMD, JSD-P, original JSD-O, CR and UR; no NLL. System result:
per-beat ISR_f with the existing nominal + 0.1 tick deadline. Preserve startup
and catch-up in the evaluation window. Metrics are a subsequent stage, not
silently computed from an unfinished or mixed-version collection.
