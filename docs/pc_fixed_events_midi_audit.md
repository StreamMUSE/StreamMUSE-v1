# H200 P+C: fixed events versus ordinary MIDI-file simulation

## Scope

Compare Backend raw from the same ten Melody MIDI files, using P+C with
Rule If-Else N=10 and tonal/empty-token constraints. This is a new same-H200
paired experiment, not a claim that old human performances are reproduced.
The separate completed pure-Lekai experiment is not P+C evidence.

No production source or scheduler changes. Production source is pinned to
`ec843a79`; the audit checks `src` and `transformers` against that revision.
The audit branch adds a current-tick event-input adapter and post-call raw
logging only. It does not replace generation, sampling or the playback loop.

## Matched settings

- Playback 90 BPM; model conditioning 80 BPM; four steps per beat; 4/4.
- Four-beat count-in; eight-beat Prompt; I=4 ticks, GL=4 ticks.
- Prompt/Continuation: temperature 1.1, top-p .95, top-k 50, repetition penalty 1.
- Rule If-Else N=10; tonal constraint and empty-token guard enabled.
- Context 32 beats / 128 ticks; generic late recovery and rehydration disabled.
- Prompt and Continuation each reset to seed 1051154023138951872 for every run.
- Two repeats per input path per piece: 40 full-song runs.
- All four runs of a piece stay on one GPU. Only H200 GPU 0/1/2 are allowed.
- Input hashes, stop ticks and source-session mapping are retained from the
  completed fixed-event input experiment. No new trimming or retiming.

## Input paths

A (`fixed`): read the exported Melody MIDI, retain integer event ticks,
velocity and same-tick MIDI ordering. Inject only the current tick's events
into the existing P+C input-window and playback queues, immediately before
the ordinary queue drain. No future events or wall-clock requantization.

B (`midi`): use the unmodified CLI `--input-mode midi_file` path on the same
file, with the same session configuration. Its native velocity and ordering
behavior is retained; model-input equality is measured, not presumed.

Both use the normal P+C input-window worker, scheduler, model and playback.
This tests already-quantized event user simulation, not physical MIDI-device
arrival jitter. The added post-call logging is not a latency benchmark.

## Acceptance

Primary: match every generation call by `generation_start_tick`, then compare
raw generated tokens, structural tokens, decoded raw events and accumulated
raw history, plus the generated Prompt history. Missing calls or incomplete
traces do not pass. Empty equal outputs are reported as empty, not good music.

Also record Prompt input/selection and continuation input tokens separately.
Exported accompaniment MIDI equality is secondary and never substituted for
raw equality. Playback differences can exist despite matching raw output.

Each pair and each same-path repeat gets a comparison JSON. Differences are
retained and remaining cases continue; no retries to obtain equality. Backend
failures stop that worker and remain in its progress/log files.

## Files

- `plan.json`, `inputs/<piece>/`: immutable settings and source inputs.
- `worker0..2/`: owned backend logs, source identity, environment and raw calls.
- `cases/<piece>/<path>_repeat_1..2/`: input, configuration, logs and native artifacts.
- `cases/<piece>/comparison_repeat_1..2.json`: A/B comparisons.
- `cases/<piece>/<path>_self_repeat.json`: repeated-run comparisons.
- `progress_gpu0..2.json`: progress or explicit failures.

Launch each worker in a separate tmux session with
`bash scripts/run_pc_fixed_events_midi_audit.sh RESULT_ROOT GPU`.
Workers stop only their own backend when finished. No Spark process is used.
