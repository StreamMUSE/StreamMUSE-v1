# Lekai public client first-sound and late-drop measurements

Date: 2026-06-12

Environment:

- Repository branch: `integrate-prompt-continuation-switch`
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4GB
- Runtime: `cuda`, `float16`
- Tempo: 120 BPM
- Timing grid: 4 ticks per beat, so 1 tick = 0.125 seconds
- MIDI input: `prompts/A_major/pop909_291_mel.mid`
- Public client path: `streammuse.presentation.cli.cli`

## First audible model event

| Mode | Input alignment | First user event | First ready tick | First model output | Notes |
|---|---:|---:|---:|---:|---|
| Standard realtime | Original MIDI | tick 149 / 18.625s | n/a | tick 13 / 1.625s | Realtime can emit model notes before the first user note because the MIDI has a long leading rest. |
| Standard realtime | `--midi-file-trim-leading-rest` | tick 0 / 0.000s | n/a | tick 33 / 4.125s | Melody starts immediately; first model output is much earlier than prompt-continuation. |
| Prompt-continuation | Original MIDI | tick 149 / 18.625s | tick 69 / 8.625s | tick 77 / 9.625s | Ready does not mean audible immediately; early playable batches were fully stale. |
| Prompt-continuation | `--midi-file-trim-leading-rest` | tick 0 / 0.000s | tick 61 / 7.625s | tick 69 / 8.625s | Even with immediate melody, first audible model event lagged standard realtime by about 4.5s. |

## Prompt-continuation late-drop evidence

The prompt-continuation client fetches playable history only after backend catch-up reports ready. The frontend/client then schedules only events that are still playable at the current tick. Fully past notes are dropped.

### Original MIDI

Trace:

- Start sent at observed tick 32 with `melody_event_count=0`.
- First playable fetch happened 4.759s after start.
- First playable fetch contained 72 accompaniment events.
- Backend status at first playable fetch: `melody_history_beats=17`, `accompaniment_history_beats=18`, `continuation_calls=10`.

Client scheduling after ready:

| Tick context | Scheduled events | Dropped past events | Clipped sustaining notes |
|---:|---:|---:|---:|
| 70 | 0 | 71 | 0 |
| 74 | 0 | 75 | 0 |
| 77 | 2 | 77 | 1 |
| 82 | 0 | 83 | 0 |
| 86 | 0 | 87 | 0 |
| 90 | 0 | 91 | 0 |
| 93 | 2 | 93 | 1 |
| 97 | 2 | 97 | 1 |

Aggregates:

- Schedule reports: 43
- `Scheduled 0` reports: 34
- `Scheduled 0` reports before first model output: 2
- Dropped past events before first model output: 146
- Dropped past events across all schedule reports: 6647
- First model output: tick 77 / 9.625s

### Trimmed-leading-rest MIDI

Trace:

- Start sent at observed tick 32 with `melody_event_count=23`.
- First playable fetch happened 3.858s after start.
- First playable fetch contained 32 accompaniment events.
- Backend status at first playable fetch: `melody_history_beats=15`, `accompaniment_history_beats=16`, `continuation_calls=8`.

Client scheduling after ready:

| Tick context | Scheduled events | Dropped past events | Clipped sustaining notes |
|---:|---:|---:|---:|
| 62 | 0 | 32 | 0 |
| 66 | 0 | 36 | 0 |
| 69 | 2 | 38 | 1 |
| 73 | 2 | 42 | 1 |
| 77 | 0 | 47 | 0 |
| 82 | 0 | 50 | 0 |
| 86 | 0 | 50 | 0 |
| 90 | 0 | 54 | 0 |

Aggregates:

- Schedule reports: 20
- `Scheduled 0` reports: 17
- `Scheduled 0` reports before first model output: 2
- Dropped past events before first model output: 68
- Dropped past events across all schedule reports: 1197
- First model output: tick 69 / 8.625s

## Interpretation

There is concrete evidence that prompt-continuation can successfully generate and fetch accompaniment, but the public client may still output no sound for that batch because the returned events are already stale.

The most direct evidence is:

```text
Scheduled 0 playable accompaniment event(s); dropped 32 past event(s)
Scheduled 0 playable accompaniment event(s); dropped 36 past event(s)
```

in the trimmed-leading-rest run, and:

```text
Scheduled 0 playable accompaniment event(s); dropped 71 past event(s)
Scheduled 0 playable accompaniment event(s); dropped 75 past event(s)
```

in the original-MIDI run.

So the issue is not simply "model did not generate". A generated accompaniment batch can arrive after its intended playback time, then be dropped by the client scheduling policy.

## Relevant logs

- `logs/public_client_realtime_default_console.log`
- `logs/public_client_realtime_trim_console.log`
- `logs/public_client_prompt_cont_default_console.log`
- `logs/public_client_prompt_cont_default_trace.jsonl`
- `logs/public_client_prompt_cont_trim_console.log`
- `logs/public_client_prompt_cont_trim_trace.jsonl`
