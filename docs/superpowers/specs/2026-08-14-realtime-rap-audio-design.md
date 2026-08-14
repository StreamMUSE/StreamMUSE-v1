# Realtime Rap Audio Design

Date: 2026-08-14

## Status

Approved design for the first audible realtime rap prototype. This document
defines the implementation contract for a Mac client that generates, renders,
plays, records, and monitors beat-aligned rap while using an H200-hosted LLM
only for lyric candidate generation.

## Context

The existing rap showcase already provides the text-side pipeline:

1. A scheduled scenario chooses topics and MCFlow-derived flow templates.
2. An OpenAI-compatible local-chat generator requests lyric candidates.
3. CMUdict-backed prosody analysis identifies syllables, stress, rhyme tails,
   and out-of-vocabulary words.
4. Candidate ranking rejects structurally invalid lines and selects a line for
   the required flow template.
5. Exact alignment assigns one analyzed syllable to each occupied flow slot.
6. A rolling controller reserves prevalidated fallbacks, replaces them before
   their deadline, and emits detailed monitoring events.
7. Terminal and web monitors expose generation, scoring, alignment, timing,
   and fallback behavior.

The current runtime emits text events on a monotonic tick loop but does not
produce speech or percussion. It also commits selected text immediately, which
is insufficient once the selected line must be synthesized before playback.

## Goal

Produce continuous audible rap in which every planned syllable begins at its
assigned flow slot. The first version prioritizes timing, inspectability, and
continuous operation over natural voice quality.

The demonstration must run with:

- the LLM and vLLM server on an H200 host;
- all planning, scoring, synthesis, mixing, playback, monitoring, and web UI on
  the user's Mac;
- prescheduled topics, tempos, and flow templates;
- a deliberately robotic but phonetically controlled local voice;
- a stable local drum reference;
- rolling bar-by-bar operation with prevalidated no-gap fallbacks.

## Non-Goals

This version does not include:

- live keyboard drum input;
- live microphone topic input;
- automatic tempo following;
- a natural neural rap voice;
- server-side TTS or audio streaming from the H200;
- browser-owned audio playback;
- interactive tempo, topic, voice, candidate-count, lookahead, or mixer
  controls;
- a claim that MCFlow contains an instrumental drum performance;
- direct measurement of physical loudspeaker latency.

Those capabilities remain future work and must not complicate the initial
audio path.

## Approved Product Decisions

1. The H200 hosts only the LLM endpoint. It returns text and diagnostics, never
   audio.
2. A Python process on the Mac owns the authoritative musical and audio clock.
3. Every aligned syllable receives its own scheduled audio onset.
4. Robotic speech is acceptable; best-effort pronunciation is acceptable when
   an exact pronunciation is unavailable.
5. Pronunciation uncertainty never invalidates or lowers the score of a lyric
   candidate. It produces a visible and recorded warning.
6. A syllable that exceeds its available duration is compressed first and may
   overlap the following syllable. The following onset is never delayed.
7. Drums use a stable boom-bap foundation with subtle flow-slot accents.
8. The website exposes only Start, Stop, and Reset.
9. Stop is bar-quantized: the current bar finishes before playback stops.
10. Tempo, topics, lookahead, candidates, drums, and voice are startup
    configuration.

## System Architecture

```text
H200 host
  vLLM OpenAI-compatible endpoint
       |
       | HTTP lyric candidate batches
       v
Mac Python process
  Generation Coordinator
       |
       v
  MCFlow + Prosody + Candidate Ranking
       |
       | exact syllable schedule
       v
  Phoneme Speech Renderer
       |
       | one waveform per syllable
       v
  Bar Audio Renderer -------- Drum Renderer
       |                          |
       +------ vocals + drums ----+
       |
       | immutable complete PCM bars
       v
  Python Audio Engine
       |
       v
  Mac speakers

  Monitoring events -> FastAPI/WebSocket -> Local website
```

The audio path does not depend on a connected browser. Refreshing or closing
the website cannot interrupt planning or playback.

## Initial Configuration

The first demonstration configuration is:

| Setting | Initial value |
|---|---:|
| Playback tempo | 60 BPM |
| Meter | 4/4 |
| Ticks per beat | 4 |
| Audio sample rate | 48,000 Hz |
| Channels | 2 |
| Candidate count | 12 per LLM request |
| Lookahead target | 3 bars |
| Voice | eSpeak NG English phoneme voice |
| Drum style | Stable boom-bap |
| Browser controls | Start, Stop, Reset |

The existing scenario tempo remains usable. A CLI tempo override allows the
audio demonstration to start at 60 BPM without rewriting scenario topic and
template schedules. Later tests increase playback to 92 BPM while holding all
other inputs fixed.

## Timing Model

The audio engine uses an absolute sample clock. It does not accumulate sleep
durations or schedule individual syllables from wall-clock callbacks.

At 60 BPM with four ticks per beat:

```text
seconds_per_beat = 60 / 60 = 1.0
seconds_per_tick = 1.0 / 4 = 0.25
seconds_per_bar  = 4.0
samples_per_bar = 4.0 * 48,000 = 192,000
```

For a slot inside a bar:

```text
onset_sample = bar_start_sample
             + round(tick_in_bar * sample_rate * 60
                     / (bpm * ticks_per_beat))
```

Every onset is derived from the absolute bar start. No onset is calculated by
adding the previous syllable's duration, so rounding cannot accumulate across
syllables or bars.

The software can guarantee sample placement inside the rendered waveform. It
cannot guarantee or directly measure when the physical speaker produces that
sample without an external loopback recording. Monitoring must distinguish
software placement, output callback status, and unmeasured device latency.

## Planning And Audio Commitment

The current rolling controller reserves fallback text immediately and replaces
it when a primary candidate is selected before the bar freezes. The audio
runtime extends this into an atomic text-and-audio commitment pipeline:

```text
FALLBACK_READY
    -> CANDIDATE_SELECTED
    -> CANDIDATE_AUDIO_RENDERING
    -> CANDIDATE_AUDIO_READY
    -> COMMITTED
    -> PLAYING
    -> COMPLETED
```

Fallback bars are synthesized as soon as they are reserved. This produces a
ready fallback but does not commit it early, so a completed primary candidate
can still replace it before the deadline.

A primary candidate does not replace the ready fallback when ranking finishes.
Its syllables are first synthesized and mixed into a complete bar. Only a
`CANDIDATE_AUDIO_READY` result received before the commitment deadline becomes
the preferred source. At the deadline, the coordinator commits either that
primary source or the ready fallback. The lyric text, scheduled syllables,
warnings, and PCM buffer are committed as one immutable playable-bar object.

The default commitment deadline is one musical tick before the target bar
starts. At 60 BPM this is a 250 ms guard; at 92 BPM it is approximately 163 ms.
The already-rendered bar only needs to be inserted into the playback queue
during this guard. Once committed, a bar cannot be replaced.

If generation, ranking, or rendering finishes late, the coordinator commits or
retains the fallback. Late results are recorded for research but cannot mutate
a committed, playing, or completed bar.

## Candidate Generation And Lookahead

Each target bar uses one batched LLM request for 12 candidates. The local
generator parses the response once, analyzes every candidate, and ranks the
batch deterministically.

The rolling planner targets three bars of lookahead. Generation remains
sequential initially so selected prior lines can provide coherent context to
the next request. Prevalidated fallbacks permit the clock to continue while the
planner catches up.

H200 benchmarking will sweep candidate counts around the initial value while
recording response and decision slack. Increasing candidates is accepted only
when the selected candidate still reaches audio commitment reliably. Adaptive
candidate counts are future work; the first runtime uses an explicit startup
value so runs remain comparable.

## Pronunciation Pipeline

The existing prosody analyzer stores one CMUdict ARPAbet phoneme tuple per
syllable. The initial speech adapter maps those phonemes to eSpeak NG phoneme
mnemonics and renders the unit locally.

Example:

```text
word: moving
CMUdict: M UW1 V IH0 NG
slot A: M UW1 V
slot B: IH0 NG
```

The renderer does not submit guessed strings such as `mov` and `ing` to an
ordinary sentence TTS engine. Explicit phonemes preserve the intended vowel
and consonant sounds even when each syllable is independently scheduled.

Pronunciation fallback order is:

1. CMUdict syllable phonemes mapped to eSpeak phonemes.
2. eSpeak best-effort grapheme-to-phoneme output for an unknown word, split on
   detected vowel nuclei to match the planned syllable count.
3. Deterministic heuristic grapheme fragments rendered independently at the
   expected syllable positions.
4. Silence only when every synthesis attempt fails.

Levels 2 through 4 emit a visible `pronunciation_fallback` warning. Warnings do
not affect candidate validity or score.

The `SpeechSynthesizer` boundary allows eSpeak NG to be replaced later by a
neural or singing-oriented engine without changing alignment, timing, or
playback code.

## Syllable Duration Fitting

Each rendered syllable has a target onset and an available interval ending at
the next scheduled syllable or the end of the bar.

The bar renderer performs these steps:

1. Decode the synthesizer output and resample it to 48 kHz.
2. Remove leading and trailing synthesis silence using a deterministic
   amplitude threshold with short safety padding.
3. Preserve the waveform unchanged when it fits its interval.
4. Time-compress audio by up to 2.0x when needed.
5. At the 2.0x cap, allow the remaining tail to overlap the next syllable.
6. Place the next syllable at its original sample offset regardless of overlap.
7. Mix with headroom and apply a final peak limiter to prevent clipping.

Silence trimming uses a default -45 dBFS threshold with 5 ms of safety padding.
Compression above 1.1x and every overlap produce a `timing_pressure` event
containing the available duration, source duration, compression ratio, overlap
duration, and action. These values are startup configuration, not web controls.

An overlap may continue through later vocal onsets but may not cross the bar
boundary in the initial implementation. A final syllable that still exceeds
the bar after 2.0x compression receives emergency compression to the remaining
bar duration and a `forced_bar_fit` warning. This preserves bar-quantized Stop
and fixed-size immutable PCM bars.

The renderer never silently drops a successfully synthesized syllable.

## Drum Rendering

MCFlow describes vocal flow slots, accents, boundaries, and rhyme groups. It
does not provide an instrumental drum pattern.

The first drum renderer derives a predictable reference groove:

- kick on beats 1 and 3;
- snare on beats 2 and 4;
- closed hi-hat on every sixteenth-note tick;
- stronger hat velocity on beat boundaries;
- subtle additional hat emphasis on occupied high-accent MCFlow slots.

The foundation remains stable between templates so listeners can compare vocal
timing. Only hat emphasis responds to MCFlow in the initial version. Drum
samples are generated or bundled locally, loaded once, and mixed into each bar
using the same sample calculations as vocal onsets.

## Playback Engine

The live sink uses a PortAudio callback through Python `sounddevice` and the
Mac CoreAudio device. The callback consumes immutable float32 PCM bars from a
prepared-bar queue.

The callback is intentionally minimal. It may:

- copy prepared PCM into the requested output block;
- advance the authoritative frame counter;
- switch to the next prepared bar at an exact boundary;
- set lock-free or minimal-lock status flags for an observer.

It may not perform:

- HTTP requests;
- lyric generation or ranking;
- speech synthesis;
- waveform rendering;
- disk I/O;
- WebSocket delivery;
- terminal formatting;
- synchronous event logging.

A separate playback observer translates sample positions and callback status
into monitoring events. Monitoring failure therefore cannot interrupt audio.

The audio output boundary has four implementations:

1. `SoundDeviceAudioSink` for live playback.
2. `WavAudioSink` for the exact mixed research artifact.
3. `CompositeAudioSink` for simultaneous playback and recording.
4. `NullAudioSink` for deterministic tests.

## Runtime State Machine

```text
STOPPED -> PRIMING -> RUNNING -> STOP_REQUESTED -> STOPPED
   ^                       |
   +--------- RESET <-----+
```

The web and CLI use identical runtime commands.

### Start

Start reserves and prepares fallback bars, starts planning, opens the audio
stream, and begins at a complete bar boundary. Starting from a previously
stopped session continues with the next complete bar.

### Stop

Stop changes the state to `STOP_REQUESTED`. The currently playing bar finishes
normally. The playback engine does not dequeue another bar, then enters
`STOPPED`. The planner stops submitting new work. An in-flight response may be
recorded but cannot trigger unexpected playback.

### Reset

Reset stops playback, clears prepared bars, pending proposals, lyric history,
rhyme anchors, warnings, counters, and timing origins, and returns to bar zero
in `STOPPED`. Playback begins again only after Start.

The website contains no other controls. Configuration changes require a new
process or Reset plus a future configuration mechanism that is outside this
scope.

## Application Interfaces

```python
class SpeechSynthesizer(Protocol):
    def synthesize(
        self,
        request: SyllableRenderRequest,
    ) -> RenderedSyllable: ...


class RapBarRenderer(Protocol):
    def render(self, plan: PlannedRapBar) -> PreparedRapBar: ...


class RapAudioSink(Protocol):
    def start(self) -> None: ...
    def enqueue(self, bar: PreparedRapBar) -> None: ...
    def request_stop_after_bar(self) -> None: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
```

`SyllableRenderRequest` contains the word, syllable index, phonemes, stress,
pronunciation source, and voice configuration. `RenderedSyllable` contains
immutable PCM plus synthesis and warning metadata.

`PreparedRapBar` atomically contains the selected plan, exact frame count,
mixed PCM, per-syllable render diagnostics, drum diagnostics, and warnings.
Domain and application contracts use immutable byte-backed PCM data rather
than exposing NumPy arrays across layer boundaries.

## Code Placement

The implementation follows the repository's existing layer structure.

### Domain

- Add rap audio value objects, warning codes, and playback state.
- Extend rap event types without adding infrastructure dependencies.

### Application

- Add speech and audio sink protocols.
- Add the deterministic syllable fitting and bar rendering service.
- Add the playback state machine and prepared-bar queue coordinator.
- Refactor rolling candidate commitment so text and audio commit atomically.
- Replace wall-clock syllable emission as the playback authority with the
  audio sample position observer.

### Infrastructure

- Add the eSpeak NG phoneme synthesizer and best-effort fallback adapter.
- Add the stable drum renderer.
- Add sounddevice, WAV, composite, and null audio sinks.
- Reuse existing session recording and event dispatch patterns.

### Presentation

- Extend the rap-demo CLI with audio, tempo override, voice, device, recording,
  and fitting configuration.
- Add Start, Stop, and Reset API operations.
- Add exactly three matching website controls.
- Add audio readiness, playback, queue, warning, and timing panels to the
  existing dense research monitor.

## Monitoring Contract

The runtime adds these canonical events:

- `audio_render_started`
- `audio_render_completed`
- `pronunciation_fallback`
- `timing_pressure`
- `bar_audio_ready`
- `bar_audio_committed`
- `bar_playback_started`
- `bar_playback_completed`
- `stop_requested`
- `session_reset`
- `audio_underrun`
- `audio_device_failed`

Per-syllable diagnostics include:

- word, syllable index, and displayed label;
- input phonemes and renderer phonemes;
- pronunciation and synthesis source;
- target bar, tick, and absolute sample;
- original and fitted durations;
- available duration;
- compression ratio and overlap;
- warning code and action;
- software placement error in samples.

Session and bar summaries include:

- audio state and selected output device;
- prepared-bar queue depth and buffered seconds;
- LLM response, decision, synthesis, and bar-render latency;
- response, decision, and audio-commit deadline slack;
- callback block size and underrun count;
- fallback, pronunciation, compression, and overlap counts;
- recording path and format.

The terminal and web UI project the same canonical events. Presentation
failures remain isolated from planning and playback.

## Error Handling

| Failure | Required behavior |
|---|---|
| H200 unavailable | Continue with prevalidated fallback bars and report generation failure |
| LLM response late | Record it, retain the committed fallback |
| No valid candidate | Retain the committed fallback |
| Candidate audio late | Retain the committed fallback and report audio deadline miss |
| CMUdict word missing | Use best-effort pronunciation and warn |
| Phoneme mapping incomplete | Use text fallback and warn |
| Syllable too long | Compress, then overlap if required, and warn |
| Syllable synthesis fails | Try degraded synthesis; use silence only after all attempts fail |
| Browser disconnects | Continue audio and session recording |
| Event sink fails | Isolate the sink and continue audio |
| Audio queue underruns | Output silence for the affected block, warn, and recover at the next prepared boundary |
| Audio device fails to open | Keep the monitor alive, report failure, and preserve WAV rendering when configured |
| Stop requested | Finish the current bar and stop before the next bar |
| Reset requested | Clear all session state and return to stopped bar zero |

## Testing Strategy

### Unit Tests

Unit tests cover:

- ARPAbet-to-eSpeak mapping and stress handling;
- pronunciation fallback and warning payloads;
- silence trimming and resampling;
- absolute tick-to-sample calculations at multiple tempos;
- no cumulative rounding drift across at least 100 bars;
- fitting behavior for natural, compressed, and overlapping syllables;
- exact kick, snare, hi-hat, and flow-accent positions;
- clipping prevention and exact output frame counts;
- fallback-first and audio-ready atomic commitment;
- Start, bar-quantized Stop, resume, and Reset transitions;
- observer event ordering and warning aggregation;
- web control API validation and idempotency.

Tests use fake synthesizers and fake audio sinks. They do not require speakers,
an H200, eSpeak, or realtime sleeping.

### Integration Tests

Integration tests cover:

- rolling planning, rendering, and playback through a fake sample clock;
- delayed and failed LLM generation with continuous fallback audio;
- late candidate audio that cannot replace committed fallback audio;
- 100-bar operation with exact total sample count and zero cumulative software
  drift;
- stopping at the exact end sample of the active bar;
- resetting all planner, renderer, playback, and monitor state;
- browser connection, refresh, and disconnection without audio-side effects;
- simultaneous live and WAV sinks receiving identical PCM.

### Local Mac Tests

The Mac test suite includes:

- an eSpeak NG smoke test for representative mono- and multisyllabic words;
- a pronunciation listening list containing CMUdict and fallback words;
- per-syllable and per-bar synthesis latency measurements;
- a sounddevice/CoreAudio startup and sustained-playback test;
- a ten-minute run at 60 BPM followed by a run at 92 BPM;
- comparison of the recorded WAV timeline against scheduled sample positions.

### H200 Tests

The H200 test suite includes:

- the real local-chat/vLLM workflow with the configured rap model;
- candidate-count sweeps that include 8, 12, and 16 candidates;
- response, decision, and audio-commit slack at 60 and 92 BPM;
- forced endpoint failure to confirm uninterrupted fallback playback on Mac.

## Acceptance Criteria

The first audible prototype is complete when:

1. The Mac client connects to the H200 vLLM endpoint and generates rolling rap
   bars from the existing scheduled scenarios.
2. Every occupied flow slot has a corresponding scheduled vocal audio unit or
   an explicit synthesis-failure warning.
3. Every successfully rendered unit is mixed at its exact target sample.
4. Syllable timing never shifts because a prior syllable is long.
5. Timing compression, overlap, and uncertain pronunciation are visible in the
   terminal, website, and session log.
6. Stable boom-bap drums and vocals play from one Python-owned audio stream.
7. The session produces a WAV artifact identical to the live mixed PCM.
8. H200 failure or lateness produces fallback audio without stopping the
   musical clock.
9. Stop finishes the current bar and prevents the next bar from starting.
10. Reset returns the planner, audio engine, and monitors to stopped bar zero.
11. The website contains only Start, Stop, and Reset controls.
12. A sustained 60 BPM run and a 92 BPM run complete without cumulative sample
    drift; any callback underrun is prominently reported.
13. Setup and SSH forwarding instructions are documented for the split Mac and
    H200 workflow.

## Future Work

Deferred extensions include:

- live keyboard drums and tempo estimation;
- live microphone topics and lower-latency topic switching;
- adaptive candidate counts based on measured deadline slack;
- natural neural TTS, singing synthesis, forced alignment, and time warping;
- improved cross-syllable coarticulation;
- user-selectable voices and drum kits;
- separate vocal and drum mixer controls;
- interactive tempo, topic, lookahead, and candidate controls;
- flow-derived kick and snare variation;
- server-side synthesis experiments;
- loopback microphone measurement of physical acoustic latency.
