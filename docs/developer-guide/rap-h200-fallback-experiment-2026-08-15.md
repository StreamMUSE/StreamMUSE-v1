# H200 Fallback Experiment - 2026-08-15

## Question

Where does the current realtime rap pipeline fall back when real H200 lyric
generation is combined with local eSpeak rendering and timed WAV playback?

This is a bounded prototype experiment, not a statistically powered benchmark.

## Setup

- Client revision: `edbf4467` on `feature/real_rap_audio`
- Client host: macOS, using the current Python runtime and eSpeak renderer
- Model host: `Andrew.Yang@masdar`, unused H200 GPU 0
- Model: `Qwen/Qwen2.5-7B-Instruct`, served as `qwen-rap` by vLLM 0.24.0
- Transport: SSH tunnel from Mac port 18001 to H200 loopback port 18001
- Audio output: device-free timed IEEE-float WAV, stereo, 48 kHz
- Lookahead: 3 bars
- Minimum lyric score: 0.3
- Trial length: 8 played bars

The model discovery endpoint and a minimal chat completion both passed before
data collection. Five trials were run. The first 8-candidate trial includes a
cold large-request latency spike; a second 8-candidate trial was run after the
server was warm.

## Per-Trial Results

`Lyric fallback` excludes the mandatory prevalidated first bar. Candidate
validity is exact-flow validity among response lines parsed as candidates.

| Trial | Returned / requested lines | Valid candidates | Avoidable lyric fallback | Reasons | Forced-fit slots | Underruns |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 8 candidates, 60 BPM, cold | 32 / 56 (57.1%) | 12 / 32 (37.5%) | 4 / 7 (57.1%) | 4 deadline misses | 0 / 72 | 0 |
| 8 candidates, 60 BPM, warm | 31 / 56 (55.4%) | 10 / 31 (32.3%) | 2 / 7 (28.6%) | 2 no-valid-candidate | 1 / 72 | 0 |
| 12 candidates, 60 BPM | 63 / 84 (75.0%) | 21 / 63 (33.3%) | 2 / 7 (28.6%) | 2 no-valid-candidate | 0 / 72 | 0 |
| 16 candidates, 60 BPM | 81 / 112 (72.3%) | 15 / 81 (18.5%) | 2 / 7 (28.6%) | 1 no-valid-candidate, 1 below score threshold | 0 / 72 | 0 |
| 12 candidates, 92 BPM | 74 / 84 (88.1%) | 27 / 74 (36.5%) | 1 / 7 (14.3%) | 1 no-valid-candidate | 7 / 72 | 0 |

Every trial also used one intentional initial fallback bar. Including that bar,
16 of 40 committed bars used prevalidated lyrics. There were no cases where a
generated lyric was frozen but audio failure independently forced a fallback.

## Pipeline Rates

### H200 Transport And Generation

- Generator errors: 0 / 35 requests (0.0%)
- Requests returning fewer lines than requested: 30 / 35 (85.7%)
- Returned-line yield: 281 / 392 requested lines (71.7%)
- Deadline misses: 4 / 35 requests (11.4%), all in the first cold trial
- Warm-trial deadline misses: 0 / 28 requests (0.0%)

The model usually returned fewer newline-delimited candidates than requested.
This was diagnostic shortfall, not automatically a bar fallback: valid lines
were still gated and ranked when at least one usable candidate remained.

### Candidate Validation And Lyric Selection

- Exact-flow valid candidates: 85 / 281 parsed candidates (30.2%)
- Mandatory initial fallback: 5 / 40 played bars (12.5%)
- Avoidable lyric fallback: 11 / 35 eligible generated bars (31.4%)
- Warm-only avoidable lyric fallback: 7 / 28 eligible bars (25.0%)
- Warm-only causes: 6 no-valid-candidate and 1 below-score-threshold

The main steady-state loss is therefore the strict lyric gate, not transport
failure. The small sample does not show that requesting 16 candidates improves
fallback rate: that run returned more lines but a lower fraction passed the
exact-flow gate.

### Pronunciation And Audio Rendering

- Planning/prosody pronunciation fallback: 10 / 1,672 analyzed candidate
  words (0.60%)
- Committed audio pronunciation fallback: 0 / 360 syllable slots (0.0%)
- Committed synthesis failures: 0 / 360 syllable slots (0.0%)
- Audio-specific committed-bar fallback: 0 / 40 bars (0.0%)
- Playback underruns: 0 / 40 bars (0.0%)
- Valid 48 kHz float WAV artifacts: 5 / 5 trials (100%)

The planning fallback rate and audio pronunciation fallback rate are different
metrics. The first describes lexical analysis used while scoring all candidate
lines. The second describes the renderer used for the 360 syllable slots that
were actually committed to playback.

### Timing Degradation, Not Fallback

- Forced-fit syllable slots: 8 / 360 (2.2%)
- Timing-pressure syllable slots: 112 / 360 (31.1%)
- At 60 BPM: 1 / 288 forced-fit slots (0.35%) and 72 / 288 timing-pressure
  slots (25.0%)
- At 92 BPM: 7 / 72 forced-fit slots (9.7%), affecting 7 / 8 bars, and
  40 / 72 timing-pressure slots (55.6%)

These warnings mean the audio renderer compressed, overlapped, or forced a
syllable to the bar boundary. They preserve the schedule and are not counted as
fallbacks, but they are an important quality cost at higher tempo.

## Integrity Checks

All five canonical `events.jsonl` files regenerated `summary.json` and
`bars.csv` byte-for-byte. All WAV files were recognized as stereo, 48 kHz IEEE
float RIFF/WAVE, and every trial completed its expected frame count with zero
reported underruns.

Local artifacts are under:

`/tmp/streammuse-h200-fallback-20260815/`

## Limitations

- Only eight bars were played per trial.
- Conditions other than 8 candidates at 60 BPM were not repeated.
- The same deterministic scenario/topic schedule was used throughout.
- WAV generation validates the audio pipeline but not speaker audibility or
  physical output latency.
- The first cold 8-candidate run is reported rather than discarded; warm-only
  rates are also shown so startup and steady-state behavior are not conflated.
