# Voice Length-Sweep Benchmark Design

## Goal

Measure how short-form speech conversion latency changes with sequence length on
the local Mac and the H200, using the fastest viable backends found in the
initial voice microbenchmark. The benchmark remains outside the StreamMUSE
runtime so it isolates speech conversion cost from capture, endpointing,
networking, playback, and LLM execution.

## Configurations

| Modality | Mac | H200 |
|---|---|---|
| STT | `faster-whisper tiny.en`, CPU int8 | `faster-whisper tiny.en`, GPU float16 |
| TTS | `espeak-ng` | Persistent Piper `en_US-amy-low` |

The H200 run must select an idle GPU and record the selected device and CUDA
environment. No claim about process GPU memory is made when the machine has
unrelated GPU work.

## Workload

- Word-count buckets: `1`, `2`, `4`, `8`, `16`, `32`, `64`.
- Five deterministic English phrases per bucket.
- Ten warmed repetitions per phrase, randomized within each bucket.
- Three unmeasured warmup operations after model/client setup.

The same phrase corpus is used for STT and TTS. STT input WAVs are generated
once, normalized to 16 kHz mono PCM, and copied unchanged to H200. This is a
reproducible synthetic-speech workload, not a real-microphone accuracy study.

## Measurements

Every configuration records three timing phases:

1. Setup: a new process creates and loads the model/client until ready to
   accept a request. This is a cold process measurement, not a cold disk-cache
   measurement.
2. First request: the first request after setup.
3. Warmed requests: the timed repetitions after three unmeasured warmups.

STT trial fields include phrase identifier, input word count, input audio
duration, transcription latency, and transcript.

TTS trial fields include phrase identifier, input word count, generation time
(request start until full WAV is available), generated audio duration, and
real-time factor (`generation_time / audio_duration`). Audio duration is a
separate output property and is never conflated with generation latency.

## Outputs

Each run directory contains a manifest, raw per-trial JSON/CSV data, summary
statistics, and PNG line plots. Each plot shows individual trials plus p50 and
p95 curves.

- STT, each device: latency vs. word count; latency vs. input-audio duration.
- TTS, each device: generation time vs. word count; generated-audio duration
  vs. word count; real-time factor vs. word count.

The final report compares setup, first-request, warmed p50/p95/mean/max, and
identifies the largest word-count bucket that remains plausible for a
one-second interaction budget.

## Validation

Unit tests cover corpus construction, repeat accounting, duration extraction,
summary calculations, and plot-data assembly. A small local smoke run verifies
the artifacts before full Mac and H200 benchmark runs.
