# Voice Microbenchmark Results

Date: 2026-07-10

This benchmark intentionally peels voice away from StreamMUSE. It measures short speech/text conversion only: 10 fixed 1-3 word samples for STT and the same 10 phrases for TTS. It does not include microphone capture, VAD endpointing, LLM latency, task runtime, or speaker playback.

## Metric Definitions

For STT, `setup ms` is model/client initialization before any sample is transcribed. `first run ms` is the first transcription after setup, which often includes lazy kernels, caches, or runtime initialization. `steady mean ms` is the mean of samples 2-10. `all-run mean ms` includes all 10 samples. Edit distance is simple Levenshtein distance after basic text normalization, reported at both character and word level.

For TTS, latency can mean several different things, so this report separates the parts that matter for a realtime system:

- `setup ms`: model/client loading before any synthesis request.
- `first run ms`: first text-to-audio request after setup; this may include lazy warmup.
- `steady mean ms`: synthesis path after warmup, measured over samples 2-10.
- `all-run mean ms`: mean over all 10 requests, including the first run.
- synthesis time: the actual text-to-audio generation path being timed here, including writing the generated WAV artifact for file-producing backends.
- real-time factor: `synthesis_seconds / generated_audio_duration_seconds`. Values below `1.0` mean the backend generates faster than playback. This is not filled in yet because the current report records synthesis timing but does not consistently extract generated audio duration for every TTS backend.

## Samples

The sample WAV files were generated on macOS with `say`, copied to H200, and kept in the workspace:

`voice_bench_runs/voice_microbench_20260710-000106/samples/`

| phrase | duration s | sample rate Hz | frames |
|---|---:|---:|---:|
| cat | 0.534 | 16000 | 8546 |
| dog | 0.441 | 16000 | 7060 |
| apple | 0.464 | 16000 | 7431 |
| red apple | 0.697 | 16000 | 11147 |
| zip zap | 0.697 | 16000 | 11147 |
| green tea | 0.697 | 16000 | 11147 |
| blue car | 0.650 | 16000 | 10403 |
| hello world | 0.836 | 16000 | 13376 |
| animal name | 0.882 | 16000 | 14119 |
| coffee cup | 0.859 | 16000 | 13747 |

## STT Table

`faster-whisper` and Vosk load the model once, then process the 10 sample WAV files. `whisper.cpp` was measured through its CLI per sample; the earlier Mac row used a warmed run because the first run had a large first-use setup outlier. Accuracy is exact normalized phrase match against the sample filename, so it is a strict metric. `X` means unavailable or failed in the tested environment.

| device/config | technology | setup ms | first run ms | steady mean ms | all-run mean ms | exact match | edit distance mean | mismatch | peak RSS MB | GPU memory note | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| Mac, isolated Python venv, CPU | faster-whisper tiny.en int8 | 1676.6 | 202.7 | 159.9 | 164.2 | 9/10 | char 0.1, word 0.1 | `zip zap` -> `Zip sap.` | 653.3 | X | OK |
| Mac, isolated Python venv | Vosk small en-us 0.15 | 288.6 | 318.7 | 345.0 | 342.4 | 10/10 | char 0.0, word 0.0 | none | 653.3 | X | OK |
| Mac, Homebrew CLI | whisper.cpp tiny.en | X | included in CLI call | X | 283.5 | 9/10 | not recorded in latest run | `zip zap` -> `Sip sap.` | 641.0 | X | OK |
| Mac | sherpa-onnx ASR | X | X | X | X | X | X | X | package installed later, but no ASR model config yet |
| H200, isolated Python venv, CPU | faster-whisper tiny.en int8 | 510.0 | 215.4 | 189.1 | 191.8 | 9/10 | char 0.1, word 0.1 | `zip zap` -> `Zip sap.` | 983.8 | ambient H200 GPU load present, not process-specific | OK |
| H200, isolated Python venv, GPU 7 | faster-whisper tiny.en float16 | 1571.4 | 268.1 | 47.4 | 69.5 | 8/10 | char 0.2, word 0.2 | `dog` -> `Doug.`, `zip zap` -> `Zip sap.` | 1306.8 | ambient H200 GPU load present, not process-specific | OK after CUDA 12 venv repair |
| H200, isolated Python venv | Vosk small en-us 0.15 | 312.4 | 352.1 | 362.6 | 361.5 | 9/10 | char 0.6, word 0.1 | `apple` -> `however` | 983.8 | ambient H200 GPU load present, not process-specific | OK |
| H200, built from source CPU CLI | whisper.cpp tiny.en | X | included in CLI call | X | 398.2 | 9/10 | not recorded in latest run | `zip zap` -> `Sip sap.` | 731.9 | ambient H200 GPU load present, not process-specific | OK |
| H200 | sherpa-onnx ASR | X | X | X | X | X | X | X | package installed, but no ASR model config yet |

H200 GPU repair note: CTranslate2/faster-whisper failed initially because the environment exposed CUDA 13 libraries while the wheel wanted `libcublas.so.12`. Installing `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` into the isolated benchmark venv and running with:

```bash
LD_LIBRARY_PATH=/data/home/Andrew.Yang/StreamMUSE/voice-bench-env/lib/python3.12/site-packages/nvidia/cublas/lib:/data/home/Andrew.Yang/StreamMUSE/voice-bench-env/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

made GPU inference work.

## TTS Table

Kokoro and Piper load once, then synthesize the 10 phrases, except the older Piper CLI row is superseded by the persistent Piper row. Direct `espeak-ng` is intentionally included as a low-quality, high-speed baseline.

| device/config | technology | setup ms | first run ms | steady mean ms | all-run mean ms | median ms | min ms | max ms | peak RSS MB | GPU memory note | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Mac, built-in | system `say` | X | included in CLI call | X | 577.7 | 551.5 | 514.0 | 688.8 | 39.8 | X | OK |
| Mac, Homebrew command | `espeak-ng` | X | 80.9 | 27.0 | 32.4 | 27.5 | 25.6 | 80.9 | 5.5 | X | OK |
| Mac, isolated Python venv | Piper, `en_US-amy-low` | X | X | X | X | X | X | X | X | X | failed: packaged `espeak-ng-data/phontab` path missing |
| Mac, isolated Python venv + Homebrew espeak | Kokoro | 4884.8 | 793.5 | 309.7 | 358.1 | 316.8 | 280.1 | 793.5 | 1631.4 | X | OK |
| Mac | sherpa-onnx TTS | X | X | X | X | X | X | X | X | X | package installed, but no TTS model config yet |
| H200, isolated Python venv, persistent Python API | Piper, `en_US-amy-low` | 1265.2 | 23.4 | 27.0 | 26.7 | 26.8 | 21.0 | 31.6 | 137.3 | ambient H200 GPU load present, not process-specific | OK |
| H200, isolated Python venv | Kokoro | 2616.8 | 997.4 | 108.8 | 197.7 | 117.7 | 35.8 | 997.4 | 2374.3 | ambient H200 GPU load present, not process-specific | OK |
| H200 | system TTS | X | X | X | X | X | X | X | X | X | no `say`, `espeak`, or `espeak-ng` command found |
| H200 | `espeak-ng` | X | X | X | X | X | X | X | X | X | apt package exists but sudo password is required |
| H200 | sherpa-onnx TTS | X | X | X | X | X | X | X | X | X | package installed, but no TTS model config yet |

## Raw Artifacts

Local reports:

- `voice_bench_runs/voice_microbench_20260710-000106/`
- `voice_bench_runs/voice_microbench_20260710-001753/`
- `voice_bench_runs/voice_microbench_20260710-011219/`
- `voice_bench_runs/voice_microbench_20260710-011644/`
- `voice_bench_runs/voice_microbench_20260710-012014/`
- `voice_bench_runs/voice_microbench_20260710-081724/`
- `voice_bench_runs/voice_microbench_20260710-084003/`
- `voice_bench_runs/voice_microbench_20260710-084106/`

Copied H200 reports:

- `voice_bench_runs/h200_voice_microbench_20260710-090717/`
- `voice_bench_runs/h200_voice_microbench_20260710-090834/`
- `voice_bench_runs/h200_voice_microbench_20260710-090951/`
- `voice_bench_runs/h200_voice_microbench_20260710-101651/`
- `voice_bench_runs/h200_voice_microbench_20260710-101748/`
- `voice_bench_runs/h200_voice_microbench_20260710-102311/`
- `voice_bench_runs/h200_voice_microbench_20260710-171733/`
- `voice_bench_runs/h200_voice_microbench_20260710-174039/`
- `voice_bench_runs/h200_voice_microbench_20260710-174043/`
- `voice_bench_runs/h200_voice_microbench_20260710-174116/`

Remote H200 working paths:

- `/data/home/Andrew.Yang/StreamMUSE/StreamMUSE-v1/voice_bench_samples_mac/`
- `/data/home/Andrew.Yang/StreamMUSE/voice-bench-env/`
- `/data/home/Andrew.Yang/StreamMUSE/StreamMUSE-v1/voice_bench_models/piper/`

Local Mac working paths:

- `voice-bench-env-mac/`
- `voice_bench_models/piper/`
- `voice_bench_models/hf/`
- `voice_bench_models/vosk/`
- `voice_bench_models/whisper_cpp/`

## Interpretation

For STT, `faster-whisper tiny.en` remains the best general direction. CPU steady-state was about 160 ms/sample on Mac and 189 ms/sample on H200. After repairing CUDA 12 library visibility, H200 GPU steady-state fell to about 47 ms/sample, but the GPU row used `float16` and exact accuracy dropped to 8/10 on these synthetic samples. The GPU setup and first run are much slower than steady state, so a realtime integration should keep the model warm in a resident process.

For TTS, direct `espeak-ng` is the simplest speed baseline on Mac: after startup, it produced short WAVs in about 27 ms. It is low quality, but appropriate as a performance floor. Persistent Piper on H200 is the best measured neural-speed row at about 27 ms steady-state after a 1.3 second setup. H200 Kokoro is heavier to load but still reasonable after warmup at about 109 ms steady-state.

Mac Kokoro now runs after installing Homebrew `espeak-ng` and overriding Kokoro's espeak loader to use the system library/data. Mac Piper still fails because the macOS Piper wheel appears to use a hard-coded build-machine `espeak-ng-data` path that does not respect the explicit data path passed to the Python API.

H200 GPU memory columns should not be interpreted as process memory. At benchmark time, `nvidia-smi` showed active unrelated jobs using multiple GPUs, including VLLM processes. A clean GPU allocation is needed before making claims about H200 GPU memory or GPU speed for STT/TTS.

## Next Benchmark Steps

1. Replace synthetic macOS `say` STT samples with real human recordings for the same 10 phrases; current accuracy is useful but may overfit synthetic speech quirks.
2. Try H200 GPU `faster-whisper` with `int8_float16` or another GPU compute type to see whether accuracy returns to 9/10 while preserving most of the speedup.
3. Add explicit sherpa-onnx ASR/TTS model configs and rerun; this is still a promising lightweight/offline stack.
4. Fix Mac Piper by testing another package version or invoking a standalone binary that uses the correct `espeak-ng-data` path.
5. Add audio duration and real-time factor for generated TTS files.
6. Add persistent process benchmarks for all CLI-based tools, especially `whisper.cpp` and `espeak-ng`, to separate startup cost from per-turn cost.
