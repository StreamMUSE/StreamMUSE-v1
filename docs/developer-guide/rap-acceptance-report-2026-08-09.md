# Real-Time Rap Prototype Acceptance Report

Original symbolic acceptance date: 2026-08-09

## Task 10 Audio Extension (2026-08-14)

This appendix is the acceptance record for the split Mac/H200 audio-demo
documentation. It does not replace the historical symbolic evidence below. All
new results were collected on macOS 14.6.1 (23G93), arm64, Python 3.10.18, from
`feature/real_rap` at `eb0cabce15585e1ecb7ed2a6bdaf20813ba8eca4`.

### Automated Evidence

```text
Command: uv run pytest tests/ -q --tb=no
Result: 1140 passed, 4 skipped, 1 warning in 33.15s
```

The suite includes audio coordinator/playback, WAV sink, failure/fallback, and
Start/Stop/Reset lifecycle coverage. This is automated coverage, not a claim
about a physical audio device.

### Local Real Runs

All paths below are ignored local artifacts under
`logs/rap/task10-20260814/`; they were not committed.

| Run | Command shape | Artifact | Observed result |
| --- | --- | --- | --- |
| Device-free phrase bank | `phrase_bank --audio-output wav --tempo 60 --candidate-count 12 --lookahead-bars 3 --max-bars 4 --no-web` | `rap-20260814T152209Z-3f1a2554` | 4 completed bars, 768,000 frames, 0 underruns, 4 commits at 250 ms slack; `mixed.wav` is 48 kHz stereo IEEE float32 (format tag 3), 6,144,000 data bytes. |
| CoreAudio smoke | `phrase_bank --audio-output composite --max-bars 1 --no-web` | `coreaudio-composite/rap-20260814T153505Z-e3c8eba7` | 1 completed bar, 192,000 frames, 0 underruns, and no device-failure event. This proves callback startup/completion, not speaker audibility or latency. |
| Failure fallback | `scripted_failure --audio-output wav --max-bars 2 --no-web` | `scripted-failure/rap-20260814T153521Z-7f5350b7` | 2/2 frozen fallback bars completed, 384,000 frames, 0 underruns; the recorded generator error was retained. |

For the four-bar device-free run, `events.jsonl` contained 36 emitted
syllables and zero nonzero `software_error_samples`; its four committed frame
counts were all 192,000. The WAV frame count equals the `summary.json`
completed-frame total. Derived artifacts were regenerated with
`uv run python scripts/summarize_rap_session.py <session-dir>`.

The eSpeak vocal path is **not accepted** on this Mac. All 63 vocal syllables
in the phrase-bank run recorded `synthesis_failed`; drums made the WAV nonempty
but audible speech was not established. Direct eSpeak output was valid 16-bit
mono PCM at 22,050 Hz but carried a streaming header whose advertised frame
count was 1,073,739,776. The current decoder trusts that count and rejects the
payload. This is a local implementation/environment blocker, not a successful
vocal acceptance result.

Physical speaker latency was not measured. The evidence establishes software
sample placement, callback progress where used, and artifact integrity only.

### Remote Real Attempt

The H200 check used `Andrew.Yang@masdar`. `nvidia-smi` showed GPUs 1 and 2
occupied; GPU 0 was empty, so a single bounded Qwen server was assigned to GPU
0 on loopback port 18001. The first launch failed before serving because
FlashInfer could not find `ninja` in non-interactive `PATH`. The corrected
launch prefixed the existing StreamMUSE environment `bin` directory, became
healthy, and was then tunneled to Mac port 18001.

Two reduced three-bar Mac sessions ran through that tunnel with candidate counts
8 and 12:

```text
logs/rap/task10-20260814/remote-candidates-8/rap-20260814T153206Z-59a8efcd
logs/rap/task10-20260814/remote-candidates-12/rap-20260814T153229Z-37a6b0b4
```

Both sessions completed with deterministic fallbacks, but each recorded two
`generation_error` batches with zero candidates, zero reported latency, and an
empty error string. The temporary vLLM log contained no chat-completion request
line for those batches. Therefore 12-candidate plausibility, remote latency,
and audio-commit latency are **not measured/accepted**; the remaining blocker
is the Mac client/tunnel request path, not H200 GPU availability. The tunnel was
closed, and only the server PID started for this task (`1381284`) was signaled.
Verification afterwards showed GPU 0 at 0 MiB/0% and port 18001 unbound.

### Compatibility And Scope

The accepted phrase-bank `wav` run and the scripted-failure `wav` run are the
bounded executable smokes for fallback/audio behavior. The static planner and
legacy text-only checks remain part of the final Task 10 compatibility gate.
No H200-hosted audio, browser-owned audio, extra website controls, or physical
speaker-latency result is claimed here.

## Scope and Revision

This report closes the symbolic real-time rap prototype tasks. It validates lyric
candidate generation, exact syllable gating, transparent ranking, no-gap
fallbacks, terminal/web observability, deterministic artifacts, and clean
interruption. Live drum input, speech synthesis, audio output, and perceptual
evaluation remain separate research phases.

- Branch: `feature/real_rap`
- Runtime revision used by recorded sessions: `803c2ba65200133950a3bd42d196d825e3623ca4`
- Final tested code revision: `8f0cc65f7cb5ce0c463d2a95b428678831f84a8a`
- Final revision difference: one test-only fix for Rich line wrapping on H200
- H200 checkout: `/data/home/Andrew.Yang/StreamMUSE/real_rap`
- Python: 3.10.20
- Model: `Qwen/Qwen2.5-7B-Instruct`, served as `qwen-rap`
- Server: vLLM 0.24.0 on H200 GPU 1, loopback port 8001
- Temporary accepted-run server PID: 3115764, stopped after testing

## Closed Engineering Tasks

1. Added the deferred MCFlow quantization-tolerance boundary test.
2. Added the empty-corpus catalog CLI artifact test.
3. Made active local-chat requests cancellable during HTTP waits and retry sleep.
4. Preserved one persistent HTTP connection pool across bars.
5. Made shutdown reject new work, cancel once, await loop-side cleanup, join the
   planner, close the transport, and support concurrent `close()` callers.
6. Preserved the historical `requests.Timeout` compatibility contract.
7. Added race tests for pre-registration stop, active HTTP cancellation,
   `KeyboardInterrupt`, delayed async cleanup, repeated cancellation, abort
   errors, and concurrent close.

Client cancellation closes StreamMUSE's HTTP work promptly. It does not prove
that a remote model server always stops inference immediately after disconnect.

## Automated Verification

Local macOS worktree:

```text
993 passed, 4 skipped, 1 pretty_midi/pkg_resources warning
```

H200 affected suite after the terminal-width portability fix:

```text
315 passed, 1 Starlette TestClient deprecation warning
```

The H200 suite covered rap application logic, task runtime compatibility,
inference transport, MCFlow extraction, both rap CLIs, terminal/web monitoring,
and the sample-catalog CLI.

## H200 Workflow Matrix

| Workflow | Session | Result |
| --- | --- | --- |
| 12-bar phrase bank | `rap-20260808T171431Z-c17b02f0` | 12 frozen bars; 1 initial fallback; 85/88 valid candidates; 0 generator errors; 0 deadline misses |
| 12-bar scripted failure | `rap-20260808T171434Z-4ea76674` | 12 frozen fallbacks; 11/11 expected generator errors; 0 deadline misses; continuous output |
| Continuous real Qwen plus Ctrl-C | `rap-20260808T172125Z-2c02733d` | 8 frozen bars; 1 initial fallback; 14/36 valid candidates; 0 generator errors; 0 deadline misses; exit 0 in about 0.53 seconds |
| Real Qwen plus web monitor | `rap-20260808T172254Z-50354e0d` | 21 frozen bars; 5 fallbacks; 44/100 valid candidates; 0 generator errors; 0 deadline misses; REST and WebSocket verified |

Selected latency evidence:

| Workflow | Generation p50 / p95 | Emission jitter p50 / p95 |
| --- | --- | --- |
| Phrase bank | 0.0 / 0.0 ms | 0.245 / 1.591 ms |
| Scripted failure | 0.0 / 0.0 ms | 0.242 / 1.950 ms |
| Continuous Qwen | 248.793 / 315.774 ms | 0.243 / 3.454 ms |
| Qwen web monitor | 251.153 / 435.439 ms | 0.245 / 3.558 ms |

The Qwen sessions commonly returned four lines when eight were requested. This
was retained as an explicit `requested_8_received_4` diagnostic instead of being
hidden. Exact syllable gating rejected malformed lines and ranked the remaining
candidates before each freeze deadline.

## Monitor Evidence

During the live web run, `/api/state` reported session
`rap-20260808T172254Z-50354e0d`, an advancing tick and sequence, and the complete
`baseline_staggered_9` request/template context. `/api/session` returned the
session metadata and artifact directory.

The WebSocket connection received:

```text
snapshot sequence=614
event    sequence=615 event_type=tick
event    sequence=616 event_type=syllable_emitted
```

This verifies snapshot-first connection semantics and ordered live events while
the real model-backed runtime was active.

## Reproduction Commands

Model server used for accepted Qwen runs:

```bash
PATH=/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron/bin:$PATH \
CUDA_VISIBLE_DEVICES=1 \
/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron/bin/vllm serve \
  Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 8001 --served-model-name qwen-rap \
  --max-model-len 2048 --max-num-seqs 8 --gpu-memory-utilization 0.25
```

All demo runs used the following common research settings:

```text
--candidate-count 8 --lookahead-bars 2 --minimum-score 0.3
--seed 20260809 --terminal-layout stream --terminal-detail full
```

The continuous terminal-only real-model command was:

```bash
/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron/bin/streammuse-rap-demo \
  --generator local_chat \
  --model-url http://127.0.0.1:8001/v1 --model qwen-rap \
  --timeout-s 10 --max-bars 0 \
  --candidate-count 8 --lookahead-bars 2 --minimum-score 0.3 \
  --seed 20260809 --terminal-layout stream --terminal-detail full \
  --log-dir /data/home/Andrew.Yang/StreamMUSE/rap_acceptance/20260809/local_chat_qwen_interrupt \
  --no-web
```

The real-model browser/WebSocket command was:

```bash
/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron/bin/streammuse-rap-demo \
  --generator local_chat \
  --model-url http://127.0.0.1:8001/v1 --model qwen-rap \
  --timeout-s 10 --max-bars 0 \
  --candidate-count 8 --lookahead-bars 2 --minimum-score 0.3 \
  --seed 20260809 --terminal-layout stream --terminal-detail full \
  --log-dir /data/home/Andrew.Yang/StreamMUSE/rap_acceptance/20260809/websocket_qwen \
  --host 127.0.0.1 --port 8012
```

The phrase-bank and scripted-failure commands changed `--generator`, used
`--max-bars 12 --no-web`, and wrote to the corresponding `phrase_bank` and
`scripted_failure` directories under the same acceptance root.

## Artifact Reproducibility

For all four accepted sessions, the following command regenerated metrics and
per-bar output from canonical events:

```bash
python scripts/summarize_rap_session.py <session-directory>
```

All eight comparisons passed byte-for-byte:

```text
summary.json == summary.regenerated.json   (4/4 sessions)
bars.csv    == bars.regenerated.csv        (4/4 sessions)
```

Raw H200 evidence is under
`/data/home/Andrew.Yang/StreamMUSE/rap_acceptance/20260809/`.

## Excluded Attempts and Environment Notes

- Session `rap-20260808T171614Z-7a80f02b` is excluded from real-model metrics.
  The previously running vLLM process disappeared before its requests, so the
  demo correctly logged connection errors and retained fallbacks.
- The first fresh vLLM launch failed because non-interactive SSH omitted the
  existing Conda `ninja` binary from `PATH`. Relaunching with the environment's
  `bin` directory on `PATH` succeeded without installing or modifying packages.
- The first H200 affected-suite run exposed a Rich-version wrapping difference
  in one dashboard assertion. The assertion now normalizes whitespace and the
  complete affected remote suite passes.
- The temporary accepted-run vLLM server was terminated after testing. No
  StreamMUSE rap or vLLM process was left running, and GPU 1 returned to zero
  allocated memory.

## Remaining Research Work

- Connect live keyboard/drum rhythm and changing tempo to flow-template choice.
- Add beat-aware speech or singing synthesis and scheduled audio playback.
- Expand beyond the three built-in nine-slot templates using the sampled MCFlow
  catalog and evaluate how template diversity affects generated delivery.
- Evaluate lyrical quality, perceived flow, topic adherence, and fallback
  transitions with listeners rather than treating heuristic scores as quality.
- Measure end-to-end audio latency and server-side cancellation behavior.
