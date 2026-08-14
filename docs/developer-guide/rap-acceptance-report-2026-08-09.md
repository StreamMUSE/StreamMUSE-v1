# Real-Time Rap Prototype Acceptance Report

Original symbolic acceptance date: 2026-08-09

## Task 10 Audio Extension (2026-08-14, Round 1 Fix)

This appendix records the split Mac/H200 audio-demo acceptance after repairing
Homebrew eSpeak streaming WAV decoding and local-chat transport diagnostics. It
does not replace the historical symbolic evidence below. Results were collected
on macOS 14.6.1 (23G93), arm64, Python 3.10.18, from `feature/real_rap`.
Artifacts below are ignored local files and were not committed.

### Automated Evidence

```text
uv run pytest tests/ -q --tb=no
1145 passed, 4 skipped, 1 warning in 28.21s

Focused decoder/transport/audio diagnostics
70 passed in 1.45s

HTTP/runtime lifecycle tests
3 passed in 0.83s
```

The added coverage includes an oversized eSpeak 1.52-style RIFF/data fixture,
a real local `espeak-ng` smoke with nonempty PCM and no `synthesis_failed`,
misaligned PCM rejection, a blank `httpx.ReadTimeout` diagnostic with a
sanitized URL, and generator-warning propagation. The lifecycle tests exercise
Start/Stop/Reset/Start/Stop, complete-bar stopping, and reset epoch cleanup.
This is automated evidence, not a physical-device claim.

### Local Real Evidence

The accepted device-free command was:

```bash
uv run streammuse-rap-demo --generator phrase_bank --audio-output wav \
  --tempo 60 --candidate-count 12 --lookahead-bars 3 --max-bars 12 \
  --terminal-layout stream --terminal-detail summary --no-web \
  --log-dir logs/rap/task10-round1/phrase-bank-12bar-rerun
```

Artifact:
`logs/rap/task10-round1/phrase-bank-12bar-rerun/rap-20260814T155002Z-7ab94c2a`.
It completed 12 bars and 2,304,000 frames with zero underruns and zero
nonzero `software_error_samples`. `mixed.wav` is nonempty 48 kHz stereo IEEE
float32 (format tag 3), 32-bit, with 2,304,000 frames. The regenerated summary
matched the canonical events.

The 24 accepted `audio_render_completed` events report 216 vocal syllables,
4,095,213 vocal source frames, 3,662,315 fitted vocal frames, and
`{"cmudict_arpabet": 216}` pronunciation sources. `synthesis_failed` is zero.
Those diagnostics establish nonempty eSpeak vocal PCM independently of the
drum bed; they support the quickstart's audible-vocal-PCM claim. Physical
speaker audibility and speaker latency were not measured.

The bounded failure command was:

```bash
uv run streammuse-rap-demo --generator scripted_failure --audio-output wav \
  --tempo 60 --candidate-count 12 --lookahead-bars 3 --max-bars 8 \
  --terminal-layout stream --terminal-detail summary --no-web \
  --log-dir logs/rap/task10-round1/scripted-failure-8bar
```

Artifact:
`logs/rap/task10-round1/scripted-failure-8bar/rap-20260814T155113Z-b9b9a5ba`.
All 8 bars completed (1,536,000 frames), all froze to fallback, 7/7 planner
requests retained `generation_error`, and underruns remained zero.

### Remote Real Evidence

The H200 host was `Andrew.Yang@masdar`. A read-only `nvidia-smi` preflight found
GPU 2 occupied by PID 1211284 and GPU 0 empty; one temporary Qwen vLLM process
(PID 1702486) was bound to GPU 0 and loopback port 18001. Its environment put
`/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron/bin` first in
`PATH` for `ninja`/FlashInfer.

The temporary SSH tunnel passed both preflights:

```text
GET http://127.0.0.1:18001/v1/models: qwen-rap listed
POST http://127.0.0.1:18001/v1/chat/completions: content "ok"
```

Three bounded three-bar, 60 BPM, `--audio-output none --no-web` runs then used
the verified tunnel:

| Requested candidates | Artifact | Returned batches | Result |
| --- | --- | --- | --- |
| 8 | `h200-candidates-8/rap-20260814T155515Z-069a492b` | 4, 4 | no transport errors; both late; `requested_8_received_4` recorded |
| 12 | `h200-candidates-12/rap-20260814T155536Z-2c935d84` | 11, 8 | no transport errors or deadline misses; shortfall warnings recorded |
| 16 | `h200-candidates-16/rap-20260814T155557Z-cabe7685` | 8, 8 | no transport errors or deadline misses; shortfall warnings recorded |

This establishes that requests configured for 8, 12, and 16 candidates reach
the H200 through the tunnel at 60 BPM. It does **not** establish that the model
reliably returns 12 complete candidates: the 12-request batches returned 11
and 8, and the adapter retained those discrepancies as warnings. The tunnel was
closed and only PID 1702486 was stopped. Post-stop checks showed GPU 0 at
0 MiB/0% and port 18001 unbound.

### Compatibility And Scope

The accepted phrase-bank `wav` run and the scripted-failure `wav` run are the
bounded executable smokes for fallback/audio behavior. The static planner and
legacy text-only checks remain part of the final Task 10 compatibility gate.
No H200-hosted audio, browser-owned audio, extra website controls, or physical
speaker-latency result is claimed here.

## Scope and Revision

The historical symbolic report validates lyric candidate generation, exact
syllable gating, transparent ranking, no-gap fallbacks, terminal/web
observability, deterministic artifacts, and clean interruption. The Task 10
appendix adds device-free eSpeak vocal PCM and WAV acceptance; live drum input,
perceptual evaluation, and physical speaker latency remain separate research
work.

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
