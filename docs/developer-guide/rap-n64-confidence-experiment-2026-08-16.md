# Rap API `n=64` Confidence Experiment - 2026-08-16

## Question

Does one API-level `n=64` request produce a sufficiently reliable candidate
pool for the 90 BPM prototype, and does it complete within a strict two-bar
lookahead window?

This experiment measures the H200-hosted text stage through the intended Mac
client and SSH-tunnel architecture. It does not include speech rendering,
audio transfer, or scheduler overhead.

## Configuration

- Model: `Qwen/Qwen2.5-7B-Instruct`, served as `qwen-rap`
- Server: vLLM 0.24.0 on otherwise idle H200 GPU 0
- vLLM concurrency: `--max-num-seqs 128`
- Client: Mac, connected through an SSH local-forward tunnel
- Contexts: 300 saved rolling contexts, 100 per topic/template pair
- Generation: 64 API-level choices, one line per choice
- Sampling: temperature `1.0`, `top_p=0.95`, no API sampling seed
- Per-choice limit: 32 completion tokens
- Hard gate: exactly nine CMU-analyzed syllables
- Selectable threshold: hard-valid and score at least `0.30`
- Retry policy: retry once only after a transport error

## Candidate Confidence

The first pass returned 294 successful model responses and six transient
`RemoteProtocolError` failures. Every successful response contained at least
one selectable candidate. Retrying only the six failed transports recovered
all six.

| Final pool target | Successful rounds | Observed rate | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| At least one selectable | 300 / 300 | **100.00%** | 98.74-100.00% |
| At least three selectable | 299 / 300 | **99.67%** | 98.14-99.94% |
| At least five selectable | 287 / 300 | 95.67% | 92.73-97.45% |
| At least ten selectable | 252 / 300 | 84.00% | 79.43-87.71% |

The final pools averaged 53.50 unique, 17.81 hard-valid, and 16.63 selectable
candidates. The smallest pool contained two selectable candidates; no final
round required a no-candidate fallback.

| Topic/template | Mean selectable | At least one | At least three |
| --- | ---: | ---: | ---: |
| Code/staggered | 13.77 | 100 / 100 | 100 / 100 |
| Deep sea/straight | 20.73 | 100 / 100 | 100 / 100 |
| Space/syncopated | 15.40 | 100 / 100 | 99 / 100 |

## Latency At 90 BPM

Two four-beat bars at 90 BPM provide exactly `480 / 90 = 5.333` seconds.

| Latency view | Mean | Median | P95 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final successful response | 3.477 s | 3.127 s | 5.932 s | 18.744 s | 25.826 s |
| Retry-inclusive policy | 3.497 s | 3.137 s | **6.266 s** | 18.744 s | 25.826 s |

Retry-inclusive candidate-stage deadline coverage was:

| Candidate deadline | Completed rounds | Rate | Remaining two-bar time |
| ---: | ---: | ---: | ---: |
| 4.0 s | 227 / 300 | 75.67% | 1.333 s |
| 4.5 s | 264 / 300 | 88.00% | 0.833 s |
| 5.0 s | 274 / 300 | 91.33% | 0.333 s |
| 5.333 s | 279 / 300 | **93.00%** | 0 s |

The full two-bar candidate-only deadline was missed in 21 of 300 rounds. The
integrated hit rate must be lower because audio still needs a nonzero reserve.
The 25.8-second maximum also demonstrates that a hard deadline and ready
fallback are mandatory even when candidate quality is strong.

## Server And Transport Audit

The H200 log contained 301 successful chat-completion POSTs: one smoke call,
294 successful first attempts, and six successful retries. It contained no
HTTP 5xx response, traceback, engine-death message, or vLLM exception. The six
first-attempt `RemoteProtocolError` calls did not reach vLLM, so they are
classified as SSH/HTTP transport failures rather than model failures.

## Decision

`n=64` satisfies the candidate-availability objective: the measured one-retry
policy produced at least one selectable candidate in every round and at least
three in 299 of 300 rounds. It does **not** by itself satisfy strict two-bar
operation at 90 BPM: candidate generation alone met that complete window in
only 93% of rounds and its P95 exceeded the window by 0.932 seconds.

The next realtime policy should therefore keep `n=64` as the quality target
only behind a hard candidate deadline and prevalidated fallback. Before fixing
that deadline, test lower per-choice token limits and smaller API choice counts
to recover audio reserve while measuring the resulting pool-confidence loss.

## Artifacts

- Harness: `/tmp/rap_multichoice_benchmark.py`
- Retry-inclusive analysis: `/tmp/analyze_n64_confidence.py`
- Valid raw records and summary:
  `/tmp/streammuse-rap-multichoice-n64-confidence-20260816-valid/`
- Rejected sandbox-only run:
  `/tmp/streammuse-rap-multichoice-n64-confidence-20260816/`
- H200 log: `/tmp/vllm-rap-n64-confidence-20260816.log`
