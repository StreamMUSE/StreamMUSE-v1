# Rap API Multi-Choice Generation Experiment - 2026-08-16

## Question

Can the OpenAI-compatible `n=N` chat-completion parameter use the H200 to
generate more than 100 short lyric candidates in one request, quickly enough
to replace the current single multiline completion and nearly eliminate
candidate-quality fallbacks?

This is a throwaway H200 feasibility experiment. It does not change the
production client or candidate policy.

## Mechanism

The current generator asks one assistant completion to print many newline-
separated lyric lines. Those lines are produced by one autoregressive sequence,
so later lines condition on earlier lines.

The multi-choice path instead sends one prompt that asks for one lyric line and
sets `n=N`. vLLM returns `N` indexed choices. Each choice is a separate short
decode from the shared prompt. The choices are sampled independently enough to
increase diversity, but they are not guaranteed to be unique. The client must
parse every `choices[i]`, normalize, deduplicate, analyze, and rerank the merged
pool.

The committed client currently reads only `choices[0]`, so this experiment used
a temporary raw HTTP harness.

## Setup

- Model: `Qwen/Qwen2.5-7B-Instruct`, served as `qwen-rap`
- Host: otherwise idle H200 GPU 0
- vLLM: 0.24.0, `--max-num-seqs 128`, model endpoint bound to loopback
- Client: current production `count_verify` prompt and saved rolling contexts
- Multi-choice sampling: temperature `1.0`, `top_p=0.95`, no API sampling seed
- Auditability: the deterministic variation seed remains in the prompt
- Per-choice limit: 32 completion tokens and one parsed lyric line
- Evaluator: production CMU prosody analyzer and ranker
- Hard gate: exactly nine spoken syllables
- Selectable gate: hard-valid and total score at least `0.30`
- Topics/templates: equal coverage of space/syncopated, deep sea/straight, and
  code/staggered

## Sampling Probe

The first `n=100` code request used temperature `0.8` and one shared API seed.
It collapsed to five unique lines and no exact nine-syllable line. Omitting the
API seed and increasing sampling diversity changed the result:

| Code-context `n=100` configuration | Unique | Exact nine-syllable | Latency |
| --- | ---: | ---: | ---: |
| Temperature 0.8, shared API seed | 5 | 0 | 5.40 s |
| Temperature 0.8, no API seed | 12 | 1 | 5.59 s |
| Temperature 1.0, no API seed | 35 | 5 | 4.07 s |
| Temperature 1.2, no API seed | 60 | 13 | 4.82 s |
| Temperature 1.0, `top_p=0.95`, no API seed | **69** | **13** | **1.76 s** |

This established that raw fan-out is not sufficient. Sampling diversity is a
required part of the policy.

## Matched 30-Context Sweep

Each row contains 30 matched contexts, 10 per topic/template pair. Counts are
means per round. Latency is request wall time.

| Policy | Unique | Hard-valid | Selectable | >=1 | >=3 | Mean / P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current multiline 36 | 33.10 | 4.77 | 2.83 | 73.3% | 43.3% | 2.39 / 3.66 s |
| API choices 36 | 29.97 | 6.03 | 5.50 | 93.3% | 66.7% | **1.12 / 1.26 s** |
| API choices 64 | 51.07 | 11.40 | 10.50 | **100%** | 93.3% | 1.73 / 2.38 s |
| API choices 100 | 74.30 | 17.63 | 16.47 | **100%** | 93.3% | 2.57 / 3.23 s |
| API choices 128 | 92.97 | 21.20 | 19.37 | **100%** | **96.7%** | 3.11 / 3.87 s |

`n=128` requested 3.56 times as many raw lines as the multiline-36 baseline,
yet its P95 was only about 0.21 seconds slower in this matched sweep. It
produced 6.84 times as many selectable candidates on average and a higher mean
best-candidate score (`0.632` versus `0.438`).

## Saturation Probe

Increasing beyond 128 continued to add candidates but had diminishing returns.
Successful calls only:

| Policy | Successful calls | Unique | Hard-valid | Selectable | Mean / P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| API choices 192 | 26 / 30 | 132.46 | 35.23 | 32.04 | 4.51 / 8.31 s |
| API choices 256 | 27 / 30 | 159.48 | 38.26 | 34.15 | 7.83 / 13.47 s |

Moving from 192 to 256 added only about 2.1 selectable candidates while adding
3.3 seconds to mean latency. Both configurations also crossed the current
SSH path's reliable operating range more often. They are not useful defaults
for the realtime prototype.

## 300-Context Confidence Run

The selected `n=128` configuration was run across all 300 saved rolling
contexts, 100 per topic/template pair.

### First Attempt Only

- Successful model responses: 288 / 300
- SSH transport failures: 12 / 300
- Successful responses with at least one selectable line: **288 / 288**
- Successful responses with at least three selectable lines: **287 / 288**
- Mean successful pool: 99.54 unique, 33.85 hard-valid, 31.55 selectable
- Successful latency: 4.63 s mean, 3.81 s median, 8.78 s P95, 21.89 s maximum

Every response received from the model contained a selectable candidate. The
only first-attempt zero pools were transport failures.

### One Transport-Only Retry

Exactly the 12 transport failures were retried once. Eleven recovered and one
failed again.

| Final pool target | Successful rounds | Observed rate | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| At least one selectable | 299 / 300 | **99.67%** | 98.14-99.94% |
| At least three selectable | 298 / 300 | **99.33%** | 97.60-99.82% |
| At least five selectable | 297 / 300 | **99.00%** | 97.10-99.66% |
| At least ten selectable | 286 / 300 | 95.33% | 92.32-97.20% |

Final successful rounds averaged 99.13 unique, 33.55 hard-valid, and 31.31
selectable candidates. The policy averaged 1.04 requests and 133.12 requested
choices per round.

End-to-end latency, including the first failed attempt before a retry, was
4.82 seconds mean, 3.90 seconds median, 9.48 seconds P95, and 22.99 seconds
maximum. The fallback catalog remains necessary for the one residual transport
failure and for hard realtime deadlines.

The H200 log contained 521 successful chat-completion POSTs and no HTTP 5xx,
traceback, or engine error. The observed request failures were SSH transport
disconnects rather than rejected or failed model generations.

## Realtime Implications

For a four-beat bar, the candidate-only P95 of 9.48 seconds corresponds to
these conservative tempo ceilings:

| Lookahead | P95 tempo ceiling |
| --- | ---: |
| 1 bar | 25.3 BPM |
| 2 bars | 50.6 BPM |
| 3 bars | 75.9 BPM |
| 4 bars | 101.2 BPM |

At 60 BPM, three bars of lookahead cover the sustained-run candidate P95. At
92 BPM, four bars are the conservative configuration. These values exclude
audio rendering and scheduler overhead. Deadline expiry must still select a
ready fallback rather than wait for a late response.

## Decision

The API multi-choice path is substantially better than asking one completion
to print 36 lines. For the current quality-first prototype:

1. Request `n=128` one-line choices with temperature `1.0`, `top_p=0.95`, and
   no API sampling seed.
2. Keep the prompt-visible variation seed for context auditability.
3. Parse all choices, normalize, deduplicate, analyze, and rank one merged pool.
4. Retry once only when the HTTP request fails. Do not request another batch
   after a successful response merely to enlarge an already deep pool.
5. Retain the prevalidated fallback for the residual transport failure and
   deadline misses.
6. Use three bars of lookahead for a 60 BPM demo and four bars near 92 BPM
   until end-to-end audio measurements justify a smaller window.

`n=64` is the likely latency-optimized future policy, but it has only a
30-context sample. `n=128` has the stronger 300-context confidence result and
is the safer next integration target.

## Follow-Up: `n=64` Confidence Run

A later 300-context `n=64` run supersedes the 30-context uncertainty noted
above. With one transport-only retry, it produced at least one selectable
candidate in 300/300 rounds and at least three in 299/300, averaging 16.63
selectable candidates. However, its retry-inclusive P95 was 6.266 seconds and
only 279/300 candidate stages completed within the full 5.333-second two-bar
window at 90 BPM. See
[`rap-n64-confidence-experiment-2026-08-16.md`](rap-n64-confidence-experiment-2026-08-16.md)
for the full result.

## Artifacts

- Harness: `/tmp/rap_multichoice_benchmark.py`
- Initial semantic smoke: `/tmp/streammuse-rap-multichoice-smoke-20260816/`
- Matched count sweep: `/tmp/streammuse-rap-multichoice-t10p95-20260816/`
- Saturation sweep: `/tmp/streammuse-rap-multichoice-scaling-20260816/`
- 300-context run and retry attempts:
  `/tmp/streammuse-rap-multichoice-n128-confidence-20260816/`
