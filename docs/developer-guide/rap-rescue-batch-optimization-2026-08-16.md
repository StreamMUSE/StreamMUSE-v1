# Rap Rescue Batch Optimization - 2026-08-16

## Question

Can the realtime rap generator preserve near-guaranteed candidate coverage
without requesting another 36 lines every time the first pool is too small?

The experiment retained the previous configuration: Qwen2.5-7B-Instruct,
production `count_verify` prompt, 0.30 minimum score, and an initial batch of
36 candidates. A successful policy still had to achieve:

- at least one selectable candidate in at least 99% of rounds;
- at least three selectable candidates in at least 95% of rounds;
- at least five selectable candidates on average; and
- zero selectable candidates in at most 1% of rounds.

## Rescue Size Sweep

Six rescue sizes were tested through 636 real H200 requests over the same 106
low-pool contexts from the 300-round confidence experiment. Each size had 106
requests and no request errors.

| Rescue size | Mean call latency | P95 call latency | Returned/requested | >=1 after one rescue | >=3 after one rescue |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.815 s | 0.978 s | 67.8% | 94.67% | 76.33% |
| 12 | 1.073 s | 1.248 s | 86.6% | 97.00% | 84.00% |
| 16 | 1.227 s | 1.508 s | 83.6% | 96.33% | 84.00% |
| 20 | 1.412 s | 1.577 s | 88.8% | 98.00% | 88.67% |
| 24 | 1.683 s | 1.953 s | 90.5% | 98.00% | 93.00% |
| 36 | 2.208 s | 3.045 s | 83.5% | 98.67% | 90.67% |

Larger requests generally cost more, but coverage was not monotonic because
the model can stop early or enter a context-specific wrong-length pattern.
Requested count is therefore a completion budget, not a guaranteed number of
returned or valid lines.

## Policy Search

The measured batches were scored with the production prosody analyzer and
ranker. A search evaluated 3,136 monotonic two-rescue policies, where larger
candidate deficits could not receive smaller budgets. Of these, 684 met the
targets on the sweep sample.

The lowest-P95 search policy was:

`36 -> 20 -> 12 when still at zero, otherwise 8`

It was rejected after independent confirmation. One 300-round confirmation
passed, but another reached only 98.67% with at least one candidate and 90.00%
with at least three. Eight-line final rescues were too variable on contexts
that had already failed twice.

## Selected Policy

The best stable policy tested was deliberately simpler:

1. Generate 36 candidates.
2. If fewer than three are selectable, generate 24 more.
3. If the combined pool still has fewer than three, generate 20 more.
4. Normalize, deduplicate, and rerank the combined pool after each batch.
5. Retain the prevalidated fallback as the final safety path.

It uses no deficit-specific branches after the stop condition. The adaptive
behavior comes from stopping as soon as three selectable lines exist.

## Independent Confirmations

Two fresh-seed confirmations replayed all 300 logical rounds independently.
Both used the same saved first-pass contexts but newly sampled rescue batches.

| Metric | Confirmation A | Confirmation B | Combined |
| --- | ---: | ---: | ---: |
| Logical rounds | 300 | 300 | 600 |
| At least one selectable | 300 (100%) | 300 (100%) | **600 (100%)** |
| At least three selectable | 291 (97.00%) | 290 (96.67%) | **581 (96.83%)** |
| At least five selectable | 202 (67.33%) | 203 (67.67%) | 405 (67.50%) |
| Mean selectable pool | 6.64 | 6.64 | **6.64** |
| P05 selectable pool | 3 | 3 | **3** |
| Mean hard-valid pool | 9.98 | 9.95 | **9.97** |
| Fallback rounds | 0 | 0 | **0** |
| Second batches | 106 | 106 | 212 |
| Third batches | 39 | 39 | 78 |
| Logical policy calls | 445 | 445 | 890 |
| Mean requested lines/round | 47.08 | 47.08 | **47.08** |

Each confirmation made 145 new H200 rescue calls: 106 second batches and 39
third batches. The logical totals include the 300 saved first-pass calls that
would occur in a live run.

Under an approximate independence model, the combined 95% Wilson lower bound
was 99.36% for at least one candidate and 95.11% for at least three. The first
pass and contexts were repeated, so these intervals should be treated as
descriptive evidence rather than a fully independent 600-round trial.

## Latency And BPM

Candidate-stage latency is reconstructed from the saved initial wall-clock
duration plus the new rescue-call, prosody-analysis, and reranking durations.
It excludes audio synthesis, scheduler overhead outside this stage, and
playback.

Across the 600 confirmation rounds:

- mean: 3.055 seconds;
- median: 2.664 seconds;
- P95: 5.330 seconds; and
- observed maximum: 7.601 seconds.

For 4/4 time, one bar lasts `240 / BPM` seconds. With `L` bars of lookahead,
the candidate stage fits when `latency <= 240 * L / BPM`. The resulting
candidate-only BPM ceilings are:

| Lookahead | P95 ceiling | Observed-maximum ceiling |
| ---: | ---: | ---: |
| 1 bar | 45.0 BPM | **31.6 BPM** |
| 2 bars | 90.1 BPM | **63.2 BPM** |
| 3 bars | 135.1 BPM | **94.7 BPM** |

The observed-maximum column is the stricter answer for the measured sample.
It is not an end-to-end safe operating BPM because audio and scheduling still
need time. Rounding those ceilings to 30 BPM with one bar, 60 BPM with two
bars, or 90 BPM with three bars leaves only about 0.4 seconds beyond the
observed candidate-stage maximum. At 60 BPM, three bars provide a 12-second
window and approximately 4.4 seconds for the remaining pipeline, so three
bars at 60 BPM remains the recommended integrated-demo starting point.

## Comparison With Fixed 36-Line Rescues

The old policy requested 36 lines for both rescue stages. Its latency has been
recomputed with wall-clock timeout accounting.

| Metric | Fixed `36/36/36` | Adaptive `36/24/20` | Change |
| --- | ---: | ---: | ---: |
| Mean requested lines/round | 51.12 | 47.08 | -7.9% |
| Mean latency | 3.244 s | 3.055 s | -5.8% |
| P95 latency | 5.807 s | 5.330 s | -8.2% |
| Observed maximum | 9.392 s | 7.601 s | -19.1% |
| Mean selectable pool | 7.17 | 6.64 | -7.4% |

The optimized policy makes slightly more calls because a 24-line second batch
leaves more rounds requiring a third batch. It still reduces generated lines
and latency while retaining the target pool coverage across both confirmation
runs.

## Decision

Use `36 -> 24 -> 20`, stopping as soon as the combined pool reaches three
selectable candidates. This is the best tested policy, not a global optimum:
it was selected from six rescue sizes, one scenario, three topic/template
pairs, and two independent rescue confirmations. The next milestone is to
implement it behind the current generator interface and validate end-to-end
deadline, audio-rendering, and underrun rates in the rolling system.

## Artifacts

- `/tmp/streammuse-rap-rescue-sweep-20260816-8-20/`
- `/tmp/streammuse-rap-rescue-sweep-20260816-12-24/`
- `/tmp/streammuse-rap-rescue-sweep-20260816-16-36/`
- `/tmp/streammuse-rap-rescue-policy-search-20260816.json`
- `/tmp/streammuse-rap-adaptive-confirm-20260816/`
- `/tmp/streammuse-rap-adaptive-confirm-b-20260816/`
- `/tmp/streammuse-rap-adaptive-24-20-confirm-a-20260816/`
- `/tmp/streammuse-rap-adaptive-24-20-confirm-b-20260816/`
