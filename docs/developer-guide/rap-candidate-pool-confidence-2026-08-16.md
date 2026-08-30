# Rap Candidate Pool Confidence Experiment - 2026-08-16

## Question

Does the best current lyric prompt produce enough usable candidates per round
that the realtime system can avoid catalog fallbacks while retaining multiple
lines for ranking?

The primary practical targets were:

- at least one selectable candidate in at least 99% of rounds;
- at least three selectable candidates in at least 95% of rounds;
- at least five selectable candidates on average; and
- no selectable candidate in at most 1% of rounds.

In this report, **hard-valid** means that a candidate passes the production
prosody analyzer's hard constraints, including exactly nine spoken syllables.
**Selectable** means hard-valid and ranked at or above the production
experiment's minimum total score of 0.30. The distinction matters because a
line can fit the syllable template but still be rejected for weak stress,
topic, rhyme, continuity, boundary, or novelty scores.

## Configuration

- Client revision: `dcfcd61e` on `feature/real_rap_audio`
- Model host: `Andrew.Yang@masdar`, otherwise unused H200 GPU 0
- Model: `Qwen/Qwen2.5-7B-Instruct`, served as `qwen-rap`
- Serving: vLLM 0.24.0 over an SSH loopback tunnel
- Prompt: production `count_verify` prompt
- Sampling: 36 requested candidates, temperature 0.8, deterministic request
  seed sequence, five-second timeout, and no per-call retry
- Evaluator: production `CmuProsodyAnalyzer` and `rank_candidates`
- Minimum score: 0.30
- Score weights: stress 30%, topic 20%, rhyme 20%, continuity 15%, boundary
  10%, novelty 5%
- Scenario: 300 sequential rounds of the default research demo, evenly split
  among `space`, `deep sea`, and `code`
- Context: the prior four sequentially frozen lines and current rhyme anchors

The three topics map one-to-one to the syncopated, straight, and staggered
nine-slot templates in this scenario. Topic and template effects therefore
cannot be separated in these results.

## Policies

The initial pass made one 36-candidate request in every round. Two adaptive
policies were then replayed from the saved round state:

1. Make a second independent 36-candidate request only when the first
   selectable pool contains fewer than three lines.
2. Make a third independent 36-candidate request only when the combined
   two-batch pool still contains fewer than three lines.

Candidates from additional batches were normalized, deduplicated, analyzed,
and reranked together against the same topic, template, history, and rhyme
anchors as the original round.

## Results

| Policy | Calls | Calls / round | Mean hard-valid | Mean selectable | Median selectable | P05 selectable | >=1 | >=3 | >=5 | Zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One batch | 300 | 1.00 | 7.62 | 5.13 | 4 | 0 | 89.00% | 64.67% | 47.33% | 11.00% |
| Conditional second batch | 406 | 1.35 | 10.01 | 6.82 | 6 | 2 | 98.33% | 93.33% | 69.00% | 1.67% |
| Conditional third batch | 426 | 1.42 | **10.52** | **7.17** | **7** | **3** | **99.67%** | **98.67%** | **73.33%** | **0.33%** |

The second batch was needed in 106 of 300 rounds (35.3%). Only 20 rounds
(6.7%) needed a third batch. The final policy requested 15,336 lines across
426 calls and received 12,885 parseable lines. Every round contained at least
two hard-valid candidates after the adaptive batches.

The final policy passed all four observed targets. Its 95% Wilson intervals
were 98.14-99.94% for at least one selectable candidate and 96.62-99.48% for
at least three. Therefore, the sample demonstrates a 99.67% observed success
rate but does not prove that the true lower-bound success rate is at least 99%.

### Final Policy By Topic And Template

| Topic | Template | Mean selectable | Minimum | P10 | >=1 | >=3 | Zero |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Space | Syncopated | 6.62 | 1 | 3 | 100% | 98% | 0% |
| Deep sea | Straight | 9.18 | 3 | 4 | 100% | 100% | 0% |
| Code | Staggered | 5.70 | 0 | 3 | 99% | 98% | 1% |

### Latency

| Policy | Median | P95 | Maximum |
| --- | ---: | ---: | ---: |
| One batch | 2.287 s | 3.150 s | 5.002 s |
| Conditional second batch | 2.667 s | 5.250 s | 7.352 s |
| Conditional third batch | 2.667 s | 5.807 s | 9.392 s |

The 20 three-batch rounds averaged 6.485 seconds and had an 8.320-second P95.
Across all rounds under the final policy, 262 of 300 (87.33%) completed within
five seconds. The policy therefore meets the candidate-coverage target but
does not meet a strict five-second generation deadline. It may still fit the
prototype's multi-bar lookahead at 60 BPM, which must be verified in the full
generation, audio-rendering, and playback pipeline.

These wall-clock values correct the initial summary, which used the HTTP
client's model-latency field. That field was zero for one request that timed
out after five seconds, causing the original latency table to undercount that
round. Coverage and candidate-pool statistics are unaffected.

## Residual Failure

The final policy produced no selectable candidate in one round. This was not
a syllable-control failure: the combined pool contained 12 hard-valid lines,
but its highest total score was 0.287, below the 0.30 gate. Its best line was:

`Streams process data, with streaming speed`

Recomputing the saved pools at a 0.28 threshold removed all zero-candidate
rounds, but threshold tuning changes the quality contract and was not adopted
as part of this experiment. At the original 0.30 threshold, four rounds had
fewer than three selectable candidates after all three batches.

## Decision

The previous one-call configuration is not reliable enough by itself despite
its healthy average pool. The bounded adaptive policy is the current best
prototype configuration:

1. Request 36 candidates.
2. If fewer than three are selectable, request and merge another 36.
3. If the merged pool still has fewer than three, request and merge a final 36.
4. Preserve the 0.30 score threshold and the existing fallback as a last-resort
   safety path.

This is sufficient to move from isolated lyric generation to an integrated
rolling test. The next test must measure complete deadline success, audio
rendering success, playback underruns, and fallback rates together. If the
five-second budget is strict, parallel speculative batches or a threshold
policy should be evaluated instead of sequential third calls.

## Limitations

- The 300 rounds cover only three topic/template pairs from one scenario.
- Rounds carry sequential context and are not statistically independent; the
  Wilson intervals are descriptive under an approximate independence model.
- Adaptive batches were replayed against saved first-pass round state. Their
  alternate selected lines were not fed into later prompts, so a true rolling
  adaptive run can produce a different context trajectory.
- The benchmark isolates candidate generation and ranking. It excludes audio
  synthesis, scheduler overhead, browser transport, and playback.
- Fixed seeds improve reproducibility but do not characterize all sampling
  behavior or other model-server configurations.

## Artifacts

The full prompts, raw model responses, candidate evaluations, and summaries
are retained for this session under:

- `/tmp/streammuse-rap-confidence-20260816-300r/`
- `/tmp/streammuse-rap-confidence-20260816-retry-under3/`
- `/tmp/streammuse-rap-confidence-20260816-third-under3/`

Each policy JSONL contains 300 records. The summary SHA-256 hashes are:

- one batch: `6b790b31a9dd4ece7bba922918ed79c68e0c3f40a8d6031ce83ea51da8619981`
- two batches: `237761991c9f48b97f25e2e3dca3f7689f2e79a2f0da8518fcde07367644e1d7`
- three batches: `eec5e45ff0122170e0e401e35d1dd912846fc3317346ac5033413fc5e7484a10`
