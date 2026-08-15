# Rap Prompt And Model Experiment - 2026-08-15

## Question

Can prompt engineering improve exact nine-syllable validity and candidate
ranking quality, and does Qwen2.5-14B understand the constraint better than
Qwen2.5-7B?

This is a bounded prototype experiment, not a statistically powered language
model evaluation. Results are useful for choosing the next demo configuration,
but stochastic runs should not be interpreted as stable model-wide rates.

## Setup

- Client base revision: `38d94a7b` on `feature/real_rap_audio`
- Model host: `Andrew.Yang@masdar`, unused H200 GPU 0
- Models: `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-14B-Instruct`
- Serving: vLLM 0.24.0 over an SSH loopback tunnel
- Evaluator: production `CmuProsodyAnalyzer` and `rank_candidates`
- Hard gate: exactly nine CMU-analyzed spoken syllables
- Score weights: stress 30%, topic 20%, rhyme 20%, continuity 15%,
  boundary 10%, novelty 5%
- Sampling: temperature 0.8 with matched fixed seeds in offline comparisons

The broad case set contained six topic/template/history combinations across
the straight, syncopated, and staggered nine-slot templates. A second case set
matched the first eight bars of the actual looping scenario: four `space` bars
followed by four `deep sea` bars.

## Prompt Sweep

Eighteen one-call prompt strategies were tested on 7B. They included the
production baseline, internal count verification, stress explanations,
verified examples, lexical word shapes, slot scaffolds, explicit score
weights, concise score priorities, literal-topic requirements, nine-word
monosyllabic constraints, topic-specific length anchors, contrastive wrong and
right examples, and calibrated ten- or eleven-syllable requests. An additional
analyzer-feedback repair strategy used a second model call.

The first broad sweep established that concise internal verification was more
effective than detailed metrical instruction:

| Prompt | Exact valid / returned | Valid yield / requested | Requests with valid | Mean top score | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 12 / 44 (27.3%) | 16.7% | 5 / 6 | 0.468 | 1.05 s |
| Count and verify | 21 / 67 (31.3%) | 29.2% | 5 / 6 | **0.587** | 1.10 s |
| Stress legend | 12 / 70 (17.1%) | 16.7% | 5 / 6 | 0.464 | 1.06 s |
| Few-shot form | 5 / 66 (7.6%) | 6.9% | 4 / 6 | 0.438 | 1.13 s |
| Slot scaffold | 11 / 61 (18.0%) | 15.3% | 4 / 6 | 0.463 | 1.00 s |
| Full combined prompt | 15 / 70 (21.4%) | 20.8% | 4 / 6 | 0.479 | 1.10 s |

More instruction was not consistently better. The nine-one-syllable-word
prompts had only 4.3-7.6% exact validity because the model still selected
multisyllabic words. Topic anchors and calibrated ten/eleven-syllable targets
also failed to outperform direct verification. The analyzer-feedback repair
pass reached 20.0% exact validity but cost two calls and covered only three of
six requests, so it was rejected for this version.

On the exact runtime topic distribution with 12 requested candidates:

| Prompt | Exact validity | Valid yield | Requests with valid | Mean top score | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 11.9% | 10.4% | 3 / 8 | 0.467 | 1.13 s |
| Count and verify | 20.5% | 17.7% | 7 / 8 | **0.499** | 1.11 s |
| Compact rank priorities | **25.6%** | **24.0%** | 7 / 8 | 0.480 | 1.33 s |
| Count plus topic anchor | 24.2% | 22.9% | 7 / 8 | 0.480 | 1.08 s |

`compact_rank` had the best aggregate validity but was unstable on individual
`space` requests. `count_verify` had the best top score, remained concise, and
was more consistent across the broad and scenario-specific sweeps. It was
therefore selected for the production prompt.

## Candidate Count

The rolling system had substantial unused lookahead. A 36-candidate sweep on
the exact runtime distribution tested whether that budget could improve the
selection pool.

| Prompt, 36 requested | Returned | Exact valid | Requests with valid | Mean top score | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 266 / 288 | 36 (13.5%) | 7 / 8 | 0.474 | 2.63 s |
| Count and verify | 277 / 288 | **40 (14.4%)** | 7 / 8 | **0.531** | **2.31 s** |
| Compact rank priorities | 287 / 288 | 39 (13.6%) | 6 / 8 | 0.520 | 2.90 s |

Requesting 36 lines did not substantially improve the per-line validity rate.
It increased the absolute valid pool from 2.1 to 5.0 candidates per request and
raised the selected-score ceiling while remaining within the slow-tempo
lookahead budget.

## Model Size

The three finalist prompts were repeated on Qwen2.5-14B using the same broad
cases and seeds.

| Prompt | 7B exact validity | 14B exact validity | 7B latency | 14B latency |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 27.3% | 13.9% | 1.05 s | 1.89 s |
| Count and verify | **31.3%** | 21.5% | 1.10 s | 1.76 s |
| Compact rank priorities | **41.4%** | 22.7% | 1.03 s | 1.71 s |

The 14B model was slower and less syllable-valid in this sample. Larger model
size did not substitute for an external pronunciation analyzer or constrained
decoding. The prototype should retain 7B.

## Rolling Confirmation

The selected production prompt was tested in the real Mac client plus H200
server pipeline at 60 BPM, three bars of lookahead, 36 candidates, and WAV
audio output. Two score policies were separated:

| Minimum score | Exact candidates | Generated eligible bars | Avoidable fallbacks | Median generation | Deadline misses | Underruns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.55 CLI default | 46 / 233 (19.7%) | 4 / 7 | 3 / 7 | 2.35 s | 0 | 0 |
| 0.30 prototype | **98 / 240 (40.8%)** | **7 / 7** | **0 / 7** | 2.68 s | 0 | 0 |

The first bar is intentionally prevalidated in both runs. At threshold 0.55,
all three avoidable fallbacks had exact candidates, but their best total score
was below the quality threshold. At threshold 0.30, every eligible bar selected
and rendered a generated nine-syllable line. The final run had no generator
errors, pronunciation fallbacks in committed audio, synthesis failures,
deadline misses, forced bar fits, or playback underruns.

Selected lines in the final run included:

- `Infinite space, our cosmic muse`
- `shooting stars ignite the endless night`
- `Stars ignite like distant torches bright`
- `Into the unknown, into the night`
- `frozen waves embrace the deep, dark night`
- `ancient ruins lie beneath the waves`
- `Flight into the deep, where spirits light`

## Decision

1. Keep Qwen2.5-7B for the current prototype.
2. Use the concise internal spoken-count verification prompt now committed to
   the local-chat generator.
3. For the slow live demo, request 36 candidates, use three bars of lookahead,
   and set `--minimum-score 0.30` explicitly.
4. Keep CLI defaults unchanged until candidate count and score threshold are
   tested across faster tempos and more scenarios.
5. Treat prompt-only syllable control as probabilistic. A future reliability
   milestone should test analyzer-guided repair or multiple bounded batches,
   but the simple second-call repair prompt tested here is not adequate.

## Artifacts

Raw offline JSON and rolling session artifacts are retained locally under
`/tmp/streammuse-rap-prompt-cycle/`. The two final rolling WAV files are also
there. These are temporary experiment artifacts; the tables and decision above
are the durable record.
