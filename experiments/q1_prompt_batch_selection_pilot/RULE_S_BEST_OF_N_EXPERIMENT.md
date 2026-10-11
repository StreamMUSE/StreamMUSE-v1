# Rule-S Best-of-N Prompt Experiment

## System path

The production-oriented path is:

```text
one Melody condition
-> one Prompt Model generate call with batch_size=N
-> N independently sampled accompaniment Prompts
-> empty/incomplete filtering
-> per-candidate raw-logit PPL and symbolic metrics
-> within-batch Rule-S ranking
-> only the highest-scoring Prompt enters Continuation
```

The ordinary condition uses candidate 1 from the same batch. This nested
comparison controls the Prompt sampling distribution:

```text
single_sample: candidate 1 -> Continuation
best_of_N_rule_s: argmax Rule-S(candidate 1..N) -> Continuation
```

The two conditions use the same continuation seeds.

## Frozen Prompt sampling

Initial configuration:

```text
candidate_count = 5
temperature = 0.8
top_p = 0.95
top_k = 50
repetition_penalty = 1.0
prompt_prefix_beats = 8
```

PPL is recomputed from raw Prompt Model logits. It is not computed from
temperature-scaled or top-k/top-p-truncated sampling probabilities.

## Rule-S

After rejecting empty or incomplete Prompt candidates:

```text
S =
  -0.450 * rank(PPL)
  +0.477 * rank(Pitch Range)
  +0.467 * rank(Pitch-Class Entropy)
  +0.496 * rank(Average Voice Number)
```

Ranks are normalized within the current Prompt batch. The coefficients are
piece-centered Spearman correlations frozen from the prior 3-piece discovery
pilot.

## Rule-S v2 candidate: duration compatibility

Rule-S v1 remains frozen. A possible v2 adds a Melody-Accompaniment note
duration compatibility penalty:

```text
r_duration = (median_duration_acc + epsilon)
             / (median_duration_mel + epsilon)

D_duration = abs(log(r_duration) - mu_expected)

S_v2 = S_v1 - lambda_duration * rank(D_duration)
```

Durations are measured in beats inside the 8-beat Prompt window. Empty or
incomplete Prompt candidates must still be rejected before scoring; `epsilon`
must not turn an empty Prompt into an eligible candidate. `mu_expected` is an
expected log duration ratio. If the expected value is specified as a raw ratio
`r_expected`, use `mu_expected = log(r_expected)`.

This exact ratio and mismatch penalty were not part of Rule-S v1. The previous
pilot included only indirect duration information:

- Qualified Note Rate: fraction of notes at least 0.75 beat long.
- Average Voice Number: duration-integrated simultaneous accompaniment voices.

An exploratory recomputation on the original discovery pilot produced the
following piece-centered Spearman correlations. Only 8 non-empty Prompt
candidates were available, so these are design signals rather than significant
results.

| Metric | Overall rating rho | Melody-acc fit rho | Acc quality rho |
| --- | ---: | ---: | ---: |
| Median duration ratio `r_duration` | +0.108 | +0.108 | +0.253 |
| `D_duration`, `mu_expected = 0` | -0.229 | -0.229 | -0.494 |
| `D_duration`, `mu_expected = log(2)` | -0.194 | -0.194 | -0.449 |

The negative direction for `D_duration` is expected: a larger mismatch tends
to accompany a lower rating. However, none of these pilot correlations is
statistically significant. The three discovery GT Prompts also have median
duration ratios `0.75`, `8.0`, and `2.0`, so the pilot cannot establish a
reliable universal `mu_expected`.

For a first v2 ablation, keep v1 unchanged and compare it against a separately
named selector using `mu_expected = 0` and provisional
`lambda_duration = 0.49`. Do not treat this coefficient as frozen until it is
validated on additional rated Prompt batches. The comparison must report both
overall listening quality and whether obvious Melody-Accompaniment duration
mismatches decrease.

## Evaluation order

1. Run the old 3 pieces as a seen-Melody/new-sample sanity check.
2. Check unique Prompt rate, empty/incomplete rate, selected candidate, logs,
   and MIDI validity.
3. Do not change the frozen selector using the new 4 pieces.
4. Run the new 4 pieces as the held-out listening comparison.

The old 3-piece run is a development check, not evidence of cross-Melody
generalization.

## Commands

Old 3-piece sanity check on GPU7:

```bash
DEVICE=7 bash q1_prompt_batch_selection_pilot/run_rule_s_best_of_5_old3.sh
```

New 4-piece held-out run:

```bash
DEVICE=7 bash q1_prompt_batch_selection_pilot/run_rule_s_best_of_5_new4.sh
```

Each piece produces only the two final conditions, with two paired
continuation seeds. Unselected Prompt candidates are retained for audit but do
not enter the Continuation Model.
