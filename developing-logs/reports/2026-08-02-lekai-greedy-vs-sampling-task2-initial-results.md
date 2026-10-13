# Lekai Greedy Versus Sampling: Task 2 Initial Results

Date: 2026-08-02

Status: nucleus-only sampling run following the original-author convention.

On 2026-08-03, the complete condition was rerun with the same inputs, prompt, seeds,
checkpoint, and explicit parameters. All 18 generated listening MIDI files had exactly
the same SHA-256 hashes as the first run. The fixed-seed offline pipeline is therefore
reproducible for this condition; the rerun did not create a second local listening
bundle.

## Question

Can stochastic decoding reduce monotonous accompaniment patterns compared with
`top_k=1`, without causing unacceptable instability or lower musical quality?

## Setting

The comparison uses the same three clean inputs and fixed 8-beat GT accompaniment
prompt as the valid Task 1 condition. All six runs have selection fingerprint
`53da6d5619e23e7c082359237877874a800c6be1`.

```text
Greedy:   temperature=1.0, top_k=1, top_p=1.0, seed=0
Sampling: temperature=1.1, top_k=0, top_p=0.95, seeds=0..4
```

There are three greedy outputs and fifteen sampling outputs. All 18 continuation
outputs contain at least one accompaniment note. Metrics exclude the common 8-beat
prompt.

This condition deliberately uses `top_k=0` because top-k and top-p are treated as
alternative sampling policies. Although `inference_new.py` currently defaults to
`top_k=10` and `top_p=0.95` together, that CLI combination conflicts with the intended
policy and is not treated as authoritative.

## Aggregate diagnostics

| Metric | Greedy | Sampling |
| --- | ---: | ---: |
| Pitch-class entropy | 0.793 | 2.993 |
| Dominant pitch-class rate | 78.5% | 22.7% |
| Median duration, beats | 134.33 | 1.35 |
| Notes longer than 8 beats | 60.5% | 1.0% |
| Notes closed at MIDI end | 59.4% | 0.5% |
| Exact repeat rate, adjacent non-empty 4-beat segments | 23.2% | 2.9% |
| Mean maximum identical non-empty run | 4.67 | 2.33 |
| Empty 4-beat segment rate | 0.2% | 15.5% |
| Mean maximum empty run | 0.33 | 13.60 |

The greedy duration aggregate is not a normal musical-duration distribution. In piece
`2581346`, 141 of 158 evaluated notes close at the MIDI end; in `43038`, 360 of 419 do
so. This indicates widespread unmatched note-ons or missing timely note-offs, followed
by final MIDI cleanup. The same two outputs also have very low pitch-class entropy.

For piece `3410421`, greedy shows a different monotony failure: 69.7% adjacent
non-empty 4-beat segments are exactly repeated, with a maximum run of 12 identical
segments. All five sampling versions have zero exact adjacent non-empty repeats and a
maximum run of one.

## Sampling trade-off

Sampling strongly reduces literal repetition, pitch collapse, and hanging-note
behavior. It also creates sparse or silent regions for some seeds. The clearest failure
is `2581346` sampling seed 4:

```text
continuation accompaniment notes: 16
empty 4-beat segment rate: 89.9%
maximum empty run: 142 segments
```

Other seeds on the same piece range from 9.4% to 49.1% empty segments. Results on
`3410421` and `43038` are substantially denser. Sampling sensitivity is therefore both
seed-dependent and piece-dependent.

Pairwise sampling-seed note Jaccard is only 0.008 to 0.024 across the three pieces, so
the five seeds provide genuinely different generations rather than duplicate outputs.

## Human listening results

A single listener rated all 18 outputs. Score `0` denotes a failed output that could
not be rated normally. Means below exclude those failures; success counts are reported
separately.

| Decoder / piece | Evaluable | Variety | Musical quality | Melody-acc fit | Stability |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy / `43038` | 0/1 | - | - | - | - |
| Greedy / `3410421` | 1/1 | 2.00 | 3.20 | 3.00 | 4.00 |
| Greedy / `2581346` | 0/1 | - | - | - | - |
| Sampling / `43038` | 5/5 | 3.30 | 3.20 | 2.70 | 3.88 |
| Sampling / `3410421` | 5/5 | 4.00 | 3.20 | 2.80 | 4.00 |
| Sampling / `2581346` | 3/5 | 4.00 | 3.33 | 2.70 | 3.73 |
| Sampling / all evaluable | 13/15 | 3.73 | 3.23 | 2.74 | 3.89 |

Sampling raised the evaluable-output rate from 1/3 for greedy to 13/15 across sampling
seeds. This is descriptive only: the seeds are repeated generations of three pieces,
not 15 independent pieces.

For the only piece with an evaluable greedy baseline (`3410421`), sampling increased
variety from 2.0 to 4.0, left musical quality unchanged at 3.2, and reduced
Melody-accompaniment fit from 3.0 to 2.8. All five sampling seeds received the same
scores on this piece.

Across all 13 evaluable sampling outputs, no Melody-accompaniment fit score exceeded
3.0 and no musical-quality score exceeded 3.4. The comments repeatedly identify the
same issue: the accompaniment is more varied but does not align well with the Melody.
For `2581346`, two of five sampling outputs were not evaluable: one was perceived as
drum-like and one failed to generate usable accompaniment.

## Interpretation

The combined evidence supports a narrow claim: nucleus sampling reduces several
greedy degeneration patterns and increases perceived variety. It does not support the
claim that sampling makes the accompaniment sound better overall. Musical quality
remained around 3.2, while Melody-accompaniment fit was consistently the weakest human
dimension.

Silence diagnostics are not a proxy for musical quality. The reported concern for this
round is primarily that several outputs sound poor even when they are non-empty. The
quality conclusion must therefore come from the human `musical_quality`,
`melody_accompaniment_fit`, and free-text ratings, not from empty-segment statistics.

The ratings use variety, musical quality, Melody-accompaniment fit, and stability, with
higher values always meaning better. Because this pilot has only three pieces and one
listener, it is suitable for identifying failure modes and planning the next decoder
comparison, not for a significance claim.

## Artifacts

- Local: `remote_results/lekai_sampling_monotony_task2_20260802`
- Remote:
  `/data/home/yuanxin/StreamMUSE-v1/output/lekai_sampling_monotony/task2_greedy_vs_topp_20260802`
- Exact rerun:
  `/data/home/yuanxin/StreamMUSE-v1/output/lekai_sampling_monotony/task2_greedy_vs_nucleus_sampling_rerun_20260803`
