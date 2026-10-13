# Lekai Melody Timing Robustness: Task 1 Initial Results

Date: 2026-08-01

## Question

Does the Lekai continuation model remain stable when only Melody note onsets are
perturbed?

## Quantization preflight

The offline input path quantizes MIDI positions with:

```text
model_step = round(beat_position * 4)
```

The earlier independent `[-20, +20] ms` onset jitter was too small for this path. All
9/9 checked inputs produced exactly the same 4tpb note records as their unperturbed
versions. That result only demonstrates quantizer invariance; it cannot test model
robustness.

The formal experiment therefore uses an exact one-model-step displacement:

```text
one step = 0.25 beat
```

Its duration in milliseconds depends on BPM:

| Piece | BPM | One model step |
| --- | ---: | ---: |
| 2581346 | 168 | 89.3 ms |
| 3410421 | 80 | 187.5 ms |
| 43038 | 120 | 125.0 ms |

## Frozen setting

- Pieces: `2581346`, `3410421`, `43038`.
- Perturbed data: Melody note onset only.
- Displacement: independently selected `-1` or `+1` model step.
- Quantized duration, pitch, accompaniment, and metadata: unchanged.
- Selected-note rates: 10%, 25%, and 50%.
- Perturbation seeds: 0, 1, and 2.
- Inference: official `RT-accompanimentV2/inference_new.py`.
- Condition: fixed 8-beat GT accompaniment prompt.
- Decoding: greedy, `top_k=1`.
- Evaluation excludes accompaniment notes beginning inside the shared 8-beat prompt.

All 27 perturbed inputs passed the post-quantization checks: Melody note records and
pianoroll cells changed, accompaniment cells remained exactly equal, and note pitch and
duration shapes were preserved.

## Failed no-prompt control

The first greedy run used no accompaniment prompt. Its three baseline pieces and all
27 perturbed conditions generated zero continuation accompaniment notes. The resulting
exact matches are empty-output matches and are not robustness evidence.

The formal run therefore uses the same fixed 8-beat GT accompaniment prompt for every
baseline and perturbed condition.

## Formal initial result

Baseline continuation accompaniment note counts were 159, 205, and 419 for the three
pieces. All 27 perturbed outputs were non-empty, but none was exactly identical to its
piece baseline after beat 8.

| Perturbed Melody notes | Samples | Exact output | Empty output | Exact-beat rate | Note Jaccard | Pianoroll-cell Jaccard |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 9 | 0/9 | 0/9 | 0.263 | 0.186 | 0.288 |
| 25% | 9 | 0/9 | 0/9 | 0.141 | 0.040 | 0.147 |
| 50% | 9 | 0/9 | 0/9 | 0.123 | 0.017 | 0.141 |

The aggregate trend is consistent: increasing the number of displaced Melody notes
reduces overlap with the unperturbed greedy continuation. Sensitivity is strongly
piece-dependent, and the three-seed per-piece results are not perfectly monotonic.

## Interpretation boundary

This experiment establishes that one-step Melody timing changes survive the actual
offline encoder and can substantially alter the autoregressive continuation. It does
not yet establish that the altered continuation is musically worse. Exact-match and
Jaccard metrics measure trajectory retention, while robustness in musical quality must
be judged with the prepared listening ratings.

## Post-hoc validity issue

Task 2 diagnostics later showed that the unperturbed greedy baseline is already
musically degenerate for two of the three pieces. After the shared 8-beat prompt,
`43038` closes 360 of 419 accompaniment notes only at the MIDI end, while `2581346`
closes 141 of 158 at the MIDI end. Their median note durations are approximately 151
and 250 beats.

Consequently, the current Task 1 bundle is suspended for listening-based robustness
claims. A failed baseline cannot identify whether a perturbed output is worse because
of timing sensitivity or simply different from an already failed greedy trajectory.
The post-quantization perturbation validation and trajectory-difference measurements
remain valid as exploratory sensitivity evidence.

The next valid conclusion requires rating baseline/perturbed pairs for musical quality,
Melody-accompaniment fit, and noticeable degradation. Duration perturbation should
remain a separate follow-up condition.

## Artifacts

- Local listening and metrics:
  `remote_results/lekai_melody_timing_robustness_task1_20260801`
- Formal remote run:
  `/data/home/yuanxin/StreamMUSE-v1/output/lekai_melody_timing_robustness/task1_grid_onset_gtprompt8_greedy_20260801`
- Failed no-prompt control:
  `/data/home/yuanxin/StreamMUSE-v1/output/lekai_melody_timing_robustness/task1_grid_onset_greedy_20260801`
