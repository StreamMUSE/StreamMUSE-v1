# Lekai fail-case label freeze

This directory freezes the first-stage cohort, labels, and feature definitions for fail-case analysis. It does not train or select a predictor.

## Prediction points and leakage boundary

The analysis has two distinct risk layers:

1. **Melody-only, pre-generation:** estimate risk before any accompaniment is generated.
2. **Melody + selected prompt, pre-continuation:** estimate risk after prompt selection but before continuation.

A predictor must never use continuation output or any feature computed from continuation output. The second layer may use only the input melody and the already selected prompt.

`quality_floor` (the minimum component score) is the primary quality label. `quality_mean` is retained only for sensitivity analysis. `severe_fail` means `quality_floor <= 2`, and `issue_any` means that at least one fail type was recorded.

Current ratings support the narrower observation that Rule-S mainly reduces `insufficient_output`. They do **not** establish improvement for `mismatch` or `repetition`. The first stage therefore freezes only label and feature definitions; it does not make a predictor-performance claim.

## Build

```bash
python experiments/lekai_failcase_analysis/build_labels.py \
  --ratings path/to/rating.csv \
  --output-dir experiments/lekai_failcase_analysis/derived
```

The default cohort excludes `jazz_blues` by style and the pieces listed in `cohort_exclusions.csv`. Outputs are `labels_runs.csv`, `labels_components.csv`, and `cohort_summary.json`.

The expected clean cohort is **23 pieces / 46 piece-seed paired rows / 92 condition outputs**.
