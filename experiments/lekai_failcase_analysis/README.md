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

The default cohort excludes `jazz_blues` by style, the pieces listed in `cohort_exclusions.csv`, and every piece not marked `include_4_4` in `meter_audit.csv`. Outputs are `labels_runs.csv`, `labels_components.csv`, and `cohort_summary.json`.

Meter is a cohort gate, not a prediction feature. Non-4/4 pieces are held for later analysis rather than deleted; a later stage can use them for `meter_mismatch` diagnosis. The current gate is a first-pass audit of NPZ metadata and does not claim that each source MusicXML is purely 4/4 across the full piece. A future MusicXML audit can upgrade that determination.

Rebuild the audit from the selection bundle with:

```bash
python experiments/lekai_failcase_analysis/build_meter_audit.py \
  --selection-manifest path/to/metadata/selection_manifest.csv \
  --output experiments/lekai_failcase_analysis/meter_audit.csv
```

The expected clean 4/4 cohort is **14 pieces / 28 piece-seed paired rows / 56 condition outputs**.

## Feature extraction

`feature_spec.csv` freezes each metric's analysis role, output column or pattern,
exact formula, unit, and undefined-value policy. `gate` fields define or audit the
cohort; `primary` fields enter the first KDE analysis; `secondary` full-piece copies
are diagnostics; `v2` fields are unavailable in the current artifacts. PPL is never
fabricated from MIDI: it requires selected-sample Prompt Model token log
probabilities.

Beat coordinates use `PrettyMIDI.time_to_tick(time) / resolution`. The model-facing
grid then intentionally matches `MidiConverter.py`:

```text
start_tick = int(round(start_beat * 4))
end_tick = int(round(end_beat * 4))
duration_tick = max(1, end_tick - start_tick)
```

This uses Python's `round` behavior. The timeline is never shifted to the first
note, so leading silence remains visible. The prefix is `[0, 8)` beats and the full
Melody horizon ends at its maximum quantized note end.

```bash
python experiments/lekai_failcase_analysis/extract_features.py \
  --bundle-dir path/to/style7x5_single_n1_vs_rule_s_n5_2seed_20260821 \
  --labels-runs experiments/lekai_failcase_analysis/results/4_4_v1/labels_runs.csv \
  --output-dir experiments/lekai_failcase_analysis/results/4_4_v1
```

The outputs are `melody_features.csv`, `prompt_features.csv`, and
`feature_audit.json`. Track names are matched case-insensitively. A Prompt MIDI
containing only a named Melody track is a valid semantically empty accompaniment:
`prompt_has_accompaniment=0`; count, density, coverage, and average voice number are
zero; `empty_beat_rate=1`; mathematically undefined pitch, duration, entropy, and
groove values remain blank. Melody is never substituted for accompaniment. A Prompt
containing only a named Accompaniment track is valid because relation features use
the separate source Melody MIDI.

## KDE screening

```bash
python experiments/lekai_failcase_analysis/analyze_kde_associations.py \
  --input-dir experiments/lekai_failcase_analysis/results/4_4_v1 \
  --output-dir experiments/lekai_failcase_analysis/results/4_4_v1/kde_v1
```

This stage uses a 1D Nadaraya-Watson/KDE screen per feature with nested
leave-one-piece-out evaluation. Strict screening runs 99 label permutations only for
positive-skill primary candidates, applies BH FDR to the resulting p-values, and
adds 200 group bootstraps for uncertainty. It does not fit any high-dimensional KDE.

Current 14-piece / 56-run / 28-pair findings are:

1. Leading blank diagnostic: skill `.426`, `p=.010`, `q=.030`. This is not music
   law: `technical/metadata/run_condition.sh` omitted `--trim-leading-rest`, so the
   signal is a pipeline segmentation diagnostic.
2. All-run Prompt density mismatch: skill `.268`, `p=.020`, `q=.240`, falling to
   `.035` on nonempty-only pieces.
3. Paired average voice-number delta: skill `.263`, `rho=.443`, `p=.010`,
   `q=.180`, suggestive only.

Generated outputs include
[`REPORT.md`](results/4_4_v1/kde_v1/REPORT.md) and
[`associations.csv`](results/4_4_v1/kde_v1/associations.csv). `results/` is
gitignored and should be regenerated locally.
