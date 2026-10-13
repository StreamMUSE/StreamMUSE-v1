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

Meter is a cohort gate, not a prediction feature. Non-4/4 pieces are held for later analysis rather than deleted; a later stage can use them for `meter_mismatch` diagnosis. When `--musicxml-dir` is supplied, the audit scans every time-signature event in the full source MusicXML and includes a piece only when its set of unique signatures is exactly `{4/4}`. Without `--musicxml-dir`, the audit falls back to the first signature recorded in NPZ metadata; that fallback cannot prove that the full piece is constantly 4/4.

Rebuild the audit from the selection bundle and full-piece MusicXML files with:

```bash
python experiments/lekai_failcase_analysis/build_meter_audit.py \
  --selection-manifest <selection-manifest> \
  --musicxml-dir <musicxml-dir> \
  --output experiments/lekai_failcase_analysis/meter_audit.csv
```

The current clean 4/4 cohort is **13 pieces / 26 piece-seed paired rows / 52 condition outputs**. `142159` (`Bohemian Rhapsody`) is excluded because its full XML contains `2/4 | 4/4 | 5/4`.

## Feature extraction

`feature_spec.csv` freezes each metric's analysis role, output column or pattern,
exact formula, unit, and undefined-value policy. `gate` fields define or audit the
cohort; `primary` fields enter the first KDE analysis; `secondary` full-piece copies
are diagnostics. The symbolic harmony `v2` fields require regenerating feature
artifacts with the current extractor. PPL is never fabricated from MIDI: it remains
unavailable unless selected-sample Prompt Model token log probabilities were
retained.

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
`feature_audit.json`. Track names are matched case-insensitively using either the
exact semantic names `Melody` / `Accompaniment` or the generated-name suffixes
`-mel` / `-acc`. Multiple matches remain an error. A Prompt MIDI containing only a
semantic Melody track is a valid semantically empty accompaniment:
`prompt_has_accompaniment=0`; count, density, coverage, and average voice number are
zero; `empty_beat_rate=1`; mathematically undefined pitch, duration, entropy, and
groove values remain blank. Melody is never substituted for accompaniment. A Prompt
containing only a named Accompaniment track is valid because relation features use
the separate source Melody MIDI.

## Harmony relation feature extraction

The v2 relation family is computed only from the input Melody and selected
accompaniment Prompt over `[0, 8)` beats. The window contains exactly 32
`0.25`-beat cells. A cell is harmony-valid only when both tracks have positive
note-duration overlap in that cell.

- `relation_pcs_{mean|p10}` averages the frozen interval-class scores over all
  cross-track active-note pairs within each valid cell, then summarizes cells.
- `relation_mctd_{mean|p90}` uses duration-overlap-weighted 12D PCPs and the frozen
  six-dimensional tonal-centroid mapping in `feature_spec.csv`.
- `relation_local_chroma_cosine_{mean|p10|std}` summarizes local PCP cosine
  similarity and is secondary because it overlaps conceptually with pitch-class
  divergence.
- `relation_harmony_valid_cell_ratio` is the primary coverage control: valid cells
  divided by 32.

PCS, MCTD, and local chroma outputs are `NaN` when there is no valid cell. The
valid-cell ratio is always defined and equals zero in that case. Regenerate the v2
feature tables with:

```bash
python experiments/lekai_failcase_analysis/extract_features.py \
  --bundle-dir <bundle-root> \
  --labels-runs experiments/lekai_failcase_analysis/results/4_4_v1/labels_runs.csv \
  --output-dir experiments/lekai_failcase_analysis/results/4_4_v2
```

These are strictly pre-continuation features. Continuation notes, continuation
tokens, output-track statistics, and any other information produced after the
selected Prompt are forbidden as inputs. Full continuation ratings supply only the
fail labels and never enter feature calculation.

Screen the harmony relations against the three non-exclusive targets
`insufficient_output`, `repetition`, and `melody_mismatch` with:

```bash
python experiments/lekai_failcase_analysis/analyze_harmony_fail_associations.py \
  --input-dir experiments/lekai_failcase_analysis/results/4_4_v2 \
  --output-dir experiments/lekai_failcase_analysis/results/4_4_v2/harmony_kde_v2
```

On the current 13-piece clean cohort, the primary signal is the effective
Melody--Prompt temporal-overlap ratio: `skill=.378`, `rho=-.490`, `p=.020`,
`q=.160`. This is suggestive, not FDR-supported. Restricting to the 10 pieces
whose four runs all have positive overlap gives `skill=.027`, `p=.180`, so the
association is largely driven by low or zero overlap. PCS, MCTD, and local Chroma
features are unsupported for `melody_mismatch`; `repetition` remains too small for
a stable claim.

The analyzer keeps all four runs from a piece together and drops the complete piece
for a feature when any of its four values is undefined. It warns when a target has
fewer than 10 positive outputs or fewer than 5 positive pieces. In the current clean
4/4 cohort, `repetition` has only 4 positive outputs, so its KDE curves and p/q
values are exploratory and cannot support a stable association claim.

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

Current 13-piece / 52-run / 26-pair findings are:

1. Leading blank diagnostic: skill `.342`, `rho=.686`, `p=.010`, `q=.030`. This is not music
   law: `technical/metadata/run_condition.sh` omitted `--trim-leading-rest`, so the
   signal is a pipeline segmentation diagnostic.
2. All-run Prompt density mismatch: skill `.308`, `rho=.407`, `p=.010`, `q=.180`,
   suggestive only.
3. Paired average voice-number delta: skill `.241`, `rho=.401`, `p=.010`,
   `q=.180`, suggestive only.

Generated outputs include
[`REPORT.md`](results/4_4_v1/kde_v1/REPORT.md) and
[`associations.csv`](results/4_4_v1/kde_v1/associations.csv). `results/` is
gitignored and should be regenerated locally.
