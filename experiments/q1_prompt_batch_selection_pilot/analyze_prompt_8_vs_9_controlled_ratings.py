"""Analyze the controlled paired Prompt 8-beat vs 9-beat ratings."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, wilcoxon


METRICS = [
    "overall_score_1_to_5",
    "prompt_continuation_transition_1_to_5",
    "melody_fit_1_to_5",
    "accompaniment_quality_1_to_5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def paired_tests(pairs: pd.DataFrame, subset: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for metric in METRICS:
        delta = pairs[f"{metric}__delta_09_minus_08"]
        wins = int((delta > 0).sum())
        ties = int((delta == 0).sum())
        losses = int((delta < 0).sum())
        non_ties = wins + losses
        sign_p = (
            float(binomtest(wins, non_ties, 0.5).pvalue)
            if non_ties
            else math.nan
        )
        wilcoxon_p = (
            float(wilcoxon(delta, zero_method="wilcox").pvalue)
            if non_ties
            else math.nan
        )
        records.append(
            {
                "subset": subset,
                "metric": metric,
                "pair_count": len(delta),
                "mean_delta_09_minus_08": delta.mean(),
                "median_delta_09_minus_08": delta.median(),
                "mean_absolute_delta": delta.abs().mean(),
                "wins_09": wins,
                "ties": ties,
                "losses_09": losses,
                "exact_sign_p_two_sided": sign_p,
                "wilcoxon_p_two_sided": wilcoxon_p,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = run_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    ratings = pd.read_csv(run_dir / "ratings.csv", dtype=str).fillna("")
    manifest = pd.read_csv(run_dir / "manifest.csv", dtype=str)
    rows = manifest.merge(ratings, on="file_name", validate="one_to_one")
    for metric in METRICS:
        rows[metric] = pd.to_numeric(rows[metric], errors="raise")
    rows["prefix_beats"] = pd.to_numeric(rows["prefix_beats"], errors="raise")
    rows["prompt_sample"] = pd.to_numeric(rows["prompt_sample"], errors="raise")
    rows["identical_pair"] = rows["identical_pair"].str.lower() == "true"

    keys = ["piece_order", "piece_id", "prompt_sample"]
    pair_base = rows[keys + ["identical_pair"]].drop_duplicates()
    if len(pair_base) != 21:
        raise ValueError(f"Expected 21 pairs, found {len(pair_base)}")
    pairs = pair_base.sort_values(keys).reset_index(drop=True)
    for metric in METRICS:
        wide = rows.pivot(index=keys, columns="prefix_beats", values=metric)
        pairs[f"{metric}__08beat"] = [
            wide.loc[(piece_order, piece_id, prompt_sample), 8]
            for piece_order, piece_id, prompt_sample in pairs[keys].itertuples(
                index=False, name=None
            )
        ]
        pairs[f"{metric}__09beat"] = [
            wide.loc[(piece_order, piece_id, prompt_sample), 9]
            for piece_order, piece_id, prompt_sample in pairs[keys].itertuples(
                index=False, name=None
            )
        ]
        pairs[f"{metric}__delta_09_minus_08"] = (
            pairs[f"{metric}__09beat"] - pairs[f"{metric}__08beat"]
        )

    condition_records: list[dict[str, object]] = []
    for metric in METRICS:
        for beats in (8, 9):
            values = rows.loc[rows["prefix_beats"] == beats, metric]
            condition_records.append(
                {
                    "metric": metric,
                    "prefix_beats": beats,
                    "n": len(values),
                    "mean": values.mean(),
                    "sample_sd": values.std(ddof=1),
                    "median": values.median(),
                    "zero_count": int((values == 0).sum()),
                }
            )
    condition_summary = pd.DataFrame(condition_records)

    tests = paired_tests(pairs, "all_pairs")
    tests.extend(
        paired_tests(
            pairs.loc[~pairs["identical_pair"]].reset_index(drop=True),
            "nonidentical_midi_pairs",
        )
    )
    tests_df = pd.DataFrame(tests)

    piece_columns = [
        column
        for column in pairs.columns
        if "__08beat" in column
        or "__09beat" in column
        or "__delta_09_minus_08" in column
    ]
    piece_summary = pairs.groupby(
        ["piece_order", "piece_id"], as_index=False
    )[piece_columns].mean()

    spread_records: list[dict[str, object]] = []
    for (piece_order, piece_id, beats), group in rows.groupby(
        ["piece_order", "piece_id", "prefix_beats"]
    ):
        values = group["overall_score_1_to_5"]
        spread_records.append(
            {
                "piece_order": piece_order,
                "piece_id": piece_id,
                "prefix_beats": beats,
                "prompt_sample_min_overall": values.min(),
                "prompt_sample_max_overall": values.max(),
                "prompt_sample_range_overall": values.max() - values.min(),
            }
        )
    prompt_spread = pd.DataFrame(spread_records)

    consistency_records: list[dict[str, object]] = []
    identical_pairs = pairs.loc[pairs["identical_pair"]]
    for metric in METRICS:
        delta = identical_pairs[f"{metric}__delta_09_minus_08"]
        consistency_records.append(
            {
                "metric": metric,
                "identical_pair_count": len(delta),
                "rating_mismatch_count": int((delta != 0).sum()),
                "maximum_absolute_mismatch": delta.abs().max(),
            }
        )
    consistency = pd.DataFrame(consistency_records)

    rows.to_csv(output_dir / "resolved_scores.csv", index=False)
    pairs.to_csv(output_dir / "paired_results.csv", index=False)
    condition_summary.to_csv(output_dir / "condition_summary.csv", index=False)
    tests_df.to_csv(output_dir / "paired_statistical_tests.csv", index=False)
    piece_summary.to_csv(output_dir / "piece_summary.csv", index=False)
    prompt_spread.to_csv(output_dir / "prompt_sample_spread.csv", index=False)
    consistency.to_csv(
        output_dir / "identical_pair_rating_consistency.csv", index=False
    )

    overall_test = tests_df[
        (tests_df["subset"] == "all_pairs")
        & (tests_df["metric"] == "overall_score_1_to_5")
    ].iloc[0]
    overall_means = condition_summary[
        condition_summary["metric"] == "overall_score_1_to_5"
    ].set_index("prefix_beats")
    mean_prompt_range = prompt_spread["prompt_sample_range_overall"].mean()
    mismatch = consistency[
        consistency["metric"] == "melody_fit_1_to_5"
    ].iloc[0]

    report = f"""# Controlled Prompt 8-vs-9 Rating Analysis

## Result

| Metric | 8 beats | 9 beats | Paired delta | 9-beat W/T/L |
|---|---:|---:|---:|---:|
| Overall | {overall_means.loc[8, 'mean']:.3f} | {overall_means.loc[9, 'mean']:.3f} | {overall_test['mean_delta_09_minus_08']:.3f} | {int(overall_test['wins_09'])}/{int(overall_test['ties'])}/{int(overall_test['losses_09'])} |

- Exact sign test: p={overall_test['exact_sign_p_two_sided']:.3f}.
- Wilcoxon signed-rank test: p={overall_test['wilcoxon_p_two_sided']:.3f}.
- 12/21 MIDI pairs are byte-identical; 9/21 differ.
- All 3 zero-score failures occur under both 8-beat and 9-beat conditions.

## Interpretation

The ninth Prompt beat has no observed average effect on overall score in this
controlled run. Positive and negative changes cancel exactly. It also does not
rescue any failed Prompt sample.

Prompt sample selection is the larger source of variation. The mean per-piece,
per-condition overall-score range across the three Prompt samples is
{mean_prompt_range:.3f}, compared with a mean absolute paired 8-vs-9 change of
{overall_test['mean_absolute_delta']:.3f}.

## Rating consistency

Byte-identical pairs should receive identical ratings. Overall, transition,
and accompaniment-quality ratings satisfy this. Melody fit has
{int(mismatch['rating_mismatch_count'])} mismatch among 12 identical pairs,
with maximum difference {mismatch['maximum_absolute_mismatch']:.3f}. The raw
rating is preserved and treated as single-rater measurement noise.

## Limits

- One listener and seven melodies.
- Three Prompt samples per melody are repeated observations, not 21 independent melodies.
- Condition names were visible during A/B rating.
- `top_k=1` is greedy-equivalent for this model except in the unlikely case of exact top-logit ties.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
