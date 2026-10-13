"""Analyze paired blind ratings for the Rule-S best-of-N experiment."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, wilcoxon


CONDITIONS = ("single_sample", "best_of_05_rule_s")
SCORE_COLUMNS = {
    "overall_strict": "overall_score_1_to_5",
    "melody_fit": "melody_fit_1_to_5",
    "accompaniment_quality": "accompaniment_quality_1_to_5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def parse_exact_score(value: object) -> float:
    text = str(value).strip()
    if not text:
        return math.nan
    return float(text)


def parse_overall_score(value: object) -> tuple[float, float, bool]:
    """Return failure-aware, audible-fragment, and ambiguity values."""
    text = str(value).strip()
    try:
        score = float(text)
        return score, score, False
    except ValueError:
        pass

    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if ("?" in text or "？" in text) and len(numbers) >= 2 and numbers[0] == 0:
        return 0.0, numbers[-1], True
    raise ValueError(f"Unsupported overall rating: {text!r}")


def summarize_conditions(rows: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for metric in metrics:
        for condition in CONDITIONS:
            values = rows.loc[rows["condition"] == condition, metric].astype(float)
            records.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "n": len(values),
                    "mean": values.mean(),
                    "sample_sd": values.std(ddof=1),
                    "median": values.median(),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(records)


def build_pairs(rows: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    keys = ["piece_id", "continuation_seed"]
    pairs = rows[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    for metric in metrics:
        wide = rows.pivot(index=keys, columns="condition", values=metric)
        for condition in CONDITIONS:
            pairs[f"{metric}__{condition}"] = [
                wide.loc[(piece, seed), condition]
                for piece, seed in pairs[keys].itertuples(index=False, name=None)
            ]
        pairs[f"{metric}__delta_rule_s_minus_single"] = (
            pairs[f"{metric}__best_of_05_rule_s"]
            - pairs[f"{metric}__single_sample"]
        )
    return pairs


def paired_tests(pairs: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for metric in metrics:
        deltas = pairs[f"{metric}__delta_rule_s_minus_single"].astype(float)
        wins = int((deltas > 0).sum())
        ties = int((deltas == 0).sum())
        losses = int((deltas < 0).sum())
        non_ties = wins + losses
        sign_p = (
            float(binomtest(wins, non_ties, 0.5, alternative="two-sided").pvalue)
            if non_ties
            else math.nan
        )
        wilcoxon_p = (
            float(wilcoxon(deltas, zero_method="wilcox", alternative="two-sided").pvalue)
            if non_ties
            else math.nan
        )
        records.append(
            {
                "metric": metric,
                "pair_count": len(deltas),
                "mean_paired_delta": deltas.mean(),
                "median_paired_delta": deltas.median(),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "two_sided_exact_sign_p": sign_p,
                "two_sided_wilcoxon_p": wilcoxon_p,
            }
        )
    return pd.DataFrame(records)


def summarize_pieces(pairs: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    columns = [
        f"{metric}__{condition}"
        for metric in metrics
        for condition in CONDITIONS
    ] + [f"{metric}__delta_rule_s_minus_single" for metric in metrics]
    return pairs.groupby("piece_id", as_index=False)[columns].mean()


def failure_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    wide = rows.pivot(
        index=["piece_id", "continuation_seed"],
        columns="condition",
        values="overall_strict",
    )
    single_failed = wide["single_sample"] == 0
    rule_s_failed = wide["best_of_05_rule_s"] == 0
    single_only = int((single_failed & ~rule_s_failed).sum())
    rule_s_only = int((~single_failed & rule_s_failed).sum())
    discordant = single_only + rule_s_only
    p_value = (
        float(
            binomtest(
                min(single_only, rule_s_only),
                discordant,
                0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else math.nan
    )
    return pd.DataFrame(
        [
            {
                "pair_count": len(wide),
                "single_sample_failure_count": int(single_failed.sum()),
                "rule_s_failure_count": int(rule_s_failed.sum()),
                "single_failed_rule_s_succeeded": single_only,
                "single_succeeded_rule_s_failed": rule_s_only,
                "two_sided_exact_mcnemar_p": p_value,
            }
        ]
    )


def successful_pair_comparison(pairs: pd.DataFrame) -> pd.DataFrame:
    successful = pairs[
        (pairs["overall_strict__single_sample"] > 0)
        & (pairs["overall_strict__best_of_05_rule_s"] > 0)
    ].copy()
    delta = successful["overall_strict__delta_rule_s_minus_single"]
    successful["outcome"] = delta.map(
        lambda value: "rule_s_win"
        if value > 0
        else ("tie" if value == 0 else "single_sample_win")
    )
    return successful


def format_number(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.3f}"


def write_report(
    output_path: Path,
    condition_summary: pd.DataFrame,
    tests: pd.DataFrame,
    pieces: pd.DataFrame,
    failures: pd.DataFrame,
    successful_pairs: pd.DataFrame,
    ambiguous_count: int,
) -> None:
    def condition_mean(metric: str, condition: str) -> float:
        row = condition_summary[
            (condition_summary["metric"] == metric)
            & (condition_summary["condition"] == condition)
        ]
        return float(row.iloc[0]["mean"])

    def test_row(metric: str) -> pd.Series:
        return tests[tests["metric"] == metric].iloc[0]

    lines = [
        "# Rule-S Best-of-5 Held-out Rating Analysis",
        "",
        "## Scope",
        "",
        "- 4 held-out melodies.",
        "- 2 paired continuation seeds per melody.",
        "- 8 paired comparisons and 16 blind ratings from one rater.",
        "- The same Prompt batch and continuation seeds were used by both conditions.",
        "",
        "## Main result",
        "",
        "| Metric | Single sample | Rule-S best-of-5 | Paired delta | W/T/L | Sign p | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in ("overall_strict", "melody_fit", "accompaniment_quality"):
        test = test_row(metric)
        lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    format_number(condition_mean(metric, "single_sample")),
                    format_number(condition_mean(metric, "best_of_05_rule_s")),
                    format_number(float(test["mean_paired_delta"])),
                    f"{int(test['wins'])}/{int(test['ties'])}/{int(test['losses'])}",
                    format_number(float(test["two_sided_exact_sign_p"])),
                    format_number(float(test["two_sided_wilcoxon_p"])),
                ]
            )
            + " |"
        )

    failure = failures.iloc[0]
    sensitivity = test_row("overall_audible_fragment")
    successful_delta = successful_pairs[
        "overall_strict__delta_rule_s_minus_single"
    ]
    successful_wins = int((successful_delta > 0).sum())
    successful_ties = int((successful_delta == 0).sum())
    successful_losses = int((successful_delta < 0).sum())
    lines.extend(
        [
            "",
            "## Failure-aware interpretation",
            "",
            (
                f"- Single sample produced {int(failure['single_sample_failure_count'])}/8 "
                "zero-score failures; Rule-S produced "
                f"{int(failure['rule_s_failure_count'])}/8."
            ),
            (
                "- Exact paired failure comparison: "
                f"p={format_number(float(failure['two_sided_exact_mcnemar_p']))}."
            ),
            (
                f"- {ambiguous_count} overall ratings used the form `0?x`. The main "
                "analysis treats them as failures because melody fit and accompaniment "
                "quality were both rated zero and the notes say most of the output was empty."
            ),
            (
                f"- Among the {len(successful_pairs)} pairs where both conditions "
                "succeeded, the mean overall delta was "
                f"{format_number(float(successful_delta.mean()))}, with "
                f"W/T/L={successful_wins}/{successful_ties}/{successful_losses}."
            ),
            "",
            "## Sensitivity analysis",
            "",
            (
                "- If the audible fragments of the two `0?x` samples are instead scored "
                f"as x, the overall paired delta remains "
                f"{format_number(float(sensitivity['mean_paired_delta']))}, with "
                f"W/T/L={int(sensitivity['wins'])}/{int(sensitivity['ties'])}/"
                f"{int(sensitivity['losses'])}."
            ),
            "",
            "## Per-piece overall result",
            "",
            "| Piece | Single sample | Rule-S best-of-5 | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in pieces.itertuples(index=False):
        lines.append(
            f"| {row.piece_id} | "
            f"{getattr(row, 'overall_strict__single_sample'):.3f} | "
            f"{getattr(row, 'overall_strict__best_of_05_rule_s'):.3f} | "
            f"{getattr(row, 'overall_strict__delta_rule_s_minus_single'):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This is positive held-out pilot evidence, not a definitive significance claim.",
            "- The effective independent melody count is 4; two seeds from one melody share the same selected Prompt.",
            "- Ratings come from one listener.",
            "- The observed gain is failure avoidance; this run does not show a quality gain among outputs where both conditions succeeded.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else run_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(run_dir / "blind_ratings.csv", dtype=str).fillna("")
    mapping = pd.read_csv(run_dir / "blind_mapping_private.csv", dtype=str)
    rows = mapping.merge(ratings, on="blind_id", validate="one_to_one")

    parsed_overall = rows["overall_score_1_to_5"].map(parse_overall_score)
    rows["overall_strict"] = parsed_overall.map(lambda item: item[0])
    rows["overall_audible_fragment"] = parsed_overall.map(lambda item: item[1])
    rows["overall_was_ambiguous"] = parsed_overall.map(lambda item: item[2])
    rows["melody_fit"] = rows["melody_fit_1_to_5"].map(parse_exact_score)
    rows["accompaniment_quality"] = rows[
        "accompaniment_quality_1_to_5"
    ].map(parse_exact_score)

    note_columns = [
        column
        for column in ("preferred_in_piece_seed_pair", "comments")
        if column in rows.columns
    ]
    rows["rater_note"] = rows[note_columns].apply(
        lambda values: " | ".join(
            value.strip() for value in values.astype(str) if value.strip()
        ),
        axis=1,
    )

    metrics = [
        "overall_strict",
        "overall_audible_fragment",
        "melody_fit",
        "accompaniment_quality",
    ]
    condition_summary = summarize_conditions(rows, metrics)
    pairs = build_pairs(rows, metrics)
    tests = paired_tests(pairs, metrics)
    pieces = summarize_pieces(pairs, metrics)
    failures = failure_comparison(rows)
    successful_pairs = successful_pair_comparison(pairs)

    rows.to_csv(output_dir / "resolved_sample_scores.csv", index=False)
    condition_summary.to_csv(output_dir / "condition_summary.csv", index=False)
    pairs.to_csv(output_dir / "paired_results.csv", index=False)
    tests.to_csv(output_dir / "paired_statistical_tests.csv", index=False)
    pieces.to_csv(output_dir / "piece_summary.csv", index=False)
    failures.to_csv(output_dir / "failure_comparison.csv", index=False)
    successful_pairs.to_csv(
        output_dir / "both_conditions_successful_pairs.csv", index=False
    )
    write_report(
        output_dir / "REPORT.md",
        condition_summary,
        tests,
        pieces,
        failures,
        successful_pairs,
        int(rows["overall_was_ambiguous"].sum()),
    )
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
