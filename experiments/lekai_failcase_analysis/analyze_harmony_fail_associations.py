"""Screen Prompt harmony features for associations with labeled fail cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__:
    from experiments.lekai_failcase_analysis.kde_model import (
        bh_qvalues,
        group_block_permutation_skill,
        group_bootstrap_curve_bands,
        nested_logo_cv,
        select_logo_bandwidth_and_curve,
        spearman_nan,
    )
else:  # Direct execution by repository-relative script path.
    from kde_model import (  # type: ignore[no-redef]
        bh_qvalues,
        group_block_permutation_skill,
        group_bootstrap_curve_bands,
        nested_logo_cv,
        select_logo_bandwidth_and_curve,
        spearman_nan,
    )


FEATURES = (
    "relation_pcs_mean",
    "relation_pcs_p10",
    "relation_mctd_mean",
    "relation_mctd_p90",
    "relation_local_chroma_cosine_mean",
    "relation_local_chroma_cosine_p10",
    "relation_local_chroma_cosine_std",
    "relation_harmony_valid_cell_ratio",
)
TARGETS = ("insufficient_output", "repetition", "melody_mismatch")
EXPECTED_RUNS_PER_PIECE = 4
FDR_THRESHOLD = 0.10
LOW_POSITIVE_RUN_THRESHOLD = 10
LOW_POSITIVE_GROUP_THRESHOLD = 5
OVERLAP_FEATURE = "relation_harmony_valid_cell_ratio"
OVERLAP_TARGET = "insufficient_output"

ASSOCIATION_FIELDS = (
    "target",
    "feature",
    "n_rows_total",
    "n_rows",
    "n_groups_total",
    "n_groups",
    "dropped_group_count",
    "dropped_groups",
    "positive_count",
    "positive_group_count",
    "prevalence",
    "valid",
    "invalid_reason",
    "cv_skill",
    "cv_mse",
    "null_mse",
    "rho",
    "permutation_status",
    "n_permutations",
    "p_value",
    "q_value",
    "evidence_status",
)
CURVE_FIELDS = (
    "target",
    "feature",
    "selection_rank",
    "curve_index",
    "x",
    "pred",
    "lower",
    "upper",
    "bandwidth",
    "n_rows",
    "n_groups",
)
OVERLAP_SENSITIVITY_FIELDS = (
    "cohort",
    "definition",
    "pieces",
    "runs",
    "positives",
    "zero_ratio",
    "prevalence",
    "valid",
    "invalid_reason",
    "cv_skill",
    "rho",
    "permutation_status",
    "n_permutations",
    "p_value",
    "dropped_piece_count",
    "dropped_pieces",
)
OVERLAP_COHORTS = (
    ("all", "all complete include_main piece blocks"),
    (
        "all_four_have_acc",
        "pieces whose four runs all have prompt_has_accompaniment=1",
    ),
    (
        "all_four_ratio_positive",
        "pieces whose four runs all have relation_harmony_valid_cell_ratio>0",
    ),
)
RATIO_BIN_LABELS = ("[0]", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1]")


def _as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_fail_types(raw: object) -> set[str]:
    """Parse the slash-separated, non-exclusive fail labels."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return set()
    return {token.strip().casefold() for token in str(raw).split("/") if token.strip()}


def _csv_value(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    return value


def _write_csv(path: Path, rows: list[dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return (int(base) + int.from_bytes(digest[:4], "big")) % (2**32)


def _run_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    seed_raw = str(row.get("seed", "")).strip()
    try:
        seed_key: tuple[int, object] = (0, int(seed_raw))
    except ValueError:
        seed_key = (1, seed_raw)
    return seed_key, str(row.get("condition", "")), int(row["_source_index"])


def load_runs(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load only include_main rows and freeze the analysis-facing columns."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        required = {
            "piece",
            "include_main",
            "fail_type_raw",
            "prompt_has_accompaniment",
            *FEATURES,
        }
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"prompt_features.csv missing columns: {missing}")
        source_rows = list(reader)

    runs: list[dict[str, object]] = []
    unknown = Counter()
    for source_index, source in enumerate(source_rows):
        if not _as_bool(source.get("include_main")):
            continue
        piece = str(source.get("piece", "")).strip()
        if not piece:
            raise ValueError(f"include_main row {source_index + 2} has an empty piece")
        fail_types = parse_fail_types(source.get("fail_type_raw"))
        unknown.update(fail_types - set(TARGETS))
        row: dict[str, object] = {
            "piece": piece,
            "seed": source.get("seed", ""),
            "condition": source.get("condition", ""),
            "prompt_has_accompaniment": _as_float(source.get("prompt_has_accompaniment")),
            "_source_index": source_index,
        }
        row.update({target: int(target in fail_types) for target in TARGETS})
        row.update({feature: _as_float(source.get(feature)) for feature in FEATURES})
        runs.append(row)

    if not runs:
        raise ValueError("prompt_features.csv has no include_main=True rows")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in runs:
        grouped[str(row["piece"])].append(row)
    bad_counts = {piece: len(block) for piece, block in grouped.items() if len(block) != EXPECTED_RUNS_PER_PIECE}
    if bad_counts:
        detail = ", ".join(f"{piece}={count}" for piece, count in sorted(bad_counts.items()))
        raise ValueError(f"each include_main piece must have exactly four runs; found {detail}")

    ordered: list[dict[str, object]] = []
    for piece in sorted(grouped):
        block = sorted(grouped[piece], key=_run_sort_key)
        for order, row in enumerate(block):
            row["_within_group_order"] = order
            ordered.append(row)

    audit = {
        "input_row_count": len(source_rows),
        "included_row_count": len(ordered),
        "excluded_row_count": len(source_rows) - len(ordered),
        "included_piece_count": len(grouped),
        "unknown_fail_type_tokens": dict(sorted(unknown.items())),
        "input_column_count": len(fieldnames),
    }
    return ordered, audit


def complete_groups(
    rows: list[dict[str, object]], feature: str, target: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Drop an entire piece unless all four feature/target values are finite."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["piece"])].append(row)

    kept: list[dict[str, object]] = []
    dropped: list[str] = []
    for piece in sorted(grouped):
        block = sorted(grouped[piece], key=lambda row: int(row["_within_group_order"]))
        complete = len(block) == EXPECTED_RUNS_PER_PIECE and all(
            math.isfinite(_as_float(row.get(feature))) and math.isfinite(_as_float(row.get(target)))
            for row in block
        )
        if complete:
            kept.extend(block)
        else:
            dropped.append(piece)

    x = np.asarray([_as_float(row[feature]) for row in kept], dtype=float)
    y = np.asarray([_as_float(row[target]) for row in kept], dtype=float)
    groups = np.asarray([str(row["piece"]) for row in kept], dtype=object)
    order = np.asarray([int(row["_within_group_order"]) for row in kept], dtype=float)
    return x, y, groups, order, dropped


def analyze_associations(
    rows: list[dict[str, object]], permutations: int, seed: int
) -> list[dict[str, object]]:
    all_groups = sorted({str(row["piece"]) for row in rows})
    results: list[dict[str, object]] = []
    for target in TARGETS:
        target_indices: list[int] = []
        for feature in FEATURES:
            x, y, groups, order, dropped = complete_groups(rows, feature, target)
            cv = nested_logo_cv(x, y, groups)
            valid = bool(cv["valid"])
            skill = _as_float(cv.get("skill")) if valid else float("nan")
            p_value = 1.0
            n_permutations = 0
            if valid and math.isfinite(skill) and skill > 0.0:
                permutation = group_block_permutation_skill(
                    x,
                    y,
                    groups,
                    n_perm=permutations,
                    seed=_stable_seed(seed, "permutation", target, feature),
                    within_group_order=order,
                )
                if permutation["valid"]:
                    p_value = _as_float(permutation.get("p_value"))
                    n_permutations = permutations
                    permutation_status = "computed_positive_skill"
                else:
                    permutation_status = "invalid_permutation"
            elif valid:
                permutation_status = "not_tested_nonpositive_skill"
            else:
                permutation_status = "not_tested_invalid_association"

            labels = sorted(set(groups.tolist()))
            positive_groups = sum(bool(np.any(y[groups == group] == 1.0)) for group in labels)
            result: dict[str, object] = {
                "target": target,
                "feature": feature,
                "n_rows_total": len(rows),
                "n_rows": len(x),
                "n_groups_total": len(all_groups),
                "n_groups": len(labels),
                "dropped_group_count": len(dropped),
                "dropped_groups": ";".join(dropped),
                "positive_count": int(np.sum(y == 1.0)),
                "positive_group_count": int(positive_groups),
                "prevalence": float(np.mean(y)) if y.size else float("nan"),
                "valid": valid,
                "invalid_reason": "" if valid else json.dumps(cv.get("reason"), sort_keys=True),
                "cv_skill": skill,
                "cv_mse": _as_float(cv.get("cv_mse")),
                "null_mse": _as_float(cv.get("null_mse")),
                "rho": spearman_nan(x, y),
                "permutation_status": permutation_status,
                "n_permutations": n_permutations,
                "p_value": p_value,
                "q_value": float("nan"),
                "evidence_status": "unsupported",
            }
            target_indices.append(len(results))
            results.append(result)

        if len(target_indices) != len(FEATURES):
            raise AssertionError("BH-FDR family must contain exactly the eight frozen harmony features")
        q_values = bh_qvalues([_as_float(results[index]["p_value"]) for index in target_indices])
        for index, q_value in zip(target_indices, q_values):
            result = results[index]
            result["q_value"] = float(q_value)
            skill = _as_float(result["cv_skill"])
            p_value = _as_float(result["p_value"])
            if math.isfinite(skill) and skill > 0.0 and q_value <= FDR_THRESHOLD:
                result["evidence_status"] = "fdr_supported_screening"
            elif math.isfinite(skill) and skill > 0.0 and p_value <= FDR_THRESHOLD:
                result["evidence_status"] = "suggestive_screening"
            else:
                result["evidence_status"] = "unsupported"
    return results


def analyze_overlap_sensitivity(
    rows: list[dict[str, object]], permutations: int, seed: int
) -> list[dict[str, object]]:
    """Run the preregistered overlap-ratio sensitivity cohorts."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["piece"])].append(row)
    all_pieces = set(grouped)

    cohort_piece_sets: dict[str, set[str]] = {
        "all": set(all_pieces),
        "all_four_have_acc": {
            piece
            for piece, block in grouped.items()
            if len(block) == EXPECTED_RUNS_PER_PIECE
            and all(
                math.isfinite(_as_float(row["prompt_has_accompaniment"]))
                and _as_float(row["prompt_has_accompaniment"]) == 1.0
                for row in block
            )
        },
        "all_four_ratio_positive": {
            piece
            for piece, block in grouped.items()
            if len(block) == EXPECTED_RUNS_PER_PIECE
            and all(
                math.isfinite(_as_float(row[OVERLAP_FEATURE]))
                and _as_float(row[OVERLAP_FEATURE]) > 0.0
                for row in block
            )
        },
    }

    results: list[dict[str, object]] = []
    for cohort, definition in OVERLAP_COHORTS:
        candidate_pieces = cohort_piece_sets[cohort]
        candidate_rows = [row for row in rows if str(row["piece"]) in candidate_pieces]
        x, y, groups, order, incomplete = complete_groups(
            candidate_rows, OVERLAP_FEATURE, OVERLAP_TARGET
        )
        used_pieces = set(groups.tolist())
        dropped = sorted((all_pieces - candidate_pieces) | set(incomplete))
        cv = nested_logo_cv(x, y, groups)
        valid = bool(cv["valid"])
        skill = _as_float(cv.get("skill")) if valid else float("nan")
        p_value = 1.0
        n_permutations = 0
        if valid and math.isfinite(skill) and skill > 0.0:
            permutation_seed = (
                _stable_seed(seed, "permutation", OVERLAP_TARGET, OVERLAP_FEATURE)
                if cohort == "all"
                else _stable_seed(seed, "overlap_sensitivity", cohort)
            )
            permutation = group_block_permutation_skill(
                x,
                y,
                groups,
                n_perm=permutations,
                seed=permutation_seed,
                within_group_order=order,
            )
            if permutation["valid"]:
                p_value = _as_float(permutation.get("p_value"))
                n_permutations = permutations
                permutation_status = "computed_positive_skill"
            else:
                permutation_status = "invalid_permutation"
        elif valid:
            permutation_status = "not_tested_nonpositive_skill"
        else:
            permutation_status = "not_tested_invalid_association"

        results.append(
            {
                "cohort": cohort,
                "definition": definition,
                "pieces": len(used_pieces),
                "runs": len(x),
                "positives": int(np.sum(y == 1.0)),
                "zero_ratio": int(np.sum(x == 0.0)),
                "prevalence": float(np.mean(y)) if y.size else float("nan"),
                "valid": valid,
                "invalid_reason": "" if valid else json.dumps(cv.get("reason"), sort_keys=True),
                "cv_skill": skill,
                "rho": spearman_nan(x, y),
                "permutation_status": permutation_status,
                "n_permutations": n_permutations,
                "p_value": p_value,
                "dropped_piece_count": len(dropped),
                "dropped_pieces": ";".join(dropped),
            }
        )
    return results


def overlap_ratio_bins(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Summarize descriptive ratio bins on the complete all cohort."""
    x, y, groups, _, dropped = complete_groups(rows, OVERLAP_FEATURE, OVERLAP_TARGET)
    masks = (
        x == 0.0,
        (x > 0.0) & (x <= 0.25),
        (x > 0.25) & (x <= 0.50),
        (x > 0.50) & (x <= 0.75),
        (x > 0.75) & (x <= 1.0),
    )
    result: list[dict[str, object]] = []
    covered = np.zeros(x.shape, dtype=bool)
    for label, mask in zip(RATIO_BIN_LABELS, masks):
        covered |= mask
        count = int(np.sum(mask))
        fails = int(np.sum(y[mask] == 1.0))
        result.append(
            {
                "bin": label,
                "n": count,
                "fail": fails,
                "rate": fails / count if count else float("nan"),
            }
        )
    audit = {
        "cohort": "all",
        "pieces": len(set(groups.tolist())),
        "runs": len(x),
        "dropped_pieces": dropped,
        "unbinned_count": int(np.sum(~covered)),
        "description": "descriptive only; no permutation or FDR correction",
    }
    return result, audit


def build_curves(
    rows: list[dict[str, object]],
    associations: list[dict[str, object]],
    bootstraps: int,
    curve_points: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    curve_rows: list[dict[str, object]] = []
    curve_audit: list[dict[str, object]] = []
    for target in TARGETS:
        eligible = [
            row
            for row in associations
            if row["target"] == target
            and bool(row["valid"])
            and math.isfinite(_as_float(row["cv_skill"]))
            and _as_float(row["cv_skill"]) > 0.0
        ]
        selected = sorted(eligible, key=lambda row: (-_as_float(row["cv_skill"]), str(row["feature"])))[:2]
        for rank, association in enumerate(selected, start=1):
            feature = str(association["feature"])
            x, y, groups, _, dropped = complete_groups(rows, feature, target)
            fitted = select_logo_bandwidth_and_curve(x, y, groups, n_curve=curve_points)
            if not fitted["valid"]:
                continue
            bands = group_bootstrap_curve_bands(
                x,
                y,
                groups,
                curve_x=fitted["curve_x"],
                bandwidth=fitted["bandwidth"],
                n_boot=bootstraps,
                seed=_stable_seed(seed, "bootstrap", target, feature),
            )
            if not bands["valid"]:
                continue
            for curve_index, (x_value, pred, lower, upper) in enumerate(
                zip(bands["curve_x"], bands["curve_y"], bands["lower"], bands["upper"])
            ):
                curve_rows.append(
                    {
                        "target": target,
                        "feature": feature,
                        "selection_rank": rank,
                        "curve_index": curve_index,
                        "x": float(x_value),
                        "pred": float(pred),
                        "lower": float(lower),
                        "upper": float(upper),
                        "bandwidth": float(fitted["bandwidth"]),
                        "n_rows": len(x),
                        "n_groups": len(set(groups.tolist())),
                    }
                )
            curve_audit.append(
                {
                    "target": target,
                    "feature": feature,
                    "selection_rank": rank,
                    "cv_skill": _as_float(association["cv_skill"]),
                    "bandwidth": float(fitted["bandwidth"]),
                    "n_rows": len(x),
                    "n_groups": len(set(groups.tolist())),
                    "dropped_groups": dropped,
                }
            )
    return curve_rows, curve_audit


def _target_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    groups = sorted({str(row["piece"]) for row in rows})
    for target in TARGETS:
        positives = int(sum(int(row[target]) for row in rows))
        positive_groups = sum(any(int(row[target]) for row in rows if row["piece"] == piece) for piece in groups)
        summary[target] = {
            "positive_count": positives,
            "positive_group_count": int(positive_groups),
            "prevalence": positives / len(rows),
        }
    return summary


def _fmt(value: object) -> str:
    number = _as_float(value)
    return f"{number:.3f}" if math.isfinite(number) else "NA"


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def make_report(
    rows: list[dict[str, object]],
    associations: list[dict[str, object]],
    target_summary: dict[str, dict[str, object]],
    overlap_sensitivity: list[dict[str, object]],
    ratio_bins: list[dict[str, object]],
    permutations: int,
    bootstraps: int,
) -> str:
    pieces = len({str(row["piece"]) for row in rows})
    lines = [
        "# Harmony fail association screening",
        "",
        "This analysis screens associations only. It does not establish causality or estimate treatment effects.",
        "",
        "## Cohort and method",
        "",
        f"- Observations: {len(rows)} outputs from {pieces} independent piece groups.",
        f"- Every included piece contributes exactly {EXPECTED_RUNS_PER_PIECE} runs.",
        "- A piece is dropped from a target-feature analysis if any of its four feature or target values is non-finite.",
        f"- Positive-skill associations use {permutations} group-block permutations; eight features receive BH-FDR correction separately within each target.",
        f"- At most two positive-skill features per target receive {bootstraps} piece-group bootstrap curve bands.",
        "",
        "## Target prevalence",
        "",
        "| Target | Positive outputs | Positive pieces | Prevalence |",
        "|---|---:|---:|---:|",
    ]
    for target in TARGETS:
        summary = target_summary[target]
        lines.append(
            f"| `{target}` | {summary['positive_count']} | {summary['positive_group_count']} | "
            f"{float(summary['prevalence']):.3f} |"
        )

    repetition = target_summary["repetition"]
    if (
        int(repetition["positive_count"]) < LOW_POSITIVE_RUN_THRESHOLD
        or int(repetition["positive_group_count"]) < LOW_POSITIVE_GROUP_THRESHOLD
    ):
        lines.extend(
            [
                "",
                "**Warning:** repetition positives are too few for stable inference; treat its KDE curves and p/q values as exploratory only.",
            ]
        )

    lines.extend(
        [
            "",
            "## Overlap-ratio sensitivity for insufficient_output",
            "",
            f"This limited follow-up uses only `{OVERLAP_FEATURE}` and keeps each piece as a complete four-run block.",
            "",
            "| Cohort | Pieces | Runs | Positives | Zero ratio | Skill | Rho | p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in overlap_sensitivity:
        lines.append(
            f"| `{row['cohort']}` | {row['pieces']} | {row['runs']} | {row['positives']} | "
            f"{row['zero_ratio']} | {_fmt(row['cv_skill'])} | {_fmt(row['rho'])} | "
            f"{_fmt(row['p_value'])} |"
        )
    lines.extend(
        [
            "",
            "### Descriptive overlap-ratio bins",
            "",
            "These bins use the complete `all` cohort and are descriptive only; they receive no permutation test or FDR correction.",
            "",
            "| Ratio bin | n | insufficient_output | Rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in ratio_bins:
        lines.append(
            f"| `{row['bin']}` | {row['n']} | {row['fail']} | {_fmt(row['rate'])} |"
        )

    lines.extend(["", "## Screening results", ""])
    for target in TARGETS:
        lines.extend(
            [
                f"### {target}",
                "",
                "| Feature | Groups | Skill | Rho | p | q | Evidence |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        ranked = sorted(
            [row for row in associations if row["target"] == target],
            key=lambda row: (
                -_as_float(row["cv_skill"]) if math.isfinite(_as_float(row["cv_skill"])) else math.inf,
                str(row["feature"]),
            ),
        )
        for row in ranked:
            lines.append(
                f"| `{row['feature']}` | {row['n_groups']} | {_fmt(row['cv_skill'])} | "
                f"{_fmt(row['rho'])} | {_fmt(row['p_value'])} | {_fmt(row['q_value'])} | "
                f"{row['evidence_status']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- Outputs from the same piece are dependent repeats and are kept together in validation, permutation, and bootstrap operations.",
            "- Association strength can reflect dataset composition, labeling noise, or an omitted musical variable.",
            "- A positive CV skill means the one-dimensional KDE improved on its outer-training prevalence baseline; it is not proof of a reliable classifier.",
            "- Curve bands describe piece-group bootstrap uncertainty and should not be interpreted as causal response curves.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    permutations: int = 99,
    bootstraps: int = 200,
    curve_points: int = 101,
    seed: int = 20260821,
) -> dict[str, object]:
    if permutations < 1 or bootstraps < 1 or curve_points < 2:
        raise ValueError("permutations/bootstraps must be positive and curve_points must be at least 2")

    input_path = input_dir / "prompt_features.csv"
    rows, load_audit = load_runs(input_path)
    associations = analyze_associations(rows, permutations=permutations, seed=seed)
    overlap_sensitivity = analyze_overlap_sensitivity(rows, permutations=permutations, seed=seed)
    ratio_bins, ratio_bin_audit = overlap_ratio_bins(rows)
    curves, curve_audit = build_curves(
        rows,
        associations,
        bootstraps=bootstraps,
        curve_points=curve_points,
        seed=seed,
    )
    summary = _target_summary(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "associations.csv", associations, ASSOCIATION_FIELDS)
    _write_csv(output_dir / "curve_points.csv", curves, CURVE_FIELDS)
    _write_csv(
        output_dir / "overlap_sensitivity.csv",
        overlap_sensitivity,
        OVERLAP_SENSITIVITY_FIELDS,
    )

    dropped_groups = {target: {} for target in TARGETS}
    for association in associations:
        raw = str(association["dropped_groups"])
        dropped_groups[str(association["target"])][str(association["feature"])] = (
            raw.split(";") if raw else []
        )
    audit: dict[str, object] = {
        "analysis_kind": "harmony_fail_association_screening",
        "causal_interpretation": False,
        "input": str(input_path.resolve()),
        "features": list(FEATURES),
        "targets": list(TARGETS),
        "observation_unit": "output",
        "independent_group": "piece",
        "required_runs_per_piece": EXPECTED_RUNS_PER_PIECE,
        "complete_group_policy": "retain a piece only when all four target and feature values are finite",
        "counts": {
            **load_audit,
            "association_count": len(associations),
            "overlap_sensitivity_count": len(overlap_sensitivity),
            "curve_model_count": len(curve_audit),
            "curve_point_count": len(curves),
        },
        "target_summary": summary,
        "settings": {
            "seed": seed,
            "permutations": permutations,
            "permutation_resolution": 1.0 / (permutations + 1),
            "bootstraps": bootstraps,
            "curve_points": curve_points,
            "fdr_scope": "eight frozen features corrected separately within each target",
            "fdr_threshold": FDR_THRESHOLD,
            "curve_selection": "up to two valid positive-skill features per target, ranked by cv_skill",
            "overlap_sensitivity": (
                "insufficient_output against relation_harmony_valid_cell_ratio in three complete-piece cohorts"
            ),
        },
        "dropped_groups": dropped_groups,
        "permutation_status_counts": dict(sorted(Counter(str(row["permutation_status"]) for row in associations).items())),
        "evidence_status_counts": dict(sorted(Counter(str(row["evidence_status"]) for row in associations).items())),
        "overlap_sensitivity": _json_safe(overlap_sensitivity),
        "overlap_ratio_bins": _json_safe(ratio_bins),
        "overlap_ratio_bin_audit": ratio_bin_audit,
        "curve_models": curve_audit,
    }
    with (output_dir / "analysis_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (output_dir / "REPORT.md").write_text(
        make_report(
            rows,
            associations,
            summary,
            overlap_sensitivity,
            ratio_bins,
            permutations,
            bootstraps,
        ),
        encoding="utf-8",
    )
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--bootstraps", type=int, default=200)
    parser.add_argument("--curve-points", type=int, default=101)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args(argv)
    if args.permutations < 1 or args.bootstraps < 1 or args.curve_points < 2:
        parser.error("permutations/bootstraps must be positive and curve-points must be at least 2")
    audit = run_analysis(
        args.input_dir,
        args.output_dir,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        curve_points=args.curve_points,
        seed=args.seed,
    )
    print(json.dumps(audit["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
