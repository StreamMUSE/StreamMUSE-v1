"""Run the frozen 4/4 fail-case KDE association screening analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
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
else:
    from kde_model import (  # type: ignore[no-redef]
        bh_qvalues,
        group_block_permutation_skill,
        group_bootstrap_curve_bands,
        nested_logo_cv,
        select_logo_bandwidth_and_curve,
        spearman_nan,
    )


FAIL_TYPES = ("insufficient_output", "melody_mismatch", "repetition")
PRIMARY_TARGET = {
    "melody": "severe_fail_rate",
    "prompt": "quality_risk",
    "paired": "quality_gain",
}
MELODY_TARGETS = (
    "severe_fail_rate", "quality_risk", "issue_any_rate",
    "insufficient_output_rate", "melody_mismatch_rate", "repetition_rate",
)
PROMPT_TARGETS = (
    "quality_risk", "severe_fail", "issue_any",
    "insufficient_output", "melody_mismatch", "repetition",
)
PAIRED_TARGETS = ("quality_gain", "severe_fail_reduction", "insufficient_output_reduction")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str] | None = None) -> None:
    fields = list(fieldnames or [])
    if not fields:
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def csv_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def parse_fail_types(raw: object) -> set[str]:
    return {part.strip().casefold() for part in str(raw or "").split("/") if part.strip()}


def load_spec(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    required = {"layer", "feature", "role", "family", "rationale"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"analysis spec missing columns: {sorted(required - set(rows[0] if rows else {}))}")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["layer"], row["feature"])
        if key in seen:
            raise ValueError(f"duplicate analysis spec feature: {key}")
        seen.add(key)
    return rows


def audit_run_condition_trim_flag(input_dir: Path) -> dict[str, object]:
    feature_audit_path = input_dir / "feature_audit.json"
    unknown = {
        "status": "unknown",
        "path": None,
        "trim_leading_rest_flag_present": None,
        "interpretation": "Run-condition audit unavailable because feature_audit.json or technical/metadata/run_condition.sh is missing.",
    }
    if not feature_audit_path.exists():
        return unknown
    try:
        feature_audit = json.loads(feature_audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return unknown
    bundle_dir = feature_audit.get("bundle_dir")
    if not bundle_dir:
        return unknown
    script_path = Path(str(bundle_dir)) / "technical" / "metadata" / "run_condition.sh"
    if not script_path.exists():
        return {
            "status": "unknown",
            "path": str(script_path),
            "trim_leading_rest_flag_present": None,
            "interpretation": "Run-condition audit unavailable because technical/metadata/run_condition.sh is missing.",
        }
    script_text = script_path.read_text(encoding="utf-8")
    has_flag = "--trim-leading-rest" in script_text
    if has_flag:
        interpretation = (
            "run_condition.sh passes --trim-leading-rest, so leading blank beats were trimmed before model input."
        )
    else:
        interpretation = (
            "run_condition.sh calls scripts/run_lekai_prompt_continuation_offline.py without "
            "--trim-leading-rest, so the leading blank remained part of the actual model input."
        )
    return {
        "status": "ok",
        "path": str(script_path),
        "trim_leading_rest_flag_present": has_flag,
        "interpretation": interpretation,
    }


def build_outcomes(
    melody_rows: list[dict[str, str]], prompt_rows: list[dict[str, str]], spec: list[dict[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    melody_features = [row["feature"] for row in spec if row["layer"] == "melody" and row["role"] != "unavailable"]
    prompt_features = [row["feature"] for row in spec if row["layer"] == "prompt" and row["role"] != "unavailable"]
    melody_main = {row["piece"]: row for row in melody_rows if as_bool(row.get("include_main"))}
    prompt_main = [row for row in prompt_rows if as_bool(row.get("include_main"))]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    unknown = Counter()
    for row in prompt_main:
        grouped[row["piece"]].append(row)
        unknown.update(parse_fail_types(row.get("fail_type_raw")) - set(FAIL_TYPES))
    if set(grouped) != set(melody_main):
        raise ValueError("included Melody and Prompt piece sets differ")

    melody_outcomes: list[dict[str, object]] = []
    for piece in sorted(melody_main):
        runs = grouped[piece]
        if len(runs) != 4:
            raise ValueError(f"expected four included runs for {piece}, found {len(runs)}")
        source = melody_main[piece]
        floors = np.array([as_float(row["quality_floor"]) for row in runs])
        result: dict[str, object] = {
            "style": source["style"], "piece": piece, "run_count": len(runs),
            "severe_fail_rate": float(np.mean([as_bool(row["severe_fail"]) for row in runs])),
            "quality_floor_mean": float(np.mean(floors)),
            "quality_risk": float(5.0 - np.mean(floors)),
            "issue_any_rate": float(np.mean([as_bool(row["issue_any"]) for row in runs])),
        }
        for fail_type in FAIL_TYPES:
            result[f"{fail_type}_rate"] = float(np.mean([fail_type in parse_fail_types(row["fail_type_raw"]) for row in runs]))
        result.update({feature: as_float(source.get(feature)) for feature in melody_features})
        melody_outcomes.append(result)

    run_outcomes: list[dict[str, object]] = []
    by_pair: dict[tuple[str, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for source in sorted(prompt_main, key=lambda row: (row["piece"], int(row["seed"]), row["condition"])):
        types = parse_fail_types(source.get("fail_type_raw"))
        row: dict[str, object] = {
            "style": source["style"], "piece": source["piece"], "seed": int(source["seed"]),
            "condition": source["condition"], "quality_floor": as_float(source["quality_floor"]),
            "quality_risk": 5.0 - as_float(source["quality_floor"]),
            "severe_fail": int(as_bool(source["severe_fail"])),
            "issue_any": int(as_bool(source["issue_any"])),
            "fail_type_raw": source.get("fail_type_raw", ""),
            "prompt_has_accompaniment": int(as_float(source.get("prompt_has_accompaniment")) == 1.0),
        }
        row.update({fail_type: int(fail_type in types) for fail_type in FAIL_TYPES})
        row.update({feature: as_float(source.get(feature)) for feature in prompt_features})
        run_outcomes.append(row)
        key = (str(row["piece"]), int(row["seed"]))
        condition = str(row["condition"])
        if condition in by_pair[key]:
            raise ValueError(f"duplicate condition in pair: {key} {condition}")
        by_pair[key][condition] = row

    paired: list[dict[str, object]] = []
    for (piece, seed), conditions in sorted(by_pair.items()):
        if set(conditions) != {"single_n1", "rule_s_n5"}:
            raise ValueError(f"incomplete exact pair for {(piece, seed)}: {sorted(conditions)}")
        single, rule = conditions["single_n1"], conditions["rule_s_n5"]
        row = {
            "style": single["style"], "piece": piece, "seed": seed,
            "quality_gain": as_float(rule["quality_floor"]) - as_float(single["quality_floor"]),
            "severe_fail_reduction": as_float(single["severe_fail"]) - as_float(rule["severe_fail"]),
            "insufficient_output_reduction": as_float(single["insufficient_output"]) - as_float(rule["insufficient_output"]),
        }
        for feature in prompt_features:
            row[f"delta_{feature}"] = as_float(rule[feature]) - as_float(single[feature])
        paired.append(row)
    return melody_outcomes, run_outcomes, paired, dict(sorted(unknown.items()))


def complete_groups(
    rows: list[dict[str, object]], feature: str, target: str, expected_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]], list[str]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["piece"])].append(row)
    kept: list[dict[str, object]] = []
    dropped: list[str] = []
    for piece in sorted(grouped):
        block = grouped[piece]
        complete = len(block) == expected_size and all(
            math.isfinite(as_float(row.get(feature))) and math.isfinite(as_float(row.get(target))) for row in block
        )
        if complete:
            kept.extend(sorted(block, key=lambda row: (int(row.get("seed", 0)), str(row.get("condition", "")))))
        else:
            dropped.append(piece)
    x = np.array([as_float(row[feature]) for row in kept], dtype=float)
    y = np.array([as_float(row[target]) for row in kept], dtype=float)
    groups = np.array([str(row["piece"]) for row in kept], dtype=object)
    return x, y, groups, kept, dropped


def stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return (int(base) + int.from_bytes(digest[:4], "big")) % (2**32)


def centered_spearman(rows: list[dict[str, object]], feature: str, target: str) -> float:
    by_piece: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_piece[str(row["piece"])].append(row)
    xc: list[float] = []
    yc: list[float] = []
    for block in by_piece.values():
        xb = np.array([as_float(row[feature]) for row in block])
        yb = np.array([as_float(row[target]) for row in block])
        xc.extend((xb - xb.mean()).tolist())
        yc.extend((yb - yb.mean()).tolist())
    return spearman_nan(xc, yc)


def analyze_associations(
    melody: list[dict[str, object]], prompt: list[dict[str, object]], paired: list[dict[str, object]],
    spec: list[dict[str, str]], n_perm: int, seed: int,
) -> list[dict[str, object]]:
    analyses = (
        ("melody", melody, MELODY_TARGETS, 1),
        ("prompt", prompt, PROMPT_TARGETS, 4),
        ("paired", paired, PAIRED_TARGETS, 2),
    )
    spec_by_layer = {
        "melody": [row for row in spec if row["layer"] == "melody" and row["role"] != "unavailable"],
        "prompt": [row for row in spec if row["layer"] == "prompt" and row["role"] != "unavailable"],
        "paired": [row for row in spec if row["layer"] == "prompt" and row["role"] != "unavailable"],
    }
    results: list[dict[str, object]] = []
    for layer, rows, targets, expected_size in analyses:
        all_groups = sorted({str(row["piece"]) for row in rows})
        for target in targets:
            for item in spec_by_layer[layer]:
                base_feature = item["feature"]
                feature = f"delta_{base_feature}" if layer == "paired" else base_feature
                x, y, groups, used, dropped = complete_groups(rows, feature, target, expected_size)
                cv = nested_logo_cv(x, y, groups)
                valid = bool(cv["valid"])
                skill = as_float(cv.get("skill")) if valid else float("nan")
                primary_test = target == PRIMARY_TARGET[layer]
                permutation_status = "descriptive_only"
                p_value = float("nan")
                if primary_test and valid and skill > 0.0:
                    order = np.arange(len(x), dtype=float)
                    if layer != "melody":
                        order = np.array([
                            int(row.get("seed", 0)) * 2 + (1 if row.get("condition") == "single_n1" else 0)
                            for row in used
                        ], dtype=float)
                    perm = group_block_permutation_skill(
                        x, y, groups, n_perm=n_perm,
                        seed=stable_seed(seed, layer, target, base_feature), within_group_order=order,
                    )
                    p_value = as_float(perm.get("p_value"))
                    permutation_status = "computed_screening" if perm["valid"] else "invalid_permutation"
                elif primary_test and valid:
                    p_value = 1.0
                    permutation_status = "not_tested_nonpositive_skill"
                elif primary_test:
                    p_value = 1.0
                    permutation_status = "not_tested_invalid_feature"
                result: dict[str, object] = {
                    "layer": layer, "target": target, "role": item["role"], "family": item["family"],
                    "feature": base_feature, "analyzed_column": feature,
                    "n_rows_total": len(rows), "n_rows_used": len(x),
                    "n_groups_total": len(all_groups), "n_groups_used": len(set(groups.tolist())),
                    "dropped_group_count": len(dropped), "dropped_groups": ";".join(dropped),
                    "valid": valid,
                    "invalid_reason": "" if valid else json.dumps(cv.get("reason"), sort_keys=True),
                    "cv_skill": skill, "cv_mse": as_float(cv.get("cv_mse")),
                    "null_mse": as_float(cv.get("null_mse")),
                    "spearman_rho": spearman_nan(x, y),
                    "within_piece_centered_spearman": centered_spearman(used, feature, target) if layer == "prompt" and used else float("nan"),
                    "permutation_status": permutation_status, "n_permutations": n_perm if permutation_status == "computed_screening" else 0,
                    "p_value": p_value, "q_value": float("nan"), "evidence_status": "descriptive_only",
                    "sensitivity_status": "not_top_prompt_association",
                    "nonempty_only_cv_skill": float("nan"), "nonempty_only_cv_mse": float("nan"),
                    "nonempty_only_null_mse": float("nan"), "nonempty_only_n_groups": 0,
                    "nonempty_only_dropped_groups": "",
                }
                results.append(result)

    # BH spans every preregistered feature in the same role family. The thematic
    # `family` column remains descriptive and must not split correction into tiny sets.
    families: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(results):
        if row["target"] == PRIMARY_TARGET[row["layer"]]:
            families[(str(row["layer"]), str(row["target"]), str(row["role"]))].append(index)
    for indices in families.values():
        qvalues = bh_qvalues([as_float(results[index]["p_value"]) for index in indices])
        for index, qvalue in zip(indices, qvalues):
            row = results[index]
            row["q_value"] = float(qvalue)
            skill = as_float(row["cv_skill"])
            pvalue = as_float(row["p_value"])
            if skill > 0.0 and qvalue <= 0.10:
                row["evidence_status"] = "fdr_supported_screening"
            elif skill > 0.0 and pvalue <= 0.10:
                row["evidence_status"] = "suggestive_screening"
            else:
                row["evidence_status"] = "unsupported"
    return results


def top_primary_associations(associations: list[dict[str, object]], layer: str) -> list[dict[str, object]]:
    candidates = [
        row for row in associations
        if row["layer"] == layer and row["target"] == PRIMARY_TARGET[layer]
        and row["role"] == "primary" and as_bool(row["valid"]) and as_float(row["cv_skill"]) > 0.0
    ]
    return sorted(candidates, key=lambda row: as_float(row["cv_skill"]), reverse=True)[:3]


def empty_prompt_contingency(prompt: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    empty_cases: list[dict[str, object]] = []
    for row in prompt:
        present = int(as_float(row.get("prompt_has_accompaniment")) == 1.0)
        grouped[(str(row["condition"]), present)].append(row)
        if not present:
            empty_cases.append({
                "style": row["style"], "piece": row["piece"], "seed": row["seed"],
                "condition": row["condition"], "quality_floor": row["quality_floor"],
                "severe_fail": row["severe_fail"], "issue_any": row["issue_any"],
                "insufficient_output": row["insufficient_output"], "fail_type_raw": row["fail_type_raw"],
            })
    rows: list[dict[str, object]] = []
    for condition in sorted({str(row["condition"]) for row in prompt}):
        for present in (0, 1):
            block = grouped.get((condition, present), [])
            rows.append({
                "condition": condition, "prompt_has_accompaniment": present, "n_runs": len(block),
                "severe_fail_count": int(sum(as_float(row["severe_fail"]) for row in block)),
                "issue_any_count": int(sum(as_float(row["issue_any"]) for row in block)),
                "insufficient_output_count": int(sum(as_float(row["insufficient_output"]) for row in block)),
                "quality_floor_mean": float(np.mean([as_float(row["quality_floor"]) for row in block])) if block else float("nan"),
                "pieces": ";".join(sorted({str(row["piece"]) for row in block})),
            })
    return rows, sorted(empty_cases, key=lambda row: (str(row["piece"]), int(row["seed"]), str(row["condition"])))


def add_nonempty_prompt_sensitivity(
    associations: list[dict[str, object]], prompt: list[dict[str, object]]
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in prompt:
        grouped[str(row["piece"])].append(row)
    complete_nonempty = {
        piece for piece, block in grouped.items()
        if len(block) == 4 and all(as_float(row.get("prompt_has_accompaniment")) == 1.0 for row in block)
    }
    nonempty_rows = [row for row in prompt if str(row["piece"]) in complete_nonempty]
    all_groups = set(grouped)
    for item in top_primary_associations(associations, "prompt"):
        feature, target = str(item["analyzed_column"]), str(item["target"])
        x, y, groups, _, feature_dropped = complete_groups(nonempty_rows, feature, target, 4)
        cv = nested_logo_cv(x, y, groups)
        dropped = sorted((all_groups - complete_nonempty) | set(feature_dropped))
        item["sensitivity_status"] = "nonempty_only_computed" if cv["valid"] else "nonempty_only_invalid"
        item["nonempty_only_cv_skill"] = as_float(cv.get("skill")) if cv["valid"] else float("nan")
        item["nonempty_only_cv_mse"] = as_float(cv.get("cv_mse"))
        item["nonempty_only_null_mse"] = as_float(cv.get("null_mse"))
        item["nonempty_only_n_groups"] = len(set(groups.tolist()))
        item["nonempty_only_dropped_groups"] = ";".join(dropped)
    return associations


def make_curves(
    associations: list[dict[str, object]], melody: list[dict[str, object]], prompt: list[dict[str, object]],
    paired: list[dict[str, object]], n_boot: int, n_curve: int, seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source = {"melody": (melody, 1), "prompt": (prompt, 4), "paired": (paired, 2)}
    selected: list[dict[str, object]] = []
    for layer in ("melody", "prompt", "paired"):
        selected.extend(top_primary_associations(associations, layer))
    curves: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for item in selected:
        layer, target = str(item["layer"]), str(item["target"])
        rows, expected = source[layer]
        feature = str(item["analyzed_column"])
        x, y, groups, _, dropped = complete_groups(rows, feature, target, expected)
        fitted = select_logo_bandwidth_and_curve(x, y, groups, n_curve=n_curve)
        bands = group_bootstrap_curve_bands(
            x, y, groups, curve_x=fitted["curve_x"], bandwidth=fitted["bandwidth"],
            n_boot=n_boot, seed=stable_seed(seed, "bootstrap", layer, target, str(item["feature"])),
        )
        for xv, pred, lower, upper in zip(bands["curve_x"], bands["curve_y"], bands["lower"], bands["upper"]):
            curves.append({
                "layer": layer, "feature": item["feature"], "target": target,
                "x": float(xv), "pred": float(pred), "lower": float(lower), "upper": float(upper),
            })
        audit.append({
            "layer": layer, "feature": item["feature"], "target": target,
            "cv_skill": item["cv_skill"], "bandwidth": fitted["bandwidth"],
            "n_groups": len(set(groups.tolist())), "dropped_groups": dropped,
        })
    return curves, audit


def report_markdown(
    associations: list[dict[str, object]], melody: list[dict[str, object]], prompt: list[dict[str, object]],
    paired: list[dict[str, object]], contingency: list[dict[str, object]], empty_cases: list[dict[str, object]],
    n_perm: int, n_boot: int, elapsed: float, run_condition_audit: dict[str, object],
) -> str:
    lines = [
        "# Frozen 4/4 KDE association screening", "",
        "This is a first-pass nonlinear screening analysis, not a confirmatory model or classifier.", "",
        "## Cohort and outcomes", "",
        f"- Independent groups: {len(melody)} pieces; Prompt runs: {len(prompt)}; exact condition pairs: {len(paired)}.",
        f"- Mean severe-fail rate: {np.mean([as_float(row['severe_fail_rate']) for row in melody]):.3f}; mean quality risk: {np.mean([as_float(row['quality_risk']) for row in prompt]):.3f}.",
        f"- Strict nested group permutations: {n_perm} (minimum attainable p={(1/(n_perm+1)):.3f}); group bootstraps for selected curves: {n_boot}.",
        "- PPL is unavailable because raw Prompt token log probabilities were not retained.", "",
        "## Empty-Prompt contingency", "",
        f"There are exactly {len(empty_cases)} valid empty Prompts. Density and relation associations on all runs may be driven by these cases.", "",
    ]
    for row in contingency:
        if int(row["n_runs"]) > 0:
            lines.append(
                f"- `{row['condition']}`, has_accompaniment={row['prompt_has_accompaniment']}: "
                f"n={row['n_runs']}, severe={row['severe_fail_count']}, issue={row['issue_any_count']}, "
                f"insufficient={row['insufficient_output_count']}, mean floor={as_float(row['quality_floor_mean']):.3f}."
            )
    for row in empty_cases:
        lines.append(
            f"- Empty case: `{row['piece']}`, seed={row['seed']}, `{row['condition']}`, "
            f"floor={as_float(row['quality_floor']):.1f}, fail_type=`{row['fail_type_raw']}`."
        )
    lines.extend(["", "## Association screening", ""])
    for layer in ("melody", "prompt", "paired"):
        primary = [row for row in associations if row["layer"] == layer and row["target"] == PRIMARY_TARGET[layer]]
        positive = sum(as_float(row["cv_skill"]) > 0 for row in primary if as_bool(row["valid"]))
        negative = sum(as_float(row["cv_skill"]) <= 0 for row in primary if as_bool(row["valid"]))
        lines.append(f"### {layer}: {PRIMARY_TARGET[layer]}")
        lines.append("")
        lines.append(f"Positive-skill features: {positive}; nonpositive-skill features: {negative}. Top numeric results:")
        lines.append("")
        ranked = sorted([row for row in primary if as_bool(row["valid"])], key=lambda row: as_float(row["cv_skill"]), reverse=True)[:5]
        for row in ranked:
            p = as_float(row["p_value"]); q = as_float(row["q_value"])
            ptext = f"{p:.3f}" if math.isfinite(p) else "NA"
            qtext = f"{q:.3f}" if math.isfinite(q) else "NA"
            lines.append(f"- `{row['feature']}`: skill={as_float(row['cv_skill']):.3f}, rho={as_float(row['spearman_rho']):.3f}, p={ptext}, q={qtext}, {row['evidence_status']}.")
            if row["feature"] == "first8_leading_blank_beats":
                trim_flag = run_condition_audit.get("trim_leading_rest_flag_present")
                lines.append(
                    "  Pipeline diagnostic: the actual run script calls "
                    "`scripts/run_lekai_prompt_continuation_offline.py` without `--trim-leading-rest`, "
                    "so this association is best read as an input-segmentation diagnostic rather than an inherent "
                    "musical style or difficulty effect."
                    if trim_flag is False else
                    "  Pipeline diagnostic: interpret leading-blank effects as an input-segmentation audit, not an "
                    "inherent musical style or difficulty effect."
                )
            if row["sensitivity_status"] != "not_top_prompt_association":
                lines.append(
                    f"  Nonempty-only sensitivity: skill={as_float(row['nonempty_only_cv_skill']):.3f}, "
                    f"groups={row['nonempty_only_n_groups']} ({row['sensitivity_status']}); no extra permutation."
                )
        lines.append("")
    lines.extend([
        "## Interpretation limits", "",
        "- Piece is always the independent group; seeds and conditions are never randomly split.",
        "- Secondary outcomes are descriptive only and have no p/q values.",
        "- Features with undefined pitch or duration values drop the complete piece group, not isolated runs.",
        "- Curve shape alone is not evidence. Negative CV skill is retained and means KDE underperformed the outer-training constant prior.",
        "- Rerun only surviving top candidates with 999 strict nested permutations before making confirmatory claims.",
        f"- Elapsed time: {elapsed:.1f} seconds.", "",
    ])
    return "\n".join(lines)


def run_analysis(
    input_dir: Path, output_dir: Path, spec_path: Path, n_perm: int = 99,
    n_boot: int = 200, n_curve: int = 101, seed: int = 20260821,
) -> dict[str, object]:
    started = time.perf_counter()
    spec = load_spec(spec_path)
    melody_source = read_csv(input_dir / "melody_features.csv")
    prompt_source = read_csv(input_dir / "prompt_features.csv")
    melody, prompt, paired, unknown = build_outcomes(melody_source, prompt_source, spec)
    associations = analyze_associations(melody, prompt, paired, spec, n_perm=n_perm, seed=seed)
    associations = add_nonempty_prompt_sensitivity(associations, prompt)
    contingency, empty_cases = empty_prompt_contingency(prompt)
    curves, curve_audit = make_curves(associations, melody, prompt, paired, n_boot=n_boot, n_curve=n_curve, seed=seed)
    run_condition_audit = audit_run_condition_trim_flag(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "melody_outcomes.csv", melody)
    write_csv(output_dir / "prompt_run_outcomes.csv", prompt)
    write_csv(output_dir / "paired_deltas.csv", paired)
    write_csv(output_dir / "empty_prompt_contingency.csv", contingency)
    write_csv(output_dir / "empty_prompt_cases.csv", empty_cases, (
        "style", "piece", "seed", "condition", "quality_floor", "severe_fail", "issue_any",
        "insufficient_output", "fail_type_raw",
    ))
    write_csv(output_dir / "associations.csv", associations)
    write_csv(output_dir / "curve_points.csv", curves, ("layer", "feature", "target", "x", "pred", "lower", "upper"))
    elapsed = time.perf_counter() - started
    audit = {
        "analysis_kind": "first_pass_kde_screening",
        "inputs": {"melody_features": str((input_dir / 'melody_features.csv').resolve()),
                   "prompt_features": str((input_dir / 'prompt_features.csv').resolve()),
                   "analysis_spec": str(spec_path.resolve())},
        "counts": {"pieces": len(melody), "prompt_runs": len(prompt), "exact_pairs": len(paired),
                   "empty_prompts": len(empty_cases),
                   "associations": len(associations), "curves": len(curve_audit)},
        "settings": {"seed": seed, "n_permutations": n_perm, "permutation_resolution": 1.0 / (n_perm + 1),
                     "n_bootstraps": n_boot, "n_curve_points": n_curve,
                     "permutation_scope": "three_primary_layer_targets_positive_skill_only"},
        "permutation_status_counts": dict(sorted(Counter(str(row["permutation_status"]) for row in associations).items())),
        "evidence_status_counts": dict(sorted(Counter(str(row["evidence_status"]) for row in associations).items())),
        "unknown_fail_type_tokens": unknown,
        "run_condition_audit": run_condition_audit,
        "curve_models": curve_audit,
        "empty_prompt_cases": empty_cases,
        "elapsed_seconds": elapsed,
    }
    with (output_dir / "analysis_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (output_dir / "REPORT.md").write_text(
        report_markdown(
            associations, melody, prompt, paired, contingency, empty_cases, n_perm, n_boot, elapsed, run_condition_audit
        ), encoding="utf-8"
    )
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=here / "results" / "4_4_v1")
    parser.add_argument("--output-dir", type=Path, default=here / "results" / "4_4_v1" / "kde_v1")
    parser.add_argument("--analysis-spec", type=Path, default=here / "analysis_spec.csv")
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--bootstraps", type=int, default=200)
    parser.add_argument("--curve-points", type=int, default=101)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args(argv)
    if args.permutations < 1 or args.bootstraps < 1 or args.curve_points < 2:
        parser.error("permutations/bootstraps must be positive and curve-points must be at least 2")
    audit = run_analysis(args.input_dir, args.output_dir, args.analysis_spec,
                         args.permutations, args.bootstraps, args.curve_points, args.seed)
    print(json.dumps({"counts": audit["counts"], "elapsed_seconds": audit["elapsed_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
