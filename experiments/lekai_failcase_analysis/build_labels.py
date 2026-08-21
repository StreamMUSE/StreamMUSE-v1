"""Build frozen run- and component-level labels for Lekai fail-case analysis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CONDITION_COLUMNS = {
    "single_n1": ("single_n1_score", "single_n1_fail_type"),
    "rule_s_n5": ("rule_s_n5_score", "rule_s_n5_fail_type"),
}
RUN_FIELDS = (
    "style",
    "piece",
    "npz_time_signature",
    "meter_status",
    "seed",
    "condition",
    "score_raw",
    "fail_type_raw",
    "quality_floor",
    "quality_mean",
    "issue_any",
    "severe_fail",
    "label_status",
    "include_main",
    "exclusion_reason",
    "comments",
)
COMPONENT_FIELDS = (
    "style",
    "piece",
    "seed",
    "condition",
    "component_index",
    "score",
    "fail_type",
    "include_main",
)


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.strip().split("/")]


def parse_components(score_raw: str, fail_type_raw: str) -> list[tuple[float, str]]:
    """Parse positional score/fail-type components and normalize known typos."""
    scores = _split(score_raw)
    if not score_raw.strip() or any(not score for score in scores):
        raise ValueError(f"invalid score: {score_raw!r}")

    fail_types = _split(fail_type_raw) if fail_type_raw.strip() else [""]
    if len(scores) != len(fail_types):
        raise ValueError(
            "score/fail type component count mismatch: "
            f"{score_raw!r} vs {fail_type_raw!r}"
        )

    result = []
    for score, fail_type in zip(scores, fail_types):
        numeric_score = float(score)
        if not 1 <= numeric_score <= 5:
            raise ValueError(f"score must be between 1 and 5: {score!r}")
        normalized = "unstable" if fail_type == "unsatable" else fail_type
        result.append((numeric_score, normalized))
    return result


def load_exclusions(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"piece", "reason"} <= set(reader.fieldnames):
            raise ValueError("exclusions CSV must contain piece and reason columns")
        return {row["piece"].strip(): row["reason"].strip() for row in reader}


def load_meter_audit(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "piece",
            "meter_status",
            "npz_time_signature",
            "meter_hold_reason",
        }
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"meter audit missing columns: {', '.join(missing)}")

        audit: dict[str, dict[str, str]] = {}
        for row_number, row in enumerate(reader, start=2):
            piece = (row["piece"] or "").strip()
            if piece in audit:
                raise ValueError(f"duplicate piece in meter audit at row {row_number}: {piece!r}")
            status = (row["meter_status"] or "").strip()
            if status not in {"include_4_4", "hold_non_4_4", "hold_unknown"}:
                raise ValueError(
                    f"invalid meter_status in meter audit at row {row_number}: {status!r}"
                )
            hold_reason = (row["meter_hold_reason"] or "").strip()
            if status != "include_4_4" and not hold_reason:
                raise ValueError(
                    f"held meter audit row {row_number} has no meter_hold_reason"
                )
            audit[piece] = {
                "meter_status": status,
                "npz_time_signature": (row["npz_time_signature"] or "").strip(),
                "meter_hold_reason": hold_reason,
            }
        return audit


def build_labels(
    ratings_path: Path,
    output_dir: Path,
    exclusions_path: Path,
    excluded_styles: list[str],
    meter_audit_path: Path | None = None,
) -> dict[str, object]:
    exclusions = load_exclusions(exclusions_path)
    meter_audit = load_meter_audit(
        meter_audit_path or Path(__file__).with_name("meter_audit.csv")
    )
    runs: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    included_pairs: set[tuple[str, str, str]] = set()
    included_pieces: set[str] = set()
    excluded_pairs: dict[tuple[str, str, str], str] = {}
    meter_pieces: dict[str, set[str]] = defaultdict(set)
    meter_pairs: dict[str, set[tuple[str, str, str]]] = defaultdict(set)

    with ratings_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"style", "piece", "seed", "comments"}
        for score_column, fail_type_column in CONDITION_COLUMNS.values():
            required.update((score_column, fail_type_column))
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"ratings CSV missing columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            style = row["style"].strip()
            piece = row["piece"].strip()
            seed = row["seed"].strip()
            if piece not in meter_audit:
                raise ValueError(
                    f"ratings row {row_number}: piece missing from meter audit: {piece!r}"
                )
            meter = meter_audit[piece]
            meter_status = meter["meter_status"]
            npz_time_signature = meter["npz_time_signature"]
            reason = exclusions.get(piece, "")
            if style in excluded_styles:
                reason = f"style:{style}"
            if not reason and meter_status != "include_4_4":
                reason = meter["meter_hold_reason"]
            include_main = not reason
            pair_key = (style, piece, seed)
            meter_pieces[meter_status].add(piece)
            meter_pairs[meter_status].add(pair_key)
            if include_main:
                included_pairs.add(pair_key)
                included_pieces.add(piece)
            else:
                excluded_pairs[pair_key] = reason

            for condition, (score_column, fail_type_column) in CONDITION_COLUMNS.items():
                score_raw = (row[score_column] or "").strip()
                fail_type_raw = (row[fail_type_column] or "").strip()
                comments = (row["comments"] or "").strip()
                if not score_raw:
                    if include_main:
                        raise ValueError(
                            f"ratings row {row_number}, {condition}: "
                            "included row has an empty score"
                        )
                    runs.append(
                        {
                            "style": style,
                            "piece": piece,
                            "npz_time_signature": npz_time_signature,
                            "meter_status": meter_status,
                            "seed": seed,
                            "condition": condition,
                            "score_raw": "",
                            "fail_type_raw": fail_type_raw,
                            "quality_floor": "",
                            "quality_mean": "",
                            "issue_any": "",
                            "severe_fail": "",
                            "label_status": "missing_excluded",
                            "include_main": False,
                            "exclusion_reason": reason,
                            "comments": comments,
                        }
                    )
                    continue
                try:
                    parsed = parse_components(score_raw, fail_type_raw)
                except ValueError as exc:
                    raise ValueError(f"ratings row {row_number}, {condition}: {exc}") from exc

                scores = [score for score, _ in parsed]
                issue_any = any(fail_type for _, fail_type in parsed)
                run = {
                    "style": style,
                    "piece": piece,
                    "npz_time_signature": npz_time_signature,
                    "meter_status": meter_status,
                    "seed": seed,
                    "condition": condition,
                    "score_raw": score_raw,
                    "fail_type_raw": fail_type_raw,
                    "quality_floor": min(scores),
                    "quality_mean": sum(scores) / len(scores),
                    "issue_any": issue_any,
                    "severe_fail": min(scores) <= 2,
                    "label_status": "rated",
                    "include_main": include_main,
                    "exclusion_reason": reason,
                    "comments": comments,
                }
                runs.append(run)
                for index, (score, fail_type) in enumerate(parsed, start=1):
                    components.append(
                        {
                            "style": style,
                            "piece": piece,
                            "seed": seed,
                            "condition": condition,
                            "component_index": index,
                            "score": score,
                            "fail_type": fail_type,
                            "include_main": include_main,
                        }
                    )

    reason_pair_counts = Counter(excluded_pairs.values())
    reason_output_counts = Counter(
        str(run["exclusion_reason"]) for run in runs if not run["include_main"]
    )
    meter_output_counts = Counter(str(run["meter_status"]) for run in runs)
    summary: dict[str, object] = {
        "main_piece_count": len(included_pieces),
        "main_paired_row_count": len(included_pairs),
        "main_condition_output_count": sum(bool(run["include_main"]) for run in runs),
        "meter_status_counts": {
            status: {
                "pieces": len(meter_pieces[status]),
                "paired_rows": len(meter_pairs[status]),
                "condition_outputs": meter_output_counts[status],
            }
            for status in sorted(meter_pieces)
        },
        "exclusion_reason_counts": {
            reason: {
                "paired_rows": reason_pair_counts[reason],
                "condition_outputs": reason_output_counts[reason],
            }
            for reason in sorted(reason_pair_counts)
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "labels_runs.csv", RUN_FIELDS, runs)
    _write_csv(output_dir / "labels_components.csv", COMPONENT_FIELDS, components)
    with (output_dir / "cohort_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=Path(__file__).with_name("cohort_exclusions.csv"),
    )
    parser.add_argument(
        "--meter-audit",
        type=Path,
        default=Path(__file__).with_name("meter_audit.csv"),
    )
    parser.add_argument("--exclude-style", action="append", default=None)
    args = parser.parse_args(argv)
    build_labels(
        args.ratings,
        args.output_dir,
        args.exclusions,
        args.exclude_style if args.exclude_style is not None else ["jazz_blues"],
        args.meter_audit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
