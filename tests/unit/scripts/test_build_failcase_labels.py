import csv
import json
from pathlib import Path

import pytest

from experiments.lekai_failcase_analysis.build_labels import build_labels, parse_components


REPO_ROOT = Path(__file__).resolve().parents[3]
EXCLUSIONS = REPO_ROOT / "experiments" / "lekai_failcase_analysis" / "cohort_exclusions.csv"


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_component_parsing():
    assert parse_components("1/3", "insufficient_output/repetition") == [
        (1.0, "insufficient_output"),
        (3.0, "repetition"),
    ]
    assert parse_components("3.5", "melody_mismatch") == [(3.5, "melody_mismatch")]
    assert parse_components("4", "") == [(4.0, "")]
    assert parse_components("2", "unsatable") == [(2.0, "unstable")]
    with pytest.raises(ValueError, match="count mismatch"):
        parse_components("1/3", "repetition")
    with pytest.raises(ValueError, match="between 1 and 5"):
        parse_components("5.5", "")


def test_small_csv_conditions_and_exclusions(tmp_path):
    ratings = tmp_path / "rating.csv"
    output = tmp_path / "out"
    fields = [
        "style",
        "piece",
        "seed",
        "single_n1_score",
        "single_n1_fail_type",
        "rule_s_n5_score",
        "rule_s_n5_fail_type",
        "preference",
        "comments",
    ]
    _write_csv(
        ratings,
        fields,
        [
            {
                "style": "classical",
                "piece": "kept_piece",
                "seed": "0",
                "single_n1_score": "1/3",
                "single_n1_fail_type": "insufficient_output/repetition",
                "rule_s_n5_score": "3.5",
                "rule_s_n5_fail_type": "melody_mismatch",
                "comments": "shared comment",
            },
            {
                "style": "folk",
                "piece": "folk_traditional_traditional_greensleeves",
                "seed": "0",
                "single_n1_score": "4",
                "rule_s_n5_score": "2",
                "rule_s_n5_fail_type": "unsatable",
            },
            {
                "style": "jazz_blues",
                "piece": "jazz_piece",
                "seed": "1",
                "single_n1_score": "",
                "rule_s_n5_score": "",
                "comments": "not rated",
            },
        ],
    )

    summary = build_labels(ratings, output, EXCLUSIONS, ["jazz_blues"])
    assert summary["main_piece_count"] == 1
    assert summary["main_paired_row_count"] == 1
    assert summary["main_condition_output_count"] == 2
    assert summary["exclusion_reason_counts"] == {
        "invalid_melody_block_chords": {"paired_rows": 1, "condition_outputs": 2},
        "style:jazz_blues": {"paired_rows": 1, "condition_outputs": 2},
    }

    with (output / "labels_runs.csv").open(encoding="utf-8", newline="") as handle:
        runs = list(csv.DictReader(handle))
    assert len(runs) == 6
    assert runs[0]["quality_floor"] == "1.0"
    assert runs[0]["quality_mean"] == "2.0"
    assert runs[0]["severe_fail"] == "True"
    assert runs[0]["label_status"] == "rated"
    assert runs[0]["comments"] == "shared comment"
    assert runs[1]["comments"] == "shared comment"
    assert runs[1]["score_raw"] == "3.5"
    assert runs[3]["fail_type_raw"] == "unsatable"
    assert runs[4]["label_status"] == "missing_excluded"
    assert runs[4]["quality_floor"] == ""

    with (output / "labels_components.csv").open(encoding="utf-8", newline="") as handle:
        components = list(csv.DictReader(handle))
    assert [(row["score"], row["fail_type"]) for row in components[:2]] == [
        ("1.0", "insufficient_output"),
        ("3.0", "repetition"),
    ]
    assert components[4]["fail_type"] == "unstable"
    assert json.loads((output / "cohort_summary.json").read_text(encoding="utf-8")) == summary


def test_empty_score_in_main_cohort_is_invalid(tmp_path):
    ratings = tmp_path / "rating.csv"
    _write_csv(
        ratings,
        [
            "style",
            "piece",
            "seed",
            "single_n1_score",
            "single_n1_fail_type",
            "rule_s_n5_score",
            "rule_s_n5_fail_type",
            "preference",
            "comments",
        ],
        [
            {
                "style": "classical",
                "piece": "kept_piece",
                "seed": "0",
                "single_n1_score": "",
                "rule_s_n5_score": "4",
            }
        ],
    )
    with pytest.raises(ValueError, match="included row has an empty score"):
        build_labels(ratings, tmp_path / "out", EXCLUSIONS, ["jazz_blues"])


def test_real_ratings_smoke_when_present(tmp_path):
    ratings = REPO_ROOT / (
        "remote_results/03_robustness_and_style_202608/"
        "style7x5_single_n1_vs_rule_s_n5_2seed_20260821/rating.csv"
    )
    if not ratings.exists():
        pytest.skip("real rating.csv not present at the expected path")

    summary = build_labels(ratings, tmp_path / "real", EXCLUSIONS, ["jazz_blues"])
    assert summary["main_piece_count"] == 23
    assert summary["main_paired_row_count"] == 46
    assert summary["main_condition_output_count"] == 92
