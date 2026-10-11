import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from experiments.lekai_failcase_analysis.analyze_harmony_fail_associations import (
    FEATURES,
    TARGETS,
    analyze_overlap_sensitivity,
    complete_groups,
    load_runs,
    parse_fail_types,
    run_analysis,
)
from experiments.lekai_failcase_analysis.kde_model import bh_qvalues


def _write_prompt_features(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fixture(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    rows: list[dict[str, object]] = []
    for piece_index in range(8):
        piece = f"piece_{piece_index}"
        insufficient = piece_index >= 4
        repetition = piece_index in {1, 6}
        mismatch = piece_index in {2, 3, 6, 7}
        labels = [
            name
            for name, present in (
                ("insufficient_output", insufficient),
                ("repetition", repetition),
                ("melody_mismatch", mismatch),
            )
            if present
        ]
        for run_index, (seed, condition) in enumerate(
            ((0, "single_n1"), (0, "rule_s_n5"), (1, "single_n1"), (1, "rule_s_n5"))
        ):
            jitter = piece_index * 0.002 + run_index * 0.0001
            if piece_index == 0:
                overlap_ratio = 0.0
            elif piece_index in {1, 2}:
                overlap_ratio = 0.0 if run_index == 0 else 0.10
            else:
                overlap_ratio = {3: 0.20, 4: 0.40, 5: 0.60, 6: 0.80, 7: 1.0}[piece_index]
            rows.append(
                {
                    "style": "test",
                    "piece": piece,
                    "seed": seed,
                    "condition": condition,
                    "include_main": "True",
                    "fail_type_raw": "/".join(labels),
                    "prompt_has_accompaniment": int(piece_index >= 2),
                    "relation_pcs_mean": float(insufficient) + jitter,
                    "relation_pcs_p10": float(repetition) + jitter,
                    "relation_mctd_mean": float(mismatch) + jitter,
                    "relation_mctd_p90": abs(piece_index - 3.5) + run_index * 0.01,
                    "relation_local_chroma_cosine_mean": piece_index / 7.0 + run_index * 0.001,
                    "relation_local_chroma_cosine_p10": (piece_index % 3) / 2.0 + run_index * 0.001,
                    "relation_local_chroma_cosine_std": (
                        "" if piece_index == 2 and run_index == 0 else (7 - piece_index) / 7.0 + run_index * 0.001
                    ),
                    "relation_harmony_valid_cell_ratio": overlap_ratio,
                    "continuation_note_count": 999,
                }
            )

    # Excluded rows do not participate in the four-run group contract.
    rows.append(
        {
            "style": "test",
            "piece": "excluded_piece",
            "seed": 0,
            "condition": "single_n1",
            "include_main": "False",
            "fail_type_raw": "repetition",
            "prompt_has_accompaniment": 1,
            **{feature: 123.0 for feature in FEATURES},
            "continuation_note_count": 1,
        }
    )
    _write_prompt_features(input_dir / "prompt_features.csv", rows)
    return input_dir


def test_slash_labels_are_nonexclusive_and_position_independent():
    assert parse_fail_types(" melody_mismatch / repetition / insufficient_output ") == set(TARGETS)
    assert parse_fail_types("") == set()
    assert parse_fail_types(None) == set()


def test_analysis_is_deterministic_grouped_and_fdr_corrected(tmp_path):
    input_dir = _fixture(tmp_path)
    output_a = tmp_path / "out_a"
    output_b = tmp_path / "out_b"

    audit_a = run_analysis(input_dir, output_a, permutations=7, bootstraps=8, curve_points=9, seed=41)
    audit_b = run_analysis(input_dir, output_b, permutations=7, bootstraps=8, curve_points=9, seed=41)

    for name in (
        "associations.csv",
        "curve_points.csv",
        "overlap_sensitivity.csv",
        "analysis_audit.json",
        "REPORT.md",
    ):
        assert (output_a / name).read_bytes() == (output_b / name).read_bytes()
    assert audit_a == audit_b
    assert audit_a["observation_unit"] == "output"
    assert audit_a["independent_group"] == "piece"
    assert audit_a["required_runs_per_piece"] == 4
    assert audit_a["counts"]["included_row_count"] == 32
    assert audit_a["counts"]["included_piece_count"] == 8
    assert audit_a["target_summary"]["insufficient_output"]["positive_count"] == 16
    assert audit_a["target_summary"]["repetition"]["positive_count"] == 8
    assert audit_a["target_summary"]["melody_mismatch"]["positive_count"] == 16
    assert audit_a["counts"]["overlap_sensitivity_count"] == 3

    associations = _read_csv(output_a / "associations.csv")
    assert len(associations) == len(TARGETS) * len(FEATURES)
    assert Counter(row["target"] for row in associations) == {target: 8 for target in TARGETS}
    assert {row["feature"] for row in associations} == set(FEATURES)
    assert all(int(row["n_rows"]) == 4 * int(row["n_groups"]) for row in associations)

    for target in TARGETS:
        family = [row for row in associations if row["target"] == target]
        expected_q = bh_qvalues([float(row["p_value"]) for row in family])
        assert [float(row["q_value"]) for row in family] == pytest.approx(expected_q.tolist())

    missing_feature_rows = [
        row for row in associations if row["feature"] == "relation_local_chroma_cosine_std"
    ]
    assert all(row["dropped_groups"] == "piece_2" for row in missing_feature_rows)
    assert all(row["n_groups"] == "7" and row["n_rows"] == "28" for row in missing_feature_rows)

    curve_rows = _read_csv(output_a / "curve_points.csv")
    for target in TARGETS:
        target_curves = [row for row in curve_rows if row["target"] == target]
        selected = {row["feature"] for row in target_curves}
        assert len(selected) <= 2
        assert all(sum(row["feature"] == feature for row in target_curves) == 9 for feature in selected)

    sensitivity = _read_csv(output_a / "overlap_sensitivity.csv")
    assert [row["cohort"] for row in sensitivity] == [
        "all",
        "all_four_have_acc",
        "all_four_ratio_positive",
    ]
    assert [(row["pieces"], row["runs"], row["zero_ratio"]) for row in sensitivity] == [
        ("8", "32", "6"),
        ("6", "24", "1"),
        ("5", "20", "0"),
    ]
    assert all(
        int(row["n_permutations"]) == (7 if float(row["cv_skill"]) > 0.0 else 0)
        for row in sensitivity
    )
    main_overlap = next(
        row
        for row in associations
        if row["target"] == "insufficient_output"
        and row["feature"] == "relation_harmony_valid_cell_ratio"
    )
    assert float(sensitivity[0]["cv_skill"]) == pytest.approx(float(main_overlap["cv_skill"]))
    assert float(sensitivity[0]["rho"]) == pytest.approx(float(main_overlap["rho"]))
    assert float(sensitivity[0]["p_value"]) == pytest.approx(float(main_overlap["p_value"]))

    assert [row["bin"] for row in audit_a["overlap_ratio_bins"]] == [
        "[0]",
        "(0,.25]",
        "(.25,.5]",
        "(.5,.75]",
        "(.75,1]",
    ]
    assert [row["n"] for row in audit_a["overlap_ratio_bins"]] == [6, 10, 4, 4, 8]
    assert [row["fail"] for row in audit_a["overlap_ratio_bins"]] == [0, 0, 4, 4, 8]
    assert audit_a["overlap_ratio_bin_audit"]["unbinned_count"] == 0

    for name in (
        "associations.csv",
        "curve_points.csv",
        "overlap_sensitivity.csv",
        "analysis_audit.json",
        "REPORT.md",
    ):
        assert "continuation" not in (output_a / name).read_text(encoding="utf-8").casefold()
    report = (output_a / "REPORT.md").read_text(encoding="utf-8")
    assert "does not establish causality" in report
    assert "repetition positives are too few" in report
    assert "descriptive only" in report
    assert "`all_four_have_acc`" in report
    assert "`(.75,1]`" in report


def test_complete_group_drop_removes_all_four_runs(tmp_path):
    rows, _ = load_runs(_fixture(tmp_path) / "prompt_features.csv")

    x, y, groups, order, dropped = complete_groups(
        rows, "relation_local_chroma_cosine_std", "melody_mismatch"
    )

    assert dropped == ["piece_2"]
    assert len(x) == len(y) == len(groups) == len(order) == 28
    assert Counter(groups.tolist()) == {f"piece_{index}": 4 for index in range(8) if index != 2}
    assert set(order.tolist()) == {0.0, 1.0, 2.0, 3.0}


def test_overlap_sensitivity_keeps_piece_blocks_and_uses_positive_skill_permutations(tmp_path):
    rows, _ = load_runs(_fixture(tmp_path) / "prompt_features.csv")

    sensitivity = analyze_overlap_sensitivity(rows, permutations=3, seed=17)

    assert [row["runs"] for row in sensitivity] == [32, 24, 20]
    assert all(int(row["runs"]) == 4 * int(row["pieces"]) for row in sensitivity)
    for row in sensitivity:
        if np.isfinite(float(row["cv_skill"])) and float(row["cv_skill"]) > 0.0:
            assert row["permutation_status"] == "computed_positive_skill"
            assert row["n_permutations"] == 3
        else:
            assert row["permutation_status"] != "computed_positive_skill"
            assert row["n_permutations"] == 0


def test_included_piece_must_have_exactly_four_runs(tmp_path):
    input_dir = _fixture(tmp_path)
    rows = _read_csv(input_dir / "prompt_features.csv")
    rows = [row for row in rows if not (row["piece"] == "piece_0" and row["seed"] == "0" and row["condition"] == "single_n1")]
    _write_prompt_features(input_dir / "prompt_features.csv", rows)

    with pytest.raises(ValueError, match=r"exactly four runs; found piece_0=3"):
        load_runs(input_dir / "prompt_features.csv")
