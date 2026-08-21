import csv
import json
from pathlib import Path

from experiments.lekai_failcase_analysis.analyze_kde_associations import (
    audit_run_condition_trim_flag,
    build_outcomes,
    load_spec,
    parse_fail_types,
    read_csv,
    run_analysis,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "input"
    spec_path = tmp_path / "analysis_spec.csv"
    _write(spec_path, [
        {"layer": "melody", "feature": "first8_onset_note_density", "role": "primary", "family": "density", "rationale": "test"},
        {"layer": "melody", "feature": "first8_leading_blank_beats", "role": "diagnostic", "family": "diagnostic", "rationale": "test"},
        {"layer": "prompt", "feature": "prompt_onset_note_density", "role": "primary", "family": "density", "rationale": "test"},
        {"layer": "prompt", "feature": "prompt_duration_median", "role": "primary", "family": "duration", "rationale": "test"},
        {"layer": "prompt", "feature": "prompt_has_accompaniment", "role": "primary", "family": "coverage", "rationale": "test"},
        {"layer": "prompt", "feature": "PPL", "role": "unavailable", "family": "not_analyzed", "rationale": "unavailable"},
    ])
    melody = []
    prompt = []
    fail_types = {
        "p0": ["", "", "", ""],
        "p1": ["melody_mismatch", "", "melody_mismatch", ""],
        "p2": ["insufficient_output/melody_mismatch", "insufficient_output", "melody_mismatch/insufficient_output", ""],
        "p3": ["repetition/melody_mismatch", "repetition", "melody_mismatch/repetition", "other"],
    }
    floors = {"p0": [5, 5, 5, 5], "p1": [4, 4, 4, 4], "p2": [2, 3, 2, 3], "p3": [1, 2, 1, 2]}
    for piece_index, piece in enumerate(("p0", "p1", "p2", "p3")):
        melody.append({
            "style": "test", "piece": piece, "include_main": "True",
            "first8_onset_note_density": piece_index,
            "first8_leading_blank_beats": piece_index % 2,
            "continuation_note_count": 999,
        })
        run_index = 0
        for seed in (0, 1):
            for condition in ("single_n1", "rule_s_n5"):
                floor = floors[piece][run_index]
                fail = fail_types[piece][run_index]
                duration = "" if piece == "p2" and seed == 0 and condition == "single_n1" else 0.5 + piece_index
                prompt.append({
                    "style": "test", "piece": piece, "seed": seed, "condition": condition,
                    "include_main": "True", "quality_floor": floor,
                    "severe_fail": str(floor <= 2), "issue_any": str(bool(fail)),
                    "fail_type_raw": fail,
                    "prompt_has_accompaniment": 0 if piece == "p0" and seed == 0 and condition == "single_n1" else 1,
                    "prompt_onset_note_density": piece_index + (0.8 if condition == "rule_s_n5" else 0.0) + seed * 0.1,
                    "prompt_duration_median": duration,
                    "continuation_note_count": 777,
                })
                run_index += 1
    _write(input_dir / "melody_features.csv", melody)
    _write(input_dir / "prompt_features.csv", prompt)
    return input_dir, spec_path


def _write_run_condition_fixture(tmp_path: Path, script_body: str | None) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = tmp_path / "bundle"
    (input_dir / "feature_audit.json").write_text(
        json.dumps({"bundle_dir": str(bundle_dir)}),
        encoding="utf-8",
    )
    if script_body is not None:
        script_path = bundle_dir / "technical" / "metadata" / "run_condition.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_body, encoding="utf-8")
    return input_dir


def test_fail_type_parser_is_position_independent():
    assert parse_fail_types(" repetition / melody_mismatch ") == {"repetition", "melody_mismatch"}
    assert parse_fail_types("") == set()


def test_outcomes_use_exact_pairs_and_parse_slash_labels(tmp_path):
    input_dir, spec_path = _fixture(tmp_path)
    melody, prompt, paired, unknown = build_outcomes(
        read_csv(input_dir / "melody_features.csv"),
        read_csv(input_dir / "prompt_features.csv"),
        load_spec(spec_path),
    )

    assert len(melody) == 4
    assert len(prompt) == 16
    assert len(paired) == 8
    p3 = next(row for row in melody if row["piece"] == "p3")
    assert p3["melody_mismatch_rate"] == 0.5
    assert p3["repetition_rate"] == 0.75
    pair = next(row for row in paired if row["piece"] == "p0" and row["seed"] == 0)
    assert pair["delta_prompt_onset_note_density"] == 0.8
    assert unknown == {"other": 1}


def test_real_analysis_contract_is_grouped_deterministic_and_excludes_continuation(tmp_path):
    input_dir, spec_path = _fixture(tmp_path)
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    audit1 = run_analysis(input_dir, out1, spec_path, n_perm=3, n_boot=4, n_curve=7, seed=31)
    audit2 = run_analysis(input_dir, out2, spec_path, n_perm=3, n_boot=4, n_curve=7, seed=31)

    associations1 = read_csv(out1 / "associations.csv")
    associations2 = read_csv(out2 / "associations.csv")
    assert associations1 == associations2
    assert audit1["counts"] == audit2["counts"]
    assert audit1["settings"] == audit2["settings"]
    assert audit1["settings"]["permutation_resolution"] == 0.25

    duration = next(row for row in associations1 if row["layer"] == "prompt"
                    and row["target"] == "quality_risk" and row["feature"] == "prompt_duration_median")
    assert duration["n_groups_total"] == "4"
    assert duration["n_groups_used"] == "3"
    assert duration["dropped_groups"] == "p2"

    secondary = next(row for row in associations1 if row["layer"] == "prompt"
                     and row["target"] == "melody_mismatch" and row["feature"] == "prompt_onset_note_density")
    assert secondary["permutation_status"] == "descriptive_only"
    assert secondary["p_value"] == ""
    assert secondary["q_value"] == ""

    for output_name in ("melody_outcomes.csv", "prompt_run_outcomes.csv", "paired_deltas.csv", "associations.csv"):
        header = (out1 / output_name).read_text(encoding="utf-8").splitlines()[0]
        assert "continuation" not in header
    assert (out1 / "curve_points.csv").exists()
    contingency = read_csv(out1 / "empty_prompt_contingency.csv")
    empty = [row for row in contingency if row["prompt_has_accompaniment"] == "0" and row["n_runs"] != "0"]
    assert len(empty) == 1
    assert empty[0]["condition"] == "single_n1"
    assert empty[0]["n_runs"] == "1"
    sensitivity = [row for row in associations1 if row["sensitivity_status"] != "not_top_prompt_association"]
    assert len(sensitivity) <= 3
    assert all(2 <= int(row["nonempty_only_n_groups"]) <= 3 for row in sensitivity)
    assert all("p0" in row["nonempty_only_dropped_groups"] for row in sensitivity)
    report = (out1 / "REPORT.md").read_text(encoding="utf-8")
    assert "first-pass nonlinear screening" in report
    assert "999 strict nested permutations" in report
    audit_json = json.loads((out1 / "analysis_audit.json").read_text(encoding="utf-8"))
    assert audit_json["unknown_fail_type_tokens"] == {"other": 1}
    assert audit_json["run_condition_audit"]["status"] == "unknown"
    assert audit_json["run_condition_audit"]["trim_leading_rest_flag_present"] is None


def test_run_condition_audit_is_unknown_when_audit_or_script_is_missing(tmp_path):
    missing_input = tmp_path / "missing"
    audit = audit_run_condition_trim_flag(missing_input)
    assert audit["status"] == "unknown"
    assert audit["path"] is None
    assert audit["trim_leading_rest_flag_present"] is None

    input_dir = _write_run_condition_fixture(tmp_path / "no_script", None)
    audit = audit_run_condition_trim_flag(input_dir)
    assert audit["status"] == "unknown"
    assert audit["path"].endswith("technical\\metadata\\run_condition.sh")
    assert audit["trim_leading_rest_flag_present"] is None


def test_run_condition_audit_marks_missing_trim_flag_as_false(tmp_path):
    input_dir = _write_run_condition_fixture(
        tmp_path / "without_flag",
        "#!/usr/bin/env bash\npython scripts/run_lekai_prompt_continuation_offline.py --prompt-length-ticks 32\n",
    )
    audit = audit_run_condition_trim_flag(input_dir)
    assert audit["status"] == "ok"
    assert audit["trim_leading_rest_flag_present"] is False
    assert "without --trim-leading-rest" in audit["interpretation"]


def test_run_condition_audit_marks_trim_flag_as_true(tmp_path):
    input_dir = _write_run_condition_fixture(
        tmp_path / "with_flag",
        "#!/usr/bin/env bash\npython scripts/run_lekai_prompt_continuation_offline.py --trim-leading-rest --prompt-length-ticks 32\n",
    )
    audit = audit_run_condition_trim_flag(input_dir)
    assert audit["status"] == "ok"
    assert audit["trim_leading_rest_flag_present"] is True
    assert "were trimmed before model input" in audit["interpretation"]
