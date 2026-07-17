from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pytest

from streammuse.experiments.melody_robustness import (
    file_sha256,
    read_jsonl,
    validate_campaign_config,
)
from streammuse.experiments.robustness_metrics import Roll, write_roll_midi


def _plan(qualify, fixture, output_dir: Path) -> tuple[Path, Path, list[dict]]:
    qualify.plan(
        argparse.Namespace(
            checkpoint=str(fixture.checkpoint),
            input_manifest=str(fixture.input_manifest),
            code_identity="b" * 40,
            dense_song="song-1",
            tail_song=["song-2", "song-3"],
            output_dir=str(output_dir),
        )
    )
    config_path = output_dir / "qualification_config.json"
    schedule_path = output_dir / "qualification_manifest.jsonl"
    return config_path, schedule_path, read_jsonl(schedule_path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_test_midi(path: Path, pitch: int = 60) -> None:
    write_roll_midi(
        Roll(
            end_tick=16,
            sustain=frozenset((tick, pitch) for tick in range(16)),
            onsets=frozenset({(0, pitch)}),
        ),
        path,
        bpm=120,
    )


def _valid_rt_trace() -> dict:
    return {
        "content": {
            "request_tick_contract_valid": True,
            "analysis_request_coverage": 1.0,
        },
        "requests": [
            {
                "generation_start_tick": tick,
                "raw_token_digest": f"raw-{tick}",
                "input_increment_digest": f"increment-{tick}",
                "input_cumulative_digest": f"cumulative-{tick}",
                "part0_roll_digest": f"roll-{tick}",
                "part0_token_digest": f"tokens-{tick}",
                "context_start_tick": 0,
            }
            for tick in (4, 8)
        ],
    }


def _static_summary(fixture) -> dict:
    manifest = json.loads(fixture.input_manifest.read_text(encoding="utf-8"))
    results = []
    for entry in manifest["entries"]:
        results.append(
            {
                "stem": entry["stem"],
                "status": "converted",
                "npz_sha256": entry["paths"]["npz"]["sha256"],
                "roll_gate": {
                    "differing_cells": 0,
                    "horizon_ticks": entry["validation_horizon_ticks"],
                },
            }
        )
    return {
        "schema_version": "streammuse.midi_to_npz_summary.v1",
        "status": "ok",
        "expected": 40,
        "converted": 40,
        "skipped": 0,
        "exact_stem_set": True,
        "ticks_per_beat": 4,
        "updated_manifest_sha256": file_sha256(fixture.input_manifest),
        "errors": [],
        "results": results,
    }


def _materialize_successful_qualification(root: Path, rows: list[dict]) -> None:
    for row in rows:
        attempt = root / "runs" / row["run_id"] / "attempt-001"
        attempt.mkdir(parents=True)
        verdict = {
            "run_id": row["run_id"],
            "attempt_id": "attempt-001",
            "content_valid": True,
            "operational_valid": True,
        }
        _write_json(attempt.parent / "latest_verdict.json", verdict)
        kind = row["qualification_kind"]
        if kind == "determinism_offline":
            _write_json(
                attempt / f"{row['run_id']}_tokens.json",
                {"sampled_tokens": [11, 22, 33, 44]},
            )
            _write_test_midi(attempt / f"{row['run_id']}_generated.mid")
        if row["pipeline"] == "rt":
            _write_json(attempt / "validity.json", _valid_rt_trace())
        if kind == "determinism_rt":
            _write_test_midi(attempt / "theoretical_model.mid")
        # Tempo 60 is selected first; equal 8/16/24 symbolic outputs choose the
        # shortest valid tail according to the frozen decision rule.
        if kind.startswith("tail_60_"):
            _write_test_midi(attempt / "theoretical_model.mid")


def test_qualification_plan_is_a_reproducible_complete_20_run_design(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    output_dir = tmp_path / "qualification"

    config_path, schedule_path, rows = _plan(qualify, robustness_fixture, output_dir)
    first_config = config_path.read_bytes()
    first_schedule = schedule_path.read_bytes()
    _plan(qualify, robustness_fixture, output_dir)

    assert config_path.read_bytes() == first_config
    assert schedule_path.read_bytes() == first_schedule
    assert len(rows) == 20
    assert len({row["run_id"] for row in rows}) == 20
    assert [row["schedule_index"] for row in rows] == list(range(20))
    assert Counter(row["qualification_kind"] for row in rows) == {
        "determinism_offline": 2,
        "determinism_rt": 2,
        "tempo_60": 2,
        "tempo_30": 2,
        "tail_60_song-2": 3,
        "tail_60_song-3": 3,
        "tail_30_song-2": 3,
        "tail_30_song-3": 3,
    }
    assert {row["qualification_replicate"] for row in rows[:4]} == {"A", "B"}
    assert all(row["run_id"].startswith("qual-") for row in rows)
    plan = json.loads((output_dir / "qualification_plan.json").read_text())
    assert plan["row_count"] == 20
    assert plan["config_sha256"] == file_sha256(config_path)
    assert plan["schedule_sha256"] == file_sha256(schedule_path)
    assert f"--schedule-sha256 {file_sha256(schedule_path)}" in plan["execute_command"]


def test_evaluate_then_freeze_selects_first_valid_tempo_and_shortest_stable_tail(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    output_dir = tmp_path / "qualification"
    config_path, schedule_path, rows = _plan(qualify, robustness_fixture, output_dir)
    run_root = tmp_path / "qualification-runs"
    _materialize_successful_qualification(run_root, rows)
    static_summary = tmp_path / "static-summary.json"
    _write_json(static_summary, _static_summary(robustness_fixture))
    result_path = tmp_path / "qualification-result.json"

    qualify.evaluate(
        argparse.Namespace(
            config=str(config_path),
            schedule=str(schedule_path),
            output_root=str(run_root),
            static_summary=str(static_summary),
            output=str(result_path),
        )
    )

    result = json.loads(result_path.read_text())
    assert result["passed"]
    assert result["offline_deterministic"]
    assert result["rt_deterministic"]
    assert result["static_input_gate"]["valid"]
    assert result["tempo"] == {"checks": {"30": True, "60": True}, "selected": 60}
    assert result["tail"]["selected"] == 8
    assert {row["decision"] for row in result["tail"]["checks"].values()} == {8}

    frozen_path = tmp_path / "campaign-frozen.json"
    qualify.freeze(
        argparse.Namespace(
            candidate_config=str(config_path),
            qualification_result=str(result_path),
            listening_manifest=str(robustness_fixture.listening_manifest),
            output=str(frozen_path),
        )
    )
    frozen = json.loads(frozen_path.read_text())
    validate_campaign_config(frozen)
    assert frozen["runtime"]["playback_tempo"] == 60
    assert frozen["runtime"]["tail_beats"] == 8
    assert frozen["qualification_result"]["sha256"] == file_sha256(result_path)
    assert frozen["listening"]["selection_manifest_sha256"] == file_sha256(
        robustness_fixture.listening_manifest
    )


@pytest.mark.parametrize(
    ("failure", "result_field"),
    [
        ("static_missing_exact", ("static_input_gate", "valid")),
        ("offline_empty_tokens", ("offline_deterministic",)),
        ("rt_empty_requests", ("rt_deterministic",)),
        ("rt_blank_digest", ("rt_deterministic",)),
    ],
)
def test_evaluate_rejects_implicit_static_success_and_empty_determinism_evidence(
    load_script, robustness_fixture, tmp_path, failure, result_field
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, schedule_path, rows = _plan(
        qualify, robustness_fixture, tmp_path / "qualification"
    )
    run_root = tmp_path / "qualification-runs"
    _materialize_successful_qualification(run_root, rows)
    static = _static_summary(robustness_fixture)
    if failure == "static_missing_exact":
        static.pop("exact_stem_set")
    elif failure == "offline_empty_tokens":
        for row in rows:
            if row["qualification_kind"] == "determinism_offline":
                attempt = run_root / "runs" / row["run_id"] / "attempt-001"
                token_path = next(attempt.glob("*_tokens.json"))
                _write_json(token_path, {"sampled_tokens": []})
    elif failure in {"rt_empty_requests", "rt_blank_digest"}:
        for row in rows:
            if row["qualification_kind"] == "determinism_rt":
                attempt = run_root / "runs" / row["run_id"] / "attempt-001"
                requests = []
                if failure == "rt_blank_digest":
                    requests = [{"generation_start_tick": 4, "raw_token_digest": "  "}]
                _write_json(attempt / "validity.json", {"requests": requests})
    static_path = tmp_path / "static.json"
    _write_json(static_path, static)
    result_path = tmp_path / "rejected-result.json"

    with pytest.raises(SystemExit) as exit_info:
        qualify.evaluate(
            argparse.Namespace(
                config=str(config_path),
                schedule=str(schedule_path),
                output_root=str(run_root),
                static_summary=str(static_path),
                output=str(result_path),
            )
        )
    assert exit_info.value.code == 2
    result = json.loads(result_path.read_text())
    value = result
    for key in result_field:
        value = value[key]
    assert value is False
    assert not result["passed"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda result, _candidate: result.update(passed=False), "failed qualification"),
        (
            lambda result, _candidate: result.update(candidate_config_sha256="0" * 64),
            "different candidate config",
        ),
    ],
)
def test_freeze_rejects_failed_or_cross_config_qualification(
    load_script, robustness_fixture, tmp_path, mutation, message
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, _schedule_path, _rows = _plan(
        qualify, robustness_fixture, tmp_path / "qualification"
    )
    result = {
        "passed": True,
        "candidate_config_sha256": file_sha256(config_path),
        "tempo": {"selected": 60},
        "tail": {"selected": 24},
    }
    mutation(result, config_path)
    result_path = tmp_path / "result.json"
    _write_json(result_path, result)

    with pytest.raises(RuntimeError, match=message):
        qualify.freeze(
            argparse.Namespace(
                candidate_config=str(config_path),
                qualification_result=str(result_path),
                listening_manifest=str(robustness_fixture.listening_manifest),
                output=str(tmp_path / "must-not-freeze.json"),
            )
        )
