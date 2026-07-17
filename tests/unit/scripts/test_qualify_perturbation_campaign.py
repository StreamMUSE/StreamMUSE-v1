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
    validate_qualification_result,
    write_canonical_json,
    write_jsonl,
)
from streammuse.experiments.robustness_metrics import Roll, write_roll_midi


def _plan(qualify, fixture, output_dir: Path) -> tuple[Path, Path, list[dict]]:
    qualify.plan(
        argparse.Namespace(
            checkpoint=str(fixture.checkpoint),
            input_manifest=str(fixture.input_manifest),
            code_identity="a" * 40,
            attestation_dir=str(fixture.attestation_dir),
            dense_song="2",
            tail_song=["2", "4"],
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


def test_attest_writes_four_hash_pinned_records_from_clean_head(
    load_script, robustness_fixture, tmp_path, monkeypatch
):
    qualify = load_script("qualify_perturbation_campaign")
    commit = "a" * 40
    environment = json.loads(
        (robustness_fixture.attestation_dir / "environment.json").read_text()
    )
    monkeypatch.setattr(
        qualify,
        "_git",
        lambda *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        qualify,
        "_environment_identity",
        lambda code_identity: {**environment, "code_identity": code_identity},
    )
    output = tmp_path / "fresh-attestation"

    qualify.attest(
        argparse.Namespace(
            checkpoint=str(robustness_fixture.checkpoint),
            output_dir=str(output),
        )
    )

    assert {
        path.name for path in output.glob("*.json")
    } == {
        "code_identity.json",
        "checkpoint_identity.json",
        "environment.json",
        "qualification_spec.json",
    }
    for path in output.glob("*.json"):
        assert path.with_name(path.name + ".sha256").is_file()
    assert json.loads((output / "code_identity.json").read_text())["git_commit"] == commit
    assert json.loads((output / "checkpoint_identity.json").read_text())[
        "sha256"
    ] == file_sha256(robustness_fixture.checkpoint)


def test_attest_rejects_dirty_head_before_writing(
    load_script, robustness_fixture, tmp_path, monkeypatch
):
    qualify = load_script("qualify_perturbation_campaign")
    monkeypatch.setattr(
        qualify,
        "_git",
        lambda *args: "a" * 40
        if args == ("rev-parse", "HEAD")
        else " M src/dirty.py",
    )
    output = tmp_path / "must-not-attest"

    with pytest.raises(RuntimeError, match="clean worktree"):
        qualify.attest(
            argparse.Namespace(
                checkpoint=str(robustness_fixture.checkpoint),
                output_dir=str(output),
            )
        )
    assert not output.exists()


def test_plan_rejects_checkpoint_or_code_identity_mismatched_from_attestation(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    other_checkpoint = tmp_path / "other.safetensors"
    other_checkpoint.write_bytes(b"other checkpoint")

    for checkpoint, code_identity, message in (
        (other_checkpoint, "a" * 40, "attested checkpoint path"),
        (robustness_fixture.checkpoint, "b" * 40, "attested git commit"),
    ):
        with pytest.raises(ValueError, match=message):
            qualify.plan(
                argparse.Namespace(
                    checkpoint=str(checkpoint),
                    input_manifest=str(robustness_fixture.input_manifest),
                    code_identity=code_identity,
                    attestation_dir=str(robustness_fixture.attestation_dir),
                    dense_song="2",
                    tail_song=["2", "4"],
                    output_dir=str(tmp_path / f"must-not-plan-{code_identity[0]}"),
                )
            )


def test_candidate_attestation_rejects_wrong_hash_and_unbound_recomputed_payloads(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, _schedule_path, _rows = _plan(
        qualify, robustness_fixture, tmp_path / "qualification"
    )
    original = json.loads(config_path.read_text())

    wrong_hash = json.loads(config_path.read_text())
    wrong_hash["attestation"]["environment"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="attestation.environment hash mismatch"):
        validate_campaign_config(
            wrong_hash,
            require_frozen=False,
            verify_attestations=True,
        )

    code_path = Path(original["attestation"]["code_identity"]["path"])
    code = json.loads(code_path.read_text())
    code["git_commit"] = "b" * 40
    write_canonical_json(code_path, code)
    recomputed_code = json.loads(config_path.read_text())
    recomputed_code["attestation"]["code_identity"]["sha256"] = file_sha256(
        code_path
    )
    with pytest.raises(ValueError, match="attested git commit"):
        validate_campaign_config(
            recomputed_code,
            require_frozen=False,
            verify_attestations=True,
        )


def test_candidate_attestation_rejects_tamper_even_when_record_hash_is_recomputed(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, _schedule_path, _rows = _plan(
        qualify, robustness_fixture, tmp_path / "qualification"
    )
    config = json.loads(config_path.read_text())
    spec_path = Path(config["attestation"]["qualification_spec"]["path"])
    spec = json.loads(spec_path.read_text())
    spec["qualification"]["dense_song"] = "1"
    write_canonical_json(spec_path, spec)
    config["attestation"]["qualification_spec"]["sha256"] = file_sha256(spec_path)

    with pytest.raises(ValueError, match="qualification attestation spec"):
        validate_campaign_config(
            config,
            require_frozen=False,
            verify_attestations=True,
        )


def _materialize_successful_qualification(
    root: Path,
    rows: list[dict],
    *,
    config_path: Path,
    schedule_path: Path,
) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    binding_path = root / "campaign_binding.json"
    binding = {
        "schema_version": "streammuse.melody_robustness.campaign_binding.v1",
        "qualification": True,
        "campaign_config_path": str(config_path.resolve()),
        "campaign_config_sha256": file_sha256(config_path),
        "run_schedule_path": str(schedule_path.resolve()),
        "run_schedule_sha256": file_sha256(schedule_path),
        "input_manifest_path": str(Path(config["input_manifest"]["path"]).resolve()),
        "input_manifest_sha256": config["input_manifest"]["sha256"],
        "checkpoint_path": str(Path(config["checkpoint"]["path"]).resolve()),
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "code_identity": config["code_identity"],
    }
    _write_json(binding_path, binding)
    binding_fields = {
        key: binding[key]
        for key in (
            "campaign_config_sha256",
            "run_schedule_sha256",
            "input_manifest_sha256",
            "checkpoint_sha256",
            "code_identity",
        )
    }
    binding_fields["campaign_binding_sha256"] = file_sha256(binding_path)
    for row in rows:
        attempt = root / "runs" / row["run_id"] / "attempt-001"
        attempt.mkdir(parents=True)
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
        if kind.startswith("tail_"):
            _write_test_midi(attempt / "theoretical_model.mid")
        artifact_index = [
            {
                "path": str(path.relative_to(attempt)),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(attempt.rglob("*"))
            if path.is_file()
        ]
        verdict = {
            "schema_version": "streammuse.melody_robustness.verdict.v1",
            "run_id": row["run_id"],
            "attempt_id": "attempt-001",
            "pipeline": row["pipeline"],
            "content_valid": True,
            "operational_valid": True,
            "validity": {},
            "artifact_index": artifact_index,
            **binding_fields,
        }
        _write_json(attempt / "verdict.json", verdict)
        _write_json(attempt.parent / "latest_verdict.json", verdict)


def _refresh_attempt_verdict(attempt: Path) -> None:
    verdict_path = attempt / "verdict.json"
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    verdict["artifact_index"] = [
        {
            "path": str(path.relative_to(attempt)),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(attempt.rglob("*"))
        if path.is_file() and path.name != "verdict.json"
    ]
    _write_json(verdict_path, verdict)
    _write_json(attempt.parent / "latest_verdict.json", verdict)


def _evaluate_successful_fixture(
    qualify,
    fixture,
    tmp_path: Path,
    *,
    development_only: bool = False,
) -> tuple[Path, Path, list[dict], Path, Path]:
    config_path, schedule_path, rows = _plan(
        qualify, fixture, tmp_path / "qualification"
    )
    run_root = tmp_path / "qualification-runs"
    _materialize_successful_qualification(
        run_root,
        rows,
        config_path=config_path,
        schedule_path=schedule_path,
    )
    static_path = tmp_path / "static-summary.json"
    _write_json(static_path, _static_summary(fixture))
    result_path = tmp_path / "qualification-result.json"
    qualify.evaluate(
        argparse.Namespace(
            config=str(config_path),
            config_sha256=file_sha256(config_path),
            schedule=str(schedule_path),
            schedule_sha256=file_sha256(schedule_path),
            output_root=str(run_root),
            static_summary=str(static_path),
            output=str(result_path),
            development_only=development_only,
        )
    )
    return config_path, schedule_path, rows, run_root, result_path


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
        "tail_60_2": 3,
        "tail_60_4": 3,
        "tail_30_2": 3,
        "tail_30_4": 3,
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
    _materialize_successful_qualification(
        run_root,
        rows,
        config_path=config_path,
        schedule_path=schedule_path,
    )
    static_summary = tmp_path / "static-summary.json"
    _write_json(static_summary, _static_summary(robustness_fixture))
    result_path = tmp_path / "qualification-result.json"

    qualify.evaluate(
        argparse.Namespace(
            config=str(config_path),
            config_sha256=file_sha256(config_path),
            schedule=str(schedule_path),
            schedule_sha256=file_sha256(schedule_path),
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
            renderer_identity=str(robustness_fixture.renderer_identity),
            output=str(frozen_path),
        )
    )
    frozen = json.loads(frozen_path.read_text())
    validate_campaign_config(frozen)
    assert frozen["runtime"]["playback_tempo"] == 60
    assert frozen["runtime"]["tail_beats"] == 8
    assert frozen["qualification_result"]["sha256"] == file_sha256(result_path)
    candidate = json.loads(config_path.read_text())
    assert frozen["attestation"] == candidate["attestation"]
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
    _materialize_successful_qualification(
        run_root,
        rows,
        config_path=config_path,
        schedule_path=schedule_path,
    )
    static = _static_summary(robustness_fixture)
    if failure == "static_missing_exact":
        static.pop("exact_stem_set")
    elif failure == "offline_empty_tokens":
        for row in rows:
            if row["qualification_kind"] == "determinism_offline":
                attempt = run_root / "runs" / row["run_id"] / "attempt-001"
                token_path = next(attempt.glob("*_tokens.json"))
                _write_json(token_path, {"sampled_tokens": []})
                _refresh_attempt_verdict(attempt)
    elif failure in {"rt_empty_requests", "rt_blank_digest"}:
        for row in rows:
            if row["qualification_kind"] == "determinism_rt":
                attempt = run_root / "runs" / row["run_id"] / "attempt-001"
                requests = []
                if failure == "rt_blank_digest":
                    requests = [{"generation_start_tick": 4, "raw_token_digest": "  "}]
                _write_json(attempt / "validity.json", {"requests": requests})
                _refresh_attempt_verdict(attempt)
    static_path = tmp_path / "static.json"
    _write_json(static_path, static)
    result_path = tmp_path / "rejected-result.json"

    with pytest.raises(SystemExit) as exit_info:
        qualify.evaluate(
            argparse.Namespace(
                config=str(config_path),
                config_sha256=file_sha256(config_path),
                schedule=str(schedule_path),
                schedule_sha256=file_sha256(schedule_path),
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
                renderer_identity=str(robustness_fixture.renderer_identity),
                output=str(tmp_path / "must-not-freeze.json"),
            )
        )


def test_evaluate_rejects_reordered_rehashed_qualification_schedule(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, schedule_path, rows = _plan(
        qualify, robustness_fixture, tmp_path / "qualification"
    )
    write_jsonl(schedule_path, list(reversed(rows)))

    with pytest.raises(RuntimeError, match="canonical 20-row design"):
        qualify.evaluate(
            argparse.Namespace(
                config=str(config_path),
                config_sha256=file_sha256(config_path),
                schedule=str(schedule_path),
                schedule_sha256=file_sha256(schedule_path),
                output_root=str(tmp_path / "unused-runs"),
                static_summary=str(tmp_path / "unused-static.json"),
                output=str(tmp_path / "must-not-exist.json"),
            )
        )


def test_evaluate_rejects_cross_campaign_qualification_binding(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, schedule_path, rows = _plan(
        qualify, robustness_fixture, tmp_path / "qualification"
    )
    run_root = tmp_path / "qualification-runs"
    _materialize_successful_qualification(
        run_root,
        rows,
        config_path=config_path,
        schedule_path=schedule_path,
    )
    binding_path = run_root / "campaign_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["code_identity"] = "0" * 40
    _write_json(binding_path, binding)

    with pytest.raises(RuntimeError, match="qualification output binding"):
        qualify.evaluate(
            argparse.Namespace(
                config=str(config_path),
                config_sha256=file_sha256(config_path),
                schedule=str(schedule_path),
                schedule_sha256=file_sha256(schedule_path),
                output_root=str(run_root),
                static_summary=str(tmp_path / "unused-static.json"),
                output=str(tmp_path / "must-not-exist.json"),
            )
        )


def test_freeze_rejects_handwritten_minimal_passed_result(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, _schedule_path, _rows = _plan(
        qualify, robustness_fixture, tmp_path / "qualification"
    )
    result_path = tmp_path / "handwritten-result.json"
    _write_json(
        result_path,
        {
            "passed": True,
            "candidate_config_sha256": file_sha256(config_path),
            "tempo": {"selected": 60},
            "tail": {"selected": 24},
        },
    )

    with pytest.raises(ValueError, match="schema_version"):
        qualify.freeze(
            argparse.Namespace(
                candidate_config=str(config_path),
                qualification_result=str(result_path),
                listening_manifest=str(robustness_fixture.listening_manifest),
                renderer_identity=str(robustness_fixture.renderer_identity),
                output=str(tmp_path / "must-not-freeze.json"),
            )
        )


@pytest.mark.parametrize(
    ("dense_song", "tail_songs", "message"),
    [
        ("1", ["2", "4"], "dense song is fixed"),
        ("2", ["4", "2"], "tail songs are fixed in order"),
        ("2", ["2", "3"], "tail songs are fixed in order"),
    ],
)
def test_plan_rejects_noncanonical_qualification_song_selectors(
    load_script,
    robustness_fixture,
    tmp_path,
    dense_song,
    tail_songs,
    message,
):
    qualify = load_script("qualify_perturbation_campaign")

    with pytest.raises(ValueError, match=message):
        qualify.plan(
            argparse.Namespace(
                checkpoint=str(robustness_fixture.checkpoint),
                input_manifest=str(robustness_fixture.input_manifest),
                code_identity="a" * 40,
                attestation_dir=str(robustness_fixture.attestation_dir),
                dense_song=dense_song,
                tail_song=tail_songs,
                output_dir=str(tmp_path / "must-not-plan"),
            )
        )


def test_evaluate_rejects_any_retry_directory_even_when_latest_is_attempt_001(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, schedule_path, rows = _plan(
        qualify, robustness_fixture, tmp_path / "qualification"
    )
    run_root = tmp_path / "qualification-runs"
    _materialize_successful_qualification(
        run_root,
        rows,
        config_path=config_path,
        schedule_path=schedule_path,
    )
    first_run = run_root / "runs" / rows[0]["run_id"]
    (first_run / "attempt-002").mkdir()
    static_path = tmp_path / "static.json"
    _write_json(static_path, _static_summary(robustness_fixture))

    with pytest.raises(ValueError, match="attempt directory set"):
        qualify.evaluate(
            argparse.Namespace(
                config=str(config_path),
                config_sha256=file_sha256(config_path),
                schedule=str(schedule_path),
                schedule_sha256=file_sha256(schedule_path),
                output_root=str(run_root),
                static_summary=str(static_path),
                output=str(tmp_path / "must-not-exist.json"),
            )
        )


def test_evaluate_rejects_extra_qualification_run_directory(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, schedule_path, rows = _plan(
        qualify, robustness_fixture, tmp_path / "qualification"
    )
    run_root = tmp_path / "qualification-runs"
    _materialize_successful_qualification(
        run_root,
        rows,
        config_path=config_path,
        schedule_path=schedule_path,
    )
    (run_root / "runs" / "forged-extra-run").mkdir()
    static_path = tmp_path / "static.json"
    _write_json(static_path, _static_summary(robustness_fixture))

    with pytest.raises(RuntimeError, match="canonical 20-run design"):
        qualify.evaluate(
            argparse.Namespace(
                config=str(config_path),
                config_sha256=file_sha256(config_path),
                schedule=str(schedule_path),
                schedule_sha256=file_sha256(schedule_path),
                output_root=str(run_root),
                static_summary=str(static_path),
                output=str(tmp_path / "must-not-exist.json"),
            )
        )


@pytest.mark.parametrize(
    "forgery",
    ["passed", "offline", "rt", "static", "tempo", "tail"],
)
def test_validator_rederives_every_decision_field_from_immutable_artifacts(
    load_script, robustness_fixture, tmp_path, forgery
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, _schedule_path, _rows, _run_root, result_path = (
        _evaluate_successful_fixture(qualify, robustness_fixture, tmp_path)
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if forgery == "passed":
        result["passed"] = False
    elif forgery == "offline":
        result["offline_deterministic"] = False
        result["passed"] = False
        result["failure_reasons"] = ["offline_determinism_failed"]
    elif forgery == "rt":
        result["rt_deterministic"] = False
        result["passed"] = False
        result["failure_reasons"] = ["rt_determinism_failed"]
    elif forgery == "static":
        result["static_input_gate"]["valid"] = False
        result["static_input_gate"]["errors"] = ["forged static error"]
        result["passed"] = False
        result["failure_reasons"] = ["static_input_gate_failed"]
    elif forgery == "tempo":
        result["tempo"] = {
            "checks": {"30": True, "60": False},
            "selected": 30,
        }
    elif forgery == "tail":
        for check in result["tail"]["checks"].values():
            check["8_eq_16"] = False
            check["decision"] = 16
            check["reason"] = "16_24_converged"
        result["tail"]["selected"] = 16

    candidate = json.loads(config_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="qualification"):
        validate_qualification_result(
            result,
            candidate_config=candidate,
            verify_files=True,
            require_passed=False,
        )


def test_freeze_rejects_internally_consistent_tail_decision_forgery(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, _schedule_path, _rows, _run_root, result_path = (
        _evaluate_successful_fixture(qualify, robustness_fixture, tmp_path)
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for check in result["tail"]["checks"].values():
        check["8_eq_16"] = False
        check["decision"] = 16
        check["reason"] = "16_24_converged"
    result["tail"]["selected"] = 16
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="artifact-derived qualification tail"):
        qualify.freeze(
            argparse.Namespace(
                candidate_config=str(config_path),
                qualification_result=str(result_path),
                listening_manifest=str(robustness_fixture.listening_manifest),
                renderer_identity=str(robustness_fixture.renderer_identity),
                output=str(tmp_path / "must-not-freeze.json"),
            )
        )


def test_development_only_result_can_be_inspected_but_never_frozen(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, _schedule_path, _rows, _run_root, result_path = (
        _evaluate_successful_fixture(
            qualify,
            robustness_fixture,
            tmp_path,
            development_only=True,
        )
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["development_only"] is True

    with pytest.raises(RuntimeError, match="development-only qualification"):
        qualify.freeze(
            argparse.Namespace(
                candidate_config=str(config_path),
                qualification_result=str(result_path),
                listening_manifest=str(robustness_fixture.listening_manifest),
                renderer_identity=str(robustness_fixture.renderer_identity),
                output=str(tmp_path / "must-not-freeze.json"),
            )
        )


def test_tail_nonconvergence_stops_instead_of_falling_back_to_24(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, schedule_path, rows = _plan(
        qualify, robustness_fixture, tmp_path / "qualification"
    )
    run_root = tmp_path / "qualification-runs"
    _materialize_successful_qualification(
        run_root,
        rows,
        config_path=config_path,
        schedule_path=schedule_path,
    )
    divergent = next(
        row
        for row in rows
        if row["qualification_kind"] == "tail_60_2"
        and row["runtime_overrides"]["tail_beats"] == 24
    )
    divergent_attempt = (
        run_root / "runs" / divergent["run_id"] / "attempt-001"
    )
    _write_test_midi(divergent_attempt / "theoretical_model.mid", pitch=61)
    _refresh_attempt_verdict(divergent_attempt)
    static_path = tmp_path / "static.json"
    _write_json(static_path, _static_summary(robustness_fixture))
    result_path = tmp_path / "failed-result.json"

    with pytest.raises(SystemExit) as exit_info:
        qualify.evaluate(
            argparse.Namespace(
                config=str(config_path),
                config_sha256=file_sha256(config_path),
                schedule=str(schedule_path),
                schedule_sha256=file_sha256(schedule_path),
                output_root=str(run_root),
                static_summary=str(static_path),
                output=str(result_path),
            )
        )
    assert exit_info.value.code == 2
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["tail"]["checks"]["2"]["decision"] is None
    assert result["tail"]["checks"]["2"]["reason"] == "tail_not_converged"
    assert result["tail"]["selected"] is None
    assert 24 not in {
        check["decision"] for check in result["tail"]["checks"].values()
    }
    assert result["passed"] is False


def test_validator_rejects_artifact_drift_after_evaluation(
    load_script, robustness_fixture, tmp_path
):
    qualify = load_script("qualify_perturbation_campaign")
    config_path, _schedule_path, rows, run_root, result_path = (
        _evaluate_successful_fixture(qualify, robustness_fixture, tmp_path)
    )
    offline = next(
        row for row in rows if row["qualification_kind"] == "determinism_offline"
    )
    token_path = next(
        (run_root / "runs" / offline["run_id"] / "attempt-001").glob(
            "*_tokens.json"
        )
    )
    _write_json(token_path, {"sampled_tokens": [999]})
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidate = json.loads(config_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="indexed artifact size/hash mismatch"):
        validate_qualification_result(
            result,
            candidate_config=candidate,
            verify_files=True,
            require_passed=True,
        )
