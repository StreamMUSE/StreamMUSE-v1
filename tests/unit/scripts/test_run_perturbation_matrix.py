from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from streammuse.experiments.melody_robustness import (
    file_sha256,
    read_jsonl,
)


def _freeze_and_schedule(driver, fixture, tmp_path: Path) -> tuple[Path, Path, dict, list[dict]]:
    config_path = tmp_path / "campaign.json"
    driver.command_freeze(
        argparse.Namespace(
            checkpoint=str(fixture.checkpoint),
            checkpoint_sha256=file_sha256(fixture.checkpoint),
            input_manifest=str(fixture.input_manifest),
            listening_manifest=str(fixture.listening_manifest),
            code_identity="a" * 40,
            playback_tempo=60,
            tail_beats=24,
            output=str(config_path),
        )
    )
    schedule_path = tmp_path / "run_manifest.jsonl"
    driver.command_schedule(
        argparse.Namespace(
            config=str(config_path),
            config_sha256=file_sha256(config_path),
            output=str(schedule_path),
        )
    )
    return (
        config_path,
        schedule_path,
        json.loads(config_path.read_text(encoding="utf-8")),
        read_jsonl(schedule_path),
    )


def _offline_gate_fixture(driver, tmp_path: Path) -> dict:
    run_id = "mr-offline-gate"
    stem = "condition-20"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for index in range(40):
        path = input_dir / f"condition-{index:02d}.npz"
        if index == 20:
            roll = np.zeros((4, 88, 16), dtype=np.uint8)
            roll[0, 40, 0:4] = 1
            np.savez(
                path,
                metadata=np.array({"num_measures": 1}, dtype=object),
                measure_0=roll,
            )
        else:
            path.write_bytes(b"not-selected")
    npz_path = input_dir / f"{stem}.npz"
    source_midi = tmp_path / f"{stem}.mid"
    source_midi.write_bytes(b"MThd-frozen-perturbed-input")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"frozen-model")
    npz_sha = file_sha256(npz_path)
    source_sha = file_sha256(source_midi)
    checkpoint_sha = file_sha256(checkpoint)
    row = {
        "run_id": run_id,
        "pipeline": "offline",
        "input_stem": stem,
        "sample_seed": 2026071101,
        "input": {
            "paths": {
                "npz": {"path": str(npz_path), "sha256": npz_sha},
                "output_midi": {
                    "path": str(source_midi), "sha256": source_sha,
                },
            }
        },
    }
    config = {
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "input_manifest": {"sha256": "c" * 64},
        "code_identity": "d" * 40,
        "sampling": {
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
        },
        "runtime": {
            "device": "cuda",
            "dtype": "float16",
            "model_condition_bpm": 120,
        },
    }
    binding = {
        "campaign_config_sha256": "a" * 64,
        "run_schedule_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
        "checkpoint_sha256": checkpoint_sha,
        "code_identity": "d" * 40,
        "campaign_binding_sha256": "e" * 64,
    }
    attempt = tmp_path / "attempt-001"
    offline_dir = attempt / "offline"
    offline_dir.mkdir(parents=True)
    names = {
        "config": f"{run_id}_run_config.json",
        "trace": f"{run_id}_tokens.json",
        "generated": f"{run_id}_generated.mid",
        "gt": f"{run_id}_gt.mid",
    }
    generated = offline_dir / names["generated"]
    gt = offline_dir / names["gt"]
    trace_path = offline_dir / names["trace"]
    config_path = offline_dir / names["config"]
    generated.write_bytes(b"MThd-generated")
    gt.write_bytes(b"MThd-ground-truth")
    part0_shape, part0_roll_sha = driver._npz_part0_identity(npz_path)
    part0_tokens_sha = hashlib.sha256(b"part0-beat-tokens").hexdigest()
    trace = {
        "schema_version": 1,
        "run_id": run_id,
        "seed": row["sample_seed"],
        "sampled_tokens": [169, 171],
        "full_interleaved_sequence": [255, 169, 171],
        "part1_beats": [[169], [171]],
        "part0_beat_tokens_sha256": part0_tokens_sha,
    }
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    run_config = {
        "schema_version": 1,
        "run_id": run_id,
        "pipeline": "offline",
        "input": {
            "dataset_index": 20,
            "npz_path": str(npz_path.resolve()),
            "npz_stem": stem,
            "npz_sha256": npz_sha,
            "source_midi": {
                "path": str(source_midi.resolve()), "sha256": source_sha,
            },
            "part0_roundtrip": {
                "valid": True,
                "expected_shape": part0_shape,
                "decoded_shape": part0_shape,
                "differing_cells": 0,
                "expected_roll_sha256": part0_roll_sha,
                "decoded_roll_sha256": part0_roll_sha,
                "part0_beat_tokens_sha256": part0_tokens_sha,
                "bar_token": 255,
                "pad_marker": 173,
                "part0_end_marker": 170,
            },
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()), "sha256": checkpoint_sha,
        },
        "sampling": {
            "seed": row["sample_seed"],
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "gt_prefix_beats": 0,
            "delay_beats": -1,
            "bpm_override": 120,
        },
        "runtime": {"device": "cuda:0", "dtype": "float16"},
        "outputs": {
            "generated_midi": names["generated"],
            "generated_midi_sha256": file_sha256(generated),
            "ground_truth_midi": names["gt"],
            "ground_truth_midi_sha256": file_sha256(gt),
            "token_trace": names["trace"],
            "token_trace_sha256": file_sha256(trace_path),
            "sampled_token_count": 2,
            "full_token_count": 3,
        },
    }
    config_path.write_text(json.dumps(run_config), encoding="utf-8")
    return {
        "attempt": attempt,
        "row": row,
        "config": config,
        "binding": binding,
        "npz": npz_path,
        "source": source_midi,
        "run_config_path": config_path,
        "trace_path": trace_path,
        "generated_path": generated,
    }


def _run_offline_gate(driver, fixture: dict, *, returncode: int = 0) -> dict:
    return driver._offline_postrun_gate(
        fixture["attempt"],
        returncode,
        fixture["row"],
        fixture["config"],
        manifest_dir=fixture["attempt"].parent,
        expected_npz=fixture["npz"],
        expected_source_midi=fixture["source"],
        campaign_binding=fixture["binding"],
    )


def test_freeze_schedule_and_single_rt_dry_run_are_hashed_deterministic_and_offline(
    load_script, robustness_fixture, tmp_path, monkeypatch
):
    driver = load_script("run_perturbation_matrix")
    config_path, schedule_path, config, schedule = _freeze_and_schedule(
        driver, robustness_fixture, tmp_path
    )

    first_config_bytes = config_path.read_bytes()
    first_schedule_bytes = schedule_path.read_bytes()
    _freeze_and_schedule(driver, robustness_fixture, tmp_path)

    assert config_path.read_bytes() == first_config_bytes
    assert schedule_path.read_bytes() == first_schedule_bytes
    assert len(schedule) == 160
    assert len({row["run_id"] for row in schedule}) == 160
    assert [row["pipeline"] for row in schedule[:80]] == ["offline"] * 80
    assert [row["pipeline"] for row in schedule[80:]] == ["rt"] * 80
    assert config_path.with_name(config_path.name + ".sha256").is_file()
    assert schedule_path.with_name(schedule_path.name + ".sha256").is_file()

    # A dry-run must exercise all local hash/path preflight checks, but must never
    # start or contact the inference server.
    monkeypatch.setattr(
        driver,
        "_require_clean_identity",
        lambda expected, *, allow_dirty: expected,
    )
    monkeypatch.setattr(
        driver.requests,
        "post",
        lambda *args, **kwargs: pytest.fail("dry-run contacted reset_session"),
    )
    monkeypatch.setattr(
        driver.requests,
        "get",
        lambda *args, **kwargs: pytest.fail("dry-run contacted runtime_info"),
    )
    rt_row = next(row for row in schedule if row["pipeline"] == "rt")
    output_root = tmp_path / "dry-run-output"
    with pytest.raises(RuntimeError, match="run schedule hash mismatch"):
        driver.command_run(
            argparse.Namespace(
                config=str(config_path),
                config_sha256=file_sha256(config_path),
                schedule=str(schedule_path),
                schedule_sha256="0" * 64,
                output_root=str(tmp_path / "rejected-output"),
                server_url="http://127.0.0.1:65530/generate_accompaniment",
                server_start_timeout=0.01,
                run_id=rt_row["run_id"],
                limit=None,
                dry_run=True,
                allow_dirty=False,
                qualification=False,
            )
        )
    driver.command_run(
        argparse.Namespace(
            config=str(config_path),
            config_sha256=file_sha256(config_path),
            schedule=str(schedule_path),
            schedule_sha256=file_sha256(schedule_path),
            output_root=str(output_root),
            server_url="http://127.0.0.1:65530/generate_accompaniment",
            server_start_timeout=0.01,
            run_id=rt_row["run_id"],
            limit=None,
            dry_run=True,
            allow_dirty=False,
            qualification=False,
        )
    )

    attempt = output_root / "runs" / rt_row["run_id"] / "attempt-001"
    command = json.loads((attempt / "command.json").read_text(encoding="utf-8"))
    assert command["run_id"] == rt_row["run_id"]
    assert command["pipeline"] == "rt"
    assert command["reset_ack"] is None
    assert command["runtime_info"] is None
    assert "--analysis-end-tick" in command["command"]
    assert not (attempt / "process.log").exists()
    assert json.loads((output_root / "last_execution.json").read_text())["results"][0][
        "dry_run"
    ]


def test_offline_postrun_gate_accepts_only_a_fully_bound_hashed_run(
    load_script, tmp_path
):
    driver = load_script("run_perturbation_matrix")
    fixture = _offline_gate_fixture(driver, tmp_path)

    gate = _run_offline_gate(driver, fixture)

    assert gate["content_valid"] is True
    assert gate["operational_valid"] is True
    assert gate["errors"] == []
    assert gate["sampled_token_count"] == 2
    assert set(gate["required_artifacts"]) == {
        "run_config", "token_trace", "generated_midi", "ground_truth_midi"
    }
    assert gate["campaign_binding"] == driver._verdict_binding_fields(
        fixture["binding"]
    )


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    [
        ("wrong_seed", "sampling.seed mismatch"),
        ("empty_tokens", "sampled_tokens must be a non-empty"),
        ("tampered_midi", "outputs.generated_midi_sha256 mismatch"),
        ("wrong_part0", "decoded_roll_sha256 mismatch"),
        ("wrong_npz_identity", "input.npz_stem mismatch"),
        ("wrong_binding", "campaign binding checkpoint_sha256 mismatch"),
        ("nonzero_process", "nonzero status 9"),
    ],
)
def test_offline_postrun_gate_fails_closed_on_identity_content_or_binding_drift(
    load_script, tmp_path, mutation, error_fragment
):
    driver = load_script("run_perturbation_matrix")
    fixture = _offline_gate_fixture(driver, tmp_path)
    returncode = 0
    if mutation in {"wrong_seed", "wrong_part0", "wrong_npz_identity"}:
        payload = json.loads(fixture["run_config_path"].read_text())
        if mutation == "wrong_seed":
            payload["sampling"]["seed"] += 1
        elif mutation == "wrong_part0":
            payload["input"]["part0_roundtrip"]["decoded_roll_sha256"] = "0" * 64
        else:
            payload["input"]["npz_stem"] = "another-condition"
        fixture["run_config_path"].write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "empty_tokens":
        payload = json.loads(fixture["trace_path"].read_text())
        payload["sampled_tokens"] = []
        fixture["trace_path"].write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "tampered_midi":
        fixture["generated_path"].write_bytes(b"tampered-after-run-config")
    elif mutation == "wrong_binding":
        fixture["binding"] = {
            **fixture["binding"], "checkpoint_sha256": "0" * 64,
        }
    elif mutation == "nonzero_process":
        returncode = 9

    gate = _run_offline_gate(driver, fixture, returncode=returncode)

    assert gate["content_valid"] is False
    assert gate["operational_valid"] is False
    assert any(error_fragment in error for error in gate["errors"])


def test_execute_row_writes_invalid_offline_verdict_when_postrun_gate_rejects(
    load_script, tmp_path, monkeypatch
):
    driver = load_script("run_perturbation_matrix")
    source = tmp_path / "input.mid"
    npz = tmp_path / "input.npz"
    acc = tmp_path / "acc.mid"
    for path in (source, npz, acc):
        path.write_bytes(b"fixture")
    row = {
        "run_id": "mr-rejected-offline",
        "pipeline": "offline",
        "sample_seed": 7,
        "input": {},
    }
    config = {
        "runtime": {"device": "cuda", "dtype": "float16", "model_condition_bpm": 120},
        "sampling": {
            "temperature": 0.8, "top_k": 50, "top_p": 0.95,
            "repetition_penalty": 1.2,
        },
        "checkpoint": {"path": str(tmp_path / "model"), "sha256": "c" * 64},
    }
    rejected = {
        "content_valid": False,
        "operational_valid": False,
        "returncode": 0,
        "errors": ["independent offline gate rejected fixture"],
    }
    monkeypatch.setattr(driver, "_preflight_entry", lambda *args: (source, npz, acc))
    monkeypatch.setattr(driver, "_run_process", lambda *args, **kwargs: 0)
    monkeypatch.setattr(driver, "_offline_postrun_gate", lambda *args, **kwargs: rejected)

    verdict = driver._execute_row(
        row,
        config,
        output_root=tmp_path / "campaign",
        manifest_dir=tmp_path,
        generate_url="http://127.0.0.1:8000/generate_accompaniment",
        dry_run=False,
        campaign_binding=None,
    )

    assert verdict["content_valid"] is False
    assert verdict["operational_valid"] is False
    assert verdict["validity"] == rejected
    latest = json.loads(
        (tmp_path / "campaign" / "runs" / row["run_id"] / "latest_verdict.json").read_text()
    )
    assert latest["content_valid"] is False


def test_hash_preflight_and_verified_attempt_cache_fail_closed_after_tampering(
    load_script, tmp_path
):
    driver = load_script("run_perturbation_matrix")
    run_dir = tmp_path / "runs" / "mr-test"
    attempt = run_dir / "attempt-001"
    attempt.mkdir(parents=True)
    artifact = attempt / "generated.mid"
    artifact.write_bytes(b"immutable output")
    record = {
        "path": "generated.mid",
        "size": artifact.stat().st_size,
        "sha256": file_sha256(artifact),
    }
    verdict = {
        "run_id": "mr-test",
        "attempt_id": "attempt-001",
        "content_valid": True,
        "artifact_index": [record],
    }
    (attempt / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    (run_dir / "latest_verdict.json").write_text(json.dumps(verdict), encoding="utf-8")

    assert driver._verified_existing_verdict(run_dir) == verdict
    assert driver._verify_file(artifact, record["sha256"], "generated MIDI") == record[
        "sha256"
    ]

    extra = attempt / "unindexed.bin"
    extra.write_bytes(b"must invalidate exact artifact set")
    assert driver._verified_existing_verdict(run_dir) is None
    extra.unlink()

    artifact.write_bytes(b"tampered output")
    assert driver._verified_existing_verdict(run_dir) is None
    with pytest.raises(RuntimeError, match="hash mismatch"):
        driver._verify_file(artifact, record["sha256"], "generated MIDI")


def test_resume_cache_rejects_empty_index_pointer_mismatch_and_path_escape(
    load_script, tmp_path
):
    driver = load_script("run_perturbation_matrix")
    run_dir = tmp_path / "runs" / "mr-test"
    attempt = run_dir / "attempt-001"
    attempt.mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"must not be indexable through traversal")

    def install(verdict):
        (attempt / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
        (run_dir / "latest_verdict.json").write_text(json.dumps(verdict), encoding="utf-8")

    empty = {
        "run_id": "mr-test",
        "attempt_id": "attempt-001",
        "content_valid": True,
        "artifact_index": [],
    }
    install(empty)
    assert driver._verified_existing_verdict(run_dir) is None

    escaped = {
        **empty,
        "artifact_index": [
            {"path": "../../../outside.bin", "sha256": file_sha256(outside)}
        ],
    }
    install(escaped)
    assert driver._verified_existing_verdict(run_dir) is None

    pointer_mismatch = {**escaped, "run_id": "another-run"}
    install(pointer_mismatch)
    assert driver._verified_existing_verdict(run_dir) is None


def test_campaign_audit_verifies_schedule_hash_and_each_indexed_artifact(
    load_script, robustness_fixture, tmp_path
):
    driver = load_script("run_perturbation_matrix")
    config_path, schedule_path, config, schedule = _freeze_and_schedule(
        driver, robustness_fixture, tmp_path
    )
    output_root = tmp_path / "output"
    binding = driver._bind_campaign_root(
        output_root,
        driver._campaign_binding(
            config_path=config_path,
            schedule_path=schedule_path,
            config=config,
            qualification=False,
        ),
        create=True,
    )
    row = schedule[0]
    run_dir = output_root / "runs" / row["run_id"]
    attempt = run_dir / "attempt-001"
    attempt.mkdir(parents=True)
    artifact = attempt / "generated.mid"
    artifact.write_bytes(b"verified")
    verdict = {
        "run_id": row["run_id"],
        "attempt_id": "attempt-001",
        "pipeline": row["pipeline"],
        "content_valid": True,
        "operational_valid": True,
        "artifact_index": [
            {
                "path": "generated.mid",
                "size": artifact.stat().st_size,
                "sha256": file_sha256(artifact),
            }
        ],
        **driver._verdict_binding_fields(binding),
    }
    (attempt / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    (run_dir / "latest_verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    args = argparse.Namespace(
        config=str(config_path),
        config_sha256=file_sha256(config_path),
        schedule=str(schedule_path),
        schedule_sha256=file_sha256(schedule_path),
        output_root=str(output_root),
        output=str(audit_path),
    )

    # Only one of the real 160 frozen rows is installed, so the detailed audit
    # is valid evidence but intentionally incomplete.
    with pytest.raises(SystemExit) as exit_info:
        driver.command_audit(args)
    assert exit_info.value.code == 2
    audit = json.loads(audit_path.read_text())
    assert audit["content_valid"] == 1
    assert audit["invalid"] == 0
    assert audit["missing"] == 159

    with pytest.raises(RuntimeError, match="run schedule hash mismatch"):
        driver.command_audit(
            argparse.Namespace(**{**vars(args), "schedule_sha256": "0" * 64})
        )

    artifact.write_bytes(b"corrupted after verdict")
    with pytest.raises(SystemExit):
        driver.command_audit(args)
    audit = json.loads(audit_path.read_text())
    assert audit["content_valid"] == 0
    assert audit["invalid"] == 1
    failed = next(item for item in audit["runs"] if item["run_id"] == row["run_id"])
    assert "strict verification" in failed["reason"]

def test_runtime_contract_reports_every_missing_or_mismatched_field(load_script):
    driver = load_script("run_perturbation_matrix")
    checkpoint_path = Path("/frozen/model.safetensors").resolve()
    config = {
        "checkpoint": {"path": str(checkpoint_path), "sha256": "c" * 64},
        "code_identity": "d" * 40,
        "runtime": {
            "model_name": "lekai",
            "inference_mode": "sliding_window",
            "device": "cuda",
            "dtype": "float16",
            "use_cache": True,
            "model_condition_bpm": 120,
            "prompt_context_beats": 128,
            "history_retention_ticks": 512,
            "max_generation_length_frames": None,
            "max_prompt_ticks": None,
            "time_signature_index": 4,
            "ticks_per_beat": 4,
            "generation_interval_ticks": 4,
            "generation_length_frames": 4,
        },
        "sampling": {
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
        },
    }
    valid = {
        "has_real_model": True,
        "checkpoint_sha256": "c" * 64,
        "checkpoint_path": str(checkpoint_path),
        "code_identity": "d" * 40,
        "resolved_device": "cuda",
        "resolved_dtype": "float16",
        "use_cache": True,
        "effective_bpm": 120,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "prompt_context_beats": 128,
        "history_retention_ticks": 512,
        "max_generation_length_frames": None,
        "max_prompt_ticks": None,
        "time_signature_index": 4,
        "ticks_per_beat": 4,
        "boundary_generation_order": "synchronous",
        "fallback_reason": None,
        "session_id": "experiment-session-1",
        "session_epoch": 1,
    }

    assert driver._runtime_errors(valid, config) == []
    post_request = {
        **valid,
        "runtime_model_name": "lekai",
        "runtime_inference_mode": "sliding_window",
        "generation_interval_ticks": 4,
        "generation_length_frames": 4,
        "prompt_length_ticks": None,
    }
    assert driver._runtime_errors(
        post_request, config, require_request_config=True
    ) == []

    invalid = dict(valid)
    invalid.pop("checkpoint_sha256")
    invalid["has_real_model"] = False
    invalid["temperature"] = 0.9
    invalid["fallback_reason"] = "model load failed"
    invalid["session_id"] = ""
    invalid["session_epoch"] = 0
    errors = driver._runtime_errors(invalid, config)

    assert any("missing checkpoint_sha256" in error for error in errors)
    assert any("has_real_model" in error for error in errors)
    assert any("temperature" in error for error in errors)
    assert any("fallback is forbidden" in error for error in errors)
    assert any("no reset experiment session" in error for error in errors)


def test_reset_contract_binds_seed_session_epoch_and_drained_state(load_script):
    driver = load_script("run_perturbation_matrix")
    expected_seed = 2026071101
    ack = {
        "success": True,
        "effective_seed": expected_seed,
        "session_id": "session-7",
        "session_epoch": 7,
        "pending_boundary_generations": 0,
    }
    runtime = {
        "sample_seed": expected_seed,
        "session_id": "session-7",
        "session_epoch": 7,
        "accepting_requests": True,
        "pending_boundary_generations": 0,
    }
    assert driver._reset_contract_errors(ack, runtime, expected_seed) == []

    invalid_ack = {**ack, "effective_seed": expected_seed + 1, "pending_boundary_generations": 1}
    invalid_runtime = {
        **runtime,
        "sample_seed": expected_seed + 2,
        "session_id": "stale-session",
        "accepting_requests": False,
        "pending_boundary_generations": 1,
    }
    errors = driver._reset_contract_errors(invalid_ack, invalid_runtime, expected_seed)
    assert any("effective_seed mismatch" in error for error in errors)
    assert any("retained pending" in error for error in errors)
    assert any("session_id does not match" in error for error in errors)
    assert any("sample_seed mismatch" in error for error in errors)
    assert any("not accepting" in error for error in errors)
    assert any("reports pending" in error for error in errors)


@pytest.mark.parametrize(
    ("returncode", "payload", "expected"),
    [
        (0, {"content_valid": True, "operational_valid": True}, (True, True)),
        (1, {"content_valid": True, "operational_valid": True}, (False, False)),
        (0, {"content_valid": True, "operational_valid": False}, (True, False)),
    ],
)
def test_rt_validity_never_turns_process_failure_into_success(
    load_script, tmp_path, returncode, payload, expected
):
    driver = load_script("run_perturbation_matrix")
    attempt = tmp_path / f"attempt-{returncode}-{expected[1]}"
    attempt.mkdir()
    (attempt / "validity.json").write_text(json.dumps(payload), encoding="utf-8")

    content_valid, operational_valid, _ = driver._rt_validity(attempt, returncode)

    assert (content_valid, operational_valid) == expected
