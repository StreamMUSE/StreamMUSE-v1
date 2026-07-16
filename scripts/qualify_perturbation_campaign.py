#!/usr/bin/env python3
"""Create/evaluate the deterministic tempo/tail qualification campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from streammuse.experiments.melody_robustness import (
    build_qualification_schedule,
    default_campaign_config,
    file_sha256,
    read_jsonl,
    validate_campaign_config,
    validate_frozen_qualification,
    validate_listening_selection_manifest,
    validate_qualification_result,
    validate_staged_input_manifest,
    verify_attempt_verdict,
    write_canonical_json,
    write_jsonl,
)
from streammuse.experiments.robustness_metrics import load_midi_roll


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def plan(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    manifest_path = Path(args.input_manifest).resolve()
    manifest = _read(manifest_path)
    validate_staged_input_manifest(
        manifest, manifest_path=manifest_path, verify_files=True
    )
    songs = {str(entry.get("song", entry.get("source_stem"))) for entry in manifest["entries"]}
    if args.dense_song not in songs:
        raise ValueError(f"dense song is not in input manifest: {args.dense_song}")
    if len(args.tail_song) != 2 or len(set(args.tail_song)) != 2:
        raise ValueError("qualification requires exactly two distinct --tail-song values")
    if not set(args.tail_song).issubset(songs):
        raise ValueError(f"tail songs are not in input manifest: {args.tail_song}")
    config = default_campaign_config(
        code_identity=args.code_identity,
        checkpoint_path=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
        input_manifest_path=str(manifest_path), input_manifest_sha256=file_sha256(manifest_path),
        playback_tempo=60, tail_beats=24,
    )
    # Candidate config is executable only with driver --qualification; it is
    # explicitly not the final C5 freeze.
    config["qualification"] = {
        "dense_song": args.dense_song, "tail_songs": args.tail_song,
        "sample_seed": int(config["seeds"]["sample"][0]),
        "perturb_seed": int(config["seeds"]["perturb"][0]),
        "tempo_candidates": [60, 30], "tail_candidates": [8, 16, 24],
        "decision_order": ["determinism", "static_input_gate", "tempo", "tail"],
    }
    validate_campaign_config(config, require_frozen=False)
    config_path = Path(args.output_dir).resolve() / "qualification_config.json"
    write_canonical_json(config_path, config)
    rows = build_qualification_schedule(manifest, config)
    schedule_path = Path(args.output_dir).resolve() / "qualification_manifest.jsonl"
    write_jsonl(schedule_path, rows)
    write_canonical_json(Path(args.output_dir).resolve() / "qualification_plan.json", {
        "config_path": str(config_path), "config_sha256": file_sha256(config_path),
        "schedule_path": str(schedule_path), "schedule_sha256": file_sha256(schedule_path),
        "row_count": len(rows), "rows": [
            {key: row.get(key) for key in ("run_id", "qualification_kind", "qualification_replicate", "song", "condition", "pipeline", "runtime_overrides")}
            for row in rows
        ],
        "execute_command": (
            "python scripts/run_perturbation_matrix.py run --qualification "
            f"--config {config_path} --config-sha256 {file_sha256(config_path)} "
            f"--schedule {schedule_path} --schedule-sha256 {file_sha256(schedule_path)} "
            "--output-root <qualification-output>"
        ),
        "evaluate_command": (
            "python scripts/qualify_perturbation_campaign.py evaluate "
            f"--config {config_path} --config-sha256 {file_sha256(config_path)} "
            f"--schedule {schedule_path} --schedule-sha256 {file_sha256(schedule_path)} "
            "--output-root <qualification-output> --static-summary <conversion-summary> "
            "--output <qualification-result>"
        ),
    })


def _qualification_binding(
    root: Path,
    *,
    config_path: Path,
    schedule_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / "campaign_binding.json"
    expected = {
        "schema_version": "streammuse.melody_robustness.campaign_binding.v1",
        "qualification": True,
        "campaign_config_path": str(config_path.resolve()),
        "campaign_config_sha256": file_sha256(config_path),
        "run_schedule_path": str(schedule_path.resolve()),
        "run_schedule_sha256": file_sha256(schedule_path),
        "input_manifest_path": str(Path(config["input_manifest"]["path"]).resolve()),
        "input_manifest_sha256": str(config["input_manifest"]["sha256"]),
        "checkpoint_path": str(Path(config["checkpoint"]["path"]).resolve()),
        "checkpoint_sha256": str(config["checkpoint"]["sha256"]),
        "code_identity": str(config["code_identity"]),
    }
    if not path.is_file() or _read(path) != expected:
        raise RuntimeError("qualification output binding does not match config/schedule")
    return {**expected, "campaign_binding_sha256": file_sha256(path)}


def _attempt(
    root: Path,
    row: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    attempt, verdict, _ = verify_attempt_verdict(
        root / "runs" / str(row["run_id"]),
        row,
        binding,
    )
    return attempt, verdict


def _single(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} beneath {root}, got {matches}")
    return matches[0]


def _offline_tokens(attempt: Path) -> list[int]:
    payload = _read(_single(attempt, "*_tokens.json"))
    return [int(value) for value in payload["sampled_tokens"]]


def _rt_trace_signature(attempt: Path) -> list[dict[str, Any]]:
    validity = _read(_single(attempt, "validity.json"))
    content = validity.get("content", {})
    if (
        content.get("request_tick_contract_valid") is not True
        or float(content.get("analysis_request_coverage", 0.0)) != 1.0
    ):
        return []
    requests = sorted(
        validity.get("requests", []),
        key=lambda row: int(row["generation_start_tick"]),
    )
    fields = (
        "generation_start_tick",
        "raw_token_digest",
        "input_increment_digest",
        "input_cumulative_digest",
        "part0_roll_digest",
        "part0_token_digest",
        "context_start_tick",
    )
    signature = [{field: row.get(field) for field in fields} for row in requests]
    if not signature:
        return []
    for row in signature:
        for field in fields:
            value = row[field]
            if field == "generation_start_tick":
                if isinstance(value, bool) or not isinstance(value, int):
                    return []
            elif field == "context_start_tick":
                if isinstance(value, bool) or not isinstance(value, int):
                    return []
            elif not isinstance(value, str) or not value.strip():
                return []
    return signature


def _static_gate_errors(static: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema_version": "streammuse.midi_to_npz_summary.v1",
        "status": "ok",
        "expected": 40,
        "converted": 40,
        "skipped": 0,
        "exact_stem_set": True,
        "ticks_per_beat": 4,
        "updated_manifest_sha256": config["input_manifest"]["sha256"],
    }
    for field, wanted in expected.items():
        if static.get(field) != wanted:
            errors.append(
                f"static summary {field}: expected={wanted!r}, got={static.get(field)!r}"
            )
    if static.get("errors") != []:
        errors.append("static summary contains conversion errors")
    manifest = _read(Path(config["input_manifest"]["path"]).resolve())
    entries = {
        str(entry.get("stem", entry.get("input_id"))): entry
        for entry in manifest["entries"]
    }
    results = static.get("results")
    if not isinstance(results, list) or len(results) != 40:
        errors.append("static summary must contain 40 per-input results")
        return errors
    result_stems = [str(row.get("stem")) for row in results if isinstance(row, Mapping)]
    if sorted(result_stems) != sorted(entries):
        errors.append("static summary result stem set differs from frozen input manifest")
    for row in results:
        if not isinstance(row, Mapping):
            errors.append("static summary result is not an object")
            continue
        stem = str(row.get("stem"))
        entry = entries.get(stem)
        roll = row.get("roll_gate")
        if row.get("status") != "converted" or not isinstance(roll, Mapping):
            errors.append(f"{stem}: missing converted roll gate")
            continue
        if int(roll.get("differing_cells", -1)) != 0:
            errors.append(f"{stem}: MIDI/NPZ roll differs")
        if entry is not None:
            if int(roll.get("horizon_ticks", -1)) != int(
                entry["validation_horizon_ticks"]
            ):
                errors.append(f"{stem}: validation horizon mismatch")
            npz = entry.get("npz", entry.get("paths", {}).get("npz", {}))
            if row.get("npz_sha256") != npz.get("sha256"):
                errors.append(f"{stem}: NPZ hash mismatch")
    return errors


def _roll_identity(left: Path, right: Path, end: int) -> bool:
    return load_midi_roll(left, end_tick=end) == load_midi_roll(right, end_tick=end)


def evaluate(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    schedule_path = Path(args.schedule).resolve()
    config = _read(config_path)
    validate_campaign_config(config, require_frozen=False)
    if config.get("status") != "qualification_candidate":
        raise ValueError("qualification evaluation requires a candidate config")
    if file_sha256(config_path) != args.config_sha256:
        raise RuntimeError("qualification candidate config hash mismatch")
    if file_sha256(schedule_path) != args.schedule_sha256:
        raise RuntimeError("qualification schedule hash mismatch")
    manifest_path = Path(config["input_manifest"]["path"]).resolve()
    if file_sha256(manifest_path) != config["input_manifest"]["sha256"]:
        raise RuntimeError("qualification input manifest hash mismatch")
    manifest = _read(manifest_path)
    validate_staged_input_manifest(
        manifest,
        manifest_path=manifest_path,
        verify_files=True,
    )
    rows = read_jsonl(schedule_path)
    if rows != build_qualification_schedule(manifest, config):
        raise RuntimeError("qualification schedule is not the canonical 20-row design")
    root = Path(args.output_root).resolve()
    binding = _qualification_binding(
        root,
        config_path=config_path,
        schedule_path=schedule_path,
        config=config,
    )
    grouped: dict[str, list[tuple[dict[str, Any], Path, dict[str, Any]]]] = {}
    run_evidence: list[dict[str, Any]] = []
    for row in rows:
        attempt, verdict = _attempt(root, row, binding)
        grouped.setdefault(row["qualification_kind"], []).append((row, attempt, verdict))
        immutable_verdict = attempt / "verdict.json"
        run_evidence.append(
            {
                "run_id": row["run_id"],
                "attempt_id": verdict["attempt_id"],
                "verdict": {
                    "path": str(immutable_verdict.resolve()),
                    "sha256": file_sha256(immutable_verdict),
                },
            }
        )
    off_runs = grouped["determinism_offline"]
    rt_runs = grouped["determinism_rt"]
    offline_deterministic = False
    if len(off_runs) == 2:
        offline_tokens = [_offline_tokens(item[1]) for item in off_runs]
        offline_deterministic = bool(
            all(item[2].get("content_valid") for item in off_runs)
            and all(tokens for tokens in offline_tokens)
            and offline_tokens[0] == offline_tokens[1]
            and file_sha256(_single(off_runs[0][1], "*_generated.mid"))
            == file_sha256(_single(off_runs[1][1], "*_generated.mid"))
        )
    rt_deterministic = False
    if len(rt_runs) == 2:
        rt_digests = [_rt_trace_signature(item[1]) for item in rt_runs]
        valid_digests = all(rt_digests)
        rt_end = int(rt_runs[0][0]["input"]["analysis_end_tick"])
        rt_deterministic = bool(
            all(item[2].get("content_valid") for item in rt_runs)
            and valid_digests
            and rt_digests[0] == rt_digests[1]
            and _roll_identity(
                _single(rt_runs[0][1], "theoretical_model.mid"),
                _single(rt_runs[1][1], "theoretical_model.mid"), rt_end,
            )
        )
    static_path = Path(args.static_summary).resolve()
    static = _read(static_path)
    static_errors = _static_gate_errors(static, config)
    static_valid = not static_errors
    tempo_checks = {}
    selected_tempo = None
    for tempo in (60, 30):
        attempts = grouped[f"tempo_{tempo}"]
        passed = len(attempts) == 2 and all(
            bool(verdict.get("content_valid")) and bool(verdict.get("operational_valid"))
            for _row_value, _attempt_value, verdict in attempts
        )
        tempo_checks[str(tempo)] = passed
        if selected_tempo is None and passed:
            selected_tempo = tempo
    tail_checks: dict[str, Any] = {}
    selected_by_song: dict[str, int | None] = {}
    for song in config["qualification"]["tail_songs"]:
        attempts = grouped[f"tail_{selected_tempo}_{song}"] if selected_tempo is not None else []
        if not attempts:
            selected_by_song[song] = None
            tail_checks[song] = {
                "content_valid": False,
                "trace_and_coverage_valid": False,
                "8_eq_16": False,
                "16_eq_24": False,
                "decision": None,
                "reason": "no_selected_tempo_tail_runs",
            }
            continue
        by_tail = {int(row["runtime_overrides"]["tail_beats"]): (row, attempt, verdict) for row, attempt, verdict in attempts}
        valid = all(bool(item[2].get("content_valid")) for item in by_tail.values())
        end = int(next(iter(by_tail.values()))[0]["input"]["analysis_end_tick"])
        paths = {tail: _single(item[1], "theoretical_model.mid") for tail, item in by_tail.items()}
        signatures = {tail: _rt_trace_signature(item[1]) for tail, item in by_tail.items()}
        trace_valid = all(signatures.values())
        eq_8_16 = bool(
            valid and trace_valid and signatures[8] == signatures[16]
            and _roll_identity(paths[8], paths[16], end)
        )
        eq_16_24 = bool(
            valid and trace_valid and signatures[16] == signatures[24]
            and _roll_identity(paths[16], paths[24], end)
        )
        if eq_8_16 and eq_16_24:
            decision = 8
        elif eq_16_24:
            decision = 16
        else:
            decision = None
        selected_by_song[song] = decision
        tail_checks[song] = {
            "content_valid": valid,
            "trace_and_coverage_valid": trace_valid,
            "8_eq_16": eq_8_16,
            "16_eq_24": eq_16_24,
            "decision": decision,
        }
    selected_tail = (
        max(int(value) for value in selected_by_song.values() if value is not None)
        if selected_by_song and all(value is not None for value in selected_by_song.values()) else None
    )
    passed = bool(
        offline_deterministic and rt_deterministic and static_valid
        and selected_tempo is not None and selected_tail is not None
    )
    result = {
        "schema_version": "streammuse.melody_robustness.qualification.v1",
        "passed": passed,
        "candidate_config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "candidate_config_sha256": file_sha256(config_path),
        "qualification_schedule": {
            "path": str(schedule_path),
            "sha256": file_sha256(schedule_path),
        },
        "qualification_campaign_binding": {
            "path": str(root / "campaign_binding.json"),
            "sha256": binding["campaign_binding_sha256"],
        },
        "run_evidence": run_evidence,
        "offline_deterministic": offline_deterministic,
        "rt_deterministic": rt_deterministic,
        "static_input_gate": {
            "valid": static_valid,
            "errors": static_errors,
            "summary_path": str(static_path),
            "sha256": file_sha256(static_path),
        },
        "tempo": {"checks": tempo_checks, "selected": selected_tempo},
        "tail": {"checks": tail_checks, "selected": selected_tail},
        "rule": "8=16=24->8; 16=24->16; otherwise stop_and_investigate",
    }
    validate_qualification_result(
        result,
        candidate_config=config,
        verify_files=True,
        require_passed=False,
    )
    write_canonical_json(Path(args.output), result)
    if not passed:
        raise SystemExit(2)


def freeze(args: argparse.Namespace) -> None:
    candidate_path = Path(args.candidate_config).resolve()
    result_path = Path(args.qualification_result).resolve()
    listening_path = Path(args.listening_manifest).resolve()
    config = _read(candidate_path)
    validate_campaign_config(config, require_frozen=False)
    if config.get("status") != "qualification_candidate":
        raise RuntimeError("C5 freeze requires a qualification candidate config")
    result = _read(result_path)
    if not result.get("passed"):
        raise RuntimeError("cannot freeze a failed qualification")
    if result.get("candidate_config_sha256") != file_sha256(candidate_path):
        raise RuntimeError("qualification result belongs to a different candidate config")
    validate_qualification_result(
        result,
        candidate_config=config,
        verify_files=True,
        require_passed=True,
    )
    listening = _read(listening_path)
    manifest_path = Path(config["input_manifest"]["path"]).resolve()
    validate_listening_selection_manifest(
        listening,
        _read(manifest_path),
        manifest_path=manifest_path,
        verify_files=True,
    )
    config["status"] = "qualified_frozen"
    config["runtime"]["playback_tempo"] = int(result["tempo"]["selected"])
    config["runtime"]["tail_beats"] = int(result["tail"]["selected"])
    config["qualification_candidate"] = {
        "path": str(candidate_path),
        "sha256": file_sha256(candidate_path),
    }
    config["qualification_result"] = {
        "path": str(result_path),
        "sha256": file_sha256(result_path),
    }
    config["listening"]["selection_manifest_path"] = str(listening_path)
    config["listening"]["selection_manifest_sha256"] = file_sha256(listening_path)
    validate_campaign_config(config)
    validate_frozen_qualification(config, verify_files=True)
    write_canonical_json(Path(args.output), config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("plan")
    make.add_argument("--checkpoint", required=True)
    make.add_argument("--input-manifest", required=True)
    make.add_argument("--code-identity", required=True)
    make.add_argument("--dense-song", required=True)
    make.add_argument("--tail-song", action="append", required=True)
    make.add_argument("--output-dir", required=True)
    make.set_defaults(func=plan)
    check = sub.add_parser("evaluate")
    check.add_argument("--config", required=True)
    check.add_argument("--config-sha256", required=True)
    check.add_argument("--schedule", required=True)
    check.add_argument("--schedule-sha256", required=True)
    check.add_argument("--output-root", required=True)
    check.add_argument("--static-summary", required=True)
    check.add_argument("--output", required=True)
    check.set_defaults(func=evaluate)
    final = sub.add_parser("freeze")
    final.add_argument("--candidate-config", required=True)
    final.add_argument("--qualification-result", required=True)
    final.add_argument("--listening-manifest", required=True)
    final.add_argument("--output", required=True)
    final.set_defaults(func=freeze)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
