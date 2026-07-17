#!/usr/bin/env python3
"""Create/evaluate the deterministic tempo/tail qualification campaign."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from streammuse.experiments.melody_robustness import (
    ATTESTATION_BUNDLE_SCHEMA_VERSION,
    CHECKPOINT_IDENTITY_SCHEMA_VERSION,
    CODE_IDENTITY_SCHEMA_VERSION,
    ENVIRONMENT_IDENTITY_SCHEMA_VERSION,
    QUALIFICATION_DECISION_ORDER,
    QUALIFICATION_DENSE_SONG,
    QUALIFICATION_TAIL_CANDIDATES,
    QUALIFICATION_TAIL_SONGS,
    QUALIFICATION_TEMPO_CANDIDATES,
    LISTENING_SCHEMA_VERSION,
    TRIANGLE_LISTENING_SCHEMA_VERSION,
    build_qualification_artifact_evidence,
    build_qualification_schedule,
    default_campaign_config,
    derive_qualification_decision,
    file_sha256,
    qualification_config_contract,
    qualification_spec_contract,
    read_jsonl,
    validate_campaign_attestation,
    validate_campaign_config,
    validate_frozen_qualification,
    validate_listening_selection_manifest,
    validate_qualification_result,
    validate_staged_input_manifest,
    verify_attempt_verdict,
    write_canonical_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _file_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def _attestation_records(directory: Path) -> dict[str, Any]:
    root = directory.resolve()
    return {
        "schema_version": ATTESTATION_BUNDLE_SCHEMA_VERSION,
        "code_identity": _file_record(root / "code_identity.json"),
        "checkpoint_identity": _file_record(root / "checkpoint_identity.json"),
        "environment": _file_record(root / "environment.json"),
        "qualification_spec": _file_record(root / "qualification_spec.json"),
    }


def _nvidia_smi_identity() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    rows: list[dict[str, Any]] = []
    drivers: set[str] = set()
    for raw in result.stdout.splitlines():
        if not raw.strip():
            continue
        fields = [field.strip() for field in raw.split(",", maxsplit=4)]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi identity row: {raw!r}")
        index, uuid, name, memory_mib, driver = fields
        rows.append(
            {
                "physical_index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(memory_mib),
            }
        )
        drivers.add(driver)
    if not rows or len(drivers) != 1:
        raise RuntimeError("nvidia-smi must report GPUs with one driver version")
    return {"driver_version": next(iter(drivers)), "gpus": rows}


def _environment_identity(code_identity: str) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("formal attestation requires at least one CUDA GPU")
    gpus = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        raw_uuid = getattr(properties, "uuid", None)
        gpus.append(
            {
                "visible_index": index,
                "name": str(properties.name),
                "uuid": None if raw_uuid is None else str(raw_uuid),
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
            }
        )
    uv_lock = ROOT / "uv.lock"
    pyproject = ROOT / "pyproject.toml"
    if not uv_lock.is_file() or not pyproject.is_file():
        raise FileNotFoundError("attestation requires repository uv.lock and pyproject.toml")
    return {
        "schema_version": ENVIRONMENT_IDENTITY_SCHEMA_VERSION,
        "code_identity": code_identity,
        "dependency_files": {
            "uv_lock": _file_record(uv_lock),
            "pyproject": _file_record(pyproject),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": sys.version,
            "version_info": [
                int(sys.version_info.major),
                int(sys.version_info.minor),
                int(sys.version_info.micro),
            ],
            "executable": str(Path(sys.executable).resolve()),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_version": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": True,
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpus,
        "nvidia_smi": _nvidia_smi_identity(),
    }


def attest(args: argparse.Namespace) -> None:
    """Freeze clean code, checkpoint, dependency/runtime, GPU, and qual spec."""

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"checkpoint is missing or empty: {checkpoint}")
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError("formal attestation requires a clean worktree")
    output = Path(args.output_dir).resolve()
    destinations = {
        "code_identity": output / "code_identity.json",
        "checkpoint_identity": output / "checkpoint_identity.json",
        "environment": output / "environment.json",
        "qualification_spec": output / "qualification_spec.json",
    }
    existing = [
        path
        for path in destinations.values()
        if path.exists() or path.with_name(path.name + ".sha256").exists()
    ]
    if existing:
        raise FileExistsError(f"attestation outputs already exist: {existing}")
    code_payload = {
        "schema_version": CODE_IDENTITY_SCHEMA_VERSION,
        "repository_root": str(ROOT.resolve()),
        "git_commit": commit,
        "git_clean": True,
        "git_status_porcelain": "",
    }
    checkpoint_payload = {
        "schema_version": CHECKPOINT_IDENTITY_SCHEMA_VERSION,
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
    }
    write_canonical_json(destinations["code_identity"], code_payload)
    write_canonical_json(destinations["checkpoint_identity"], checkpoint_payload)
    write_canonical_json(
        destinations["environment"], _environment_identity(commit)
    )
    write_canonical_json(
        destinations["qualification_spec"], qualification_spec_contract()
    )
    records = _attestation_records(output)
    validate_campaign_attestation(
        records,
        code_identity=commit,
        checkpoint={"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        qualification=qualification_config_contract(),
        verify_files=True,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "code_identity": commit,
                "attestation": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


def plan(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    manifest_path = Path(args.input_manifest).resolve()
    manifest = _read(manifest_path)
    validate_staged_input_manifest(
        manifest, manifest_path=manifest_path, verify_files=True
    )
    songs = {str(entry.get("song", entry.get("source_stem"))) for entry in manifest["entries"]}
    if args.dense_song != QUALIFICATION_DENSE_SONG:
        raise ValueError(
            f"qualification dense song is fixed to {QUALIFICATION_DENSE_SONG!r}"
        )
    if args.tail_song != list(QUALIFICATION_TAIL_SONGS):
        raise ValueError(
            "qualification tail songs are fixed in order to "
            f"{list(QUALIFICATION_TAIL_SONGS)!r}"
        )
    if args.dense_song not in songs or not set(args.tail_song).issubset(songs):
        raise ValueError("fixed qualification songs are not in the input manifest")
    qualification = qualification_config_contract()
    attestation = _attestation_records(Path(args.attestation_dir))
    checkpoint_record = {
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
    }
    validate_campaign_attestation(
        attestation,
        code_identity=args.code_identity,
        checkpoint=checkpoint_record,
        qualification=qualification,
        verify_files=True,
    )
    config = default_campaign_config(
        code_identity=args.code_identity,
        checkpoint_path=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
        input_manifest_path=str(manifest_path), input_manifest_sha256=file_sha256(manifest_path),
        attestation=attestation,
        playback_tempo=60, tail_beats=24,
    )
    # Candidate config is executable only with driver --qualification; it is
    # explicitly not the final C5 freeze.
    config["qualification"] = qualification
    validate_campaign_config(
        config, require_frozen=False, verify_attestations=True
    )
    config_path = Path(args.output_dir).resolve() / "qualification_config.json"
    write_canonical_json(config_path, config)
    rows = build_qualification_schedule(manifest, config)
    schedule_path = Path(args.output_dir).resolve() / "qualification_manifest.jsonl"
    write_jsonl(schedule_path, rows)
    write_canonical_json(Path(args.output_dir).resolve() / "qualification_plan.json", {
        "config_path": str(config_path), "config_sha256": file_sha256(config_path),
        "schedule_path": str(schedule_path), "schedule_sha256": file_sha256(schedule_path),
        "attestation": config["attestation"],
        "row_count": len(rows), "rows": [
            {key: row.get(key) for key in ("run_id", "qualification_kind", "qualification_replicate", "song", "condition", "pipeline", "runtime_overrides")}
            for row in rows
        ],
        "execution_semantics": (
            "all 20 rows are pre-run; decision_order controls artifact-derived "
            "evaluation only and is not execution short-circuiting"
        ),
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
        expected_attempt_id="attempt-001",
        forbid_other_attempts=True,
    )
    return attempt, verdict


def evaluate(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    schedule_path = Path(args.schedule).resolve()
    config = _read(config_path)
    validate_campaign_config(
        config, require_frozen=False, verify_attestations=True
    )
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
    runs_root = root / "runs"
    if not runs_root.is_dir():
        raise FileNotFoundError(f"qualification runs directory missing: {runs_root}")
    actual_run_directories = {
        path.name for path in runs_root.iterdir() if path.is_dir()
    }
    expected_run_directories = {str(row["run_id"]) for row in rows}
    if actual_run_directories != expected_run_directories:
        raise RuntimeError(
            "qualification run directory set is not the canonical 20-run design: "
            f"missing={sorted(expected_run_directories - actual_run_directories)}, "
            f"extra={sorted(actual_run_directories - expected_run_directories)}"
        )
    attempts: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    run_evidence: list[dict[str, Any]] = []
    for row in rows:
        attempt, verdict = _attempt(root, row, binding)
        if attempt.name != "attempt-001":
            raise RuntimeError("qualification accepts only attempt-001")
        attempts[str(row["run_id"])] = (attempt, verdict)
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
    static_path = Path(args.static_summary).resolve()
    static = _read(static_path)
    artifact_evidence = build_qualification_artifact_evidence(
        config,
        rows,
        static,
        attempts,
    )
    decision = derive_qualification_decision(artifact_evidence)
    static_decision = decision["static_input_gate"]
    result = {
        "schema_version": "streammuse.melody_robustness.qualification.v1",
        "development_only": bool(getattr(args, "development_only", False)),
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
        "static_input_gate": {
            "valid": static_decision["valid"],
            "errors": static_decision["errors"],
            "summary_path": str(static_path),
            "sha256": file_sha256(static_path),
        },
        **{
            key: value
            for key, value in decision.items()
            if key != "static_input_gate"
        },
    }
    validate_qualification_result(
        result,
        candidate_config=config,
        verify_files=True,
        require_passed=False,
    )
    write_canonical_json(Path(args.output), result)
    if not decision["passed"]:
        raise SystemExit(2)


def freeze(args: argparse.Namespace) -> None:
    candidate_path = Path(args.candidate_config).resolve()
    result_path = Path(args.qualification_result).resolve()
    listening_path = Path(args.listening_manifest).resolve()
    config = _read(candidate_path)
    validate_campaign_config(
        config, require_frozen=False, verify_attestations=True
    )
    if config.get("status") != "qualification_candidate":
        raise RuntimeError("C5 freeze requires a qualification candidate config")
    result = _read(result_path)
    if result.get("development_only") is True:
        raise RuntimeError("development-only qualification can never be frozen")
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
    listening_schema = listening.get("schema_version")
    config_listening_schema = config.get("listening", {}).get("schema_version")
    if listening_schema == TRIANGLE_LISTENING_SCHEMA_VERSION:
        if config_listening_schema != TRIANGLE_LISTENING_SCHEMA_VERSION:
            raise RuntimeError("triangle selection does not match candidate listening contract")
        # Lazy import avoids making the historical v1 qualification/read path
        # depend on the v2 listening workflow module.
        from streammuse.experiments.triangle_listening import (
            validate_triangle_renderer_identity,
            validate_triangle_selection_manifest,
        )

        validate_triangle_selection_manifest(
            listening,
            _read(manifest_path),
            manifest_path=manifest_path,
            verify_files=True,
        )
        raw_renderer_path = getattr(args, "renderer_identity", None)
        if not raw_renderer_path:
            raise ValueError("triangle C5 freeze requires --renderer-identity")
        renderer_path = Path(raw_renderer_path).resolve()
        renderer_identity = _read(renderer_path)
        validate_triangle_renderer_identity(renderer_identity, verify_files=True)
    elif listening_schema == LISTENING_SCHEMA_VERSION:
        if config_listening_schema not in {None, LISTENING_SCHEMA_VERSION}:
            raise RuntimeError("legacy selection does not match candidate listening contract")
        validate_listening_selection_manifest(
            listening,
            _read(manifest_path),
            manifest_path=manifest_path,
            verify_files=True,
        )
    else:
        raise ValueError(f"unsupported listening selection schema: {listening_schema!r}")
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
    if listening_schema == TRIANGLE_LISTENING_SCHEMA_VERSION:
        config["listening"]["renderer_identity"] = {
            "path": str(renderer_path),
            "sha256": file_sha256(renderer_path),
        }
    validate_campaign_config(config)
    validate_frozen_qualification(config, verify_files=True)
    write_canonical_json(Path(args.output), config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    attestation = sub.add_parser(
        "attest",
        help="freeze clean code/checkpoint/dependency/CUDA/GPU qualification identity",
    )
    attestation.add_argument("--checkpoint", required=True)
    attestation.add_argument("--output-dir", required=True)
    attestation.set_defaults(func=attest)
    make = sub.add_parser("plan")
    make.add_argument("--checkpoint", required=True)
    make.add_argument("--input-manifest", required=True)
    make.add_argument("--code-identity", required=True)
    make.add_argument("--attestation-dir", required=True)
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
    check.add_argument(
        "--development-only",
        action="store_true",
        help="mark a clean smoke evaluation as permanently ineligible for C5 freeze",
    )
    check.set_defaults(func=evaluate)
    final = sub.add_parser("freeze")
    final.add_argument("--candidate-config", required=True)
    final.add_argument("--qualification-result", required=True)
    final.add_argument("--listening-manifest", required=True)
    final.add_argument(
        "--renderer-identity",
        help="required for triangle v2; hash-pinned FluidSynth/soundfont identity",
    )
    final.add_argument("--output", required=True)
    final.set_defaults(func=freeze)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
