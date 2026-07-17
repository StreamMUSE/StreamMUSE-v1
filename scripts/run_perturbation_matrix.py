#!/usr/bin/env python3
"""Schedule, execute, and audit the 160-run melody robustness campaign.

The driver intentionally fails closed.  Formal execution requires a clean
worktree, immutable hashed inputs/config/schedule, a dedicated real-model
server whose runtime contract matches the campaign, and a listening selection
manifest frozen before any model output is generated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import numpy as np
import requests

from streammuse.experiments.melody_robustness import (
    OFFLINE_REQUIRED_ARTIFACT_SCHEMA,
    OFFLINE_REQUIRED_ATTEMPT_ARTIFACTS,
    RT_REQUIRED_ARTIFACT_SCHEMA,
    RT_REQUIRED_ATTEMPT_ARTIFACTS,
    build_qualification_schedule,
    build_run_schedule,
    canonical_sha256,
    file_sha256,
    read_jsonl,
    validate_campaign_config,
    validate_frozen_qualification,
    validate_listening_selection_manifest,
    validate_staged_input_manifest,
    verify_attempt_verdict,
    write_canonical_json,
    write_jsonl,
)
from streammuse.infrastructure.inference.lekai_http_backend import (
    decode_part1_token_trace,
)


ROOT = Path(__file__).resolve().parents[1]

OFFLINE_ARTIFACT_CONTRACT_SCHEMA = OFFLINE_REQUIRED_ARTIFACT_SCHEMA
RT_ARTIFACT_CONTRACT_SCHEMA = RT_REQUIRED_ARTIFACT_SCHEMA
OFFLINE_REQUIRED_ARTIFACTS = OFFLINE_REQUIRED_ATTEMPT_ARTIFACTS
RT_REQUIRED_ARTIFACTS = RT_REQUIRED_ATTEMPT_ARTIFACTS


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _require_clean_identity(expected: str, *, allow_dirty: bool) -> str:
    commit = _git("rev-parse", "HEAD")
    if commit != expected:
        raise RuntimeError(f"code identity mismatch: config={expected}, HEAD={commit}")
    dirty = _git("status", "--porcelain")
    if dirty and not allow_dirty:
        raise RuntimeError("formal execution requires a clean worktree")
    return commit


def _verify_file(path: Path, expected_sha: str | None, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = file_sha256(path)
    if expected_sha and actual != expected_sha:
        raise RuntimeError(f"{label} hash mismatch: expected={expected_sha}, actual={actual}")
    return actual


def _path_record(entry: Mapping[str, Any], *names: str) -> tuple[str, str | None]:
    containers: list[Mapping[str, Any]] = [entry]
    for container_name in ("paths", "artifacts", "files"):
        nested = entry.get(container_name)
        if isinstance(nested, Mapping):
            containers.append(nested)
    for container in containers:
        for name in names:
            value = container.get(name)
            if isinstance(value, str):
                return value, container.get(f"{name}_sha256")  # type: ignore[return-value]
            if isinstance(value, Mapping) and value.get("path"):
                return str(value["path"]), (
                    str(value["sha256"]) if value.get("sha256") else None
                )
    raise KeyError(f"input manifest entry lacks any of {names}")


def _resolve_artifact(
    entry: Mapping[str, Any], manifest_dir: Path, *names: str
) -> tuple[Path, str | None]:
    raw, digest = _path_record(entry, *names)
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve(), digest


def _server_base(generate_url: str) -> str:
    return generate_url.rsplit("/", 1)[0]


def _server_port(generate_url: str) -> int:
    parsed = urlparse(generate_url)
    if parsed.port:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def _require_dedicated_loopback(generate_url: str) -> None:
    parsed = urlparse(generate_url)
    if parsed.scheme != "http":
        raise ValueError("experiment server URL must use local HTTP")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("experiment server URL must resolve to loopback")
    if parsed.path.rstrip("/") != "/generate_accompaniment":
        raise ValueError("experiment server URL must target /generate_accompaniment")


def _wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.ok:
                return
        except Exception as exc:  # server is expected to be unavailable initially
            last_error = exc
        time.sleep(0.25)
    raise TimeoutError(f"dedicated server did not become healthy: {last_error}")


def _runtime_errors(
    info: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    require_request_config: bool = False,
) -> list[str]:
    runtime = config["runtime"]
    sampling = config["sampling"]
    expected = {
        "has_real_model": True,
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "checkpoint_path": str(Path(config["checkpoint"]["path"]).resolve()),
        "code_identity": config["code_identity"],
        "resolved_device": runtime["device"],
        "resolved_dtype": runtime["dtype"],
        "use_cache": runtime["use_cache"],
        "effective_bpm": runtime["model_condition_bpm"],
        "temperature": sampling["temperature"],
        "top_k": sampling["top_k"],
        "top_p": sampling["top_p"],
        "repetition_penalty": sampling["repetition_penalty"],
        "prompt_context_beats": runtime["prompt_context_beats"],
        "history_retention_ticks": runtime["history_retention_ticks"],
        "max_generation_length_frames": runtime["max_generation_length_frames"],
        "max_prompt_ticks": runtime["max_prompt_ticks"],
        "time_signature_index": runtime["time_signature_index"],
        "ticks_per_beat": runtime["ticks_per_beat"],
        "boundary_generation_order": "synchronous",
    }
    if require_request_config:
        expected.update(
            {
                "runtime_model_name": runtime["model_name"],
                "runtime_inference_mode": runtime["inference_mode"],
                "generation_interval_ticks": runtime[
                    "generation_interval_ticks"
                ],
                "generation_length_frames": runtime["generation_length_frames"],
                "prompt_length_ticks": None,
            }
        )
    errors: list[str] = []
    for key, wanted in expected.items():
        if key not in info:
            errors.append(f"runtime_info missing {key}")
        elif info[key] != wanted:
            errors.append(f"runtime_info.{key}: expected {wanted!r}, got {info[key]!r}")
    if info.get("fallback_reason"):
        errors.append(f"runtime fallback is forbidden: {info['fallback_reason']}")
    if not info.get("session_id") or int(info.get("session_epoch", 0)) <= 0:
        errors.append("runtime has no reset experiment session")
    return errors


def _reset_contract_errors(
    ack: Mapping[str, Any], info: Mapping[str, Any], expected_seed: int
) -> list[str]:
    errors: list[str] = []
    if ack.get("success") is not True:
        errors.append("reset ack did not report success=true")
    if int(ack.get("effective_seed", -1)) != int(expected_seed):
        errors.append(
            f"reset effective_seed mismatch: expected {expected_seed}, got {ack.get('effective_seed')}"
        )
    if not ack.get("session_id") or int(ack.get("session_epoch", 0)) <= 0:
        errors.append("reset ack lacks a valid session_id/session_epoch")
    if int(ack.get("pending_boundary_generations", -1)) != 0:
        errors.append("reset ack retained pending boundary generation")
    for key in ("session_id", "session_epoch"):
        if info.get(key) != ack.get(key):
            errors.append(
                f"runtime_info.{key} does not match reset ack: {info.get(key)!r} != {ack.get(key)!r}"
            )
    if int(info.get("sample_seed", -1)) != int(expected_seed):
        errors.append(
            f"runtime_info.sample_seed mismatch: expected {expected_seed}, got {info.get('sample_seed')}"
        )
    if not info.get("accepting_requests"):
        errors.append("runtime is not accepting requests after reset")
    if int(info.get("pending_boundary_generations", -1)) != 0:
        errors.append("runtime reports pending boundary generation after reset")
    return errors


def _hash_index(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return rows


def _next_attempt(run_dir: Path) -> tuple[str, Path]:
    existing = sorted(path for path in run_dir.glob("attempt-*") if path.is_dir())
    attempt_id = f"attempt-{len(existing) + 1:03d}"
    attempt_dir = run_dir / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return attempt_id, attempt_dir


_BINDING_VERDICT_FIELDS = (
    "campaign_config_sha256",
    "run_schedule_sha256",
    "input_manifest_sha256",
    "checkpoint_sha256",
    "code_identity",
    "campaign_binding_sha256",
)


def _campaign_binding(
    *,
    config_path: Path,
    schedule_path: Path,
    config: Mapping[str, Any],
    qualification: bool,
) -> dict[str, Any]:
    binding = {
        "schema_version": "streammuse.melody_robustness.campaign_binding.v1",
        "qualification": bool(qualification),
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
    if not qualification:
        binding["qualification_result_sha256"] = str(
            config["qualification_result"]["sha256"]
        )
    return binding


def _bind_campaign_root(
    output_root: Path, binding: Mapping[str, Any], *, create: bool
) -> dict[str, Any]:
    path = output_root / "campaign_binding.json"
    expected = dict(binding)
    if path.is_file():
        if _read_json(path) != expected:
            raise RuntimeError(
                "output root is bound to a different campaign config/schedule"
            )
    elif create:
        runs_dir = output_root / "runs"
        if runs_dir.exists() and any(runs_dir.iterdir()):
            raise RuntimeError("refusing to bind an output root that already has runs")
        write_canonical_json(path, expected)
    else:
        raise FileNotFoundError(f"campaign binding missing: {path}")
    digest = file_sha256(path)
    return {**expected, "campaign_binding_sha256": digest}


def _verdict_binding_fields(binding: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "campaign_config_sha256": binding["campaign_config_sha256"],
        "run_schedule_sha256": binding["run_schedule_sha256"],
        "input_manifest_sha256": binding["input_manifest_sha256"],
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "code_identity": binding["code_identity"],
        "campaign_binding_sha256": binding["campaign_binding_sha256"],
    }
    if "qualification_result_sha256" in binding:
        fields["qualification_result_sha256"] = binding[
            "qualification_result_sha256"
        ]
    return fields


def _verified_existing_verdict(
    run_dir: Path, expected_binding: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    pointer = run_dir / "latest_verdict.json"
    if not pointer.is_file():
        return None
    verdict = _read_json(pointer)
    if not verdict.get("content_valid"):
        return None
    attempt_id = str(verdict.get("attempt_id", ""))
    if re.fullmatch(r"attempt-[0-9]{3}", attempt_id) is None:
        return None
    if verdict.get("run_id") != run_dir.name:
        return None
    if expected_binding is not None:
        expected_fields = _verdict_binding_fields(expected_binding)
        if any(verdict.get(key) != value for key, value in expected_fields.items()):
            return None
    attempt = (run_dir / attempt_id).resolve()
    try:
        if not attempt.is_dir() or not attempt.is_relative_to(run_dir.resolve()):
            return None
    except ValueError:
        return None
    immutable_path = attempt / "verdict.json"
    if not immutable_path.is_file() or _read_json(immutable_path) != verdict:
        return None
    index = verdict.get("artifact_index")
    if not isinstance(index, list) or not index:
        return None
    seen: set[str] = set()
    indexed_records: dict[str, Mapping[str, Any]] = {}
    for record in index:
        if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
            return None
        relative = str(record["path"])
        if relative in seen:
            return None
        seen.add(relative)
        indexed_records[relative] = record
        path = (attempt / relative).resolve()
        try:
            if not path.is_relative_to(attempt):
                return None
        except ValueError:
            return None
        size = record.get("size")
        if (
            not path.is_file()
            or isinstance(size, bool)
            or not isinstance(size, int)
            or path.stat().st_size != size
            or file_sha256(path) != record.get("sha256")
        ):
            return None
    actual = {
        str(path.relative_to(attempt))
        for path in attempt.rglob("*")
        if path.is_file()
        and path.name not in {"verdict.json", "verdict.json.sha256"}
    }
    if actual != seen:
        return None
    # Formal attempts must carry a pipeline-specific, independently generated
    # required-artifact gate.  Qualification attempts use their own evidence
    # validator and intentionally do not claim this formal contract.
    if expected_binding is not None and "qualification_result_sha256" in expected_binding:
        if verdict.get("schema_version") != "streammuse.melody_robustness.verdict.v1":
            return None
        pipeline = verdict.get("pipeline")
        if pipeline == "rt":
            expected_schema = RT_ARTIFACT_CONTRACT_SCHEMA
            expected_labels = set(RT_REQUIRED_ARTIFACTS)
            expected_gate = "rt_artifact_gate.json"
        elif pipeline == "offline":
            expected_schema = OFFLINE_ARTIFACT_CONTRACT_SCHEMA
            expected_labels = set(OFFLINE_REQUIRED_ARTIFACTS)
            expected_gate = "offline_post_run_gate.json"
        else:
            return None
        contract = verdict.get("required_artifact_contract")
        if not isinstance(contract, Mapping):
            return None
        if (
            contract.get("schema_version") != expected_schema
            or contract.get("enforced") is not True
            or contract.get("gate_path") != expected_gate
            or contract.get("required_labels") != sorted(expected_labels)
        ):
            return None
        gate_path = attempt / expected_gate
        if expected_gate not in indexed_records or not gate_path.is_file():
            return None
        try:
            gate = _read_json(gate_path)
        except Exception:
            return None
        required = gate.get("required_artifacts")
        if (
            gate.get("schema_version") != expected_schema
            or gate.get("enforced", True) is not True
            or gate.get("content_valid") is not True
            or not isinstance(required, Mapping)
            or set(required) != expected_labels
        ):
            return None
        for label in expected_labels:
            record = required.get(label)
            if not isinstance(record, Mapping):
                return None
            relative = record.get("path")
            if not isinstance(relative, str):
                return None
            basename = Path(relative).name
            expected_name = (
                RT_REQUIRED_ARTIFACTS[label]
                if pipeline == "rt"
                else OFFLINE_REQUIRED_ARTIFACTS[label]
            )
            if (
                (expected_name.startswith("_") and not basename.endswith(expected_name))
                or (not expected_name.startswith("_") and basename != expected_name)
            ):
                return None
            indexed = indexed_records.get(relative)
            if indexed is None or dict(indexed) != dict(record):
                return None
        if pipeline == "rt":
            semantic = gate.get("semantic_evidence")
            if not isinstance(semantic, Mapping) or any(
                semantic.get(field) is not True
                for field in (
                    "raw_token_trace_complete",
                    "token_decode_reconciliation_complete",
                    "output_event_digest_complete",
                    "trace_inference_equal",
                    "theoretical_semantic_equal",
                    "combined_schedule_consistent",
                    "combined_semantic_equal",
                )
            ):
                return None
        try:
            _attempt, shared_verdict, _indexed = verify_attempt_verdict(
                run_dir,
                {"run_id": run_dir.name, "pipeline": pipeline},
                expected_binding,
                require_content_valid=True,
            )
        except Exception:
            return None
        if shared_verdict != verdict:
            return None
    return verdict


_ATTEMPT_ID_PATTERN = re.compile(r"attempt-([0-9]{3})")


def _matched_retry_block_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["pipeline"]),
        str(row["song"]),
        int(row["sample_seed"]),
    )


def _formal_retry_blocks(
    rows: Iterable[Mapping[str, Any]],
) -> list[tuple[tuple[str, str, int], list[Mapping[str, Any]]]]:
    """Group the canonical formal schedule into its immutable 8-run blocks."""
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_matched_retry_block_key(row), []).append(row)
    blocks = list(grouped.items())
    for key, block_rows in blocks:
        selectors = {
            (
                str(row.get("condition")),
                (
                    None
                    if row.get("perturb_seed") is None
                    else int(row["perturb_seed"])
                ),
            )
            for row in block_rows
        }
        if len(block_rows) != 8 or len(selectors) != 8:
            raise ValueError(
                "formal matched retry block must contain exactly 8 distinct "
                f"condition/perturb-seed runs: block={key!r}, "
                f"rows={len(block_rows)}, selectors={len(selectors)}"
            )
    return blocks


def _attempt_directory_ids(run_dir: Path) -> list[str]:
    """Return a contiguous immutable attempt sequence, rejecting lookalikes."""
    if not run_dir.exists():
        return []
    if not run_dir.is_dir():
        raise ValueError(f"run path is not a directory: {run_dir}")
    attempt_ids: list[str] = []
    for path in run_dir.iterdir():
        if not path.name.startswith("attempt-"):
            continue
        match = _ATTEMPT_ID_PATTERN.fullmatch(path.name)
        if match is None or not path.is_dir():
            raise ValueError(f"invalid attempt entry: {path}")
        attempt_ids.append(path.name)
    attempt_ids.sort()
    expected = [f"attempt-{index:03d}" for index in range(1, len(attempt_ids) + 1)]
    if attempt_ids != expected:
        raise ValueError(
            f"attempt IDs are not contiguous from attempt-001: {run_dir}: "
            f"actual={attempt_ids!r}, expected={expected!r}"
        )
    return attempt_ids


def _verified_attempt_history(
    *,
    row: Mapping[str, Any],
    output_root: Path,
    campaign_binding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Verify every immutable attempt plus the pointer to the highest one."""
    run_dir = output_root / "runs" / str(row["run_id"])
    attempt_ids = _attempt_directory_ids(run_dir)
    pointer = run_dir / "latest_verdict.json"
    if not attempt_ids:
        if pointer.exists():
            raise ValueError(f"latest verdict exists without an immutable attempt: {pointer}")
        return {}
    history: dict[str, dict[str, Any]] = {}
    for attempt_id in attempt_ids:
        _attempt, verdict, _indexed = verify_attempt_verdict(
            run_dir,
            row,
            campaign_binding,
            expected_attempt_id=attempt_id,
            require_content_valid=None,
            require_latest_pointer=False,
        )
        history[attempt_id] = verdict
    highest = attempt_ids[-1]
    _attempt, latest, _indexed = verify_attempt_verdict(
        run_dir,
        row,
        campaign_binding,
        expected_attempt_id=highest,
        require_content_valid=None,
        require_latest_pointer=True,
    )
    if latest != history[highest]:
        raise ValueError(f"latest verdict is not the highest immutable attempt: {run_dir}")
    return history


def _inspect_formal_retry_block(
    *,
    key: tuple[str, str, int],
    block_rows: list[Mapping[str, Any]],
    output_root: Path,
    campaign_binding: Mapping[str, Any],
    allow_partial_latest: bool,
    maximum_attempts: int | None = None,
) -> dict[str, Any]:
    """Verify block history and permit only an interruptible highest attempt."""
    if len(block_rows) != 8:
        raise ValueError(f"matched retry block must contain 8 rows: {key!r}")
    histories = {
        str(row["run_id"]): _verified_attempt_history(
            row=row,
            output_root=output_root,
            campaign_binding=campaign_binding,
        )
        for row in block_rows
    }
    id_sets = {run_id: set(history) for run_id, history in histories.items()}
    union = sorted(set().union(*id_sets.values())) if id_sets else []
    if maximum_attempts is not None and len(union) > maximum_attempts:
        raise ValueError(
            f"block {key!r} exceeds frozen retry policy: "
            f"attempts={union!r}, maximum={maximum_attempts}"
        )
    for attempt_id in union[:-1]:
        missing = sorted(run_id for run_id, ids in id_sets.items() if attempt_id not in ids)
        if missing:
            raise ValueError(
                f"block {key!r} has a non-highest partial {attempt_id}: missing={missing!r}"
            )
    partial_latest = False
    missing_latest: list[str] = []
    if union:
        highest = union[-1]
        missing_latest = sorted(
            run_id for run_id, ids in id_sets.items() if highest not in ids
        )
        partial_latest = bool(missing_latest)
        if partial_latest and not allow_partial_latest:
            raise ValueError(
                f"block {key!r} attempt ID sets are not aligned at {highest}: "
                f"missing={missing_latest!r}"
            )
    else:
        highest = None
    return {
        "key": key,
        "rows": block_rows,
        "histories": histories,
        "attempt_ids": union,
        "highest_attempt_id": highest,
        "partial_latest": partial_latest,
        "missing_latest_run_ids": missing_latest,
        "per_run_attempt_ids": {
            run_id: sorted(ids) for run_id, ids in id_sets.items()
        },
    }


def _preaudit_formal_retry_blocks(
    *,
    blocks: list[tuple[tuple[str, str, int], list[Mapping[str, Any]]]],
    output_root: Path,
    campaign_binding: Mapping[str, Any],
    allow_partial_latest: bool,
    maximum_attempts: int | None = None,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Audit all blocks before the driver creates any new attempt directory."""
    return {
        key: _inspect_formal_retry_block(
            key=key,
            block_rows=block_rows,
            output_root=output_root,
            campaign_binding=campaign_binding,
            allow_partial_latest=allow_partial_latest,
            maximum_attempts=maximum_attempts,
        )
        for key, block_rows in blocks
    }


def _formal_retry_action(
    state: Mapping[str, Any], *, maximum_attempts: int
) -> dict[str, Any] | None:
    """Choose one aligned attempt action; ``None`` means block is terminal."""
    if maximum_attempts < 1:
        raise ValueError("maximum_attempts must include at least attempt-001")
    rows = list(state["rows"])
    attempt_ids = list(state["attempt_ids"])
    highest = state.get("highest_attempt_id")
    histories = state["histories"]
    if state.get("partial_latest"):
        assert isinstance(highest, str)
        missing = set(state["missing_latest_run_ids"])
        return {
            "attempt_id": highest,
            "rows": [row for row in rows if str(row["run_id"]) in missing],
            "resume_partial": True,
        }
    if not attempt_ids:
        return {
            "attempt_id": "attempt-001",
            "rows": rows,
            "resume_partial": False,
        }
    assert isinstance(highest, str)
    latest = [histories[str(row["run_id"])][highest] for row in rows]
    if all(verdict.get("content_valid") is True for verdict in latest):
        return None
    if len(attempt_ids) >= maximum_attempts:
        return None
    return {
        "attempt_id": f"attempt-{len(attempt_ids) + 1:03d}",
        "rows": rows,
        "resume_partial": False,
    }


def _selected_formal_block_keys(
    *,
    rows: list[Mapping[str, Any]],
    run_id: str | None,
    limit: int | None,
) -> set[tuple[str, str, int]]:
    """Expand row-oriented CLI selectors to complete matched retry blocks."""
    if run_id is not None:
        matches = [row for row in rows if row["run_id"] == run_id]
        if len(matches) != 1:
            raise KeyError(f"unknown run_id: {run_id}")
        selected = matches
    elif limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        selected = rows[:limit]
    else:
        selected = rows
    return {_matched_retry_block_key(row) for row in selected}


def _validate_frozen_listening_selection(
    config: Mapping[str, Any], input_manifest: Mapping[str, Any]
) -> Path:
    raw = config.get("listening", {}).get("selection_manifest_path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("frozen config requires listening selection_manifest_path")
    selection_path = Path(raw).resolve()
    _verify_file(
        selection_path,
        config["listening"].get("selection_manifest_sha256"),
        "listening selection manifest",
    )
    selection = _read_json(selection_path)
    if (
        selection.get("schema_version")
        == "streammuse.melody_robustness.listening_triangle_selection.v2"
    ):
        from streammuse.experiments.triangle_listening import (
            validate_triangle_selection_manifest,
        )

        validate_triangle_selection_manifest(
            selection,
            input_manifest,
            manifest_path=Path(config["input_manifest"]["path"]).resolve(),
            verify_files=True,
        )
    else:
        validate_listening_selection_manifest(
            selection,
            input_manifest,
            manifest_path=Path(config["input_manifest"]["path"]).resolve(),
            verify_files=True,
        )
    return selection_path


def _run_process(command: list[str], *, cwd: Path, env: Mapping[str, str], output: Path) -> int:
    with output.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(result.returncode)


def _find_single(root: Path, name: str) -> Path | None:
    matches = list(root.rglob(name))
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous {name} beneath {root}: {matches}")
    return matches[0] if matches else None


def _artifact_record(path: Path, attempt_dir: Path) -> dict[str, Any]:
    resolved = path.resolve()
    attempt = attempt_dir.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(attempt):
        raise ValueError(f"artifact is missing or escapes attempt: {path}")
    return {
        "path": str(resolved.relative_to(attempt)),
        "size": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _read_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {line_number} is not an object")
        rows.append(value)
    return rows


def _event_fingerprint(
    row: Mapping[str, Any], *, tick_field: str, include_key: bool
) -> tuple[Any, ...]:
    event_type = row.get("type")
    if event_type not in {"note_on", "note_off"}:
        raise ValueError(f"invalid model event type: {event_type!r}")
    tick = row.get(tick_field)
    if isinstance(tick, bool) or not isinstance(tick, int):
        raise ValueError(f"model event lacks integer {tick_field}")
    pitch = row.get("pitch")
    velocity = row.get("velocity")
    if isinstance(pitch, bool) or not isinstance(pitch, int) or not -1 <= pitch <= 127:
        raise ValueError("model event pitch is invalid")
    if (
        isinstance(velocity, bool)
        or not isinstance(velocity, int)
        or not 0 <= velocity <= 127
    ):
        raise ValueError("model event velocity is invalid")
    base: tuple[Any, ...] = (event_type, pitch, tick, velocity)
    if not include_key:
        return base
    channel = row.get("channel", 0)
    program = row.get("program", 0)
    if (
        isinstance(channel, bool)
        or not isinstance(channel, int)
        or isinstance(program, bool)
        or not isinstance(program, int)
    ):
        raise ValueError("model event channel/program is invalid")
    return (*base, channel, program)


def _canonical_notes_from_events(
    rows: Iterable[Mapping[str, Any]], *, tick_field: str, final_tick: int | None = None
) -> list[tuple[int, int, int, int]]:
    """Mirror MidiFileOutputSink note reconstruction in integer tick space."""
    events = [
        _event_fingerprint(row, tick_field=tick_field, include_key=False)
        for row in rows
        if int(row.get("pitch", -1)) != -1
    ]
    events.sort(
        key=lambda event: (
            int(event[2]),
            0 if event[0] == "note_off" else 1,
            int(event[1]),
        )
    )
    active: dict[int, tuple[int, int]] = {}
    notes: list[tuple[int, int, int, int]] = []
    max_tick = int(final_tick or 0)
    for event_type, raw_pitch, raw_tick, raw_velocity in events:
        pitch = int(raw_pitch)
        tick = int(raw_tick)
        velocity = int(raw_velocity)
        max_tick = max(max_tick, tick)
        if event_type == "note_on" and velocity > 0:
            previous = active.pop(pitch, None)
            if previous is not None and tick > previous[0]:
                notes.append((previous[0], tick, pitch, previous[1]))
            active[pitch] = (tick, velocity)
        elif event_type == "note_off":
            previous = active.pop(pitch, None)
            if previous is not None and tick > previous[0]:
                notes.append((previous[0], tick, pitch, previous[1]))
    for pitch, (start, velocity) in active.items():
        if max_tick > start:
            notes.append((start, max_tick, pitch, velocity))
    return sorted(notes)


def _midi_track_notes(
    path: Path,
    *,
    track_name: str,
    bpm: float,
    ticks_per_beat: int,
    allowed_other_tracks: set[str],
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    import pretty_midi

    errors: list[str] = []
    midi = pretty_midi.PrettyMIDI(str(path))
    matching = [instrument for instrument in midi.instruments if instrument.name == track_name]
    if len(matching) > 1:
        errors.append(f"MIDI contains multiple {track_name!r} tracks")
    for instrument in midi.instruments:
        if instrument.name != track_name and instrument.notes:
            if instrument.name not in allowed_other_tracks:
                errors.append(f"MIDI contains unexpected non-empty track {instrument.name!r}")
    seconds_per_tick = (60.0 / float(bpm)) / int(ticks_per_beat)
    notes: list[tuple[int, int, int, int]] = []
    for instrument in matching:
        for note in instrument.notes:
            raw_start = float(note.start) / seconds_per_tick
            raw_end = float(note.end) / seconds_per_tick
            start = int(round(raw_start))
            end = int(round(raw_end))
            if abs(raw_start - start) > 1e-4 or abs(raw_end - end) > 1e-4:
                errors.append("MIDI note is not aligned to the frozen logical tick grid")
            notes.append((start, end, int(note.pitch), int(note.velocity)))
    return sorted(notes), errors


def _resolve_formal_rt_required_artifacts(
    attempt_dir: Path,
) -> tuple[dict[str, Path], list[str]]:
    errors: list[str] = []
    paths: dict[str, Path] = {}
    root_labels = {
        "command",
        "process_log",
        "rt_input_gate",
        "post_run_runtime_info",
    }
    for label, name in RT_REQUIRED_ARTIFACTS.items():
        try:
            path = attempt_dir / name if label in root_labels else _find_single(attempt_dir, name)
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        if path is None or not path.is_file():
            errors.append(f"missing required RT artifact {label}: {name}")
            continue
        paths[label] = path.resolve()
    session_labels = set(RT_REQUIRED_ARTIFACTS) - root_labels
    session_dirs = {paths[label].parent for label in session_labels if label in paths}
    if len(session_dirs) != 1:
        errors.append(
            "required RT session artifacts do not resolve to one session directory"
        )
    return paths, errors


def _rt_postrun_artifact_gate(
    attempt_dir: Path,
    *,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    returncode: int,
    base_content_valid: bool,
    base_operational_valid: bool,
    enforce_formal_contract: bool,
) -> dict[str, Any]:
    """Independently validate formal RT evidence and generated MIDI semantics."""
    if not enforce_formal_contract:
        return {
            "schema_version": RT_ARTIFACT_CONTRACT_SCHEMA,
            "enforced": False,
            "content_valid": bool(base_content_valid),
            "operational_valid": bool(base_operational_valid),
            "required_artifacts": {},
            "semantic_evidence": {},
            "errors": [],
        }

    errors: list[str] = []
    paths, resolution_errors = _resolve_formal_rt_required_artifacts(attempt_dir)
    errors.extend(resolution_errors)
    required_records: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        try:
            required_records[label] = _artifact_record(path, attempt_dir)
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    if int(returncode) != 0:
        errors.append(f"RT process returned nonzero status {returncode}")
    if not base_content_valid:
        errors.append("RT service validity gate rejected content")

    command: dict[str, Any] = {}
    input_gate: dict[str, Any] = {}
    runtime_gate: dict[str, Any] = {}
    validity: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    session_config: dict[str, Any] = {}
    for label, destination in (
        ("command", command),
        ("rt_input_gate", input_gate),
        ("post_run_runtime_info", runtime_gate),
        ("validity", validity),
        ("theoretical_model_summary", summary),
        ("session_config", session_config),
    ):
        path = paths.get(label)
        if path is None:
            continue
        try:
            destination.update(_read_json(path))
        except Exception as exc:
            errors.append(f"cannot parse {label}: {type(exc).__name__}: {exc}")

    expected_attempt_id = attempt_dir.name
    for label, actual, expected in (
        ("command.run_id", command.get("run_id"), row.get("run_id")),
        ("command.attempt_id", command.get("attempt_id"), expected_attempt_id),
        ("command.pipeline", command.get("pipeline"), "rt"),
        ("input gate valid", input_gate.get("valid"), True),
        ("post runtime gate valid", runtime_gate.get("valid"), True),
        ("session output_type", session_config.get("output_type"), "session"),
        ("session artifact tier", session_config.get("session_artifact_tier"), "debug"),
    ):
        if actual != expected or type(actual) is not type(expected):
            errors.append(f"{label} mismatch: expected={expected!r}, actual={actual!r}")
    reset_ack = command.get("reset_ack")
    runtime_info = command.get("runtime_info")
    if not isinstance(reset_ack, Mapping) or not isinstance(runtime_info, Mapping):
        errors.append("command artifact lacks reset/runtime evidence")
    content = validity.get("content")
    if not isinstance(content, Mapping) or content.get("valid") is not True:
        errors.append("validity.json does not contain content.valid=true")

    trace_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    actual_rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    for label, destination in (
        ("model_schedule_trace", trace_rows),
        ("actual_event_trace", actual_rows),
        ("request_lifecycle", lifecycle_rows),
    ):
        path = paths.get(label)
        if path is None:
            continue
        try:
            destination.extend(_read_jsonl_objects(path, label))
        except Exception as exc:
            errors.append(f"cannot parse {label}: {type(exc).__name__}: {exc}")
    inference_path = paths.get("full_inference_trace")
    if inference_path is not None:
        try:
            raw_inferences = json.loads(inference_path.read_text(encoding="utf-8"))
            if not isinstance(raw_inferences, list) or any(
                not isinstance(item, dict) for item in raw_inferences
            ):
                raise ValueError("inferences.json must be a list of objects")
            inference_rows.extend(raw_inferences)
        except Exception as exc:
            errors.append(
                f"cannot parse full_inference_trace: {type(exc).__name__}: {exc}"
            )

    inference_events: list[Mapping[str, Any]] = []
    inference_request_ids: list[str] = []
    raw_token_trace_complete = True
    token_decode_reconciliation_complete = True
    output_event_digest_complete = True
    validity_requests = validity.get("requests")
    validity_by_request: dict[str, Mapping[str, Any]] = {}
    if isinstance(validity_requests, list):
        for request in validity_requests:
            if isinstance(request, Mapping) and isinstance(request.get("request_id"), str):
                validity_by_request[str(request["request_id"])] = request
    else:
        errors.append("validity.json requests is not a list")
        raw_token_trace_complete = False
        output_event_digest_complete = False
    for index, inference in enumerate(inference_rows):
        request_data = inference.get("request_data")
        response_data = inference.get("response_data")
        if not isinstance(request_data, Mapping) or not isinstance(response_data, Mapping):
            errors.append(f"inference {index} lacks request/response objects")
            raw_token_trace_complete = False
            output_event_digest_complete = False
            continue
        request_id = request_data.get("lifecycle_request_id")
        if not isinstance(request_id, str) or not request_id:
            errors.append(f"inference {index} lacks lifecycle_request_id")
            raw_token_trace_complete = False
            output_event_digest_complete = False
            request_id = f"<invalid-inference-{index}>"
        inference_request_ids.append(request_id)
        if request_data.get("log_detail") != "full" or not isinstance(
            request_data.get("melody_notes"), list
        ):
            errors.append(f"inference {index} is not a full-detail request trace")
            raw_token_trace_complete = False
        accompaniment = response_data.get("accompaniment")
        metadata = response_data.get("response_metadata")
        if not isinstance(accompaniment, list) or any(
            not isinstance(event, Mapping) for event in accompaniment
        ):
            errors.append(f"inference {index} lacks full accompaniment events")
            accompaniment = []
            raw_token_trace_complete = False
            output_event_digest_complete = False
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("raw_tokens"), list
        ) or not isinstance(metadata.get("structural_tokens"), list):
            errors.append(f"inference {index} lacks the full raw-token trace")
            raw_token_trace_complete = False
            output_event_digest_complete = False
        else:
            raw_tokens = metadata["raw_tokens"]
            structural_tokens = metadata["structural_tokens"]
            raw_digest = _wire_digest(
                {"raw": raw_tokens, "structural": structural_tokens}
            )
            if metadata.get("raw_token_digest") != raw_digest:
                errors.append(f"inference {index} raw-token digest mismatch")
                raw_token_trace_complete = False
            decode_beats = metadata.get("token_decode_beats")
            decode_initial = metadata.get("token_decode_initial_active_pitches")
            decode_digest = metadata.get("token_decode_digest")
            decode_payload = {
                "initial_active_pitches": decode_initial,
                "beats": decode_beats,
            }
            if (
                not isinstance(decode_beats, list)
                or any(not isinstance(record, Mapping) for record in decode_beats)
                or not isinstance(decode_initial, list)
                or any(
                    isinstance(pitch, bool) or not isinstance(pitch, int)
                    for pitch in decode_initial
                )
                or decode_digest != _wire_digest(decode_payload)
            ):
                errors.append(f"inference {index} token-decode trace/digest mismatch")
                token_decode_reconciliation_complete = False
            else:
                flattened_raw = [
                    token
                    for record in decode_beats
                    for token in record.get("raw_tokens", [])
                ]
                flattened_structural = [
                    token
                    for record in decode_beats
                    for token in record.get("boundary_tokens", [])
                ]
                if flattened_raw != raw_tokens or flattened_structural != structural_tokens:
                    errors.append(
                        f"inference {index} token partitions differ from raw/structural tokens"
                    )
                    token_decode_reconciliation_complete = False
                try:
                    independently_decoded = decode_part1_token_trace(
                        decode_beats,
                        initial_active_pitches=decode_initial,
                    )
                    canonical_wire_events = [
                        {
                            "type": str(event.get("type")),
                            "pitch": int(event.get("pitch")),
                            "tick": int(event.get("tick")),
                        }
                        for event in accompaniment
                    ]
                    decoded_note_events = [
                        {
                            "type": str(event["type"]),
                            "pitch": int(event["pitch"]),
                            "tick": int(event["tick"]),
                        }
                        for event in independently_decoded
                    ]
                    # The PianoLLaMA token vocabulary encodes onset/sustain,
                    # pitch and time, but not wire velocity.  Velocity remains
                    # independently protected by output_event_digest.
                    if decoded_note_events != canonical_wire_events:
                        errors.append(
                            f"inference {index} raw-token decode differs from accompaniment events"
                        )
                        token_decode_reconciliation_complete = False
                except Exception as exc:
                    errors.append(
                        f"inference {index} raw-token decode failed: {type(exc).__name__}: {exc}"
                    )
                    token_decode_reconciliation_complete = False
            request_record = validity_by_request.get(str(request_id))
            canonical_decoded_events = [
                {
                    "type": event.get("type"),
                    "pitch": event.get("pitch"),
                    "tick": event.get("tick"),
                    "velocity": event.get("velocity"),
                }
                for event in accompaniment
            ]
            output_digest = metadata.get("output_event_digest")
            if output_digest != _wire_digest(canonical_decoded_events):
                errors.append(f"inference {index} output-event digest mismatch")
                output_event_digest_complete = False
            elif request_record is None or request_record.get(
                "output_event_digest"
            ) != output_digest:
                errors.append(
                    f"inference {index} output-event digest is not bound to validity request"
                )
                output_event_digest_complete = False
            if request_record is None or request_record.get("raw_token_digest") != raw_digest:
                errors.append(
                    f"inference {index} raw-token digest is not bound to validity request"
                )
                raw_token_trace_complete = False
            if request_record is None or request_record.get(
                "token_decode_digest"
            ) != decode_digest:
                errors.append(
                    f"inference {index} token-decode digest is not bound to validity request"
                )
                token_decode_reconciliation_complete = False
        inference_events.extend(accompaniment)
    if isinstance(content, Mapping):
        request_count = content.get("request_count")
        if (
            isinstance(request_count, bool)
            or not isinstance(request_count, int)
            or request_count <= 0
            or len(inference_rows) != request_count
        ):
            errors.append(
                "full inference trace count does not equal the positive validity request_count"
            )

    lifecycle_reconciliation_complete = True
    try:
        validity_request_ids = [
            str(request["request_id"])
            for request in validity_requests
            if isinstance(request, Mapping) and isinstance(request.get("request_id"), str)
        ] if isinstance(validity_requests, list) else []
        if len(validity_request_ids) != len(set(validity_request_ids)):
            raise ValueError("validity request IDs are duplicated")
        validity_id_set = set(validity_request_ids)
        if not validity_id_set:
            raise ValueError("validity request ID set is empty")
        if len(inference_request_ids) != len(set(inference_request_ids)):
            raise ValueError("inference lifecycle_request_id values are duplicated")
        if set(inference_request_ids) != validity_id_set:
            raise ValueError("inference request IDs differ from validity request IDs")
        if not isinstance(content, Mapping):
            raise ValueError("validity content is not an object")
        for field in (
            "expected_request_ids",
            "succeeded_request_ids",
            "processed_request_ids",
        ):
            raw_ids = content.get(field)
            if (
                not isinstance(raw_ids, list)
                or any(not isinstance(value, str) for value in raw_ids)
                or len(raw_ids) != len(set(raw_ids))
                or set(raw_ids) != validity_id_set
            ):
                raise ValueError(f"validity {field} differs from request records")
        for field in (
            "failed_request_ids",
            "stale_dropped_request_ids",
            "metadata_invalid_request_ids",
            "pending_at_stop_request_ids",
        ):
            if content.get(field) != []:
                raise ValueError(f"validity {field} must be empty for content-valid RT")
        lifecycle_by_event: dict[str, list[str]] = {}
        metadata_lifecycle: dict[str, dict[str, Mapping[str, Any]]] = {
            "succeeded": {},
            "processed": {},
        }
        for lifecycle in lifecycle_rows:
            event = lifecycle.get("event")
            request_id = lifecycle.get("request_id")
            if not isinstance(event, str) or request_id is None:
                continue
            if not isinstance(request_id, str) or request_id not in validity_id_set:
                raise ValueError("lifecycle row references an unknown request ID")
            lifecycle_by_event.setdefault(event, []).append(request_id)
            if event in metadata_lifecycle:
                metadata_lifecycle[event][request_id] = lifecycle
        for event in ("expected", "enqueued", "started", "succeeded", "processed"):
            ids = lifecycle_by_event.get(event, [])
            if len(ids) != len(set(ids)) or set(ids) != validity_id_set:
                raise ValueError(f"lifecycle {event} rows do not exactly cover request IDs")
        for forbidden in ("failed", "stale_dropped", "pending_at_stop"):
            if lifecycle_by_event.get(forbidden):
                raise ValueError(f"content-valid lifecycle contains {forbidden}")
        inference_by_id = {
            request_id: inference
            for request_id, inference in zip(inference_request_ids, inference_rows)
        }
        for request_id in sorted(validity_id_set):
            record = validity_by_request[request_id]
            if any(
                record.get(field) is not expected
                for field, expected in (
                    ("expected", True),
                    ("enqueued", True),
                    ("started", True),
                    ("succeeded", True),
                    ("processed", True),
                    ("failed", False),
                    ("stale_dropped", False),
                )
            ):
                raise ValueError(f"validity request flags are inconsistent: {request_id}")
            inference = inference_by_id[request_id]
            request_data = inference["request_data"]
            response_data = inference["response_data"]
            if int(request_data.get("generation_start_tick", -1)) != int(
                record.get("generation_start_tick", -2)
            ):
                raise ValueError(f"generation tick differs for request {request_id}")
            metadata = response_data.get("response_metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError(f"inference response metadata is invalid for {request_id}")
            for event in ("succeeded", "processed"):
                lifecycle_metadata = metadata_lifecycle[event][request_id].get(
                    "response_metadata"
                )
                if not isinstance(lifecycle_metadata, Mapping) or dict(
                    lifecycle_metadata
                ) != dict(metadata):
                    raise ValueError(
                        f"{event} lifecycle metadata differs for {request_id}"
                    )
            if record.get("raw_token_digest") != metadata.get("raw_token_digest"):
                raise ValueError(f"raw-token digest differs for request {request_id}")
            if record.get("output_event_digest") != metadata.get("output_event_digest"):
                raise ValueError(f"output-event digest differs for request {request_id}")
    except Exception as exc:
        lifecycle_reconciliation_complete = False
        errors.append(f"request lifecycle reconciliation failed: {type(exc).__name__}: {exc}")

    non_forced_trace = [
        row for row in trace_rows if row.get("policy") != "forced_note_off"
    ]
    trace_inference_equal = False
    try:
        trace_counter = Counter(
            _event_fingerprint(row, tick_field="logical_tick", include_key=True)
            for row in non_forced_trace
        )
        inference_counter = Counter(
            _event_fingerprint(row, tick_field="tick", include_key=True)
            for row in inference_events
        )
        trace_inference_equal = trace_counter == inference_counter
        if not trace_inference_equal:
            errors.append("full inference events and model schedule trace differ")
    except Exception as exc:
        errors.append(f"cannot compare inference/trace events: {type(exc).__name__}: {exc}")

    theoretical_event_rows: list[dict[str, Any]] = []
    try:
        theoretical_event_rows = [
            row
            for row in non_forced_trace
            if int(row.get("pitch", -1)) != -1
        ]
    except Exception as exc:
        errors.append(
            f"cannot identify theoretical trace events: {type(exc).__name__}: {exc}"
        )
    expected_theoretical_notes: list[tuple[int, int, int, int]] = []
    actual_theoretical_notes: list[tuple[int, int, int, int]] = []
    theoretical_semantic_equal = False
    theoretical_path = paths.get("theoretical_model_midi")
    try:
        expected_theoretical_notes = _canonical_notes_from_events(
            theoretical_event_rows, tick_field="logical_tick"
        )
        if theoretical_path is None:
            raise FileNotFoundError("theoretical_model.mid is missing")
        actual_theoretical_notes, midi_errors = _midi_track_notes(
            theoretical_path,
            track_name="Theoretical Accompaniment",
            bpm=float(config["runtime"]["playback_tempo"]),
            ticks_per_beat=int(config["runtime"]["ticks_per_beat"]),
            allowed_other_tracks={"Melody"},
        )
        errors.extend(f"theoretical MIDI: {error}" for error in midi_errors)
        theoretical_semantic_equal = (
            expected_theoretical_notes == actual_theoretical_notes and not midi_errors
        )
        if not theoretical_semantic_equal:
            errors.append("model schedule trace and theoretical_model.mid differ")
        if theoretical_event_rows and not actual_theoretical_notes:
            errors.append("non-empty theoretical event trace exported to silent MIDI")
    except Exception as exc:
        errors.append(f"cannot verify theoretical MIDI: {type(exc).__name__}: {exc}")

    source_empty = not actual_theoretical_notes
    if summary:
        summary_expectations = {
            "schema_version": 2,
            "export_status": "success",
            "trace_path": "model_schedule_trace.jsonl",
            "trace_sha256": (
                file_sha256(paths["model_schedule_trace"])
                if "model_schedule_trace" in paths
                else None
            ),
            "trace_row_count": len(trace_rows),
            "forced_note_off_excluded_count": len(trace_rows) - len(non_forced_trace),
            "event_count": len(theoretical_event_rows),
            "note_count": len(actual_theoretical_notes),
            "empty": source_empty,
            "midi_path": "theoretical_model.mid",
        }
        for field, expected in summary_expectations.items():
            actual = summary.get(field)
            if actual != expected or type(actual) is not type(expected):
                errors.append(
                    f"theoretical summary {field} mismatch: "
                    f"expected={expected!r}, actual={actual!r}"
                )

    scheduled_counter: Counter[tuple[Any, ...]] = Counter()
    actual_model_event_rows: list[dict[str, Any]] = []
    combined_schedule_consistent = False
    combined_semantic_equal = False
    try:
        scheduled_counter = Counter(
            _event_fingerprint(row, tick_field="scheduled_tick", include_key=False)
            for row in trace_rows
            if row.get("action") == "scheduled"
            and row.get("scheduled_tick") is not None
            and int(row.get("pitch", -1)) != -1
        )
        for row in actual_rows:
            data = row.get("data")
            if not isinstance(data, Mapping) or data.get("source") != "model":
                continue
            actual_model_event_rows.append(
                {
                    "type": row.get("type"),
                    "pitch": data.get("pitch"),
                    "tick": row.get("tick"),
                    "velocity": data.get("velocity"),
                }
            )
        actual_counter = Counter(
            _event_fingerprint(row, tick_field="tick", include_key=False)
            for row in actual_model_event_rows
        )
        combined_schedule_consistent = not bool(actual_counter - scheduled_counter)
        if not combined_schedule_consistent:
            errors.append("actual combined model events are absent from scheduled trace")
        all_actual_ticks = [
            int(row["tick"])
            for row in actual_rows
            if isinstance(row.get("tick"), int)
            and not isinstance(row.get("tick"), bool)
        ]
        expected_combined_notes = _canonical_notes_from_events(
            actual_model_event_rows,
            tick_field="tick",
            final_tick=max(all_actual_ticks, default=0),
        )
        combined_path = paths.get("combined_midi")
        if combined_path is None:
            raise FileNotFoundError("combined.mid is missing")
        actual_combined_notes, midi_errors = _midi_track_notes(
            combined_path,
            track_name="Accompaniment",
            bpm=float(config["runtime"]["playback_tempo"]),
            ticks_per_beat=int(config["runtime"]["ticks_per_beat"]),
            allowed_other_tracks={"Melody"},
        )
        errors.extend(f"combined MIDI: {error}" for error in midi_errors)
        combined_semantic_equal = (
            expected_combined_notes == actual_combined_notes and not midi_errors
        )
        if not combined_semantic_equal:
            errors.append("actual model event trace and combined.mid accompaniment differ")
    except Exception as exc:
        errors.append(f"cannot verify combined MIDI: {type(exc).__name__}: {exc}")

    content_valid = not errors
    return {
        "schema_version": RT_ARTIFACT_CONTRACT_SCHEMA,
        "enforced": True,
        "content_valid": content_valid,
        "operational_valid": bool(content_valid and base_operational_valid),
        "returncode": int(returncode),
        "required_artifacts": required_records,
        "source_empty": source_empty,
        "semantic_evidence": {
            "raw_token_trace_complete": raw_token_trace_complete,
            "token_decode_reconciliation_complete": token_decode_reconciliation_complete,
            "output_event_digest_complete": output_event_digest_complete,
            "lifecycle_reconciliation_complete": lifecycle_reconciliation_complete,
            "lifecycle_row_count": len(lifecycle_rows),
            "inference_event_count": len(inference_events),
            "schedule_trace_row_count": len(trace_rows),
            "trace_inference_equal": trace_inference_equal,
            "theoretical_event_count": len(theoretical_event_rows),
            "theoretical_note_count": len(actual_theoretical_notes),
            "theoretical_semantic_equal": theoretical_semantic_equal,
            "actual_model_event_count": len(actual_model_event_rows),
            "combined_schedule_consistent": combined_schedule_consistent,
            "combined_semantic_equal": combined_semantic_equal,
        },
        "errors": errors,
    }


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _offline_device_matches(expected: Any, actual: Any) -> bool:
    """Match the requested device to the concrete model-parameter device."""
    requested = str(expected)
    observed = str(actual)
    if requested == "cuda":
        return re.fullmatch(r"cuda(?::[0-9]+)?", observed) is not None
    return observed == requested


def _npz_part0_identity(npz_path: Path) -> tuple[list[int], str]:
    """Recover the exact NPZ melody roll checked by the offline encoder gate."""
    with np.load(npz_path, allow_pickle=True) as payload:
        metadata = payload["metadata"].item()
        if not isinstance(metadata, Mapping):
            raise ValueError("NPZ metadata must be a mapping")
        num_measures = int(metadata["num_measures"])
        if num_measures < 0:
            raise ValueError("NPZ num_measures must be non-negative")
        measures: list[np.ndarray] = []
        for index in range(num_measures):
            measure = np.asarray(payload[f"measure_{index}"], dtype=np.uint8)
            if measure.ndim != 3 or measure.shape[0] < 2 or measure.shape[1] != 88:
                raise ValueError(
                    f"measure_{index} has invalid shape {tuple(measure.shape)}"
                )
            measures.append(measure[:2])
    roll = (
        np.concatenate(measures, axis=2)
        if measures
        else np.zeros((2, 88, 0), dtype=np.uint8)
    )
    return [int(value) for value in roll.shape], hashlib.sha256(
        roll.tobytes(order="C")
    ).hexdigest()


def _offline_postrun_gate(
    attempt_dir: Path,
    returncode: int,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    manifest_dir: Path,
    expected_npz: Path,
    expected_source_midi: Path,
    campaign_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Independently validate one offline runner result against frozen inputs.

    The runner's JSON is evidence, not an authority: every identity/hash field is
    compared with the frozen schedule/config and every referenced output is hashed
    again here. Malformed or incomplete evidence becomes an invalid verdict instead
    of aborting before a verdict can be written.
    """
    errors: list[str] = []
    run_id = str(row.get("run_id", ""))
    offline_dir = (attempt_dir / "offline").resolve()
    expected_names = {
        "run_config": f"{run_id}_run_config.json",
        "token_trace": f"{run_id}_tokens.json",
        "generated_midi": f"{run_id}_generated.mid",
        "ground_truth_midi": f"{run_id}_gt.mid",
    }
    artifact_paths = {
        key: offline_dir / name for key, name in expected_names.items()
    }
    artifact_records: dict[str, dict[str, Any]] = {}

    if int(returncode) != 0:
        errors.append(f"offline process returned nonzero status {returncode}")
    if not offline_dir.is_dir():
        errors.append(f"offline output directory missing: {offline_dir}")
    else:
        actual_files = sorted(
            path.name for path in offline_dir.iterdir() if path.is_file()
        )
        actual_dirs = sorted(
            path.name for path in offline_dir.iterdir() if path.is_dir()
        )
        if actual_files != sorted(expected_names.values()):
            errors.append(
                "offline artifact exact set mismatch: "
                f"expected={sorted(expected_names.values())}, actual={actual_files}"
            )
        if actual_dirs:
            errors.append(f"unexpected directories in offline output: {actual_dirs}")
    for label, path in artifact_paths.items():
        if not path.is_file():
            errors.append(f"missing offline {label}: {path}")
            continue
        size = path.stat().st_size
        artifact_records[label] = _artifact_record(path, attempt_dir)
        if size <= 0:
            errors.append(f"offline {label} is empty: {path}")
    for label, path in (
        ("command", attempt_dir / "command.json"),
        ("process_log", attempt_dir / "process.log"),
    ):
        if not path.is_file():
            errors.append(f"missing offline {label}: {path}")
            continue
        artifact_records[label] = _artifact_record(path, attempt_dir)

    run_config: dict[str, Any] = {}
    token_trace: dict[str, Any] = {}
    for label, destination in (
        ("run_config", run_config), ("token_trace", token_trace)
    ):
        path = artifact_paths[label]
        if not path.is_file():
            continue
        try:
            destination.update(_read_json(path))
        except Exception as exc:
            errors.append(
                f"cannot parse offline {label}: {type(exc).__name__}: {exc}"
            )

    def expect(label: str, actual: Any, expected: Any) -> None:
        if actual != expected or type(actual) is not type(expected):
            errors.append(f"{label} mismatch: expected={expected!r}, actual={actual!r}")

    expect("run_config.schema_version", run_config.get("schema_version"), 1)
    expect("run_config.run_id", run_config.get("run_id"), run_id)
    expect("run_config.pipeline", run_config.get("pipeline"), "offline")
    expect("token_trace.schema_version", token_trace.get("schema_version"), 1)
    expect("token_trace.run_id", token_trace.get("run_id"), run_id)

    input_record = run_config.get("input")
    if not isinstance(input_record, Mapping):
        errors.append("run_config.input must be an object")
        input_record = {}
    expected_npz = expected_npz.resolve()
    expected_source_midi = expected_source_midi.resolve()
    try:
        manifest_npz, manifest_npz_sha = _resolve_artifact(
            row["input"], manifest_dir, "npz", "npz_path"
        )
        manifest_source, manifest_source_sha = _resolve_artifact(
            row["input"], manifest_dir, "output_midi", "melody_midi"
        )
        expect("manifest NPZ path", expected_npz, manifest_npz)
        expect("manifest source MIDI path", expected_source_midi, manifest_source)
    except Exception as exc:
        manifest_npz_sha = None
        manifest_source_sha = None
        errors.append(f"cannot resolve frozen input records: {type(exc).__name__}: {exc}")

    actual_npz_path = input_record.get("npz_path")
    try:
        actual_npz_resolved = Path(str(actual_npz_path)).expanduser().resolve()
    except Exception:
        actual_npz_resolved = None
    expect("input.npz_path", actual_npz_resolved, expected_npz)
    expected_stem = str(row.get("input_stem", ""))
    expect("frozen NPZ filename stem", expected_npz.stem, expected_stem)
    expect(
        "frozen source MIDI filename stem",
        expected_source_midi.stem,
        expected_stem,
    )
    expect("input.npz_stem", input_record.get("npz_stem"), expected_stem)
    if expected_npz.is_file():
        live_npz_sha = file_sha256(expected_npz)
        expect("input.npz_sha256 vs frozen manifest", live_npz_sha, manifest_npz_sha)
        expect("run_config input.npz_sha256", input_record.get("npz_sha256"), live_npz_sha)
        dataset_files = sorted(path.name for path in expected_npz.parent.glob("*.npz"))
        expect("offline dataset size", len(dataset_files), 40)
        if expected_npz.name in dataset_files:
            expect(
                "input.dataset_index",
                input_record.get("dataset_index"),
                dataset_files.index(expected_npz.name),
            )
        else:
            errors.append(f"selected NPZ is absent from its dataset directory: {expected_npz}")
    else:
        errors.append(f"expected NPZ disappeared after generation: {expected_npz}")

    source_record = input_record.get("source_midi")
    if not isinstance(source_record, Mapping):
        errors.append("run_config.input.source_midi must be an object")
        source_record = {}
    try:
        actual_source_resolved = Path(str(source_record.get("path"))).expanduser().resolve()
    except Exception:
        actual_source_resolved = None
    expect("input.source_midi.path", actual_source_resolved, expected_source_midi)
    if expected_source_midi.is_file():
        live_source_sha = file_sha256(expected_source_midi)
        expect(
            "input source MIDI hash vs frozen manifest",
            live_source_sha,
            manifest_source_sha,
        )
        expect("input.source_midi.sha256", source_record.get("sha256"), live_source_sha)
    else:
        errors.append(
            f"expected source MIDI disappeared after generation: {expected_source_midi}"
        )

    checkpoint = run_config.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        errors.append("run_config.checkpoint must be an object")
        checkpoint = {}
    frozen_checkpoint = Path(str(config["checkpoint"]["path"])).expanduser().resolve()
    try:
        actual_checkpoint = Path(str(checkpoint.get("path"))).expanduser().resolve()
    except Exception:
        actual_checkpoint = None
    expect("checkpoint.path", actual_checkpoint, frozen_checkpoint)
    if frozen_checkpoint.is_file():
        live_checkpoint_sha = file_sha256(frozen_checkpoint)
        expect(
            "checkpoint file vs frozen config",
            live_checkpoint_sha,
            config["checkpoint"]["sha256"],
        )
        expect("checkpoint.sha256", checkpoint.get("sha256"), live_checkpoint_sha)
    else:
        errors.append(f"frozen checkpoint disappeared after generation: {frozen_checkpoint}")

    sampling = run_config.get("sampling")
    if not isinstance(sampling, Mapping):
        errors.append("run_config.sampling must be an object")
        sampling = {}
    sample_seed = int(row.get("sample_seed", -1))
    sampling_expectations = {
        "seed": sample_seed,
        "temperature": config["sampling"]["temperature"],
        "top_k": config["sampling"]["top_k"],
        "top_p": config["sampling"]["top_p"],
        "repetition_penalty": config["sampling"]["repetition_penalty"],
        "gt_prefix_beats": 0,
        "delay_beats": -1,
        "bpm_override": config["runtime"]["model_condition_bpm"],
    }
    for key, expected in sampling_expectations.items():
        expect(f"sampling.{key}", sampling.get(key), expected)
    expect("token_trace.seed", token_trace.get("seed"), sample_seed)

    runtime = run_config.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append("run_config.runtime must be an object")
        runtime = {}
    if not _offline_device_matches(config["runtime"]["device"], runtime.get("device")):
        errors.append(
            "runtime.device mismatch: "
            f"requested={config['runtime']['device']!r}, actual={runtime.get('device')!r}"
        )
    expect("runtime.dtype", runtime.get("dtype"), config["runtime"]["dtype"])

    part0 = input_record.get("part0_roundtrip")
    if not isinstance(part0, Mapping):
        errors.append("run_config.input.part0_roundtrip must be an object")
        part0 = {}
    try:
        expected_roll_shape, expected_roll_sha = _npz_part0_identity(expected_npz)
    except Exception as exc:
        expected_roll_shape, expected_roll_sha = [], None
        errors.append(f"cannot independently read NPZ part0 roll: {type(exc).__name__}: {exc}")
    expect("part0_roundtrip.valid", part0.get("valid"), True)
    expect("part0_roundtrip.differing_cells", part0.get("differing_cells"), 0)
    expect("part0_roundtrip.expected_shape", part0.get("expected_shape"), expected_roll_shape)
    expect("part0_roundtrip.decoded_shape", part0.get("decoded_shape"), expected_roll_shape)
    expect("part0_roundtrip.expected_roll_sha256", part0.get("expected_roll_sha256"), expected_roll_sha)
    expect("part0_roundtrip.decoded_roll_sha256", part0.get("decoded_roll_sha256"), expected_roll_sha)
    expect("part0_roundtrip.bar_token", part0.get("bar_token"), 255)
    expect("part0_roundtrip.pad_marker", part0.get("pad_marker"), 173)
    expect("part0_roundtrip.part0_end_marker", part0.get("part0_end_marker"), 170)
    part0_tokens_sha = part0.get("part0_beat_tokens_sha256")
    if not _is_lower_sha256(part0_tokens_sha):
        errors.append("part0_roundtrip.part0_beat_tokens_sha256 is not a SHA-256")
    expect(
        "token_trace.part0_beat_tokens_sha256",
        token_trace.get("part0_beat_tokens_sha256"),
        part0_tokens_sha,
    )

    sampled_tokens = token_trace.get("sampled_tokens")
    full_tokens = token_trace.get("full_interleaved_sequence")
    if (
        not isinstance(sampled_tokens, list)
        or not sampled_tokens
        or any(isinstance(token, bool) or not isinstance(token, int) for token in sampled_tokens)
        or any(token < 0 or token >= 268 for token in sampled_tokens)
    ):
        errors.append("token_trace.sampled_tokens must be a non-empty integer list")
        sampled_tokens = []
    if (
        not isinstance(full_tokens, list)
        or not full_tokens
        or any(isinstance(token, bool) or not isinstance(token, int) for token in full_tokens)
        or any(token < 0 or token >= 268 for token in full_tokens)
    ):
        errors.append("token_trace.full_interleaved_sequence must be a non-empty integer list")
        full_tokens = []
    part1_beats = token_trace.get("part1_beats")
    flattened_part1: list[int] = []
    if not isinstance(part1_beats, list) or any(
        not isinstance(beat, list)
        or not beat
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= 268
            for token in beat
        )
        for beat in part1_beats
    ):
        errors.append("token_trace.part1_beats must be a list of non-empty token lists")
    else:
        flattened_part1 = [token for beat in part1_beats for token in beat]
        expect(
            "token_trace sampled_tokens vs flattened part1_beats",
            sampled_tokens,
            flattened_part1,
        )

    outputs = run_config.get("outputs")
    if not isinstance(outputs, Mapping):
        errors.append("run_config.outputs must be an object")
        outputs = {}
    output_contract = (
        ("generated_midi", "generated_midi_sha256", "generated_midi"),
        ("ground_truth_midi", "ground_truth_midi_sha256", "ground_truth_midi"),
        ("token_trace", "token_trace_sha256", "token_trace"),
    )
    for name_key, sha_key, artifact_key in output_contract:
        expect(f"outputs.{name_key}", outputs.get(name_key), expected_names[artifact_key])
        record = artifact_records.get(artifact_key)
        expected_sha = record.get("sha256") if record else None
        expect(f"outputs.{sha_key}", outputs.get(sha_key), expected_sha)
        if not _is_lower_sha256(outputs.get(sha_key)):
            errors.append(f"outputs.{sha_key} is not a SHA-256")
    expect("outputs.sampled_token_count", outputs.get("sampled_token_count"), len(sampled_tokens))
    expect("outputs.full_token_count", outputs.get("full_token_count"), len(full_tokens))

    binding_record: dict[str, Any] = {}
    if campaign_binding is None:
        errors.append("offline post-run gate lacks campaign binding")
    else:
        try:
            binding_record = _verdict_binding_fields(campaign_binding)
        except Exception as exc:
            errors.append(f"campaign binding is incomplete: {type(exc).__name__}: {exc}")
        if binding_record:
            expect(
                "campaign binding checkpoint_sha256",
                binding_record.get("checkpoint_sha256"),
                config["checkpoint"]["sha256"],
            )
            expect(
                "campaign binding input_manifest_sha256",
                binding_record.get("input_manifest_sha256"),
                config["input_manifest"]["sha256"],
            )
            expect(
                "campaign binding code_identity",
                binding_record.get("code_identity"),
                config["code_identity"],
            )
            for key in (
                "campaign_config_sha256", "run_schedule_sha256",
                "input_manifest_sha256", "checkpoint_sha256",
                "campaign_binding_sha256",
            ):
                if not _is_lower_sha256(binding_record.get(key)):
                    errors.append(f"campaign binding {key} is not a SHA-256")

    content_valid = not errors
    return {
        "schema_version": OFFLINE_ARTIFACT_CONTRACT_SCHEMA,
        "content_valid": content_valid,
        "operational_valid": content_valid,
        "returncode": int(returncode),
        "required_artifacts": artifact_records,
        "run_config_sha256": artifact_records.get("run_config", {}).get("sha256"),
        "token_trace_sha256": artifact_records.get("token_trace", {}).get("sha256"),
        "sampled_token_count": len(sampled_tokens),
        "full_token_count": len(full_tokens),
        "campaign_binding": binding_record,
        "errors": errors,
    }


def _rt_validity(attempt_dir: Path, returncode: int) -> tuple[bool, bool, dict[str, Any]]:
    try:
        path = _find_single(attempt_dir, "validity.json")
    except Exception as exc:
        return False, False, {
            "reason": f"invalid validity artifact: {type(exc).__name__}: {exc}",
            "returncode": returncode,
        }
    if path is None:
        return False, False, {"reason": "missing validity.json", "returncode": returncode}
    try:
        validity = _read_json(path)
    except Exception as exc:
        return False, False, {
            "reason": f"cannot parse validity.json: {type(exc).__name__}: {exc}",
            "returncode": returncode,
        }
    content = validity.get("content", {})
    operational = validity.get("operational", {})
    content_valid = bool(
        validity.get("content_valid", content.get("valid", False))
    ) and returncode == 0
    operational_valid = bool(
        validity.get("operational_valid", operational.get("valid", False))
    ) and content_valid
    return content_valid, operational_valid, validity


def _wire_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_rt_input_trace(
    midi_path: Path,
    entry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Independently reconstruct what each beat-tail request must contain."""
    from streammuse.infrastructure.inference.lekai_model.MidiConverter import (
        MidiConverter,
    )
    from streammuse.infrastructure.input.midi_file import MidiFileInput

    runtime = config["runtime"]
    ticks_per_beat = int(runtime["ticks_per_beat"])
    interval = int(runtime["generation_interval_ticks"])
    if interval != ticks_per_beat:
        raise ValueError("formal RT input trace requires interval == ticks_per_beat")
    notes, _resolution, _max_tick = MidiFileInput._midi_to_notes(
        str(midi_path),
        beat_div=ticks_per_beat,
        min_pitch=0,
        max_pitch=127,
        program=None,
        max_tick=None,
    )
    schedule: dict[int, list[dict[str, Any]]] = {}
    for note in notes:
        onset = int(note["tick"])
        offset = onset + int(note["duration"])
        schedule.setdefault(onset, []).append(
            {
                "type": "note_on",
                "pitch": int(note["pitch"]),
                "tick": onset,
                "velocity": 64,
                "channel": 0,
                "program": 0,
            }
        )
        schedule.setdefault(offset, []).append(
            {
                "type": "note_off",
                "pitch": int(note["pitch"]),
                "tick": offset,
                "velocity": 0,
                "channel": 0,
                "program": 0,
            }
        )
    events = [event for tick in sorted(schedule) for event in schedule[tick]]
    analysis_end = int(entry["analysis_end_tick"])
    cutoff = ((analysis_end - 1) // interval) * interval
    generation_ticks = list(range(interval, cutoff + 1, interval))
    converter = MidiConverter(ticks_per_beat=ticks_per_beat)
    context_ticks = int(runtime["prompt_context_beats"]) * ticks_per_beat
    generation_beats = max(
        1,
        (int(runtime["generation_length_frames"]) + ticks_per_beat - 1)
        // ticks_per_beat,
    )

    def active_before(rows: list[dict[str, Any]], tick: int) -> set[int]:
        active: set[int] = set()
        ordered = sorted(
            rows,
            key=lambda event: (
                int(event["tick"]),
                0 if event["type"] == "note_off" else 1,
            ),
        )
        for event in ordered:
            if int(event["tick"]) >= tick:
                break
            if event["type"] == "note_on":
                active.add(int(event["pitch"]))
            else:
                active.discard(int(event["pitch"]))
        return active

    trace: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    previous_tick = 0
    for generation_tick in generation_ticks:
        increment = [
            event
            for event in events
            if previous_tick <= int(event["tick"]) < generation_tick
        ]
        cumulative.extend(increment)
        context_start = max(0, generation_tick - context_ticks)
        roll_end = generation_tick + generation_beats * ticks_per_beat
        roll = converter.events_to_pianoroll(
            events=cumulative,
            start_tick=context_start,
            end_tick=roll_end,
            active_pitches=active_before(cumulative, context_start),
        )
        roll_bytes_sha = hashlib.sha256(roll.tobytes(order="C")).hexdigest()
        shape = [int(value) for value in roll.shape]
        trace.append(
            {
                "generation_start_tick": generation_tick,
                "input_event_count": len(increment),
                "input_increment_digest": _wire_digest(increment),
                "input_cumulative_digest": _wire_digest(cumulative),
                "context_start_tick": context_start,
                "part0_roll_start_tick": context_start,
                "part0_roll_end_tick": roll_end,
                "part0_roll_shape": shape,
                "part0_roll_bytes_sha256": roll_bytes_sha,
                "part0_roll_digest": _wire_digest(
                    {"shape": shape, "bytes_sha256": roll_bytes_sha}
                ),
            }
        )
        previous_tick = generation_tick
    return trace


def _validate_rt_input_trace(
    validity: Mapping[str, Any],
    midi_path: Path,
    entry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _expected_rt_input_trace(midi_path, entry, config)
    actual_rows = validity.get("requests", [])
    errors: list[str] = []
    if not isinstance(actual_rows, list):
        actual_rows = []
        errors.append("validity.requests is not a list")
    actual_by_tick: dict[int, dict[str, Any]] = {}
    for raw in actual_rows:
        if not isinstance(raw, Mapping) or raw.get("generation_start_tick") is None:
            errors.append("request record lacks generation_start_tick")
            continue
        tick = int(raw["generation_start_tick"])
        if tick in actual_by_tick:
            errors.append(f"duplicate request record at generation tick {tick}")
        actual_by_tick[tick] = dict(raw)
    expected_ticks = [int(row["generation_start_tick"]) for row in expected]
    if sorted(actual_by_tick) != expected_ticks:
        errors.append(
            f"generation tick set mismatch: expected={expected_ticks}, actual={sorted(actual_by_tick)}"
        )
    fields = (
        "input_event_count",
        "input_increment_digest",
        "input_cumulative_digest",
        "context_start_tick",
        "part0_roll_start_tick",
        "part0_roll_end_tick",
        "part0_roll_shape",
        "part0_roll_bytes_sha256",
        "part0_roll_digest",
    )
    for wanted in expected:
        tick = int(wanted["generation_start_tick"])
        got = actual_by_tick.get(tick)
        if got is None:
            continue
        for field in fields:
            if got.get(field) != wanted[field]:
                errors.append(
                    f"tick {tick} {field}: expected={wanted[field]!r}, got={got.get(field)!r}"
                )
        if got.get("server_input_increment_digest") != wanted["input_increment_digest"]:
            errors.append(f"tick {tick} server input increment digest mismatch")
        if got.get("server_input_cumulative_digest") != wanted["input_cumulative_digest"]:
            errors.append(f"tick {tick} server cumulative input digest mismatch")
        if int(got.get("effective_bpm", -1)) != int(
            config["runtime"]["model_condition_bpm"]
        ):
            errors.append(f"tick {tick} effective model BPM mismatch")
        if not isinstance(got.get("part0_token_digest"), str) or not got[
            "part0_token_digest"
        ]:
            errors.append(f"tick {tick} lacks part0 token digest")
    return {
        "valid": not errors,
        "midi_path": str(midi_path),
        "midi_sha256": file_sha256(midi_path),
        "expected_request_count": len(expected),
        "actual_request_count": len(actual_by_tick),
        "expected": expected,
        "errors": errors,
    }


def _offline_command(
    row: Mapping[str, Any], config: Mapping[str, Any], npz_path: Path,
    source_midi: Path, attempt_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_lekai_offline.py"),
        "--checkpoint", str(Path(config["checkpoint"]["path"]).resolve()),
        "--device", str(config["runtime"]["device"]),
        "--dtype", str(config["runtime"]["dtype"]),
        "--npz-dir", str(npz_path.parent),
        "--output-dir", str(attempt_dir / "offline"),
        "--condition-path", str(npz_path),
        "--source-midi", str(source_midi),
        "--require-source-midi",
        "--seed", str(row["sample_seed"]),
        "--run-id", str(row["run_id"]),
        "--bpm", str(config["runtime"]["model_condition_bpm"]),
        "--gt-prefix-beats", "0",
        "--temperature", str(config["sampling"]["temperature"]),
        "--top-k", str(config["sampling"]["top_k"]),
        "--top-p", str(config["sampling"]["top_p"]),
        "--repetition-penalty", str(config["sampling"]["repetition_penalty"]),
        "--expected-dataset-size", "40",
    ]


def _rt_command(
    row: Mapping[str, Any], config: Mapping[str, Any], midi_path: Path,
    attempt_dir: Path, generate_url: str,
) -> list[str]:
    entry = row["input"]
    analysis_end = int(entry["analysis_end_tick"])
    last_note_off = int(entry.get("last_input_note_off_tick", analysis_end))
    interval = int(config["runtime"]["generation_interval_ticks"])
    request_cutoff = ((analysis_end - 1) // interval) * interval
    run_stop = max(last_note_off, request_cutoff) + int(config["runtime"]["tail_beats"]) * 4
    return [
        sys.executable, "-m", "streammuse.presentation.cli.cli",
        "--tempo", str(config["runtime"]["playback_tempo"]),
        "--ticks-per-beat", "4", "--beats-per-bar", "4",
        "--input-mode", "midi_file", "--midi-file-path", str(midi_path),
        "--output-type", "session", "--log-dir", str(attempt_dir / "rt"),
        "--session-artifact-tier", "debug", "--inference-log-detail", "full",
        "--inference-type", "http", "--server-url", generate_url,
        "--model-name", "lekai", "--inference-mode", "sliding_window",
        "--model-condition-bpm", str(config["runtime"]["model_condition_bpm"]),
        "--generation-length-frames", str(config["runtime"]["generation_length_frames"]),
        "--generation-interval-ticks", str(interval),
        "--max-ticks", str(run_stop),
        "--analysis-end-tick", str(analysis_end),
        "--last-input-note-off-tick", str(last_note_off),
        "--request-cutoff-tick", str(request_cutoff),
        "--run-stop-tick", str(run_stop),
        "--drain-timeout-s", "60",
        "--count-in-beats", "0",
    ]


def _preflight_entry(
    row: Mapping[str, Any], manifest_dir: Path
) -> tuple[Path, Path | None, Path]:
    entry = row["input"]
    midi, midi_sha = _resolve_artifact(
        entry, manifest_dir, "output_midi", "melody_midi", "source_midi"
    )
    _verify_file(midi, midi_sha, "melody MIDI")
    npz: Path | None = None
    try:
        npz, npz_sha = _resolve_artifact(entry, manifest_dir, "npz", "npz_path")
        _verify_file(npz, npz_sha, "NPZ")
    except KeyError:
        if row["pipeline"] == "offline":
            raise
    acc, acc_sha = _resolve_artifact(entry, manifest_dir, "acc_copy", "acc", "accompaniment_midi")
    _verify_file(acc, acc_sha, "accompaniment copy")
    source_midi, source_midi_sha = _resolve_artifact(
        entry, manifest_dir, "source_midi", "clean_midi", "clean_source_midi"
    )
    _verify_file(source_midi, source_midi_sha, "clean source MIDI")
    source_acc, source_acc_sha = _resolve_artifact(
        entry, manifest_dir, "source_acc", "clean_acc", "clean_accompaniment_midi"
    )
    _verify_file(source_acc, source_acc_sha, "clean source accompaniment")
    if file_sha256(source_acc) != file_sha256(acc):
        raise RuntimeError("staged accompaniment copy differs from clean source")
    sidecar, sidecar_sha = _resolve_artifact(
        entry, manifest_dir, "sidecar", "perturbation_sidecar"
    )
    _verify_file(sidecar, sidecar_sha, "perturbation sidecar")
    return midi, npz, acc


def _execute_row(
    row: Mapping[str, Any], config: Mapping[str, Any], *, output_root: Path,
    manifest_dir: Path, generate_url: str, dry_run: bool,
    campaign_binding: Mapping[str, Any] | None = None,
    force_attempt: bool = False,
    attempt_id_override: str | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(config))
    if isinstance(row.get("runtime_overrides"), Mapping):
        config["runtime"].update(dict(row["runtime_overrides"]))
    run_dir = output_root / "runs" / str(row["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if not force_attempt:
        existing = _verified_existing_verdict(run_dir, campaign_binding)
        if existing is not None:
            return {**existing, "skipped_verified_existing": True}
    if attempt_id_override is None:
        attempt_id, attempt_dir = _next_attempt(run_dir)
    else:
        if not force_attempt:
            raise ValueError("attempt_id_override requires force_attempt=True")
        if re.fullmatch(r"attempt-[0-9]{3}", attempt_id_override) is None:
            raise ValueError(f"invalid explicit attempt ID: {attempt_id_override!r}")
        attempt_id = attempt_id_override
        attempt_dir = run_dir / attempt_id
        attempt_dir.mkdir(parents=False, exist_ok=False)
    midi, npz, _acc = _preflight_entry(row, manifest_dir)
    base_env = dict(os.environ)
    command: list[str]
    reset_ack: dict[str, Any] | None = None
    runtime_info: dict[str, Any] | None = None
    if row["pipeline"] == "offline":
        if npz is None:
            raise RuntimeError(f"offline row has no NPZ: {row['run_id']}")
        command = _offline_command(row, config, npz, midi, attempt_dir)
    else:
        base_url = _server_base(generate_url)
        if not dry_run:
            response = requests.post(
                f"{base_url}/debug/reset_session",
                json={"seed": int(row["sample_seed"])}, timeout=60,
            )
            response.raise_for_status()
            reset_ack = dict(response.json())
            runtime_response = requests.get(f"{base_url}/runtime_info", timeout=30)
            runtime_response.raise_for_status()
            runtime_info = dict(runtime_response.json())
            errors = _runtime_errors(runtime_info, config) + _reset_contract_errors(
                reset_ack, runtime_info, int(row["sample_seed"])
            )
            if errors:
                raise RuntimeError("runtime contract rejected:\n- " + "\n- ".join(errors))
            base_env["LEKAI_SESSION_ID"] = str(reset_ack["session_id"])
            base_env["LEKAI_SESSION_EPOCH"] = str(reset_ack["session_epoch"])
            base_env["LEKAI_EFFECTIVE_SEED"] = str(reset_ack["effective_seed"])
        command = _rt_command(row, config, midi, attempt_dir, generate_url)
    command_record = {
        "run_id": row["run_id"], "attempt_id": attempt_id,
        "pipeline": row["pipeline"], "command": command,
        "cwd": str(ROOT), "reset_ack": reset_ack, "runtime_info": runtime_info,
    }
    write_canonical_json(attempt_dir / "command.json", command_record)
    if dry_run:
        return {**command_record, "dry_run": True}
    returncode = _run_process(
        command, cwd=ROOT, env=base_env, output=attempt_dir / "process.log"
    )
    if row["pipeline"] == "rt":
        content_valid, operational_valid, validity = _rt_validity(attempt_dir, returncode)
        input_gate = _validate_rt_input_trace(validity, midi, row["input"], config)
        write_canonical_json(attempt_dir / "rt_input_gate.json", input_gate)
        post_runtime_info: dict[str, Any] = {}
        post_runtime_errors: list[str] = []
        try:
            post_response = requests.get(
                f"{_server_base(generate_url)}/runtime_info", timeout=30
            )
            post_response.raise_for_status()
            post_runtime_info = dict(post_response.json())
            post_runtime_errors.extend(
                _runtime_errors(
                    post_runtime_info, config, require_request_config=True
                )
            )
            if reset_ack is None:
                post_runtime_errors.append("missing reset ack for post-run runtime check")
            else:
                post_runtime_errors.extend(
                    _reset_contract_errors(
                        reset_ack, post_runtime_info, int(row["sample_seed"])
                    )
                )
        except Exception as exc:
            post_runtime_errors.append(
                f"post-run runtime_info failed: {type(exc).__name__}: {exc}"
            )
        post_runtime_gate = {
            "valid": not post_runtime_errors,
            "runtime_info": post_runtime_info,
            "errors": post_runtime_errors,
        }
        write_canonical_json(
            attempt_dir / "post_run_runtime_info.json", post_runtime_gate
        )
        validity["driver_input_gate"] = input_gate
        validity["driver_post_runtime_gate"] = post_runtime_gate
        enforce_formal_contract = bool(
            campaign_binding is not None
            and not campaign_binding.get("qualification", False)
        )
        artifact_gate = _rt_postrun_artifact_gate(
            attempt_dir,
            row=row,
            config=config,
            returncode=returncode,
            base_content_valid=bool(content_valid),
            base_operational_valid=bool(operational_valid),
            enforce_formal_contract=enforce_formal_contract,
        )
        write_canonical_json(attempt_dir / "rt_artifact_gate.json", artifact_gate)
        validity["driver_artifact_gate"] = artifact_gate
        content_valid = bool(
            content_valid
            and input_gate["valid"]
            and post_runtime_gate["valid"]
            and artifact_gate["content_valid"]
        )
        operational_valid = bool(operational_valid and content_valid)
    else:
        assert npz is not None
        validity = _offline_postrun_gate(
            attempt_dir,
            returncode,
            row,
            config,
            manifest_dir=manifest_dir,
            expected_npz=npz,
            expected_source_midi=midi,
            campaign_binding=campaign_binding,
        )
        write_canonical_json(attempt_dir / "offline_post_run_gate.json", validity)
        content_valid = bool(validity["content_valid"])
        operational_valid = bool(validity["operational_valid"])
    verdict = {
        "schema_version": "streammuse.melody_robustness.verdict.v1",
        "run_id": row["run_id"], "attempt_id": attempt_id,
        "pipeline": row["pipeline"], "content_valid": content_valid,
        "operational_valid": operational_valid, "validity": validity,
        "artifact_index": _hash_index(attempt_dir),
        "required_artifact_contract": {
            "schema_version": (
                RT_ARTIFACT_CONTRACT_SCHEMA
                if row["pipeline"] == "rt"
                else OFFLINE_ARTIFACT_CONTRACT_SCHEMA
            ),
            "enforced": bool(
                row["pipeline"] == "offline"
                or (
                    campaign_binding is not None
                    and not campaign_binding.get("qualification", False)
                )
            ),
            "gate_path": (
                "rt_artifact_gate.json"
                if row["pipeline"] == "rt"
                else "offline_post_run_gate.json"
            ),
            "required_labels": sorted(
                (
                    RT_REQUIRED_ARTIFACTS
                    if row["pipeline"] == "rt"
                    else OFFLINE_REQUIRED_ARTIFACTS
                )
            ),
        },
        **(
            _verdict_binding_fields(campaign_binding)
            if campaign_binding is not None
            else {}
        ),
    }
    write_canonical_json(attempt_dir / "verdict.json", verdict)
    # This is a mutable pointer/index only; immutable attempt verdicts remain preserved.
    write_canonical_json(run_dir / "latest_verdict.json", verdict)
    return verdict


def _start_server(config: Mapping[str, Any], generate_url: str, log_dir: Path) -> tuple[subprocess.Popen[str], Any]:
    port = _server_port(generate_url)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is already in use; refusing to share an experiment server")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (log_dir / "server.log").open("w", encoding="utf-8")
    # Do not inherit experiment-changing LEKAI_* knobs from the caller. Keep
    # normal process/CUDA/HF environment, then set every model knob explicitly.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LEKAI_") and key != "STREAMMUSE_CODE_SHA"
    }
    env.update(
        {
            "LEKAI_CHECKPOINT_PATH": str(Path(config["checkpoint"]["path"]).resolve()),
            "LEKAI_SERVER_HOST": "127.0.0.1", "LEKAI_SERVER_PORT": str(port),
            "STREAMMUSE_CODE_SHA": str(config["code_identity"]),
            "LEKAI_DEFAULT_BPM": str(config["runtime"]["model_condition_bpm"]),
            "LEKAI_DEVICE": str(config["runtime"]["device"]),
            "LEKAI_DTYPE": str(config["runtime"]["dtype"]),
            "LEKAI_TIME_SIGNATURE_INDEX": str(
                config["runtime"]["time_signature_index"]
            ),
            "LEKAI_RT_TEMPERATURE": str(config["sampling"]["temperature"]),
            "LEKAI_RT_TOP_K": str(config["sampling"]["top_k"]),
            "LEKAI_RT_TOP_P": str(config["sampling"]["top_p"]),
            "LEKAI_RT_REPETITION_PENALTY": str(config["sampling"]["repetition_penalty"]),
            "LEKAI_PROMPT_CONTEXT_BEATS": str(config["runtime"]["prompt_context_beats"]),
            "LEKAI_HISTORY_MAX_TICKS": str(config["runtime"]["history_retention_ticks"]),
            "LEKAI_DETERMINISTIC_BOUNDARY_GENERATION": "1", "LEKAI_REQUIRE_SESSION": "1",
            "LEKAI_ENABLE_DEBUG_RESET": "1",
            "LEKAI_RT_LOG_DIR": str((log_dir / "generation").resolve()),
        }
    )
    if config["runtime"].get("max_generation_length_frames") is not None:
        env["LEKAI_MAX_GENERATION_LENGTH_FRAMES"] = str(
            config["runtime"]["max_generation_length_frames"]
        )
    if config["runtime"].get("max_prompt_ticks") is not None:
        env["LEKAI_MAX_PROMPT_TICKS"] = str(config["runtime"]["max_prompt_ticks"])
    command = [sys.executable, "-m", "streammuse.infrastructure.inference.server_lekai"]
    process = subprocess.Popen(
        command,
        cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True,
    )
    allowlist_keys = [
        "LEKAI_CHECKPOINT_PATH", "LEKAI_SERVER_HOST", "LEKAI_SERVER_PORT",
        "STREAMMUSE_CODE_SHA", "LEKAI_DEFAULT_BPM", "LEKAI_RT_TEMPERATURE",
        "LEKAI_DEVICE", "LEKAI_DTYPE", "LEKAI_TIME_SIGNATURE_INDEX",
        "LEKAI_RT_TOP_K", "LEKAI_RT_TOP_P", "LEKAI_RT_REPETITION_PENALTY",
        "LEKAI_PROMPT_CONTEXT_BEATS", "LEKAI_HISTORY_MAX_TICKS",
        "LEKAI_DETERMINISTIC_BOUNDARY_GENERATION", "LEKAI_REQUIRE_SESSION",
        "LEKAI_ENABLE_DEBUG_RESET", "LEKAI_RT_LOG_DIR",
    ]
    if config["runtime"].get("max_generation_length_frames") is not None:
        allowlist_keys.append("LEKAI_MAX_GENERATION_LENGTH_FRAMES")
    if config["runtime"].get("max_prompt_ticks") is not None:
        allowlist_keys.append("LEKAI_MAX_PROMPT_TICKS")
    write_canonical_json(
        log_dir / "server_process.json",
        {
            "command": command,
            "pid": process.pid,
            "cwd": str(ROOT),
            "environment_allowlist": {key: env[key] for key in allowlist_keys},
            "code_identity": config["code_identity"],
            "checkpoint_sha256": config["checkpoint"]["sha256"],
        },
    )
    return process, log_handle


def command_freeze(args: argparse.Namespace) -> None:
    raise RuntimeError(
        "direct campaign freeze is disabled; use "
        "scripts/qualify_perturbation_campaign.py freeze so C5 is bound to "
        "canonical qualification evidence"
    )


def command_schedule(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = _read_json(config_path)
    validate_campaign_config(config)
    validate_frozen_qualification(config, verify_files=True)
    _verify_file(config_path, args.config_sha256, "campaign config")
    checkpoint_path = Path(config["checkpoint"]["path"]).resolve()
    _verify_file(checkpoint_path, config["checkpoint"]["sha256"], "checkpoint")
    manifest_path = Path(config["input_manifest"]["path"]).resolve()
    _verify_file(manifest_path, config["input_manifest"]["sha256"], "input manifest")
    validate_staged_input_manifest(
        _read_json(manifest_path), manifest_path=manifest_path, verify_files=True
    )
    _validate_frozen_listening_selection(config, _read_json(manifest_path))
    schedule = build_run_schedule(_read_json(manifest_path), config)
    digest = write_jsonl(Path(args.output), schedule)
    print(json.dumps({"run_manifest": str(Path(args.output).resolve()), "rows": 160, "sha256": digest}))


def command_run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    schedule_path = Path(args.schedule).resolve()
    config = _read_json(config_path)
    if args.allow_dirty and args.qualification:
        raise ValueError("--allow-dirty is forbidden for formal qualification")
    _require_dedicated_loopback(args.server_url)
    validate_campaign_config(
        config,
        require_frozen=not args.qualification,
        verify_attestations=True,
    )
    if args.qualification:
        if config.get("status") != "qualification_candidate":
            raise ValueError("--qualification requires a qualification candidate config")
    else:
        validate_frozen_qualification(config, verify_files=True)
    if args.allow_dirty and not args.qualification and not args.dry_run:
        raise ValueError("--allow-dirty is forbidden for formal execution")
    _verify_file(config_path, args.config_sha256, "campaign config")
    _verify_file(
        Path(config["checkpoint"]["path"]).resolve(),
        config["checkpoint"]["sha256"],
        "checkpoint",
    )
    if not args.qualification and not config["listening"].get("selection_manifest_sha256"):
        raise RuntimeError("listening selection manifest was not frozen before formal execution")
    _require_clean_identity(config["code_identity"], allow_dirty=bool(args.allow_dirty or args.dry_run))
    rows = read_jsonl(schedule_path)
    _verify_file(schedule_path, args.schedule_sha256, "run schedule")
    if len({row.get("run_id") for row in rows}) != len(rows):
        raise ValueError("run schedule contains duplicate run IDs")
    if not args.qualification and len(rows) != 160:
        raise ValueError("formal run schedule must contain exactly 160 unique rows")
    if not args.qualification and [row["pipeline"] for row in rows] != ["offline"] * 80 + ["rt"] * 80:
        raise ValueError("formal schedule must place the 80 offline runs before the 80 RT runs")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config["input_manifest"]["path"]).resolve()
    _verify_file(
        manifest_path, config["input_manifest"]["sha256"], "input manifest"
    )
    validate_staged_input_manifest(
        _read_json(manifest_path), manifest_path=manifest_path, verify_files=True
    )
    if args.qualification:
        expected_rows = build_qualification_schedule(_read_json(manifest_path), config)
        if rows != expected_rows:
            raise ValueError(
                "qualification schedule is not the canonical 20-row design"
            )
    else:
        _validate_frozen_listening_selection(config, _read_json(manifest_path))
        expected_rows = build_run_schedule(_read_json(manifest_path), config)
        if rows != expected_rows:
            raise ValueError(
                "formal schedule is not the deterministic schedule derived from the frozen config/input manifest"
            )
    campaign_binding = _bind_campaign_root(
        output_root,
        _campaign_binding(
            config_path=config_path,
            schedule_path=schedule_path,
            config=config,
            qualification=bool(args.qualification),
        ),
        create=True,
    )
    process: subprocess.Popen[str] | None = None
    server_log = None
    server_start_index = 0

    def stop_server() -> None:
        nonlocal process, server_log
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            process = None
        if server_log is not None:
            server_log.close()
            server_log = None

    def ensure_server() -> None:
        nonlocal process, server_log, server_start_index
        if process is not None or args.dry_run:
            return
        server_start_index += 1
        process, server_log = _start_server(
            config, args.server_url, output_root / f"server-{server_start_index:02d}"
        )
        _wait_for_health(_server_base(args.server_url), args.server_start_timeout)
    try:
        results: list[dict[str, Any]] = []
        if not args.dry_run and not args.qualification:
            blocks = _formal_retry_blocks(rows)
            maximum_attempts = 1 + int(
                config["validity"]["retry"]["content_failure_max_retries"]
            )
            # This is deliberately global, including blocks not selected by a
            # resume-oriented --run-id/--limit invocation.  No new directory is
            # created until every existing immutable attempt is re-verified.
            states = _preaudit_formal_retry_blocks(
                blocks=blocks,
                output_root=output_root,
                campaign_binding=campaign_binding,
                allow_partial_latest=True,
                maximum_attempts=maximum_attempts,
            )
            selected_keys = _selected_formal_block_keys(
                rows=rows,
                run_id=args.run_id,
                limit=args.limit,
            )
            for key, block_rows in blocks:
                if key not in selected_keys:
                    continue
                if key[0] == "offline" and process is not None:
                    raise RuntimeError(
                        "schedule returned to offline after starting the GPU server; "
                        "this would run two model processes concurrently"
                    )
                state = states[key]
                executed_any = False
                previous_action_attempt: str | None = None
                while True:
                    action = _formal_retry_action(
                        state, maximum_attempts=maximum_attempts
                    )
                    if action is None:
                        break
                    if key[0] == "rt":
                        if previous_action_attempt is not None:
                            # A retry attempt gets a fresh server process.  A
                            # resumed partial highest attempt starts with the
                            # fresh process owned by this invocation.
                            stop_server()
                        ensure_server()
                    for row in action["rows"]:
                        _execute_row(
                            row,
                            config,
                            output_root=output_root,
                            manifest_dir=manifest_path.parent,
                            generate_url=args.server_url,
                            dry_run=False,
                            campaign_binding=campaign_binding,
                            force_attempt=True,
                            attempt_id_override=str(action["attempt_id"]),
                        )
                        executed_any = True
                    previous_action_attempt = str(action["attempt_id"])
                    state = _inspect_formal_retry_block(
                        key=key,
                        block_rows=block_rows,
                        output_root=output_root,
                        campaign_binding=campaign_binding,
                        allow_partial_latest=False,
                        maximum_attempts=maximum_attempts,
                    )
                highest = state.get("highest_attempt_id")
                if not isinstance(highest, str):
                    raise RuntimeError(f"selected formal block produced no attempt: {key!r}")
                for row in block_rows:
                    verdict = dict(state["histories"][str(row["run_id"])][highest])
                    if not executed_any:
                        verdict["skipped_verified_existing"] = True
                    results.append(verdict)
        else:
            selected = rows
            if args.run_id:
                selected = [row for row in rows if row["run_id"] == args.run_id]
                if len(selected) != 1:
                    raise KeyError(f"unknown run_id: {args.run_id}")
            elif args.limit is not None:
                if args.limit < 0:
                    raise ValueError("--limit must be non-negative")
                selected = rows[: args.limit]
            for row in selected:
                if row["pipeline"] == "rt" and process is None and not args.dry_run:
                    ensure_server()
                if row["pipeline"] == "offline" and process is not None:
                    raise RuntimeError(
                        "schedule returned to offline after starting the GPU server; "
                        "this would run two model processes concurrently"
                    )
                results.append(
                    _execute_row(
                        row, config, output_root=output_root,
                        manifest_dir=manifest_path.parent, generate_url=args.server_url,
                        dry_run=bool(args.dry_run), campaign_binding=campaign_binding,
                    )
                )
        write_canonical_json(output_root / "last_execution.json", {"results": results})
    finally:
        stop_server()


def _formal_listening_selectors(
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the unique formal model-output selectors used by listening."""
    schema = selection.get("schema_version")
    selectors: list[dict[str, Any]] = []
    if schema == "streammuse.melody_robustness.listening_triangle_selection.v2":
        trials = selection.get("trials", selection.get("scored_trials"))
        if not isinstance(trials, list):
            raise ValueError("triangle selection lacks scored trials")
        for trial in trials:
            if not isinstance(trial, Mapping):
                raise ValueError("triangle listening trial is not an object")
            sources = trial.get("sources")
            if not isinstance(sources, Mapping):
                raise ValueError("triangle listening trial lacks sources")
            for side in ("a", "b"):
                source = sources.get(side)
                if not isinstance(source, Mapping):
                    raise ValueError(f"triangle listening trial lacks source {side}")
                if source.get("kind") != "formal":
                    continue
                selectors.append(
                    {
                        "formal_pipeline": source.get("formal_pipeline"),
                        "source_artifact": source.get("source_artifact"),
                        "presentation": source.get("presentation"),
                        "song": source.get("song"),
                        "condition": source.get("condition"),
                        "perturb_seed": source.get("perturb_seed"),
                        "sample_seed": source.get("sample_seed"),
                    }
                )
    else:
        clips = selection.get("clips")
        if not isinstance(clips, list):
            raise ValueError("listening selection lacks clips")
        for clip in clips:
            if not isinstance(clip, Mapping):
                raise ValueError("listening selection clip is not an object")
            pipeline = clip.get("pipeline")
            if pipeline not in {"rt_theoretical", "rt_combined"}:
                continue
            selectors.append(
                {
                    "formal_pipeline": "rt",
                    "source_artifact": (
                        "theoretical_model"
                        if pipeline == "rt_theoretical"
                        else "combined"
                    ),
                    "presentation": clip.get("block"),
                    "song": clip.get("song"),
                    "condition": clip.get("condition"),
                    "perturb_seed": clip.get("perturb_seed"),
                    "sample_seed": clip.get("sample_seed"),
                }
            )
    unique: dict[str, dict[str, Any]] = {}
    for selector in selectors:
        if selector["formal_pipeline"] != "rt":
            raise ValueError("listening source must use the formal RT pipeline")
        if selector["source_artifact"] not in {"theoretical_model", "combined"}:
            raise ValueError("listening source artifact is unsupported")
        key = json.dumps(selector, sort_keys=True, separators=(",", ":"))
        unique[key] = selector
    return [unique[key] for key in sorted(unique)]


def _listening_source_readiness(
    *,
    selection: Mapping[str, Any],
    schedule: list[dict[str, Any]],
    output_root: Path,
    campaign_binding: Mapping[str, Any],
) -> dict[str, Any]:
    selectors = _formal_listening_selectors(selection)
    rows: list[dict[str, Any]] = []
    for selector in selectors:
        matches = [
            row
            for row in schedule
            if row.get("pipeline") == selector["formal_pipeline"]
            and row.get("song") == selector["song"]
            and row.get("condition") == selector["condition"]
            and row.get("perturb_seed") == selector["perturb_seed"]
            and row.get("sample_seed") == selector["sample_seed"]
        ]
        result: dict[str, Any] = {"selector": selector, "ready": False}
        if len(matches) != 1:
            result["reason"] = f"selector matched {len(matches)} formal schedule rows"
            rows.append(result)
            continue
        schedule_row = matches[0]
        result["run_id"] = schedule_row["run_id"]
        run_dir = output_root / "runs" / str(schedule_row["run_id"])
        verdict = _verified_existing_verdict(run_dir, campaign_binding)
        if verdict is None:
            result["reason"] = "formal attempt/verdict/artifact contract is not valid"
            rows.append(result)
            continue
        attempt = run_dir / str(verdict["attempt_id"])
        gate_path = attempt / "rt_artifact_gate.json"
        try:
            gate = _read_json(gate_path)
            artifact_label = (
                "theoretical_model_midi"
                if selector["source_artifact"] == "theoretical_model"
                else "combined_midi"
            )
            record = gate["required_artifacts"][artifact_label]
            if not isinstance(record, Mapping):
                raise ValueError("listening source record is not an object")
            artifact_path = (attempt / str(record["path"])).resolve()
            if (
                not artifact_path.is_relative_to(attempt.resolve())
                or not artifact_path.is_file()
                or artifact_path.stat().st_size != record.get("size")
                or file_sha256(artifact_path) != record.get("sha256")
            ):
                raise ValueError("listening source artifact size/hash mismatch")
        except Exception as exc:
            result["reason"] = (
                f"cannot resolve verified listening source: {type(exc).__name__}: {exc}"
            )
            rows.append(result)
            continue
        result.update(
            {
                "ready": True,
                "attempt_id": verdict["attempt_id"],
                "operational_valid": bool(verdict.get("operational_valid")),
                "source_empty": bool(gate.get("source_empty")),
                "artifact": dict(record),
            }
        )
        rows.append(result)
    ready_count = sum(row["ready"] for row in rows)
    return {
        "schema_version": (
            "streammuse.melody_robustness.listening_source_readiness.v1"
        ),
        "selection_schema_version": selection.get("schema_version"),
        "expected_unique_sources": len(selectors),
        "ready_sources": ready_count,
        "not_ready_sources": len(selectors) - ready_count,
        "ready": bool(selectors) and ready_count == len(selectors),
        "sources": rows,
    }


def command_audit(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    schedule_path = Path(args.schedule).resolve()
    config = _read_json(config_path)
    validate_campaign_config(config)
    validate_frozen_qualification(config, verify_files=True)
    _verify_file(config_path, args.config_sha256, "campaign config")
    _verify_file(schedule_path, args.schedule_sha256, "run schedule")
    schedule = read_jsonl(schedule_path)
    manifest_path = Path(config["input_manifest"]["path"]).resolve()
    _verify_file(
        manifest_path, config["input_manifest"]["sha256"], "input manifest"
    )
    validate_staged_input_manifest(
        _read_json(manifest_path), manifest_path=manifest_path, verify_files=True
    )
    selection_path = _validate_frozen_listening_selection(
        config, _read_json(manifest_path)
    )
    _verify_file(
        Path(config["checkpoint"]["path"]).resolve(),
        config["checkpoint"]["sha256"],
        "checkpoint",
    )
    expected_schedule = build_run_schedule(_read_json(manifest_path), config)
    if schedule != expected_schedule:
        raise ValueError(
            "audit schedule is not the exact 160-row schedule derived from frozen config/input"
        )
    root = Path(args.output_root).resolve()
    binding = _bind_campaign_root(
        root,
        _campaign_binding(
            config_path=config_path,
            schedule_path=schedule_path,
            config=config,
            qualification=False,
        ),
        create=False,
    )
    expected_run_ids = {str(row["run_id"]) for row in schedule}
    runs_dir = root / "runs"
    actual_run_ids = {
        path.name for path in runs_dir.iterdir() if path.is_dir()
    } if runs_dir.is_dir() else set()
    extra_run_ids = sorted(actual_run_ids - expected_run_ids)
    retry_blocks = _formal_retry_blocks(schedule)
    block_states: dict[tuple[str, str, int], dict[str, Any]] = {}
    attempt_block_errors: list[dict[str, Any]] = []
    for key, block_rows in retry_blocks:
        try:
            block_states[key] = _inspect_formal_retry_block(
                key=key,
                block_rows=block_rows,
                output_root=root,
                campaign_binding=binding,
                allow_partial_latest=False,
                maximum_attempts=(
                    1
                    + int(
                        config["validity"]["retry"][
                            "content_failure_max_retries"
                        ]
                    )
                ),
            )
        except Exception as exc:
            attempt_block_errors.append(
                {
                    "pipeline": key[0],
                    "song": key[1],
                    "sample_seed": key[2],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    rows: list[dict[str, Any]] = []
    for expected in schedule:
        run_dir = root / "runs" / expected["run_id"]
        verdict_path = run_dir / "latest_verdict.json"
        if not verdict_path.is_file():
            rows.append({"run_id": expected["run_id"], "status": "missing"})
            continue
        verdict = _verified_existing_verdict(run_dir, binding)
        if verdict is not None and verdict.get("pipeline") != expected["pipeline"]:
            verdict = None
        try:
            attempt_ids = _attempt_directory_ids(run_dir)
        except Exception:
            attempt_ids = sorted(
                path.name for path in run_dir.glob("attempt-*") if path.is_dir()
            )
        attempts = len(attempt_ids)
        if verdict is None:
            rows.append(
                {
                    "run_id": expected["run_id"],
                    "pipeline": expected["pipeline"],
                    "status": "invalid",
                    "operational_valid": False,
                    "empty_success": False,
                    "source_empty": False,
                    "attempts": attempts,
                    "attempt_ids": attempt_ids,
                    "reason": "latest verdict or indexed artifacts failed strict verification",
                }
            )
            continue
        rows.append(
            {
                "run_id": expected["run_id"], "pipeline": expected["pipeline"],
                "status": "valid" if verdict.get("content_valid") else "invalid",
                "operational_valid": bool(verdict.get("operational_valid")),
                "empty_success": bool(
                    verdict.get("validity", {})
                    .get("content", {})
                    .get("empty_success", False)
                ),
                "source_empty": bool(
                    verdict.get("validity", {})
                    .get("driver_artifact_gate", {})
                    .get("source_empty", False)
                ),
                "attempts": attempts,
                "attempt_ids": attempt_ids,
            }
        )
    listening_readiness = _listening_source_readiness(
        selection=_read_json(selection_path),
        schedule=schedule,
        output_root=root,
        campaign_binding=binding,
    )
    summary = {
        "campaign_config_sha256": binding["campaign_config_sha256"],
        "run_schedule_sha256": binding["run_schedule_sha256"],
        "campaign_binding_sha256": binding["campaign_binding_sha256"],
        "qualification_result_sha256": binding["qualification_result_sha256"],
        "expected": len(schedule), "present": sum(row["status"] != "missing" for row in rows),
        "content_valid": sum(row["status"] == "valid" for row in rows),
        "missing": sum(row["status"] == "missing" for row in rows),
        "invalid": sum(row["status"] == "invalid" for row in rows),
        "retried": sum(int(row.get("attempts", 0)) > 1 for row in rows),
        "operational_invalid": sum(
            row["status"] == "valid" and not row.get("operational_valid", False) for row in rows
        ),
        "source_empty": sum(
            row["status"] == "valid" and row.get("source_empty", False)
            for row in rows
        ),
        "extra_run_ids": extra_run_ids,
        "attempt_block_errors": attempt_block_errors,
        "retry_blocks": [
            {
                "pipeline": key[0],
                "song": key[1],
                "sample_seed": key[2],
                "attempt_ids": state["attempt_ids"],
            }
            for key, state in block_states.items()
        ],
        "listening_source_readiness": listening_readiness,
        "runs": rows,
    }
    write_canonical_json(Path(args.output), summary)
    if (
        summary["expected"] != 160
        or summary["missing"]
        or summary["invalid"]
        or extra_run_ids
        or attempt_block_errors
        or not listening_readiness["ready"]
    ):
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    schedule = sub.add_parser("schedule")
    schedule.add_argument("--config", required=True)
    schedule.add_argument("--config-sha256")
    schedule.add_argument("--output", required=True)
    schedule.set_defaults(func=command_schedule)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--config-sha256", required=True)
    run.add_argument("--schedule", required=True)
    run.add_argument("--schedule-sha256", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--server-url", default="http://127.0.0.1:8765/generate_accompaniment")
    run.add_argument("--server-start-timeout", type=float, default=180.0)
    run.add_argument("--run-id")
    run.add_argument("--limit", type=int)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development dry-runs only; formal and qualification runs reject it",
    )
    run.add_argument(
        "--qualification", action="store_true",
        help="accept a pre-formal qualification schedule and candidate config",
    )
    run.set_defaults(func=command_run)
    audit = sub.add_parser("audit")
    audit.add_argument("--config", required=True)
    audit.add_argument("--config-sha256", required=True)
    audit.add_argument("--schedule", required=True)
    audit.add_argument("--schedule-sha256", required=True)
    audit.add_argument("--output-root", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=command_audit)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
