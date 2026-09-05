#!/usr/bin/env python
"""Run matched StreamMUSE v1/v2 system trials from a cohort manifest.

This runner intentionally covers system execution and artifact validation only.
It does not run offline inference or compute music/system metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_IDS = (
    "streammuse_v1_standard",
    "streammuse_v2_prompt_continuation",
)
BPM = 120
TICKS_PER_BEAT = 4
BEATS_PER_BAR = 4
WINDOW_START_TICK = 0
WINDOW_END_TICK = 128
PROMPT_LENGTH_TICKS = 32
GENERATION_INTERVAL_TICKS = 4
GENERATION_LENGTH_FRAMES = 4
HISTORY_MAX_TICKS = 128
PROMPT_CONTEXT_BEATS = 32
CHECKPOINT_TIME_SIGNATURE = "4/4"
SAMPLING = {
    "temperature": 1.05,
    "top_p": 0.98,
    "top_k": 0,
    "repetition_penalty": 1.0,
}
PROMPT_CANDIDATES = 5
PROMPT_SELECTION_MODES = ("single", "rule_s", "rule_s_v3")
EVAL_MANIFEST_FIELDS = (
    "piece_id",
    "seed",
    "system_id",
    "session_dir",
    "run_status",
    "melody_input_sha256",
    "failure_reason",
)
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class CohortPiece:
    piece_id: str
    midi_path: Path
    melody_input_sha256: str
    canonical_melody_input_sha256: str | None = None


@dataclass
class ServerHandle:
    system_id: str
    base_url: str
    process: subprocess.Popen[str]
    log_handle: TextIO
    environment: dict[str, str]
    startup_runtime: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched StreamMUSE v1/v2 MIDI-file system trials."
    )
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt-checkpoint", type=Path, required=True)
    parser.add_argument("--continuation-checkpoint", type=Path, required=True)
    parser.add_argument("--time-signature-index", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--systems",
        default=",".join(SYSTEM_IDS),
        help=f"Comma-separated subset of: {','.join(SYSTEM_IDS)}",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--prompt-selection-mode",
        choices=PROMPT_SELECTION_MODES,
        default="rule_s",
    )
    parser.add_argument(
        "--prompt-batch-candidates", type=int, default=PROMPT_CANDIDATES
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=0,
        help="Use only the first N pieces; 0 uses the full cohort.",
    )
    parser.add_argument("--server-start-timeout-s", type=float, default=600.0)
    parser.add_argument("--trial-timeout-s", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def effective_prompt_candidate_count(selection_mode: str, requested: int) -> int:
    if selection_mode not in PROMPT_SELECTION_MODES:
        raise ValueError(f"unsupported prompt selection mode: {selection_mode}")
    if selection_mode == "single":
        return 1
    if requested < 2:
        raise ValueError(
            f"--prompt-batch-candidates must be at least 2 for {selection_mode}"
        )
    return int(requested)


def _checkpoint_conditioning(time_signature_index: int) -> dict[str, Any]:
    return {
        "time_signature": CHECKPOINT_TIME_SIGNATURE,
        "continuation_time_signature_index": time_signature_index,
        "prompt_time_signature_index": time_signature_index,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _eval_run_status(record: Mapping[str, Any]) -> str:
    status = str(record.get("run_status", "pending"))
    if status == "complete":
        return "complete"
    if status == "failed":
        return "failed"
    return "missing"


def _relative_session_dir(record: Mapping[str, Any], manifest_path: Path) -> str:
    value = record.get("session_dir")
    if not value:
        return ""
    session_path = Path(str(value)).expanduser()
    if not session_path.is_absolute():
        session_path = manifest_path.parent / session_path
    return Path(
        os.path.relpath(session_path.resolve(), manifest_path.parent.resolve())
    ).as_posix()


def write_eval_manifest(path: Path, trials: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write the evaluator's exact seven-column trial manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVAL_MANIFEST_FIELDS))
        writer.writeheader()
        for trial in trials:
            status = _eval_run_status(trial)
            session_dir = _relative_session_dir(trial, path)
            if status == "complete" and not session_dir:
                raise ValueError("complete eval rows require session_dir")
            if status == "complete":
                failure_reason = ""
            else:
                failure_reason = str(trial.get("failure_reason") or "").strip()
                if not failure_reason:
                    source_status = str(trial.get("run_status", "pending"))
                    failure_reason = {
                        "pending": "not_run_yet",
                        "running": "trial_in_progress",
                        "dry_run": "dry_run_not_executed",
                    }.get(source_status, "trial_not_available")
            writer.writerow(
                {
                    "piece_id": trial["piece_id"],
                    "seed": trial["seed"],
                    "system_id": trial["system_id"],
                    "session_dir": session_dir,
                    "run_status": status,
                    "melody_input_sha256": trial["melody_input_sha256"],
                    "failure_reason": failure_reason,
                }
            )
    temporary.replace(path)


def persist_run_manifests(output_root: Path, manifest: Mapping[str, Any]) -> None:
    write_json(output_root / "run_manifest.json", manifest)
    write_eval_manifest(output_root / "eval_manifest.csv", manifest["trials"])


def _parse_unique_ints(raw: str, label: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique integers")
    return values


def _parse_systems(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--systems must contain unique system IDs")
    unknown = sorted(set(values) - set(SYSTEM_IDS))
    if unknown:
        raise ValueError(f"unknown system IDs: {', '.join(unknown)}")
    return values


def _read_manifest_rows(path: Path) -> tuple[list[Mapping[str, Any]], str]:
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        schema = "jsonl"
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
            schema = "list"
        elif isinstance(payload, dict):
            if isinstance(payload.get("samples"), list):
                rows = payload["samples"]
                schema = "samples"
            elif isinstance(payload.get("pieces"), list):
                rows = payload["pieces"]
                schema = "pieces"
            else:
                rows = payload.get("entries")
                schema = "entries"
        else:
            rows = None
            schema = "unknown"
        if not isinstance(rows, list):
            raise ValueError(
                "cohort manifest must be a list or contain samples, pieces, or entries"
            )
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("every cohort manifest row must be an object")
    return rows, schema


def load_cohort_manifest(path: Path, smoke_limit: int = 0) -> list[CohortPiece]:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cohort manifest not found: {manifest_path}")
    if smoke_limit < 0:
        raise ValueError("--smoke-limit must be >= 0")
    rows, manifest_schema = _read_manifest_rows(manifest_path)
    if smoke_limit:
        rows = rows[:smoke_limit]
    if not rows:
        raise ValueError("cohort manifest selects no pieces")

    pieces: list[CohortPiece] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        piece_id = str(row.get("piece_id", "")).strip()
        if not piece_id:
            raise ValueError(f"cohort row {index} has no piece_id")
        if piece_id in seen_ids:
            raise ValueError(f"duplicate piece_id: {piece_id}")
        seen_ids.add(piece_id)

        artifact = row.get("midi_path", row.get("melody_midi"))
        expected_hash = row.get(
            "melody_midi_sha256", row.get("melody_input_sha256")
        )
        if isinstance(artifact, Mapping):
            expected_hash = artifact.get("sha256", expected_hash)
            artifact = artifact.get("path")
        if not isinstance(artifact, str) or not artifact.strip():
            raise ValueError(f"cohort row {piece_id} has no midi_path")
        midi_path = Path(artifact).expanduser()
        if not midi_path.is_absolute():
            midi_path = manifest_path.parent / midi_path
        midi_path = midi_path.resolve()
        if not midi_path.is_file():
            raise FileNotFoundError(f"MIDI input not found for {piece_id}: {midi_path}")
        if manifest_schema == "samples" and row.get("melody_midi_sha256") is None:
            raise ValueError(
                f"cohort sample {piece_id} has no melody_midi_sha256"
            )
        if expected_hash is not None and not SHA256_PATTERN.fullmatch(
            str(expected_hash)
        ):
            raise ValueError(f"invalid MIDI SHA-256 for {piece_id}: {expected_hash}")
        canonical_hash = row.get("canonical_melody_input_sha256")
        if canonical_hash is not None and not SHA256_PATTERN.fullmatch(
            str(canonical_hash)
        ):
            raise ValueError(
                f"invalid canonical melody SHA-256 for {piece_id}: {canonical_hash}"
            )
        actual_hash = file_sha256(midi_path)
        if expected_hash is not None and str(expected_hash).lower() != actual_hash:
            raise ValueError(
                f"MIDI hash mismatch for {piece_id}: expected {expected_hash}, "
                f"got {actual_hash}"
            )
        pieces.append(
            CohortPiece(
                piece_id=piece_id,
                midi_path=midi_path,
                melody_input_sha256=actual_hash,
                canonical_melody_input_sha256=(
                    str(canonical_hash).lower() if canonical_hash is not None else None
                ),
            )
        )
    return pieces


def checkpoint_identity(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {resolved}")
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def code_identity(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "repository_root": str(repo_root.resolve()),
        "git_commit": commit,
        "git_clean": not bool(status),
        "git_status_porcelain": status,
    }


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _controlled_environment(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LEKAI_") and key != "STREAMMUSE_CODE_SHA"
    }
    python_paths = [str((repo_root / "src").resolve())]
    bundled_transformers = repo_root / "transformers" / "src"
    if bundled_transformers.is_dir():
        python_paths.append(str(bundled_transformers.resolve()))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def build_server_environment(
    system_id: str,
    *,
    port: int,
    gpu: str,
    server_dir: Path,
    code: Mapping[str, Any],
    prompt_checkpoint: Mapping[str, Any],
    continuation_checkpoint: Mapping[str, Any],
    time_signature_index: int,
    prompt_selection_mode: str = "rule_s",
    prompt_batch_candidates: int = PROMPT_CANDIDATES,
) -> dict[str, str]:
    if system_id not in SYSTEM_IDS:
        raise ValueError(f"unknown system: {system_id}")
    env = _controlled_environment()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "STREAMMUSE_CODE_SHA": str(code["git_commit"]),
            "LEKAI_SERVER_HOST": "127.0.0.1",
            "LEKAI_SERVER_PORT": str(port),
            "LEKAI_ENABLE_DEBUG_RESET": "true",
            "LEKAI_DEFAULT_BPM": str(BPM),
            "LEKAI_DEVICE": "cuda",
            "LEKAI_DTYPE": "float16",
            "LEKAI_TIME_SIGNATURE_INDEX": str(time_signature_index),
            "LEKAI_PROMPT_TIME_SIGNATURE_INDEX": str(time_signature_index),
            "LEKAI_RT_TEMPERATURE": str(SAMPLING["temperature"]),
            "LEKAI_RT_TOP_P": str(SAMPLING["top_p"]),
            "LEKAI_RT_TOP_K": str(SAMPLING["top_k"]),
            "LEKAI_RT_REPETITION_PENALTY": str(
                SAMPLING["repetition_penalty"]
            ),
            "LEKAI_PROMPT_CONTEXT_BEATS": str(PROMPT_CONTEXT_BEATS),
            "LEKAI_HISTORY_MAX_TICKS": str(HISTORY_MAX_TICKS),
            "LEKAI_DETERMINISTIC_BOUNDARY_GENERATION": "1",
            "LEKAI_RT_LOG_DIR": str((server_dir / "generation").resolve()),
        }
    )
    if system_id == "streammuse_v1_standard":
        env.update(
            {
                "LEKAI_CHECKPOINT_PATH": str(continuation_checkpoint["path"]),
                "LEKAI_REQUIRE_SESSION": "1",
            }
        )
    else:
        effective_candidates = effective_prompt_candidate_count(
            prompt_selection_mode, prompt_batch_candidates
        )
        env.update(
            {
                "LEKAI_PROMPT_CHECKPOINT_PATH": str(prompt_checkpoint["path"]),
                "LEKAI_CONTINUATION_CHECKPOINT_PATH": str(
                    continuation_checkpoint["path"]
                ),
                "LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS": "1",
                "LEKAI_PROMPT_CONTINUATION_ENGINE": "standard",
                "LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP": "1",
                "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS": "0",
                "LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY": "0",
                "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS": "0",
                "LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES": "0",
                "LEKAI_PROMPT_SELECTION_MODE": prompt_selection_mode,
                "LEKAI_PROMPT_BATCH_CANDIDATES": str(effective_candidates),
                "LEKAI_PROMPT_SEED": "0",
                "LEKAI_PROMPT_BPM": str(BPM),
                "LEKAI_PROMPT_DEVICE": "cuda",
                "LEKAI_PROMPT_DTYPE": "float16",
                "LEKAI_PROMPT_TEMPERATURE": str(SAMPLING["temperature"]),
                "LEKAI_PROMPT_TOP_P": str(SAMPLING["top_p"]),
                "LEKAI_PROMPT_TOP_K": str(SAMPLING["top_k"]),
                "LEKAI_PROMPT_REPETITION_PENALTY": str(
                    SAMPLING["repetition_penalty"]
                ),
            }
        )
    return env


def _environment_record(env: Mapping[str, str]) -> dict[str, str]:
    return {
        key: env[key]
        for key in sorted(env)
        if key.startswith("LEKAI_")
        or key in {"CUDA_VISIBLE_DEVICES", "STREAMMUSE_CODE_SHA", "PYTHONPATH"}
    }


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"expected object response from {url}")
    return parsed


def _runtime_endpoint(system_id: str) -> str:
    if system_id == "streammuse_v1_standard":
        return "/runtime_info"
    return "/prompt_continuation/runtime_info"


def _resolved_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).expanduser().resolve()


def runtime_contract_errors(
    system_id: str,
    runtime: Mapping[str, Any],
    *,
    code: Mapping[str, Any],
    prompt_checkpoint: Mapping[str, Any],
    continuation_checkpoint: Mapping[str, Any],
    time_signature_index: int,
    expected_seed: int | None = None,
    reset_ack: Mapping[str, Any] | None = None,
    prompt_selection_mode: str = "rule_s",
    prompt_batch_candidates: int = PROMPT_CANDIDATES,
) -> list[str]:
    errors: list[str] = []
    if runtime.get("has_real_model") is not True:
        errors.append("continuation model is not real")
    if runtime.get("fallback_reason") not in (None, ""):
        errors.append(f"continuation fallback: {runtime.get('fallback_reason')}")
    if runtime.get("checkpoint_sha256") != continuation_checkpoint["sha256"]:
        errors.append("continuation checkpoint SHA-256 mismatch")
    if runtime.get("code_identity") != code["git_commit"]:
        errors.append("runtime code identity mismatch")
    if not str(runtime.get("resolved_device", "")).startswith("cuda"):
        errors.append(f"continuation device is not CUDA: {runtime.get('resolved_device')}")
    expected_values = {
        "effective_bpm": BPM,
        "ticks_per_beat": TICKS_PER_BEAT,
        "prompt_context_beats": PROMPT_CONTEXT_BEATS,
        "history_retention_ticks": HISTORY_MAX_TICKS,
        "time_signature_index": time_signature_index,
        **SAMPLING,
    }
    for key, expected in expected_values.items():
        actual = runtime.get(key)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-9
            ):
                errors.append(f"runtime {key}={actual!r}, expected {expected!r}")
        elif actual != expected:
            errors.append(f"runtime {key}={actual!r}, expected {expected!r}")
    if system_id == "streammuse_v2_prompt_continuation":
        effective_candidates = effective_prompt_candidate_count(
            prompt_selection_mode, prompt_batch_candidates
        )
        if runtime.get("prompt_has_real_model") is not True:
            errors.append("prompt model is not real")
        if runtime.get("prompt_fallback_reason") not in (None, ""):
            errors.append(f"prompt fallback: {runtime.get('prompt_fallback_reason')}")
        if _resolved_path(runtime.get("prompt_checkpoint_path")) != Path(
            str(prompt_checkpoint["path"])
        ).resolve():
            errors.append("prompt checkpoint path mismatch")
        if runtime.get("prompt_selection_mode") != prompt_selection_mode:
            errors.append(
                "prompt selection mode "
                f"is {runtime.get('prompt_selection_mode')!r}, "
                f"expected {prompt_selection_mode!r}"
            )
        if runtime.get("prompt_batch_candidate_count") != effective_candidates:
            errors.append(
                "prompt candidate count "
                f"is {runtime.get('prompt_batch_candidate_count')!r}, "
                f"expected {effective_candidates}"
            )
    if expected_seed is not None:
        if runtime.get("sample_seed") != expected_seed:
            errors.append(
                f"runtime continuation seed={runtime.get('sample_seed')!r}, "
                f"expected {expected_seed}"
            )
        if reset_ack is None:
            errors.append("missing reset acknowledgement")
        else:
            if runtime.get("session_id") != reset_ack.get("session_id"):
                errors.append("runtime session_id differs from reset acknowledgement")
            if runtime.get("session_epoch") != reset_ack.get("session_epoch"):
                errors.append("runtime session_epoch differs from reset acknowledgement")
    return errors


def wait_for_server(
    handle: ServerHandle,
    *,
    timeout_s: float,
    code: Mapping[str, Any],
    prompt_checkpoint: Mapping[str, Any],
    continuation_checkpoint: Mapping[str, Any],
    time_signature_index: int,
    prompt_selection_mode: str = "rule_s",
    prompt_batch_candidates: int = PROMPT_CANDIDATES,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = "server not contacted"
    while time.monotonic() < deadline:
        if handle.process.poll() is not None:
            raise RuntimeError(
                f"{handle.system_id} server exited with {handle.process.returncode}"
            )
        try:
            health = request_json(f"{handle.base_url}/health", timeout=3.0)
            if health.get("status") == "ok":
                runtime = request_json(
                    f"{handle.base_url}{_runtime_endpoint(handle.system_id)}",
                    timeout=30.0,
                )
                errors = runtime_contract_errors(
                    handle.system_id,
                    runtime,
                    code=code,
                    prompt_checkpoint=prompt_checkpoint,
                    continuation_checkpoint=continuation_checkpoint,
                    time_signature_index=time_signature_index,
                    prompt_selection_mode=prompt_selection_mode,
                    prompt_batch_candidates=prompt_batch_candidates,
                )
                if errors:
                    raise RuntimeError("; ".join(errors))
                return runtime
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1.0)
    raise TimeoutError(
        f"{handle.system_id} server did not become ready in {timeout_s}s: {last_error}"
    )


def start_server(
    system_id: str,
    *,
    output_root: Path,
    python_bin: str,
    gpu: str,
    timeout_s: float,
    code: Mapping[str, Any],
    prompt_checkpoint: Mapping[str, Any],
    continuation_checkpoint: Mapping[str, Any],
    time_signature_index: int,
    prompt_selection_mode: str = "rule_s",
    prompt_batch_candidates: int = PROMPT_CANDIDATES,
) -> ServerHandle:
    server_dir = output_root / "servers" / system_id
    server_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = build_server_environment(
        system_id,
        port=port,
        gpu=gpu,
        server_dir=server_dir,
        code=code,
        prompt_checkpoint=prompt_checkpoint,
        continuation_checkpoint=continuation_checkpoint,
        time_signature_index=time_signature_index,
        prompt_selection_mode=prompt_selection_mode,
        prompt_batch_candidates=prompt_batch_candidates,
    )
    command = [python_bin, "-m", "streammuse.infrastructure.inference.server_lekai"]
    log_handle = (server_dir / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    handle = ServerHandle(
        system_id=system_id,
        base_url=base_url,
        process=process,
        log_handle=log_handle,
        environment=env,
        startup_runtime={},
    )
    try:
        handle.startup_runtime = wait_for_server(
            handle,
            timeout_s=timeout_s,
            code=code,
            prompt_checkpoint=prompt_checkpoint,
            continuation_checkpoint=continuation_checkpoint,
            time_signature_index=time_signature_index,
            prompt_selection_mode=prompt_selection_mode,
            prompt_batch_candidates=prompt_batch_candidates,
        )
    except Exception:
        stop_server(handle)
        raise
    write_json(
        server_dir / "server_identity.json",
        {
            "system_id": system_id,
            "pid": process.pid,
            "base_url": base_url,
            "command": command,
            "environment": _environment_record(env),
            "startup_runtime": handle.startup_runtime,
            "code_identity": code,
            "prompt_checkpoint": prompt_checkpoint,
            "continuation_checkpoint": continuation_checkpoint,
        },
    )
    return handle


def stop_server(handle: ServerHandle) -> None:
    if handle.process.poll() is None:
        handle.process.terminate()
        try:
            handle.process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait(timeout=15.0)
    handle.log_handle.close()


def reset_trial(system_id: str, base_url: str, seed: int) -> dict[str, Any]:
    if system_id == "streammuse_v1_standard":
        ack = request_json(
            f"{base_url}/debug/reset_session",
            method="POST",
            payload={"seed": int(seed)},
            timeout=120.0,
        )
        if ack.get("effective_seed") != seed:
            raise RuntimeError("standard reset returned the wrong effective seed")
    elif system_id == "streammuse_v2_prompt_continuation":
        ack = request_json(
            f"{base_url}/prompt_continuation/debug/reset_session",
            method="POST",
            payload={"prompt_seed": int(seed), "continuation_seed": int(seed)},
            timeout=120.0,
        )
        if ack.get("prompt_seed") != seed:
            raise RuntimeError("P+C reset returned the wrong prompt seed")
        if ack.get("continuation_effective_seed") != seed:
            raise RuntimeError("P+C reset returned the wrong continuation seed")
        if ack.get("scheduler_phase") != "idle" or ack.get("scheduler_is_running"):
            raise RuntimeError("P+C scheduler was not idle after reset")
    else:
        raise ValueError(f"unknown system: {system_id}")
    if not ack.get("session_id") or not isinstance(ack.get("session_epoch"), int):
        raise RuntimeError("reset acknowledgement has no session identity")
    if ack.get("pending_boundary_generations") != 0:
        raise RuntimeError("reset left pending boundary generations")
    return ack


def build_cli_command(
    system_id: str,
    *,
    python_bin: str,
    midi_path: Path,
    log_root: Path,
    base_url: str,
) -> list[str]:
    continuation_mode = (
        "standard"
        if system_id == "streammuse_v1_standard"
        else "prompt_continuation"
    )
    command = [
        python_bin,
        "-m",
        "streammuse.presentation.cli.cli",
        "--tempo",
        str(BPM),
        "--ticks-per-beat",
        str(TICKS_PER_BEAT),
        "--beats-per-bar",
        str(BEATS_PER_BAR),
        "--input-mode",
        "midi_file",
        "--midi-file-path",
        str(midi_path),
        "--output-type",
        "session",
        "--log-dir",
        str(log_root),
        "--session-artifact-tier",
        "debug",
        "--inference-log-detail",
        "full",
        "--inference-type",
        "http",
        "--server-url",
        f"{base_url}/generate_accompaniment",
        "--model-name",
        "lekai",
        "--inference-mode",
        "sliding_window",
        "--model-condition-bpm",
        str(BPM),
        "--generation-length-frames",
        str(GENERATION_LENGTH_FRAMES),
        "--generation-interval-ticks",
        str(GENERATION_INTERVAL_TICKS),
        "--run-stop-tick",
        str(WINDOW_END_TICK),
        "--count-in-beats",
        "0",
        "--continuation-mode",
        continuation_mode,
    ]
    if continuation_mode == "prompt_continuation":
        command.extend(["--prompt-length-ticks", str(PROMPT_LENGTH_TICKS)])
    return command


def build_client_environment(
    system_id: str, reset_ack: Mapping[str, Any] | None
) -> dict[str, str]:
    env = _controlled_environment()
    env.update(
        {
            "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS": "0",
            "LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY": "0",
            "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS": "0",
            "LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES": "0",
        }
    )
    if system_id == "streammuse_v1_standard" and reset_ack is not None:
        env.update(
            {
                "LEKAI_SESSION_ID": str(reset_ack["session_id"]),
                "LEKAI_SESSION_EPOCH": str(reset_ack["session_epoch"]),
                "LEKAI_EFFECTIVE_SEED": str(reset_ack["effective_seed"]),
            }
        )
    elif system_id == "streammuse_v2_prompt_continuation" and reset_ack is not None:
        env.update(
            {
                "LEKAI_PROMPT_REQUESTED_SEED": str(reset_ack["prompt_seed"]),
                "LEKAI_PROMPT_EFFECTIVE_SEED": str(reset_ack["prompt_seed"]),
                "LEKAI_CONTINUATION_REQUESTED_SEED": str(
                    reset_ack["continuation_effective_seed"]
                ),
                "LEKAI_CONTINUATION_EFFECTIVE_SEED": str(
                    reset_ack["continuation_effective_seed"]
                ),
                "LEKAI_PROMPT_SESSION_ID": str(reset_ack["session_id"]),
                "LEKAI_PROMPT_SESSION_EPOCH": str(reset_ack["session_epoch"]),
            }
        )
    return env


def _find_session(log_root: Path) -> Path:
    configs = sorted(log_root.rglob("session_config.json"))
    if len(configs) != 1:
        raise RuntimeError(
            f"expected exactly one session_config.json under {log_root}, "
            f"found {len(configs)}"
        )
    return configs[0].parent


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def validate_session(system_id: str, log_root: Path) -> dict[str, Any]:
    session_dir = _find_session(log_root)
    config = json.loads((session_dir / "session_config.json").read_text(encoding="utf-8"))
    expected_mode = (
        "standard"
        if system_id == "streammuse_v1_standard"
        else "prompt_continuation"
    )
    expected_config = {
        "tempo_bpm": float(BPM),
        "ticks_per_beat": TICKS_PER_BEAT,
        "beats_per_bar": BEATS_PER_BAR,
        "input_type": "midi_file",
        "output_type": "session",
        "session_artifact_tier": "debug",
        "continuation_mode": expected_mode,
        "generation_interval_ticks": GENERATION_INTERVAL_TICKS,
        "generation_length_frames": GENERATION_LENGTH_FRAMES,
        "count_in_beats": 0,
    }
    errors = [
        f"session_config {key}={config.get(key)!r}, expected {expected!r}"
        for key, expected in expected_config.items()
        if config.get(key) != expected
    ]
    trace_path = session_dir / "system_trace.jsonl"
    if not trace_path.is_file():
        errors.append("missing system_trace.jsonl")
        deadline_rows: list[dict[str, Any]] = []
    else:
        trace_rows = _read_jsonl(trace_path)
        deadline_rows = [
            row for row in trace_rows if row.get("record_type") == "frame_deadline"
        ]
        if any(row.get("schema_version") != 2 for row in deadline_rows):
            errors.append("frame deadlines are not all schema v2")
        ticks = [row.get("tick") for row in deadline_rows]
        expected_ticks = list(range(WINDOW_START_TICK, WINDOW_END_TICK))
        if ticks != expected_ticks:
            errors.append(
                "schema v2 frame deadlines do not continuously cover ticks 0..127"
            )
        expected_condition = (
            "standard"
            if system_id == "streammuse_v1_standard"
            else "prompt_continuation"
        )
        if any(row.get("condition") != expected_condition for row in deadline_rows):
            errors.append("frame deadline condition does not match system")
    if errors:
        raise RuntimeError("session contract rejected:\n- " + "\n- ".join(errors))
    return {
        "session_dir": str(session_dir.resolve()),
        "session_config": str((session_dir / "session_config.json").resolve()),
        "system_trace": str(trace_path.resolve()),
        "deadline_schema_version": 2,
        "deadline_count": len(deadline_rows),
        "deadline_start_tick": WINDOW_START_TICK,
        "deadline_end_tick_exclusive": WINDOW_END_TICK,
        "deadline_ticks_contiguous": True,
    }


def _safe_piece_name(piece_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", piece_id).strip("-.")
    return safe or "piece"


def _run_client(
    command: list[str],
    *,
    env: Mapping[str, str],
    trial_dir: Path,
    timeout_s: float,
) -> int:
    stdout_path = trial_dir / "client.stdout.log"
    stderr_path = trial_dir / "client.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=dict(env),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15.0)
            raise TimeoutError(f"client exceeded trial timeout {timeout_s}s")


def _trial_record(
    *,
    piece: CohortPiece,
    seed: int,
    system_id: str,
    trial_dir: Path,
    code: Mapping[str, Any],
    prompt_checkpoint: Mapping[str, Any],
    continuation_checkpoint: Mapping[str, Any],
    time_signature_index: int,
) -> dict[str, Any]:
    return {
        "piece_id": piece.piece_id,
        "seed": seed,
        "system_id": system_id,
        "session_dir": None,
        "run_status": "pending",
        "melody_input_path": str(piece.midi_path),
        "melody_input_sha256": piece.melody_input_sha256,
        "canonical_melody_input_sha256": piece.canonical_melody_input_sha256,
        "failure_reason": None,
        "trial_dir": str(trial_dir.resolve()),
        "requested_seeds": {
            "prompt_seed": seed if system_id.endswith("prompt_continuation") else None,
            "continuation_seed": seed,
        },
        "actual_seeds": None,
        "reset_ack": None,
        "runtime_before_trial": None,
        "runtime_after_trial": None,
        "checkpoint_conditioning": _checkpoint_conditioning(time_signature_index),
        "code_identity": dict(code),
        "checkpoint_identities": {
            "prompt": (
                dict(prompt_checkpoint)
                if system_id.endswith("prompt_continuation")
                else None
            ),
            "continuation": dict(continuation_checkpoint),
        },
    }


def run_trial(
    *,
    handle: ServerHandle,
    piece: CohortPiece,
    seed: int,
    trial_dir: Path,
    python_bin: str,
    timeout_s: float,
    code: Mapping[str, Any],
    prompt_checkpoint: Mapping[str, Any],
    continuation_checkpoint: Mapping[str, Any],
    time_signature_index: int,
    prompt_selection_mode: str = "rule_s",
    prompt_batch_candidates: int = PROMPT_CANDIDATES,
) -> dict[str, Any]:
    trial_dir.mkdir(parents=True, exist_ok=False)
    record = _trial_record(
        piece=piece,
        seed=seed,
        system_id=handle.system_id,
        trial_dir=trial_dir,
        code=code,
        prompt_checkpoint=prompt_checkpoint,
        continuation_checkpoint=continuation_checkpoint,
        time_signature_index=time_signature_index,
    )
    record["run_status"] = "running"
    write_json(trial_dir / "trial_manifest.json", record)
    log_root = trial_dir / "session_logs"
    try:
        reset_ack = reset_trial(handle.system_id, handle.base_url, seed)
        runtime_before = request_json(
            f"{handle.base_url}{_runtime_endpoint(handle.system_id)}", timeout=30.0
        )
        errors = runtime_contract_errors(
            handle.system_id,
            runtime_before,
            code=code,
            prompt_checkpoint=prompt_checkpoint,
            continuation_checkpoint=continuation_checkpoint,
            time_signature_index=time_signature_index,
            expected_seed=seed,
            reset_ack=reset_ack,
            prompt_selection_mode=prompt_selection_mode,
            prompt_batch_candidates=prompt_batch_candidates,
        )
        if errors:
            raise RuntimeError("pre-trial runtime rejected: " + "; ".join(errors))
        command = build_cli_command(
            handle.system_id,
            python_bin=python_bin,
            midi_path=piece.midi_path,
            log_root=log_root,
            base_url=handle.base_url,
        )
        client_env = build_client_environment(handle.system_id, reset_ack)
        record.update(
            {
                "reset_ack": reset_ack,
                "actual_seeds": {
                    "prompt_seed": reset_ack.get("prompt_seed"),
                    "continuation_seed": reset_ack.get(
                        "continuation_effective_seed",
                        reset_ack.get("effective_seed"),
                    ),
                },
                "runtime_before_trial": runtime_before,
                "client_command": command,
                "client_environment": _environment_record(client_env),
            }
        )
        write_json(trial_dir / "trial_manifest.json", record)
        returncode = _run_client(
            command,
            env=client_env,
            trial_dir=trial_dir,
            timeout_s=timeout_s,
        )
        record["client_returncode"] = returncode
        if returncode != 0:
            raise RuntimeError(f"StreamMUSE client exited with {returncode}")
        session_validation = validate_session(handle.system_id, log_root)
        record["session_dir"] = session_validation["session_dir"]
        runtime_after = request_json(
            f"{handle.base_url}{_runtime_endpoint(handle.system_id)}", timeout=30.0
        )
        errors = runtime_contract_errors(
            handle.system_id,
            runtime_after,
            code=code,
            prompt_checkpoint=prompt_checkpoint,
            continuation_checkpoint=continuation_checkpoint,
            time_signature_index=time_signature_index,
            expected_seed=seed,
            reset_ack=reset_ack,
            prompt_selection_mode=prompt_selection_mode,
            prompt_batch_candidates=prompt_batch_candidates,
        )
        if errors:
            raise RuntimeError("post-trial runtime rejected: " + "; ".join(errors))
        if file_sha256(piece.midi_path) != piece.melody_input_sha256:
            raise RuntimeError("MIDI input changed during the trial")
        record.update(
            {
                "runtime_after_trial": runtime_after,
                "session_validation": session_validation,
                "run_status": "complete",
                "failure_reason": None,
            }
        )
    except Exception as exc:
        try:
            record["session_dir"] = str(_find_session(log_root).resolve())
        except Exception:
            pass
        record["run_status"] = "failed"
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
    write_json(trial_dir / "trial_manifest.json", record)
    return record


def _dry_run_trial(
    *,
    system_id: str,
    piece: CohortPiece,
    seed: int,
    trial_dir: Path,
    python_bin: str,
    base_url: str,
    code: Mapping[str, Any],
    prompt_checkpoint: Mapping[str, Any],
    continuation_checkpoint: Mapping[str, Any],
    time_signature_index: int,
) -> dict[str, Any]:
    trial_dir.mkdir(parents=True, exist_ok=False)
    record = _trial_record(
        piece=piece,
        seed=seed,
        system_id=system_id,
        trial_dir=trial_dir,
        code=code,
        prompt_checkpoint=prompt_checkpoint,
        continuation_checkpoint=continuation_checkpoint,
        time_signature_index=time_signature_index,
    )
    record.update(
        {
            "run_status": "dry_run",
            "client_command": build_cli_command(
                system_id,
                python_bin=python_bin,
                midi_path=piece.midi_path,
                log_root=trial_dir / "session_logs",
                base_url=base_url,
            ),
            "failure_reason": None,
        }
    )
    write_json(trial_dir / "trial_manifest.json", record)
    return record


def _trial_key(record: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(record["piece_id"]),
        int(record["seed"]),
        str(record["system_id"]),
    )


def _replace_trial(manifest: dict[str, Any], record: Mapping[str, Any]) -> None:
    key = _trial_key(record)
    matches = [
        index
        for index, current in enumerate(manifest["trials"])
        if _trial_key(current) == key
    ]
    if len(matches) != 1:
        raise RuntimeError(f"planned trial key occurs {len(matches)} times: {key}")
    manifest["trials"][matches[0]] = dict(record)


def _mark_failed_system_trials(
    *,
    manifest: dict[str, Any],
    output_root: Path,
    system_id: str,
    pending_trials: Sequence[tuple[CohortPiece, int]],
    reason: str,
    code: Mapping[str, Any],
    prompt_checkpoint: Mapping[str, Any],
    continuation_checkpoint: Mapping[str, Any],
    time_signature_index: int,
) -> None:
    for piece, seed in pending_trials:
        trial_dir = (
            output_root
            / "trials"
            / system_id
            / _safe_piece_name(piece.piece_id)
            / f"seed_{seed}"
        )
        trial_dir.mkdir(parents=True, exist_ok=False)
        record = _trial_record(
            piece=piece,
            seed=seed,
            system_id=system_id,
            trial_dir=trial_dir,
            code=code,
            prompt_checkpoint=prompt_checkpoint,
            continuation_checkpoint=continuation_checkpoint,
            time_signature_index=time_signature_index,
        )
        record["run_status"] = "failed"
        record["failure_reason"] = reason
        write_json(trial_dir / "trial_manifest.json", record)
        _replace_trial(manifest, record)
        persist_run_manifests(output_root, manifest)


def _validate_matched_hashes(trials: Sequence[Mapping[str, Any]]) -> None:
    hashes_by_piece: dict[str, set[str]] = {}
    for trial in trials:
        hashes_by_piece.setdefault(str(trial["piece_id"]), set()).add(
            str(trial["melody_input_sha256"])
        )
    mismatches = [piece_id for piece_id, hashes in hashes_by_piece.items() if len(hashes) != 1]
    if mismatches:
        raise RuntimeError(
            "matched trials use different MIDI hashes for: " + ", ".join(mismatches)
        )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    prompt_candidate_count = effective_prompt_candidate_count(
        args.prompt_selection_mode, args.prompt_batch_candidates
    )
    pieces = load_cohort_manifest(args.cohort_manifest, args.smoke_limit)
    seeds = _parse_unique_ints(args.seeds, "--seeds")
    systems = _parse_systems(args.systems)
    if args.server_start_timeout_s <= 0 or args.trial_timeout_s <= 0:
        raise ValueError("timeouts must be positive")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    eval_manifest_path = output_root / "eval_manifest.csv"
    if manifest_path.exists() or eval_manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_root}")

    prompt_checkpoint = checkpoint_identity(args.prompt_checkpoint, "prompt")
    continuation_checkpoint = checkpoint_identity(
        args.continuation_checkpoint, "continuation"
    )
    code = code_identity()
    planned_trials = [
        _trial_record(
            piece=piece,
            seed=seed,
            system_id=system_id,
            trial_dir=(
                output_root
                / "trials"
                / system_id
                / _safe_piece_name(piece.piece_id)
                / f"seed_{seed}"
            ),
            code=code,
            prompt_checkpoint=prompt_checkpoint,
            continuation_checkpoint=continuation_checkpoint,
            time_signature_index=args.time_signature_index,
        )
        for system_id in systems
        for piece in pieces
        for seed in seeds
    ]
    manifest: dict[str, Any] = {
        "schema_version": "streammuse.matched_system_eval.run_manifest.v1",
        "created_at_unix_s": time.time(),
        "run_status": "running",
        "dry_run": bool(args.dry_run),
        "cohort_manifest": {
            "path": str(args.cohort_manifest.expanduser().resolve()),
            "sha256": file_sha256(args.cohort_manifest.expanduser().resolve()),
        },
        "systems": systems,
        "seeds": seeds,
        "piece_count": len(pieces),
        "evaluation_contract": {
            "bpm": BPM,
            "ticks_per_beat": TICKS_PER_BEAT,
            "beats_per_bar": BEATS_PER_BAR,
            "window_start_tick": WINDOW_START_TICK,
            "window_end_tick_exclusive": WINDOW_END_TICK,
            "generation_interval_ticks": GENERATION_INTERVAL_TICKS,
            "generation_length_frames": GENERATION_LENGTH_FRAMES,
            "late_recovery": False,
            "checkpoint_conditioning": _checkpoint_conditioning(
                args.time_signature_index
            ),
            "streammuse_v2_prompt": {
                "selection_mode": args.prompt_selection_mode,
                "candidate_count": prompt_candidate_count,
                "prompt_length_ticks": PROMPT_LENGTH_TICKS,
            },
            "sampling": dict(SAMPLING),
        },
        "code_identity": code,
        "checkpoint_identities": {
            "prompt": prompt_checkpoint,
            "continuation": continuation_checkpoint,
        },
        "trials": planned_trials,
    }
    persist_run_manifests(output_root, manifest)

    for system_index, system_id in enumerate(systems):
        if args.dry_run:
            base_url = f"http://127.0.0.1:{18000 + system_index}"
            for piece in pieces:
                for seed in seeds:
                    trial_dir = (
                        output_root
                        / "trials"
                        / system_id
                        / _safe_piece_name(piece.piece_id)
                        / f"seed_{seed}"
                    )
                    record = _dry_run_trial(
                        system_id=system_id,
                        piece=piece,
                        seed=seed,
                        trial_dir=trial_dir,
                        python_bin=args.python_bin,
                        base_url=base_url,
                        code=code,
                        prompt_checkpoint=prompt_checkpoint,
                        continuation_checkpoint=continuation_checkpoint,
                        time_signature_index=args.time_signature_index,
                    )
                    _replace_trial(manifest, record)
                    persist_run_manifests(output_root, manifest)
            continue

        handle: ServerHandle | None = None
        try:
            handle = start_server(
                system_id,
                output_root=output_root,
                python_bin=args.python_bin,
                gpu=args.gpu,
                timeout_s=args.server_start_timeout_s,
                code=code,
                prompt_checkpoint=prompt_checkpoint,
                continuation_checkpoint=continuation_checkpoint,
                time_signature_index=args.time_signature_index,
                prompt_selection_mode=args.prompt_selection_mode,
                prompt_batch_candidates=prompt_candidate_count,
            )
            for piece in pieces:
                for seed in seeds:
                    trial_dir = (
                        output_root
                        / "trials"
                        / system_id
                        / _safe_piece_name(piece.piece_id)
                        / f"seed_{seed}"
                    )
                    record = run_trial(
                        handle=handle,
                        piece=piece,
                        seed=seed,
                        trial_dir=trial_dir,
                        python_bin=args.python_bin,
                        timeout_s=args.trial_timeout_s,
                        code=code,
                        prompt_checkpoint=prompt_checkpoint,
                        continuation_checkpoint=continuation_checkpoint,
                        time_signature_index=args.time_signature_index,
                        prompt_selection_mode=args.prompt_selection_mode,
                        prompt_batch_candidates=prompt_candidate_count,
                    )
                    _replace_trial(manifest, record)
                    persist_run_manifests(output_root, manifest)
        except Exception as exc:
            pending_trials = [
                (piece, seed)
                for piece in pieces
                for seed in seeds
                if any(
                    _trial_key(row) == (piece.piece_id, seed, system_id)
                    and row["run_status"] == "pending"
                    for row in manifest["trials"]
                )
            ]
            _mark_failed_system_trials(
                manifest=manifest,
                output_root=output_root,
                system_id=system_id,
                pending_trials=pending_trials,
                reason=f"server_failure: {type(exc).__name__}: {exc}",
                code=code,
                prompt_checkpoint=prompt_checkpoint,
                continuation_checkpoint=continuation_checkpoint,
                time_signature_index=args.time_signature_index,
            )
        finally:
            if handle is not None:
                stop_server(handle)

    _validate_matched_hashes(manifest["trials"])
    statuses = [str(trial["run_status"]) for trial in manifest["trials"]]
    if args.dry_run:
        manifest["run_status"] = "dry_run"
    elif statuses and all(status == "complete" for status in statuses):
        manifest["run_status"] = "success"
    else:
        manifest["run_status"] = "completed_with_failures"
    manifest["completed_at_unix_s"] = time.time()
    manifest["summary"] = {
        status: statuses.count(status) for status in sorted(set(statuses))
    }
    persist_run_manifests(output_root, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_evaluation(args)
    print(json.dumps(manifest["summary"], sort_keys=True))
    return 0 if manifest["run_status"] in {"success", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
