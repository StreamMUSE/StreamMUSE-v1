#!/usr/bin/env python3
"""Validate and optionally launch one pinned SGLang-Omni MOSS service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from streammuse.infrastructure.rap.moss_tts import (
    MossInvalidOutput,
    read_valid_mono_wav_bytes,
)


MANIFEST_SCHEMA_VERSION = "streammuse.sglang_omni_launch.v1"
_IMPLEMENTATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PINNED_FORBIDDEN = {"main", "master", "latest", "unknown", "unavailable"}
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_MAX_CAPTURE_BYTES = 256 * 1024
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_REFERENCE_BYTES = 64 * 1024 * 1024
_MAX_TEXT_BYTES = 64 * 1024
_REQUIRED_SERVE_FLAGS = (
    "--model-path",
    "--config",
    "--allowed-local-media-path",
    "--host",
    "--port",
)


class PreflightError(RuntimeError):
    """A bounded, user-actionable preflight failure."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a pinned SGLang-Omni MOSS launch"
    )
    parser.add_argument("--implementation-id", required=True)
    parser.add_argument("--sgl-omni-bin", default="sgl-omni")
    parser.add_argument("--ninja-bin", default="ninja")
    parser.add_argument("--cxx-bin", default="c++")
    parser.add_argument("--nvcc-bin", default="nvcc")
    parser.add_argument("--git-bin", default="git")
    parser.add_argument(
        "--runtime-patch-file",
        help="patch applied to the pinned sglang_omni package (paths a/sglang_omni/...)",
    )
    parser.add_argument(
        "--runtime-patch-root",
        help="directory holding the patched sglang_omni package; it is put first on "
        "PYTHONPATH for the launched service",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-wav", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--service-reference-file", required=True)
    parser.add_argument("--service-reference-uri", required=True)
    parser.add_argument("--allowed-local-media-path", required=True)
    parser.add_argument("--runtime-environment-file", required=True)
    parser.add_argument("--sglang-omni-version", required=True)
    parser.add_argument("--sglang-omni-revision", required=True)
    parser.add_argument("--sglang-version", required=True)
    parser.add_argument("--sglang-revision", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launch", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    execvpe: Callable[[str, list[str], Mapping[str, str]], Any] = os.execvpe,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.dry_run and args.launch:
            raise PreflightError("--dry-run and --launch cannot be combined")
        inputs = validate_inputs(args)
        tools = (
            dry_run_tool_evidence(args)
            if args.dry_run
            else qualify_tools(args, which=which, run=run)
        )
        if inputs["runtime_patch_sha256"] is not None:
            tools["runtime-patch"] = (
                {"status": "not_executed"}
                if args.dry_run
                else qualify_runtime_patch(args, which=which, run=run)
            )
        launch_argv = build_launch_argv(args, tools["sgl-omni"]["path"])
        manifest = build_manifest(
            args,
            inputs=inputs,
            tools=tools,
            launch_argv=launch_argv,
        )
        write_manifest(Path(args.output_manifest), manifest)
    except (OSError, PreflightError, ValueError) as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2

    print(shlex.join(launch_argv))
    if args.launch:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
        if inputs["runtime_patch_root"] is not None:
            environment["PYTHONPATH"] = os.pathsep.join(
                item
                for item in (inputs["runtime_patch_root"], environment.get("PYTHONPATH"))
                if item
            )
        execvpe(launch_argv[0], launch_argv, environment)
    return 0


def validate_inputs(args: argparse.Namespace) -> dict[str, object]:
    if not _IMPLEMENTATION_ID.fullmatch(args.implementation_id):
        raise PreflightError("implementation id is invalid")
    for value, name in (
        (args.model_id, "model id"),
        (args.model_revision, "model revision"),
        (args.sglang_omni_version, "SGLang-Omni version"),
        (args.sglang_omni_revision, "SGLang-Omni revision"),
        (args.sglang_version, "SGLang version"),
        (args.sglang_revision, "SGLang revision"),
    ):
        _require_pin(value, name)
    if args.host not in _LOOPBACK_HOSTS:
        raise PreflightError("SGLang-Omni must bind to a loopback host")
    if isinstance(args.port, bool) or not 1 <= args.port <= 65535:
        raise PreflightError("SGLang-Omni port is invalid")
    if not re.fullmatch(r"[0-9]+", args.gpu):
        raise PreflightError("--gpu must identify exactly one CUDA-visible device")

    model_path = _absolute_directory(args.model_path, "model path")
    config_path = _absolute_file(args.config, "runtime config")
    reference_path = _absolute_file(args.reference_wav, "reference WAV")
    text_path = _absolute_file(args.reference_text, "reference transcript")
    service_reference = _absolute_file(
        args.service_reference_file, "service reference WAV"
    )
    allowlist = _absolute_directory(
        args.allowed_local_media_path, "local media allowlist"
    )
    environment_path = _absolute_file(
        args.runtime_environment_file, "runtime environment evidence"
    )
    output_path = Path(args.output_manifest)
    if not output_path.is_absolute():
        raise PreflightError("output manifest path must be absolute")

    try:
        service_reference.resolve().relative_to(allowlist.resolve())
    except ValueError as exc:
        raise PreflightError(
            "service reference WAV is outside the local media allowlist"
        ) from exc
    uri_path = _local_file_uri_path(args.service_reference_uri)
    if uri_path.resolve() != service_reference.resolve():
        raise PreflightError(
            "service reference URI does not resolve to service reference file"
        )

    config_bytes = _bounded_read(config_path, _MAX_CONFIG_BYTES, "runtime config")
    reference_bytes = _bounded_read(
        reference_path, _MAX_REFERENCE_BYTES, "reference WAV"
    )
    service_reference_bytes = _bounded_read(
        service_reference,
        _MAX_REFERENCE_BYTES,
        "service reference WAV",
    )
    if reference_bytes != service_reference_bytes:
        raise PreflightError("host and service reference WAV bytes differ")
    try:
        sample_rate_hz, samples = read_valid_mono_wav_bytes(reference_bytes)
    except MossInvalidOutput as exc:
        raise PreflightError("reference WAV failed audio validation") from exc
    duration_seconds = samples.shape[0] / sample_rate_hz
    if duration_seconds > 120.0:
        raise PreflightError("reference WAV exceeds 120 seconds")

    text_bytes = _bounded_read(text_path, _MAX_TEXT_BYTES, "reference transcript")
    try:
        transcript = text_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreflightError("reference transcript must be UTF-8") from exc
    if transcript != transcript.strip():
        raise PreflightError("reference transcript has surrounding whitespace")
    environment_bytes = _bounded_read(
        environment_path,
        16 * 1024 * 1024,
        "runtime environment evidence",
    )
    if (args.runtime_patch_file is None) != (args.runtime_patch_root is None):
        raise PreflightError(
            "--runtime-patch-file and --runtime-patch-root must be given together"
        )
    patch_sha256 = patch_path = patch_root = None
    if args.runtime_patch_file is not None:
        patch_file = _absolute_file(args.runtime_patch_file, "runtime patch")
        patch_sha256 = _sha256(_bounded_read(patch_file, 1024 * 1024, "runtime patch"))
        root = _absolute_directory(args.runtime_patch_root, "runtime patch root")
        if not (root / "sglang_omni" / "__init__.py").is_file():
            raise PreflightError("runtime patch root does not hold a sglang_omni package")
        patch_path, patch_root = str(patch_file), str(root)
    return {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_bytes),
        "reference_audio_sha256": _sha256(reference_bytes),
        "reference_text_sha256": _sha256(text_bytes),
        "runtime_environment_path": str(environment_path),
        "runtime_environment_sha256": _sha256(environment_bytes),
        "reference_sample_rate_hz": sample_rate_hz,
        "reference_frame_count": int(samples.shape[0]),
        "reference_duration_seconds": duration_seconds,
        "runtime_patch_path": patch_path,
        "runtime_patch_sha256": patch_sha256,
        "runtime_patch_root": patch_root,
    }


def qualify_tools(
    args: argparse.Namespace,
    *,
    which: Callable[[str], str | None],
    run: Callable[..., subprocess.CompletedProcess[bytes]],
) -> dict[str, dict[str, object]]:
    requested = {
        "sgl-omni": args.sgl_omni_bin,
        "ninja": args.ninja_bin,
        "cxx": args.cxx_bin,
        "nvcc": args.nvcc_bin,
    }
    resolved: dict[str, str] = {}
    for name, command in requested.items():
        path = which(command)
        if path is None:
            raise PreflightError(f"required executable is missing: {name}")
        resolved[name] = path

    evidence: dict[str, dict[str, object]] = {}
    for name, path in resolved.items():
        output = _run_bounded([path, "--version"], run=run)
        evidence[name] = {
            "path": path,
            "version_output_sha256": _sha256(output),
            "version_summary": _safe_summary(output),
            "status": "validated",
        }
    help_output = _run_bounded(
        [resolved["sgl-omni"], "serve", "--help"],
        run=run,
    )
    help_text = help_output.decode("utf-8", errors="replace")
    missing_flags = [flag for flag in _REQUIRED_SERVE_FLAGS if flag not in help_text]
    if missing_flags:
        raise PreflightError(
            "pinned sgl-omni serve CLI lacks required flags: "
            + ", ".join(missing_flags)
        )
    evidence["sgl-omni"]["serve_help_sha256"] = _sha256(help_output)
    evidence["sgl-omni"]["required_serve_flags"] = list(_REQUIRED_SERVE_FLAGS)
    return evidence


def qualify_runtime_patch(
    args: argparse.Namespace,
    *,
    which: Callable[[str], str | None],
    run: Callable[..., subprocess.CompletedProcess[bytes]],
) -> dict[str, object]:
    """Prove the patch is applied in the root the service will import from."""
    git = which(args.git_bin)
    if git is None:
        raise PreflightError("required executable is missing: git")
    try:
        _run_bounded(
            [
                git,
                "-C",
                str(Path(args.runtime_patch_root)),
                "apply",
                "--reverse",
                "--check",
                "-p1",
                str(Path(args.runtime_patch_file)),
            ],
            run=run,
        )
    except PreflightError as exc:
        raise PreflightError(
            "runtime patch is not applied under --runtime-patch-root"
        ) from exc
    return {"status": "applied", "git": git}


def dry_run_tool_evidence(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    return {
        "sgl-omni": {"path": args.sgl_omni_bin, "status": "not_executed"},
        "ninja": {"path": args.ninja_bin, "status": "not_executed"},
        "cxx": {"path": args.cxx_bin, "status": "not_executed"},
        "nvcc": {"path": args.nvcc_bin, "status": "not_executed"},
    }


def build_launch_argv(args: argparse.Namespace, executable: object) -> list[str]:
    if not isinstance(executable, str) or not executable:
        raise PreflightError("resolved sgl-omni executable is invalid")
    return [
        executable,
        "serve",
        "--model-path",
        str(Path(args.model_path)),
        "--config",
        str(Path(args.config)),
        "--allowed-local-media-path",
        str(Path(args.allowed_local_media_path)),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]


def build_manifest(
    args: argparse.Namespace,
    *,
    inputs: Mapping[str, object],
    tools: Mapping[str, object],
    launch_argv: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "implementation_id": args.implementation_id,
        "preflight_status": "dry-run" if args.dry_run else "validated",
        "qualification_status": "pending_h200_runtime_qualification",
        "model": {
            "id": args.model_id,
            "revision": args.model_revision,
            "path": inputs["model_path"],
        },
        "runtime": {
            "sglang_omni_version": args.sglang_omni_version,
            "sglang_omni_revision": args.sglang_omni_revision,
            "sglang_version": args.sglang_version,
            "sglang_revision": args.sglang_revision,
            "environment_path": inputs["runtime_environment_path"],
            "environment_sha256": inputs["runtime_environment_sha256"],
            "config_path": inputs["config_path"],
            "config_sha256": inputs["config_sha256"],
            **(
                {
                    "patch_path": inputs["runtime_patch_path"],
                    "patch_sha256": inputs["runtime_patch_sha256"],
                    "patch_root": inputs["runtime_patch_root"],
                }
                if inputs["runtime_patch_sha256"] is not None
                else {}
            ),
        },
        "reference": {
            "audio_sha256": inputs["reference_audio_sha256"],
            "text_sha256": inputs["reference_text_sha256"],
            "sample_rate_hz": inputs["reference_sample_rate_hz"],
            "frame_count": inputs["reference_frame_count"],
            "duration_seconds": inputs["reference_duration_seconds"],
            "service_uri": args.service_reference_uri,
        },
        "network": {"host": args.host, "port": args.port, "loopback_only": True},
        "gpu": {"cuda_visible_devices": args.gpu, "logical_device": 0},
        "launch": {"argv": list(launch_argv), "shell": False},
        "tools": dict(tools),
        "capabilities": [
            {
                "id": "non_streaming_wav",
                "required": True,
                "status": "pending_h200_probe",
            },
            {
                "id": "reference_audio_and_text",
                "required": True,
                "status": "pending_h200_probe",
            },
            {
                "id": "generation_knobs",
                "required": True,
                "status": "pending_h200_probe",
            },
            {
                "id": "disconnect_deadline_recovery",
                "required": True,
                "status": "pending_h200_probe",
            },
            {
                "id": "request_profiler_correlation",
                "required": False,
                "status": "pending_h200_probe",
            },
            {
                "id": "raw_pcm_streaming",
                "required": False,
                "status": "phase_9_only",
            },
        ],
    }


def write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    """Publish a launch manifest once, allowing only byte-identical replays."""
    data = json.dumps(
        manifest,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return
        raise PreflightError(
            f"launch manifest already exists with different bytes: {path}"
        )
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if path.read_bytes() != data:
                raise PreflightError(
                    f"launch manifest already exists with different bytes: {path}"
                ) from exc
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _require_pin(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 512
        or value.strip().lower() in _PINNED_FORBIDDEN
        or any(ord(character) < 32 for character in value)
    ):
        raise PreflightError(f"{name} must be an immutable pin")


def _absolute_file(value: object, name: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PreflightError(f"{name} must be an absolute regular file")
    return path


def _absolute_directory(value: object, name: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise PreflightError(f"{name} must be an absolute directory")
    return path


def _local_file_uri_path(value: object) -> Path:
    if not isinstance(value, str) or len(value) > 2048:
        raise PreflightError("service reference URI is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise PreflightError("service reference URI must be an absolute local file URI")
    return Path(unquote(parsed.path))


def _bounded_read(path: Path, limit: int, name: str) -> bytes:
    size = path.stat().st_size
    if size <= 0 or size > limit:
        raise PreflightError(f"{name} size is invalid")
    data = path.read_bytes()
    if len(data) != size:
        raise PreflightError(f"{name} changed while it was being read")
    return data


def _run_bounded(
    argv: list[str],
    *,
    run: Callable[..., subprocess.CompletedProcess[bytes]],
) -> bytes:
    try:
        completed = run(
            argv,
            check=False,
            capture_output=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError(f"tool probe failed: {Path(argv[0]).name}") from exc
    output = bytes(completed.stdout or b"") + bytes(completed.stderr or b"")
    if completed.returncode != 0:
        raise PreflightError(f"tool probe returned non-zero: {Path(argv[0]).name}")
    if len(output) > _MAX_CAPTURE_BYTES:
        raise PreflightError(f"tool probe output is too large: {Path(argv[0]).name}")
    return output


def _safe_summary(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace").strip().splitlines()
    summary = " ".join(text[0].split())[:256] if text else "empty"
    if not summary or "://" in summary or summary.startswith(("/", "~", "\\")):
        return "redacted"
    return summary


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
