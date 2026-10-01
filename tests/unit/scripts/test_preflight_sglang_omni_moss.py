from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from scripts import preflight_sglang_omni_moss as preflight


def _write_wav(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(struct.pack("<h", 1_000) * 2_400)
    return path.read_bytes()


def _inputs(tmp_path: Path) -> tuple[list[str], Path]:
    model = tmp_path / "models" / "moss-snapshot"
    model.mkdir(parents=True)
    config = tmp_path / "runtime" / "moss_tts.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model: pinned\n", encoding="utf-8")
    host_reference = tmp_path / "host" / "reference.wav"
    reference_bytes = _write_wav(host_reference)
    allowlist = tmp_path / "service-media"
    service_reference = allowlist / "reference.wav"
    _write_wav(service_reference)
    assert service_reference.read_bytes() == reference_bytes
    transcript = tmp_path / "host" / "reference.txt"
    transcript.write_text("reference words", encoding="utf-8")
    environment = tmp_path / "runtime" / "environment.lock"
    environment.write_text(
        "sglang-omni==0.1.4\nsglang==0.5.2\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "evidence" / "launch-manifest.json"
    return (
        [
            "--implementation-id",
            "sglang-moss-20260904-a",
            "--model-id",
            "OpenMOSS-Team/MOSS-TTS-v1.5",
            "--model-path",
            str(model),
            "--model-revision",
            "moss-snapshot-a1",
            "--config",
            str(config),
            "--reference-wav",
            str(host_reference),
            "--reference-text",
            str(transcript),
            "--service-reference-file",
            str(service_reference),
            "--service-reference-uri",
            service_reference.as_uri(),
            "--allowed-local-media-path",
            str(allowlist),
            "--runtime-environment-file",
            str(environment),
            "--sglang-omni-version",
            "0.1.4",
            "--sglang-omni-revision",
            "omni-commit-a1",
            "--sglang-version",
            "0.5.2",
            "--sglang-revision",
            "sglang-commit-b2",
            "--gpu",
            "3",
            "--output-manifest",
            str(manifest),
        ],
        manifest,
    )


def test_dry_run_validates_inputs_and_writes_reproducible_manifest(
    tmp_path: Path,
) -> None:
    argv, manifest_path = _inputs(tmp_path)

    status = preflight.main(
        [*argv, "--dry-run"],
        which=lambda _command: (_ for _ in ()).throw(
            AssertionError("dry-run must not inspect host executables")
        ),
    )

    assert status == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == preflight.MANIFEST_SCHEMA_VERSION
    assert manifest["preflight_status"] == "dry-run"
    assert manifest["qualification_status"] == "pending_h200_runtime_qualification"
    assert manifest["network"] == {
        "host": "127.0.0.1",
        "loopback_only": True,
        "port": 8030,
    }
    assert manifest["gpu"] == {"cuda_visible_devices": "3", "logical_device": 0}
    assert manifest["launch"]["shell"] is False
    assert manifest["launch"]["argv"][1:3] == ["serve", "--model-path"]
    assert manifest["reference"]["audio_sha256"] == hashlib.sha256(
        Path(argv[argv.index("--reference-wav") + 1]).read_bytes()
    ).hexdigest()
    serialized = manifest_path.read_text(encoding="utf-8")
    assert "reference words" not in serialized
    assert not list(manifest_path.parent.glob(".*.tmp"))


def test_launch_manifest_is_immutable_but_allows_identical_replay(
    tmp_path: Path,
) -> None:
    argv, manifest_path = _inputs(tmp_path)

    assert preflight.main([*argv, "--dry-run"]) == 0
    original = manifest_path.read_bytes()
    assert preflight.main([*argv, "--dry-run"]) == 0

    changed = list(argv)
    changed[changed.index("--gpu") + 1] = "4"
    assert preflight.main([*changed, "--dry-run"]) == 2
    assert manifest_path.read_bytes() == original
    assert not list(manifest_path.parent.glob(".*.tmp"))


def test_preflight_rejects_public_bind_and_reference_mismatch(tmp_path: Path) -> None:
    argv, manifest_path = _inputs(tmp_path)
    public = [*argv, "--host", "0.0.0.0", "--dry-run"]
    assert preflight.main(public) == 2
    assert not manifest_path.exists()

    argv, manifest_path = _inputs(tmp_path / "mismatch")
    service_reference = Path(argv[argv.index("--service-reference-file") + 1])
    _write_wav(service_reference)
    data = bytearray(service_reference.read_bytes())
    data[-1] ^= 1
    service_reference.write_bytes(data)
    assert preflight.main([*argv, "--dry-run"]) == 2
    assert not manifest_path.exists()


def test_preflight_fails_when_required_build_tool_is_missing(tmp_path: Path) -> None:
    argv, manifest_path = _inputs(tmp_path)

    status = preflight.main(
        argv,
        which=lambda command: None if command == "ninja" else f"/tools/{command}",
    )

    assert status == 2
    assert not manifest_path.exists()


def test_validated_launch_uses_exact_argv_and_exec_environment(
    tmp_path: Path,
) -> None:
    argv, manifest_path = _inputs(tmp_path)
    exec_calls: list[tuple[str, list[str], dict[str, str]]] = []

    def run(command, **_kwargs):
        output = b"tool version 1\n"
        if command[-2:] == ["serve", "--help"]:
            output = (
                b"--model-path --config --allowed-local-media-path --host --port\n"
            )
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr=b"")

    def execvpe(executable, command, environment):
        exec_calls.append((executable, command, dict(environment)))

    status = preflight.main(
        [*argv, "--launch"],
        which=lambda command: f"/tools/{command}",
        run=run,
        execvpe=execvpe,
    )

    assert status == 0
    assert len(exec_calls) == 1
    executable, command, environment = exec_calls[0]
    assert executable == "/tools/sgl-omni"
    assert command[0] == executable
    assert command[-4:] == ["--host", "127.0.0.1", "--port", "8030"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["preflight_status"] == "validated"
    assert manifest["tools"]["ninja"]["status"] == "validated"
    assert manifest["tools"]["sgl-omni"]["required_serve_flags"] == [
        "--model-path",
        "--config",
        "--allowed-local-media-path",
        "--host",
        "--port",
    ]


def _patched_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "patched-site"
    (root / "sglang_omni").mkdir(parents=True)
    (root / "sglang_omni" / "__init__.py").write_text("", encoding="utf-8")
    patch = tmp_path / "runtime.patch"
    patch.write_text("--- a/sglang_omni/x.py\n+++ b/sglang_omni/x.py\n", encoding="utf-8")
    return root, patch


def _validated_run(git_returncode: int, git_calls: list[list[str]]):
    def run(command, **_kwargs):
        if "apply" in command:
            git_calls.append(list(command))
            return subprocess.CompletedProcess(command, git_returncode, stdout=b"", stderr=b"")
        output = b"tool version 1\n"
        if command[-2:] == ["serve", "--help"]:
            output = b"--model-path --config --allowed-local-media-path --host --port\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr=b"")

    return run


def test_runtime_patch_is_verified_recorded_and_put_first_on_pythonpath(
    tmp_path: Path, monkeypatch
) -> None:
    argv, manifest_path = _inputs(tmp_path)
    root, patch = _patched_root(tmp_path)
    git_calls: list[list[str]] = []
    exec_calls: list[dict[str, str]] = []
    monkeypatch.setenv("PYTHONPATH", "/existing")

    status = preflight.main(
        [*argv, "--runtime-patch-file", str(patch), "--runtime-patch-root", str(root), "--launch"],
        which=lambda command: f"/tools/{command}",
        run=_validated_run(0, git_calls),
        execvpe=lambda _executable, _command, environment: exec_calls.append(dict(environment)),
    )

    assert status == 0
    assert git_calls == [
        ["/tools/git", "-C", str(root), "apply", "--reverse", "--check", "-p1", str(patch)]
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime"]["patch_sha256"] == hashlib.sha256(patch.read_bytes()).hexdigest()
    assert manifest["runtime"]["patch_root"] == str(root)
    assert manifest["tools"]["runtime-patch"]["status"] == "applied"
    assert exec_calls[0]["PYTHONPATH"] == f"{root}:/existing"


def test_runtime_patch_that_is_not_applied_fails_preflight(tmp_path: Path) -> None:
    argv, manifest_path = _inputs(tmp_path)
    root, patch = _patched_root(tmp_path)

    status = preflight.main(
        [*argv, "--runtime-patch-file", str(patch), "--runtime-patch-root", str(root)],
        which=lambda command: f"/tools/{command}",
        run=_validated_run(1, []),
    )

    assert status == 2
    assert not manifest_path.exists()


def test_runtime_patch_flags_must_come_together_and_absent_keeps_manifest(
    tmp_path: Path,
) -> None:
    argv, manifest_path = _inputs(tmp_path)
    _, patch = _patched_root(tmp_path)

    assert preflight.main([*argv, "--runtime-patch-file", str(patch), "--dry-run"]) == 2
    assert preflight.main([*argv, "--dry-run"]) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "patch_sha256" not in manifest["runtime"]
    assert "runtime-patch" not in manifest["tools"]


def test_repository_patch_fails_the_check_on_an_unpatched_tree(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is unavailable")
    patch = Path(preflight.__file__).resolve().parents[1] / "patches" / (
        "sglang-omni-0.1.4-af3ab61-moss-latency.patch"
    )
    root = tmp_path / "site"
    (root / "sglang_omni").mkdir(parents=True)
    completed = subprocess.run(
        [git, "-C", str(root), "apply", "--reverse", "--check", "-p1", str(patch)],
        capture_output=True,
    )
    assert completed.returncode != 0
