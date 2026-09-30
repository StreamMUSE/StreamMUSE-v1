from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import wave
from pathlib import Path

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
