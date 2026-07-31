from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from streammuse.application.tasks import SpeechOutputConfig
from streammuse.infrastructure.voice import SpeechSynthesisError
from streammuse.infrastructure.voice.synthesizer import (
    CommandSpeechSynthesizer,
    KokoroSpeechSynthesizer,
)


class FakeProcess:
    def __init__(
        self,
        command,
        *,
        timeout: bool = False,
        **kwargs,
    ) -> None:
        self.command = list(command)
        self.kwargs = kwargs
        self.returncode = 0
        self.inputs: list[bytes | None] = []
        self.timeout = timeout
        self.communicate_count = 0
        self.killed = False
        self.terminated = False

    def communicate(self, input_value=None, timeout=None):
        self.communicate_count += 1
        self.inputs.append(input_value)
        if self.timeout and self.communicate_count == 1:
            raise subprocess.TimeoutExpired(self.command, timeout)
        if "-o" in self.command:
            path = Path(self.command[self.command.index("-o") + 1])
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(22_050)
                handle.writeframes(b"\x00\x00" * 16)
        if "sw_vers" in self.command[0]:
            return b"15.1\n", b""
        return b"", b""

    def kill(self) -> None:
        self.killed = True

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> int:
        return 0


class FakePopenFactory:
    def __init__(self, *, timeout: bool = False) -> None:
        self.timeout = timeout
        self.processes: list[FakeProcess] = []

    def __call__(self, command, **kwargs):
        process = FakeProcess(command, timeout=self.timeout, **kwargs)
        self.processes.append(process)
        return process


def test_say_uses_shell_false_stdin_and_cleans_temporary_file(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "streammuse.infrastructure.voice.synthesizer.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    factory = FakePopenFactory()
    synthesizer = CommandSpeechSynthesizer(
        SpeechOutputConfig(mode="audio", backend="system"),
        backend="say",
        popen_factory=factory,
    )
    synthesizer.start()

    audio = synthesizer.synthesize("-Zip Zap")
    process = next(item for item in factory.processes if "-o" in item.command)
    output_path = Path(process.command[process.command.index("-o") + 1])

    assert audio.sample_rate_hz == 22_050
    assert process.kwargs["shell"] is False
    assert process.inputs == [b"-Zip Zap"]
    assert "-Zip Zap" not in process.command
    assert "-f" in process.command
    assert "-" in process.command
    assert not output_path.exists()
    assert synthesizer.provenance["platform_version"] == "15.1"


def test_command_synthesis_timeout_kills_child_and_cleans_file(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "streammuse.infrastructure.voice.synthesizer.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    factory = FakePopenFactory(timeout=True)
    synthesizer = CommandSpeechSynthesizer(
        SpeechOutputConfig(
            mode="audio",
            backend="system",
            synthesis_timeout_s=0.01,
        ),
        backend="say",
        popen_factory=factory,
    )

    synthesizer.start()
    with pytest.raises(SpeechSynthesisError, match="timed out"):
        synthesizer.synthesize("Zip")

    assert factory.processes[-1].killed is True


def test_kokoro_loads_only_explicit_local_snapshot_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "voices").mkdir()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "kokoro-v1_0.pth").write_bytes(b"model")
    (tmp_path / "voices" / "af_heart.pt").write_bytes(b"voice")
    constructed: dict[str, object] = {}

    class FakeModel:
        def __init__(self, **kwargs) -> None:
            constructed["model_kwargs"] = kwargs

        def to(self, device: str):
            constructed["device"] = device
            return self

        def eval(self):
            return self

    class FakePipeline:
        def __init__(self, **kwargs) -> None:
            constructed["pipeline_kwargs"] = kwargs

    fake_tensor = object()
    monkeypatch.setitem(
        sys.modules,
        "kokoro",
        SimpleNamespace(KModel=FakeModel, KPipeline=FakePipeline),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(load=lambda *args, **kwargs: fake_tensor),
    )
    monkeypatch.setattr(
        "streammuse.infrastructure.voice.synthesizer.importlib.metadata.version",
        lambda name: "0.9.4",
    )
    synthesizer = KokoroSpeechSynthesizer(
        SpeechOutputConfig(
            mode="audio",
            backend="kokoro",
            model=str(tmp_path),
            model_revision="fixed-commit",
        )
    )

    synthesizer.start()

    assert constructed["model_kwargs"] == {
        "config": str(tmp_path / "config.json"),
        "model": str(tmp_path / "kokoro-v1_0.pth"),
    }
    pipeline_kwargs = constructed["pipeline_kwargs"]
    assert pipeline_kwargs["repo_id"] == str(tmp_path)
    assert pipeline_kwargs["model"].__class__ is FakeModel
    assert synthesizer._voice_tensor is fake_tensor
    assert synthesizer.provenance["model_revision"] == "fixed-commit"
    assert synthesizer.provenance["voice_asset_sha256"]
