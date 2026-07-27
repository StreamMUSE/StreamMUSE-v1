from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streammuse.application.factories.human_input_factory import HumanInputFactory
from streammuse.application.tasks.human_input import (
    HumanInputConfig,
    TerminalHumanResponseSource,
    VoiceInputConfig,
)


class FakeTerminal:
    def write(self, text: str) -> None:
        _ = text

    def prompt(self, text: str) -> str:
        _ = text
        return ""

    def prompt_with_timeout(self, text: str, timeout_s: float):
        raise AssertionError((text, timeout_s))


def test_factory_builds_terminal_source_without_importing_voice() -> None:
    terminal = FakeTerminal()

    source = HumanInputFactory.create(HumanInputConfig(), terminal=terminal)

    assert isinstance(source, TerminalHumanResponseSource)


def test_factory_lazily_builds_voice_source_with_artifact_root(tmp_path: Path) -> None:
    config = HumanInputConfig(mode="voice", voice=VoiceInputConfig())
    sentinel = object()

    with patch("streammuse.infrastructure.voice.VoiceHumanResponseSource", return_value=sentinel) as source_type:
        source = HumanInputFactory.create(config, terminal=FakeTerminal(), artifact_root=tmp_path)

    assert source is sentinel
    source_type.assert_called_once_with(config.voice, artifact_root=tmp_path)
