from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from streammuse.application.tasks.human_input import (
    HumanInputConfig,
    TerminalHumanResponseSource,
    TimedPromptResult,
    VoiceInputConfig,
)
from streammuse.domain.tasks import HumanResponseRequest


class FakeTerminal:
    def __init__(self) -> None:
        self.prompt_result = "Zip"
        self.timed_result = TimedPromptResult(text="Zap")
        self.prompts: list[str] = []
        self.timeout_prompts: list[tuple[str, float]] = []

    def write(self, text: str) -> None:
        _ = text

    def prompt(self, text: str) -> str:
        self.prompts.append(text)
        return self.prompt_result

    def prompt_with_timeout(self, text: str, timeout_s: float) -> TimedPromptResult:
        self.timeout_prompts.append((text, timeout_s))
        return self.timed_result


def test_human_input_config_requires_configuration_matching_the_mode() -> None:
    assert HumanInputConfig() == HumanInputConfig(mode="terminal", voice=None)
    voice = VoiceInputConfig()
    assert HumanInputConfig(mode="voice", voice=voice).voice is voice

    with pytest.raises(ValueError, match="terminal mode does not accept"):
        HumanInputConfig(mode="terminal", voice=voice)
    with pytest.raises(ValueError, match="voice mode requires"):
        HumanInputConfig(mode="voice")
    with pytest.raises(ValueError, match="mode must be"):
        HumanInputConfig(mode="other")  # type: ignore[arg-type]


def test_voice_input_config_defaults_are_explicit_and_frozen() -> None:
    config = VoiceInputConfig()

    assert config.model == "tiny.en"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.local_files_only is False
    assert config.save_audio is False
    assert config.max_utterance_ms == 5000.0
    assert config.vad_aggressiveness == 2
    with pytest.raises(FrozenInstanceError):
        config.model = "base.en"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"model": ""}, ValueError, "model must not be empty"),
        ({"microphone_device": -1}, ValueError, "index must be >= 0"),
        ({"microphone_device": True}, TypeError, "device name"),
        ({"local_files_only": 1}, TypeError, "must be a boolean"),
        ({"start_timeout_ms": 0}, ValueError, "must be > 0"),
        ({"end_silence_ms": float("inf")}, ValueError, "must be finite"),
        ({"max_utterance_ms": 100, "end_silence_ms": 101}, ValueError, "must be <="),
        ({"max_utterance_ms": 100, "pre_roll_ms": 101}, ValueError, "must be <="),
        ({"vad_aggressiveness": 4}, ValueError, "between 0 and 3"),
        ({"queue_max_chunks": 0}, ValueError, "must be > 0"),
    ],
)
def test_voice_input_config_rejects_invalid_values(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        VoiceInputConfig(**kwargs)  # type: ignore[arg-type]


def test_terminal_response_source_uses_plain_prompt_without_a_timeout() -> None:
    terminal = FakeTerminal()
    source = TerminalHumanResponseSource(terminal)

    source.start()
    response = source.read_response(HumanResponseRequest(turn_id=7, prompt="You > ", timeout_s=None))
    source.close()
    source.close()

    assert response.text == "Zip"
    assert response.status == "ok"
    assert response.deadline_expired is False
    assert response.metadata == {"mode": "terminal", "status": "ok"}
    assert terminal.prompts == ["You > "]
    assert terminal.timeout_prompts == []


def test_terminal_response_source_maps_timed_prompt_expiration_to_deadline() -> None:
    terminal = FakeTerminal()
    terminal.timed_result = TimedPromptResult(text="", timed_out=True)
    source = TerminalHumanResponseSource(terminal)

    response = source.read_response(HumanResponseRequest(turn_id=8, prompt="You > ", timeout_s=0.25))

    assert response.text == ""
    assert response.deadline_expired is True
    assert terminal.timeout_prompts == [("You > ", 0.25)]


def test_terminal_response_source_provenance_is_a_defensive_snapshot() -> None:
    source = TerminalHumanResponseSource(FakeTerminal())

    provenance = source.provenance
    provenance["mode"] = "changed"

    assert source.mode == "terminal"
    assert source.provenance == {"mode": "terminal"}
