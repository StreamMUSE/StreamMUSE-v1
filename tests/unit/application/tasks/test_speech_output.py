from __future__ import annotations

import pytest

from streammuse.application.tasks.speech_output import (
    SilentSpeechOutput,
    SpeechOutputConfig,
)
from streammuse.domain.tasks import SpeechRequest


def test_speech_output_config_defaults_to_dependency_free_off_mode() -> None:
    config = SpeechOutputConfig()
    sink = SilentSpeechOutput()

    playback = sink.speak(
        SpeechRequest(turn_id=0, actor="llm", text="Zip", source_text="Zip")
    )

    assert config.mode == "off"
    assert sink.mode == "silent"
    assert sink.provenance == {"mode": "silent"}
    assert playback.status == "disabled"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rate": 0.0},
        {"rate": float("nan")},
        {"guard_ms": -1.0},
        {"cache_max_entries": 0},
        {"cache_max_bytes": 0},
        {"speaker_device": -1},
    ],
)
def test_speech_output_config_rejects_invalid_bounds(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SpeechOutputConfig(**kwargs)


def test_audio_end_requires_audio_output() -> None:
    with pytest.raises(ValueError, match="audio_end"):
        SpeechOutputConfig(llm_deadline_basis="audio_end")


def test_kokoro_requires_explicit_model_and_revision() -> None:
    with pytest.raises(ValueError, match="model and model revision"):
        SpeechOutputConfig(mode="audio", backend="kokoro")


def test_model_options_are_rejected_for_non_kokoro_backend() -> None:
    with pytest.raises(ValueError, match="kokoro"):
        SpeechOutputConfig(
            mode="audio",
            backend="system",
            model="model",
            model_revision="revision",
        )
