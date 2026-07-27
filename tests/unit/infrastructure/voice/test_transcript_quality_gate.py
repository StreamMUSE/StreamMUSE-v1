from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from streammuse.application.tasks.human_input import VoiceInputConfig
from streammuse.infrastructure.voice import FasterWhisperRecognizer, TranscriptionResult


_FAKE_SNAPSHOT = str(Path(__file__).parent.resolve())


@dataclass(frozen=True)
class _Segment:
    text: str
    id: int
    start: float = 0.0
    end: float = 0.1
    avg_logprob: float = -0.1
    no_speech_prob: float = 0.01
    compression_ratio: float = 1.0


class _Info:
    language = "en"
    language_probability = 0.99
    duration = 0.2
    duration_after_vad = 0.2


class _ScriptedModel:
    def __init__(self, outputs: list[tuple[_Segment, ...]]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[np.ndarray, dict[str, Any]]] = []
        self.completed_calls: list[int] = []

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, _Info]:
        call_index = len(self.calls)
        self.calls.append((audio, kwargs))
        output = self.outputs[call_index]

        def segments() -> Any:
            yield from output
            self.completed_calls.append(call_index)

        return segments(), _Info()


def _started_recognizer(
    *business_outputs: tuple[_Segment, ...],
) -> tuple[FasterWhisperRecognizer, _ScriptedModel]:
    model = _ScriptedModel(
        [
            (_Segment("", 0),),
            *business_outputs,
        ]
    )
    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(),
        model_factory=lambda *args, **kwargs: model,
        model_downloader=lambda *args, **kwargs: _FAKE_SNAPSHOT,
    )
    recognizer.start()
    return recognizer, model


def test_transcription_result_defaults_to_accepted_for_backwards_compatibility() -> None:
    result = TranscriptionResult(text="Zip", latency_ms=1.0)

    assert result.accepted is True
    assert result.rejection_reasons == ()


def test_pathological_repetition_and_compression_are_rejected_without_losing_raw_text() -> None:
    raw_text = " ".join(["Zap"] * 112)
    recognizer, model = _started_recognizer(
        (_Segment(f" {raw_text}", 0, compression_ratio=29.421),),
    )

    result = recognizer.transcribe(np.ones(1_600, dtype=np.float32))

    assert result.text == raw_text
    assert result.accepted is False
    assert set(result.rejection_reasons) == {
        "excessive_token_repetition",
        "excessive_compression_ratio",
    }
    quality_gate = result.diagnostics["quality_gate"]
    assert quality_gate["accepted"] is False
    assert set(quality_gate["reasons"]) == set(result.rejection_reasons)
    assert quality_gate["token_count"] == 112
    assert quality_gate["max_consecutive_token_repetitions"] == 112
    assert quality_gate["max_segment_compression_ratio"] == pytest.approx(29.421)
    assert model.completed_calls == [0, 1]


def test_normal_short_game_transcript_is_accepted() -> None:
    recognizer, _ = _started_recognizer(
        (
            _Segment(" Zip", 0, compression_ratio=1.1),
            _Segment(" zap.", 1, start=0.1, end=0.2, compression_ratio=1.2),
        ),
    )

    result = recognizer.transcribe(np.ones(1_600, dtype=np.float32))

    assert result.text == "Zip zap."
    assert result.accepted is True
    assert result.rejection_reasons == ()
    quality_gate = result.diagnostics["quality_gate"]
    assert quality_gate["accepted"] is True
    assert quality_gate["reasons"] == []
    assert quality_gate["token_count"] == 2
    assert quality_gate["max_consecutive_token_repetitions"] == 1
    assert quality_gate["max_segment_compression_ratio"] == pytest.approx(1.2)


def test_consecutive_transcriptions_are_isolated_and_disable_previous_text_conditioning() -> None:
    recognizer, model = _started_recognizer(
        (
            _Segment(" Zip", 0),
            _Segment(" zap.", 1, start=0.1, end=0.2),
        ),
        (_Segment(" Zop", 0),),
    )

    first = recognizer.transcribe(np.ones(800, dtype=np.float32))
    second = recognizer.transcribe(np.ones(800, dtype=np.float32))

    assert first.text == "Zip zap."
    assert second.text == "Zop"
    assert first.accepted is True
    assert second.accepted is True
    assert model.completed_calls == [0, 1, 2]
    assert model.calls[1][1]["condition_on_previous_text"] is False
    assert model.calls[2][1]["condition_on_previous_text"] is False


@pytest.mark.parametrize(
    ("character_count", "expected_reason"),
    [(512, None), (513, "transcript_too_long")],
)
def test_transcript_length_boundary(
    character_count: int,
    expected_reason: str | None,
) -> None:
    recognizer, _ = _started_recognizer(
        (_Segment("x" * character_count, 0),),
    )

    result = recognizer.transcribe(np.ones(1_600, dtype=np.float32))

    assert ("transcript_too_long" in result.rejection_reasons) is (
        expected_reason is not None
    )


@pytest.mark.parametrize(
    ("repeat_count", "rejected"),
    [(15, False), (16, True)],
)
def test_token_repetition_boundary(repeat_count: int, rejected: bool) -> None:
    recognizer, _ = _started_recognizer(
        (_Segment(" ".join(["Zap"] * repeat_count), 0),),
    )

    result = recognizer.transcribe(np.ones(1_600, dtype=np.float32))

    assert ("excessive_token_repetition" in result.rejection_reasons) is rejected


@pytest.mark.parametrize(
    ("compression_ratio", "rejected"),
    [(10.0, False), (10.0001, True)],
)
def test_compression_ratio_boundary(
    compression_ratio: float,
    rejected: bool,
) -> None:
    recognizer, _ = _started_recognizer(
        (_Segment("Zip", 0, compression_ratio=compression_ratio),),
    )

    result = recognizer.transcribe(np.ones(1_600, dtype=np.float32))

    assert ("excessive_compression_ratio" in result.rejection_reasons) is rejected


@pytest.mark.parametrize("compression_ratio", [float("nan"), float("inf"), "invalid"])
def test_invalid_compression_ratio_fails_closed_and_remains_json_safe(
    compression_ratio: object,
) -> None:
    recognizer, _ = _started_recognizer(
        (_Segment("Zip", 0, compression_ratio=compression_ratio),),  # type: ignore[arg-type]
    )

    result = recognizer.transcribe(np.ones(1_600, dtype=np.float32))

    assert result.rejection_reasons == ("invalid_compression_ratio",)
    assert result.diagnostics["quality_gate"]["invalid_compression_ratio_count"] == 1
    json.dumps(result.diagnostics, allow_nan=False)
