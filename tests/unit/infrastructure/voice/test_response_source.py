from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from streammuse.application.tasks.human_input import VoiceInputConfig
from streammuse.domain.tasks import HumanResponseRequest, SpeechContext
from streammuse.infrastructure.voice import (
    CapturedUtterance,
    TranscriptionResult,
    VoiceHumanResponseSource,
    VoiceInfrastructureError,
)


def _utterance(*, speech: bool = True, deadline_expired: bool = False) -> CapturedUtterance:
    return CapturedUtterance(
        audio=np.ones(1_600, dtype=np.float32) if speech else np.zeros(0, dtype=np.float32),
        sample_rate_hz=16_000,
        capture_sample_rate_hz=48_000,
        endpoint_reason="trailing_silence" if speech else "start_timeout",
        deadline_expired=deadline_expired,
        wait_for_speech_ms=120.0,
        utterance_ms=500.0 if speech else 0.0,
        endpoint_silence_ms=300.0 if speech else 0.0,
        last_voiced_offset_ms=620.0 if speech else None,
        endpoint_detected_offset_ms=920.0,
    )


class FakeMicrophone:
    provenance = {"capture_sample_rate_hz": 48_000}

    def __init__(self, captured: CapturedUtterance) -> None:
        self.captured = captured
        self.started = 0
        self.closed = 0
        self.timeouts: list[float | None] = []

    def start(self) -> None:
        self.started += 1

    def capture(self, *, timeout_s: float | None) -> CapturedUtterance:
        self.timeouts.append(timeout_s)
        return self.captured

    def close(self) -> None:
        self.closed += 1


class FakeRecognizer:
    provenance = {"model": "tiny.en"}

    def __init__(self, result: TranscriptionResult) -> None:
        self.result = result
        self.started = 0
        self.closed = 0
        self.calls: list[tuple[np.ndarray, SpeechContext | None]] = []

    def start(self) -> None:
        self.started += 1

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        speech_context: SpeechContext | None,
    ) -> TranscriptionResult:
        self.calls.append((audio, speech_context))
        return self.result

    def close(self) -> None:
        self.closed += 1


class SteppedClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class MutableClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_start_is_idempotent_and_provenance_is_defensive() -> None:
    microphone = FakeMicrophone(_utterance())
    recognizer = FakeRecognizer(TranscriptionResult("Zip", 10.0))
    source = VoiceHumanResponseSource(
        VoiceInputConfig(),
        microphone=microphone,  # type: ignore[arg-type]
        recognizer=recognizer,  # type: ignore[arg-type]
    )

    source.start()
    source.start()
    provenance = source.provenance
    provenance["microphone"]["capture_sample_rate_hz"] = 1

    assert microphone.started == 1
    assert recognizer.started == 1
    assert source.provenance["microphone"]["capture_sample_rate_hz"] == 48_000


def test_start_failure_closes_both_components_and_preserves_original_error() -> None:
    microphone = FakeMicrophone(_utterance())
    recognizer = FakeRecognizer(TranscriptionResult("Zip", 1.0))
    startup_error = RuntimeError("warmup failed")

    def fail_start() -> None:
        recognizer.started += 1
        raise startup_error

    recognizer.start = fail_start  # type: ignore[method-assign]
    source = VoiceHumanResponseSource(
        VoiceInputConfig(),
        microphone=microphone,  # type: ignore[arg-type]
        recognizer=recognizer,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError) as exc_info:
        source.start()

    assert exc_info.value is startup_error
    assert microphone.closed == 1
    assert recognizer.closed == 1
    source.close()
    assert microphone.closed == 1
    assert recognizer.closed == 1


def test_read_response_composes_capture_asr_context_and_metadata() -> None:
    captured = _utterance()
    microphone = FakeMicrophone(captured)
    recognizer = FakeRecognizer(
        TranscriptionResult("Zip zap.", 165.0, {"hotwords": "Zip Zap Zop"})
    )
    source = VoiceHumanResponseSource(
        VoiceInputConfig(),
        microphone=microphone,  # type: ignore[arg-type]
        recognizer=recognizer,  # type: ignore[arg-type]
        now=SteppedClock([10.0, 10.93, 11.095]),
    )
    context = SpeechContext(initial_prompt="Zip Zap Zop", hotwords=("Zip", "Zap", "Zop"))
    source.start()

    response = source.read_response(
        HumanResponseRequest(turn_id=2, prompt="Speak", timeout_s=2.0, speech_context=context)
    )

    assert response.text == "Zip zap."
    assert response.status == "ok"
    assert response.deadline_expired is False
    assert microphone.timeouts == [2.0]
    assert recognizer.calls[0][0] is captured.audio
    assert recognizer.calls[0][1] == context
    assert response.metadata["raw_transcript"] == "Zip zap."
    assert response.metadata["last_voiced_offset_ms"] == 620.0
    assert response.metadata["asr_start_offset_ms"] == pytest.approx(930.0)
    assert response.metadata["asr_end_offset_ms"] == pytest.approx(1095.0)
    assert response.metadata["asr_latency_ms"] == 165.0
    assert response.latency_ms == pytest.approx(1095.0)
    assert response.metadata["artifact_persistence_ms"] == 0.0
    assert response.metadata["asr"]["hotwords"] == "Zip Zap Zop"


def test_empty_transcript_has_distinct_status() -> None:
    source = VoiceHumanResponseSource(
        VoiceInputConfig(),
        microphone=FakeMicrophone(_utterance()),  # type: ignore[arg-type]
        recognizer=FakeRecognizer(TranscriptionResult("", 10.0)),  # type: ignore[arg-type]
    )
    source.start()

    response = source.read_response(HumanResponseRequest(turn_id=0, prompt="Speak", timeout_s=None))

    assert response.status == "empty_transcript"
    assert response.text == ""


def test_quality_rejection_takes_precedence_over_an_empty_raw_transcript() -> None:
    source = VoiceHumanResponseSource(
        VoiceInputConfig(),
        microphone=FakeMicrophone(_utterance()),  # type: ignore[arg-type]
        recognizer=FakeRecognizer(  # type: ignore[arg-type]
            TranscriptionResult(
                "",
                10.0,
                {"quality_gate": {"accepted": False}},
                ("invalid_compression_ratio",),
            )
        ),
    )
    source.start()

    response = source.read_response(
        HumanResponseRequest(turn_id=0, prompt="Speak", timeout_s=None)
    )

    assert response.status == "rejected_transcript"
    assert response.text == ""
    assert response.metadata["transcript_rejection_reasons"] == [
        "invalid_compression_ratio"
    ]


def test_rejected_transcript_preserves_raw_text_without_becoming_a_game_answer() -> None:
    raw_text = ", ".join(["Zap"] * 112)
    result = TranscriptionResult(
        raw_text,
        10.0,
        {"quality_gate": {"accepted": False}},
        ("excessive_token_repetition", "excessive_compression_ratio"),
    )
    source = VoiceHumanResponseSource(
        VoiceInputConfig(),
        microphone=FakeMicrophone(_utterance()),  # type: ignore[arg-type]
        recognizer=FakeRecognizer(result),  # type: ignore[arg-type]
    )
    source.start()

    response = source.read_response(
        HumanResponseRequest(turn_id=0, prompt="Speak", timeout_s=None)
    )

    assert response.status == "rejected_transcript"
    assert response.text == raw_text
    assert response.metadata["raw_transcript"] == raw_text
    assert response.metadata["transcript_rejection_reasons"] == [
        "excessive_token_repetition",
        "excessive_compression_ratio",
    ]


def test_response_metadata_is_strictly_json_safe() -> None:
    recognizer = FakeRecognizer(
        TranscriptionResult(
            "Zip",
            10.0,
            {
                "probability": np.float32(0.5),
                "not_finite": np.float64(np.nan),
                "vector": np.array([1, 2], dtype=np.int64),
                "cache_path": Path("models/cache"),
            },
        )
    )
    source = VoiceHumanResponseSource(
        VoiceInputConfig(),
        microphone=FakeMicrophone(_utterance()),  # type: ignore[arg-type]
        recognizer=recognizer,  # type: ignore[arg-type]
    )
    source.start()

    response = source.read_response(HumanResponseRequest(turn_id=0, prompt="Speak", timeout_s=None))

    json.dumps(response.metadata, allow_nan=False)
    assert response.metadata["asr"]["probability"] == 0.5
    assert response.metadata["asr"]["not_finite"] is None
    assert response.metadata["asr"]["vector"] == [1, 2]
    assert response.metadata["asr"]["cache_path"] == "models/cache"


def test_no_speech_skips_asr_and_preserves_soft_safety_semantics() -> None:
    microphone = FakeMicrophone(_utterance(speech=False, deadline_expired=False))
    recognizer = FakeRecognizer(TranscriptionResult("should not run", 1.0))
    source = VoiceHumanResponseSource(
        VoiceInputConfig(),
        microphone=microphone,  # type: ignore[arg-type]
        recognizer=recognizer,  # type: ignore[arg-type]
    )
    source.start()

    response = source.read_response(HumanResponseRequest(turn_id=0, prompt="Speak", timeout_s=None))

    assert response.status == "no_speech"
    assert response.deadline_expired is False
    assert response.metadata["endpoint_reason"] == "start_timeout"
    assert recognizer.calls == []


def test_asr_completion_after_game_budget_marks_deadline_expired() -> None:
    source = VoiceHumanResponseSource(
        VoiceInputConfig(),
        microphone=FakeMicrophone(_utterance()),  # type: ignore[arg-type]
        recognizer=FakeRecognizer(TranscriptionResult("Zip", 200.0)),  # type: ignore[arg-type]
        now=SteppedClock([0.0, 0.9, 1.1]),
    )
    source.start()

    response = source.read_response(HumanResponseRequest(turn_id=0, prompt="Speak", timeout_s=1.0))

    assert response.deadline_expired is True


def test_opt_in_audio_persistence_uses_stable_relative_artifact_path(tmp_path: Path) -> None:
    writes: list[tuple[Path, np.ndarray, int]] = []

    def writer(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        writes.append((path, audio, sample_rate))

    source = VoiceHumanResponseSource(
        VoiceInputConfig(save_audio=True),
        artifact_root=tmp_path / "run" / "artifacts" / "turn",
        microphone=FakeMicrophone(_utterance()),  # type: ignore[arg-type]
        recognizer=FakeRecognizer(TranscriptionResult("Zip", 1.0)),  # type: ignore[arg-type]
        wave_writer=writer,
    )
    source.start()

    response = source.read_response(HumanResponseRequest(turn_id=3, prompt="Speak", timeout_s=None))

    assert writes[0][0].name == "0004_turn_0003_human.wav"
    assert writes[0][2] == 16_000
    assert response.metadata["audio_artifact"] == "artifacts/turn/0004_turn_0003_human.wav"


def test_audio_persistence_time_is_excluded_from_scoring_latency(tmp_path: Path) -> None:
    clock = MutableClock()

    def slow_writer(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        _ = path, audio, sample_rate
        clock.advance(5.0)

    source = VoiceHumanResponseSource(
        VoiceInputConfig(save_audio=True),
        artifact_root=tmp_path / "artifacts" / "turn",
        microphone=FakeMicrophone(_utterance()),  # type: ignore[arg-type]
        recognizer=FakeRecognizer(TranscriptionResult("Zip", 1.0)),  # type: ignore[arg-type]
        wave_writer=slow_writer,
        now=clock,
    )
    source.start()

    response = source.read_response(HumanResponseRequest(turn_id=0, prompt="Speak", timeout_s=2.0))

    assert response.latency_ms == pytest.approx(921.0)
    assert response.metadata["total_latency_ms"] == pytest.approx(921.0)
    assert response.metadata["artifact_persistence_ms"] == pytest.approx(5000.0)
    assert response.deadline_expired is False


def test_audio_artifact_path_uses_innermost_run_artifacts_directory(tmp_path: Path) -> None:
    source = VoiceHumanResponseSource(
        VoiceInputConfig(save_audio=True),
        artifact_root=tmp_path / "artifacts" / "parent" / "artifacts" / "turn",
        microphone=FakeMicrophone(_utterance()),  # type: ignore[arg-type]
        recognizer=FakeRecognizer(TranscriptionResult("Zip", 1.0)),  # type: ignore[arg-type]
        wave_writer=lambda path, audio, rate: None,
    )
    source.start()

    response = source.read_response(
        HumanResponseRequest(turn_id=0, prompt="Speak", timeout_s=None)
    )

    assert response.metadata["audio_artifact"] == "artifacts/turn/0001_turn_0000_human.wav"


def test_default_wave_writer_persists_mono_16bit_16khz_audio(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "turn"
    source = VoiceHumanResponseSource(
        VoiceInputConfig(save_audio=True),
        artifact_root=artifact_root,
        microphone=FakeMicrophone(_utterance()),  # type: ignore[arg-type]
        recognizer=FakeRecognizer(TranscriptionResult("Zip", 1.0)),  # type: ignore[arg-type]
    )
    source.start()

    source.read_response(HumanResponseRequest(turn_id=0, prompt="Speak", timeout_s=None))

    with wave.open(str(artifact_root / "0001_turn_0000_human.wav"), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16_000
        assert handle.getnframes() == 1_600


def test_audio_is_not_persisted_by_default(tmp_path: Path) -> None:
    writes: list[Any] = []
    source = VoiceHumanResponseSource(
        VoiceInputConfig(save_audio=False),
        artifact_root=tmp_path,
        microphone=FakeMicrophone(_utterance()),  # type: ignore[arg-type]
        recognizer=FakeRecognizer(TranscriptionResult("Zip", 1.0)),  # type: ignore[arg-type]
        wave_writer=lambda *args: writes.append(args),
    )
    source.start()

    response = source.read_response(HumanResponseRequest(turn_id=0, prompt="Speak", timeout_s=None))

    assert writes == []
    assert response.metadata["audio_artifact"] is None


def test_save_audio_requires_artifact_root() -> None:
    with pytest.raises(ValueError, match="artifact_root"):
        VoiceHumanResponseSource(
            VoiceInputConfig(save_audio=True),
            microphone=FakeMicrophone(_utterance()),  # type: ignore[arg-type]
            recognizer=FakeRecognizer(TranscriptionResult("Zip", 1.0)),  # type: ignore[arg-type]
        )


def test_close_attempts_both_components_and_is_idempotent() -> None:
    microphone = FakeMicrophone(_utterance())
    recognizer = FakeRecognizer(TranscriptionResult("Zip", 1.0))
    source = VoiceHumanResponseSource(
        VoiceInputConfig(),
        microphone=microphone,  # type: ignore[arg-type]
        recognizer=recognizer,  # type: ignore[arg-type]
    )
    source.start()

    source.close()
    source.close()

    assert microphone.closed == 1
    assert recognizer.closed == 1


def test_read_before_start_fails() -> None:
    source = VoiceHumanResponseSource(
        VoiceInputConfig(),
        microphone=FakeMicrophone(_utterance()),  # type: ignore[arg-type]
        recognizer=FakeRecognizer(TranscriptionResult("Zip", 1.0)),  # type: ignore[arg-type]
    )

    with pytest.raises(VoiceInfrastructureError, match="start"):
        source.read_response(HumanResponseRequest(turn_id=0, prompt="Speak", timeout_s=None))
