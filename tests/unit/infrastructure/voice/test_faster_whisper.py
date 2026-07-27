from __future__ import annotations

import builtins
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from streammuse.application.tasks.human_input import VoiceInputConfig
from streammuse.domain.tasks import SpeechContext
from streammuse.infrastructure.voice import (
    FasterWhisperRecognizer,
    SpeechRecognitionError,
    VoiceDependencyError,
)
from streammuse.infrastructure.voice import faster_whisper as faster_whisper_module


FAKE_SNAPSHOT = str(Path(__file__).parent.resolve())


def _fake_model_downloader(*args: Any, **kwargs: Any) -> str:
    del args, kwargs
    return FAKE_SNAPSHOT


@dataclass
class Segment:
    text: str
    id: int
    start: float = 0.0
    end: float = 0.1
    avg_logprob: float = -0.1
    no_speech_prob: float = 0.01
    compression_ratio: float = 1.0


class Info:
    language = "en"
    language_probability = 0.99
    duration = 0.2
    duration_after_vad = 0.2


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, dict[str, Any]]] = []
        self.consumed = 0
        self.closed = 0

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, Info]:
        self.calls.append((audio, kwargs))
        call_index = len(self.calls)

        def segments() -> Any:
            if call_index == 1:
                self.consumed += 1
                yield Segment("", 0)
            else:
                self.consumed += 1
                yield Segment(" Zip", 0)
                self.consumed += 1
                yield Segment(" zap.", 1, start=0.1, end=0.2)

        return segments(), Info()

    def close(self) -> None:
        self.closed += 1


def test_model_is_loaded_and_warmed_once_with_explicit_cpu_int8_defaults() -> None:
    model = FakeModel()
    factory_calls: list[tuple[str, dict[str, Any]]] = []

    def factory(model_name: str, **kwargs: Any) -> FakeModel:
        factory_calls.append((model_name, kwargs))
        return model

    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(),
        model_factory=factory,
        model_downloader=_fake_model_downloader,
    )
    recognizer.start()
    recognizer.start()

    assert factory_calls == [
        (
            FAKE_SNAPSHOT,
            {"device": "cpu", "compute_type": "int8"},
        )
    ]
    assert len(model.calls) == 1
    assert model.consumed == 1
    assert model.calls[0][0].dtype == np.float32
    assert model.calls[0][0].shape == (16_000,)
    assert model.calls[0][1] == {
        "language": "en",
        "beam_size": 1,
        "condition_on_previous_text": False,
        "task": "transcribe",
        "temperature": 0,
        "vad_filter": False,
        "word_timestamps": False,
        "without_timestamps": True,
    }


def test_transcribe_consumes_all_segments_and_preserves_raw_text() -> None:
    model = FakeModel()
    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(),
        model_factory=lambda *args, **kwargs: model,
        model_downloader=_fake_model_downloader,
    )
    recognizer.start()

    result = recognizer.transcribe(
        np.ones(800, dtype=np.float64),
        speech_context=SpeechContext(
            initial_prompt="Game words: Zip, Zap, Zop.",
            hotwords=("Zip", "Zap", "Zip", "Zop Zop"),
        ),
    )

    assert result.text == "Zip zap."
    assert result.accepted is True
    assert result.rejection_reasons == ()
    assert model.consumed == 3
    waveform, kwargs = model.calls[1]
    assert waveform.dtype == np.float32
    assert waveform.flags.c_contiguous
    assert kwargs["hotwords"] == "Zip Zap Zop Zop"
    assert kwargs["initial_prompt"] == "Game words: Zip, Zap, Zop."
    assert result.diagnostics["segment_count"] == 2
    assert result.diagnostics["language"] == "en"
    assert result.diagnostics["segments"][1]["id"] == 1
    assert result.diagnostics["quality_gate"]["accepted"] is True


def test_provenance_is_a_defensive_snapshot_and_reports_timings() -> None:
    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(),
        model_factory=lambda *args, **kwargs: FakeModel(),
        model_downloader=_fake_model_downloader,
    )
    recognizer.start()

    provenance = recognizer.provenance
    provenance["decode"]["beam_size"] = 99

    assert recognizer.provenance["decode"]["beam_size"] == 1
    assert recognizer.provenance["model"] == "tiny.en"
    assert recognizer.provenance["model_resolution_ms"] is not None
    assert recognizer.provenance["model_load_ms"] is not None
    assert recognizer.provenance["warmup_ms"] is not None
    assert recognizer.provenance["transcript_quality_gate"] == {
        "max_characters": 512,
        "reject_at_consecutive_token_repetitions": 16,
        "max_segment_compression_ratio": 10.0,
    }


def test_local_snapshot_path_records_resolved_revision(tmp_path: Any) -> None:
    snapshot = tmp_path / "snapshots" / "cache" / "models--repo" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(model=str(snapshot), local_files_only=True),
        model_factory=lambda *args, **kwargs: FakeModel(),
    )

    recognizer.start()

    assert recognizer.provenance["model_path_resolved"] == str(snapshot.resolve())
    assert recognizer.provenance["model_revision_resolved"] == "abc123"


def test_existing_local_model_file_is_rejected_before_model_construction(tmp_path: Any) -> None:
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"not a model directory")
    factory_calls: list[Any] = []
    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(model=str(model_file)),
        model_factory=lambda *args, **kwargs: factory_calls.append((args, kwargs)),
    )

    with pytest.raises(SpeechRecognitionError, match="must be a directory"):
        recognizer.start()

    assert factory_calls == []


def test_remote_model_is_resolved_with_requested_policy_before_local_model_load(tmp_path: Any) -> None:
    factory_calls: list[tuple[str, dict[str, Any]]] = []
    downloader_calls: list[tuple[str, dict[str, Any]]] = []
    snapshot = tmp_path / "cache" / "models--Systran--faster-whisper-tiny.en" / "snapshots" / "resolved456"
    snapshot.mkdir(parents=True)

    def factory(model_path: str, **kwargs: Any) -> FakeModel:
        factory_calls.append((model_path, kwargs))
        return FakeModel()

    def downloader(model_name: str, **kwargs: Any) -> str:
        downloader_calls.append((model_name, kwargs))
        return str(snapshot)

    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(
            model="tiny.en",
            model_revision="requested123",
            model_cache=str(tmp_path / "cache"),
            local_files_only=True,
        ),
        model_factory=factory,
        model_downloader=downloader,
    )

    recognizer.start()

    assert factory_calls == [
        (
            str(snapshot.resolve()),
            {"device": "cpu", "compute_type": "int8"},
        )
    ]
    assert downloader_calls == [
        (
            "tiny.en",
            {
                "local_files_only": True,
                "cache_dir": str(tmp_path / "cache"),
                "revision": "requested123",
            },
        )
    ]
    assert recognizer.provenance["model_revision_requested"] == "requested123"
    assert recognizer.provenance["model_revision_resolved"] == "resolved456"
    assert recognizer.provenance["model_path_resolved"] == str(snapshot.resolve())


def test_local_model_directory_does_not_invoke_downloader(tmp_path: Any) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    downloader_calls: list[Any] = []
    factory_calls: list[tuple[str, dict[str, Any]]] = []

    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(model=str(model_dir), local_files_only=True),
        model_factory=lambda model_path, **kwargs: (
            factory_calls.append((model_path, kwargs)) or FakeModel()
        ),
        model_downloader=lambda *args, **kwargs: downloader_calls.append((args, kwargs)),
    )

    recognizer.start()

    assert downloader_calls == []
    assert factory_calls == [
        (str(model_dir.resolve()), {"device": "cpu", "compute_type": "int8"})
    ]


def test_remote_resolution_must_return_an_existing_directory(tmp_path: Any) -> None:
    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(),
        model_factory=lambda *args, **kwargs: FakeModel(),
        model_downloader=lambda *args, **kwargs: str(tmp_path / "missing"),
    )

    with pytest.raises(SpeechRecognitionError, match="local snapshot directory"):
        recognizer.start()


def test_real_downloader_honors_offline_empty_cache_contract(tmp_path: Any) -> None:
    pytest.importorskip("faster_whisper")
    factory_calls: list[Any] = []
    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(
            model="tiny.en",
            model_cache=str(tmp_path / "empty-cache"),
            local_files_only=True,
        ),
        model_factory=lambda *args, **kwargs: factory_calls.append((args, kwargs)),
    )

    with pytest.raises(SpeechRecognitionError, match="model snapshot") as exc_info:
        recognizer.start()

    assert not isinstance(exc_info.value.__cause__, TypeError)
    assert factory_calls == []


def test_model_load_and_transcription_errors_are_typed_and_actionable() -> None:
    def broken_factory(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("bad compute type")

    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(),
        model_factory=broken_factory,
        model_downloader=_fake_model_downloader,
    )
    with pytest.raises(SpeechRecognitionError, match="CTranslate2"):
        recognizer.start()

    class BrokenModel(FakeModel):
        def transcribe(self, audio: np.ndarray, **kwargs: Any) -> Any:
            if not self.calls:
                self.calls.append((audio, kwargs))
                return iter(()), Info()
            raise RuntimeError("decode failed")

    broken = FasterWhisperRecognizer(
        VoiceInputConfig(),
        model_factory=lambda *args, **kwargs: BrokenModel(),
        model_downloader=_fake_model_downloader,
    )
    broken.start()
    with pytest.raises(SpeechRecognitionError, match="decode failed"):
        broken.transcribe(np.zeros(100, dtype=np.float32))


def test_warmup_generator_failure_preserves_error_when_model_cleanup_also_fails() -> None:
    class WarmupFailureModel(FakeModel):
        def transcribe(self, audio: np.ndarray, **kwargs: Any) -> Any:
            def fail_during_consumption() -> Any:
                raise RuntimeError("warmup decode failed")
                yield  # pragma: no cover

            return fail_during_consumption(), Info()

        def close(self) -> None:
            self.closed += 1
            raise RuntimeError("cleanup failed")

    model = WarmupFailureModel()
    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(),
        model_factory=lambda *args, **kwargs: model,
        model_downloader=_fake_model_downloader,
    )

    with pytest.raises(SpeechRecognitionError, match="warmup decode failed"):
        recognizer.start()

    assert model.closed == 1


def test_close_releases_model_once() -> None:
    model = FakeModel()
    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(),
        model_factory=lambda *args, **kwargs: model,
        model_downloader=_fake_model_downloader,
    )
    recognizer.start()

    recognizer.close()
    recognizer.close()

    assert model.closed == 1


def test_close_wraps_model_cleanup_error_and_remains_idempotent() -> None:
    class CloseFailureModel(FakeModel):
        def close(self) -> None:
            self.closed += 1
            raise RuntimeError("model cleanup failed")

    model = CloseFailureModel()
    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(),
        model_factory=lambda *args, **kwargs: model,
        model_downloader=_fake_model_downloader,
    )
    recognizer.start()

    with pytest.raises(SpeechRecognitionError, match="model cleanup failed") as exc_info:
        recognizer.close()
    recognizer.close()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert model.closed == 1


@pytest.mark.parametrize("import_target", ["faster_whisper", "faster_whisper.utils"])
def test_faster_whisper_import_oserror_is_wrapped_as_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
    import_target: str,
) -> None:
    original_import = builtins.__import__

    def broken_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == import_target:
            raise OSError("shared library missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    operation = (
        faster_whisper_module._default_model_factory
        if import_target == "faster_whisper"
        else faster_whisper_module._default_model_downloader
    )

    with pytest.raises(VoiceDependencyError, match="voice extra") as exc_info:
        operation("tiny.en")

    assert isinstance(exc_info.value.__cause__, OSError)
