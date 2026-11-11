"""Bounded in-memory utterances; no microphone access or audio policy changes."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import io
import os
import struct
from threading import Lock
from time import perf_counter
from typing import Any, Callable
import wave

import numpy as np


MAX_DURATION_SECONDS = 15.0
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000
MAX_WAV_BYTES = int(MAX_DURATION_SECONDS * MAX_SAMPLE_RATE * 2) + 4096


class InvalidUtterance(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class TranscriberUnavailable(RuntimeError):
    """The optional local recognition dependency or model is unavailable."""


@dataclass(frozen=True)
class ValidatedUtterance:
    duration_seconds: float
    sample_rate: int
    silence: bool


@dataclass(frozen=True)
class UtteranceTranscript:
    text: str
    duration_seconds: float
    sample_rate: int
    transcription_ms: float
    silence: bool


def validate_wav(payload: bytes) -> ValidatedUtterance:
    """Accept complete RIFF/WAVE mono PCM16 only, with bounded metadata/audio."""
    if len(payload) > MAX_WAV_BYTES:
        raise InvalidUtterance("WAV exceeds the upload size limit", status_code=413)
    if len(payload) < 44 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise InvalidUtterance("Expected a complete PCM WAV body")
    if struct.unpack_from("<I", payload, 4)[0] != len(payload) - 8:
        raise InvalidUtterance("WAV size does not match its body")

    # wave tolerates truncated chunks; verify the enclosing RIFF structure first.
    offset = 12
    chunks: dict[bytes, tuple[int, int]] = {}
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise InvalidUtterance("Truncated WAV chunk header")
        name, size = struct.unpack_from("<4sI", payload, offset)
        start = offset + 8
        end = start + size
        if end + (size % 2) > len(payload):
            raise InvalidUtterance("Truncated WAV chunk")
        if name in (b"fmt ", b"data"):
            if name in chunks:
                raise InvalidUtterance("Duplicate WAV format or data chunk")
            chunks[name] = (start, size)
        offset = end + (size % 2)
    if b"fmt " not in chunks or b"data" not in chunks or chunks[b"fmt "][1] < 16:
        raise InvalidUtterance("WAV format and data chunks are required")
    fmt, channels, rate, byte_rate, alignment, bits = struct.unpack_from(
        "<HHIIHH", payload, chunks[b"fmt "][0]
    )
    if fmt != 1 or channels != 1 or bits != 16 or alignment != 2 or byte_rate != rate * 2:
        raise InvalidUtterance("Only uncompressed mono PCM16 WAV is supported")
    if not MIN_SAMPLE_RATE <= rate <= MAX_SAMPLE_RATE:
        raise InvalidUtterance("WAV sample rate must be between 8000 and 48000 Hz")
    data_size = chunks[b"data"][1]
    if not data_size or data_size % 2:
        raise InvalidUtterance("WAV must contain complete nonempty PCM16 frames")
    duration = data_size / (rate * 2)
    if duration > MAX_DURATION_SECONDS:
        raise InvalidUtterance("WAV duration exceeds 15 seconds", status_code=413)
    try:
        with wave.open(io.BytesIO(payload), "rb") as audio:
            frames = audio.readframes(audio.getnframes())
            if len(frames) != data_size:
                raise InvalidUtterance("WAV frame count does not match its data")
    except (wave.Error, EOFError) as error:
        raise InvalidUtterance("Invalid PCM WAV structure") from error
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    silence = float(np.sqrt(np.mean(samples * samples))) < 0.0001
    return ValidatedUtterance(duration_seconds=duration, sample_rate=rate, silence=silence)


class FasterWhisperUtteranceTranscriber:
    """One lazy CPU model per instance, restricted to already cached English models."""

    def __init__(
        self,
        *,
        model: str | None = None,
        enabled: bool | None = None,
        whisper_model_factory: Callable[..., Any] | None = None,
    ) -> None:
        model = model or os.environ.get("STREAMMUSE_ASR_MODEL", "base.en")
        self.model = {"base": "base.en", "small": "small.en"}.get(model, model)
        if self.model not in {"base.en", "small.en"}:
            raise ValueError("STREAMMUSE_ASR_MODEL must be base.en or small.en")
        if enabled is None:
            enabled = os.environ.get("STREAMMUSE_MICROPHONE_ENABLED", "1").lower() not in {
                "0", "false", "no", "off",
            }
        self._factory = whisper_model_factory
        self._model: Any | None = None
        self._model_lock = Lock()
        self._state_lock = Lock()
        self._state = "unloaded"
        self._reason: str | None = None
        if not enabled:
            self._state = "disabled"
            self._reason = "Microphone transcription is disabled"
        elif whisper_model_factory is None and importlib.util.find_spec("faster_whisper") is None:
            self._state = "unavailable"
            self._reason = "Optional faster-whisper dependency is not installed"

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "available": self._state not in {"disabled", "unavailable"},
                "model": self.model,
                "state": self._state,
                "reason": self._reason,
                "device": "cpu",
                "compute_type": "int8",
                "language": "en",
            }

    def transcribe(self, payload: bytes) -> UtteranceTranscript:
        started = perf_counter()
        audio = validate_wav(payload)
        if not self.status()["available"]:
            raise TranscriberUnavailable(self.status()["reason"])
        text = ""
        if not audio.silence:
            with self._model_lock:
                model = self._load_model()
                with io.BytesIO(payload) as source:
                    segments, _info = model.transcribe(
                        source, language="en", vad_filter=True,
                        condition_on_previous_text=False, beam_size=1,
                        word_timestamps=False,
                    )
                    text = " ".join(segment.text.strip() for segment in segments).strip()
        return UtteranceTranscript(
            text=text,
            duration_seconds=audio.duration_seconds,
            sample_rate=audio.sample_rate,
            transcription_ms=(perf_counter() - started) * 1000.0,
            silence=not bool(text),
        )

    def _load_model(self) -> Any:
        with self._state_lock:
            if self._state == "unavailable":
                raise TranscriberUnavailable(self._reason)
            if self._model is not None:
                return self._model
            self._state = "loading"
        try:
            factory = self._factory or importlib.import_module("faster_whisper").WhisperModel
            model = factory(
                model_size_or_path=self.model,
                device="cpu", compute_type="int8", cpu_threads=2, num_workers=1,
                local_files_only=True,
            )
        except Exception as error:
            with self._state_lock:
                self._state = "unavailable"
                self._reason = "Local Whisper model could not be loaded; install the dependency and cache the model"
            raise TranscriberUnavailable(self._reason) from error
        with self._state_lock:
            self._model = model
            self._state = "ready"
        return model
