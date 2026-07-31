"""Speech-output configuration and the dependency-free silent sink."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from streammuse.domain.tasks import SpeechPlayback, SpeechRequest


@dataclass(frozen=True)
class SpeechOutputConfig:
    mode: Literal["off", "audio"] = "off"
    backend: Literal["system", "espeak_ng", "kokoro", "null"] = "system"
    voice: str | None = None
    rate: float = 1.0
    speaker_device: str | int | None = None
    model: str | None = None
    model_cache: str | None = None
    model_revision: str | None = None
    local_files_only: bool = False
    synthesis_timeout_s: float = 10.0
    prewarm: bool = True
    cache_miss: Literal["synthesize", "skip"] = "synthesize"
    cache_max_entries: int = 512
    cache_max_bytes: int = 67_108_864
    guard_ms: float = 200.0
    on_error: Literal["fail", "warn"] = "fail"
    save_audio: bool = False
    llm_deadline_basis: Literal["text", "audio_end"] = "text"

    def __post_init__(self) -> None:
        if self.mode not in {"off", "audio"}:
            raise ValueError("mode must be 'off' or 'audio'")
        if self.backend not in {"system", "espeak_ng", "kokoro", "null"}:
            raise ValueError("unsupported speech backend")
        for name in ("voice", "model", "model_cache", "model_revision"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        device = self.speaker_device
        if isinstance(device, bool) or not isinstance(device, (str, int, type(None))):
            raise TypeError("speaker_device must be a device name, non-negative index, or None")
        if isinstance(device, str) and not device.strip():
            raise ValueError("speaker_device must not be empty")
        if isinstance(device, int) and device < 0:
            raise ValueError("speaker_device index must be >= 0")
        self._finite_between("rate", self.rate, minimum=0.25, maximum=4.0)
        self._finite_between(
            "synthesis_timeout_s",
            self.synthesis_timeout_s,
            minimum=0.001,
        )
        self._finite_between("guard_ms", self.guard_ms, minimum=0.0)
        for name in ("cache_max_entries", "cache_max_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.cache_miss not in {"synthesize", "skip"}:
            raise ValueError("cache_miss must be 'synthesize' or 'skip'")
        if self.on_error not in {"fail", "warn"}:
            raise ValueError("on_error must be 'fail' or 'warn'")
        if self.llm_deadline_basis not in {"text", "audio_end"}:
            raise ValueError("llm_deadline_basis must be 'text' or 'audio_end'")
        for name in ("local_files_only", "prewarm", "save_audio"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.mode == "off" and self.llm_deadline_basis == "audio_end":
            raise ValueError("audio_end deadline basis requires speech output")
        model_options = any(
            value is not None
            for value in (self.model, self.model_cache, self.model_revision)
        ) or self.local_files_only
        if self.backend != "kokoro" and model_options:
            raise ValueError("speech model options require the kokoro backend")
        if self.mode == "audio" and self.backend == "kokoro":
            if self.model is None or self.model_revision is None:
                raise ValueError(
                    "kokoro requires explicit speech model and model revision"
                )

    @staticmethod
    def _finite_between(
        name: str,
        value: float,
        *,
        minimum: float,
        maximum: float | None = None,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < minimum:
            raise ValueError(f"{name} must be finite and >= {minimum}")
        if maximum is not None and numeric > maximum:
            raise ValueError(f"{name} must be <= {maximum}")


class SilentSpeechOutput:
    mode: Literal["silent"] = "silent"

    def start(self) -> None:
        return None

    @property
    def provenance(self) -> dict[str, Any]:
        return {"mode": self.mode}

    def prepare(self, phrases: tuple[str, ...]) -> None:
        _ = phrases

    def speak(self, request: SpeechRequest) -> SpeechPlayback:
        return SpeechPlayback(status="disabled", spoken_text=request.text)

    def drain(self) -> None:
        return None

    def close(self) -> None:
        return None
