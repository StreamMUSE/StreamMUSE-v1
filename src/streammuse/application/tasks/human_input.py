"""Human-response configuration and terminal input adapter."""

from __future__ import annotations

import math
import select
import sys
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from streammuse.domain.tasks import HumanResponse, HumanResponseRequest


@dataclass(frozen=True)
class TimedPromptResult:
    text: str
    timed_out: bool = False


class TerminalIO(Protocol):
    def write(self, text: str) -> None: ...

    def prompt(self, text: str) -> str: ...

    def prompt_with_timeout(self, text: str, timeout_s: float) -> TimedPromptResult: ...


class StdTerminalIO:
    def write(self, text: str) -> None:
        print(text, flush=True)

    def prompt(self, text: str) -> str:
        return input(text)

    def prompt_with_timeout(self, text: str, timeout_s: float) -> TimedPromptResult:
        sys.stdout.write(text)
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], max(0.0, float(timeout_s)))
        if not ready:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return TimedPromptResult(text="", timed_out=True)
        return TimedPromptResult(text=sys.stdin.readline().rstrip("\n"), timed_out=False)


@dataclass(frozen=True)
class VoiceInputConfig:
    model: str = "tiny.en"
    device: str = "cpu"
    compute_type: str = "int8"
    microphone_device: str | int | None = None
    model_cache: str | None = None
    model_revision: str | None = None
    local_files_only: bool = False
    save_audio: bool = False
    start_timeout_ms: float = 5000.0
    end_silence_ms: float = 300.0
    max_utterance_ms: float = 5000.0
    vad_aggressiveness: int = 2
    pre_roll_ms: float = 100.0
    queue_max_chunks: int = 128

    def __post_init__(self) -> None:
        _require_nonempty_string("model", self.model)
        _require_nonempty_string("device", self.device)
        _require_nonempty_string("compute_type", self.compute_type)
        _require_optional_nonempty_string("model_cache", self.model_cache)
        _require_optional_nonempty_string("model_revision", self.model_revision)

        microphone_device = self.microphone_device
        if isinstance(microphone_device, bool) or not isinstance(microphone_device, (str, int, type(None))):
            raise TypeError("microphone_device must be a device name, non-negative index, or None")
        if isinstance(microphone_device, str) and not microphone_device.strip():
            raise ValueError("microphone_device must not be empty")
        if isinstance(microphone_device, int) and microphone_device < 0:
            raise ValueError("microphone_device index must be >= 0")

        _require_bool("local_files_only", self.local_files_only)
        _require_bool("save_audio", self.save_audio)
        _require_positive_number("start_timeout_ms", self.start_timeout_ms)
        _require_positive_number("end_silence_ms", self.end_silence_ms)
        _require_positive_number("max_utterance_ms", self.max_utterance_ms)
        _require_nonnegative_number("pre_roll_ms", self.pre_roll_ms)

        if self.end_silence_ms > self.max_utterance_ms:
            raise ValueError("end_silence_ms must be <= max_utterance_ms")
        if self.pre_roll_ms > self.max_utterance_ms:
            raise ValueError("pre_roll_ms must be <= max_utterance_ms")
        if isinstance(self.vad_aggressiveness, bool) or not isinstance(self.vad_aggressiveness, int):
            raise TypeError("vad_aggressiveness must be an integer from 0 to 3")
        if self.vad_aggressiveness not in {0, 1, 2, 3}:
            raise ValueError("vad_aggressiveness must be between 0 and 3")
        if isinstance(self.queue_max_chunks, bool) or not isinstance(self.queue_max_chunks, int):
            raise TypeError("queue_max_chunks must be an integer")
        if self.queue_max_chunks <= 0:
            raise ValueError("queue_max_chunks must be > 0")


@dataclass(frozen=True)
class HumanInputConfig:
    mode: Literal["terminal", "voice"] = "terminal"
    voice: VoiceInputConfig | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"terminal", "voice"}:
            raise ValueError("mode must be 'terminal' or 'voice'")
        if self.mode == "terminal" and self.voice is not None:
            raise ValueError("terminal mode does not accept voice configuration")
        if self.mode == "voice" and self.voice is None:
            raise ValueError("voice mode requires voice configuration")
        if self.voice is not None and not isinstance(self.voice, VoiceInputConfig):
            raise TypeError("voice must be a VoiceInputConfig instance or None")


class TerminalHumanResponseSource:
    mode: Literal["terminal"] = "terminal"

    def __init__(self, terminal: TerminalIO) -> None:
        self._terminal = terminal

    def start(self) -> None:
        return None

    @property
    def provenance(self) -> dict[str, Any]:
        return {"mode": self.mode}

    def read_response(self, request: HumanResponseRequest) -> HumanResponse:
        if request.timeout_s is None:
            text = self._terminal.prompt(request.prompt)
            timed_out = False
        else:
            result = self._terminal.prompt_with_timeout(request.prompt, request.timeout_s)
            text = result.text
            timed_out = result.timed_out
        return HumanResponse(
            text=text,
            deadline_expired=timed_out,
            metadata={"mode": self.mode, "status": "ok"},
        )

    def close(self) -> None:
        return None


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_optional_nonempty_string(name: str, value: object) -> None:
    if value is None:
        return
    _require_nonempty_string(name, value)


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _require_positive_number(name: str, value: object) -> None:
    _require_number(name, value)
    if value <= 0:  # type: ignore[operator]
        raise ValueError(f"{name} must be > 0")


def _require_nonnegative_number(name: str, value: object) -> None:
    _require_number(name, value)
    if value < 0:  # type: ignore[operator]
        raise ValueError(f"{name} must be >= 0")


def _require_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
