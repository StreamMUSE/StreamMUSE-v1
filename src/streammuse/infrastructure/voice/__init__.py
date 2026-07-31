"""Optional microphone and speech-recognition infrastructure.

The optional third-party packages are imported only when a voice component is
started or devices are enumerated. Importing this package is therefore safe in
the default terminal-only installation.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


def _json_safe(value: Any) -> Any:
    """Return a defensive value accepted by ``json.dumps(..., allow_nan=False)``."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, bytes):
        return value.hex()
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]

    # NumPy scalars are common in model diagnostics but NumPy should not be an
    # import-time requirement of this small serialization helper.
    if type(value).__module__.startswith("numpy"):
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return _json_safe(tolist())
        item = getattr(value, "item", None)
        if callable(item):
            return _json_safe(item())
    return str(value)


class VoiceInfrastructureError(RuntimeError):
    """Base class for actionable voice input or output infrastructure failures."""


class VoiceDependencyError(VoiceInfrastructureError):
    """A package from the optional ``voice`` dependency group is missing."""


class VoiceStartupError(VoiceInfrastructureError):
    """Voice input could not be initialized before the game started."""


class MicrophoneError(VoiceInfrastructureError):
    """Base class for microphone selection and capture failures."""


class MicrophoneDeviceError(MicrophoneError, VoiceStartupError):
    """No usable microphone configuration could be selected."""


class MicrophoneCaptureError(MicrophoneError):
    """A running microphone stream failed."""


class AudioQueueOverflowError(MicrophoneCaptureError):
    """The real-time callback queue filled before it could be consumed."""


class SpeechRecognitionError(VoiceInfrastructureError):
    """The speech recognizer failed to load, warm up, or transcribe audio."""


class SpeechOutputError(VoiceInfrastructureError):
    """Base class for speech synthesis, playback, and artifact failures."""


class SpeechSynthesisError(SpeechOutputError):
    """A configured speech synthesizer failed."""


class SpeakerDeviceError(SpeechOutputError, VoiceStartupError):
    """No usable speaker configuration could be selected."""


class SpeakerPlaybackError(SpeechOutputError):
    """A running speaker stream failed."""


class SpeechArtifactError(SpeechOutputError):
    """A synthesized speech artifact could not be persisted."""


from .faster_whisper import FasterWhisperRecognizer, TranscriptionResult
from .microphone import (
    SUPPORTED_VAD_SAMPLE_RATES,
    CapturedUtterance,
    MicrophoneCapture,
    MicrophoneDevice,
    enumerate_input_devices,
)
from .response_source import VoiceHumanResponseSource

__all__ = [
    "AudioQueueOverflowError",
    "CapturedUtterance",
    "FasterWhisperRecognizer",
    "MicrophoneCapture",
    "MicrophoneCaptureError",
    "MicrophoneDevice",
    "MicrophoneDeviceError",
    "MicrophoneError",
    "SUPPORTED_VAD_SAMPLE_RATES",
    "SpeechRecognitionError",
    "SpeechArtifactError",
    "SpeechOutputError",
    "SpeechSynthesisError",
    "SpeakerDeviceError",
    "SpeakerPlaybackError",
    "TranscriptionResult",
    "VoiceDependencyError",
    "VoiceHumanResponseSource",
    "VoiceInfrastructureError",
    "VoiceStartupError",
    "enumerate_input_devices",
]
