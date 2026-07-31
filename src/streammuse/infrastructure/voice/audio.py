"""Shared audio conversion and PortAudio clock helpers."""

from __future__ import annotations

import math
from math import gcd
from typing import Any

import numpy as np

from . import VoiceDependencyError


def time_info_seconds(time_info: Any, field: str) -> float | None:
    """Read a finite PortAudio timestamp from mapping or CFFI-like data."""

    try:
        if hasattr(time_info, "get"):
            value = time_info.get(field)
        else:
            value = getattr(time_info, field)
    except (AttributeError, IndexError, KeyError, TypeError):
        try:
            value = getattr(time_info[0], field)
        except (AttributeError, IndexError, KeyError, TypeError):
            return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) else None


class PortAudioClockMapper:
    """Map one PortAudio buffer timestamp onto the local monotonic clock."""

    def __init__(self, buffer_time_field: str) -> None:
        self.buffer_time_field = str(buffer_time_field)
        self._offset_s: float | None = None

    def calibrate(self, *, portaudio_s: float, local_s: float) -> None:
        if math.isfinite(portaudio_s) and math.isfinite(local_s):
            self._offset_s = local_s - portaudio_s

    def buffer_start(
        self,
        time_info: Any,
        *,
        callback_local_s: float,
    ) -> float:
        callback_portaudio_s = time_info_seconds(time_info, "currentTime")
        offset_s = self._offset_s
        if offset_s is None and callback_portaudio_s is not None:
            offset_s = callback_local_s - callback_portaudio_s
            self._offset_s = offset_s
        buffer_s = time_info_seconds(time_info, self.buffer_time_field)
        if buffer_s is not None and offset_s is not None:
            return buffer_s + offset_s
        return callback_local_s

    def capture_interval(
        self,
        time_info: Any,
        *,
        callback_local_s: float,
        frame_count: int,
        sample_rate: int,
    ) -> tuple[float, float]:
        duration_s = max(0, int(frame_count)) / int(sample_rate)
        buffer_start_s = self.buffer_start(
            time_info,
            callback_local_s=callback_local_s,
        )
        if time_info_seconds(time_info, self.buffer_time_field) is not None:
            return buffer_start_s, buffer_start_s + duration_s
        return callback_local_s - duration_s, callback_local_s


def resample_float32(
    audio: np.ndarray,
    source_rate_hz: int,
    target_rate_hz: int,
) -> np.ndarray:
    """Return contiguous mono float32 audio at ``target_rate_hz``."""

    source_rate = int(source_rate_hz)
    target_rate = int(target_rate_hz)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    source = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    if source.size == 0 or source_rate == target_rate:
        return source
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - SciPy is a base dependency.
        raise VoiceDependencyError("Audio resampling requires scipy.") from exc
    divisor = gcd(source_rate, target_rate)
    result = resample_poly(
        source,
        target_rate // divisor,
        source_rate // divisor,
    )
    return np.ascontiguousarray(result, dtype=np.float32)
