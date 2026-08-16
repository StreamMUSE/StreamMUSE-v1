"""Smooth loudness emphasis for flow-template stress targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from streammuse.experiments.rap_audio_protocols.contracts import SyllableTarget


@dataclass(frozen=True)
class StressRenderResult:
    samples: np.ndarray
    gain_envelope: np.ndarray
    input_rms: float
    output_rms: float
    peak_limited: bool
    syllable_gain_db: tuple[float, ...]


def apply_stress_envelope(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
    syllables: Sequence[SyllableTarget],
    weak_db: float = -1.0,
    strong_db: float = 2.5,
    ramp_seconds: float = 0.025,
    peak_limit: float = 0.999,
) -> StressRenderResult:
    """Apply target-stress loudness while preserving overall chunk loudness."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if not syllables:
        raise ValueError("at least one syllable is required")
    if ramp_seconds < 0:
        raise ValueError("ramp_seconds must not be negative")
    if not 0 < peak_limit <= 1:
        raise ValueError("peak_limit must lie in (0, 1]")

    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(mono) == 0:
        raise ValueError("samples must not be empty")
    target_seconds = tuple(float(syllable.target_seconds) for syllable in syllables)
    if any(right <= left for left, right in zip(target_seconds, target_seconds[1:])):
        raise ValueError("syllable targets must be strictly monotonic")
    if any(not 0.0 <= syllable.target_stress <= 1.0 for syllable in syllables):
        raise ValueError("target stress must lie in [0, 1]")

    target_samples = tuple(
        min(len(mono) - 1, max(0, round(seconds * sample_rate_hz)))
        for seconds in target_seconds
    )
    if any(right <= left for left, right in zip(target_samples, target_samples[1:])):
        raise ValueError("syllable targets must remain monotonic at the audio sample rate")
    gain_db = tuple(
        weak_db + (strong_db - weak_db) * syllable.target_stress
        for syllable in syllables
    )
    db_envelope = _build_smoothed_db_envelope(
        frame_count=len(mono),
        target_samples=target_samples,
        gain_db=gain_db,
        ramp_samples=round(ramp_seconds * sample_rate_hz),
    )
    gain_envelope = np.power(10.0, db_envelope / 20.0).astype(np.float32)
    stressed = mono * gain_envelope

    input_rms = _rms(mono)
    provisional_rms = _rms(stressed)
    loudness_scale = input_rms / provisional_rms if provisional_rms > 0 else 1.0
    stressed = stressed * np.float32(loudness_scale)
    gain_envelope = gain_envelope * np.float32(loudness_scale)

    peak = float(np.max(np.abs(stressed)))
    peak_limited = peak > peak_limit
    if peak_limited:
        peak_scale = peak_limit / peak
        stressed = stressed * np.float32(peak_scale)
        gain_envelope = gain_envelope * np.float32(peak_scale)

    output = np.asarray(stressed, dtype=np.float32)
    return StressRenderResult(
        samples=output,
        gain_envelope=np.asarray(gain_envelope, dtype=np.float32),
        input_rms=input_rms,
        output_rms=_rms(output),
        peak_limited=peak_limited,
        syllable_gain_db=gain_db,
    )


def _build_smoothed_db_envelope(
    *,
    frame_count: int,
    target_samples: tuple[int, ...],
    gain_db: tuple[float, ...],
    ramp_samples: int,
) -> np.ndarray:
    envelope = np.full(frame_count, gain_db[0], dtype=np.float32)
    for index, target_sample in enumerate(target_samples):
        end = target_samples[index + 1] if index + 1 < len(target_samples) else frame_count
        envelope[target_sample:end] = gain_db[index]
    if ramp_samples <= 1:
        return envelope

    half_ramp = max(1, ramp_samples // 2)
    for index in range(1, len(target_samples)):
        center = target_samples[index]
        start = max(target_samples[index - 1], center - half_ramp)
        next_target = target_samples[index + 1] if index + 1 < len(target_samples) else frame_count
        end = min(next_target, center + half_ramp)
        width = end - start
        if width <= 1:
            continue
        phase = np.linspace(0.0, np.pi, width, dtype=np.float32)
        blend = (1.0 - np.cos(phase)) * 0.5
        envelope[start:end] = (
            gain_db[index - 1] * (1.0 - blend) + gain_db[index] * blend
        )
    return envelope


def _rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    values = np.asarray(samples, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values))))
