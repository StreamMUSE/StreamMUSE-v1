"""Pure sample timing and waveform fitting for realtime rap audio."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.signal import resample

from streammuse.domain.rap import (
    AudioFormat,
    AudioWarning,
    AudioWarningCode,
    AudioWarningSeverity,
    PcmAudio,
)
from streammuse.domain.timing import Tempo


@dataclass(frozen=True)
class FitContext:
    bar: int
    slot_index: int
    word: str


@dataclass(frozen=True)
class FittedSyllable:
    audio: PcmAudio
    compression_ratio: float
    overlap_frames: int
    warnings: tuple[AudioWarning, ...]


def bar_start_frame(bar: int, tempo: Tempo, audio_format: AudioFormat) -> int:
    _validate_bar(bar)
    _require_float32_format(audio_format)
    absolute_tick = bar * tempo.ticks_per_bar
    return round(tempo.tick_to_seconds(absolute_tick) * audio_format.sample_rate_hz)


def bar_frame_count(bar: int, tempo: Tempo, audio_format: AudioFormat) -> int:
    return bar_start_frame(bar + 1, tempo, audio_format) - bar_start_frame(bar, tempo, audio_format)


def tick_frame_in_bar(
    bar: int,
    tick_in_bar: int,
    tempo: Tempo,
    audio_format: AudioFormat,
) -> int:
    _validate_bar(bar)
    if not 0 <= tick_in_bar < tempo.ticks_per_bar:
        raise ValueError("tick_in_bar must be within the bar")
    absolute_tick = bar * tempo.ticks_per_bar + tick_in_bar
    absolute_frame = round(tempo.tick_to_seconds(absolute_tick) * audio_format.sample_rate_hz)
    return absolute_frame - bar_start_frame(bar, tempo, audio_format)


def trim_silence(audio: PcmAudio, threshold_dbfs: float, padding_ms: float) -> PcmAudio:
    """Trim quiet leading/trailing frames while retaining bounded padding."""
    _require_float32_format(audio.format)
    if padding_ms < 0:
        raise ValueError("padding_ms must be nonnegative")
    samples = _samples_from_audio(audio)
    threshold = 10 ** (threshold_dbfs / 20)
    active = np.any(np.abs(samples) >= threshold, axis=1)
    active_frames = np.flatnonzero(active)
    if active_frames.size == 0:
        return _audio_from_samples(audio.format, np.empty((0, audio.format.channels), dtype=np.float32))

    padding_frames = round(padding_ms * audio.format.sample_rate_hz / 1000)
    start = max(0, int(active_frames[0]) - padding_frames)
    end = min(audio.frame_count, int(active_frames[-1]) + padding_frames + 1)
    return _audio_from_samples(audio.format, samples[start:end])


def fit_syllable(
    audio: PcmAudio,
    available_frames: int,
    final_in_bar: bool,
    context: FitContext,
) -> FittedSyllable:
    _require_float32_format(audio.format)
    if available_frames < 1:
        raise ValueError("available_frames must be positive")
    source_frames = audio.frame_count
    if source_frames == 0:
        return FittedSyllable(audio, 1.0, 0, ())

    required_ratio = source_frames / available_frames
    capped_ratio = min(max(required_ratio, 1.0), 2.0)
    target_frames = max(1, ceil(source_frames / capped_ratio))
    forced_bar_fit = final_in_bar and target_frames > available_frames
    if forced_bar_fit:
        target_frames = available_frames

    fitted_audio = _resample_audio(audio, target_frames)
    compression_ratio = source_frames / target_frames
    overlap_frames = max(0, target_frames - available_frames)
    warnings: list[AudioWarning] = []
    if overlap_frames:
        warnings.append(
            AudioWarning(
                code=AudioWarningCode.TIMING_PRESSURE,
                severity=AudioWarningSeverity.WARNING,
                message="Syllable exceeds available timing window; overlap retained",
                bar=context.bar,
                slot_index=context.slot_index,
                word=context.word,
                available_ms=available_frames / audio.format.sample_rate_hz * 1000,
                rendered_ms=target_frames / audio.format.sample_rate_hz * 1000,
                compression_ratio=compression_ratio,
                overlap_ms=overlap_frames / audio.format.sample_rate_hz * 1000,
                action="overlap",
            )
        )
    if forced_bar_fit:
        warnings.append(
            AudioWarning(
                code=AudioWarningCode.FORCED_BAR_FIT,
                severity=AudioWarningSeverity.WARNING,
                message="Final syllable forced to the bar boundary",
                bar=context.bar,
                slot_index=context.slot_index,
                word=context.word,
                available_ms=available_frames / audio.format.sample_rate_hz * 1000,
                rendered_ms=target_frames / audio.format.sample_rate_hz * 1000,
                compression_ratio=compression_ratio,
                action="resample_to_bar",
            )
        )
    return FittedSyllable(fitted_audio, compression_ratio, overlap_frames, tuple(warnings))


def mix_at(destination: np.ndarray, source: PcmAudio | np.ndarray, onset_frame: int, gain: float) -> None:
    """Add source samples into destination at an absolute frame onset."""
    if onset_frame < 0:
        raise ValueError("onset_frame must be nonnegative")
    if isinstance(source, PcmAudio):
        _require_float32_format(source.format)
        source_samples = _samples_from_audio(source)
    else:
        source_samples = _normalise_samples(source)
    destination_samples = _normalise_samples(destination)
    if destination_samples.shape[1] != source_samples.shape[1]:
        raise ValueError("destination and source channel counts must match")
    end = min(destination_samples.shape[0], onset_frame + source_samples.shape[0])
    if end > onset_frame:
        destination_samples[onset_frame:end] += source_samples[: end - onset_frame] * np.float32(gain)


def limit_peak(samples: np.ndarray, peak: float = 0.95) -> np.ndarray:
    if peak <= 0:
        raise ValueError("peak must be positive")
    result = np.asarray(samples, dtype=np.float32).copy()
    maximum = float(np.max(np.abs(result), initial=0.0))
    if maximum > peak:
        result *= np.float32(peak / maximum)
    return result


def _validate_bar(bar: int) -> None:
    if bar < 0:
        raise ValueError("bar must be nonnegative")


def _require_float32_format(audio_format: AudioFormat) -> None:
    if audio_format.sample_width_bytes != 4:
        raise ValueError("audio rendering requires float32 PCM (sample_width_bytes=4)")


def _normalise_samples(samples: np.ndarray) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float32)
    if result.ndim == 1:
        return result.reshape(-1, 1)
    if result.ndim != 2:
        raise ValueError("samples must be a one- or two-dimensional array")
    return result


def _samples_from_audio(audio: PcmAudio) -> np.ndarray:
    samples = np.frombuffer(audio.data, dtype=np.float32)
    return samples.reshape(audio.frame_count, audio.format.channels)


def _audio_from_samples(audio_format: AudioFormat, samples: np.ndarray) -> PcmAudio:
    normalised = _normalise_samples(samples).astype(np.float32, copy=False)
    return PcmAudio(audio_format, normalised.shape[0], normalised.tobytes())


def _resample_audio(audio: PcmAudio, target_frames: int) -> PcmAudio:
    if target_frames == audio.frame_count:
        return audio
    samples = _samples_from_audio(audio)
    fitted = resample(samples, target_frames, axis=0).astype(np.float32)
    return _audio_from_samples(audio.format, fitted)
