"""Deterministic procedural boom-bap drum rendering."""

from __future__ import annotations

import numpy as np

from streammuse.application.rap.audio_rendering import bar_frame_count, mix_at, tick_frame_in_bar
from streammuse.domain.rap import AudioFormat, FlowTemplate, PcmAudio
from streammuse.domain.timing import Tempo


_SAMPLE_RATE_HZ = 48_000
_DRUM_PEAK = 0.65


class ProceduralBoomBapRenderer:
    """Render a repeatable 16th-note boom-bap pattern for one planned bar."""

    def __init__(self, *, seed: int) -> None:
        self._seed = seed
        self._kick = _normalise_hit(_kick_hit())

    def render(self, template: FlowTemplate, tempo: Tempo, audio_format: AudioFormat, bar: int) -> PcmAudio:
        _validate_format(audio_format)
        if tempo.ticks_per_bar != 16:
            raise ValueError("boom-bap rendering requires sixteen ticks per bar")
        if bar < 0:
            raise ValueError("bar must be nonnegative")

        rng = np.random.default_rng(self._seed + bar)
        snare = _normalise_hit(_snare_hit(rng))
        hat = _normalise_hit(_hat_hit(rng))
        samples = np.zeros((bar_frame_count(bar, tempo, audio_format), audio_format.channels), dtype=np.float32)
        stressed_ticks = {slot.tick_in_bar for slot in template.slots if slot.target_stress >= 0.75}

        for tick in range(tempo.ticks_per_bar):
            onset = tick_frame_in_bar(bar, tick, tempo, audio_format)
            hat_gain = 0.14
            if tick % tempo.ticks_per_beat == 0:
                hat_gain += 0.03
            if tick in stressed_ticks:
                hat_gain += 0.02
            mix_at(samples, _stereo(hit=hat, channels=audio_format.channels), onset, hat_gain)
            if tick in (0, 8):
                mix_at(samples, _stereo(hit=self._kick, channels=audio_format.channels), onset, 0.42)
            if tick in (4, 12):
                mix_at(samples, _stereo(hit=snare, channels=audio_format.channels), onset, 0.36)

        peak = float(np.max(np.abs(samples), initial=0.0))
        if peak > _DRUM_PEAK:
            samples *= np.float32(_DRUM_PEAK / peak)
        return PcmAudio(audio_format, samples.shape[0], samples.tobytes())


def _kick_hit() -> np.ndarray:
    frames = round(0.120 * _SAMPLE_RATE_HZ)
    time = np.arange(frames, dtype=np.float32) / _SAMPLE_RATE_HZ
    frequency = 45.0 + 40.0 * np.exp(-8.0 * time)
    phase = 2.0 * np.pi * np.cumsum(frequency, dtype=np.float64) / _SAMPLE_RATE_HZ
    return (np.sin(phase) * np.exp(-18.0 * time)).astype(np.float32)


def _snare_hit(rng: np.random.Generator) -> np.ndarray:
    frames = round(0.110 * _SAMPLE_RATE_HZ)
    time = np.arange(frames, dtype=np.float32) / _SAMPLE_RATE_HZ
    envelope = np.exp(-24.0 * time)
    noise = rng.standard_normal(frames).astype(np.float32)
    body = 0.16 * np.sin(2.0 * np.pi * 180.0 * time)
    return ((0.68 * noise + body) * envelope).astype(np.float32)


def _hat_hit(rng: np.random.Generator) -> np.ndarray:
    frames = round(0.035 * _SAMPLE_RATE_HZ)
    time = np.arange(frames, dtype=np.float32) / _SAMPLE_RATE_HZ
    noise = rng.standard_normal(frames + 1).astype(np.float32)
    differentiated = np.diff(noise)
    return (differentiated * np.exp(-85.0 * time)).astype(np.float32)


def _stereo(*, hit: np.ndarray, channels: int) -> np.ndarray:
    return np.repeat(hit[:, np.newaxis], channels, axis=1)


def _normalise_hit(hit: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(hit), initial=0.0))
    return hit if peak == 0 else (hit / np.float32(peak)).astype(np.float32)


def _validate_format(audio_format: AudioFormat) -> None:
    if audio_format.sample_rate_hz != _SAMPLE_RATE_HZ or audio_format.channels != 2 or audio_format.sample_width_bytes != 4:
        raise ValueError("boom-bap rendering requires 48 kHz stereo float32 audio")
