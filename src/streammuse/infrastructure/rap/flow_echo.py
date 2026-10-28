"""Local captured-onset percussion over the existing drum renderer."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

import numpy as np

from streammuse.application.rap.audio_rendering import bar_frame_count, mix_at, tick_frame_in_bar
from streammuse.application.rap.chunk_audio import mix_audio_stems, remix_prepared_bar
from streammuse.domain.rap.audio import AudioFormat, PcmAudio, PreparedRapBar, RapMixConfig
from streammuse.domain.rap.flow import FlowTemplate
from streammuse.domain.timing import Tempo


class FlowEchoRenderer:
    def __init__(self, *, drums, idle_template: FlowTemplate, drum_gain: float = 0.48, echo_gain: float = 0.35):
        if any(not isfinite(value) or not 0 <= value <= 4 for value in (drum_gain, echo_gain)):
            raise ValueError("echo and drum gains must be finite and between zero and four")
        self._drums = drums
        self._idle_template = idle_template
        self._mix_config = RapMixConfig(drum_gain=drum_gain, vocal_gain=1.0)
        self._echo_gain = echo_gain

    def set_mix_config(self, config: RapMixConfig) -> None:
        if not isinstance(config, RapMixConfig):
            raise ValueError("mix configuration must be RapMixConfig")
        # The auxiliary stem is captured percussion, not generated vocals.
        self._mix_config = replace(config, vocal_gain=1.0)

    def render(self, *, bar: int, tempo: Tempo, audio_format: AudioFormat, template: FlowTemplate | None = None, timing=None) -> PreparedRapBar:
        if audio_format.sample_width_bytes != 4:
            raise ValueError("flow echo requires float32 audio")
        render_tempo = tempo if timing is None else timing.tempo
        frames = bar_frame_count(bar, render_tempo, audio_format) if timing is None else timing.frame_count
        drums = self._drums.render(self._idle_template, render_tempo, audio_format, bar)
        if drums.format != audio_format or abs(drums.frame_count - frames) > 1:
            raise ValueError("drum audio is incompatible with the destination bar")
        drum_samples = np.zeros((frames, audio_format.channels), dtype=np.float32)
        mix_at(drum_samples, drums, 0, 1.0)
        samples = np.zeros_like(drum_samples)
        if template is not None:
            # A short two-frequency woodblock distinguishes captured taps from hats.
            times = np.arange(round(0.045 * audio_format.sample_rate_hz)) / audio_format.sample_rate_hz
            hit = ((np.cos(2 * np.pi * 1250 * times) + 0.45 * np.cos(2 * np.pi * 1870 * times)) * np.exp(-110 * times) / 1.45).astype(np.float32)
            hit = np.repeat(hit[:, None], audio_format.channels, axis=1)
            for slot in template.slots:
                onset = tick_frame_in_bar(bar, slot.tick_in_bar, render_tempo, audio_format) if timing is None else timing.tick_offsets[slot.tick_in_bar]
                mix_at(samples, hit, onset, self._echo_gain)
        config = self._mix_config
        percussion = PcmAudio(audio_format, frames, samples.tobytes())
        drum_stem = PcmAudio(audio_format, frames, drum_samples.tobytes())
        prepared = PreparedRapBar(
            bar=bar, text="", source="drums_idle" if template is None else "flow_echo",
            fallback_reason=None, scheduled=(),
            audio=mix_audio_stems(percussion, drum_stem, config),
            diagnostics=(), warnings=(), render_latency_ms=0.0,
            timing=timing, vocal_stem=percussion, drum_stem=drum_stem, mix_config=config,
        )
        return remix_prepared_bar(prepared, config)
