"""Deterministic full-bar mixing for independently synthesized rap syllables."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.signal import resample

from streammuse.application.rap.audio_rendering import (
    FitContext,
    bar_frame_count,
    fit_syllable,
    mix_at,
    tick_frame_in_bar,
    trim_silence,
)
from streammuse.application.rap.audio_service import (
    AudioTimeStretcher,
    DrumRenderer,
    SpeechSynthesizer,
)
from streammuse.application.rap.chunk_audio import mix_audio_stems, remix_prepared_bar
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.domain.rap import (
    AudioFormat,
    AudioWarning,
    AudioWarningCode,
    AudioWarningSeverity,
    FlowTemplate,
    PcmAudio,
    PreparedRapBar,
    SyllablePlacementDiagnostic,
    SyllableRenderRequest,
)
from streammuse.domain.timing import Tempo
from streammuse.domain.rap.audio import RapMixConfig, RenderedOnsetEstimate
from streammuse.domain.rap.timeline import BarTimingSnapshot


_TRIM_THRESHOLD_DBFS = -45.0
_TRIM_PADDING_MS = 5.0


class DeterministicRapBarRenderer:
    """Render every scheduled syllable into one immutable, sample-stable bar."""

    def __init__(
        self,
        *,
        tempo: Tempo,
        audio_format: AudioFormat,
        synthesizer: SpeechSynthesizer,
        drums: DrumRenderer,
        time_stretcher: AudioTimeStretcher,
        voice: str = "en-us",
        speed_wpm: int = 175,
        pitch: int = 50,
        max_compression: float = 2.0,
        mix_config: RapMixConfig | None = None,
    ) -> None:
        _validate_format(audio_format)
        self._tempo = tempo
        self._audio_format = audio_format
        self._synthesizer = synthesizer
        self._drums = drums
        self._time_stretcher = time_stretcher
        self.voice = voice
        self.speed_wpm = speed_wpm
        self.pitch = pitch
        self.max_compression = max_compression
        self.set_mix_config(mix_config or RapMixConfig())

    def set_mix_config(self, config: RapMixConfig) -> None:
        if not isinstance(config, RapMixConfig):
            raise ValueError("mix config must be a RapMixConfig")
        self._mix_config = config

    def remix(self, prepared: PreparedRapBar, config: RapMixConfig | None = None) -> PreparedRapBar:
        return remix_prepared_bar(prepared, config or self._mix_config)

    def render_drums_only(
        self, *, bar: int, template: FlowTemplate, timing: BarTimingSnapshot,
    ) -> PreparedRapBar:
        """A correctly timed local alternative with no speech/model work."""
        started = perf_counter()
        config = self._mix_config
        frames = bar_frame_count(bar, timing, self._audio_format)
        drums = self._drums.render(template, timing, self._audio_format, bar)
        vocals = PcmAudio(self._audio_format, frames, bytes(frames * self._audio_format.channels * 4))
        prepared = PreparedRapBar(
            bar=bar, text="", source="drums_only", fallback_reason="vocal_unavailable",
            scheduled=(), audio=mix_audio_stems(vocals, drums, config), diagnostics=(), warnings=(),
            render_latency_ms=(perf_counter() - started) * 1000.0,
            vocal_stem=vocals, drum_stem=drums, mix_config=config, timing=timing,
        )
        return remix_prepared_bar(prepared, config)

    def render(self, plan: PlannedRapBar, *, timing: BarTimingSnapshot | None = None) -> PreparedRapBar:
        started = perf_counter()
        mix_config = self._mix_config
        tempo = timing or self._tempo
        frames = bar_frame_count(plan.bar, tempo, self._audio_format)
        mixed = np.zeros((frames, self._audio_format.channels), dtype=np.float32)
        scheduled = tuple(sorted(plan.scheduled, key=lambda item: item.slot.slot_index))
        warnings = []
        diagnostics = []

        for index, item in enumerate(scheduled):
            tick_in_bar = item.slot.tick - plan.bar * tempo.ticks_per_bar
            target_sample = tick_frame_in_bar(plan.bar, tick_in_bar, tempo, self._audio_format)
            next_target = (
                tick_frame_in_bar(
                    plan.bar,
                    scheduled[index + 1].slot.tick - plan.bar * tempo.ticks_per_bar,
                    tempo,
                    self._audio_format,
                )
                if index + 1 < len(scheduled)
                else frames
            )
            available_frames = next_target - target_sample
            request = SyllableRenderRequest(
                bar=plan.bar,
                slot_index=item.slot.slot_index,
                word=item.syllable.word,
                index_in_word=item.syllable.index_in_word,
                syllable_count=item.syllable.syllable_count,
                phonemes=item.syllable.phonemes,
                stress=item.syllable.stress,
                analysis_source=item.syllable.analysis_source,
                voice=self.voice,
                speed_wpm=self.speed_wpm,
                pitch=self.pitch,
            )
            rendered = self._synthesizer.synthesize(request)
            trimmed = trim_silence(rendered.audio, _TRIM_THRESHOLD_DBFS, _TRIM_PADDING_MS)
            source = _to_stereo_format(trimmed, self._audio_format)
            fitted = fit_syllable(
                source,
                available_frames=available_frames,
                final_in_bar=index == len(scheduled) - 1,
                context=FitContext(plan.bar, item.slot.slot_index, item.syllable.word),
                time_stretcher=self._time_stretcher,
                max_compression=self.max_compression,
            )
            rendered_frames = min(fitted.audio.frame_count, frames - target_sample)
            cropped_frames = fitted.audio.frame_count - rendered_frames
            rendered_audio = _head(fitted.audio, rendered_frames)
            mix_at(mixed, rendered_audio, target_sample, 1.0)
            diagnostics.append(
                SyllablePlacementDiagnostic(
                    bar=plan.bar,
                    slot_index=item.slot.slot_index,
                    word=item.syllable.word,
                    target_sample=target_sample,
                    source_frames=source.frame_count,
                    fitted_frames=fitted.audio.frame_count,
                    available_frames=available_frames,
                    compression_ratio=fitted.compression_ratio,
                    overlap_frames=fitted.overlap_frames,
                    pronunciation_source=rendered.pronunciation_source,
                    software_error_samples=0,
                    renderer_phonemes=rendered.renderer_phonemes,
                    synthesis_latency_ms=rendered.synthesis_latency_ms,
                    rendered_frames=rendered_frames,
                    cropped_frames=cropped_frames,
                )
            )
            warnings.extend(rendered.warnings)
            warnings.extend(fitted.warnings)
            if cropped_frames:
                warnings.append(
                    AudioWarning(
                        code=AudioWarningCode.FORCED_BAR_FIT,
                        severity=AudioWarningSeverity.WARNING,
                        message="Syllable tail cropped at the bar boundary",
                        bar=plan.bar,
                        slot_index=item.slot.slot_index,
                        word=item.syllable.word,
                        available_ms=(frames - target_sample) / self._audio_format.sample_rate_hz * 1000.0,
                        rendered_ms=rendered_frames / self._audio_format.sample_rate_hz * 1000.0,
                        compression_ratio=fitted.compression_ratio,
                        action="crop_at_bar_boundary",
                    )
                )

        drum_audio = self._drums.render(plan.template, tempo, self._audio_format, plan.bar)
        _require_matching_format(drum_audio, self._audio_format)
        vocal_stem = PcmAudio(self._audio_format, frames, mixed.tobytes())
        audio = mix_audio_stems(vocal_stem, drum_audio, mix_config)
        prepared = PreparedRapBar(
            bar=plan.bar,
            text=plan.text,
            source=plan.source,
            fallback_reason=plan.fallback_reason,
            scheduled=plan.scheduled,
            audio=audio,
            diagnostics=tuple(diagnostics),
            warnings=tuple(warnings),
            render_latency_ms=(perf_counter() - started) * 1000.0,
            vocal_stem=vocal_stem,
            drum_stem=drum_audio,
            mix_config=mix_config,
            timing=timing,
            onset_estimates=tuple(
                RenderedOnsetEstimate(
                    item.slot_index, item.target_sample,
                    unavailable_reason="isolated syllable placement is not acoustic onset alignment",
                )
                for item in diagnostics
            ),
        )
        return remix_prepared_bar(prepared, mix_config)


def _to_stereo_format(audio: PcmAudio, audio_format: AudioFormat) -> PcmAudio:
    if audio.format.sample_width_bytes != 4:
        raise ValueError("speech synthesis must return float32 PCM")
    samples = np.frombuffer(audio.data, dtype=np.float32).reshape(audio.frame_count, audio.format.channels)
    if audio.format.sample_rate_hz != audio_format.sample_rate_hz:
        target_frames = round(audio.frame_count * audio_format.sample_rate_hz / audio.format.sample_rate_hz)
        samples = resample(samples, target_frames, axis=0).astype(np.float32)
    if samples.shape[1] == 1:
        samples = np.repeat(samples, audio_format.channels, axis=1)
    if samples.shape[1] != audio_format.channels:
        raise ValueError("speech synthesis channel count must be mono or match the output")
    return PcmAudio(audio_format, samples.shape[0], samples.astype(np.float32, copy=False).tobytes())


def _head(audio: PcmAudio, frame_count: int) -> PcmAudio:
    if frame_count == audio.frame_count:
        return audio
    bytes_per_frame = audio.format.channels * audio.format.sample_width_bytes
    return PcmAudio(audio.format, frame_count, audio.data[: frame_count * bytes_per_frame])


def _validate_format(audio_format: AudioFormat) -> None:
    if audio_format.sample_rate_hz != 48_000 or audio_format.channels != 2 or audio_format.sample_width_bytes != 4:
        raise ValueError("rap bar rendering requires 48 kHz stereo float32 audio")


def _require_matching_format(audio: PcmAudio, audio_format: AudioFormat) -> None:
    if audio.format != audio_format:
        raise ValueError("drum renderer must return the requested audio format")
