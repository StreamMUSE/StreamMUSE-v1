"""Deterministic full-bar mixing for independently synthesized rap syllables."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.signal import resample

from streammuse.application.rap.audio_rendering import (
    FitContext,
    bar_frame_count,
    fit_syllable,
    limit_peak,
    mix_at,
    tick_frame_in_bar,
    trim_silence,
)
from streammuse.application.rap.audio_service import (
    AudioTimeStretcher,
    DrumRenderer,
    SpeechSynthesizer,
)
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.domain.rap import (
    AudioFormat,
    AudioWarning,
    AudioWarningCode,
    AudioWarningSeverity,
    PcmAudio,
    PreparedRapBar,
    SyllablePlacementDiagnostic,
    SyllableRenderRequest,
)
from streammuse.domain.timing import Tempo


_TRIM_THRESHOLD_DBFS = -45.0
_TRIM_PADDING_MS = 5.0
_VOCAL_GAIN = 0.80
_DRUM_GAIN = 0.55
_FINAL_PEAK = 0.95


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

    def render(self, plan: PlannedRapBar) -> PreparedRapBar:
        started = perf_counter()
        frames = bar_frame_count(plan.bar, self._tempo, self._audio_format)
        mixed = np.zeros((frames, self._audio_format.channels), dtype=np.float32)
        scheduled = tuple(sorted(plan.scheduled, key=lambda item: item.slot.slot_index))
        warnings = []
        diagnostics = []

        for index, item in enumerate(scheduled):
            tick_in_bar = item.slot.tick - plan.bar * self._tempo.ticks_per_bar
            target_sample = tick_frame_in_bar(plan.bar, tick_in_bar, self._tempo, self._audio_format)
            next_target = (
                tick_frame_in_bar(
                    plan.bar,
                    scheduled[index + 1].slot.tick - plan.bar * self._tempo.ticks_per_bar,
                    self._tempo,
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
            mix_at(mixed, rendered_audio, target_sample, _VOCAL_GAIN)
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

        drum_audio = self._drums.render(plan.template, self._tempo, self._audio_format, plan.bar)
        _require_matching_format(drum_audio, self._audio_format)
        mix_at(mixed, drum_audio, 0, _DRUM_GAIN)
        audio = PcmAudio(self._audio_format, frames, limit_peak(mixed, _FINAL_PEAK).tobytes())
        return PreparedRapBar(
            bar=plan.bar,
            text=plan.text,
            source=plan.source,
            fallback_reason=plan.fallback_reason,
            scheduled=plan.scheduled,
            audio=audio,
            diagnostics=tuple(diagnostics),
            warnings=tuple(warnings),
            render_latency_ms=(perf_counter() - started) * 1000.0,
        )


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
