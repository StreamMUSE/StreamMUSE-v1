"""Continuous eSpeak bar rendering with the listening-tested Gate D anchor policy."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Protocol

import numpy as np

from streammuse.application.rap.audio_rendering import (
    bar_frame_count,
    limit_peak,
    mix_at,
    tick_frame_in_bar,
)
from streammuse.application.rap.audio_service import DrumRenderer, RapBarRenderer
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.domain.rap import (
    AudioFormat,
    AudioWarning,
    AudioWarningCode,
    AudioWarningSeverity,
    PcmAudio,
    PreparedRapBar,
    SyllablePlacementDiagnostic,
)
from streammuse.domain.timing import Tempo


_ANCHOR_STRESS_THRESHOLD = 0.7
_ERROR_THRESHOLD_SECONDS = 0.120
_MAX_ADAPTIVE_ANCHORS = 6
_VOCAL_NORMALIZATION_PEAK = 0.72
_VOCAL_GAIN = 0.80
_DRUM_GAIN = 0.55
_FINAL_PEAK = 0.95


class RenderedContinuousPhrase(Protocol):
    audio: PcmAudio
    onset_frames: tuple[int, ...]
    synthesis_latency_ms: float
    pronunciation_source: str
    warnings: tuple[AudioWarning, ...]


class ContinuousPhraseSynthesizer(Protocol):
    def synthesize(
        self,
        text: str,
        *,
        voice: str,
        speed_wpm: int,
        pitch: int,
    ) -> RenderedContinuousPhrase: ...


class AudioTimeMapStretcher(Protocol):
    def stretch(
        self,
        audio: PcmAudio,
        target_frames: int,
        time_map: tuple[tuple[int, int], ...],
    ) -> PcmAudio: ...


class AdaptiveContinuousRapBarRenderer:
    """Render one natural eSpeak phrase through a sparse whole-bar R3 warp."""

    def __init__(
        self,
        *,
        tempo: Tempo,
        audio_format: AudioFormat,
        phrase_synthesizer: ContinuousPhraseSynthesizer,
        drums: DrumRenderer,
        time_map_stretcher: AudioTimeMapStretcher,
        fallback_renderer: RapBarRenderer,
        voice: str = "en-us",
        speed_wpm: int = 175,
        pitch: int = 50,
    ) -> None:
        if audio_format.sample_rate_hz != 48_000 or audio_format.channels != 2:
            raise ValueError("adaptive rap rendering requires 48 kHz stereo audio")
        if audio_format.sample_width_bytes != 4:
            raise ValueError("adaptive rap rendering requires float32 PCM")
        self._tempo = tempo
        self._audio_format = audio_format
        self._phrase_synthesizer = phrase_synthesizer
        self._drums = drums
        self._time_map_stretcher = time_map_stretcher
        self._fallback_renderer = fallback_renderer
        self.voice = voice
        self.speed_wpm = speed_wpm
        self.pitch = pitch

    def render(self, plan: PlannedRapBar) -> PreparedRapBar:
        started = perf_counter()
        try:
            return self._render_continuous(plan, started=started)
        except Exception as exc:
            fallback = self._fallback_renderer.render(plan)
            warning = AudioWarning(
                code=AudioWarningCode.PRONUNCIATION_FALLBACK,
                severity=AudioWarningSeverity.WARNING,
                message=f"Adaptive continuous eSpeak failed; using isolated syllables: {exc}",
                bar=plan.bar,
                action="adaptive_phrase_to_isolated_syllables",
            )
            return replace(
                fallback,
                warnings=(*fallback.warnings, warning),
                render_latency_ms=(perf_counter() - started) * 1000.0,
            )

    def _render_continuous(self, plan: PlannedRapBar, *, started: float) -> PreparedRapBar:
        scheduled = tuple(sorted(plan.scheduled, key=lambda item: item.slot.slot_index))
        if not scheduled:
            raise ValueError("adaptive rendering requires at least one scheduled syllable")

        phrase = self._phrase_synthesizer.synthesize(
            plan.text,
            voice=self.voice,
            speed_wpm=self.speed_wpm,
            pitch=self.pitch,
        )
        _validate_phrase(phrase, expected_onsets=len(scheduled), audio_format=self._audio_format)

        frames = bar_frame_count(plan.bar, self._tempo, self._audio_format)
        targets = tuple(
            tick_frame_in_bar(
                plan.bar,
                item.slot.tick - plan.bar * self._tempo.ticks_per_bar,
                self._tempo,
                self._audio_format,
            )
            for item in scheduled
        )
        first_target = targets[0]
        target_frames = frames - first_target
        relative_targets = tuple(target - first_target for target in targets)
        selected = select_adaptive_anchor_indices(
            source_onsets=phrase.onset_frames,
            target_onsets=relative_targets,
            target_stresses=tuple(item.slot.accent for item in scheduled),
            source_frames=phrase.audio.frame_count,
            target_frames=target_frames,
            error_threshold_frames=round(
                _ERROR_THRESHOLD_SECONDS * self._audio_format.sample_rate_hz
            ),
            max_anchors=_MAX_ADAPTIVE_ANCHORS,
        )
        time_map = build_sparse_time_map(
            source_onsets=phrase.onset_frames,
            target_onsets=relative_targets,
            selected_indices=selected,
            source_frames=phrase.audio.frame_count,
            target_frames=target_frames,
        )
        warped = self._time_map_stretcher.stretch(phrase.audio, target_frames, time_map)
        if warped.format != phrase.audio.format or warped.frame_count != target_frames:
            raise ValueError("time-map stretcher returned the wrong format or duration")

        vocal_samples = np.frombuffer(warped.data, dtype=np.float32).reshape(
            warped.frame_count,
            warped.format.channels,
        )
        peak = float(np.max(np.abs(vocal_samples), initial=0.0))
        if peak > 0.0:
            vocal_samples = vocal_samples * np.float32(_VOCAL_NORMALIZATION_PEAK / peak)
        if vocal_samples.shape[1] == 1:
            vocal_samples = np.repeat(vocal_samples, self._audio_format.channels, axis=1)

        mixed = np.zeros((frames, self._audio_format.channels), dtype=np.float32)
        vocal_audio = PcmAudio(
            self._audio_format,
            len(vocal_samples),
            vocal_samples.astype(np.float32, copy=False).tobytes(),
        )
        mix_at(mixed, vocal_audio, first_target, _VOCAL_GAIN)
        drums = self._drums.render(plan.template, self._tempo, self._audio_format, plan.bar)
        if drums.format != self._audio_format or drums.frame_count != frames:
            raise ValueError("drum renderer returned the wrong format or duration")
        mix_at(mixed, drums, 0, _DRUM_GAIN)

        effective_relative = piecewise_map_frames(phrase.onset_frames, time_map)
        effective = tuple(first_target + frame for frame in effective_relative)
        diagnostics = tuple(
            _placement_diagnostic(
                plan=plan,
                index=index,
                source_onsets=phrase.onset_frames,
                source_frames=phrase.audio.frame_count,
                effective_onsets=effective,
                targets=targets,
                bar_frames=frames,
                pronunciation_source=phrase.pronunciation_source,
                synthesis_latency_ms=phrase.synthesis_latency_ms,
            )
            for index in range(len(scheduled))
        )
        audio = PcmAudio(
            self._audio_format,
            frames,
            limit_peak(mixed, _FINAL_PEAK).tobytes(),
        )
        return PreparedRapBar(
            bar=plan.bar,
            text=plan.text,
            source=plan.source,
            fallback_reason=plan.fallback_reason,
            scheduled=plan.scheduled,
            audio=audio,
            diagnostics=diagnostics,
            warnings=phrase.warnings,
            render_latency_ms=(perf_counter() - started) * 1000.0,
        )


def select_adaptive_anchor_indices(
    *,
    source_onsets: tuple[int, ...],
    target_onsets: tuple[int, ...],
    target_stresses: tuple[float, ...],
    source_frames: int,
    target_frames: int,
    error_threshold_frames: int,
    max_anchors: int,
) -> tuple[int, ...]:
    _validate_anchor_inputs(
        source_onsets,
        target_onsets,
        target_stresses,
        source_frames,
        target_frames,
    )
    selected = {
        index
        for index, stress in enumerate(target_stresses)
        if index in {0, len(target_stresses) - 1} or stress >= _ANCHOR_STRESS_THRESHOLD
    }
    while len(selected) < max_anchors:
        time_map = build_sparse_time_map(
            source_onsets=source_onsets,
            target_onsets=target_onsets,
            selected_indices=tuple(sorted(selected)),
            source_frames=source_frames,
            target_frames=target_frames,
        )
        mapped = piecewise_map_frames(source_onsets, time_map)
        errors = tuple(mapped[index] - target_onsets[index] for index in range(len(mapped)))
        candidates = tuple(index for index in range(len(source_onsets)) if index not in selected)
        if not candidates:
            break
        worst = max(candidates, key=lambda index: abs(errors[index]))
        if abs(errors[worst]) <= error_threshold_frames:
            break
        selected.add(worst)
    return tuple(sorted(selected))


def build_sparse_time_map(
    *,
    source_onsets: tuple[int, ...],
    target_onsets: tuple[int, ...],
    selected_indices: tuple[int, ...],
    source_frames: int,
    target_frames: int,
) -> tuple[tuple[int, int], ...]:
    points: list[tuple[int, int]] = [(0, 0)]
    for index in selected_indices:
        source = source_onsets[index]
        target = target_onsets[index]
        if source <= 0 or target <= 0:
            continue
        if source >= source_frames - 1 or target >= target_frames - 1:
            continue
        if source <= points[-1][0] or target <= points[-1][1]:
            raise ValueError(f"non-monotonic adaptive anchor: {(source, target)}")
        points.append((source, target))
    points.append((source_frames - 1, target_frames - 1))
    return tuple(points)


def piecewise_map_frames(
    source_frames: tuple[int, ...],
    time_map: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    mapped = []
    region = 0
    for source in source_frames:
        while region + 1 < len(time_map) - 1 and source > time_map[region + 1][0]:
            region += 1
        source_start, target_start = time_map[region]
        source_end, target_end = time_map[region + 1]
        fraction = (source - source_start) / max(1, source_end - source_start)
        mapped.append(round(target_start + fraction * (target_end - target_start)))
    return tuple(mapped)


def _validate_anchor_inputs(
    source_onsets: tuple[int, ...],
    target_onsets: tuple[int, ...],
    target_stresses: tuple[float, ...],
    source_frames: int,
    target_frames: int,
) -> None:
    if not source_onsets or len(source_onsets) != len(target_onsets):
        raise ValueError("source and target onsets must be nonempty and equal length")
    if len(source_onsets) != len(target_stresses):
        raise ValueError("one target stress is required per onset")
    if source_frames <= source_onsets[-1] or target_frames <= target_onsets[-1]:
        raise ValueError("onsets must lie within source and target audio")
    if any(right <= left for left, right in zip(source_onsets, source_onsets[1:])):
        raise ValueError("source onsets must be strictly increasing")
    if any(right <= left for left, right in zip(target_onsets, target_onsets[1:])):
        raise ValueError("target onsets must be strictly increasing")


def _validate_phrase(
    phrase: RenderedContinuousPhrase,
    *,
    expected_onsets: int,
    audio_format: AudioFormat,
) -> None:
    if phrase.audio.format.sample_rate_hz != audio_format.sample_rate_hz:
        raise ValueError("continuous phrase has the wrong sample rate")
    if phrase.audio.format.channels != 1 or phrase.audio.format.sample_width_bytes != 4:
        raise ValueError("continuous phrase must be mono float32 PCM")
    if len(phrase.onset_frames) != expected_onsets:
        raise ValueError(
            f"continuous phrase onset mismatch: {len(phrase.onset_frames)} != {expected_onsets}"
        )


def _placement_diagnostic(
    *,
    plan: PlannedRapBar,
    index: int,
    source_onsets: tuple[int, ...],
    source_frames: int,
    effective_onsets: tuple[int, ...],
    targets: tuple[int, ...],
    bar_frames: int,
    pronunciation_source: str,
    synthesis_latency_ms: float,
) -> SyllablePlacementDiagnostic:
    source_end = source_onsets[index + 1] if index + 1 < len(source_onsets) else source_frames
    effective_end = effective_onsets[index + 1] if index + 1 < len(effective_onsets) else bar_frames
    target_end = targets[index + 1] if index + 1 < len(targets) else bar_frames
    source_length = source_end - source_onsets[index]
    fitted_length = effective_end - effective_onsets[index]
    item = tuple(sorted(plan.scheduled, key=lambda entry: entry.slot.slot_index))[index]
    return SyllablePlacementDiagnostic(
        bar=plan.bar,
        slot_index=item.slot.slot_index,
        word=item.syllable.word,
        target_sample=targets[index],
        source_frames=source_length,
        fitted_frames=fitted_length,
        available_frames=target_end - targets[index],
        compression_ratio=source_length / max(1, fitted_length),
        overlap_frames=0,
        pronunciation_source=pronunciation_source,
        software_error_samples=effective_onsets[index] - targets[index],
        synthesis_latency_ms=synthesis_latency_ms,
        rendered_frames=fitted_length,
    )
