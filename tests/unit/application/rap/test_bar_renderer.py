"""Tests for complete immutable rap bar rendering."""

from __future__ import annotations

import numpy as np

from streammuse.application.rap.audio_rendering import bar_frame_count
from streammuse.application.rap.bar_renderer import DeterministicRapBarRenderer
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.domain.rap import (
    AudioFormat,
    AudioWarning,
    AudioWarningCode,
    AudioWarningSeverity,
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    PcmAudio,
    ProsodyAnalysis,
    RenderedSyllable,
    ScenarioSegment,
    ScheduledSyllable,
    Syllable,
    SyllableRenderRequest,
    materialize_flow,
)
from streammuse.domain.timing import Tempo


class ImpulseSpeechSynthesizer:
    def __init__(self, *, frames: int, warnings: tuple[AudioWarning, ...] = (), sustained: bool = False) -> None:
        self._frames = frames
        self._warnings = warnings
        self._sustained = sustained
        self.requests: list[SyllableRenderRequest] = []

    def synthesize(self, request: SyllableRenderRequest) -> RenderedSyllable:
        self.requests.append(request)
        samples = np.full(self._frames, 0.5 if self._sustained else 0.0, dtype=np.float32)
        if self._frames:
            samples[0] = 1.0
        audio = PcmAudio(AudioFormat(48_000, 1), self._frames, samples.tobytes())
        return RenderedSyllable(request, audio, (), "test", 0.0, self._warnings)


class SilentDrumRenderer:
    def render(self, template: FlowTemplate, tempo: Tempo, audio_format: AudioFormat, bar: int) -> PcmAudio:
        frames = bar_frame_count(bar, tempo, audio_format)
        return PcmAudio(audio_format, frames, bytes(frames * audio_format.channels * audio_format.sample_width_bytes))


class ExactLengthTimeStretcher:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def stretch(self, audio: PcmAudio, target_frames: int) -> PcmAudio:
        self.calls.append((audio.frame_count, target_frames))
        samples = np.frombuffer(audio.data, dtype=np.float32).reshape(
            audio.frame_count,
            audio.format.channels,
        )
        fitted = samples[np.minimum(np.arange(target_frames), audio.frame_count - 1)]
        return PcmAudio(audio.format, target_frames, fitted.astype(np.float32).tobytes())


def planned_bar_with_slots(*, bar: int, ticks: tuple[int, ...]) -> PlannedRapBar:
    template = FlowTemplate(
        template_id="test-flow",
        name="Test flow",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=tuple(FlowSlot(tick_in_bar=tick, duration_ticks=1, target_stress=0.8) for tick in ticks),
        provenance=FlowProvenance(kind="test", source="test"),
    )
    syllables = tuple(
        Syllable(
            word=f"word{index}",
            index_in_word=0,
            syllable_count=1,
            stress=1,
            phonemes=("W", "ER1", "D"),
            analysis_source="cmudict",
        )
        for index in range(len(ticks))
    )
    scheduled = tuple(
        ScheduledSyllable(slot, syllable)
        for slot, syllable in zip(materialize_flow(template, bar), syllables, strict=True)
    )
    return PlannedRapBar(
        bar=bar,
        segment=ScenarioSegment(0, 1, "test", template.template_id, ("test",)),
        template=template,
        analysis=ProsodyAnalysis("test", "test", syllables, (), (), (), ()),
        scheduled=scheduled,
        text="test",
        source="test",
        fallback_reason=None,
    )


def stereo_array(audio: PcmAudio) -> np.ndarray:
    return np.frombuffer(audio.data, dtype=np.float32).reshape(audio.frame_count, audio.format.channels)


def test_bar_renderer_places_every_syllable_at_exact_target_sample() -> None:
    plan = planned_bar_with_slots(bar=0, ticks=(0, 2, 7, 12, 15))
    synthesizer = ImpulseSpeechSynthesizer(frames=1_000)
    renderer = DeterministicRapBarRenderer(
        tempo=Tempo(60.0, 4, 4),
        audio_format=AudioFormat(48_000, 2),
        synthesizer=synthesizer,
        drums=SilentDrumRenderer(),
        time_stretcher=ExactLengthTimeStretcher(),
    )

    prepared = renderer.render(plan)
    samples = stereo_array(prepared.audio)

    assert prepared.audio.frame_count == 192_000
    assert [item.target_sample for item in prepared.diagnostics] == [0, 24_000, 84_000, 144_000, 180_000]
    assert all(item.software_error_samples == 0 for item in prepared.diagnostics)
    assert np.all(samples[[0, 24_000, 84_000, 144_000, 180_000]] == np.float32(0.8))
    assert [request.slot_index for request in synthesizer.requests] == [0, 1, 2, 3, 4]


def test_bar_renderer_uses_absolute_samples_at_92_bpm_for_nonzero_bars() -> None:
    renderer = DeterministicRapBarRenderer(
        tempo=Tempo(92.0, 4, 4),
        audio_format=AudioFormat(48_000, 2),
        synthesizer=ImpulseSpeechSynthesizer(frames=1_000),
        drums=SilentDrumRenderer(),
        time_stretcher=ExactLengthTimeStretcher(),
    )

    first = renderer.render(planned_bar_with_slots(bar=1, ticks=(0, 3, 9, 15)))
    second = renderer.render(planned_bar_with_slots(bar=2, ticks=(0, 3, 9, 15)))

    assert first.audio.frame_count == 125_218
    assert [item.target_sample for item in first.diagnostics] == [0, 23_479, 70_435, 117_392]
    assert second.audio.frame_count == 125_217
    assert [item.target_sample for item in second.diagnostics] == [0, 23_478, 70_435, 117_391]
    for prepared, offsets in (
        (first, (0, 23_479, 70_435, 117_392)),
        (second, (0, 23_478, 70_435, 117_391)),
    ):
        samples = stereo_array(prepared.audio)
        assert np.all(samples[list(offsets)] == np.float32(0.8))
        assert not np.any(samples[[offset - 1 for offset in offsets[1:]]])


def test_bar_renderer_preserves_pronunciation_and_timing_warnings() -> None:
    pronunciation_warning = AudioWarning(
        code=AudioWarningCode.PRONUNCIATION_FALLBACK,
        severity=AudioWarningSeverity.WARNING,
        message="fallback",
    )
    stretcher = ExactLengthTimeStretcher()
    renderer = DeterministicRapBarRenderer(
        tempo=Tempo(60.0, 4, 4),
        audio_format=AudioFormat(48_000, 2),
        synthesizer=ImpulseSpeechSynthesizer(frames=30_000, warnings=(pronunciation_warning,), sustained=True),
        drums=SilentDrumRenderer(),
        time_stretcher=stretcher,
    )

    prepared = renderer.render(planned_bar_with_slots(bar=0, ticks=(0, 1)))

    assert {warning.code for warning in prepared.warnings} == {
        AudioWarningCode.PRONUNCIATION_FALLBACK,
        AudioWarningCode.TIMING_PRESSURE,
    }
    assert stretcher.calls == [(30_000, 15_000)]


def test_bar_renderer_passes_its_configured_compression_cap_to_syllable_fitting() -> None:
    renderer = DeterministicRapBarRenderer(
        tempo=Tempo(60.0, 4, 4),
        audio_format=AudioFormat(48_000, 2),
        synthesizer=ImpulseSpeechSynthesizer(frames=144_000, sustained=True),
        drums=SilentDrumRenderer(),
        time_stretcher=ExactLengthTimeStretcher(),
        max_compression=3.0,
    )

    prepared = renderer.render(planned_bar_with_slots(bar=0, ticks=(0, 4)))

    assert prepared.diagnostics[0].compression_ratio == 3.0


def test_nonfinal_tail_cropped_at_bar_boundary_has_truthful_warning_and_lengths() -> None:
    renderer = DeterministicRapBarRenderer(
        tempo=Tempo(60.0, 4, 4),
        audio_format=AudioFormat(48_000, 2),
        synthesizer=ImpulseSpeechSynthesizer(frames=100_000, sustained=True),
        drums=SilentDrumRenderer(),
        time_stretcher=ExactLengthTimeStretcher(),
    )

    prepared = renderer.render(planned_bar_with_slots(bar=0, ticks=(14, 15)))

    first = prepared.diagnostics[0]
    crop_warning = next(
        warning
        for warning in prepared.warnings
        if warning.slot_index == first.slot_index and warning.action == "crop_at_bar_boundary"
    )
    assert first.fitted_frames == 50_000
    assert first.rendered_frames == 24_000
    assert first.cropped_frames == 26_000
    assert crop_warning.code == AudioWarningCode.FORCED_BAR_FIT
    assert crop_warning.rendered_ms == 500.0
