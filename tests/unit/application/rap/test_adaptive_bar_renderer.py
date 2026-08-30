"""Tests for the Gate D continuous eSpeak bar renderer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from streammuse.application.rap.audio_rendering import bar_frame_count
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.domain.rap import (
    AudioFormat,
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    PcmAudio,
    PreparedRapBar,
    ProsodyAnalysis,
    ScenarioSegment,
    ScheduledSyllable,
    Syllable,
    materialize_flow,
)
from streammuse.domain.timing import Tempo


class SilentDrumRenderer:
    def render(
        self,
        template: FlowTemplate,
        tempo: Tempo,
        audio_format: AudioFormat,
        bar: int,
    ) -> PcmAudio:
        frames = bar_frame_count(bar, tempo, audio_format)
        return PcmAudio(
            audio_format,
            frames,
            bytes(frames * audio_format.channels * audio_format.sample_width_bytes),
        )


class PhraseSynthesizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str, int, int]] = []

    def synthesize(self, text: str, *, voice: str, speed_wpm: int, pitch: int):
        self.calls.append((text, voice, speed_wpm, pitch))
        if self.fail:
            raise RuntimeError("event alignment failed")
        samples = np.full(50_000, 0.5, dtype=np.float32)
        return SimpleNamespace(
            audio=PcmAudio(AudioFormat(48_000, 1), len(samples), samples.tobytes()),
            onset_frames=(0, 10_000, 20_000),
            synthesis_latency_ms=2.5,
            pronunciation_source="espeak_continuous_events",
            warnings=(),
        )


class RecordingTimeMapStretcher:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []

    def stretch(
        self,
        audio: PcmAudio,
        target_frames: int,
        time_map: tuple[tuple[int, int], ...],
    ) -> PcmAudio:
        self.calls.append((audio.frame_count, target_frames, time_map))
        samples = np.full(target_frames, 0.5, dtype=np.float32)
        return PcmAudio(audio.format, target_frames, samples.tobytes())


class FailIfFallbackRenderer:
    def render(self, plan: PlannedRapBar) -> PreparedRapBar:
        raise AssertionError(f"fallback renderer should not run: {plan.text}")


class SentinelFallbackRenderer:
    def __init__(self, prepared: PreparedRapBar) -> None:
        self.prepared = prepared
        self.calls = 0

    def render(self, plan: PlannedRapBar) -> PreparedRapBar:
        self.calls += 1
        return self.prepared


def _plan() -> PlannedRapBar:
    ticks = (0, 4, 12)
    template = FlowTemplate(
        template_id="adaptive-test",
        name="Adaptive test",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=tuple(
            FlowSlot(
                tick_in_bar=tick,
                duration_ticks=1,
                target_stress=(0.4, 0.8, 0.5)[index],
            )
            for index, tick in enumerate(ticks)
        ),
        provenance=FlowProvenance(kind="test", source="test"),
    )
    syllables = tuple(
        Syllable(
            word=("plastic", "folders", "slide")[index],
            index_in_word=0,
            syllable_count=1,
            stress=1,
            phonemes=("AA1",),
            analysis_source="cmudict",
        )
        for index in range(3)
    )
    scheduled = tuple(
        ScheduledSyllable(slot, syllable)
        for slot, syllable in zip(materialize_flow(template, 0), syllables, strict=True)
    )
    return PlannedRapBar(
        bar=0,
        segment=ScenarioSegment(0, 1, "test", template.template_id, ("test",)),
        template=template,
        analysis=ProsodyAnalysis("plastic folders slide", "test", syllables, (), (), (), ()),
        scheduled=scheduled,
        text="Plastic folders slide",
        source="local_chat",
        fallback_reason=None,
    )


def test_adaptive_anchor_policy_reproduces_gate_d_selection() -> None:
    from streammuse.application.rap.adaptive_bar_renderer import select_adaptive_anchor_indices

    selected = select_adaptive_anchor_indices(
        source_onsets=(0, 8_921, 23_075, 36_528, 46_363, 64_292, 68_193, 78_180, 85_053),
        target_onsets=(0, 16_000, 24_000, 48_000, 56_000, 80_000, 88_000, 112_000, 120_000),
        target_stresses=(1.0, 0.2, 0.7, 0.2, 0.6, 1.0, 0.2, 0.7, 0.9),
        source_frames=102_242,
        target_frames=128_000,
        error_threshold_frames=5_760,
        max_anchors=6,
    )

    assert selected == (0, 1, 2, 5, 7, 8)


def test_adaptive_renderer_warps_one_phrase_to_an_exact_60_bpm_bar() -> None:
    from streammuse.application.rap.adaptive_bar_renderer import AdaptiveContinuousRapBarRenderer

    plan = _plan()
    phrase = PhraseSynthesizer()
    stretcher = RecordingTimeMapStretcher()
    renderer = AdaptiveContinuousRapBarRenderer(
        tempo=Tempo(60.0, 4, 4),
        audio_format=AudioFormat(48_000, 2),
        phrase_synthesizer=phrase,
        drums=SilentDrumRenderer(),
        time_map_stretcher=stretcher,
        fallback_renderer=FailIfFallbackRenderer(),
    )

    prepared = renderer.render(plan)

    assert prepared.audio.frame_count == 192_000
    assert phrase.calls == [("Plastic folders slide", "en-us", 175, 50)]
    assert stretcher.calls == [
        (
            50_000,
            192_000,
            ((0, 0), (10_000, 48_000), (20_000, 144_000), (49_999, 191_999)),
        )
    ]
    assert [item.target_sample for item in prepared.diagnostics] == [0, 48_000, 144_000]
    assert [item.software_error_samples for item in prepared.diagnostics] == [0, 0, 0]
    assert {item.pronunciation_source for item in prepared.diagnostics} == {
        "espeak_continuous_events"
    }


def test_adaptive_renderer_warns_and_uses_existing_renderer_when_phrase_alignment_fails() -> None:
    from streammuse.application.rap.adaptive_bar_renderer import AdaptiveContinuousRapBarRenderer

    plan = _plan()
    frames = 192_000
    fallback_prepared = PreparedRapBar(
        bar=0,
        text=plan.text,
        source=plan.source,
        fallback_reason=None,
        scheduled=plan.scheduled,
        audio=PcmAudio(AudioFormat(48_000, 2), frames, bytes(frames * 2 * 4)),
        diagnostics=(),
        warnings=(),
        render_latency_ms=1.0,
    )
    fallback = SentinelFallbackRenderer(fallback_prepared)
    renderer = AdaptiveContinuousRapBarRenderer(
        tempo=Tempo(60.0, 4, 4),
        audio_format=AudioFormat(48_000, 2),
        phrase_synthesizer=PhraseSynthesizer(fail=True),
        drums=SilentDrumRenderer(),
        time_map_stretcher=RecordingTimeMapStretcher(),
        fallback_renderer=fallback,
    )

    prepared = renderer.render(plan)

    assert fallback.calls == 1
    assert prepared.warnings[-1].action == "adaptive_phrase_to_isolated_syllables"
    assert "event alignment failed" in prepared.warnings[-1].message
