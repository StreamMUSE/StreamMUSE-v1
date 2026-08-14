"""Tests for deterministic procedural boom-bap rendering."""

from __future__ import annotations

import numpy as np

from streammuse.domain.rap import AudioFormat, FlowProvenance, FlowSlot, FlowTemplate, PcmAudio
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.drums import ProceduralBoomBapRenderer


def template() -> FlowTemplate:
    return FlowTemplate(
        template_id="test-flow",
        name="Test flow",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=(
            FlowSlot(tick_in_bar=0, duration_ticks=1, target_stress=1.0),
            FlowSlot(tick_in_bar=7, duration_ticks=1, target_stress=0.9),
            FlowSlot(tick_in_bar=12, duration_ticks=1, target_stress=0.4),
        ),
        provenance=FlowProvenance(kind="test", source="test"),
    )


def stereo_array(audio: PcmAudio) -> np.ndarray:
    return np.frombuffer(audio.data, dtype=np.float32).reshape(audio.frame_count, audio.format.channels)


def has_energy(samples: np.ndarray, *, frame: int, window: int) -> bool:
    return bool(np.max(np.abs(samples[frame : frame + window]), initial=0.0) > 1e-5)


def test_boom_bap_has_stable_meter_hits_and_sixteenth_hats() -> None:
    renderer = ProceduralBoomBapRenderer(seed=20260814)
    audio = renderer.render(template(), Tempo(60.0, 4, 4), AudioFormat(48_000, 2), bar=0)
    samples = stereo_array(audio)

    assert audio.frame_count == 192_000
    assert has_energy(samples, frame=0, window=2_000)
    assert has_energy(samples, frame=48_000, window=2_000)
    assert has_energy(samples, frame=96_000, window=2_000)
    assert has_energy(samples, frame=144_000, window=2_000)
    assert all(has_energy(samples, frame=tick * 12_000, window=1_500) for tick in range(16))


def test_boom_bap_is_reproducible_for_same_seed_and_bar() -> None:
    first = ProceduralBoomBapRenderer(seed=7).render(template(), Tempo(60.0, 4, 4), AudioFormat(48_000, 2), bar=3)
    second = ProceduralBoomBapRenderer(seed=7).render(template(), Tempo(60.0, 4, 4), AudioFormat(48_000, 2), bar=3)

    assert first.data == second.data
