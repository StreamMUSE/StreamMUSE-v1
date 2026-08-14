"""Tests for deterministic procedural boom-bap rendering."""

from __future__ import annotations

import numpy as np

from streammuse.domain.rap import AudioFormat, FlowProvenance, FlowSlot, FlowTemplate, PcmAudio
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.drums import ProceduralBoomBapRenderer


def template(*, slots: tuple[tuple[int, float], ...] = ((0, 1.0), (7, 0.9), (12, 0.4))) -> FlowTemplate:
    return FlowTemplate(
        template_id="test-flow",
        name="Test flow",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=tuple(FlowSlot(tick_in_bar=tick, duration_ticks=1, target_stress=stress) for tick, stress in slots),
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


def test_boom_bap_kicks_and_snares_outlast_hats_at_their_fixed_ticks() -> None:
    samples = stereo_array(
        ProceduralBoomBapRenderer(seed=20260814).render(template(), Tempo(60.0, 4, 4), AudioFormat(48_000, 2), bar=0)
    )

    assert all(has_energy(samples, frame=tick * 12_000 + 3_000, window=1_000) for tick in (0, 8))
    assert all(has_energy(samples, frame=tick * 12_000 + 3_000, window=1_000) for tick in (4, 12))
    assert not any(has_energy(samples, frame=tick * 12_000 + 3_000, window=1_000) for tick in (2, 6, 10, 14))
    assert np.max(np.abs(samples)) <= 0.65


def test_flow_stress_changes_only_hat_accents() -> None:
    renderer = ProceduralBoomBapRenderer(seed=7)
    tempo = Tempo(60.0, 4, 4)
    audio_format = AudioFormat(48_000, 2)
    low_stress = stereo_array(renderer.render(template(slots=((0, 0.2), (4, 0.2))), tempo, audio_format, bar=0))
    high_stress = stereo_array(renderer.render(template(slots=((0, 0.9), (4, 0.9))), tempo, audio_format, bar=0))
    changed = np.any(low_stress != high_stress, axis=1)
    hat_frames = round(0.035 * audio_format.sample_rate_hz)
    expected = np.zeros(changed.shape[0], dtype=bool)
    for tick in (0, 4):
        onset = tick * 12_000
        expected[onset : onset + hat_frames] = True

    assert np.any(changed & expected)
    assert not np.any(changed & ~expected)


def test_boom_bap_is_reproducible_for_same_seed_and_bar() -> None:
    first = ProceduralBoomBapRenderer(seed=7).render(template(), Tempo(60.0, 4, 4), AudioFormat(48_000, 2), bar=3)
    second = ProceduralBoomBapRenderer(seed=7).render(template(), Tempo(60.0, 4, 4), AudioFormat(48_000, 2), bar=3)

    assert first.data == second.data
