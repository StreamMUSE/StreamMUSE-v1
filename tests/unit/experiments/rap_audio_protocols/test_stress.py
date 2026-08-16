from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from streammuse.experiments.rap_audio_protocols.contracts import SyllableTarget
from streammuse.experiments.rap_audio_protocols.stress import apply_stress_envelope


def _syllable(*, target_seconds: float, target_stress: float) -> SyllableTarget:
    return SyllableTarget(
        word="flow",
        index_in_word=0,
        phonemes=("F", "L", "OW1"),
        lexical_stress=1,
        target_stress=target_stress,
        boundary_strength=0,
        absolute_tick=0,
        tick_in_chunk=0,
        target_seconds=target_seconds,
    )


def test_apply_stress_envelope_emphasises_strong_syllables_without_length_change() -> None:
    sample_rate_hz = 1_000
    samples = np.full(1_000, 0.2, dtype=np.float32)
    syllables = (
        _syllable(target_seconds=0.0, target_stress=0.0),
        replace(_syllable(target_seconds=0.5, target_stress=1.0), index_in_word=1),
    )

    rendered = apply_stress_envelope(
        samples,
        sample_rate_hz=sample_rate_hz,
        syllables=syllables,
        weak_db=-1.0,
        strong_db=2.5,
        ramp_seconds=0.030,
    )

    weak_rms = np.sqrt(np.mean(np.square(rendered.samples[100:400])))
    strong_rms = np.sqrt(np.mean(np.square(rendered.samples[600:900])))
    assert len(rendered.samples) == len(samples)
    assert rendered.samples.dtype == np.float32
    assert strong_rms / weak_rms == pytest.approx(10 ** (3.5 / 20.0), rel=0.01)
    assert rendered.output_rms == pytest.approx(rendered.input_rms, rel=0.01)


def test_apply_stress_envelope_smooths_gain_transitions_and_prevents_clipping() -> None:
    samples = np.full(1_000, 0.95, dtype=np.float32)
    syllables = (
        _syllable(target_seconds=0.0, target_stress=0.0),
        replace(_syllable(target_seconds=0.5, target_stress=1.0), index_in_word=1),
    )

    rendered = apply_stress_envelope(
        samples,
        sample_rate_hz=1_000,
        syllables=syllables,
        ramp_seconds=0.030,
    )

    assert np.max(np.abs(rendered.samples)) <= 0.999
    assert np.max(np.abs(np.diff(rendered.gain_envelope[480:520]))) < 0.05
    assert rendered.peak_limited


def test_apply_stress_envelope_rejects_non_monotonic_syllable_targets() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        apply_stress_envelope(
            np.ones(1_000, dtype=np.float32),
            sample_rate_hz=1_000,
            syllables=(
                _syllable(target_seconds=0.5, target_stress=1.0),
                _syllable(target_seconds=0.4, target_stress=0.0),
            ),
        )
