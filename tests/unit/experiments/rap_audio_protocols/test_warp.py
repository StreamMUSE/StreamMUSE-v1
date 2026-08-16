from __future__ import annotations

import hashlib

import numpy as np
import pytest

from streammuse.experiments.rap_audio_protocols.contracts import SyllableTarget
from streammuse.experiments.rap_audio_protocols.warp import (
    PhoneInterval,
    is_arpabet_vowel,
    match_vowel_anchors,
    parse_textgrid_phone_intervals,
    piecewise_pitch_preserving_warp,
)


def _syllable(word: str, phonemes: tuple[str, ...], *, target_seconds: float) -> SyllableTarget:
    return SyllableTarget(
        word=word,
        index_in_word=0,
        phonemes=phonemes,
        lexical_stress=1,
        target_stress=1.0,
        boundary_strength=0,
        absolute_tick=0,
        tick_in_chunk=0,
        target_seconds=target_seconds,
    )


def _impulse_stretcher(samples: np.ndarray, target_frames: int, sample_rate_hz: int) -> np.ndarray:
    del sample_rate_hz
    output = np.zeros(target_frames, dtype=np.float32)
    if len(samples) == 0 or target_frames == 0:
        return output
    if len(samples) == 1:
        output[:] = samples[0]
        return output
    nonzero = np.flatnonzero(np.abs(samples) > 0.5)
    for index in nonzero:
        mapped = round(index * (target_frames - 1) / (len(samples) - 1))
        output[mapped] = samples[index]
    return output


def test_is_arpabet_vowel_accepts_stressed_and_unstressed_phones() -> None:
    assert is_arpabet_vowel("AA1")
    assert is_arpabet_vowel("IY0")
    assert not is_arpabet_vowel("SH")


def test_parse_textgrid_phone_intervals_reads_only_long_form_phones_tier() -> None:
    text = """
    File type = "ooTextFile"
    Object class = "TextGrid"

    xmin = 0
    xmax = 0.42
    tiers? <exists>
    size = 2
    item []:
        item [1]:
            class = "IntervalTier"
            name = "words"
            xmin = 0
            xmax = 0.42
            intervals: size = 1
            intervals [1]:
                xmin = 0.00
                xmax = 0.42
                text = "beat"
        item [2]:
            class = "IntervalTier"
            name = "phones"
            xmin = 0
            xmax = 0.42
            intervals: size = 4
            intervals [1]:
                xmin = 0.00
                xmax = 0.08
                text = "B"
            intervals [2]:
                xmin = 0.08
                xmax = 0.18
                text = "IY1"
            intervals [3]:
                xmin = 0.18
                xmax = 0.25
                text = ""
            intervals [4]:
                xmin = 0.25
                xmax = 0.42
                text = "L"
    """

    intervals = parse_textgrid_phone_intervals(text)

    assert [interval.phone for interval in intervals] == ["B", "IY1", "L"]
    assert intervals[1].start_seconds == pytest.approx(0.08)
    assert intervals[1].end_seconds == pytest.approx(0.18)


def test_parse_textgrid_phone_intervals_supports_short_form_praat_textgrid() -> None:
    text = '''
    File type = "ooTextFile short"
    "TextGrid"

    0
    0.42
    <exists>
    2
    "IntervalTier"
    "words"
    0
    0.42
    1
    0
    0.42
    "beat"
    "IntervalTier"
    "phones"
    0
    0.42
    4
    0.00
    0.08
    "B"
    0.08
    0.18
    "IY1"
    0.18
    0.25
    ""
    0.25
    0.42
    "T"
    '''

    intervals = parse_textgrid_phone_intervals(text)

    assert [interval.phone for interval in intervals] == ["B", "IY1", "T"]
    assert intervals[1].start_seconds == pytest.approx(0.08)
    assert intervals[1].end_seconds == pytest.approx(0.18)


def test_match_vowel_anchors_rejects_source_target_count_mismatches() -> None:
    intervals = (
        PhoneInterval(start_seconds=0.00, end_seconds=0.05, phone="B"),
        PhoneInterval(start_seconds=0.05, end_seconds=0.11, phone="IY1"),
    )
    syllables = (
        _syllable("beat", ("B", "IY1", "T"), target_seconds=0.10),
        _syllable("flow", ("F", "L", "OW1"), target_seconds=0.25),
    )

    with pytest.raises(ValueError, match="aligned vowel count"):
        match_vowel_anchors(intervals, syllables, sample_rate_hz=1_000)


def test_piecewise_pitch_preserving_warp_rejects_non_monotonic_anchor_targets() -> None:
    anchors = (
        _syllable("beat", ("B", "IY1", "T"), target_seconds=0.30),
        _syllable("flow", ("F", "L", "OW1"), target_seconds=0.25),
    )
    matched = (
        match_vowel_anchors(
            (
                PhoneInterval(start_seconds=0.10, end_seconds=0.16, phone="IY1"),
                PhoneInterval(start_seconds=0.30, end_seconds=0.38, phone="OW1"),
            ),
            anchors,
            sample_rate_hz=1_000,
        )
    )

    with pytest.raises(ValueError, match="monotonic"):
        piecewise_pitch_preserving_warp(
            np.zeros(1_000, dtype=np.float32),
            sample_rate_hz=1_000,
            anchors=matched,
            target_frame_count=1_000,
            stretch_region=_impulse_stretcher,
            crossfade_seconds=0.0,
            source_sha256="a" * 64,
        )


def test_piecewise_pitch_preserving_warp_rejects_regions_shorter_than_ten_ms() -> None:
    syllables = (
        _syllable("beat", ("B", "IY1", "T"), target_seconds=0.020),
        _syllable("flow", ("F", "L", "OW1"), target_seconds=0.040),
    )
    matched = match_vowel_anchors(
        (
            PhoneInterval(start_seconds=0.000, end_seconds=0.020, phone="IY1"),
            PhoneInterval(start_seconds=0.012, end_seconds=0.024, phone="OW1"),
        ),
        syllables,
        sample_rate_hz=1_000,
    )

    with pytest.raises(ValueError, match="10 ms"):
        piecewise_pitch_preserving_warp(
            np.zeros(100, dtype=np.float32),
            sample_rate_hz=1_000,
            anchors=matched,
            target_frame_count=100,
            stretch_region=_impulse_stretcher,
            crossfade_seconds=0.0,
            source_sha256="b" * 64,
        )


def test_piecewise_pitch_preserving_warp_hits_target_anchors_and_keeps_exact_frame_count() -> None:
    sample_rate_hz = 1_000
    samples = np.zeros(1_000, dtype=np.float32)
    samples[110] = 1.0
    samples[610] = 1.0
    source_sha256 = hashlib.sha256(b"moss-source-wav").hexdigest()
    matched = match_vowel_anchors(
        (
            PhoneInterval(start_seconds=0.100, end_seconds=0.140, phone="IY1"),
            PhoneInterval(start_seconds=0.600, end_seconds=0.640, phone="OW1"),
        ),
        (
            _syllable("beat", ("B", "IY1", "T"), target_seconds=0.250),
            _syllable("flow", ("F", "L", "OW1"), target_seconds=0.650),
        ),
        sample_rate_hz=sample_rate_hz,
    )

    warped = piecewise_pitch_preserving_warp(
        samples,
        sample_rate_hz=sample_rate_hz,
        anchors=matched,
        target_frame_count=1_000,
        stretch_region=_impulse_stretcher,
        crossfade_seconds=0.0,
        source_sha256=source_sha256,
    )

    assert len(warped.samples) == 1_000
    assert warped.source_sha256 == source_sha256
    assert [anchor.target_sample for anchor in warped.anchor_map] == [250, 650]
    assert warped.samples[250] == pytest.approx(1.0)
    assert warped.samples[650] == pytest.approx(1.0)
    assert all(region.stretch_ratio > 0 for region in warped.stretch_regions)


def test_piecewise_pitch_preserving_warp_preserves_boundary_impulses_with_default_crossfade() -> None:
    sample_rate_hz = 1_000
    samples = np.zeros(1_000, dtype=np.float32)
    samples[110] = 1.0
    samples[610] = 1.0
    matched = match_vowel_anchors(
        (
            PhoneInterval(start_seconds=0.100, end_seconds=0.140, phone="IY1"),
            PhoneInterval(start_seconds=0.600, end_seconds=0.640, phone="OW1"),
        ),
        (
            _syllable("beat", ("B", "IY1", "T"), target_seconds=0.250),
            _syllable("flow", ("F", "L", "OW1"), target_seconds=0.650),
        ),
        sample_rate_hz=sample_rate_hz,
    )

    warped = piecewise_pitch_preserving_warp(
        samples,
        sample_rate_hz=sample_rate_hz,
        anchors=matched,
        target_frame_count=1_000,
        stretch_region=_impulse_stretcher,
        source_sha256="c" * 64,
    )

    assert len(warped.samples) == 1_000
    assert np.flatnonzero(np.abs(warped.samples) > 0.5).tolist() == [250, 650]
    assert warped.samples[250] == pytest.approx(1.0)
    assert warped.samples[650] == pytest.approx(1.0)
