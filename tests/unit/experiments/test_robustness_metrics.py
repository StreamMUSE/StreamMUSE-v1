from __future__ import annotations

import math

import mido
import pytest

from streammuse.experiments.robustness_metrics import (
    Roll,
    bootstrap_song_mean,
    coverage_metrics,
    dissonance_metrics,
    load_midi_roll,
    rhythmic_metrics,
    sensitivity_metrics,
    transform_roll,
    write_roll_midi,
)


def _roll(end_tick: int, notes: list[tuple[int, int, int]]) -> Roll:
    sustain = {
        (tick, pitch)
        for start, stop, pitch in notes
        for tick in range(start, stop)
    }
    onsets = {(start, pitch) for start, _stop, pitch in notes}
    return Roll(end_tick, frozenset(sustain), frozenset(onsets))


def test_canonical_roll_midi_roundtrip_preserves_note_spans(tmp_path):
    original = _roll(12, [(0, 4, 60), (4, 7, 64), (9, 12, 67)])
    path = tmp_path / "roll.mid"

    write_roll_midi(original, path, bpm=120)
    loaded = load_midi_roll(path, end_tick=12)

    assert loaded == original


def test_midi_loader_filters_named_tracks_and_channel_9_drums(tmp_path):
    midi = mido.MidiFile(type=1, ticks_per_beat=480)

    melody = mido.MidiTrack()
    melody.append(mido.MetaMessage("track_name", name="Melody", time=0))
    melody.append(mido.Message("note_on", note=72, velocity=80, channel=0, time=0))
    melody.append(mido.Message("note_off", note=72, velocity=0, channel=0, time=480))
    midi.tracks.append(melody)

    accompaniment = mido.MidiTrack()
    accompaniment.append(mido.MetaMessage("track_name", name="Accompaniment Piano", time=0))
    accompaniment.append(mido.Message("note_on", note=60, velocity=80, channel=1, time=0))
    accompaniment.append(mido.Message("note_off", note=60, velocity=0, channel=1, time=480))
    midi.tracks.append(accompaniment)

    drums = mido.MidiTrack()
    drums.append(mido.MetaMessage("track_name", name="Accompaniment Drums", time=0))
    drums.append(mido.Message("note_on", note=36, velocity=80, channel=9, time=0))
    drums.append(mido.Message("note_off", note=36, velocity=0, channel=9, time=480))
    midi.tracks.append(drums)

    path = tmp_path / "tracks.mid"
    midi.save(path)

    filtered = load_midi_roll(path, end_tick=4, track_name_contains="accompaniment")
    all_pitched = load_midi_roll(path, end_tick=4)

    assert filtered == _roll(4, [(0, 4, 60)])
    assert all_pitched == _roll(4, [(0, 4, 60), (0, 4, 72)])


def test_consonant_identity_and_targeted_m2_tritone_controls_have_expected_direction():
    melody = _roll(8, [(0, 8, 60)])
    consonant = _roll(8, [(0, 8, 60)])

    identity = transform_roll(consonant, "identity")
    minor_second = transform_roll(consonant, "harmonic_m2")
    tritone = transform_roll(consonant, "harmonic_tt")

    base = dissonance_metrics(melody, identity)
    m2 = dissonance_metrics(melody, minor_second)
    tt = dissonance_metrics(melody, tritone)

    assert identity is consonant
    assert base["pair_ticks"] == 8
    assert base["D_micro"] == 0.0
    assert base["D_macro_coactive_tick"] == 0.0
    for control in (m2, tt):
        assert control["pair_ticks"] == 8
        assert control["D_micro"] == 1.0
        assert control["D_macro_coactive_tick"] == 1.0
        assert control["harmonic_pair_coverage"] == 1.0
        assert not control["coverage_failure"]


def test_fully_empty_and_partially_covered_accompaniment_keep_D_na_separate_from_coverage():
    melody = _roll(8, [(0, 8, 60)])
    empty = Roll(8, frozenset(), frozenset())
    partial = _roll(8, [(0, 4, 60)])

    empty_coverage = coverage_metrics(empty)
    empty_quality = dissonance_metrics(melody, empty)
    partial_coverage = coverage_metrics(partial)
    partial_quality = dissonance_metrics(melody, partial)

    assert empty_coverage == {
        "onsets_per_beat": 0.0,
        "active_pitch_ticks_per_beat": 0.0,
        "empty_beat_ratio": 1.0,
        "active_tick_count": 0,
        "beat_count": 2,
        "fully_empty": True,
    }
    assert empty_quality["D_micro"] is None
    assert empty_quality["D_macro_coactive_tick"] is None
    assert empty_quality["coverage_failure"]
    assert empty_quality["na_reason"] == "zero_coactive_pair_denominator"
    assert empty_quality["harmonic_pair_coverage"] == 0.0

    assert partial_coverage["fully_empty"] is False
    assert partial_coverage["empty_beat_ratio"] == 0.5
    assert partial_coverage["active_pitch_ticks_per_beat"] == 2.0
    assert partial_quality["D_micro"] == 0.0
    assert partial_quality["pair_ticks"] == 4
    assert partial_quality["harmonic_pair_coverage"] == 0.5
    assert not partial_quality["coverage_failure"]


def test_coverage_respects_an_explicit_zero_length_analysis_window():
    accompaniment = _roll(8, [(0, 8, 60)])

    result = coverage_metrics(accompaniment, end_tick=0)

    assert result["beat_count"] == 1
    assert result["fully_empty"]
    assert result["active_tick_count"] == 0
    assert result["active_pitch_ticks_per_beat"] == 0.0
    assert result["empty_beat_ratio"] == 1.0


def test_dense_chords_report_pair_weighted_micro_and_tick_weighted_macro_D():
    melody = Roll(
        2,
        sustain=frozenset({(0, 60), (0, 64), (1, 60)}),
        onsets=frozenset({(0, 60), (0, 64), (1, 60)}),
    )
    accompaniment = Roll(
        2,
        sustain=frozenset({(0, 60), (0, 61), (0, 66), (1, 60)}),
        onsets=frozenset({(0, 60), (0, 61), (0, 66), (1, 60)}),
    )

    result = dissonance_metrics(melody, accompaniment)

    assert result["pair_ticks"] == 7
    assert result["dissonant_pair_ticks"] == 3
    assert result["D_micro"] == pytest.approx(3 / 7)
    assert result["D_macro_coactive_tick"] == pytest.approx(0.25)
    assert result["coactive_tick_count"] == 2
    assert result["harmonic_pair_coverage"] == 1.0


def test_rhythm_shift_moves_onsets_one_tick_and_is_detected_by_both_endpoints():
    original = _roll(12, [(0, 1, 60), (3, 4, 62), (8, 9, 64)])
    shifted = transform_roll(original, "rhythm_shift")

    sensitivity = sensitivity_metrics(original, shifted)
    rhythm = rhythmic_metrics(original, shifted)

    assert shifted.onsets == frozenset({(1, 60), (4, 62), (9, 64)})
    assert sensitivity["onset"]["f1"] == 0.0
    assert sensitivity["onset_distance"]["mean_ticks"] == 1.0
    assert rhythm["nearest_melody_onset_distance_ticks"] == 1.0
    assert rhythm["ioi_correlation"] == pytest.approx(1.0)


def test_identity_dropout_and_empty_controls_have_distinct_sensitivity_and_coverage():
    original = _roll(
        8,
        [(0, 2, 60), (2, 4, 61), (4, 6, 62), (6, 8, 63)],
    )
    identity = transform_roll(original, "identity")
    dropout = transform_roll(original, "coverage_dropout")
    empty = transform_roll(original, "coverage_empty")

    assert identity is original
    assert sensitivity_metrics(original, identity)["sustain"]["f1"] == 1.0
    assert dropout.onsets == frozenset({(0, 60), (4, 62)})
    assert len(dropout.sustain) == len(original.sustain) // 2
    assert coverage_metrics(dropout)["active_pitch_ticks_per_beat"] == (
        coverage_metrics(original)["active_pitch_ticks_per_beat"] / 2
    )
    assert sensitivity_metrics(original, dropout)["sustain"]["f1"] == pytest.approx(2 / 3)
    assert empty.sustain == frozenset()
    assert empty.onsets == frozenset()
    assert coverage_metrics(empty)["fully_empty"]
    assert sensitivity_metrics(original, empty)["sustain"]["flag"] == "one_empty"

    with pytest.raises(ValueError, match="unknown control transform"):
        transform_roll(original, "not-a-control")


def test_both_empty_sensitivity_is_explicitly_na_not_a_perfect_match():
    empty = Roll(4, frozenset(), frozenset())

    result = sensitivity_metrics(empty, empty)

    assert result["sustain"] == {"jaccard": None, "f1": None, "flag": "both_empty"}
    assert result["onset"] == {"jaccard": None, "f1": None, "flag": "both_empty"}
    assert result["onset_distance"] == {"mean_ticks": None, "flag": "both_empty"}


def test_song_block_bootstrap_is_equal_weighted_deterministic_and_reports_loo_range():
    effects = {"song-1": 0.0, "song-2": 0.0, "song-3": 0.0, "song-4": 0.0, "song-5": 10.0}

    first = bootstrap_song_mean(effects, seed=2026071301, iterations=2000)
    second = bootstrap_song_mean(effects, seed=2026071301, iterations=2000)

    assert first == second
    assert first["estimate"] == 2.0
    assert first["raw_song_effects"] == effects
    assert first["valid_song_count"] == 5
    assert first["leave_one_song_out_range"] == [0.0, 2.5]
    assert first["interval"][0] <= first["estimate"] <= first["interval"][1]
    assert first["interpretation"] == "descriptive_song_block_bootstrap"


@pytest.mark.parametrize(
    "effects, expected_valid, expected_pattern",
    [
        (
            {"song-1": 0.1, "song-2": None, "song-3": 0.3},
            2,
            {"song-1": False, "song-2": True, "song-3": False},
        ),
        ({}, 0, {}),
    ],
)
def test_bootstrap_with_na_or_no_song_blocks_fails_closed_without_complete_case_deletion(
    effects, expected_valid, expected_pattern
):
    result = bootstrap_song_mean(effects, seed=2026071301, iterations=100)

    assert result["estimate"] is None
    assert result["interval"] is None
    assert result["valid_song_count"] == expected_valid
    assert result["na_pattern"] == expected_pattern
    assert not any(
        isinstance(value, float) and math.isnan(value)
        for value in result.values()
    )
