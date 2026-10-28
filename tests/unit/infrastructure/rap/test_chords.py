"""Deterministic, exact-length local chord accompaniment."""

import numpy as np
import pytest

from streammuse.domain.rap.audio import AudioFormat


def test_catalog_has_only_the_two_hand_authored_cycles() -> None:
    from streammuse.domain.rap.chords import CHORD_LOOPS, CHORD_LOOP_IDS

    assert CHORD_LOOP_IDS == ("off", "minor_cycle", "major_cycle")
    assert set(CHORD_LOOPS) == {"minor_cycle", "major_cycle"}
    for loop in CHORD_LOOPS.values():
        assert loop.name
        assert len(loop.chords) == 4
        assert all(len(chord) == 4 for chord in loop.chords)
    with pytest.raises(TypeError):
        CHORD_LOOPS["unexpected"] = CHORD_LOOPS["minor_cycle"]


@pytest.mark.parametrize("loop_id, pitch_classes", [
    ("minor_cycle", ({9, 0, 4}, {5, 9, 0}, {0, 4, 7}, {7, 11, 2})),
    ("major_cycle", ({0, 4, 7}, {7, 11, 2}, {9, 0, 4}, {5, 9, 0})),
])
def test_voicings_match_the_ui_progression_labels(loop_id, pitch_classes) -> None:
    from streammuse.domain.rap.chords import CHORD_LOOPS

    actual = tuple({pitch % 12 for pitch in chord} for chord in CHORD_LOOPS[loop_id].chords)
    assert actual == pitch_classes


@pytest.mark.parametrize("loop_id", ["minor_cycle", "major_cycle"])
@pytest.mark.parametrize("bpm", [40.0, 117.3, 300.0])
@pytest.mark.parametrize("audio_format", [AudioFormat(8000, 1), AudioFormat(24000, 2), AudioFormat()])
def test_render_is_nonblank_finite_and_exact_at_any_bar_length(loop_id, bpm, audio_format) -> None:
    from streammuse.infrastructure.rap.chords import render_chord_loop

    frames = round(240 / bpm * audio_format.sample_rate_hz)
    audio = render_chord_loop(loop_id, 9, frames, audio_format)
    samples = np.frombuffer(audio.data, dtype=np.float32).reshape(frames, audio_format.channels)

    assert audio.format == audio_format
    assert audio.frame_count == frames
    assert np.isfinite(samples).all()
    assert 0.01 < np.sqrt(np.mean(samples ** 2))
    assert np.max(np.abs(samples)) <= 0.6
    assert np.all(samples[0] == 0)
    assert np.all(samples[-1] == 0)


@pytest.mark.parametrize("loop_id", ["minor_cycle", "major_cycle"])
def test_absolute_bar_selects_cycle_phase_independent_of_render_order(loop_id) -> None:
    from streammuse.infrastructure.rap.chords import render_chord_loop

    original = render_chord_loop(loop_id, 2, 16001, AudioFormat())
    other_bar = render_chord_loop(loop_id, 1, 16001, AudioFormat())
    render_chord_loop("major_cycle", 11, 9000, AudioFormat())

    assert original.data != other_bar.data
    assert original == render_chord_loop(loop_id, 2, 16001, AudioFormat())
    assert original == render_chord_loop(loop_id, 6, 16001, AudioFormat())


def test_loop_selection_changes_audible_identity() -> None:
    from streammuse.infrastructure.rap.chords import render_chord_loop

    minor = render_chord_loop("minor_cycle", 0, 48000, AudioFormat())
    major = render_chord_loop("major_cycle", 0, 48000, AudioFormat())

    assert minor.data != major.data


@pytest.mark.parametrize("loop_id", ["minor_cycle", "major_cycle"])
def test_bar_edges_have_smooth_attack_and_release(loop_id) -> None:
    from streammuse.infrastructure.rap.chords import render_chord_loop

    audio = render_chord_loop(loop_id, 3, 48000, AudioFormat())
    samples = np.frombuffer(audio.data, dtype=np.float32).reshape(-1, 2)[:, 0]

    assert samples[0] == samples[-1] == 0.0
    assert np.max(np.abs(samples[:48])) < 0.01
    assert np.max(np.abs(samples[-48:])) < 0.01
    assert abs(samples[1] - samples[0]) < 1e-4
    assert abs(samples[-1] - samples[-2]) < 1e-4
    assert np.sqrt(np.mean(samples[4800:9600] ** 2)) > 0.01


@pytest.mark.parametrize("frames", [0, 1, 2, 3, 7])
def test_tiny_buffers_remain_exact_finite_and_click_safe(frames) -> None:
    from streammuse.infrastructure.rap.chords import render_chord_loop

    audio = render_chord_loop("minor_cycle", 0, frames, AudioFormat())
    samples = np.frombuffer(audio.data, dtype=np.float32)

    assert audio.frame_count == frames
    assert len(audio.data) == frames * 8
    assert np.isfinite(samples).all()
    if frames:
        assert np.all(samples[:2] == 0) and np.all(samples[-2:] == 0)


def test_off_renders_exact_silence() -> None:
    from streammuse.infrastructure.rap.chords import render_chord_loop

    assert render_chord_loop("off", 41, 137, AudioFormat()).data == bytes(137 * 8)


@pytest.mark.parametrize("arguments", [
    ("unknown", 0, 100, AudioFormat()),
    (None, 0, 100, AudioFormat()),
    ([], 0, 100, AudioFormat()),
    ("minor_cycle", -1, 100, AudioFormat()),
    ("minor_cycle", True, 100, AudioFormat()),
    ("minor_cycle", 1.5, 100, AudioFormat()),
    ("minor_cycle", 0, -1, AudioFormat()),
    ("minor_cycle", 0, True, AudioFormat()),
    ("minor_cycle", 0, 1.5, AudioFormat()),
    ("minor_cycle", 0, 100, AudioFormat(0, 2)),
    ("minor_cycle", 0, 100, AudioFormat(48000, 0)),
    ("minor_cycle", 0, 100, AudioFormat(48000, 2, 2)),
])
def test_renderer_rejects_invalid_arguments(arguments) -> None:
    from streammuse.infrastructure.rap.chords import render_chord_loop

    with pytest.raises(ValueError):
        render_chord_loop(*arguments)
