"""Optional third-stem mixing and exact-bar re-rendering on revision."""

from dataclasses import replace

import numpy as np
import pytest

from streammuse.application.rap.chunk_audio import mix_audio_stems, remix_prepared_bar
from streammuse.domain.rap.audio import AudioFormat, PcmAudio, PreparedRapBar, RapMixConfig


def pcm(values, audio_format=AudioFormat()) -> PcmAudio:
    samples = np.asarray(values, dtype=np.float32)
    return PcmAudio(audio_format, len(samples), samples.tobytes())


def prepared_bar(bar=5, frames=12001) -> PreparedRapBar:
    vocal = pcm(np.full((frames, 2), 0.1))
    drum = pcm(np.full((frames, 2), -0.03))
    config = RapMixConfig()
    return PreparedRapBar(bar, "one", "test", None, (),
                          mix_audio_stems(vocal, drum, config), (), (), 5.0,
                          vocal_stem=vocal, drum_stem=drum, mix_config=config)


def test_third_stem_has_independent_gain_and_shared_master() -> None:
    vocal = pcm([[0.1, -0.1], [0.2, -0.2]])
    drum = pcm([[0.2, -0.2], [0.1, -0.1]])
    chord = pcm([[0.4, -0.4], [0.2, -0.2]])
    config = RapMixConfig(vocal_gain=0.5, drum_gain=1.0, master_gain=0.5,
                          chord_loop="minor_cycle", chord_gain=0.25)

    mixed = mix_audio_stems(vocal, drum, config, chord)

    assert np.frombuffer(mixed.data, dtype=np.float32) == pytest.approx([0.175, -0.175, 0.125, -0.125])
    muted = mix_audio_stems(vocal, drum, replace(config, chord_gain=0), chord)
    assert muted.data == mix_audio_stems(vocal, drum, config).data
    assert chord.data == pcm([[0.4, -0.4], [0.2, -0.2]]).data


def test_peak_limiter_includes_chords() -> None:
    silent = pcm([[0, 0], [0, 0]])
    chord = pcm([[0.8, -0.8], [0.4, -0.4]])
    config = RapMixConfig(chord_loop="major_cycle", chord_gain=2, peak_limit=0.4)

    mixed = mix_audio_stems(silent, silent, config, chord=chord)

    assert np.frombuffer(mixed.data, dtype=np.float32) == pytest.approx([0.4, -0.4, 0.2, -0.2])


@pytest.mark.parametrize("peak", [0.3, 0.95])
def test_off_is_bit_identical_to_legacy_mix_even_with_a_retained_chord(peak) -> None:
    rng = np.random.default_rng(10)
    vocal = pcm(rng.uniform(-0.8, 0.8, (97, 2)))
    drum = pcm(rng.uniform(-0.8, 0.8, (97, 2)))
    chord = pcm(np.full((97, 2), 0.4))
    config = RapMixConfig(peak_limit=peak)
    expected = (np.frombuffer(vocal.data, dtype=np.float32) * np.float32(0.8)
                + np.frombuffer(drum.data, dtype=np.float32) * np.float32(0.55)) * np.float32(1.0)
    maximum = float(np.max(np.abs(expected), initial=0.0))
    if maximum > peak:
        expected *= np.float32(peak / maximum)

    assert mix_audio_stems(vocal, drum, config).data == expected.tobytes()
    assert mix_audio_stems(vocal, drum, config, chord).data == expected.tobytes()


@pytest.mark.parametrize("chord", [
    pcm([[0, 0], [0, 0]]), pcm([[0, 0]], AudioFormat(24000, 2)),
    pcm([[float("nan"), 0]]), pcm([[float("inf"), 0]]),
])
def test_active_chord_must_match_stems_and_be_finite(chord) -> None:
    silent = pcm([[0, 0]])
    with pytest.raises(ValueError, match="same format and frame count|finite"):
        mix_audio_stems(silent, silent, RapMixConfig(chord_loop="minor_cycle"), chord)


def test_chord_gain_overflow_is_rejected_before_limiting() -> None:
    silent = pcm([[0, 0]])
    with pytest.raises(ValueError, match="finite"):
        mix_audio_stems(silent, silent, RapMixConfig(chord_loop="minor_cycle", chord_gain=2),
                        pcm([[3e38, -3e38]]))


def test_remix_first_render_creates_correct_chord_stem_and_records_revision() -> None:
    from streammuse.infrastructure.rap.chords import render_chord_loop

    original = prepared_bar()
    config = RapMixConfig(chord_loop="minor_cycle", revision=1)
    remixed = remix_prepared_bar(original, config)

    assert remixed.chord_stem == render_chord_loop("minor_cycle", 5, 12001, original.audio.format)
    assert remixed.audio == mix_audio_stems(original.vocal_stem, original.drum_stem, config, remixed.chord_stem)
    assert remixed.audio != original.audio
    assert remixed.vocal_stem is original.vocal_stem and remixed.drum_stem is original.drum_stem
    assert original.chord_stem is None
    assert remixed.mix_config.revision == 1
    assert remixed.render_latency_ms == original.render_latency_ms


def test_remix_loop_gain_and_absolute_bar_revisions_do_not_reuse_stale_chords() -> None:
    from streammuse.infrastructure.rap.chords import render_chord_loop

    original = prepared_bar()
    minor = remix_prepared_bar(original, RapMixConfig(chord_loop="minor_cycle", revision=1))
    major_config = RapMixConfig(chord_loop="major_cycle", chord_gain=0.4, revision=2)
    major = remix_prepared_bar(minor, major_config)
    moved = remix_prepared_bar(replace(major, bar=6), replace(major_config, revision=3))

    assert major.chord_stem != minor.chord_stem
    assert moved.chord_stem == render_chord_loop("major_cycle", 6, 12001, original.audio.format)
    assert moved.chord_stem != major.chord_stem
    assert moved.mix_config.revision == 3
    repeated = remix_prepared_bar(moved, moved.mix_config)
    assert repeated.audio == moved.audio
    assert repeated.chord_stem == moved.chord_stem
    assert repeated.chord_stem is not moved.chord_stem


def test_remix_off_clears_chord_and_restores_original_bytes() -> None:
    original = prepared_bar()
    enabled = remix_prepared_bar(original, RapMixConfig(chord_loop="minor_cycle", revision=1))
    disabled = remix_prepared_bar(enabled, RapMixConfig(revision=2))

    assert disabled.chord_stem is None
    assert disabled.audio == original.audio
    assert disabled.mix_config.revision == 2


def test_relocated_physical_bar_rerenders_even_when_config_is_unchanged() -> None:
    from streammuse.infrastructure.rap.chords import render_chord_loop

    config = RapMixConfig(chord_loop="minor_cycle", revision=4)
    original = remix_prepared_bar(prepared_bar(bar=0), config)
    relocated = remix_prepared_bar(replace(original, bar=3), config)

    assert relocated.chord_stem == render_chord_loop("minor_cycle", 3, 12001, original.audio.format)
    assert relocated.chord_stem != original.chord_stem
    assert relocated.audio != original.audio
    assert relocated.mix_config is config


def test_remix_uses_current_prepared_frame_count_after_tempo_change() -> None:
    from streammuse.infrastructure.rap.chords import render_chord_loop

    config = RapMixConfig(chord_loop="minor_cycle", revision=1)
    slow = remix_prepared_bar(prepared_bar(frames=12001), config)
    fast = replace(prepared_bar(frames=6001), chord_stem=slow.chord_stem)
    remixed = remix_prepared_bar(fast, replace(config, revision=2))

    assert remixed.chord_stem == render_chord_loop("minor_cycle", 5, 6001, fast.audio.format)
    assert remixed.audio.frame_count == 6001
