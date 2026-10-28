"""Optional chord controls preserve existing audio constructors."""

from dataclasses import FrozenInstanceError

import pytest

from streammuse.domain.rap.audio import AudioFormat, PcmAudio, PreparedRapBar, RapMixConfig


def test_chords_default_off_and_preserve_positional_config_arguments() -> None:
    config = RapMixConfig(0.8, 0.55, 1.0, 0.95, 7)

    assert config.chord_loop == "off"
    assert config.chord_gain == 0.20
    assert config.revision == 7
    assert RapMixConfig.with_drum_boost().chord_loop == "off"
    with pytest.raises(FrozenInstanceError):
        config.chord_gain = 1.0


@pytest.mark.parametrize("loop_id", ["off", "minor_cycle", "major_cycle"])
@pytest.mark.parametrize("gain", [0, 0.2, 2.0])
def test_config_accepts_known_loops_and_bounded_gains(loop_id, gain) -> None:
    config = RapMixConfig(chord_loop=loop_id, chord_gain=gain)

    assert config.chord_loop == loop_id
    assert config.chord_gain == gain


@pytest.mark.parametrize("fields", [
    {"chord_loop": "unknown"}, {"chord_loop": "OFF"}, {"chord_loop": None},
    {"chord_loop": True}, {"chord_loop": []},
    {"chord_gain": -0.1}, {"chord_gain": 2.1}, {"chord_gain": True},
    {"chord_gain": float("inf")}, {"chord_gain": float("nan")},
    {"chord_gain": 10 ** 400}, {"chord_gain": "0.2"},
])
def test_config_rejects_unknown_loops_and_invalid_gains(fields) -> None:
    with pytest.raises(ValueError):
        RapMixConfig(**fields)


def test_prepared_bar_appends_optional_stem_without_shifting_existing_fields() -> None:
    pcm = PcmAudio(AudioFormat(), 0, b"")
    config = RapMixConfig()
    prepared = PreparedRapBar(0, "one", "test", None, (), pcm, (), (), 5.0,
                              pcm, pcm, config, (), None, None)

    assert prepared.chord_stem is None
    assert prepared.mix_config is config
    assert prepared.timing is None
