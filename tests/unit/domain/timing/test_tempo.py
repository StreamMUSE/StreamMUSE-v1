"""Unit tests for Tempo and MusicalTime."""

import pytest

from streammuse.domain.timing import MusicalTime, Tempo


def test_tempo_basic() -> None:
    t = Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4)
    assert t.bpm == 120.0
    assert t.ticks_per_beat == 4
    assert t.beats_per_bar == 4
    assert t.ticks_per_bar == 16


def test_tempo_seconds_per_tick() -> None:
    # 120 BPM, 4 ticks per beat -> 0.125 s per tick
    t = Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4)
    assert t.seconds_per_tick == pytest.approx(0.125)


def test_tempo_tick_to_seconds() -> None:
    t = Tempo(bpm=60.0, ticks_per_beat=4, beats_per_bar=4)
    # 60 BPM, 4 tpb -> 1 beat = 1 s -> 1 tick = 0.25 s
    assert t.tick_to_seconds(4) == pytest.approx(1.0)
    assert t.tick_to_seconds(8) == pytest.approx(2.0)


def test_tempo_seconds_to_tick() -> None:
    t = Tempo(bpm=60.0, ticks_per_beat=4, beats_per_bar=4)
    assert t.seconds_to_tick(1.0) == 4
    assert t.seconds_to_tick(2.5) == 10


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bpm": 0}, "BPM must be positive"),
        ({"bpm": -1}, "BPM must be positive"),
        ({"ticks_per_beat": 0}, "Ticks per beat must be positive"),
        ({"beats_per_bar": 0}, "Beats per bar must be positive"),
    ],
)
def test_tempo_invalid_fields(kwargs, match) -> None:
    base = {"bpm": 120.0, "ticks_per_beat": 4, "beats_per_bar": 4}
    with pytest.raises(ValueError, match=match):
        Tempo(**{**base, **kwargs})


def test_musical_time_from_tick_zero() -> None:
    t = Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4)
    mt = MusicalTime.from_tick(0, t)
    assert mt.tick == 0
    assert mt.bar == 0
    assert mt.beat == 0
    assert mt.tick_in_beat == 0


def test_musical_time_from_tick_one_bar() -> None:
    t = Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4)
    mt = MusicalTime.from_tick(16, t)
    assert mt.tick == 16
    assert mt.bar == 1
    assert mt.beat == 0
    assert mt.tick_in_beat == 0


def test_musical_time_from_tick_mid_bar() -> None:
    t = Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4)
    # tick 10 = bar 0, tick_in_bar 10 -> beat 2, tick_in_beat 2
    mt = MusicalTime.from_tick(10, t)
    assert mt.tick == 10
    assert mt.bar == 0
    assert mt.beat == 2
    assert mt.tick_in_beat == 2


def test_musical_time_roundtrip() -> None:
    t = Tempo(bpm=100.0, ticks_per_beat=4, beats_per_bar=3)
    for tick in (0, 1, 4, 11, 12, 36):
        mt = MusicalTime.from_tick(tick, t)
        assert mt.tick == tick
