"""Unit tests for domain.musical Note and event↔note conversion."""

import pytest

from streammuse.domain.musical.events import EventType, MusicalEvent, Note


def test_note_minimal() -> None:
    n = Note(pitch=60, tick=0, duration=4)
    assert n.pitch == 60
    assert n.tick == 0
    assert n.duration == 4
    assert n.velocity == 100
    assert n.channel == 0
    assert n.program == 0
    assert n.is_placeholder is False


def test_note_full() -> None:
    n = Note(
        pitch=72,
        tick=10,
        duration=8,
        velocity=80,
        channel=1,
        program=2,
    )
    assert n.pitch == 72
    assert n.tick == 10
    assert n.duration == 8
    assert n.velocity == 80
    assert n.channel == 1
    assert n.program == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tick": -1}, "Invalid tick"),
        ({"duration": 0}, "Invalid duration"),
        ({"duration": -1}, "Invalid duration"),
        ({"pitch": 128}, "Invalid pitch"),
    ],
)
def test_note_invalid_fields(kwargs, match) -> None:
    base = {"pitch": 60, "tick": 0, "duration": 4}
    with pytest.raises(ValueError, match=match):
        Note(**{**base, **kwargs})


def test_note_placeholder_valid() -> None:
    n = Note(
        pitch=-1,
        tick=0,
        duration=4,
        is_placeholder=True,
    )
    assert n.is_placeholder is True
    assert n.pitch == -1


def test_note_placeholder_requires_pitch_minus_one() -> None:
    with pytest.raises(ValueError, match="Placeholder notes must have pitch=-1"):
        Note(pitch=60, tick=0, duration=4, is_placeholder=True)


def test_note_to_events() -> None:
    n = Note(pitch=60, tick=0, duration=4, velocity=90)
    events = n.to_events()
    assert len(events) == 2
    on, off = events[0], events[1]
    assert on.event_type == EventType.NOTE_ON
    assert on.tick == 0
    assert on.pitch == 60
    assert on.velocity == 90
    assert off.event_type == EventType.NOTE_OFF
    assert off.tick == 4
    assert off.pitch == 60


def test_note_to_events_roundtrip() -> None:
    n = Note(pitch=64, tick=10, duration=6, velocity=70, channel=1, program=3)
    events = n.to_events()
    back = Note.from_events(events[0], events[1])
    assert back.pitch == n.pitch
    assert back.tick == n.tick
    assert back.duration == n.duration
    assert back.velocity == n.velocity
    assert back.channel == n.channel
    assert back.program == n.program


def test_from_events_valid() -> None:
    on = MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON)
    off = MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_OFF)
    n = Note.from_events(on, off)
    assert n.pitch == 60
    assert n.tick == 0
    assert n.duration == 4


def test_from_events_first_must_be_note_on() -> None:
    on = MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_OFF)
    off = MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_ON)
    with pytest.raises(ValueError, match="First event must be NOTE_ON"):
        Note.from_events(on, off)


def test_from_events_second_must_be_note_off() -> None:
    on = MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON)
    off = MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_ON)
    with pytest.raises(ValueError, match="Second event must be NOTE_OFF"):
        Note.from_events(on, off)


def test_from_events_same_pitch() -> None:
    on = MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON)
    off = MusicalEvent(tick=4, pitch=64, event_type=EventType.NOTE_OFF)
    with pytest.raises(ValueError, match="Events must have same pitch"):
        Note.from_events(on, off)


def test_from_events_off_after_on() -> None:
    on = MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_ON)
    off = MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_OFF)
    with pytest.raises(ValueError, match="note_off tick must be after note_on"):
        Note.from_events(on, off)


def test_from_events_preserves_channel_program() -> None:
    on = MusicalEvent(
        tick=0,
        pitch=60,
        event_type=EventType.NOTE_ON,
        velocity=80,
        channel=2,
        program=5,
    )
    off = MusicalEvent(
        tick=8,
        pitch=60,
        event_type=EventType.NOTE_OFF,
        channel=2,
        program=5,
    )
    n = Note.from_events(on, off)
    assert n.channel == 2
    assert n.program == 5
    assert n.velocity == 80
    assert n.duration == 8
