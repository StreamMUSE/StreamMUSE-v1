"""Unit tests for domain.musical.events."""

import pytest

from streammuse.domain.musical.events import EventType, MusicalEvent


def test_event_type_enum() -> None:
    assert EventType.NOTE_ON.value == "note_on"
    assert EventType.NOTE_OFF.value == "note_off"


def test_musical_event_minimal() -> None:
    e = MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON)
    assert e.tick == 0
    assert e.pitch == 60
    assert e.event_type == EventType.NOTE_ON
    assert e.velocity == 100
    assert e.channel == 0
    assert e.program == 0
    assert e.is_placeholder is False


def test_musical_event_full() -> None:
    e = MusicalEvent(
        tick=10,
        pitch=72,
        event_type=EventType.NOTE_OFF,
        velocity=80,
        channel=1,
        program=2,
    )
    assert e.tick == 10
    assert e.pitch == 72
    assert e.velocity == 80
    assert e.channel == 1
    assert e.program == 2


def test_musical_event_immutable() -> None:
    e = MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON)
    with pytest.raises(AttributeError):
        e.pitch = 61  # type: ignore[misc]


def test_musical_event_invalid_tick() -> None:
    with pytest.raises(ValueError, match="Invalid tick"):
        MusicalEvent(tick=-1, pitch=60, event_type=EventType.NOTE_ON)


def test_musical_event_invalid_pitch_low() -> None:
    with pytest.raises(ValueError, match="Invalid pitch"):
        MusicalEvent(tick=0, pitch=-1, event_type=EventType.NOTE_ON)


def test_musical_event_invalid_pitch_high() -> None:
    with pytest.raises(ValueError, match="Invalid pitch"):
        MusicalEvent(tick=0, pitch=128, event_type=EventType.NOTE_ON)


def test_musical_event_invalid_velocity() -> None:
    with pytest.raises(ValueError, match="Invalid velocity"):
        MusicalEvent(
            tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=200
        )


def test_musical_event_invalid_channel() -> None:
    with pytest.raises(ValueError, match="Invalid channel"):
        MusicalEvent(
            tick=0, pitch=60, event_type=EventType.NOTE_ON, channel=16
        )


def test_musical_event_invalid_program() -> None:
    with pytest.raises(ValueError, match="Invalid program"):
        MusicalEvent(
            tick=0, pitch=60, event_type=EventType.NOTE_ON, program=128
        )


def test_musical_event_placeholder_valid() -> None:
    e = MusicalEvent(
        tick=5,
        pitch=-1,
        event_type=EventType.NOTE_ON,
        is_placeholder=True,
    )
    assert e.is_placeholder is True
    assert e.pitch == -1


def test_musical_event_placeholder_requires_pitch_minus_one() -> None:
    with pytest.raises(ValueError, match="Placeholder events must have pitch=-1"):
        MusicalEvent(
            tick=0,
            pitch=60,
            event_type=EventType.NOTE_ON,
            is_placeholder=True,
        )
