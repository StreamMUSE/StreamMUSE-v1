"""Tests for generic event model."""

from __future__ import annotations

import pytest

from streammuse.domain.events import Event, EventKind, MusicalEventPayload, TextChunkPayload
from streammuse.domain.musical import MusicalEvent
from streammuse.domain.musical.converters import generic_event_to_musical_event, musical_event_to_generic_event
from streammuse.domain.musical.events import EventType


def test_musical_event_payload_validation() -> None:
    """Test MusicalEventPayload validation."""
    # Valid payload
    payload = MusicalEventPayload(
        pitch=60,
        event_type=EventType.NOTE_ON,
        velocity=100,
        channel=0,
        program=0,
    )
    assert payload.pitch == 60

    # Invalid pitch
    with pytest.raises(ValueError, match="Invalid pitch"):
        MusicalEventPayload(pitch=128, event_type=EventType.NOTE_ON)

    # Invalid velocity
    with pytest.raises(ValueError, match="Invalid velocity"):
        MusicalEventPayload(pitch=60, event_type=EventType.NOTE_ON, velocity=128)

    # Placeholder must have pitch=-1
    with pytest.raises(ValueError, match="Placeholder events must have pitch=-1"):
        MusicalEventPayload(pitch=60, event_type=EventType.NOTE_ON, is_placeholder=True)


def test_generic_event() -> None:
    """Test generic Event."""
    musical_payload = MusicalEventPayload(pitch=60, event_type=EventType.NOTE_ON)
    event = Event(tick=10, kind=EventKind.MUSICAL, payload=musical_payload)

    assert event.tick == 10
    assert event.kind == EventKind.MUSICAL
    assert event.is_musical is True
    assert event.is_text is False
    assert event.is_audio is False

    # Invalid tick
    with pytest.raises(ValueError, match="Invalid tick"):
        Event(tick=-1, kind=EventKind.MUSICAL, payload=musical_payload)


def test_musical_event_to_generic_event() -> None:
    """Test conversion from MusicalEvent to generic Event."""
    musical_event = MusicalEvent(
        tick=10,
        pitch=60,
        event_type=EventType.NOTE_ON,
        velocity=100,
        channel=1,
        program=2,
    )

    generic_event = musical_event_to_generic_event(musical_event)

    assert generic_event.tick == 10
    assert generic_event.kind == EventKind.MUSICAL
    assert isinstance(generic_event.payload, MusicalEventPayload)
    assert generic_event.payload.pitch == 60
    assert generic_event.payload.event_type == EventType.NOTE_ON
    assert generic_event.payload.velocity == 100
    assert generic_event.payload.channel == 1
    assert generic_event.payload.program == 2


def test_generic_event_to_musical_event() -> None:
    """Test conversion from generic Event to MusicalEvent."""
    musical_payload = MusicalEventPayload(
        pitch=60,
        event_type=EventType.NOTE_ON,
        velocity=100,
        channel=1,
        program=2,
    )
    generic_event = Event(tick=10, kind=EventKind.MUSICAL, payload=musical_payload)

    musical_event = generic_event_to_musical_event(generic_event)

    assert musical_event.tick == 10
    assert musical_event.pitch == 60
    assert musical_event.event_type == EventType.NOTE_ON
    assert musical_event.velocity == 100
    assert musical_event.channel == 1
    assert musical_event.program == 2


def test_generic_event_to_musical_event_invalid_kind() -> None:
    """Test that converting non-musical event raises error."""
    text_payload = TextChunkPayload(text="Hello")
    generic_event = Event(tick=10, kind=EventKind.TEXT_CHUNK, payload=text_payload)

    with pytest.raises(ValueError, match="Cannot convert"):
        generic_event_to_musical_event(generic_event)


def test_round_trip_conversion() -> None:
    """Test round-trip conversion: MusicalEvent -> Event -> MusicalEvent."""
    original = MusicalEvent(
        tick=10,
        pitch=60,
        event_type=EventType.NOTE_ON,
        velocity=100,
        channel=1,
        program=2,
        is_placeholder=False,
    )

    generic = musical_event_to_generic_event(original)
    converted_back = generic_event_to_musical_event(generic)

    assert converted_back.tick == original.tick
    assert converted_back.pitch == original.pitch
    assert converted_back.event_type == original.event_type
    assert converted_back.velocity == original.velocity
    assert converted_back.channel == original.channel
    assert converted_back.program == original.program
    assert converted_back.is_placeholder == original.is_placeholder
