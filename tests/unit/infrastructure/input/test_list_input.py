"""Tests for ListInput adapter."""

import pytest

from streammuse.domain.musical import MusicalEvent, EventType
from streammuse.infrastructure.input import ListInput


def test_list_input_yields_events():
    """ListInput yields all events from the list."""
    events = [
        MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=80),
        MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_OFF, velocity=0),
    ]
    source = ListInput(events)
    out = list(source.read_events())
    assert len(out) == 2
    assert out[0].pitch == 60 and out[0].event_type == EventType.NOTE_ON
    assert out[1].pitch == 60 and out[1].event_type == EventType.NOTE_OFF


def test_list_input_close():
    """close() marks source closed; read_events returns empty iterator after close."""
    events = [MusicalEvent(tick=0, pitch=64, event_type=EventType.NOTE_ON)]
    source = ListInput(events)
    source.close()
    # After close, read_events yields nothing (source is closed)
    out = list(source.read_events())
    assert len(out) == 0


def test_list_input_satisfies_input_source_protocol():
    """ListInput has read_events() and close() per InputSource."""
    from streammuse.domain.interfaces import InputSource

    source = ListInput([])
    assert hasattr(source, "read_events")
    assert hasattr(source, "close")
    assert callable(source.read_events)
    assert callable(source.close)
