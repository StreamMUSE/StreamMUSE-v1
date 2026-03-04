"""Unit tests for events_to_notes (close-at-horizon policy)."""

import pytest

from streammuse.domain.musical.converters import events_to_notes
from streammuse.domain.musical.events import EventType, MusicalEvent, Note


def test_paired_events_to_one_note() -> None:
    events = [
        MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON),
        MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_OFF),
    ]
    notes = events_to_notes(events, horizon_tick=100)
    assert len(notes) == 1
    assert notes[0].pitch == 60
    assert notes[0].tick == 0
    assert notes[0].duration == 4


def test_unpaired_note_on_closed_at_horizon() -> None:
    events = [
        MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON),
    ]
    notes = events_to_notes(events, horizon_tick=10)
    assert len(notes) == 1
    assert notes[0].pitch == 60
    assert notes[0].tick == 0
    assert notes[0].duration == 10  # closed at horizon


def test_unpaired_note_on_horizon_before_note_on_omitted() -> None:
    events = [
        MusicalEvent(tick=20, pitch=60, event_type=EventType.NOTE_ON),
    ]
    notes = events_to_notes(events, horizon_tick=10)
    assert len(notes) == 0  # horizon already past note_on


def test_two_notes_same_pitch_pairing() -> None:
    events = [
        MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON),
        MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_OFF),
        MusicalEvent(tick=8, pitch=60, event_type=EventType.NOTE_ON),
        MusicalEvent(tick=12, pitch=60, event_type=EventType.NOTE_OFF),
    ]
    notes = events_to_notes(events, horizon_tick=100)
    assert len(notes) == 2
    assert notes[0].tick == 0 and notes[0].duration == 4
    assert notes[1].tick == 8 and notes[1].duration == 4


def test_retrigger_same_pitch_closes_previous_at_new_note_on() -> None:
    # First note_on at 0, second note_on at 6 (no note_off in between)
    events = [
        MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON),
        MusicalEvent(tick=6, pitch=60, event_type=EventType.NOTE_ON),
    ]
    notes = events_to_notes(events, horizon_tick=20)
    # First note closed at tick 6; second is unpaired, closed at horizon 20
    assert len(notes) == 2
    assert notes[0].tick == 0 and notes[0].duration == 6
    assert notes[1].tick == 6 and notes[1].duration == 14


def test_placeholder_events_ignored() -> None:
    events = [
        MusicalEvent(
            tick=0, pitch=-1, event_type=EventType.NOTE_ON, is_placeholder=True
        ),
    ]
    notes = events_to_notes(events, horizon_tick=10)
    assert len(notes) == 0


def test_empty_events() -> None:
    notes = events_to_notes([], horizon_tick=10)
    assert notes == []


def test_multiple_pitches() -> None:
    events = [
        MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON),
        MusicalEvent(tick=2, pitch=64, event_type=EventType.NOTE_ON),
        MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_OFF),
        MusicalEvent(tick=6, pitch=64, event_type=EventType.NOTE_OFF),
    ]
    notes = events_to_notes(events, horizon_tick=100)
    assert len(notes) == 2
    by_pitch = {n.pitch: n for n in notes}
    assert by_pitch[60].duration == 4
    assert by_pitch[64].duration == 4
