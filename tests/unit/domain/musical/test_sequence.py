"""Unit tests for MusicalSequence."""

import pytest

from streammuse.domain.musical.events import EventType, MusicalEvent
from streammuse.domain.musical.sequence import MusicalSequence
from streammuse.domain.timing import Tempo


def _tempo() -> Tempo:
    return Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4)


def test_sequence_empty() -> None:
    t = _tempo()
    seq = MusicalSequence(events=(), tempo=t)
    assert seq.events == ()
    assert seq.tempo is t


def test_get_events_in_range() -> None:
    t = _tempo()
    events = (
        MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON),
        MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_OFF),
        MusicalEvent(tick=8, pitch=64, event_type=EventType.NOTE_ON),
        MusicalEvent(tick=12, pitch=64, event_type=EventType.NOTE_OFF),
        MusicalEvent(tick=16, pitch=67, event_type=EventType.NOTE_ON),
    )
    seq = MusicalSequence(events=events, tempo=t)
    sub = seq.get_events_in_range(4, 12)
    assert len(sub.events) == 2  # tick 4 and 8
    assert sub.events[0].tick == 4
    assert sub.events[1].tick == 8
    assert sub.tempo is t


def test_get_events_in_range_empty() -> None:
    t = _tempo()
    events = (
        MusicalEvent(tick=10, pitch=60, event_type=EventType.NOTE_ON),
    )
    seq = MusicalSequence(events=events, tempo=t)
    sub = seq.get_events_in_range(0, 10)
    assert len(sub.events) == 0


def test_quantize() -> None:
    t = _tempo()
    events = (
        MusicalEvent(tick=1, pitch=60, event_type=EventType.NOTE_ON),
        MusicalEvent(tick=5, pitch=60, event_type=EventType.NOTE_OFF),
        MusicalEvent(tick=10, pitch=64, event_type=EventType.NOTE_ON),
    )
    seq = MusicalSequence(events=events, tempo=t)
    q = seq.quantize(4)
    assert q.events[0].tick == 0   # 1 -> 0
    assert q.events[1].tick == 4   # 5 -> 4
    assert q.events[2].tick == 8   # 10 -> 8
    assert q.tempo is t


def test_quantize_invalid() -> None:
    t = _tempo()
    seq = MusicalSequence(events=(), tempo=t)
    with pytest.raises(ValueError, match="quantization_ticks must be positive"):
        seq.quantize(0)
    with pytest.raises(ValueError, match="quantization_ticks must be positive"):
        seq.quantize(-1)


def test_quantize_preserves_placeholder() -> None:
    t = _tempo()
    events = (
        MusicalEvent(
            tick=2, pitch=-1, event_type=EventType.NOTE_ON, is_placeholder=True
        ),
    )
    seq = MusicalSequence(events=events, tempo=t)
    q = seq.quantize(2)
    assert len(q.events) == 1
    assert q.events[0].is_placeholder is True
    assert q.events[0].tick == 2
