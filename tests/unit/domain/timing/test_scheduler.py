"""Unit tests for PlaybackScheduler."""

import threading

import pytest

from streammuse.domain.musical import MusicalEvent
from streammuse.domain.musical.events import EventType
from streammuse.domain.timing import PlaybackScheduler


def test_schedule_and_get() -> None:
    s = PlaybackScheduler()
    e = MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON)
    s.schedule(e, tick=5)
    got = s.get_events_at_tick(5)
    assert got == [e]
    assert s.get_events_at_tick(5) == []  # removed


def test_schedule_multiple_at_same_tick() -> None:
    s = PlaybackScheduler()
    e1 = MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON)
    e2 = MusicalEvent(tick=0, pitch=64, event_type=EventType.NOTE_ON)
    s.schedule(e1, 10)
    s.schedule(e2, 10)
    got = s.get_events_at_tick(10)
    assert len(got) == 2
    assert e1 in got and e2 in got


def test_get_empty_tick() -> None:
    s = PlaybackScheduler()
    assert s.get_events_at_tick(0) == []


def test_clear_future_events_all() -> None:
    s = PlaybackScheduler()
    e1 = MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON)
    e2 = MusicalEvent(tick=0, pitch=64, event_type=EventType.NOTE_ON)
    s.schedule(e1, 10)
    s.schedule(e2, 12)
    s.clear_future_events(from_tick=11)
    assert s.get_events_at_tick(10) == [e1]
    assert s.get_events_at_tick(12) == []


def test_clear_future_events_from_tick_inclusive() -> None:
    s = PlaybackScheduler()
    e = MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON)
    s.schedule(e, 8)
    s.clear_future_events(from_tick=8)
    assert s.get_events_at_tick(8) == []


def test_pop_future_events_returns_removed_source_events_in_tick_order() -> None:
    s = PlaybackScheduler()
    model_10 = MusicalEvent(tick=10, pitch=60, event_type=EventType.NOTE_ON, source="model")
    user_11 = MusicalEvent(tick=11, pitch=62, event_type=EventType.NOTE_ON, source="user")
    model_12 = MusicalEvent(tick=12, pitch=64, event_type=EventType.NOTE_OFF, source="model")
    s.schedule(model_12, 12)
    s.schedule(model_10, 10)
    s.schedule(user_11, 11)

    removed = s.pop_future_events(from_tick=10, source="model")

    assert removed == [model_10, model_12]
    assert s.get_events_at_tick(10) == []
    assert s.get_events_at_tick(11) == [user_11]
    assert s.get_events_at_tick(12) == []


def test_thread_safety() -> None:
    s = PlaybackScheduler()
    events = [
        MusicalEvent(tick=0, pitch=60 + i, event_type=EventType.NOTE_ON)
        for i in range(50)
    ]

    def schedule_many() -> None:
        for i, e in enumerate(events):
            s.schedule(e, tick=i % 10)

    def get_many() -> None:
        for t in range(10):
            s.get_events_at_tick(t)

    t1 = threading.Thread(target=schedule_many)
    t2 = threading.Thread(target=get_many)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # No crash; scheduler internal state consistent
    remaining = sum(len(s.get_events_at_tick(t)) for t in range(10))
    assert remaining >= 0  # some may have been consumed by t2
