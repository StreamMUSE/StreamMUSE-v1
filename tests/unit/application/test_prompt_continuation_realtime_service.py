from __future__ import annotations

import queue
import threading
import time

from streammuse.application.services.prompt_continuation_realtime_service import (
    PromptContinuationRealtimeService,
)
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import PlaybackScheduler, Tempo


class _NoopInput:
    def read_events(self):
        return iter([])

    def close(self):
        return None


class _RecordingOutput:
    def __init__(self):
        self.events = []
        self.statuses = []
        self.ticks = []

    def output_event(self, event, source):
        self.events.append((event, source))

    def output_tick(self, tick, bar, beat):
        self.ticks.append((tick, bar, beat))

    def output_stats(self, **kwargs):
        return None

    def output_status(self, state, message=""):
        self.statuses.append((state, message))

    def output_config(self, config):
        return None

    def close(self):
        return None


class _FakePromptClient:
    def __init__(self):
        self.status_calls = 0
        self.playable_calls = 0
        self.playable_responses = [[]]

    def clear_history(self):
        return {"success": True}

    def start(self, **kwargs):
        return {"phase": "prompt_running", **kwargs}

    def append_melody(self, **kwargs):
        return {"phase": "catchup_running", **kwargs}

    def status(self):
        self.status_calls += 1
        return {"phase": "ready", "is_playback_ready": True}

    def playable(self):
        self.playable_calls += 1
        index = min(self.playable_calls - 1, len(self.playable_responses) - 1)
        return self.playable_responses[index], {"phase": "ready"}


def _note(pitch: int, tick: int) -> MusicalEvent:
    return MusicalEvent(tick=tick, pitch=pitch, event_type=EventType.NOTE_ON, velocity=100)


def _note_off(pitch: int, tick: int) -> MusicalEvent:
    return MusicalEvent(tick=tick, pitch=pitch, event_type=EventType.NOTE_OFF, velocity=0)


def _make_service() -> PromptContinuationRealtimeService:
    client = _FakePromptClient()
    return PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=client,
        output_sink=_RecordingOutput(),
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        prompt_length_ticks=32,
        generation_interval_ticks=4,
        now=lambda: 0.0,
        sleep=lambda _: None,
    )


def test_prompt_continuation_enqueues_start_after_prompt_window():
    service = _make_service()
    service._prompt_events = [_note(60, 0), _note(62, 31)]

    service._maybe_enqueue_start(observed_until_tick=31)
    with __import__("pytest").raises(queue.Empty):
        service._control_q.get_nowait()

    service._maybe_enqueue_start(observed_until_tick=32)
    action = service._control_q.get_nowait()

    assert action.kind == "start"
    assert action.observed_until_tick == 32
    assert [event.pitch for event in action.melody_events] == [60, 62]


def test_prompt_continuation_append_keeps_empty_rest_chunks():
    service = _make_service()
    service._start_enqueued = True
    service._last_append_observed_tick = 32

    service._maybe_enqueue_append(observed_until_tick=36)
    empty_action = service._control_q.get_nowait()
    assert empty_action.kind == "append"
    assert empty_action.observed_until_tick == 36
    assert empty_action.melody_events == []

    service._pending_append_events = [_note(64, 38)]
    service._maybe_enqueue_append(observed_until_tick=40)
    note_action = service._control_q.get_nowait()
    assert [event.pitch for event in note_action.melody_events] == [64]


def test_prompt_continuation_schedule_playable_drops_past_events():
    service = _make_service()
    output = service._output

    service._schedule_playable(
        [_note(48, 12), _note_off(48, 16), _note(50, 36), _note_off(50, 40)],
        current_tick=32,
    )

    assert service._scheduler.get_events_at_tick(12) == []
    scheduled = service._scheduler.get_events_at_tick(36)
    assert [event.pitch for event in scheduled] == [50]
    assert any("dropped 2 past" in message for _state, message in output.statuses)


def test_prompt_continuation_schedule_playable_clips_sustaining_notes():
    service = _make_service()
    output = service._output

    service._schedule_playable(
        [_note(48, 28), _note_off(48, 40), _note(50, 36), _note_off(50, 44)],
        current_tick=32,
    )

    clipped_on = service._scheduler.get_events_at_tick(32)
    future_on = service._scheduler.get_events_at_tick(36)
    note_offs = service._scheduler.get_events_at_tick(40) + service._scheduler.get_events_at_tick(44)
    assert [(event.pitch, event.event_type) for event in clipped_on] == [(48, EventType.NOTE_ON)]
    assert [(event.pitch, event.event_type) for event in future_on] == [(50, EventType.NOTE_ON)]
    assert [(event.pitch, event.event_type) for event in note_offs] == [
        (48, EventType.NOTE_OFF),
        (50, EventType.NOTE_OFF),
    ]
    assert any("clipped 1 sustaining" in message for _state, message in output.statuses)


def test_prompt_continuation_schedule_playable_skips_duplicates():
    service = _make_service()
    output = service._output

    service._schedule_playable([_note(50, 36), _note_off(50, 40)], current_tick=32)
    service._schedule_playable(
        [_note(50, 36), _note_off(50, 40), _note(52, 40), _note_off(52, 44)],
        current_tick=32,
    )

    assert [event.pitch for event in service._scheduler.get_events_at_tick(36)] == [50]
    tick_40 = service._scheduler.get_events_at_tick(40)
    assert [(event.pitch, event.event_type) for event in tick_40] == [
        (50, EventType.NOTE_OFF),
        (52, EventType.NOTE_ON),
    ]
    assert any("skipped 1 duplicate note" in message for _state, message in output.statuses)


def test_prompt_continuation_recover_late_can_drop_too_old_note_on(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "1")
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS", "4")
    service = _make_service()
    output = service._output

    service._schedule_playable(
        [
            _note(48, 20),
            _note_off(48, 24),
            _note(50, 34),
            _note_off(50, 40),
        ],
        current_tick=36,
    )

    tick_36 = service._scheduler.get_events_at_tick(36)
    assert [(event.pitch, event.event_type) for event in tick_36] == [
        (48, EventType.NOTE_OFF),
        (50, EventType.NOTE_ON),
    ]
    assert service._scheduler.get_events_at_tick(20) == []
    assert any("dropped 1 too-late note_on" in message for _state, message in output.statuses)


def test_protocol_worker_does_not_fetch_playable_before_first_append():
    service = _make_service()
    client = service._client
    service._running = True
    action_cls = __import__(
        "streammuse.application.services.prompt_continuation_realtime_service",
        fromlist=["_ControlAction"],
    )._ControlAction
    service._control_q.put(action_cls(kind="start", melody_events=[_note(60, 0)], observed_until_tick=32))

    worker = threading.Thread(target=service._protocol_worker)
    worker.start()
    time.sleep(0.15)
    service._running = False
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert client.status_calls == 0
    assert client.playable_calls == 0


def test_protocol_worker_fetches_playable_after_rest_append():
    service = _make_service()
    client = service._client
    service._running = True
    action_cls = __import__(
        "streammuse.application.services.prompt_continuation_realtime_service",
        fromlist=["_ControlAction"],
    )._ControlAction
    service._control_q.put(action_cls(kind="start", melody_events=[_note(60, 0)], observed_until_tick=32))
    service._control_q.put(action_cls(kind="append", melody_events=[], observed_until_tick=36))

    worker = threading.Thread(target=service._protocol_worker)
    worker.start()
    deadline = time.monotonic() + 1.0
    while client.playable_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    service._running = False
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert client.status_calls >= 1
    assert client.playable_calls == 1


def test_protocol_worker_fetches_playable_more_than_once_after_append():
    service = _make_service()
    client = service._client
    service._running = True
    action_cls = __import__(
        "streammuse.application.services.prompt_continuation_realtime_service",
        fromlist=["_ControlAction"],
    )._ControlAction
    service._control_q.put(action_cls(kind="start", melody_events=[_note(60, 0)], observed_until_tick=32))
    service._control_q.put(action_cls(kind="append", melody_events=[], observed_until_tick=36))
    service._control_q.put(action_cls(kind="append", melody_events=[_note(64, 38)], observed_until_tick=40))

    worker = threading.Thread(target=service._protocol_worker)
    worker.start()
    deadline = time.monotonic() + 1.0
    while client.playable_calls < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    service._running = False
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert client.status_calls >= 2
    assert client.playable_calls >= 2
