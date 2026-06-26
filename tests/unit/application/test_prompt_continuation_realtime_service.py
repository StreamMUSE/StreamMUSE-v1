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


def _event_signature(events):
    return [(event.pitch, event.event_type) for event in events]


def test_prompt_continuation_default_scheduling_mode_is_streaming(monkeypatch):
    monkeypatch.delenv("LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE", raising=False)
    service = _make_service()

    assert service._scheduling_mode == "streaming_events"


def test_prompt_continuation_schedule_playable_drops_past_events():
    service = _make_service()
    output = service._output

    service._schedule_playable(
        [_note(48, 12), _note_off(48, 16), _note(50, 36), _note_off(50, 40)],
        current_tick=32,
    )

    assert service._scheduler.get_events_at_tick(12) == []
    assert _event_signature(service._scheduler.get_events_at_tick(36)) == [(50, EventType.NOTE_ON)]
    assert _event_signature(service._scheduler.get_events_at_tick(40)) == [(50, EventType.NOTE_OFF)]
    assert any("dropped 2 past event" in message for _state, message in output.statuses)


def test_prompt_continuation_schedule_playable_rehydrates_sustaining_notes():
    service = _make_service()
    output = service._output

    service._schedule_playable(
        [_note(48, 28), _note_off(48, 40), _note(50, 36), _note_off(50, 44)],
        current_tick=32,
    )

    assert _event_signature(service._scheduler.get_events_at_tick(32)) == [(48, EventType.NOTE_ON)]
    assert _event_signature(service._scheduler.get_events_at_tick(36)) == [(50, EventType.NOTE_ON)]
    assert _event_signature(
        service._scheduler.get_events_at_tick(40) + service._scheduler.get_events_at_tick(44)
    ) == [
        (48, EventType.NOTE_OFF),
        (50, EventType.NOTE_OFF),
    ]
    assert any("rehydrated 1 active note" in message for _state, message in output.statuses)


def test_prompt_continuation_schedule_playable_schedules_same_tick_retrigger_without_pair():
    service = _make_service()

    service._schedule_playable(
        [_note_off(60, 52), _note(60, 52)],
        current_tick=52,
    )

    assert _event_signature(service._scheduler.get_events_at_tick(52)) == [
        (60, EventType.NOTE_OFF),
        (60, EventType.NOTE_ON),
    ]


def test_prompt_continuation_schedule_playable_keeps_cross_chunk_note_on_before_note_off_arrives():
    service = _make_service()

    service._schedule_playable([_note(60, 52)], current_tick=52)
    assert _event_signature(service._scheduler.get_events_at_tick(52)) == [(60, EventType.NOTE_ON)]

    service._schedule_playable([_note(60, 52), _note_off(60, 56)], current_tick=53)

    assert service._scheduler.get_events_at_tick(53) == []
    assert _event_signature(service._scheduler.get_events_at_tick(56)) == [(60, EventType.NOTE_OFF)]


def test_prompt_continuation_schedule_playable_skips_duplicates():
    service = _make_service()
    output = service._output

    service._schedule_playable([_note(50, 36), _note_off(50, 40)], current_tick=32)
    service._schedule_playable(
        [_note(50, 36), _note_off(50, 40), _note(52, 40), _note_off(52, 44)],
        current_tick=32,
    )

    assert _event_signature(service._scheduler.get_events_at_tick(36)) == [(50, EventType.NOTE_ON)]
    assert _event_signature(service._scheduler.get_events_at_tick(40)) == [
        (50, EventType.NOTE_OFF),
        (52, EventType.NOTE_ON),
    ]
    assert _event_signature(service._scheduler.get_events_at_tick(44)) == [(52, EventType.NOTE_OFF)]
    assert any("skipped 2 duplicate event" in message for _state, message in output.statuses)


def test_prompt_continuation_schedule_playable_rehydrate_does_not_repeat_for_full_history():
    service = _make_service()
    history = [_note(60, 48), _note_off(60, 56)]

    service._schedule_playable(history, current_tick=52)
    service._schedule_playable(history, current_tick=53)

    assert _event_signature(service._scheduler.get_events_at_tick(52)) == [(60, EventType.NOTE_ON)]
    assert service._scheduler.get_events_at_tick(53) == []
    assert _event_signature(service._scheduler.get_events_at_tick(56)) == [(60, EventType.NOTE_OFF)]


def test_prompt_continuation_recover_late_can_drop_too_old_note_on(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "1")
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS", "4")
    monkeypatch.delenv("LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES", raising=False)
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

    assert _event_signature(service._scheduler.get_events_at_tick(36)) == [
        (50, EventType.NOTE_ON),
        (48, EventType.NOTE_OFF),
    ]
    assert service._scheduler.get_events_at_tick(20) == []
    assert any("dropped 1 too-late note_on" in message for _state, message in output.statuses)
    assert any("rehydrated 1 active note" in message for _state, message in output.statuses)


def test_prompt_continuation_recover_late_can_rehydrate_active_notes(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "1")
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS", "4")
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES", "1")
    service = _make_service()
    output = service._output

    history = [
        _note(48, 20),
        _note_off(48, 40),
        _note(50, 34),
        _note_off(50, 42),
    ]
    service._schedule_playable(history, current_tick=36)

    assert _event_signature(service._scheduler.get_events_at_tick(36)) == [
        (48, EventType.NOTE_ON),
        (50, EventType.NOTE_ON),
    ]
    assert _event_signature(service._scheduler.get_events_at_tick(40)) == [(48, EventType.NOTE_OFF)]
    assert service._scheduler.get_events_at_tick(20) == []
    assert any("rehydrated 2 active note" in message for _state, message in output.statuses)

    service._schedule_playable(history, current_tick=37)

    assert service._scheduler.get_events_at_tick(37) == []


def test_prompt_continuation_recover_late_is_unbounded_without_bound_switch(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "1")
    monkeypatch.delenv("LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY", raising=False)
    monkeypatch.delenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS", raising=False)
    service = _make_service()

    assert service._bound_late_recovery is False
    assert service._recover_late_max_ticks is None
    service._schedule_playable(
        [
            _note(48, 20),
            _note_off(48, 24),
            _note(50, 34),
            _note_off(50, 40),
        ],
        current_tick=36,
    )

    assert _event_signature(service._scheduler.get_events_at_tick(36)) == [
        (50, EventType.NOTE_ON),
        (48, EventType.NOTE_ON),
        (48, EventType.NOTE_OFF),
    ]


def test_prompt_continuation_recover_late_bound_switch_defaults_to_generation_interval_cap(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "1")
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY", "1")
    monkeypatch.delenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS", raising=False)
    service = _make_service()

    assert service._bound_late_recovery is True
    assert service._recover_late_max_ticks == 4
    service._schedule_playable(
        [
            _note(48, 20),
            _note_off(48, 24),
            _note(50, 34),
            _note_off(50, 40),
        ],
        current_tick=36,
    )

    assert _event_signature(service._scheduler.get_events_at_tick(36)) == [
        (50, EventType.NOTE_ON),
        (48, EventType.NOTE_OFF),
    ]
    assert service._scheduler.get_events_at_tick(20) == []


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
