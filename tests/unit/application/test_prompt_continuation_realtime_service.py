from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest

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
        self.metronome_ticks = []

    def output_event(self, event, source):
        self.events.append((event, source))

    def output_tick(self, tick, bar, beat):
        self.ticks.append((tick, bar, beat))

    def output_metronome_tick(self, tick, bar, beat):
        self.metronome_ticks.append((tick, bar, beat))

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
        self.clear_history_calls = 0
        self.start_calls = 0
        self.append_calls = 0
        self.status_calls = 0
        self.playable_calls = 0
        self.playable_responses = [[]]

    def clear_history(self):
        self.clear_history_calls += 1
        return {"success": True}

    def start(self, **kwargs):
        self.start_calls += 1
        return {"phase": "prompt_running", **kwargs}

    def append_melody(self, **kwargs):
        self.append_calls += 1
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


def _event_signature(events):
    return [(event.pitch, event.event_type) for event in events]


def _make_service() -> PromptContinuationRealtimeService:
    client = _FakePromptClient()
    service = PromptContinuationRealtimeService(
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
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.0)
    return service


def _stamp_single_input_event(*, input_snap_forward_fraction: float) -> MusicalEvent:
    class _OneEventInput:
        def read_events(self):
            return iter([_note(60, 99)])

        def close(self):
            return None

    service = PromptContinuationRealtimeService(
        input_source=_OneEventInput(),
        prompt_client=_FakePromptClient(),
        output_sink=_RecordingOutput(),
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        input_snap_forward_fraction=input_snap_forward_fraction,
        now=lambda: 0.490,
        sleep=lambda _: None,
    )
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.0)
    service._running = True

    service._input_worker()

    return service._event_q.get_nowait()


def test_prompt_input_fraction_zero_keeps_floor_quantization() -> None:
    assert _stamp_single_input_event(input_snap_forward_fraction=0.0).tick == 3


def test_prompt_input_fraction_point_four_snaps_490ms_to_step_four() -> None:
    assert _stamp_single_input_event(input_snap_forward_fraction=0.4).tick == 4


def test_prompt_continuation_rejects_non_four_steps_per_beat() -> None:
    with pytest.raises(
        ValueError,
        match=r"Prompt\+Continuation requires exactly 4 steps per beat; got 8",
    ):
        PromptContinuationRealtimeService(
            input_source=_NoopInput(),
            prompt_client=_FakePromptClient(),
            output_sink=_RecordingOutput(),
            tempo=Tempo(bpm=120.0, ticks_per_beat=8, beats_per_bar=4),
            scheduler=PlaybackScheduler(),
        )


def test_prompt_count_in_emits_metronome_only_before_formal_timeline():
    client = _FakePromptClient()
    output = _RecordingOutput()
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=client,
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        count_in_beats=1,
        now=lambda: 0.0,
        sleep=lambda _: None,
    )
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.5)
    service._running = True

    service._tick_loop(max_ticks=0)

    assert [tick for tick, _bar, _beat in output.metronome_ticks] == [-4, -3, -2, -1]
    assert output.ticks == []
    assert client.clear_history_calls == 0
    assert client.start_calls == 0
    assert client.append_calls == 0
    assert client.status_calls == 0
    assert client.playable_calls == 0


def test_prompt_count_in_starts_formal_timeline_at_tick_zero():
    output = _RecordingOutput()
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=_FakePromptClient(),
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        count_in_beats=1,
        now=lambda: 0.0,
        sleep=lambda _: None,
    )
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.5)
    service._running = True

    service._tick_loop(max_ticks=1)

    assert output.ticks == [(0, 0, 0)]
    assert output.metronome_ticks[-1] == (0, 0, 0)


def test_prompt_count_in_input_waits_and_stamps_first_event_at_tick_zero():
    sleep_entered = threading.Event()
    release = threading.Event()
    read_started = threading.Event()
    now_value = [0.0]

    class _OneEventInput:
        def read_events(self):
            read_started.set()
            return iter([_note(60, 99)])

        def close(self):
            return None

    def blocking_sleep(_delay):
        sleep_entered.set()
        release.wait(timeout=1.0)

    service = PromptContinuationRealtimeService(
        input_source=_OneEventInput(),
        prompt_client=_FakePromptClient(),
        output_sink=_RecordingOutput(),
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        count_in_beats=1,
        now=lambda: now_value[0],
        sleep=blocking_sleep,
    )
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.5)
    service._running = True

    worker = threading.Thread(target=service._input_worker)
    worker.start()
    assert sleep_entered.wait(timeout=1.0)
    assert not read_started.is_set()

    now_value[0] = 0.5
    release.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert read_started.is_set()
    assert service._event_q.get_nowait().tick == 0


def test_prompt_count_in_stop_before_timeline_does_not_read_closed_input():
    sleep_entered = threading.Event()
    release = threading.Event()
    read_started = threading.Event()

    class _RecordingInput:
        def read_events(self):
            read_started.set()
            return iter([])

        def close(self):
            return None

    def blocking_sleep(_delay):
        sleep_entered.set()
        release.wait(timeout=1.0)

    service = PromptContinuationRealtimeService(
        input_source=_RecordingInput(),
        prompt_client=_FakePromptClient(),
        output_sink=_RecordingOutput(),
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        count_in_beats=1,
        now=lambda: 0.0,
        sleep=blocking_sleep,
    )
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.5)
    service._running = True

    worker = threading.Thread(target=service._input_worker)
    worker.start()
    assert sleep_entered.wait(timeout=1.0)
    service._running = False
    release.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert not read_started.is_set()


def test_prompt_count_in_protocol_waits_before_any_backend_call():
    sleep_entered = threading.Event()
    release = threading.Event()
    now_value = [0.0]
    client = _FakePromptClient()

    def blocking_sleep(_delay):
        sleep_entered.set()
        release.wait(timeout=1.0)

    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=client,
        output_sink=_RecordingOutput(),
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        count_in_beats=1,
        protocol_poll_interval_s=0.01,
        now=lambda: now_value[0],
        sleep=blocking_sleep,
    )
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.5)
    service._running = True

    worker = threading.Thread(target=service._protocol_worker)
    worker.start()
    assert sleep_entered.wait(timeout=1.0)
    assert client.clear_history_calls == 0
    assert client.start_calls == 0
    assert client.append_calls == 0
    assert client.status_calls == 0
    assert client.playable_calls == 0

    now_value[0] = 0.5
    release.set()
    deadline = time.monotonic() + 1.0
    while client.clear_history_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    service._running = False
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert client.clear_history_calls == 1
    assert client.start_calls == 0
    assert client.append_calls == 0
    assert client.status_calls == 0
    assert client.playable_calls == 0


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


def test_prompt_continuation_default_scheduling_mode_is_streaming(monkeypatch):
    monkeypatch.delenv("LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE", raising=False)
    service = _make_service()

    assert service._scheduling_mode == "streaming_events"


def test_prompt_continuation_schedule_playable_current_tick_unpaired_note_on_schedules():
    service = _make_service()

    service._schedule_playable([_note(50, 36)], current_tick=36)

    assert _event_signature(service._scheduler.get_events_at_tick(36)) == [(50, EventType.NOTE_ON)]


def test_prompt_continuation_schedule_playable_later_note_off_schedules_exactly_once():
    service = _make_service()

    service._schedule_playable([_note(60, 52)], current_tick=52)
    service._schedule_playable([_note(60, 52), _note_off(60, 56)], current_tick=53)
    service._schedule_playable([_note(60, 52), _note_off(60, 56)], current_tick=54)

    assert _event_signature(service._scheduler.get_events_at_tick(52)) == [(60, EventType.NOTE_ON)]
    assert _event_signature(service._scheduler.get_events_at_tick(56)) == [(60, EventType.NOTE_OFF)]
    assert service._scheduler.get_events_at_tick(53) == []
    assert service._scheduler.get_events_at_tick(54) == []


def test_prompt_continuation_schedule_playable_repeated_full_history_does_not_replay():
    service = _make_service()

    history = [_note(60, 52), _note_off(60, 56)]
    service._schedule_playable(history, current_tick=52)
    service._schedule_playable(history, current_tick=53)

    assert _event_signature(service._scheduler.get_events_at_tick(52)) == [(60, EventType.NOTE_ON)]
    assert _event_signature(service._scheduler.get_events_at_tick(56)) == [(60, EventType.NOTE_OFF)]
    assert service._scheduler.get_events_at_tick(53) == []


def test_prompt_continuation_schedule_playable_allows_identical_duplicate_occurrences():
    service = _make_service()

    history = [_note(60, 52), _note(60, 52), _note_off(60, 56), _note_off(60, 56)]
    service._schedule_playable(history, current_tick=52)
    service._schedule_playable(history, current_tick=53)

    assert _event_signature(service._scheduler.get_events_at_tick(52)) == [
        (60, EventType.NOTE_ON),
        (60, EventType.NOTE_ON),
    ]
    assert _event_signature(service._scheduler.get_events_at_tick(56)) == [
        (60, EventType.NOTE_OFF),
        (60, EventType.NOTE_OFF),
    ]


def test_prompt_continuation_schedule_playable_paired_mode_is_selectable(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE", "paired_future_only")
    service = _make_service()

    service._schedule_playable([_note(60, 52)], current_tick=52)
    service._schedule_playable([_note(60, 52), _note_off(60, 56)], current_tick=53)

    assert service._scheduling_mode == "paired_future_only"
    assert service._scheduler.get_events_at_tick(52) == []
    assert _event_signature(service._scheduler.get_events_at_tick(53)) == [(60, EventType.NOTE_ON)]
    assert _event_signature(service._scheduler.get_events_at_tick(56)) == [(60, EventType.NOTE_OFF)]


def test_prompt_continuation_schedule_playable_paired_mode_clips_sustaining_notes(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE", "paired_future_only")
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
    assert any("clipped 1 sustaining" in message for _state, message in output.statuses)


def test_prompt_continuation_schedule_playable_paired_mode_skips_duplicate_pairs(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE", "paired_future_only")
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
    assert any("skipped 1 duplicate note" in message for _state, message in output.statuses)


def test_prompt_continuation_schedule_playable_drops_past_events_without_recovery():
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


def test_prompt_continuation_schedule_playable_without_recovery_drops_past_note_on_only():
    service = _make_service()
    output = service._output

    service._schedule_playable(
        [_note(48, 20), _note_off(48, 40), _note(50, 34), _note_off(50, 42)],
        current_tick=36,
    )

    assert service._scheduler.get_events_at_tick(36) == []
    assert _event_signature(service._scheduler.get_events_at_tick(40)) == [(48, EventType.NOTE_OFF)]
    assert _event_signature(service._scheduler.get_events_at_tick(42)) == [(50, EventType.NOTE_OFF)]
    assert any("recovered 0 late event" in message for _state, message in output.statuses)
    assert any("dropped 2 past event" in message for _state, message in output.statuses)


def test_prompt_continuation_rehydrate_active_notes_is_independent_of_recover_late(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "0")
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY", "1")
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS", "4")
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES", "1")
    service = _make_service()
    output = service._output

    service._schedule_playable(
        [
            _note(48, 20),
            _note_off(48, 40),
            _note(50, 22),
            _note_off(50, 24),
            _note(52, 38),
            _note_off(52, 42),
        ],
        current_tick=36,
    )

    assert _event_signature(service._scheduler.get_events_at_tick(36)) == [(48, EventType.NOTE_ON)]
    assert _event_signature(service._scheduler.get_events_at_tick(38)) == [(52, EventType.NOTE_ON)]
    assert _event_signature(service._scheduler.get_events_at_tick(40)) == [(48, EventType.NOTE_OFF)]
    assert _event_signature(service._scheduler.get_events_at_tick(42)) == [(52, EventType.NOTE_OFF)]
    assert any("recovered 0 late event" in message for _state, message in output.statuses)
    assert any("dropped 2 past event" in message for _state, message in output.statuses)
    assert any("rehydrated 1 active note" in message for _state, message in output.statuses)


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

    tick_36 = service._scheduler.get_events_at_tick(36)
    assert [(event.pitch, event.event_type) for event in tick_36] == [
        (48, EventType.NOTE_OFF),
        (50, EventType.NOTE_ON),
    ]
    assert service._scheduler.get_events_at_tick(20) == []
    assert any("dropped 1 too-late note_on" in message for _state, message in output.statuses)


def test_prompt_continuation_recover_late_can_rehydrate_active_notes(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "1")
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS", "4")
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES", "1")
    service = _make_service()
    output = service._output

    service._schedule_playable(
        [
            _note(48, 20),
            _note_off(48, 40),
            _note(50, 34),
            _note_off(50, 42),
        ],
        current_tick=36,
    )

    tick_36 = service._scheduler.get_events_at_tick(36)
    assert [(event.pitch, event.event_type) for event in tick_36] == [
        (48, EventType.NOTE_ON),
        (50, EventType.NOTE_ON),
    ]
    assert [(event.pitch, event.event_type) for event in service._scheduler.get_events_at_tick(40)] == [
        (48, EventType.NOTE_OFF),
    ]
    assert service._scheduler.get_events_at_tick(20) == []
    assert any("rehydrated 1 active note" in message for _state, message in output.statuses)

    service._schedule_playable(
        [
            _note(48, 20),
            _note_off(48, 40),
            _note(50, 34),
            _note_off(50, 42),
        ],
        current_tick=37,
    )

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

    tick_36 = service._scheduler.get_events_at_tick(36)
    assert [(event.pitch, event.event_type) for event in tick_36] == [
        (48, EventType.NOTE_ON),
        (48, EventType.NOTE_OFF),
        (50, EventType.NOTE_ON),
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

    tick_36 = service._scheduler.get_events_at_tick(36)
    assert [(event.pitch, event.event_type) for event in tick_36] == [
        (48, EventType.NOTE_OFF),
        (50, EventType.NOTE_ON),
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
