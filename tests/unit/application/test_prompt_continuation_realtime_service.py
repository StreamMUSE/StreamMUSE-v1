from __future__ import annotations

import inspect
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from streammuse.application.services.prompt_continuation_realtime_service import (
    PromptContinuationRealtimeService,
    _ControlAction,
    _PlayableBatch,
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
        self.start_requests = []

    def clear_history(self):
        self.clear_history_calls += 1
        return {"success": True}

    def start(self, **kwargs):
        self.start_calls += 1
        self.start_requests.append(dict(kwargs))
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


def _placeholder(tick: int) -> MusicalEvent:
    return MusicalEvent(
        tick=tick,
        pitch=-1,
        event_type=EventType.NOTE_ON,
        is_placeholder=True,
        source="model",
    )


def _event_signature(events):
    return [(event.pitch, event.event_type) for event in events]


def _make_service(
    *, model_condition_bpm: int | None = None
) -> PromptContinuationRealtimeService:
    client = _FakePromptClient()
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=client,
        output_sink=_RecordingOutput(),
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        prompt_length_ticks=32,
        generation_interval_ticks=4,
        model_condition_bpm=model_condition_bpm,
        now=lambda: 0.0,
        sleep=lambda _: None,
    )
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.0)
    return service


def test_prompt_continuation_default_now_is_time_time() -> None:
    assert (
        inspect.signature(PromptContinuationRealtimeService).parameters["now"].default
        is time.time
    )


def test_prompt_service_resolves_and_exposes_model_condition_bpm() -> None:
    fallback = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=_FakePromptClient(),
        output_sink=_RecordingOutput(),
        tempo=Tempo(bpm=80.6, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
    )
    overridden = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=_FakePromptClient(),
        output_sink=_RecordingOutput(),
        tempo=Tempo(bpm=80.6, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        model_condition_bpm=137,
    )

    assert fallback.effective_model_bpm == 81
    assert overridden.effective_model_bpm == 137


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


def test_prompt_input_worker_uses_one_service_clock_receipt_for_stamp_and_trace():
    class _OneEventInput:
        def read_events(self):
            return iter([_note(60, 99)])

        def close(self):
            return None

    class _TraceOutput(_RecordingOutput):
        def __init__(self):
            super().__init__()
            self.rows = []

        def log_input_quantization(self, row):
            self.rows.append(dict(row))

    now_calls = []

    def now():
        now_calls.append(None)
        return 100.490

    output = _TraceOutput()
    service = PromptContinuationRealtimeService(
        input_source=_OneEventInput(),
        prompt_client=_FakePromptClient(),
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        input_snap_forward_fraction=0.4,
        input_quantization_trace_enabled=True,
        now=now,
        sleep=lambda _: None,
    )
    service._runtime = SimpleNamespace(
        session_start_time=100.0,
        timeline_start_time=100.0,
    )
    service._running = True
    service._sleep_until = lambda _target: None

    service._input_worker()

    stamped = service._event_q.get_nowait()
    row = output.rows[0]
    assert len(now_calls) == 1
    assert stamped.tick == row["quantized_tick"] == 4
    assert row["service"] == "prompt_continuation"
    assert row["clock_domain"] == "service_now"
    assert row["application_received_time_s"] == pytest.approx(100.490)
    assert row["raw_tick"] == pytest.approx(3.92)
    assert row["signed_error_ms"] == pytest.approx(10.0)


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


def test_prompt_continuation_system_trace_records_note_rest_missing() -> None:
    class _TraceOutput(_RecordingOutput):
        def __init__(self) -> None:
            super().__init__()
            self.system_trace_rows = []

        def log_system_trace(self, row):
            self.system_trace_rows.append(dict(row))

    output = _TraceOutput()
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=_FakePromptClient(),
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        now=lambda: 50.0,
        sleep=lambda _: None,
    )
    service._runtime = SimpleNamespace(session_start_time=50.0, timeline_start_time=50.0)
    service._running = True
    service._playable_q.put(
        _PlayableBatch(
            accompaniment=[_note(55, 0), _placeholder(1)],
            status={"phase": "ready"},
            arrival_time_s=50.02,
        )
    )

    service._tick_loop(max_ticks=3)

    assert [(source, event.pitch) for event, source in output.events] == [("model", 55)]
    rows = output.system_trace_rows
    assert [row["tick"] for row in rows] == [0, 1, 2]

    note_row = rows[0]
    assert note_row["schema_version"] == 2
    assert note_row["record_type"] == "frame_deadline"
    assert note_row["mode"] == "realtime"
    assert note_row["condition"] == "prompt_continuation"
    assert note_row["clock_domain"] == "service_now"
    assert note_row["nominal_tick_time_s"] == pytest.approx(50.0)
    assert note_row["deadline_time_s"] == pytest.approx(50.0125)
    assert note_row["arrival_time_s"] > note_row["deadline_time_s"]
    assert note_row["arrived_by_deadline"] is False
    assert note_row["decision"] == "note"
    assert note_row["logical_tick"] == 0
    assert note_row["scheduled_tick"] == 0
    assert note_row["generation_start_tick"] is None
    assert note_row["request_id"] is None
    assert note_row["action"] == "scheduled"
    assert note_row["policy"] == "future_event"
    assert note_row["emitted_model_note_on_count"] == 1
    assert note_row["explicit_rest"] is False

    rest_row = rows[1]
    assert rest_row["decision"] == "rest"
    assert rest_row["arrival_time_s"] == pytest.approx(50.02)
    assert rest_row["arrived_by_deadline"] is True
    assert rest_row["logical_tick"] == 1
    assert rest_row["scheduled_tick"] == 1
    assert rest_row["action"] == "scheduled"
    assert rest_row["policy"] == "future_placeholder"
    assert rest_row["emitted_model_note_on_count"] == 0
    assert rest_row["explicit_rest"] is True

    missing_row = rows[2]
    assert missing_row["decision"] == "missing"
    assert missing_row["arrived_by_deadline"] is False
    assert missing_row["arrival_time_s"] is None
    assert missing_row["logical_tick"] is None
    assert missing_row["scheduled_tick"] is None
    assert missing_row["generation_start_tick"] is None
    assert missing_row["request_id"] is None
    assert missing_row["action"] is None
    assert missing_row["policy"] is None
    assert missing_row["emitted_model_note_on_count"] == 0
    assert missing_row["explicit_rest"] is False


def test_prompt_first_late_playable_fetch_records_past_coverage() -> None:
    class _TraceOutput(_RecordingOutput):
        def __init__(self) -> None:
            super().__init__()
            self.system_trace_rows = []

        def log_system_trace(self, row):
            self.system_trace_rows.append(dict(row))

    output = _TraceOutput()
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=_FakePromptClient(),
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        prompt_length_ticks=32,
        now=lambda: 0.0,
        sleep=lambda _: None,
    )

    service._record_playable_availability(
        {"accompaniment_history_beats": 4, "request_id": "late-fetch"},
        availability_time_s=20.0,
    )

    assert output.system_trace_rows == [
        {
            "schema_version": 2,
            "record_type": "availability_span",
            "mode": "realtime",
            "condition": "prompt_continuation",
            "clock_domain": "service_now",
            "start_tick": 0,
            "end_tick_exclusive": 16,
            "availability_time_s": 20.0,
            "generation_start_tick": 0,
            "request_id": "late-fetch",
            "source_stage": "prompt",
        }
    ]


def test_prompt_playable_coverage_is_incremental_without_duplicates() -> None:
    class _TraceOutput(_RecordingOutput):
        def __init__(self) -> None:
            super().__init__()
            self.system_trace_rows = []

        def log_system_trace(self, row):
            self.system_trace_rows.append(dict(row))

    output = _TraceOutput()
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=_FakePromptClient(),
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        prompt_length_ticks=32,
        now=lambda: 0.0,
        sleep=lambda _: None,
    )

    service._record_playable_availability(
        {"accompaniment_history_beats": 8, "request_id": "fetch-1"},
        availability_time_s=1.0,
    )
    service._record_playable_availability(
        {"accompaniment_history_beats": 8, "request_id": "fetch-repeat"},
        availability_time_s=2.0,
    )
    service._record_playable_availability(
        {"accompaniment_history_beats": 10, "request_id": "fetch-2"},
        availability_time_s=3.0,
    )

    spans = output.system_trace_rows
    assert [(row["start_tick"], row["end_tick_exclusive"]) for row in spans] == [
        (0, 32),
        (32, 40),
    ]
    assert [row["request_id"] for row in spans] == ["fetch-1", "fetch-2"]
    assert [row["availability_time_s"] for row in spans] == [1.0, 3.0]


def test_prompt_playable_coverage_splits_at_prompt_boundary() -> None:
    class _TraceOutput(_RecordingOutput):
        def __init__(self) -> None:
            super().__init__()
            self.system_trace_rows = []

        def log_system_trace(self, row):
            self.system_trace_rows.append(dict(row))

    output = _TraceOutput()
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=_FakePromptClient(),
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        prompt_length_ticks=32,
        now=lambda: 0.0,
        sleep=lambda _: None,
    )

    service._record_playable_availability(
        {"accompaniment_history_beats": 10, "request_id": "cross-boundary"},
        availability_time_s=4.0,
    )

    assert [
        (
            row["start_tick"],
            row["end_tick_exclusive"],
            row["generation_start_tick"],
            row["source_stage"],
        )
        for row in output.system_trace_rows
    ] == [
        (0, 32, 0, "prompt"),
        (32, 40, 32, "continuation"),
    ]


def test_prompt_continuation_system_trace_is_inert_without_callable_logger(
    monkeypatch,
) -> None:
    monkeypatch.delenv("LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE", raising=False)
    monkeypatch.delenv("LEKAI_PROMPT_CONTINUATION_TRACE_PATH", raising=False)
    now_calls = 0

    def counting_now() -> float:
        nonlocal now_calls
        now_calls += 1
        return 0.0

    output = _RecordingOutput()
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=_FakePromptClient(),
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        now=counting_now,
        sleep=lambda _: None,
    )
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.0)
    service._running = True

    service._schedule_playable([_note(55, 0)], current_tick=0, arrival_time_s=0.01)
    service._record_playable_availability(
        {"accompaniment_history_beats": 8},
        availability_time_s=0.01,
    )
    assert service._system_trace_frames == {}
    assert service._system_trace_coverage_end_tick == 0

    service._tick_loop(max_ticks=1)

    assert now_calls == 1
    assert [(source, event.pitch) for event, source in output.events] == [("model", 55)]


def test_prompt_continuation_cumulative_duplicates_keep_first_trace_provenance(
    monkeypatch,
) -> None:
    monkeypatch.delenv("LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE", raising=False)

    class _TraceOutput(_RecordingOutput):
        def __init__(self) -> None:
            super().__init__()
            self.system_trace_rows = []

        def log_system_trace(self, row):
            self.system_trace_rows.append(dict(row))

    output = _TraceOutput()
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=_FakePromptClient(),
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        now=lambda: 0.0,
        sleep=lambda _: None,
    )
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.0)
    service._running = True
    cumulative = [_note(55, 0), _placeholder(1)]
    service._playable_q.put(
        _PlayableBatch(cumulative, {"phase": "ready"}, arrival_time_s=0.01)
    )
    service._playable_q.put(
        _PlayableBatch(cumulative, {"phase": "ready"}, arrival_time_s=0.02)
    )

    service._tick_loop(max_ticks=2)

    assert [(source, event.pitch) for event, source in output.events] == [("model", 55)]
    rows = {row["tick"]: row for row in output.system_trace_rows}
    assert rows[0]["decision"] == "note"
    assert rows[0]["arrival_time_s"] == pytest.approx(0.01)
    assert rows[0]["emitted_model_note_on_count"] == 1
    assert rows[1]["decision"] == "rest"
    assert rows[1]["arrival_time_s"] == pytest.approx(0.01)
    assert rows[1]["emitted_model_note_on_count"] == 0


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


def test_prompt_start_waits_for_closed_observation_window() -> None:
    tempo = Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4)
    now_value = [0.0]
    injected = [False]
    service = None

    def fake_now() -> float:
        return now_value[0]

    def fake_sleep(duration: float) -> None:
        nonlocal service
        tick_31_buffer_end = tempo.tick_to_seconds(31) + (
            tempo.seconds_per_tick
            * PromptContinuationRealtimeService._INPUT_BUFFER_RATIO
        )
        tick_32_boundary = tempo.tick_to_seconds(32)
        sleep_end = now_value[0] + duration
        if (
            not injected[0]
            and now_value[0] >= tick_31_buffer_end - 1e-9
            and sleep_end >= tick_32_boundary - 1e-9
        ):
            assert service is not None
            assert service._start_enqueued is False
            service._event_q.put(_note(62, 31))
            service._event_q.put(_note(64, 32))
            injected[0] = True
        now_value[0] = sleep_end

    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=_FakePromptClient(),
        output_sink=_RecordingOutput(),
        tempo=tempo,
        scheduler=PlaybackScheduler(),
        prompt_length_ticks=32,
        generation_interval_ticks=4,
        now=fake_now,
        sleep=fake_sleep,
    )
    service._runtime = SimpleNamespace(
        session_start_time=0.0,
        timeline_start_time=0.0,
    )
    service._running = True

    service._tick_loop(max_ticks=33)

    assert injected[0] is True
    action = service._control_q.get_nowait()
    assert action.kind == "start"
    assert action.observed_until_tick == 32
    assert [(event.pitch, event.tick) for event in action.melody_events] == [(62, 31)]
    assert [(event.pitch, event.tick) for event in service._pending_append_events] == [
        (64, 32)
    ]


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


def test_prompt_continuation_paired_mode_traces_future_placeholder_without_playing(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE", "paired_future_only")

    class _TraceOutput(_RecordingOutput):
        def __init__(self) -> None:
            super().__init__()
            self.system_trace_rows = []

        def log_system_trace(self, row):
            self.system_trace_rows.append(dict(row))

    output = _TraceOutput()
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=_FakePromptClient(),
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        now=lambda: 0.0,
        sleep=lambda _: None,
    )
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.0)
    service._running = True

    service._schedule_playable([_placeholder(1)], current_tick=0, arrival_time_s=0.01)
    assert service._scheduler.get_events_at_tick(1) == []
    assert 1 in service._system_trace_frames

    service._tick_loop(max_ticks=2)

    assert output.events == []
    row = output.system_trace_rows[1]
    assert row["tick"] == 1
    assert row["decision"] == "rest"
    assert row["arrival_time_s"] == pytest.approx(0.01)
    assert row["policy"] == "future_placeholder"
    assert row["emitted_model_note_on_count"] == 0
    assert row["explicit_rest"] is True


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


def test_prompt_continuation_without_recovery_closes_active_late_note_off_at_current_tick(
    monkeypatch,
):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "0")
    service = _make_service()
    output = service._output
    sounding_note = service._to_model_event(_note(60, 4), current_tick=4)
    service._output_model_event(sounding_note)
    output.events.clear()

    service._schedule_playable([_note_off(60, 6)], current_tick=8)

    scheduled = service._scheduler.get_events_at_tick(8)
    assert _event_signature(scheduled) == [(60, EventType.NOTE_OFF)]
    assert scheduled[0].tick == 8
    assert service._model_event_key(sounding_note) in service._active_model_note_keys
    service._output_model_event(scheduled[0])
    assert service._model_event_key(sounding_note) not in service._active_model_note_keys
    assert any(
        "closed 1 active note(s) from late note_off" in message
        for _state, message in output.statuses
    )


def test_prompt_continuation_without_recovery_drops_inactive_late_note_off_as_orphan(
    monkeypatch,
):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "0")
    service = _make_service()
    output = service._output

    service._schedule_playable([_note_off(60, 6)], current_tick=8)

    assert service._scheduler.get_events_at_tick(8) == []
    assert any(
        "dropped 1 orphan late note_off event(s)" in message
        for _state, message in output.statuses
    )


def test_prompt_continuation_without_recovery_still_drops_late_note_on(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "0")
    service = _make_service()

    service._schedule_playable([_note(60, 6)], current_tick=8)

    assert service._scheduler.get_events_at_tick(8) == []
    assert service._model_event_key(_note(60, 6)) not in service._active_model_note_keys


def test_prompt_continuation_emits_same_tick_note_off_before_note_on(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "0")
    service = _make_service()
    output = service._output
    service._output_model_event(service._to_model_event(_note(60, 0), current_tick=0))
    output.events.clear()

    model_note_on = service._to_model_event(_note(60, 0), current_tick=1)
    model_note_off = service._to_model_event(_note_off(60, 1), current_tick=1)
    service._scheduler.schedule(model_note_on, 1)
    service._scheduler.schedule(model_note_off, 1)
    service._running = True
    service._tick_loop(max_ticks=2)

    assert _event_signature([event for event, _source in output.events]) == [
        (60, EventType.NOTE_OFF),
        (60, EventType.NOTE_ON),
    ]
    assert service._model_event_key(_note(60, 1)) in service._active_model_note_keys


def test_prompt_continuation_repeated_history_schedules_active_late_note_off_once(
    monkeypatch,
):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "0")
    service = _make_service()
    service._output_model_event(service._to_model_event(_note(60, 4), current_tick=4))
    cumulative = [_note(60, 4), _note_off(60, 6)]

    service._schedule_playable(cumulative, current_tick=8)
    service._schedule_playable(cumulative, current_tick=8)

    scheduled = service._scheduler.get_events_at_tick(8)
    assert _event_signature(scheduled) == [(60, EventType.NOTE_OFF)]
    assert scheduled[0].tick == 8


def test_prompt_continuation_distinct_late_note_offs_reserve_one_active_closure(
    monkeypatch,
):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "0")
    service = _make_service()
    output = service._output
    sounding_note = service._to_model_event(_note(60, 4), current_tick=4)
    service._output_model_event(sounding_note)
    output.events.clear()
    key = service._model_event_key(sounding_note)

    service._schedule_playable([_note_off(60, 6)], current_tick=8)
    service._schedule_playable([_note_off(60, 7)], current_tick=8)

    assert key in service._active_model_note_keys
    assert key in service._pending_late_note_off_keys
    scheduled = service._scheduler.get_events_at_tick(8)
    assert _event_signature(scheduled) == [(60, EventType.NOTE_OFF)]
    service._output_model_event(scheduled[0])
    assert _event_signature([event for event, _source in output.events]) == [
        (60, EventType.NOTE_OFF),
    ]
    assert key not in service._active_model_note_keys
    assert key not in service._pending_late_note_off_keys
    assert any(
        "skipped 1 pending late note_off event(s)" in message
        for _state, message in output.statuses
    )


def test_prompt_continuation_case11_late_transition_only_closes_active_pitch(
    monkeypatch,
):
    monkeypatch.setenv("LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS", "0")
    service = _make_service()
    output = service._output
    sounding_note = service._to_model_event(_note(56, 280), current_tick=280)
    service._output_model_event(sounding_note)
    output.events.clear()
    pitch_56_key = service._model_event_key(sounding_note)
    pitch_52_key = service._model_event_key(_note(52, 284))

    service._schedule_playable(
        [_note_off(56, 284), _note(52, 284)],
        current_tick=285,
    )

    scheduled = service._scheduler.get_events_at_tick(285)
    assert _event_signature(scheduled) == [(56, EventType.NOTE_OFF)]
    assert scheduled[0].tick == 285
    assert pitch_56_key in service._active_model_note_keys
    assert pitch_52_key not in service._active_model_note_keys
    service._output_model_event(scheduled[0])
    assert _event_signature([event for event, _source in output.events]) == [
        (56, EventType.NOTE_OFF),
    ]
    assert pitch_56_key not in service._active_model_note_keys
    assert pitch_52_key not in service._active_model_note_keys


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
    service = _make_service(model_condition_bpm=137)
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
    assert client.start_requests[0]["bpm"] == 137
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


def test_protocol_worker_records_ordered_replay_requests_with_acknowledgements():
    class _ReplayAuditOutput(_RecordingOutput):
        def __init__(self) -> None:
            super().__init__()
            self.replay_requests = []

        def log_prompt_continuation_replay_request(self, row):
            self.replay_requests.append(dict(row))

    class _JsonPromptClient(_FakePromptClient):
        def start(self, **kwargs):
            self.start_calls += 1
            self.start_requests.append(dict(kwargs))
            return {"accepted": True, "phase": "prompt_running"}

        def append_melody(self, **kwargs):
            self.append_calls += 1
            return {"accepted": True, "phase": "catchup_running"}

    output = _ReplayAuditOutput()
    client = _JsonPromptClient()
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=client,
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        model_condition_bpm=137,
        protocol_poll_interval_s=0.01,
    )
    service._runtime = SimpleNamespace(
        session_start_time=0.0,
        timeline_start_time=0.0,
    )
    service._running = True
    service._control_q.put(
        _ControlAction(
            kind="start",
            melody_events=[_note(60, 0), _note_off(60, 4)],
            observed_until_tick=32,
        )
    )
    service._control_q.put(
        _ControlAction(
            kind="append",
            melody_events=[_note(64, 34)],
            observed_until_tick=36,
        )
    )

    worker = threading.Thread(target=service._protocol_worker)
    worker.start()
    deadline = time.monotonic() + 1.0
    while len(output.replay_requests) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    service._running = False
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert [row["sequence"] for row in output.replay_requests] == [1, 2]
    assert [row["operation"] for row in output.replay_requests] == [
        "start",
        "append",
    ]
    start_row, append_row = output.replay_requests
    assert start_row["request"] == {
        "melody_events": [
            {
                "type": "note_on",
                "pitch": 60,
                "tick": 0,
                "velocity": 100,
                "channel": 0,
                "program": 0,
            },
            {
                "type": "note_off",
                "pitch": 60,
                "tick": 4,
                "velocity": 0,
                "channel": 0,
                "program": 0,
            },
        ],
        "observed_until_tick": 32,
        "prompt_length_ticks": 32,
        "generation_interval_ticks": 4,
        "bpm": 137,
    }
    assert append_row["request"] == {
        "melody_events": [
            {
                "type": "note_on",
                "pitch": 64,
                "tick": 34,
                "velocity": 100,
                "channel": 0,
                "program": 0,
            }
        ],
        "observed_until_tick": 36,
    }
    assert append_row["protocol_context"] == {
        "prompt_length_ticks": 32,
        "generation_interval_ticks": 4,
        "bpm": 137,
    }
    assert start_row["acknowledgement"] == {
        "accepted": True,
        "phase": "prompt_running",
    }
    assert append_row["acknowledgement"] == {
        "accepted": True,
        "phase": "catchup_running",
    }


def test_protocol_worker_playable_fetch_without_system_logger_skips_trace_clock_read(
    monkeypatch,
):
    monkeypatch.delenv("LEKAI_PROMPT_CONTINUATION_TRACE_PATH", raising=False)
    now_calls = 0

    def counting_now() -> float:
        nonlocal now_calls
        now_calls += 1
        return 0.0

    class _StoppingPromptClient(_FakePromptClient):
        def __init__(self):
            super().__init__()
            self.stop = lambda: None

        def playable(self):
            response = super().playable()
            self.stop()
            return response

    client = _StoppingPromptClient()
    client.playable_responses = [[_note(55, 36)]]
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=client,
        output_sink=_RecordingOutput(),
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        protocol_poll_interval_s=0.01,
        now=counting_now,
        sleep=lambda _: None,
    )
    client.stop = lambda: setattr(service, "_running", False)
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.0)
    service._running = True
    service._control_q.put(
        _ControlAction(kind="start", melody_events=[_note(60, 0)], observed_until_tick=32)
    )
    service._control_q.put(
        _ControlAction(kind="append", melody_events=[], observed_until_tick=36)
    )

    service._protocol_worker()

    playable = service._normalize_playable_item(service._playable_q.get_nowait())
    assert client.playable_calls == 1
    assert playable.arrival_time_s is None
    assert now_calls == 1


def test_protocol_worker_records_availability_after_playable_returns(monkeypatch):
    monkeypatch.delenv("LEKAI_PROMPT_CONTINUATION_TRACE_PATH", raising=False)

    class _TraceOutput(_RecordingOutput):
        def __init__(self) -> None:
            super().__init__()
            self.system_trace_rows = []

        def log_system_trace(self, row):
            self.system_trace_rows.append(dict(row))

    class _StoppingPromptClient(_FakePromptClient):
        def __init__(self):
            super().__init__()
            self.playable_returned = False
            self.stop = lambda: None

        def playable(self):
            self.playable_calls += 1
            self.playable_returned = True
            self.stop()
            return [], {
                "phase": "ready",
                "accompaniment_history_beats": 2,
                "request_id": "http-playable",
            }

    client = _StoppingPromptClient()

    def now() -> float:
        return 7.0 if client.playable_returned else 1.0

    output = _TraceOutput()
    service = PromptContinuationRealtimeService(
        input_source=_NoopInput(),
        prompt_client=client,
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        protocol_poll_interval_s=0.01,
        now=now,
        sleep=lambda _: None,
    )
    client.stop = lambda: setattr(service, "_running", False)
    service._runtime = SimpleNamespace(session_start_time=0.0, timeline_start_time=0.0)
    service._running = True
    service._control_q.put(
        _ControlAction(kind="start", melody_events=[_note(60, 0)], observed_until_tick=32)
    )
    service._control_q.put(
        _ControlAction(kind="append", melody_events=[], observed_until_tick=36)
    )

    service._protocol_worker()

    assert output.system_trace_rows == [
        {
            "schema_version": 2,
            "record_type": "availability_span",
            "mode": "realtime",
            "condition": "prompt_continuation",
            "clock_domain": "service_now",
            "start_tick": 0,
            "end_tick_exclusive": 8,
            "availability_time_s": 7.0,
            "generation_start_tick": 0,
            "request_id": "http-playable",
            "source_stage": "prompt",
        }
    ]


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
