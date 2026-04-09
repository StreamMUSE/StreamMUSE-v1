from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.interfaces.timing_info import TimingInfo
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import PlaybackScheduler, Tempo


class _NoopInput:
    def read_events(self):
        return iter([])

    def close(self):
        return None


class _NoopOutput:
    def output_event(self, event, source):
        return None

    def output_tick(self, tick, bar, beat):
        return None

    def output_stats(self, **kwargs):
        return None

    def output_status(self, state, message=""):
        return None

    def output_config(self, config):
        return None

    def close(self):
        return None


class _NoopInference:
    def generate_accompaniment(self, *args, **kwargs):
        raise NotImplementedError

    def inject_history(self, *args, **kwargs):
        return None

    def set_injection_offset(self, offset_ticks: int):
        return None

    def clear_history(self):
        return {"success": True, "melody_history": [], "accompaniment_history": []}


def _make_service() -> RealTimeMusicService:
    return RealTimeMusicService(
        input_source=_NoopInput(),
        inference_engine=_NoopInference(),
        output_sink=_NoopOutput(),
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        now=lambda: 0.0,
        sleep=lambda _: None,
    )


def _note(pitch: int, tick: int) -> MusicalEvent:
    return MusicalEvent(tick=tick, pitch=pitch, event_type=EventType.NOTE_ON, velocity=100)


def test_tick_loop_sends_all_events_on_first_cycle():
    svc = _make_service()
    svc._melody_history = [_note(60, 0), _note(64, 1)]
    svc._runtime = SimpleNamespace(session_start_time=0.0)
    svc._running = True

    svc._tick_loop(max_ticks=1)

    generation_start_tick, sent_events = svc._inference_request_queue.get_nowait()
    assert generation_start_tick == 0
    assert [e.pitch for e in sent_events] == [60, 64]
    assert svc._last_sent_index == 2


def test_tick_loop_only_sends_incremental_events_after_first_cycle():
    svc = _make_service()
    svc._melody_history = [_note(60, 0), _note(64, 1)]
    svc._runtime = SimpleNamespace(session_start_time=0.0)
    svc._running = True
    svc._tick_loop(max_ticks=1)
    _ = svc._inference_request_queue.get_nowait()

    svc._melody_history.append(_note(67, 2))
    svc._runtime = SimpleNamespace(session_start_time=0.0)
    svc._running = True
    svc._tick_loop(max_ticks=1)

    _, sent_events = svc._inference_request_queue.get_nowait()
    assert [e.pitch for e in sent_events] == [67]
    assert svc._last_sent_index == 3


def test_tick_loop_does_not_enqueue_when_no_incremental_events():
    svc = _make_service()
    svc._melody_history = [_note(60, 0)]
    svc._runtime = SimpleNamespace(session_start_time=0.0)
    svc._running = True
    svc._tick_loop(max_ticks=1)
    _ = svc._inference_request_queue.get_nowait()

    svc._runtime = SimpleNamespace(session_start_time=0.0)
    svc._running = True
    svc._tick_loop(max_ticks=1)

    with pytest.raises(queue.Empty):
        svc._inference_request_queue.get_nowait()


def test_inference_worker_latest_only_merges_events_and_keeps_latest_tick():
    class _DebugOutput(_NoopOutput):
        def __init__(self):
            self.status_messages = []

        def output_status(self, state, message=""):
            self.status_messages.append((state, message))

    service_holder = {}

    class _LatestOnlyInference:
        def __init__(self):
            self.calls = []

        def generate_accompaniment(self, melody_events, generation_start_tick, generation_length_frames, **kwargs):
            _ = generation_length_frames, kwargs
            self.calls.append((generation_start_tick, [event.pitch for event in melody_events]))
            service_holder["svc"]._running = False
            return [], TimingInfo(
                request_arrival_time=0.0,
                response_output_time=0.0,
                preprocess_start_time=0.0,
                inference_start_time=0.0,
                inference_end_time=0.0,
                postprocess_start_time=0.0,
            )

        def inject_history(self, *args, **kwargs):
            return None

        def set_injection_offset(self, offset_ticks: int):
            _ = offset_ticks
            return None

        def clear_history(self):
            return {"success": True, "melody_history": [], "accompaniment_history": []}

    output = _DebugOutput()
    inference = _LatestOnlyInference()
    svc = RealTimeMusicService(
        input_source=_NoopInput(),
        inference_engine=inference,
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
        now=lambda: 0.0,
        sleep=lambda _: None,
    )
    service_holder["svc"] = svc

    svc._running = True
    svc._inference_request_queue.put((2, [_note(60, 0)]))
    svc._inference_request_queue.put((4, [_note(62, 1)]))
    svc._inference_request_queue.put((6, [_note(64, 2)]))

    svc._inference_worker()

    assert inference.calls == [(6, [60, 62, 64])]
    _, response_tick = svc._inference_response_queue.get_nowait()
    assert response_tick == 6
    assert any("dropped 2 stale request" in msg for _, msg in output.status_messages)
