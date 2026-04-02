from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from streammuse.application.services.real_time_music_service import RealTimeMusicService
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
        return None


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
