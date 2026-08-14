"""Tests for sample-clock-driven rap playback observation."""

from __future__ import annotations

from collections import deque
from threading import Event, Thread
from time import monotonic, sleep

import pytest

from streammuse.application.rap.playback import RapPlaybackService
from streammuse.domain.rap import (
    AudioFormat,
    AudioPlaybackNotice,
    AudioPlaybackNoticeKind,
    AudioPlaybackSnapshot,
    PcmAudio,
    PlaybackState,
    PreparedRapBar,
    RapEventType,
    ScheduledSyllable,
    Syllable,
    SyllablePlacementDiagnostic,
)
from streammuse.domain.rap.models import BeatSlot
from streammuse.domain.timing import Tempo


class RecordingPublisher:
    def __init__(self) -> None:
        self.events: list[PublishedEvent] = []

    def emit(self, event_type: RapEventType, **kwargs: object) -> None:
        self.events.append(PublishedEvent(event_type, kwargs))

    def only(self, event_type: RapEventType) -> "PublishedEvent":
        events = [event for event in self.events if event.event_type == event_type]
        assert len(events) == 1
        return events[0]


class BlockingStartPublisher(RecordingPublisher):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def emit(self, event_type: RapEventType, **kwargs: object) -> None:
        if event_type == RapEventType.SESSION_STARTED:
            self.entered.set()
            assert self.release.wait(timeout=1.0)
        super().emit(event_type, **kwargs)


class BlockingLifecyclePublisher(RecordingPublisher):
    def __init__(self, blocked_event: RapEventType) -> None:
        super().__init__()
        self._blocked_event = blocked_event
        self.entered = Event()
        self.release = Event()
        self.session_started = Event()
        self.state_reader = None
        self.state_at_blocked_event: PlaybackState | None = None

    def emit(self, event_type: RapEventType, **kwargs: object) -> None:
        if event_type == self._blocked_event:
            self.entered.set()
            assert self.release.wait(timeout=1.0)
            if self.state_reader is not None:
                self.state_at_blocked_event = self.state_reader()
        super().emit(event_type, **kwargs)
        if event_type == RapEventType.SESSION_STARTED:
            self.session_started.set()


class TimelinePublisher(RecordingPublisher):
    def __init__(self, timeline: list[object]) -> None:
        super().__init__()
        self._timeline = timeline

    def emit(self, event_type: RapEventType, **kwargs: object) -> None:
        super().emit(event_type, **kwargs)
        self._timeline.append(event_type)


class PublishedEvent:
    def __init__(self, event_type: RapEventType, values: dict[str, object]) -> None:
        self.event_type = event_type
        self.bar = values.get("bar")
        self.tick = values.get("tick")
        self.payload = values.get("payload", {})


class FakeRapAudioSink:
    def __init__(self) -> None:
        self.state = PlaybackState.STOPPED
        self.absolute_frame = 0
        self.current_bar: int | None = None
        self.last_completed_bar: int | None = None
        self.queued: list[PreparedRapBar] = []
        self.notices: deque[AudioPlaybackNotice] = deque()
        self.start_calls = 0
        self.stop_requests = 0
        self.reset_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        self.state = PlaybackState.RUNNING

    def enqueue(self, bar: PreparedRapBar) -> None:
        self.queued.append(bar)

    def request_stop_after_bar(self) -> AudioPlaybackSnapshot:
        self.stop_requests += 1
        self.state = PlaybackState.STOP_REQUESTED
        return self.snapshot()

    def reset(self) -> None:
        self.reset_calls += 1
        self.state = PlaybackState.STOPPED
        self.absolute_frame = 0
        self.current_bar = None
        self.last_completed_bar = None
        self.queued.clear()
        self.notices.clear()

    def snapshot(self) -> AudioPlaybackSnapshot:
        return AudioPlaybackSnapshot(
            state=self.state,
            current_bar=self.current_bar,
            last_completed_bar=self.last_completed_bar,
            frame_in_bar=0,
            absolute_frame=self.absolute_frame,
            queue_depth=len(self.queued),
            underrun_count=0,
            buffered_seconds=sum(bar.audio.duration_seconds for bar in self.queued),
        )

    def drain_notices(self) -> tuple[AudioPlaybackNotice, ...]:
        notices = tuple(self.notices)
        self.notices.clear()
        return notices

    def close(self) -> None:
        self.close_calls += 1
        self.state = PlaybackState.CLOSED

    def set_absolute_frame(self, frame: int, *, current_bar: int | None = 0) -> None:
        self.absolute_frame = frame
        self.current_bar = current_bar

    def publish(self, kind: AudioPlaybackNoticeKind, *, bar: int | None = None, message: str = "test") -> None:
        self.notices.append(
            AudioPlaybackNotice(
                kind=kind,
                bar=bar,
                absolute_frame=self.absolute_frame,
                queue_depth=len(self.queued),
                message=message,
                buffered_seconds=sum(item.audio.duration_seconds for item in self.queued),
            )
        )

    def complete_bar(self, bar: int) -> None:
        completed = next(item for item in self.queued if item.bar == bar)
        self.current_bar = bar
        self.absolute_frame += completed.audio.frame_count
        self.queued.remove(completed)
        self.last_completed_bar = bar
        self.publish(AudioPlaybackNoticeKind.BAR_COMPLETED, bar=bar)
        self.state = PlaybackState.STOPPED
        self.current_bar = None
        self.publish(AudioPlaybackNoticeKind.STOPPED)


class SnapshotToArmRaceSink(FakeRapAudioSink):
    """Activate the successor while atomically arming the stop boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.last_completed_bar = 1
        self.activation_attempts = 0

    def request_stop_after_bar(self):
        self._activate_next_bar()
        return super().request_stop_after_bar()

    def _activate_next_bar(self) -> None:
        if self.current_bar is None:
            self.activation_attempts += 1
            self.current_bar = 2


class BlockingSnapshotSink(FakeRapAudioSink):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()
        self.block = False

    def snapshot(self) -> AudioPlaybackSnapshot:
        if self.block:
            self.entered.set()
            assert self.release.wait(timeout=1.0)
        return super().snapshot()


class BlockingCloseSink(FakeRapAudioSink):
    def __init__(self) -> None:
        super().__init__()
        self.close_entered = Event()
        self.allow_close = Event()

    def close(self) -> None:
        self.close_entered.set()
        assert self.allow_close.wait(timeout=1.0)
        super().close()


class BlockingResetSink(FakeRapAudioSink):
    def __init__(self) -> None:
        super().__init__()
        self.reset_entered = Event()
        self.allow_reset = Event()
        self.reset_finished = Event()

    def reset(self) -> None:
        self.reset_entered.set()
        assert self.allow_reset.wait(timeout=1.0)
        super().reset()
        self.reset_finished.set()


class ResetCompletionPublisher(RecordingPublisher):
    def __init__(self, sink: BlockingResetSink) -> None:
        super().__init__()
        self._sink = sink
        self.cleanup_completed_before_session_reset: bool | None = None

    def emit(self, event_type: RapEventType, **kwargs: object) -> None:
        if event_type == RapEventType.SESSION_RESET:
            self.cleanup_completed_before_session_reset = self._sink.reset_finished.is_set()
        super().emit(event_type, **kwargs)


class StaleSnapshotSink(FakeRapAudioSink):
    def __init__(self) -> None:
        super().__init__()
        self.stale_snapshot: AudioPlaybackSnapshot | None = None

    def capture_stale_snapshot(self) -> None:
        self.stale_snapshot = super().snapshot()

    def snapshot(self) -> AudioPlaybackSnapshot:
        return self.stale_snapshot or super().snapshot()


class InterleavedSnapshotSink(FakeRapAudioSink):
    def __init__(self) -> None:
        super().__init__()
        self._frames = deque((12_000, 24_000))
        self.second_snapshot_taken = Event()

    def snapshot(self) -> AudioPlaybackSnapshot:
        frame = self._frames.popleft()
        if not self._frames:
            self.second_snapshot_taken.set()
        self.absolute_frame = frame
        return super().snapshot()


class ManualPlaybackService(RapPlaybackService):
    """Disable automatic observation so tests control every poll boundary."""

    def _observe(self, epoch: int, stop: Event) -> None:
        stop.wait(timeout=1.0)


class ContendingObserverPlaybackService(RapPlaybackService):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.allow_poll = Event()
        self.poll_attempted = Event()
        self.finished = Event()
        self.dispatch_active = Event()
        self.join_during_dispatch = Event()

    def _observe(self, epoch: int, stop: Event) -> None:
        assert self.allow_poll.wait(timeout=1.0)
        self.poll_attempted.set()
        self.poll()
        self.finished.set()

    def _join(self, observer: Thread | None, timeout: float | None = None) -> None:
        if self.dispatch_active.is_set():
            self.join_during_dispatch.set()
        RapPlaybackService._join(observer, 0.05)


class StartMarkerPlaybackService(RapPlaybackService):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.observer_started = Event()

    def _observe(self, epoch: int, stop: Event) -> None:
        self.observer_started.set()
        stop.wait(timeout=1.0)


class ShortJoinPlaybackService(RapPlaybackService):
    """Keep the pre-fix deadlock test bounded while retaining the join ordering."""

    @staticmethod
    def _join(observer: Thread | None, timeout: float | None = None) -> None:
        RapPlaybackService._join(observer, 0.05)


def tempo() -> Tempo:
    return Tempo(60.0, 4, 4)


def prepared_bar(*, bar: int, syllable_at_tick: int | None = None, source: str = "test") -> PreparedRapBar:
    audio_format = AudioFormat()
    scheduled: tuple[ScheduledSyllable, ...] = ()
    diagnostics: tuple[SyllablePlacementDiagnostic, ...] = ()
    if syllable_at_tick is not None:
        syllable = Syllable("sample", 0, 1, stress=1)
        slot = BeatSlot(
            bar=bar,
            tick=bar * tempo().ticks_per_bar + syllable_at_tick,
            beat=syllable_at_tick // tempo().ticks_per_beat,
            tick_in_beat=syllable_at_tick % tempo().ticks_per_beat,
            accent=1.0,
            slot_index=4,
        )
        scheduled = (ScheduledSyllable(slot, syllable),)
        diagnostics = (
            SyllablePlacementDiagnostic(
                bar=bar,
                slot_index=4,
                word="sample",
                target_sample=syllable_at_tick * 12_000,
                source_frames=1,
                fitted_frames=1,
                available_frames=12_000,
                compression_ratio=1.0,
                overlap_frames=0,
                pronunciation_source="test",
            ),
        )
    return PreparedRapBar(
        bar=bar,
        text="sample lyric",
        source=source,
        fallback_reason=None,
        scheduled=scheduled,
        audio=PcmAudio(audio_format, 192_000, bytes(192_000 * 2 * 4)),
        diagnostics=diagnostics,
        warnings=(),
        render_latency_ms=0.0,
    )


def running_service(
    *,
    sink: FakeRapAudioSink | None = None,
    publisher: RecordingPublisher | None = None,
    on_tick=None,
    prepared: PreparedRapBar | None = None,
) -> tuple[RapPlaybackService, FakeRapAudioSink, RecordingPublisher]:
    selected_sink = sink or FakeRapAudioSink()
    selected_publisher = publisher or RecordingPublisher()
    service = RapPlaybackService(
        tempo=tempo(),
        sink=selected_sink,
        publisher=selected_publisher,
        on_tick=on_tick or (lambda _: None),
    )
    service.prime(prepared or prepared_bar(bar=0))
    service.start()
    return service, selected_sink, selected_publisher


def test_playback_moves_through_priming_running_and_bar_quantized_stop() -> None:
    sink = FakeRapAudioSink()
    publisher = RecordingPublisher()
    service = RapPlaybackService(tempo=tempo(), sink=sink, publisher=publisher, on_tick=lambda _: None)

    service.prime(prepared_bar(bar=0))
    assert service.state == PlaybackState.PRIMING
    service.start()
    assert service.state == PlaybackState.RUNNING
    service.request_stop()
    assert service.state == PlaybackState.STOP_REQUESTED

    sink.complete_bar(0)
    service.poll()

    assert service.state == PlaybackState.STOPPED
    assert sink.stop_requests == 1
    assert [event.event_type for event in publisher.events] == [
        RapEventType.SESSION_STARTED,
        RapEventType.STOP_REQUESTED,
        RapEventType.BAR_PLAYBACK_COMPLETED,
        RapEventType.SESSION_STOPPED,
    ]


def test_playback_notice_exports_authoritative_sink_queue_and_buffer_duration() -> None:
    service, sink, publisher = running_service(prepared=prepared_bar(bar=0))
    sink.publish(AudioPlaybackNoticeKind.BAR_STARTED, bar=0)

    service.poll()

    assert publisher.only(RapEventType.BAR_PLAYBACK_STARTED).payload == {
        "absolute_frame": 0,
        "queue_depth": 1,
        "buffered_seconds": 4.0,
        "message": "test",
    }


def test_enqueue_primes_first_bar_and_rejects_later_first_bar() -> None:
    service = RapPlaybackService(tempo=tempo(), sink=FakeRapAudioSink(), publisher=None, on_tick=lambda _: None)

    service.enqueue(prepared_bar(bar=0))

    assert service.state == PlaybackState.PRIMING
    with pytest.raises(ValueError, match="bar 0"):
        RapPlaybackService(tempo=tempo(), sink=FakeRapAudioSink(), publisher=None, on_tick=lambda _: None).enqueue(
            prepared_bar(bar=1)
        )


def test_stop_successor_uses_sink_active_bar_when_observer_tick_is_stale() -> None:
    sink = FakeRapAudioSink()
    service = RapPlaybackService(tempo=tempo(), sink=sink, publisher=None, on_tick=lambda _: None)
    sink.state = PlaybackState.RUNNING
    sink.current_bar = 1
    service._current_tick = 15  # Simulate observer lag at the bar-one callback boundary.

    assert service.stop_successor_bar == 2


def test_stop_arming_prefers_new_active_bar_over_last_completed_bar() -> None:
    sink = SnapshotToArmRaceSink()
    sink.state = PlaybackState.RUNNING
    service = RapPlaybackService(tempo=tempo(), sink=sink, publisher=None, on_tick=lambda _: None)
    service._state = PlaybackState.RUNNING
    service._current_tick = 15

    assert service.request_stop() == 3
    assert sink.activation_attempts == 1


def test_stop_arming_uses_last_completed_bar_during_no_active_gap() -> None:
    sink = FakeRapAudioSink()
    sink.state = PlaybackState.RUNNING
    sink.last_completed_bar = 1
    service = RapPlaybackService(tempo=tempo(), sink=sink, publisher=None, on_tick=lambda _: None)
    service._state = PlaybackState.RUNNING
    service._current_tick = 15

    assert service.request_stop() == 2


def test_reset_requires_stopped_and_clears_audio_state() -> None:
    service, sink, publisher = running_service()
    sink.complete_bar(0)
    service.poll()

    service.reset()

    assert service.state == PlaybackState.STOPPED
    assert sink.reset_calls == 1
    assert service.current_tick is None
    assert publisher.only(RapEventType.SESSION_RESET).payload == {"playback_state": "stopped"}


def test_reset_rejects_running_playback() -> None:
    service, _sink, _publisher = running_service()

    with pytest.raises(RuntimeError, match="stopped"):
        service.reset()

    service.close()


def test_stop_during_priming_discards_queued_bar_and_allows_fresh_prime() -> None:
    sink = FakeRapAudioSink()
    service = RapPlaybackService(tempo=tempo(), sink=sink, publisher=None, on_tick=lambda _: None)
    queued = prepared_bar(bar=0, source="discarded")
    fresh = prepared_bar(bar=0, source="fresh")
    service.prime(queued)

    service.request_stop()

    assert service.state == PlaybackState.STOPPED
    assert sink.reset_calls == 1
    assert sink.queued == []
    assert service.current_tick is None
    service.prime(fresh)
    assert sink.queued == [fresh]


def test_stop_during_priming_keeps_concurrent_prime_out_until_sink_cleanup_finishes() -> None:
    sink = BlockingResetSink()
    service = RapPlaybackService(tempo=tempo(), sink=sink, publisher=None, on_tick=lambda _: None)
    discarded = prepared_bar(bar=0, source="discarded")
    fresh = prepared_bar(bar=0, source="fresh")
    errors: list[Exception] = []
    prime_completed = Event()
    service.prime(discarded)

    stopper = Thread(target=service.request_stop)
    stopper.start()
    assert sink.reset_entered.wait(timeout=1.0)
    primer = Thread(target=lambda: _prime_and_capture(service, fresh, errors, prime_completed))
    primer.start()
    try:
        assert not prime_completed.wait(timeout=0.1)
    finally:
        sink.allow_reset.set()
        stopper.join(timeout=1.0)
        primer.join(timeout=1.0)

    assert not stopper.is_alive()
    assert not primer.is_alive()
    assert errors == []
    assert service.state == PlaybackState.PRIMING
    assert sink.queued == [fresh]


def test_reset_keeps_concurrent_prime_out_until_sink_cleanup_finishes() -> None:
    sink = BlockingResetSink()
    publisher = ResetCompletionPublisher(sink)
    service = RapPlaybackService(tempo=tempo(), sink=sink, publisher=publisher, on_tick=lambda _: None)
    fresh = prepared_bar(bar=0, source="fresh")
    reset_errors: list[Exception] = []
    prime_errors: list[Exception] = []
    reset_completed = Event()
    prime_completed = Event()

    resetter = Thread(target=lambda: _reset_and_capture(service, reset_errors, reset_completed))
    resetter.start()
    assert sink.reset_entered.wait(timeout=1.0)
    primer = Thread(target=lambda: _prime_and_capture(service, fresh, prime_errors, prime_completed))
    primer.start()
    try:
        assert not prime_completed.wait(timeout=0.1)
        assert not reset_completed.is_set()
        assert publisher.events == []
    finally:
        sink.allow_reset.set()
        resetter.join(timeout=1.0)
        primer.join(timeout=1.0)

    assert not resetter.is_alive()
    assert not primer.is_alive()
    assert reset_errors == []
    assert prime_errors == []
    assert service.state == PlaybackState.PRIMING
    assert sink.queued == [fresh]
    assert publisher.only(RapEventType.SESSION_RESET).payload == {"playback_state": "stopped"}
    assert publisher.cleanup_completed_before_session_reset is True


def test_reset_event_precedes_concurrent_restart_session_started() -> None:
    publisher = BlockingLifecyclePublisher(RapEventType.SESSION_RESET)
    service = ManualPlaybackService(
        tempo=tempo(),
        sink=FakeRapAudioSink(),
        publisher=publisher,
        on_tick=lambda _: None,
    )
    errors: list[Exception] = []
    reset_completed = Event()
    start_requested = Event()

    resetter = Thread(target=lambda: _reset_and_capture(service, errors, reset_completed))
    resetter.start()
    assert publisher.entered.wait(timeout=1.0)
    service.prime(prepared_bar(bar=0, source="fresh"))
    starter = Thread(target=lambda: _start_and_capture(service, errors, start_requested))
    starter.start()
    assert start_requested.wait(timeout=1.0)
    try:
        assert not publisher.session_started.wait(timeout=0.1)
        assert starter.is_alive()
        assert not reset_completed.is_set()
    finally:
        publisher.release.set()
        resetter.join(timeout=1.0)
        starter.join(timeout=1.0)

    assert not resetter.is_alive()
    assert not starter.is_alive()
    assert errors == []
    assert [event.event_type for event in publisher.events] == [
        RapEventType.SESSION_RESET,
        RapEventType.SESSION_STARTED,
    ]
    service.close()


def test_close_waits_for_reset_event_and_prevents_a_stale_reset_after_closed() -> None:
    publisher = BlockingLifecyclePublisher(RapEventType.SESSION_RESET)
    service = ManualPlaybackService(
        tempo=tempo(),
        sink=FakeRapAudioSink(),
        publisher=publisher,
        on_tick=lambda _: None,
    )
    publisher.state_reader = lambda: service.state
    errors: list[Exception] = []
    reset_completed = Event()
    close_completed = Event()

    resetter = Thread(target=lambda: _reset_and_capture(service, errors, reset_completed))
    resetter.start()
    assert publisher.entered.wait(timeout=1.0)
    closer = Thread(target=lambda: _close_and_capture(service, errors, close_completed))
    closer.start()
    try:
        assert not close_completed.wait(timeout=0.1)
        assert service.state == PlaybackState.STOPPED
    finally:
        publisher.release.set()
        resetter.join(timeout=1.0)
        closer.join(timeout=1.0)

    assert not resetter.is_alive()
    assert not closer.is_alive()
    assert errors == []
    assert publisher.state_at_blocked_event == PlaybackState.STOPPED
    assert [event.event_type for event in publisher.events] == [RapEventType.SESSION_RESET]
    assert service.state == PlaybackState.CLOSED


def test_notice_kinds_map_to_canonical_events() -> None:
    service, sink, publisher = running_service()
    sink.publish(AudioPlaybackNoticeKind.BAR_STARTED, bar=0)
    sink.publish(AudioPlaybackNoticeKind.UNDERRUN, message="empty queue")
    sink.publish(AudioPlaybackNoticeKind.DEVICE_FAILED, message="device unavailable")

    service.poll()

    assert [event.event_type for event in publisher.events[1:]] == [
        RapEventType.BAR_PLAYBACK_STARTED,
        RapEventType.AUDIO_UNDERRUN,
        RapEventType.AUDIO_DEVICE_FAILED,
    ]
    assert publisher.events[-1].payload == {
        "absolute_frame": 0,
        "queue_depth": 1,
        "buffered_seconds": 4.0,
        "message": "device unavailable",
    }

    service.close()


def test_close_from_tick_callback_defers_observer_join_until_poll_releases_its_mutex() -> None:
    sink = FakeRapAudioSink()
    holder: dict[str, ContendingObserverPlaybackService] = {}

    def on_tick(_: int) -> None:
        service = holder["service"]
        service.allow_poll.set()
        assert service.poll_attempted.wait(timeout=1.0)
        service.dispatch_active.set()
        try:
            service.close()
        finally:
            service.dispatch_active.clear()

    service = ContendingObserverPlaybackService(tempo=tempo(), sink=sink, publisher=None, on_tick=on_tick)
    holder["service"] = service
    service.prime(prepared_bar(bar=0))
    service.start()
    caller = Thread(target=service.poll)
    caller.start()
    caller.join(timeout=1.0)

    assert not caller.is_alive()
    assert service.finished.wait(timeout=1.0)
    assert not service.join_during_dispatch.is_set()
    assert service.state == PlaybackState.CLOSED
    assert sink.close_calls == 1


def test_continuation_notice_payloads_use_global_absolute_samples() -> None:
    sink = FakeRapAudioSink()
    publisher = RecordingPublisher()
    service = ManualPlaybackService(tempo=tempo(), sink=sink, publisher=publisher, on_tick=lambda _: None)
    service.prime(prepared_bar(bar=0))
    service.start()
    service.request_stop()
    sink.complete_bar(0)
    service.poll()
    service.prime(prepared_bar(bar=1))
    service.start()
    sink.set_absolute_frame(96, current_bar=1)
    sink.publish(AudioPlaybackNoticeKind.BAR_STARTED, bar=1)
    sink.publish(AudioPlaybackNoticeKind.UNDERRUN, message="empty queue")
    sink.publish(AudioPlaybackNoticeKind.BAR_COMPLETED, bar=1)

    service.poll()

    events = [event for event in publisher.events if event.event_type in {
        RapEventType.BAR_PLAYBACK_STARTED,
        RapEventType.AUDIO_UNDERRUN,
        RapEventType.BAR_PLAYBACK_COMPLETED,
    }]
    assert [event.payload["absolute_frame"] for event in events[-3:]] == [192_096, 192_096, 192_096]

    service.close()


def test_observer_emits_each_tick_once_when_polling_skips_frames() -> None:
    ticks: list[int] = []
    service, sink, _publisher = running_service(on_tick=ticks.append)

    sink.set_absolute_frame(36_100)
    service.poll()
    service.poll()

    assert ticks == [0, 1, 2, 3]
    assert service.current_tick == 3

    service.close()


def test_syllable_event_reports_exact_mixed_sample_not_poll_delay() -> None:
    publisher = RecordingPublisher()
    service, sink, _publisher = running_service(publisher=publisher, prepared=prepared_bar(bar=0, syllable_at_tick=2))

    sink.set_absolute_frame(30_000)
    service.poll()

    event = publisher.only(RapEventType.SYLLABLE_EMITTED)
    assert event.tick == 2
    assert event.payload == {
        "scheduled_sample": 24_000,
        "software_error_samples": 0,
        "observation_delay_ms": 125.0,
        "word": "sample",
        "label": "sample",
        "stress": 1,
        "beat": 0,
        "subdivision": 2,
    }

    service.close()


def test_completed_bar_keeps_crossed_syllable_metadata_until_after_observation() -> None:
    publisher = RecordingPublisher()
    service, sink, _publisher = running_service(publisher=publisher, prepared=prepared_bar(bar=0, syllable_at_tick=2))

    sink.complete_bar(0)
    service.poll()

    assert publisher.only(RapEventType.SYLLABLE_EMITTED).payload["scheduled_sample"] == 24_000


def test_close_invalidates_an_observer_poll_already_waiting_on_the_sink() -> None:
    sink = BlockingSnapshotSink()
    service, _sink, publisher = running_service(sink=sink, prepared=prepared_bar(bar=0, syllable_at_tick=2))
    sink.block = True
    sink.set_absolute_frame(36_100)
    assert sink.entered.wait(timeout=1.0)

    closer = Thread(target=service.close)
    closer.start()
    sink.release.set()
    closer.join(timeout=1.0)

    assert not closer.is_alive()
    assert [event.event_type for event in publisher.events].count(RapEventType.SYLLABLE_EMITTED) == 0
    assert sink.close_calls == 1


def test_close_is_idempotent_and_waits_for_its_observer() -> None:
    service, sink, _publisher = running_service()
    observer = service._observer

    service.close()
    service.close()

    assert service.state == PlaybackState.CLOSED
    assert observer is not None
    assert not observer.is_alive()
    assert sink.close_calls == 1


def test_external_close_releases_lifecycle_gate_before_waiting_for_observer_callback() -> None:
    sink = BlockingCloseSink()
    callback_entered = Event()
    allow_lifecycle_call = Event()
    callback_returned = Event()
    holder: dict[str, RapPlaybackService] = {}

    def on_tick(_: int) -> None:
        callback_entered.set()
        assert allow_lifecycle_call.wait(timeout=1.0)
        holder["service"].request_stop()
        callback_returned.set()

    service = ShortJoinPlaybackService(tempo=tempo(), sink=sink, publisher=None, on_tick=on_tick)
    holder["service"] = service
    service.prime(prepared_bar(bar=0))
    service.start()
    sink.set_absolute_frame(1)
    assert callback_entered.wait(timeout=1.0)

    closer = Thread(target=service.close)
    closer.start()
    deadline = monotonic() + 1.0
    while service.state != PlaybackState.CLOSED and monotonic() < deadline:
        sleep(0.005)
    assert service.state == PlaybackState.CLOSED
    allow_lifecycle_call.set()
    assert sink.close_entered.wait(timeout=1.0)
    try:
        assert callback_returned.wait(timeout=0.2)
    finally:
        sink.allow_close.set()
        closer.join(timeout=1.0)

    assert not closer.is_alive()
    assert service.state == PlaybackState.CLOSED
    assert sink.close_calls == 1


def test_close_during_start_does_not_join_an_unstarted_observer() -> None:
    sink = FakeRapAudioSink()
    publisher = BlockingStartPublisher()
    service = StartMarkerPlaybackService(tempo=tempo(), sink=sink, publisher=publisher, on_tick=lambda _: None)
    service.prime(prepared_bar(bar=0))
    errors: list[Exception] = []

    starter = Thread(target=lambda: _capture_error(service.start, errors))
    starter.start()
    assert publisher.entered.wait(timeout=1.0)
    closer = Thread(target=lambda: _capture_error(service.close, errors))
    closer.start()
    publisher.release.set()
    starter.join(timeout=1.0)
    closer.join(timeout=1.0)

    assert not starter.is_alive()
    assert not closer.is_alive()
    assert errors == []
    assert service.state == PlaybackState.CLOSED
    assert sink.close_calls == 1
    assert not service.observer_started.is_set()


def test_restart_after_stop_discards_stale_future_bars_and_starts_next_complete_bar() -> None:
    sink = FakeRapAudioSink()
    ticks: list[int] = []
    service = ManualPlaybackService(tempo=tempo(), sink=sink, publisher=None, on_tick=ticks.append)
    first = prepared_bar(bar=0)
    stale_future = prepared_bar(bar=1, source="stale")
    continuation = prepared_bar(bar=1, source="continuation")
    service.prime(first)
    service.start()
    service.enqueue(stale_future)
    service.request_stop()
    sink.complete_bar(0)
    service.poll()

    service.prime(continuation)

    assert service.state == PlaybackState.PRIMING
    assert sink.reset_calls == 1
    assert sink.queued == [continuation]
    service.start()
    sink.set_absolute_frame(0, current_bar=1)
    service.poll()
    assert ticks == list(range(17))

    service.close()


def test_enqueue_after_stop_discards_stale_future_bars_and_primes_the_next_bar() -> None:
    sink = FakeRapAudioSink()
    service = ManualPlaybackService(tempo=tempo(), sink=sink, publisher=None, on_tick=lambda _: None)
    service.prime(prepared_bar(bar=0))
    service.start()
    service.enqueue(prepared_bar(bar=1, source="stale"))
    service.request_stop()
    sink.complete_bar(0)
    service.poll()
    continuation = prepared_bar(bar=1, source="continuation")

    service.enqueue(continuation)

    assert service.state == PlaybackState.PRIMING
    assert sink.reset_calls == 1
    assert sink.queued == [continuation]


def test_reset_after_stop_requires_a_fresh_bar_zero_session() -> None:
    sink = FakeRapAudioSink()
    service = ManualPlaybackService(tempo=tempo(), sink=sink, publisher=None, on_tick=lambda _: None)
    service.prime(prepared_bar(bar=0))
    service.start()
    service.request_stop()
    sink.complete_bar(0)
    service.poll()
    service.reset()

    with pytest.raises(ValueError, match="bar 0"):
        service.prime(prepared_bar(bar=1))
    service.prime(prepared_bar(bar=0))

    assert service.state == PlaybackState.PRIMING


def test_completion_notice_advances_observation_past_an_older_snapshot_before_metadata_removal() -> None:
    sink = StaleSnapshotSink()
    ticks: list[int] = []
    timeline: list[object] = []
    publisher = TimelinePublisher(timeline)

    def on_tick(tick: int) -> None:
        ticks.append(tick)
        timeline.append(("tick", tick))

    service = ManualPlaybackService(
        tempo=tempo(),
        sink=sink,
        publisher=publisher,
        on_tick=on_tick,
    )
    service.prime(prepared_bar(bar=0, syllable_at_tick=15))
    service.start()
    service.request_stop()
    sink.capture_stale_snapshot()
    sink.complete_bar(0)

    service.poll()

    syllable = publisher.only(RapEventType.SYLLABLE_EMITTED)
    stopped_index = next(index for index, event in enumerate(publisher.events) if event.event_type == RapEventType.SESSION_STOPPED)
    syllable_index = publisher.events.index(syllable)
    assert ticks == list(range(16))
    assert syllable.payload["scheduled_sample"] == 180_000
    assert syllable_index < stopped_index
    assert max(index for index, item in enumerate(timeline) if isinstance(item, tuple)) < timeline.index(RapEventType.SESSION_STOPPED)


def test_interleaved_polls_do_not_derive_or_execute_effects_concurrently() -> None:
    sink = InterleavedSnapshotSink()
    ticks: list[int] = []
    first_tick_entered = Event()
    release_first_tick = Event()

    def on_tick(tick: int) -> None:
        if tick == 0:
            first_tick_entered.set()
            assert release_first_tick.wait(timeout=1.0)
        ticks.append(tick)

    service = ManualPlaybackService(tempo=tempo(), sink=sink, publisher=None, on_tick=on_tick)
    service.prime(prepared_bar(bar=0))
    service.start()
    first = Thread(target=service.poll)
    first.start()
    assert first_tick_entered.wait(timeout=1.0)
    second = Thread(target=service.poll)
    second.start()
    try:
        assert not sink.second_snapshot_taken.wait(timeout=0.1)
        release_first_tick.set()
        first.join(timeout=1.0)
        second.join(timeout=1.0)

        assert not first.is_alive()
        assert not second.is_alive()
        assert ticks == [0, 1, 2]
    finally:
        release_first_tick.set()
        first.join(timeout=1.0)
        second.join(timeout=1.0)
        service.close()


def _capture_error(callback, errors: list[Exception]) -> None:
    try:
        callback()
    except Exception as error:
        errors.append(error)


def _prime_and_capture(
    service: RapPlaybackService,
    bar: PreparedRapBar,
    errors: list[Exception],
    completed: Event,
) -> None:
    try:
        service.prime(bar)
    except Exception as error:
        errors.append(error)
    finally:
        completed.set()


def _reset_and_capture(
    service: RapPlaybackService,
    errors: list[Exception],
    completed: Event,
) -> None:
    try:
        service.reset()
    except Exception as error:
        errors.append(error)
    finally:
        completed.set()


def _start_and_capture(
    service: RapPlaybackService,
    errors: list[Exception],
    requested: Event,
) -> None:
    requested.set()
    _capture_error(service.start, errors)


def _close_and_capture(
    service: RapPlaybackService,
    errors: list[Exception],
    completed: Event,
) -> None:
    _capture_error(service.close, errors)
    completed.set()
