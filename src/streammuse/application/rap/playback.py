"""Lifecycle control and non-callback observation for rap audio playback."""

from __future__ import annotations

from threading import Event, Lock, RLock, Thread, current_thread, get_ident
import time
from typing import Callable

from streammuse.application.rap.audio_rendering import bar_start_frame
from streammuse.application.rap.audio_service import RapAudioSink
from streammuse.application.rap.monitoring import RapEventPublisher
from streammuse.domain.rap import (
    AudioPlaybackNotice,
    AudioPlaybackNoticeKind,
    AudioPlaybackSnapshot,
    PlaybackState,
    PreparedRapBar,
    RapEventType,
)
from streammuse.domain.timing import Tempo


_OBSERVER_INTERVAL_SECONDS = 0.005
_NOTICE_EVENTS = {
    AudioPlaybackNoticeKind.BAR_STARTED: RapEventType.BAR_PLAYBACK_STARTED,
    AudioPlaybackNoticeKind.BAR_COMPLETED: RapEventType.BAR_PLAYBACK_COMPLETED,
    AudioPlaybackNoticeKind.UNDERRUN: RapEventType.AUDIO_UNDERRUN,
    AudioPlaybackNoticeKind.DEVICE_FAILED: RapEventType.AUDIO_DEVICE_FAILED,
}


class RapPlaybackService:
    """Own playback lifecycle while observing the sink away from its callback."""

    def __init__(
        self,
        *,
        tempo: Tempo,
        sink: RapAudioSink,
        publisher: RapEventPublisher | None,
        on_tick: Callable[[int], None],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tempo = tempo
        self._sink = sink
        self._publisher = publisher
        self._on_tick = on_tick
        self._monotonic = monotonic
        self._lock = RLock()
        self._poll_lock = Lock()
        self._dispatch_owner: int | None = None
        self._deferred_joins: list[Thread] = []
        self._sink_closed = False
        self._state = PlaybackState.STOPPED
        self._prepared: dict[int, PreparedRapBar] = {}
        self._current_tick: int | None = None
        self._emitted_syllables: set[tuple[int, int]] = set()
        self._next_start_bar = 0
        self._sample_origin_frame = 0
        self._epoch = 0
        self._observer_stop = Event()
        self._observer: Thread | None = None

    @property
    def state(self) -> PlaybackState:
        with self._lock:
            return self._state

    @property
    def current_tick(self) -> int | None:
        with self._lock:
            return self._current_tick

    def prime(self, bar: PreparedRapBar) -> None:
        with self._lock:
            self._require_state(PlaybackState.STOPPED)
            self._prime_locked(bar)

    def start(self) -> None:
        with self._lock:
            self._require_state(PlaybackState.PRIMING)
            if not self._prepared:
                raise RuntimeError("playback requires at least one queued bar")
            epoch = self._epoch

        self._sink.start()

        with self._lock:
            if epoch != self._epoch or self._state == PlaybackState.CLOSED:
                return
            self._state = PlaybackState.RUNNING
            self._observer_stop = Event()
            observer = Thread(
                target=self._observe,
                args=(epoch, self._observer_stop),
                name="streammuse-rap-playback-observer",
                daemon=True,
            )
            self._observer = observer
        self._emit(RapEventType.SESSION_STARTED, payload={"playback_state": PlaybackState.RUNNING.value})
        with self._lock:
            if epoch != self._epoch or self._state == PlaybackState.CLOSED:
                return
            observer.start()

    def enqueue(self, bar: PreparedRapBar) -> None:
        with self._lock:
            if self._state == PlaybackState.STOPPED:
                self._prime_locked(bar)
                return
            if self._state not in (PlaybackState.PRIMING, PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
                raise RuntimeError(f"cannot enqueue while playback is {self._state.value}")
            self._enqueue_locked(bar)

    def request_stop(self) -> None:
        reset_sink = False
        with self._lock:
            if self._state in (PlaybackState.STOPPED, PlaybackState.CLOSED, PlaybackState.STOP_REQUESTED):
                return
            if self._state == PlaybackState.PRIMING:
                self._state = PlaybackState.STOPPED
                self._stop_observer_locked()
                self._epoch += 1
                self._prepared.clear()
                self._current_tick = None
                self._emitted_syllables.clear()
                self._next_start_bar = 0
                self._sample_origin_frame = 0
                reset_sink = True
                emit = True
            else:
                self._state = PlaybackState.STOP_REQUESTED
                self._sink.request_stop_after_bar()
                emit = True
        if reset_sink:
            self._sink.reset()
        if emit:
            self._emit(RapEventType.STOP_REQUESTED, payload={"playback_state": PlaybackState.STOP_REQUESTED.value})

    def reset(self) -> None:
        with self._lock:
            self._require_state(PlaybackState.STOPPED)
            self._epoch += 1
            observer = self._stop_observer_locked()
            self._prepared.clear()
            self._current_tick = None
            self._emitted_syllables.clear()
            self._next_start_bar = 0
            self._sample_origin_frame = 0
        self._join(observer)
        self._sink.reset()
        self._emit(RapEventType.SESSION_RESET, payload={"playback_state": PlaybackState.STOPPED.value})

    def wait(self, timeout: float | None = None) -> None:
        with self._lock:
            observer = self._observer
        self._join(observer, timeout)

    def close(self) -> None:
        with self._lock:
            if self._state != PlaybackState.CLOSED:
                self._epoch += 1
                self._state = PlaybackState.CLOSED
                self._prepared.clear()
                self._emitted_syllables.clear()
            observer = self._stop_observer_locked()
            defer_join = self._dispatch_owner == get_ident()
            if defer_join and observer is not None:
                self._deferred_joins.append(observer)
            close_sink = not self._sink_closed
            self._sink_closed = True
        if not defer_join:
            self._join(observer)
        if close_sink:
            self._sink.close()

    def poll(self) -> None:
        """Observe current sink state explicitly for deterministic callers and tests."""
        try:
            with self._poll_lock:
                with self._lock:
                    if self._state in (PlaybackState.STOPPED, PlaybackState.CLOSED, PlaybackState.PRIMING):
                        return
                    epoch = self._epoch

                snapshot = self._sink.snapshot()
                notices = self._sink.drain_notices()
                observed_at = self._monotonic()

                with self._lock:
                    if epoch != self._epoch or self._state == PlaybackState.CLOSED:
                        return
                    effects = self._effects_locked(snapshot, notices, observed_at)

                self._dispatch_effects(epoch, effects)
        finally:
            self._join_deferred_observers()

    def _observe(self, epoch: int, stop: Event) -> None:
        while not stop.is_set():
            self.poll()
            with self._lock:
                active = epoch == self._epoch and self._state in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED)
            if not active:
                return
            stop.wait(_OBSERVER_INTERVAL_SECONDS)

    def _effects_locked(
        self,
        snapshot: AudioPlaybackSnapshot,
        notices: tuple[AudioPlaybackNotice, ...],
        observed_at: float,
    ) -> list[Callable[[], None]]:
        effects: list[Callable[[], None]] = []
        completed_bars: list[int] = []
        observed_frame = self._sample_origin_frame + snapshot.absolute_frame
        for notice in notices:
            if notice.kind == AudioPlaybackNoticeKind.BAR_COMPLETED and notice.bar is not None:
                completed_bars.append(notice.bar)
                prepared = self._prepared.get(notice.bar)
                if prepared is not None:
                    bar_end = bar_start_frame(prepared.bar, self._tempo, prepared.audio.format) + prepared.audio.frame_count
                    observed_frame = max(observed_frame, bar_end)
            observed_frame = max(observed_frame, self._sample_origin_frame + notice.absolute_frame)
            event_type = _NOTICE_EVENTS.get(notice.kind)
            if event_type is not None:
                absolute_frame = self._sample_origin_frame + notice.absolute_frame
                effects.append(
                    lambda notice=notice, event_type=event_type, absolute_frame=absolute_frame: self._emit_notice(
                        event_type, notice, absolute_frame
                    )
                )

        stopping = snapshot.state == PlaybackState.STOPPED or any(
            notice.kind == AudioPlaybackNoticeKind.STOPPED for notice in notices
        )
        final_bar = max(completed_bars) if stopping and completed_bars else None
        effects.extend(self._sample_effects_locked(observed_frame, observed_at, final_bar=final_bar))
        for bar in completed_bars:
            self._prepared.pop(bar, None)
            self._next_start_bar = max(self._next_start_bar, bar + 1)
        if stopping:
            self._transition_to_stopped_locked(effects)
        return effects

    def _sample_effects_locked(
        self,
        observed_frame: int,
        observed_at: float,
        *,
        final_bar: int | None,
    ) -> list[Callable[[], None]]:
        effects: list[Callable[[], None]] = []
        sample_rate = self._sample_rate()
        next_tick = 0 if self._current_tick is None else self._current_tick + 1
        final_tick = (final_bar + 1) * self._tempo.ticks_per_bar - 1 if final_bar is not None else None
        while self._tick_sample(next_tick) <= observed_frame and (final_tick is None or next_tick <= final_tick):
            tick = next_tick
            self._current_tick = tick
            effects.append(lambda tick=tick: self._on_tick(tick))
            next_tick += 1

        for bar in tuple(self._prepared.values()):
            if final_bar is not None and bar.bar > final_bar:
                continue
            for diagnostic in bar.diagnostics:
                key = (bar.bar, diagnostic.slot_index)
                scheduled_sample = bar_start_frame(bar.bar, self._tempo, bar.audio.format) + diagnostic.target_sample
                if key in self._emitted_syllables or scheduled_sample > observed_frame:
                    continue
                scheduled = next((item for item in bar.scheduled if item.slot.slot_index == diagnostic.slot_index), None)
                if scheduled is None:
                    continue
                self._emitted_syllables.add(key)
                payload = {
                    "scheduled_sample": scheduled_sample,
                    "software_error_samples": 0,
                    "observation_delay_ms": (observed_frame - scheduled_sample) / sample_rate * 1000.0,
                    "word": scheduled.syllable.word,
                    "label": scheduled.syllable.label,
                    "stress": scheduled.syllable.stress,
                    "beat": scheduled.slot.beat,
                    "subdivision": scheduled.slot.tick_in_beat,
                }
                effects.append(
                    lambda bar=bar.bar, tick=scheduled.slot.tick, payload=payload: self._emit(
                        RapEventType.SYLLABLE_EMITTED,
                        bar=bar,
                        tick=tick,
                        payload=payload,
                    )
                )
        return effects

    def _transition_to_stopped_locked(self, effects: list[Callable[[], None]]) -> None:
        if self._state == PlaybackState.STOPPED:
            return
        self._state = PlaybackState.STOPPED
        self._stop_observer_locked()
        effects.append(lambda: self._emit(RapEventType.SESSION_STOPPED, payload={"playback_state": PlaybackState.STOPPED.value}))

    def _enqueue_locked(self, bar: PreparedRapBar) -> None:
        self._sink.enqueue(bar)
        self._prepared[bar.bar] = bar

    def _prime_locked(self, bar: PreparedRapBar) -> None:
        if bar.bar != self._next_start_bar:
            raise ValueError(f"prime requires prepared bar {self._next_start_bar}")
        if bar.bar > 0:
            # Task 5 intentionally retains bars queued after a bar-quantized
            # stop. A continuation must start from one known complete bar.
            self._sink.reset()
            self._prepared.clear()
            self._sample_origin_frame = bar_start_frame(bar.bar, self._tempo, bar.audio.format)
            self._current_tick = bar.bar * self._tempo.ticks_per_bar - 1
        self._enqueue_locked(bar)
        self._state = PlaybackState.PRIMING

    def _dispatch_effects(self, epoch: int, effects: list[Callable[[], None]]) -> None:
        owner = get_ident()
        with self._lock:
            self._dispatch_owner = owner
        try:
            for effect in effects:
                with self._lock:
                    if epoch != self._epoch or self._state == PlaybackState.CLOSED:
                        break
                effect()
        finally:
            with self._lock:
                if self._dispatch_owner == owner:
                    self._dispatch_owner = None

    def _join_deferred_observers(self) -> None:
        with self._lock:
            observers = tuple(self._deferred_joins)
            self._deferred_joins.clear()
        for observer in observers:
            self._join(observer)

    def _emit_notice(self, event_type: RapEventType, notice: AudioPlaybackNotice, absolute_frame: int) -> None:
        self._emit(
            event_type,
            bar=notice.bar,
            tick=self._current_tick,
            payload={
                "absolute_frame": absolute_frame,
                "queue_depth": notice.queue_depth,
                "message": notice.message,
            },
        )

    def _emit(
        self,
        event_type: RapEventType,
        *,
        bar: int | None = None,
        tick: int | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        if self._publisher is not None:
            self._publisher.emit(event_type, bar=bar, tick=tick, payload=payload)

    def _tick_sample(self, tick: int) -> int:
        return round(self._tempo.tick_to_seconds(tick) * self._sample_rate())

    def _sample_rate(self) -> int:
        if self._prepared:
            return next(iter(self._prepared.values())).audio.format.sample_rate_hz
        return 48_000

    def _stop_observer_locked(self) -> Thread | None:
        self._observer_stop.set()
        return self._observer

    def _require_state(self, expected: PlaybackState) -> None:
        if self._state != expected:
            raise RuntimeError(f"playback must be {expected.value}, got {self._state.value}")

    @staticmethod
    def _join(observer: Thread | None, timeout: float | None = None) -> None:
        if observer is not None and observer.ident is not None and observer is not current_thread():
            observer.join(timeout)
