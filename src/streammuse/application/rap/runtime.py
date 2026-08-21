"""Standalone monotonic tick runtime for the rap showcase."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from threading import Event, RLock
from types import MappingProxyType
from typing import Any, Callable, ClassVar

from streammuse.application.rap.audio_service import RapAudioController
from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher
from streammuse.application.rap.realtime import RollingRapController
from streammuse.domain.rap import PlaybackState, RapEventType
from streammuse.domain.timing import Tempo


class RapTickLoop:
    """Drive absolute musical ticks from monotonic deadlines without drift."""

    def __init__(
        self,
        tempo: Tempo,
        *,
        on_tick: Callable[[int], None],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._tempo = tempo
        self._on_tick = on_tick
        self._clock = clock
        self._sleep = sleep
        self._stop = Event()

    def run(self, max_ticks: int | None = None) -> None:
        start = self._clock()
        tick = 0
        while not self._stop.is_set() and (max_ticks is None or tick < max_ticks):
            target = start + self._tempo.tick_to_seconds(tick)
            remaining = target - self._clock()
            if remaining > 0:
                self._sleep(remaining)
            self._on_tick(tick)
            tick += 1

    def stop(self) -> None:
        self._stop.set()


@dataclass
class RapDemoDependencies:
    """Own the standalone demo lifecycle exactly once."""

    autostart: ClassVar[bool] = True

    tempo: Tempo
    controller: RollingRapController
    publisher: RapEventPublisher
    dispatcher: RapEventDispatcher
    tick_loop: RapTickLoop
    session_dir: Path
    repetition_window_bars: int = 4
    session_metadata: Mapping[str, Any] = field(default_factory=dict)
    recorder: Any | None = None
    projector: Any | None = None
    websocket_queue: Any | None = None
    configured_max_bars: int = 0
    _closed: bool = field(default=False, init=False)
    _close_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repetition_window_bars, int)
            or isinstance(self.repetition_window_bars, bool)
            or self.repetition_window_bars <= 0
        ):
            raise ValueError("repetition_window_bars must be a positive integer")
        if not isinstance(self.session_metadata, Mapping):
            raise ValueError("session_metadata must be a mapping")
        if (
            not isinstance(self.configured_max_bars, int)
            or isinstance(self.configured_max_bars, bool)
            or self.configured_max_bars < 0
        ):
            raise ValueError("configured_max_bars must be a nonnegative integer")
        self.session_metadata = _freeze_metadata(self.session_metadata)

    def start(self) -> None:
        """Run with the bar limit resolved during CLI assembly."""
        self.run(max_bars=self.configured_max_bars)

    def run(self, *, max_bars: int) -> None:
        payload = _thaw_metadata(self.session_metadata)
        payload.update(
            {
                "tempo_bpm": self.tempo.bpm,
                "ticks_per_beat": self.tempo.ticks_per_beat,
                "beats_per_bar": self.tempo.beats_per_bar,
                "max_bars": max_bars,
                "repetition_window_bars": self.repetition_window_bars,
            }
        )
        self.publisher.emit(RapEventType.SESSION_STARTED, payload=payload)
        max_ticks = None if max_bars == 0 else max_bars * self.tempo.ticks_per_bar
        try:
            self.controller.start()
            self.tick_loop.run(max_ticks=max_ticks)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            failures: list[tuple[str, BaseException]] = []

            def attempt(phase: str, action: Callable[[], None]) -> None:
                try:
                    action()
                except BaseException as exc:
                    failures.append((phase, exc))

            attempt("tick loop stop", self.tick_loop.stop)
            attempt("controller close", self.controller.close)
            attempt(
                "session stopped publication",
                lambda: self.publisher.emit(RapEventType.SESSION_STOPPED, payload={}),
            )
            attempt("dispatcher close", self.dispatcher.flush_and_close)
            if self.recorder is not None:
                attempt("recorder close", self.recorder.close)

            if failures:
                first_phase, first_error = failures[0]
                add_note = getattr(first_error, "add_note", None)
                if callable(add_note):
                    add_note(f"rap runtime teardown first failed during {first_phase}")
                    for phase, error in failures[1:]:
                        add_note(f"additional teardown failure during {phase}: {type(error).__name__}: {error}")
                raise first_error


@dataclass
class RapAudioDemoDependencies:
    """Own a restartable audio demo without changing text-runtime semantics."""

    autostart: ClassVar[bool] = False

    tempo: Tempo
    controller: RapAudioController
    coordinator: Any
    playback: Any
    publisher: RapEventPublisher
    dispatcher: RapEventDispatcher
    session_dir: Path
    session_metadata: Mapping[str, Any] = field(default_factory=dict)
    recorder: Any | None = None
    projector: Any | None = None
    websocket_queue: Any | None = None
    configured_max_bars: int = 0
    _closed: bool = field(default=False, init=False)
    _restart_requires_reset: bool = field(default=False, init=False)
    _lifecycle_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.configured_max_bars, int)
            or isinstance(self.configured_max_bars, bool)
            or self.configured_max_bars < 0
        ):
            raise ValueError("configured_max_bars must be a nonnegative integer")
        if not isinstance(self.session_metadata, Mapping):
            raise ValueError("session_metadata must be a mapping")
        self.session_metadata = _freeze_metadata(self.session_metadata)

    @property
    def control_state(self) -> PlaybackState:
        return self.playback.state

    @property
    def restart_requires_reset(self) -> bool:
        return self._restart_requires_reset

    def start(self) -> None:
        """Start audio playback and block until a requested or automatic stop."""

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("rap audio runtime is closed")
            if self._restart_requires_reset:
                raise RuntimeError("reset is required before restarting a completed finite scenario")
            if self.playback.state not in (PlaybackState.STOPPED, PlaybackState.PRIMING):
                return
            payload = _thaw_metadata(self.session_metadata)
            payload.update(
                {
                    "tempo_bpm": self.tempo.bpm,
                    "ticks_per_beat": self.tempo.ticks_per_beat,
                    "beats_per_bar": self.tempo.beats_per_bar,
                    "max_bars": self.configured_max_bars,
                    "playback_state": PlaybackState.PRIMING.value,
                }
            )
            self.publisher.emit(RapEventType.SESSION_STARTED, payload=payload)
            self.controller.start()
            if self.playback.state == PlaybackState.STOPPED:
                # A prior bar-quantized stop retains an immutable reserved
                # successor. Commit it here, then prime exactly the next
                # complete musical bar after stale queue data is cleared.
                self.controller.resume_audio(self.playback.next_start_bar)
                self.controller.resume_after_stop()
            if self.playback.state == PlaybackState.STOPPED:
                raise RuntimeError("audio controller did not prepare a playback bar")
            self.playback.start()

        maximum_tick = self.configured_max_bars * self.tempo.ticks_per_bar - 1
        while self.playback.state in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
            if self.configured_max_bars and self.playback.current_tick is not None:
                if self.playback.current_tick >= maximum_tick:
                    self.request_stop()
            self.playback.wait(timeout=0.01)

    def request_stop(self) -> None:
        """Request a stop that the playback service quantizes to the current bar."""

        with self._lifecycle_lock:
            self._request_stop_locked()

    def _request_stop_locked(self) -> None:
        if self._closed:
            return
        if self.playback.state not in (PlaybackState.PRIMING, PlaybackState.RUNNING):
            return
        successor_bar = self.playback.request_stop()
        if successor_bar is None:
            successor_bar = self.playback.stop_successor_bar
        scenario = getattr(self.controller, "scenario", None)
        terminal_bar_limit = getattr(self.controller, "terminal_bar_limit", None)
        if terminal_bar_limit is None and scenario is not None and not scenario.loop:
            terminal_bar_limit = scenario.total_bars
        terminal = terminal_bar_limit is not None and successor_bar >= terminal_bar_limit
        self.controller.request_stop(successor_bar=None if terminal else successor_bar)
        if terminal:
            self._restart_requires_reset = True

    def reset(self) -> None:
        """Discard stopped session state so the next start begins at bar zero."""

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("rap audio runtime is closed")
            if self.playback.state != PlaybackState.STOPPED:
                raise RuntimeError("rap audio runtime can reset only while stopped")
            quiesce = getattr(self.playback, "quiesce_for_reset", None)
            if callable(quiesce):
                quiesce()
            coordinator_epoch = self.controller.reset()
            if not isinstance(coordinator_epoch, int) or isinstance(coordinator_epoch, bool):
                raise RuntimeError("audio controller reset did not establish a coordinator epoch")
            self.playback.reset(coordinator_epoch=coordinator_epoch)
            self._restart_requires_reset = False

    def close(self) -> None:
        """Permanently close audio, planning, publication, and recording once."""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            failures: list[tuple[str, BaseException]] = []

            def attempt(phase: str, action: Callable[[], None]) -> None:
                try:
                    action()
                except BaseException as exc:
                    failures.append((phase, exc))

            attempt("playback close", self.playback.close)
            attempt("controller close", self.controller.close)
            if self.coordinator is not None:
                attempt("coordinator close", self.coordinator.close)
            attempt("dispatcher close", self.dispatcher.flush_and_close)
            if self.recorder is not None:
                attempt("recorder close", self.recorder.close)
            if failures:
                first_phase, first_error = failures[0]
                add_note = getattr(first_error, "add_note", None)
                if callable(add_note):
                    add_note(f"rap audio runtime teardown first failed during {first_phase}")
                raise first_error


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("session_metadata keys must be strings")
            frozen[key] = _freeze_metadata(item)
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise ValueError("session_metadata must contain JSON-compatible values")


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return value
