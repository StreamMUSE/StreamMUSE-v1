"""Standalone monotonic tick runtime for the rap showcase."""

from __future__ import annotations

import logging
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
    settings_service: Any | None = None
    controller_factory: Callable[[Any, int], Any] | None = None
    _closed: bool = field(default=False, init=False)
    _restart_requires_reset: bool = field(default=False, init=False)
    _starting: bool = field(default=False, init=False)
    _device_starting: bool = field(default=False, init=False)
    _cancel_start: bool = field(default=False, init=False)
    _settings_updating: bool = field(default=False, init=False)
    _settings_groups: set[str] = field(default_factory=set, init=False, repr=False)
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
        if self._starting:
            return PlaybackState.STOP_REQUESTED if self._cancel_start else PlaybackState.PRIMING
        return self.playback.state

    @property
    def restart_requires_reset(self) -> bool:
        return self._restart_requires_reset

    def settings_state(self, group: str | None = None) -> dict[str, Any]:
        if self.settings_service is None:
            return {"supported": False, "values": {}, "capabilities": {}, "devices": []}
        if group is None:
            return self.settings_service.snapshot()
        from streammuse.application.rap.settings import SETTINGS_GROUPS
        if group not in SETTINGS_GROUPS:
            raise ValueError("unknown settings group")
        return self.settings_service.snapshot(group=group, discover_devices=group == "session")

    def record_audio_device(self, name: str) -> None:
        """Keep resumed-session metadata consistent with a successful device switch."""
        with self._lifecycle_lock:
            metadata = _thaw_metadata(self.session_metadata)
            metadata.setdefault("audio", {})["audio_device"] = name
            self.session_metadata = _freeze_metadata(metadata)
            self.publisher.emit(RapEventType.AUDIO_SETTINGS_CHANGED, payload={"audio_device": name})

    def update_settings(self, payload: object, *, group: str | None = None) -> dict[str, Any]:
        if self.settings_service is None:
            raise NotImplementedError("runtime does not support settings changes")
        from streammuse.application.rap.settings import settings_group
        selected_group = settings_group(payload, group, allow_combined=self.control_state == PlaybackState.STOPPED)
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("rap audio runtime is closed")
            independent_live_update = (
                self.control_state == PlaybackState.RUNNING
                and selected_group != "session" and "session" not in self._settings_groups
                and selected_group not in self._settings_groups
            )
            if self._starting or (self._settings_updating and not independent_live_update):
                raise RuntimeError("wait for startup or the pending settings update")
            self._settings_groups.add(selected_group)
            self._settings_updating = True
        try:
            # Capability discovery and tempo acknowledgement can block on I/O.
            # Stop must remain available while reset/start are fenced out.
            return (self.settings_service.update(payload) if group is None
                    else self.settings_service.update(payload, group=group))
        finally:
            with self._lifecycle_lock:
                self._settings_groups.discard(selected_group)
                self._settings_updating = bool(self._settings_groups)

    def interaction_state(self) -> dict[str, Any]:
        method = getattr(self.controller, "interaction_state", None)
        result = dict(method()) if callable(method) else {"supported": False, "mode": "continuous"}
        result.setdefault("supported", callable(method))
        result["state"] = self.control_state.value
        result["current_tick"] = self.playback.current_tick
        return result

    def arm_capture(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            if self.control_state != PlaybackState.RUNNING:
                raise RuntimeError("start alternating playback before arming capture")
            method = getattr(self.controller, "arm_capture", None)
            if not callable(method):
                raise NotImplementedError("capture is available only in alternating mode")
            return method()

    def record_tap(self, **payload) -> dict[str, Any]:
        with self._lifecycle_lock:
            if self.control_state != PlaybackState.RUNNING:
                raise RuntimeError("playback is not running")
            method = getattr(self.controller, "record_tap", None)
            if not callable(method):
                raise NotImplementedError("capture is available only in alternating mode")
            return method(**payload)

    def replace_controller(self, factory: Callable[[int], Any]) -> None:
        """Switch stopped interaction modes in a fresh monotonically versioned epoch."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("rap audio runtime is closed")
            if self.control_state != PlaybackState.STOPPED:
                raise RuntimeError("interaction mode changes require stopped playback")
            if len(self._settings_groups) > 1:
                raise RuntimeError("wait for the pending local settings update before rebuilding playback")
            self._reset_locked(factory)

    def start(self) -> None:
        """Start audio playback and block until a requested or automatic stop."""

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("rap audio runtime is closed")
            if self._starting:
                return
            if self._settings_updating:
                raise RuntimeError("wait for the pending settings update before starting")
            if self._restart_requires_reset:
                self.reset()
            if self.playback.state not in (PlaybackState.STOPPED, PlaybackState.PRIMING):
                return
            self._starting = True
            self._cancel_start = False
        try:
            payload = _thaw_metadata(self.session_metadata)
            timeline = getattr(self.controller, "timeline", None)
            current_tempo = (
                timeline.snapshot(self.playback.next_start_bar, 48_000).bpm
                if timeline is not None else self.tempo.bpm
            )
            payload.update(
                {
                    "tempo_bpm": current_tempo,
                    "ticks_per_beat": self.tempo.ticks_per_beat,
                    "beats_per_bar": self.tempo.beats_per_bar,
                    "max_bars": self.configured_max_bars,
                    "playback_state": PlaybackState.PRIMING.value,
                }
            )
            if self.settings_service is not None:
                settings = self.settings_service.values
                payload["interaction_mode"] = settings.mode
                payload["response_bars"] = settings.response_bars
                audio = payload.setdefault("audio", {})
                audio.update(interaction_mode=settings.mode, response_bars=settings.response_bars,
                             audio_device=settings.output_device)
                audio.setdefault("mix", {}).update(
                    drum_gain=settings.drum_gain, vocal_gain=settings.vocal_gain,
                    chord_loop=settings.chord_loop, chord_gain=settings.chord_gain,
                )
                if self.controller_factory is not None or "local_warp" in audio:
                    audio.setdefault("local_warp", {})["policy"] = settings.alignment_policy
            self.publisher.emit(RapEventType.SESSION_STARTED, payload=payload)
            self.controller.start()
            with self._lifecycle_lock:
                if self._cancel_start or self._closed:
                    if not self._closed:
                        if self.playback.state == PlaybackState.PRIMING:
                            self.playback.request_stop()
                        self._reset_locked(self._settings_factory())
                    return
                if self.playback.state == PlaybackState.STOPPED:
                    # Manual Stop retains the next immutable bar for resume.
                    self.controller.resume_audio(self.playback.next_start_bar)
                    self.controller.resume_after_stop()
                if self.playback.state == PlaybackState.STOPPED:
                    raise RuntimeError("audio controller did not prepare a playback bar")
                self._device_starting = True
            # Device open/start may block while holding playback's own gate.
            # Keep the runtime gate available to record Stop or permanent Close.
            self.playback.start()
            with self._lifecycle_lock:
                self._device_starting = False
                if self._closed:
                    return
                canceled_device_start = self._cancel_start
                if canceled_device_start:
                    self._restart_requires_reset = True
                    if self.playback.state in (PlaybackState.PRIMING, PlaybackState.RUNNING):
                        self.playback.request_stop()
                    self.controller.request_stop(successor_bar=None)
            if canceled_device_start:
                while self.playback.state == PlaybackState.STOP_REQUESTED:
                    self.playback.wait(timeout=0.01)
                with self._lifecycle_lock:
                    if not self._closed:
                        self._reset_locked(self._settings_factory())
                return
        except BaseException:
            with self._lifecycle_lock:
                if not self._closed:
                    self._restart_requires_reset = True
                    try:
                        if self.playback.state == PlaybackState.PRIMING:
                            self.playback.request_stop()
                        if self.playback.state == PlaybackState.STOPPED:
                            self._reset_locked(self._settings_factory())
                    except Exception:
                        logging.getLogger(__name__).exception("failed to reset after audio startup failure")
            raise
        finally:
            with self._lifecycle_lock:
                self._device_starting = False
                self._starting = False

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
        if self._device_starting:
            self._cancel_start = True
            return
        if self._starting and self.playback.state in (PlaybackState.STOPPED, PlaybackState.PRIMING):
            self._cancel_start = True
            self.controller.request_stop(successor_bar=None)
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
        if self.configured_max_bars:
            terminal_bar_limit = (
                self.configured_max_bars if terminal_bar_limit is None
                else min(terminal_bar_limit, self.configured_max_bars)
            )
        terminal = terminal_bar_limit is not None and successor_bar >= terminal_bar_limit
        self.controller.request_stop(successor_bar=None if terminal else successor_bar)
        if terminal:
            self._restart_requires_reset = True

    def reset(self) -> None:
        """Discard stopped session state so the next start begins at bar zero."""

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("rap audio runtime is closed")
            if self._starting:
                raise RuntimeError("rap audio runtime is still preparing; stop it before reset")
            if self._settings_updating:
                raise RuntimeError("wait for the pending settings update before reset")
            if self.playback.state != PlaybackState.STOPPED:
                raise RuntimeError("rap audio runtime can reset only while stopped")
            self._reset_locked(self._settings_factory())

    def _settings_factory(self) -> Callable[[int], Any] | None:
        if self.controller_factory is None or self.settings_service is None:
            return None
        values = self.settings_service.values
        return lambda epoch: self.controller_factory(values, epoch)

    def _install_controller_timeline(self, controller: Any) -> None:
        timeline = getattr(controller, "timeline", None)
        if timeline is not None:
            self.playback.set_timeline(timeline)

    def _reset_locked(self, factory: Callable[[int], Any] | None = None) -> None:
        # Playback is stopped: serialize teardown against starts, not active audio.
        self._restart_requires_reset = True
        quiesce = getattr(self.playback, "quiesce_for_reset", None)
        if callable(quiesce):
            quiesce()
        old = self.controller
        epoch = old.reset()
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise RuntimeError("audio controller reset did not establish a coordinator epoch")
        if self.settings_service is not None:
            self.settings_service.reset_acceptance(epoch, applied=False)
        self.playback.reset(coordinator_epoch=epoch)
        self._install_controller_timeline(old)
        if factory is not None:
            replacement = None
            try:
                replacement = factory(epoch)
                self._install_controller_timeline(replacement)
            except BaseException:
                # Existing factories may install their timeline before returning.
                self._install_controller_timeline(old)
                if replacement is not None:
                    try:
                        replacement.close()
                    except Exception:
                        logging.getLogger(__name__).exception("failed to close rejected audio controller")
                raise
            self.controller = replacement
            try:
                old.close()
            except Exception:
                # The replacement is already committed. Rejecting its settings
                # now would falsely report that the old values are still active.
                logging.getLogger(__name__).exception("failed to close retired audio controller")
        if self.settings_service is not None:
            self.settings_service.reset_acceptance(epoch)
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
