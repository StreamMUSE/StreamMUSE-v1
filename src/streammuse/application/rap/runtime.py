"""Standalone monotonic tick runtime for the rap showcase."""

from __future__ import annotations

import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from types import MappingProxyType
from typing import Any, Callable

from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher
from streammuse.application.rap.realtime import RollingRapController
from streammuse.domain.rap import RapEventType
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

    tempo: Tempo
    controller: RollingRapController
    publisher: RapEventPublisher
    dispatcher: RapEventDispatcher
    tick_loop: RapTickLoop
    session_dir: Path
    repetition_window_bars: int = 4
    session_metadata: Mapping[str, Any] = field(default_factory=dict)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repetition_window_bars, int)
            or isinstance(self.repetition_window_bars, bool)
            or self.repetition_window_bars <= 0
        ):
            raise ValueError("repetition_window_bars must be a positive integer")
        if not isinstance(self.session_metadata, Mapping):
            raise ValueError("session_metadata must be a mapping")
        self.session_metadata = _freeze_metadata(deepcopy(dict(self.session_metadata)))

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
        self.controller.start()
        max_ticks = None if max_bars == 0 else max_bars * self.tempo.ticks_per_bar
        try:
            self.tick_loop.run(max_ticks=max_ticks)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.tick_loop.stop()
        self.controller.close()
        self.publisher.emit(RapEventType.SESSION_STOPPED, payload={})
        self.dispatcher.flush_and_close()


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return deepcopy(value)
