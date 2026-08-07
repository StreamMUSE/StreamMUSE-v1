"""Standalone monotonic tick runtime for the rap showcase."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable

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
    _closed: bool = field(default=False, init=False)

    def run(self, *, max_bars: int) -> None:
        self.publisher.emit(
            RapEventType.SESSION_STARTED,
            payload={"tempo_bpm": self.tempo.bpm, "max_bars": max_bars},
        )
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
