"""Nonblocking publication and ordered dispatch of rap showcase events."""

from __future__ import annotations

from datetime import datetime, timezone
from queue import SimpleQueue
from threading import Lock, Thread
from time import monotonic_ns
from typing import Any, Callable

from streammuse.domain.rap import RapEvent, RapEventType


_SENTINEL = object()


class RapEventPublisher:
    """Assign event identity once, then enqueue without presentation I/O."""

    def __init__(
        self,
        session_id: str,
        *,
        utc_now: Callable[[], str] | None = None,
        monotonic_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        self.session_id = session_id
        self.queue: SimpleQueue[RapEvent | object] = SimpleQueue()
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc).isoformat())
        self._monotonic_ns = monotonic_ns
        self._lock = Lock()
        self._sequence = 0

    def emit(
        self,
        event_type: RapEventType,
        *,
        bar: int | None = None,
        tick: int | None = None,
        request_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RapEvent:
        with self._lock:
            self._sequence += 1
            event = RapEvent(
                session_id=self.session_id,
                sequence=self._sequence,
                event_type=event_type,
                utc_time=self._utc_now(),
                monotonic_ns=self._monotonic_ns(),
                bar=bar,
                tick=tick,
                request_id=request_id,
                payload=dict(payload or {}),
            )
        self.queue.put(event)
        return event


class RapEventDispatcher:
    """Fan one queue out to terminal-compatible sinks in sequence order."""

    def __init__(self, queue: SimpleQueue[RapEvent | object], *, sinks: tuple[Callable[[RapEvent], None], ...]) -> None:
        self._queue = queue
        self._sinks = list(sinks)
        self._thread = Thread(target=self._run, name="streammuse-rap-events", daemon=True)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def flush_and_close(self) -> None:
        if not self._started:
            return
        self._queue.put(_SENTINEL)
        self._thread.join()
        self._started = False

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            assert isinstance(item, RapEvent)
            active = []
            for sink in self._sinks:
                try:
                    sink(item)
                except Exception:
                    continue
                active.append(sink)
            self._sinks = active
