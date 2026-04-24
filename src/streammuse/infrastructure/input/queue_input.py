"""Queue-backed InputSource for real-time external pushers (e.g. web UI)."""

from __future__ import annotations

import queue
from typing import Iterator

from streammuse.domain.musical import MusicalEvent


class QueueInput:
    """
    InputSource that yields events pushed into a thread-safe queue.

    Unlike ListInput (static, replay-style), QueueInput is a live feed — the
    web UI's WebSocket handler pushes MusicalEvents via `push()` and the
    service thread consumes them via `read_events()`.
    """

    def __init__(self, *, poll_timeout_s: float = 0.05) -> None:
        self._q: "queue.Queue[MusicalEvent]" = queue.Queue()
        self._poll_timeout_s = float(poll_timeout_s)
        self._closed = False

    def push(self, event: MusicalEvent) -> None:
        if self._closed:
            return
        self._q.put(event)

    def read_events(self) -> Iterator[MusicalEvent]:
        while not self._closed:
            try:
                yield self._q.get(timeout=self._poll_timeout_s)
            except queue.Empty:
                continue

    def close(self) -> None:
        self._closed = True
