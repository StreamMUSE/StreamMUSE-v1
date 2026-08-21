"""Thread-safe snapshot fan-out for task WebSocket observers."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Callable

from streammuse.domain.tasks import TaskViewEvent
from streammuse.infrastructure.task_view.snapshot import (
    TaskViewSnapshot,
    reduce_task_view_snapshot,
)


@dataclass(frozen=True)
class TaskViewSubscription:
    id: str
    queue: queue.Queue[dict[str, object]]


class QueueTaskEventSink:
    """Reduce events and fan out complete, bounded snapshots to each viewer."""

    def __init__(
        self,
        *,
        session_id: str,
        task: str,
        capacity: int = 256,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = int(capacity)
        self._monotonic = monotonic or time.monotonic
        self._lock = threading.RLock()
        self._snapshot = TaskViewSnapshot(session_id=session_id, task=task)
        self._subscriptions: dict[str, TaskViewSubscription] = {}
        self._closed = False

    @property
    def snapshot(self) -> TaskViewSnapshot:
        with self._lock:
            return self._snapshot

    def subscribe(self) -> tuple[TaskViewSubscription, dict[str, object]]:
        with self._lock:
            subscription = TaskViewSubscription(
                id=uuid.uuid4().hex,
                queue=queue.Queue(maxsize=self._capacity),
            )
            self._subscriptions[subscription.id] = subscription
            initial = self._envelope_locked()
            return subscription, initial

    def unsubscribe(self, subscription: TaskViewSubscription) -> None:
        with self._lock:
            self._subscriptions.pop(subscription.id, None)

    def emit(self, event: TaskViewEvent) -> None:
        with self._lock:
            if self._closed:
                return
            if event.type == "turn_attempt_started":
                event = replace(
                    event,
                    payload={
                        **event.payload,
                        "started_server_ms": self._monotonic() * 1000.0,
                    },
                )
            reduced = reduce_task_view_snapshot(self._snapshot, event)
            if reduced is self._snapshot:
                return
            self._snapshot = reduced
            self._broadcast_locked()

    def update_status(self, status: str) -> None:
        with self._lock:
            if self._closed:
                return
            self._snapshot = replace(self._snapshot, status=str(status))
            self._broadcast_locked()

    def drain(
        self,
        subscription: TaskViewSubscription,
        *,
        limit: int = 32,
    ) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for _ in range(max(1, int(limit))):
            try:
                items.append(subscription.queue.get_nowait())
            except queue.Empty:
                break
        return items

    def has_pending(self) -> bool:
        with self._lock:
            return any(not sub.queue.empty() for sub in self._subscriptions.values())

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._subscriptions.clear()

    def _broadcast_locked(self) -> None:
        envelope = self._envelope_locked()
        for subscription in tuple(self._subscriptions.values()):
            try:
                subscription.queue.put_nowait(envelope)
                continue
            except queue.Full:
                pass
            dropped = 0
            while True:
                try:
                    subscription.queue.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            self._snapshot = replace(
                self._snapshot,
                dropped_event_count=self._snapshot.dropped_event_count + dropped,
            )
            subscription.queue.put_nowait(self._envelope_locked())

    def _envelope_locked(self) -> dict[str, object]:
        return self._snapshot.envelope(now_ms=self._monotonic() * 1000.0)
