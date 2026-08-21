"""Read-only Web observer infrastructure for interactive tasks."""

from streammuse.infrastructure.task_view.snapshot import (
    TaskViewSnapshot,
    reduce_task_view_snapshot,
)
from streammuse.infrastructure.task_view.websocket import (
    QueueTaskEventSink,
    TaskViewSubscription,
)

__all__ = [
    "QueueTaskEventSink",
    "TaskViewSnapshot",
    "TaskViewSubscription",
    "reduce_task_view_snapshot",
]
