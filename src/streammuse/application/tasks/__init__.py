"""Application services for generic realtime tasks."""

from streammuse.application.tasks.interactive_runtime import (
    InteractiveTaskRunResult,
    InteractiveTaskRuntime,
    InteractiveTaskRuntimeConfig,
    StdTerminalIO,
    TerminalIO,
)
from streammuse.application.tasks.runtime import TaskRunResult, TaskRuntime, TaskRuntimeConfig

__all__ = [
    "InteractiveTaskRunResult",
    "InteractiveTaskRuntime",
    "InteractiveTaskRuntimeConfig",
    "StdTerminalIO",
    "TaskRunResult",
    "TaskRuntime",
    "TaskRuntimeConfig",
    "TerminalIO",
]
