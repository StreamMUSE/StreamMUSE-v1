"""Application services for generic realtime tasks."""

from streammuse.application.tasks.interactive_runtime import (
    InteractiveTaskRunResult,
    InteractiveTaskRuntime,
    InteractiveTaskRuntimeConfig,
    TimedPromptResult,
    StdTerminalIO,
    TerminalIO,
)
from streammuse.application.tasks.runtime import TaskRunResult, TaskRuntime, TaskRuntimeConfig

__all__ = [
    "InteractiveTaskRunResult",
    "InteractiveTaskRuntime",
    "InteractiveTaskRuntimeConfig",
    "TimedPromptResult",
    "StdTerminalIO",
    "TaskRunResult",
    "TaskRuntime",
    "TaskRuntimeConfig",
    "TerminalIO",
]
