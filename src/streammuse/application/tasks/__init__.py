"""Application services for generic realtime tasks."""

from streammuse.application.tasks.interactive_runtime import (
    InteractiveTaskRunResult,
    InteractiveTaskRuntime,
    InteractiveTaskRuntimeConfig,
)
from streammuse.application.tasks.human_input import (
    HumanInputConfig,
    StdTerminalIO,
    TerminalHumanResponseSource,
    TerminalIO,
    TimedPromptResult,
    VoiceInputConfig,
)
from streammuse.application.tasks.runtime import TaskRunResult, TaskRuntime, TaskRuntimeConfig
from streammuse.application.tasks.speech_output import SpeechOutputConfig, SilentSpeechOutput
from streammuse.application.tasks.task_events import NullTaskEventSink, TaskWebConfig

__all__ = [
    "HumanInputConfig",
    "InteractiveTaskRunResult",
    "InteractiveTaskRuntime",
    "InteractiveTaskRuntimeConfig",
    "NullTaskEventSink",
    "TimedPromptResult",
    "StdTerminalIO",
    "SilentSpeechOutput",
    "SpeechOutputConfig",
    "TerminalHumanResponseSource",
    "TaskRunResult",
    "TaskRuntime",
    "TaskRuntimeConfig",
    "TaskWebConfig",
    "TerminalIO",
    "VoiceInputConfig",
]
