"""Generic realtime task contracts and reference tasks."""

from streammuse.domain.tasks.models import (
    ChatMessage,
    ChatModelResponse,
    InteractiveActor,
    InteractiveTask,
    InteractiveTurnRecord,
    LocalChatModel,
    RealtimeTask,
    TaskRefereeResult,
    TaskState,
    TaskTurn,
)
from streammuse.domain.tasks.zip_zap_zop import ZipZapZopTask

__all__ = [
    "ChatMessage",
    "ChatModelResponse",
    "InteractiveActor",
    "InteractiveTask",
    "InteractiveTurnRecord",
    "LocalChatModel",
    "RealtimeTask",
    "TaskRefereeResult",
    "TaskState",
    "TaskTurn",
    "ZipZapZopTask",
]
