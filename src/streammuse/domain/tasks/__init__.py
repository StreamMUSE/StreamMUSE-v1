"""Generic realtime task contracts and reference tasks."""

from streammuse.domain.tasks.models import (
    ChatMessage,
    ChatModelResponse,
    LocalChatModel,
    RealtimeTask,
    TaskRefereeResult,
    TaskState,
    TaskTurn,
)
from streammuse.domain.tasks.animal_naming import AnimalNamingTask
from streammuse.domain.tasks.zip_zap_zop import ZipZapZopTask

__all__ = [
    "AnimalNamingTask",
    "ChatMessage",
    "ChatModelResponse",
    "LocalChatModel",
    "RealtimeTask",
    "TaskRefereeResult",
    "TaskState",
    "TaskTurn",
    "ZipZapZopTask",
]
