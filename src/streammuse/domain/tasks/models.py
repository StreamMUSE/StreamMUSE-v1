"""Domain contracts for realtime and benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class TaskState:
    task_name: str
    turn_index: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    history: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class TaskTurn:
    task_name: str
    turn_id: int
    state: TaskState
    messages: list[ChatMessage]
    expected_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRefereeResult:
    is_valid: bool
    expected_output: str | None = None
    failure_reason: str = "NONE"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatModelResponse:
    text: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class LocalChatModel(Protocol):
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> ChatModelResponse:
        ...


class RealtimeTask(Protocol):
    name: str

    def initial_state(self) -> TaskState: ...

    def build_turn(self, state: TaskState) -> TaskTurn: ...

    def validate_response(self, state: TaskState, response_text: str) -> TaskRefereeResult: ...

    def advance_state(
        self,
        state: TaskState,
        referee_result: TaskRefereeResult,
        response_text: str,
    ) -> TaskState: ...
