"""Domain contracts for realtime, benchmark, and interactive tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


ChatMessage = dict[str, str]
InteractiveActor = Literal["human", "llm"]
DeadlineMode = Literal["menu", "soft", "hard", "challenge"]
HumanInputMode = Literal["terminal", "voice"]
HumanResponseStatus = Literal[
    "ok",
    "no_speech",
    "empty_transcript",
    "rejected_transcript",
]
SpokenResponseParseStatus = Literal["ok", "empty", "unrecognized"]


@dataclass(frozen=True)
class SpeechContext:
    initial_prompt: str | None = None
    hotwords: tuple[str, ...] = ()


@dataclass(frozen=True)
class HumanResponseRequest:
    turn_id: int
    prompt: str
    timeout_s: float | None
    speech_context: SpeechContext | None = None


@dataclass(frozen=True)
class HumanResponse:
    text: str
    status: HumanResponseStatus = "ok"
    deadline_expired: bool = False
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class HumanResponseSource(Protocol):
    mode: HumanInputMode

    def start(self) -> None: ...

    @property
    def provenance(self) -> dict[str, Any]: ...

    def read_response(self, request: HumanResponseRequest) -> HumanResponse: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SpokenResponseParseResult:
    canonical_text: str | None
    status: SpokenResponseParseStatus
    reason: str | None = None


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


@dataclass(frozen=True)
class InteractiveTurnRecord:
    turn_id: int
    actor: InteractiveActor
    number: int | None = None
    prompt: str | list[ChatMessage] | None = None
    response: str = ""
    expected: str | None = None
    is_valid: bool = False
    latency_ms: float | None = None
    deadline_missed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class LocalChatModel(Protocol):
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
        timeout_s: float | None = None,
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


class InteractiveTask(Protocol):
    name: str

    def initial_state(self) -> TaskState: ...

    def build_human_prompt(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> str: ...

    def build_llm_messages(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> list[ChatMessage]: ...

    def validate_response(
        self,
        state: TaskState,
        response_text: str,
        *,
        actor: InteractiveActor | None = None,
        transcript: list[InteractiveTurnRecord] | None = None,
    ) -> TaskRefereeResult: ...

    def advance_state(
        self,
        state: TaskState,
        referee_result: TaskRefereeResult,
        response_text: str,
        *,
        actor: InteractiveActor | None = None,
        transcript: list[InteractiveTurnRecord] | None = None,
    ) -> TaskState: ...

    def expected_for_state(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> str | None: ...

    def build_hint(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> str | None: ...


@runtime_checkable
class SpeechAwareInteractiveTask(Protocol):
    def build_speech_context(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> SpeechContext: ...

    def parse_spoken_response(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        raw_text: str,
    ) -> SpokenResponseParseResult: ...
