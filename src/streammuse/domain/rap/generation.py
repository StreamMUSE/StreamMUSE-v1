"""Immutable values at the lyric candidate generation boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from streammuse.domain.rap.flow import FlowTemplate


@dataclass(frozen=True)
class CandidateRequest:
    """Complete immutable generation input for one target bar."""

    request_id: str
    target_bar: int
    topic: str
    flow_template: FlowTemplate
    count: int
    context_lines: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("candidate request_id must be non-empty")
        if (
            not isinstance(self.target_bar, int)
            or isinstance(self.target_bar, bool)
            or self.target_bar < 0
        ):
            raise ValueError("candidate target_bar must be a non-negative integer")
        if not isinstance(self.topic, str) or not self.topic.strip():
            raise ValueError("candidate topic must be non-empty")
        if not isinstance(self.flow_template, FlowTemplate):
            raise ValueError("candidate flow_template must be a FlowTemplate")
        if (
            not isinstance(self.count, int)
            or isinstance(self.count, bool)
            or self.count <= 0
        ):
            raise ValueError("candidate count must be positive")
        if not isinstance(self.context_lines, tuple) or not all(
            isinstance(line, str) for line in self.context_lines
        ):
            raise ValueError("candidate context_lines must be a tuple of strings")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("candidate seed must be an integer")

    @property
    def template_id(self) -> str:
        """Return the identity of the authoritative flow template."""
        return self.flow_template.template_id

    @property
    def required_syllables(self) -> int:
        """Return the syllable count implied by the authoritative flow template."""
        return len(self.flow_template.slots)


@dataclass(frozen=True)
class CandidateBatch:
    """Raw candidate lines and non-secret diagnostics for one request."""

    request_id: str
    candidates: tuple[str, ...]
    source: str
    prompt: tuple[Mapping[str, str], ...]
    raw_response: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    warning: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    provider_choice_indices: tuple[int | None, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("candidate batch request_id must be non-empty")
        if not isinstance(self.candidates, tuple) or not all(
            isinstance(candidate, str) for candidate in self.candidates
        ):
            raise ValueError("candidate batch candidates must be a tuple of strings")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("candidate batch source must be non-empty")
        if not isinstance(self.prompt, tuple) or not all(
            isinstance(message, Mapping)
            and all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in message.items()
            )
            for message in self.prompt
        ):
            raise ValueError(
                "candidate batch prompt must be a tuple of string mappings"
            )
        object.__setattr__(
            self,
            "prompt",
            tuple(MappingProxyType(dict(message)) for message in self.prompt),
        )
        if not isinstance(self.raw_response, str):
            raise ValueError("candidate batch raw_response must be a string")
        if (
            not isinstance(self.latency_ms, (int, float))
            or isinstance(self.latency_ms, bool)
            or not isfinite(self.latency_ms)
            or self.latency_ms < 0
        ):
            raise ValueError(
                "candidate batch latency_ms must be finite and non-negative"
            )
        for name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(
                    f"candidate batch {name} must be a non-negative integer or None"
                )
        if (
            not isinstance(self.provider_choice_indices, tuple)
            or (
                self.provider_choice_indices
                and len(self.provider_choice_indices) != len(self.candidates)
            )
            or any(
                index is not None
                and (not isinstance(index, int) or isinstance(index, bool) or index < 0)
                for index in self.provider_choice_indices
            )
        ):
            raise ValueError(
                "candidate batch provider_choice_indices must be empty or align "
                "one-to-one with candidates using non-negative integers or None"
            )

    @property
    def prompt_json(self) -> tuple[dict[str, str], ...]:
        """Return a detached JSON-ready copy of the immutable prompt diagnostics."""
        return tuple(dict(message) for message in self.prompt)
