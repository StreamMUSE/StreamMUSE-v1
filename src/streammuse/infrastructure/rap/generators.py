"""Offline and OpenAI-compatible lyric candidate generators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from streammuse.domain.rap import CandidateBatch, CandidateRequest


_TOPIC_WORDS = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
_NUMBERED_LINE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")
_TEMPLATES = (
    "{topic} in the midnight all the speakers keep it bright",
    "we bring {topic} through the rhythm making every pulse ignite",
    "the room turns to {topic} every kick makes sparks arrive",
    "with {topic} in our hands we let the chorus cut the night",
    "we trace {topic} through the smoke and let the drums begin",
    "all of {topic} moves with every word we send tonight",
    "the bass lifts {topic} up and keeps the floor alive",
    "we make {topic} fit the pocket every time",
    "the rhythm holds {topic} while the whole crowd stays awake",
    "we turn {topic} into sparks that dance across the room",
)


class ChatClient(Protocol):
    """The small surface needed from the existing local chat client."""

    def generate(self, messages: list[dict[str, str]], **kwargs: object) -> object:
        """Produce an object with a text attribute."""


class PhraseBankGenerator:
    """Always-available baseline candidate source for local evaluation."""

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        clean_topic = _normalise_topic(request.topic)
        candidates = tuple(
            _TEMPLATES[index % len(_TEMPLATES)].format(topic=clean_topic) for index in range(request.count)
        )
        return CandidateBatch(
            request_id=request.request_id,
            candidates=candidates,
            source="phrase_bank",
            prompt=(),
            raw_response="",
            latency_ms=0.0,
        )


class LocalChatCandidateGenerator:
    """Optional observable candidate source backed by a local chat model."""

    def __init__(self, client: ChatClient) -> None:
        self._client = client

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        prompt = _build_messages(request)
        diagnostics = _ResponseDiagnostics()
        try:
            response = self._client.generate(
                [dict(message) for message in prompt],
                max_tokens=max(64, request.count * 24),
                temperature=0.8,
            )
            diagnostics, errors = _response_diagnostics(response)
            if errors:
                return _error_batch(request, prompt, "; ".join(errors), diagnostics=diagnostics)
            lines = _parse_candidate_lines(diagnostics.raw_response, request.count)
            if not lines:
                return _error_batch(
                    request,
                    prompt,
                    "local chat candidate generation returned no usable lines",
                    diagnostics=diagnostics,
                )
            return CandidateBatch(
                request_id=request.request_id,
                candidates=lines,
                source="local_chat",
                prompt=prompt,
                raw_response=diagnostics.raw_response,
                latency_ms=diagnostics.latency_ms,
                prompt_tokens=diagnostics.prompt_tokens,
                completion_tokens=diagnostics.completion_tokens,
            )
        except Exception as exc:
            return _error_batch(request, prompt, _sanitize_error(str(exc)), diagnostics=diagnostics)


def _normalise_topic(topic: str) -> str:
    words = _TOPIC_WORDS.findall(topic.lower())[:4]
    return " ".join(words) if words else "the moment"


def _build_messages(request: CandidateRequest) -> tuple[dict[str, str], ...]:
    history = "\n".join(f"- {line}" for line in request.context_lines) or "- (none)"
    return (
        {
            "role": "system",
            "content": "Write concise, clean rap lyric candidates. Return only one candidate per line.",
        },
        {
            "role": "user",
            "content": (
                f"Request {request.request_id} targets bar {request.target_bar} using flow template "
                f"{request.template_id!r}. Give {request.count} distinct one-bar lyric lines about "
                f"{request.topic!r}. Each line must contain exactly {request.required_syllables} spoken "
                "syllables, have no label or numbering, and be suitable for a four-four beat. "
                f"Recent frozen lines:\n{history}\nDeterministic variation seed: {request.seed}."
            ),
        },
    )


def _parse_candidate_lines(text: str, count: int) -> tuple[str, ...]:
    candidates: list[str] = []
    for raw_line in text.splitlines():
        line = _NUMBERED_LINE.sub("", raw_line).strip()
        if not line or line in candidates:
            continue
        candidates.append(line)
        if len(candidates) == count:
            break
    return tuple(candidates)


def _error_batch(
    request: CandidateRequest,
    prompt: tuple[dict[str, str], ...],
    message: str,
    *,
    diagnostics: "_ResponseDiagnostics | None" = None,
) -> CandidateBatch:
    diagnostics = diagnostics or _ResponseDiagnostics()
    return CandidateBatch(
        request_id=request.request_id,
        candidates=(),
        source="local_chat",
        prompt=prompt,
        raw_response=diagnostics.raw_response,
        latency_ms=diagnostics.latency_ms,
        prompt_tokens=diagnostics.prompt_tokens,
        completion_tokens=diagnostics.completion_tokens,
        error_type="generation_error",
        error_message=_sanitize_error(message),
    )


@dataclass(frozen=True)
class _ResponseDiagnostics:
    raw_response: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _response_diagnostics(response: object) -> tuple[_ResponseDiagnostics, tuple[str, ...]]:
    raw_response = ""
    latency_ms = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    errors: list[str] = []

    try:
        text = getattr(response, "text")
    except Exception as exc:
        errors.append(f"malformed chat response text: {exc}")
    else:
        if isinstance(text, str):
            raw_response = text
        else:
            errors.append("malformed chat response text must be a string")

    try:
        value = getattr(response, "latency_ms")
    except Exception as exc:
        errors.append(f"malformed chat response latency_ms: {exc}")
    else:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) and value >= 0:
            latency_ms = float(value)
        else:
            errors.append("malformed chat response latency_ms")

    for name in ("prompt_tokens", "completion_tokens"):
        try:
            value = getattr(response, name)
        except Exception as exc:
            errors.append(f"malformed chat response {name}: {exc}")
            continue
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"malformed chat response {name}")
            continue
        if name == "prompt_tokens":
            prompt_tokens = value
        else:
            completion_tokens = value

    return (
        _ResponseDiagnostics(
            raw_response=raw_response,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        tuple(errors),
    )


_SENSITIVE_ERROR_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>(?:[a-z][a-z0-9_]*_)?(?:api[-_]?key|access[-_]?key(?:[-_]?id)?|key|token|password|secret|authorization))"
    r"(?P<separator>\s*(?::|=)\s*)(?:bearer\s+)?[^\s,;}\]]+"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[^\s,;}\]]+")


def _sanitize_error(message: str) -> str:
    sanitized = _SENSITIVE_ERROR_ASSIGNMENT.sub(
        lambda match: f"{match.group('name')}{match.group('separator')}[REDACTED]",
        message,
    )
    return _BEARER_TOKEN.sub("Bearer [REDACTED]", sanitized)
