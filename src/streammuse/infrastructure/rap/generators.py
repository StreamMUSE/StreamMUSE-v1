"""Offline and OpenAI-compatible lyric candidate generators."""

from __future__ import annotations

import re
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
        try:
            response = self._client.generate(
                [dict(message) for message in prompt],
                max_tokens=max(64, request.count * 24),
                temperature=0.8,
            )
            raw_response = str(getattr(response, "text", ""))
            lines = _parse_candidate_lines(raw_response, request.count)
        except Exception as exc:
            return _error_batch(request, prompt, _sanitize_error(str(exc)))

        if not lines:
            return _error_batch(request, prompt, "local chat candidate generation returned no usable lines", raw_response=raw_response)
        return CandidateBatch(
            request_id=request.request_id,
            candidates=lines,
            source="local_chat",
            prompt=prompt,
            raw_response=raw_response,
            latency_ms=_response_latency(response),
            prompt_tokens=_response_tokens(response, "prompt_tokens"),
            completion_tokens=_response_tokens(response, "completion_tokens"),
        )


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
        line = _NUMBERED_LINE.sub("", raw_line).strip().strip('"')
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
    raw_response: str = "",
) -> CandidateBatch:
    return CandidateBatch(
        request_id=request.request_id,
        candidates=(),
        source="local_chat",
        prompt=prompt,
        raw_response=raw_response,
        latency_ms=0.0,
        error_type="generation_error",
        error_message=message,
    )


def _response_latency(response: object) -> float:
    value = getattr(response, "latency_ms", 0.0)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else 0.0


def _response_tokens(response: object, name: str) -> int | None:
    value = getattr(response, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


_SENSITIVE_ERROR_VALUE = re.compile(
    r"(?i)\b(?:authorization|api[_-]?key|token|password)\b\s*(?::|=)\s*(?:bearer\s+)?[^\s,;}\]]+"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[^\s,;}\]]+")


def _sanitize_error(message: str) -> str:
    sanitized = _SENSITIVE_ERROR_VALUE.sub("[REDACTED]", message)
    return _BEARER_TOKEN.sub("Bearer [REDACTED]", sanitized)
