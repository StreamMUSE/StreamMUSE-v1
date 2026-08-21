"""Offline and OpenAI-compatible lyric candidate generators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from streammuse.domain.rap import CandidateBatch, CandidateRequest, normalize_text
from streammuse.infrastructure.inference.local_chat_client import (
    LocalChatChoicesResponse,
)
from streammuse.infrastructure.rap.flow_prompt import format_flow_for_prompt
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer


_TOPIC_WORDS = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
_NUMBERED_LINE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")
_MONOSYLLABIC_SKELETONS = (
    "we make beats hit hard and spark the night",
    "bright drums knock while bold words shake the ground",
    "bass runs deep as sharp rhymes wake the crowd",
    "kick snare snap and make heads nod real fast",
    "words land clean when strong drums drive us home",
    "new flows cut through dark rooms all night long",
    "we ride each pulse and keep the sound tight",
    "hard truths ring when drums strike through the haze",
    "these lines move swift while bass rolls through town",
    "each bar burns bright and turns cold nights gold",
    "raw thoughts rise while kick drums hold the pace",
    "we bend each phrase till all the words lock",
)


class ChatClient(Protocol):
    """The small surface needed from the existing local chat client."""

    def generate(self, messages: list[dict[str, str]], **kwargs: object) -> object:
        """Produce an object with a text attribute."""


class IndependentChoicesClient(Protocol):
    """The API-level independent-choice surface used by the H200 planner."""

    def generate_choices(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> LocalChatChoicesResponse:
        """Return every valid independently decoded response choice."""


class PhraseBankGenerator:
    """Always-available baseline candidate source for local evaluation."""

    def __init__(self) -> None:
        self._analyzer = CmuProsodyAnalyzer()

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        clean_topic = _normalise_topic(request.topic)
        topic_words, topic_syllables = _topic_prefix(
            clean_topic,
            request.required_syllables,
            self._analyzer,
        )
        candidates = tuple(
            _fit_phrase_bank_line(
                topic_words,
                topic_syllables,
                request.required_syllables,
                index,
                request.seed,
            )
            for index in range(request.count)
        )
        return CandidateBatch(
            request_id=request.request_id,
            candidates=candidates,
            source="phrase_bank",
            prompt=(),
            raw_response="",
            latency_ms=0.0,
        )


class ScriptedFailureGenerator:
    """Deterministic failure source used to prove no-gap fallback behavior."""

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        return CandidateBatch(
            request_id=request.request_id,
            candidates=(),
            source="scripted_failure",
            prompt=(),
            raw_response="",
            latency_ms=0.0,
            error_type="generation_error",
            error_message="scripted generator failure",
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
                return _error_batch(
                    request, prompt, "; ".join(errors), diagnostics=diagnostics
                )
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
                warning=(
                    f"requested_{request.count}_received_{len(lines)}"
                    if len(lines) < request.count
                    else None
                ),
            )
        except Exception as exc:
            return _error_batch(request, prompt, str(exc), diagnostics=diagnostics)


class IndependentChoiceCandidateGenerator:
    """Generate one lyric line from each independent API-level completion."""

    def __init__(
        self,
        client: IndependentChoicesClient,
        *,
        max_tokens_per_choice: int = 32,
        temperature: float = 1.0,
    ) -> None:
        if (
            not isinstance(max_tokens_per_choice, int)
            or isinstance(max_tokens_per_choice, bool)
            or max_tokens_per_choice <= 0
        ):
            raise ValueError("max_tokens_per_choice must be a positive integer")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature must be finite and non-negative")
        self._client = client
        self._max_tokens_per_choice = max_tokens_per_choice
        self._temperature = float(temperature)

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        prompt = _build_independent_messages(request)
        try:
            response = self._client.generate_choices(
                [dict(message) for message in prompt],
                n=request.count,
                max_tokens=self._max_tokens_per_choice,
                temperature=self._temperature,
            )
            if not isinstance(response, LocalChatChoicesResponse):
                raise ValueError("malformed independent-choice response")
            candidates: list[str] = []
            normalized: set[str] = set()
            for choice in response.choices:
                lines = _parse_candidate_lines(choice.text, 1)
                if not lines:
                    continue
                line = lines[0]
                identity = normalize_text(line)
                if not identity or identity in normalized:
                    continue
                normalized.add(identity)
                candidates.append(line)
                if len(candidates) == request.count:
                    break
            if not candidates:
                return _error_batch(
                    request,
                    prompt,
                    "independent-choice generation returned no usable lines",
                    diagnostics=_ResponseDiagnostics(
                        raw_response="\n".join(
                            choice.text for choice in response.choices
                        ),
                        latency_ms=response.latency_ms,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                    ),
                    source="local_chat_independent",
                )
            warnings = list(response.warnings)
            if len(candidates) < request.count:
                warnings.append(f"requested_{request.count}_received_{len(candidates)}")
            return CandidateBatch(
                request_id=request.request_id,
                candidates=tuple(candidates),
                source="local_chat_independent",
                prompt=prompt,
                raw_response="\n".join(choice.text for choice in response.choices),
                latency_ms=response.latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                warning="; ".join(warnings) if warnings else None,
            )
        except Exception as exc:
            return _error_batch(
                request,
                prompt,
                str(exc),
                source="local_chat_independent",
            )


def _normalise_topic(topic: str) -> str:
    words = _TOPIC_WORDS.findall(topic.lower())[:4]
    return " ".join(words) if words else "the moment"


def _topic_prefix(
    topic: str,
    required_syllables: int,
    analyzer: CmuProsodyAnalyzer,
) -> tuple[tuple[str, ...], int]:
    words: list[str] = []
    syllable_count = 0
    for word in topic.split():
        word_syllables = len(analyzer.analyze(word).syllables)
        if word_syllables <= 0 or syllable_count + word_syllables >= required_syllables:
            break
        words.append(word)
        syllable_count += word_syllables
    return tuple(words), syllable_count


def _fit_phrase_bank_line(
    topic_words: tuple[str, ...],
    topic_syllables: int,
    required_syllables: int,
    index: int,
    seed: int,
) -> str:
    skeleton_index = (seed + index) % len(_MONOSYLLABIC_SKELETONS)
    cycle_index = index // len(_MONOSYLLABIC_SKELETONS)
    skeleton = _MONOSYLLABIC_SKELETONS[skeleton_index].split()
    rotation = cycle_index % len(skeleton)
    if rotation:
        skeleton = skeleton[rotation:] + skeleton[:rotation]
    filler = tuple(
        skeleton[position % len(skeleton)]
        for position in range(topic_syllables, required_syllables)
    )
    return " ".join((*topic_words, *filler))


def _build_messages(request: CandidateRequest) -> tuple[dict[str, str], ...]:
    history = "\n".join(f"- {line}" for line in request.context_lines) or "- (none)"
    flow = format_flow_for_prompt(request.flow_template)
    return (
        {
            "role": "system",
            "content": (
                "You are a meticulous rap lyric writer and pronunciation-aware prosody checker. "
                "Follow the output contract exactly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Request {request.request_id} targets bar {request.target_bar} using flow template "
                f"{request.template_id!r}. Give {request.count} distinct one-bar lyric lines about "
                f"{request.topic!r}. Each line must contain exactly {request.required_syllables} spoken "
                "syllables and be suitable for a four-four beat. Place naturally stressed syllables near "
                "stronger flow slots, close the phrase at the final slot, and return plain lyric lines without "
                "syllable markup, labels, or numbering. Work internally in two stages: draft extra lines, then "
                "count every line using normal American spoken pronunciation. Silently discard or rewrite every "
                f"line that is not exactly {request.required_syllables} syllables. Output only the checked lines. "
                "Contractions count as spoken; do not rely on spelling to count syllables. "
                f"\n{flow}\n"
                f"Recent frozen lines:\n{history}\nDeterministic variation seed: {request.seed}."
            ),
        },
    )


def _build_independent_messages(
    request: CandidateRequest,
) -> tuple[dict[str, str], ...]:
    history = "\n".join(f"- {line}" for line in request.context_lines) or "- (none)"
    flow = format_flow_for_prompt(request.flow_template)
    return (
        {
            "role": "system",
            "content": (
                "You are a meticulous rap lyric writer and pronunciation-aware prosody checker. "
                "Return exactly one plain lyric line and nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write one distinct lyric line about {request.topic!r} for bar {request.target_bar}. "
                f"It must contain exactly {request.required_syllables} spoken syllables under normal American "
                "spoken pronunciation. Match lexical stress to the stronger slots, preserve natural words, and "
                "close the phrase at the final slot. Do not include numbering, labels, explanations, syllable "
                f"markup, or alternate lines.\n{flow}\nRecent frozen lines:\n{history}\n"
                f"Variation seed: {request.seed}."
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
    source: str = "local_chat",
) -> CandidateBatch:
    diagnostics = diagnostics or _ResponseDiagnostics()
    sanitized_message = _sanitize_error(message)
    return CandidateBatch(
        request_id=request.request_id,
        candidates=(),
        source=source,
        prompt=prompt,
        raw_response=diagnostics.raw_response,
        latency_ms=diagnostics.latency_ms,
        prompt_tokens=diagnostics.prompt_tokens,
        completion_tokens=diagnostics.completion_tokens,
        warning=sanitized_message,
        error_type="generation_error",
        error_message=sanitized_message,
    )


@dataclass(frozen=True)
class _ResponseDiagnostics:
    raw_response: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _response_diagnostics(
    response: object,
) -> tuple[_ResponseDiagnostics, tuple[str, ...]]:
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
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and value >= 0
        ):
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


# Keep a Bearer prefix and its token in one alternative so a redacted marker
# cannot be left behind when the optional prefix would otherwise backtrack.
_SENSITIVE_ERROR_VALUE = (
    r"(?:(?:bearer\s+)?\[redacted\]|bearer\s+[^\s,;}\]]+|[^\s,;}\]]+)"
)
_SENSITIVE_ERROR_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>(?:[a-z][a-z0-9_]*_)?(?:api[-_]?key|access[-_]?key(?:[-_]?id)?|key|token|password|secret|authorization))"
    rf"(?P<separator>\s*(?::|=)\s*){_SENSITIVE_ERROR_VALUE}"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+(?:\[redacted\]|[^\s,;}\]]+)")


def _sanitize_error(message: str) -> str:
    sanitized = _SENSITIVE_ERROR_ASSIGNMENT.sub(
        lambda match: f"{match.group('name')}{match.group('separator')}[REDACTED]",
        message,
    )
    return _BEARER_TOKEN.sub("Bearer [REDACTED]", sanitized)
