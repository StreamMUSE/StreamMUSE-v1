"""Offline and OpenAI-compatible lyric candidate generators."""

from __future__ import annotations

import re
from typing import Protocol

from streammuse.domain.rap import CandidateBatch, analyse_syllables


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

    def generate(self, topic: str, count: int) -> CandidateBatch:
        if count <= 0:
            raise ValueError("count must be positive")
        clean_topic = _normalise_topic(topic)
        candidates = tuple(_TEMPLATES[index % len(_TEMPLATES)].format(topic=clean_topic) for index in range(count))
        return CandidateBatch(candidates=candidates, source="phrase_bank")


class LocalChatCandidateGenerator:
    """Optional candidate source which never makes text alignment depend on HTTP."""

    def __init__(self, client: ChatClient, fallback: PhraseBankGenerator) -> None:
        self._client = client
        self._fallback = fallback

    def generate(self, topic: str, count: int) -> CandidateBatch:
        try:
            response = self._client.generate(
                _build_messages(topic, count),
                max_tokens=max(64, count * 24),
                temperature=0.8,
            )
            lines = _parse_candidate_lines(str(getattr(response, "text", "")), count)
        except Exception as exc:
            return _fallback_batch(self._fallback, topic, count, f"local chat candidate generation failed: {exc}")

        if not lines:
            return _fallback_batch(
                self._fallback,
                topic,
                count,
                "local chat candidate generation returned no usable lines",
            )
        return CandidateBatch(candidates=lines, source="local_chat")


def _normalise_topic(topic: str) -> str:
    words = _TOPIC_WORDS.findall(topic.lower())[:4]
    return " ".join(words) if words else "the moment"


def _build_messages(topic: str, count: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Write concise, clean rap lyric candidates. Return only one candidate per line.",
        },
        {
            "role": "user",
            "content": (
                f"Give {count} distinct one-bar lyric lines about {topic!r}. "
                "Each line must be 8 to 16 spoken English syllables, have no label or numbering, "
                "and be suitable for a four-four beat."
            ),
        },
    ]


def _parse_candidate_lines(text: str, count: int) -> tuple[str, ...]:
    candidates: list[str] = []
    for raw_line in text.splitlines():
        line = _NUMBERED_LINE.sub("", raw_line).strip().strip('"')
        syllable_count = len(analyse_syllables(line))
        if not line or not 1 <= syllable_count <= 16 or line in candidates:
            continue
        candidates.append(line)
        if len(candidates) == count:
            break
    return tuple(candidates)


def _fallback_batch(fallback: PhraseBankGenerator, topic: str, count: int, warning: str) -> CandidateBatch:
    batch = fallback.generate(topic, count)
    return CandidateBatch(candidates=batch.candidates, source=batch.source, warning=warning)
