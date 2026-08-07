"""Immutable value objects for beat-aligned lyric planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from streammuse.domain.timing import Tempo


@dataclass(frozen=True)
class Syllable:
    """One estimated syllable within a source word."""

    word: str
    index_in_word: int
    syllable_count: int
    stress: int
    phonemes: tuple[str, ...] = ()
    analysis_source: str = "heuristic"

    @property
    def stressed(self) -> bool:
        """Preserve the historical boolean stress API."""
        return self.stress > 0

    @property
    def is_word_end(self) -> bool:
        """Whether this syllable closes its source word."""
        return self.index_in_word == self.syllable_count - 1

    @property
    def label(self) -> str:
        """Compact terminal label which preserves the word boundary."""
        return self.word if self.index_in_word == 0 else "."


@dataclass(frozen=True)
class ProsodyAnalysis:
    """Immutable diagnostics and prosody data for one input text."""

    text: str
    normalized_text: str
    syllables: tuple[Syllable, ...]
    end_rhyme_tail: tuple[str, ...]
    oov_words: tuple[str, ...]
    heuristic_words: tuple[str, ...]
    punctuation_boundary_after: tuple[int, ...]


@dataclass(frozen=True)
class BeatSlot:
    """One position in a bar that can carry a lyric syllable."""

    bar: int
    tick: int
    beat: int
    tick_in_beat: int
    accent: float
    duration_ticks: int = 1
    boundary_strength: int = 0
    rhyme_group: str | None = None
    template_id: str = "legacy"
    slot_index: int = 0


@dataclass(frozen=True)
class ScheduledSyllable:
    """A syllable assigned to one beat slot."""

    slot: BeatSlot
    syllable: Syllable


@dataclass(frozen=True)
class AlignedLine:
    """One candidate lyric line and its deterministic alignment result."""

    text: str
    syllables: tuple[Syllable, ...]
    events: tuple[ScheduledSyllable, ...]
    score: float
    overflow_count: int = 0


@dataclass(frozen=True)
class CandidateRequest:
    """Complete immutable generation input for one target bar."""

    request_id: str
    target_bar: int
    topic: str
    template_id: str
    required_syllables: int
    count: int
    context_lines: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("candidate request_id must be non-empty")
        if not isinstance(self.target_bar, int) or isinstance(self.target_bar, bool) or self.target_bar < 0:
            raise ValueError("candidate target_bar must be a non-negative integer")
        if not isinstance(self.topic, str) or not self.topic.strip():
            raise ValueError("candidate topic must be non-empty")
        if not isinstance(self.template_id, str) or not self.template_id.strip():
            raise ValueError("candidate template_id must be non-empty")
        if not isinstance(self.required_syllables, int) or isinstance(self.required_syllables, bool) or self.required_syllables <= 0:
            raise ValueError("candidate required_syllables must be positive")
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count <= 0:
            raise ValueError("candidate count must be positive")
        if not isinstance(self.context_lines, tuple) or not all(isinstance(line, str) for line in self.context_lines):
            raise ValueError("candidate context_lines must be a tuple of strings")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("candidate seed must be an integer")


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

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("candidate batch request_id must be non-empty")
        if not isinstance(self.candidates, tuple) or not all(isinstance(candidate, str) for candidate in self.candidates):
            raise ValueError("candidate batch candidates must be a tuple of strings")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("candidate batch source must be non-empty")
        if not isinstance(self.prompt, tuple) or not all(
            isinstance(message, Mapping)
            and all(isinstance(key, str) and isinstance(value, str) for key, value in message.items())
            for message in self.prompt
        ):
            raise ValueError("candidate batch prompt must be a tuple of string mappings")
        object.__setattr__(self, "prompt", tuple(MappingProxyType(dict(message)) for message in self.prompt))
        if not isinstance(self.raw_response, str):
            raise ValueError("candidate batch raw_response must be a string")
        if not isinstance(self.latency_ms, (int, float)) or isinstance(self.latency_ms, bool) or not isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("candidate batch latency_ms must be finite and non-negative")
        for name, value in (("prompt_tokens", self.prompt_tokens), ("completion_tokens", self.completion_tokens)):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"candidate batch {name} must be a non-negative integer or None")

    @property
    def prompt_json(self) -> tuple[dict[str, str], ...]:
        """Return a detached JSON-ready copy of the immutable prompt diagnostics."""
        return tuple(dict(message) for message in self.prompt)


@dataclass(frozen=True)
class RapPlan:
    """A complete multi-bar plan ready for rendering or timed playback."""

    topic: str
    tempo: Tempo
    pattern: str
    lines: tuple[AlignedLine, ...]
    candidate_source: str
    warning: str | None = None

    @property
    def events(self) -> tuple[ScheduledSyllable, ...]:
        """All scheduled syllables ordered by their planned bar and tick."""
        return tuple(event for line in self.lines for event in line.events)
