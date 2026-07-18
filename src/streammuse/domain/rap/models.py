"""Immutable value objects for beat-aligned lyric planning."""

from __future__ import annotations

from dataclasses import dataclass

from streammuse.domain.timing import Tempo


@dataclass(frozen=True)
class Syllable:
    """One estimated syllable within a source word."""

    word: str
    index_in_word: int
    syllable_count: int
    stressed: bool

    @property
    def label(self) -> str:
        """Compact terminal label which preserves the word boundary."""
        return self.word if self.index_in_word == 0 else "."


@dataclass(frozen=True)
class BeatSlot:
    """One position in a bar that can carry a lyric syllable."""

    bar: int
    tick: int
    beat: int
    tick_in_beat: int
    accent: float


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
class CandidateBatch:
    """Candidate lines supplied by an adapter and any non-fatal warning."""

    candidates: tuple[str, ...]
    source: str
    warning: str | None = None


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
