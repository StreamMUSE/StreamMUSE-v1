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
