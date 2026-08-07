"""Assembly of multi-bar, beat-aligned rap text plans."""

from __future__ import annotations

from typing import Protocol

from streammuse.application.rap.alignment import choose_best_line
from streammuse.application.rap.rhythm import build_bar_slots
from streammuse.domain.rap import CandidateBatch, ProsodyAnalysis, RapPlan
from streammuse.domain.timing import Tempo


class CandidateGenerator(Protocol):
    """Supplies lyric-line candidates for deterministic local alignment."""

    def generate(self, topic: str, count: int) -> CandidateBatch:
        """Return candidate lines and their origin metadata."""


class ProsodyAnalyzer(Protocol):
    """Replaceable boundary for lyric prosody analysis."""

    def analyze(self, text: str) -> ProsodyAnalysis:
        """Return syllable, stress, rhyme-tail, boundary, and fallback diagnostics."""


class RapPrototypeService:
    """Build an inspectable text schedule without touching accompaniment state."""

    def __init__(self, tempo: Tempo, pattern: str, generator: CandidateGenerator) -> None:
        self._tempo = tempo
        self._pattern = pattern
        self._generator = generator

    def build_plan(self, topic: str, *, bars: int, candidate_count: int) -> RapPlan:
        """Generate candidates once and select a distinct best line per bar."""
        if bars <= 0:
            raise ValueError("bars must be positive")
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")

        batch = self._generator.generate(topic, candidate_count)
        candidates = tuple(candidate.strip() for candidate in batch.candidates if candidate.strip())
        if not candidates:
            raise ValueError("candidate generator returned no usable lines")

        used_texts: set[str] = set()
        lines = []
        for bar in range(bars):
            available = tuple(candidate for candidate in candidates if candidate not in used_texts) or candidates
            line = choose_best_line(available, build_bar_slots(self._tempo, self._pattern, bar))
            if line.overflow_count or not line.events:
                raise ValueError("candidate generator returned no line that fits one bar")
            lines.append(line)
            used_texts.add(line.text)

        return RapPlan(
            topic=topic,
            tempo=self._tempo,
            pattern=self._pattern,
            lines=tuple(lines),
            candidate_source=batch.source,
            warning=batch.warning,
        )
