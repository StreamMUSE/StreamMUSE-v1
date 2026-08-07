"""Immutable, inspectable records for deterministic rap candidate evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from streammuse.domain.rap.models import ProsodyAnalysis, ScheduledSyllable


@dataclass(frozen=True)
class ScoreWeights:
    """Baseline weights for the transparent candidate-scoring proxy."""

    stress_alignment: float = 0.30
    boundary_fit: float = 0.10
    rhyme_quality: float = 0.20
    topic_coverage: float = 0.20
    lexical_continuity: float = 0.15
    novelty: float = 0.05

    def __post_init__(self) -> None:
        if abs(sum(asdict(self).values()) - 1.0) > 1e-9:
            raise ValueError("score weights must sum to one")


@dataclass(frozen=True)
class ScoreComponent:
    """One logged score value and its weighted contribution."""

    name: str
    value: float
    weight: float
    contribution: float
    method: str


@dataclass(frozen=True)
class CandidateEvaluation:
    """Complete validation, alignment, and scoring outcome for one candidate."""

    candidate_id: str
    text: str
    analysis: ProsodyAnalysis
    valid: bool
    rejection_reasons: tuple[str, ...]
    components: tuple[ScoreComponent, ...]
    total_score: float | None
    scheduled: tuple[ScheduledSyllable, ...]

    def component(self, name: str) -> ScoreComponent:
        """Return one named component for presentation or diagnostics."""
        return next(item for item in self.components if item.name == name)


@dataclass(frozen=True)
class SelectionResult:
    """All evaluated candidates and the deterministic selection outcome."""

    evaluations: tuple[CandidateEvaluation, ...]
    selected: CandidateEvaluation | None
    threshold: float
    fallback_reason: str | None
