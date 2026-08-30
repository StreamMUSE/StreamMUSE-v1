"""Deterministic selection rules for offline rap-song production."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from streammuse.domain.rap import CandidateEvaluation
from streammuse.domain.rap.prosody import extract_words, normalize_text


FLOW_TEMPLATE_CYCLE = (
    "baseline_syncopated_9",
    "baseline_straight_9",
    "baseline_syncopated_9",
    "baseline_staggered_9",
    "baseline_straight_9",
    "baseline_syncopated_9",
    "baseline_staggered_9",
    "baseline_syncopated_9",
)

_NARRATIVE_STAGES = (
    "origin and physical setting",
    "ambition and forward motion",
    "obstacle and uncertainty",
    "discovery and consequences",
    "resolution and reflection",
)

_NARRATIVE_LENSES = (
    "a concrete place",
    "a human emotion",
    "a decisive action",
    "a difficult risk",
    "a tool or technique",
    "a sensory detail",
    "a cause and its effect",
    "a community impact",
    "a future possibility",
    "a personal realization",
)


def template_id_for_bar(bar: int) -> str:
    """Return the repeated eight-bar flow arrangement for one song."""
    if bar < 0:
        raise ValueError("bar must be nonnegative")
    return FLOW_TEMPLATE_CYCLE[bar % len(FLOW_TEMPLATE_CYCLE)]


def narrative_focus_for_bar(bar: int, total_bars: int) -> str:
    """Return a deterministic stage and lens that move a song forward."""
    if bar < 0 or total_bars <= 0 or bar >= total_bars:
        raise ValueError("bar must lie within a positive song length")
    stage_index = min(len(_NARRATIVE_STAGES) - 1, bar * len(_NARRATIVE_STAGES) // total_bars)
    return f"{_NARRATIVE_STAGES[stage_index]}, viewed through {_NARRATIVE_LENSES[bar % len(_NARRATIVE_LENSES)]}"


def word_trigrams(text: str) -> frozenset[tuple[str, str, str]]:
    """Return normalized consecutive word trigrams for repetition gating."""
    words = extract_words(normalize_text(text))
    return frozenset(zip(words, words[1:], words[2:]))


def opening_word(text: str) -> str:
    """Return the normalized first spoken word, or an empty value."""
    words = extract_words(normalize_text(text))
    return words[0] if words else ""


def closing_word(text: str) -> str:
    """Return the normalized final spoken word, or an empty value."""
    words = extract_words(normalize_text(text))
    return words[-1] if words else ""


def overused_opening_words(lines: Iterable[str], *, maximum_uses: int) -> frozenset[str]:
    """Return opening words that have reached the per-song use cap."""
    if maximum_uses <= 0:
        raise ValueError("maximum_uses must be positive")
    counts = Counter(opening_word(line) for line in lines)
    counts.pop("", None)
    return frozenset(word for word, count in counts.items() if count >= maximum_uses)


def overused_closing_words(lines: Iterable[str], *, maximum_uses: int) -> frozenset[str]:
    """Return closing words that have reached the per-song use cap."""
    if maximum_uses <= 0:
        raise ValueError("maximum_uses must be positive")
    counts = Counter(closing_word(line) for line in lines)
    counts.pop("", None)
    return frozenset(word for word, count in counts.items() if count >= maximum_uses)


def select_flow_qualified(
    evaluations: Sequence[CandidateEvaluation],
    *,
    minimum_score: float,
    minimum_stress: float,
    blocked_trigrams: frozenset[tuple[str, str, str]] = frozenset(),
    blocked_opening_words: frozenset[str] = frozenset(),
    blocked_closing_words: frozenset[str] = frozenset(),
) -> tuple[tuple[CandidateEvaluation, ...], CandidateEvaluation | None]:
    """Filter exact candidates through score and stress gates, then rank."""
    qualified = tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.text.isascii()
        and evaluation.valid
        and evaluation.total_score is not None
        and evaluation.total_score >= minimum_score
        and evaluation.component("stress_alignment").value >= minimum_stress
        and not (word_trigrams(evaluation.text) & blocked_trigrams)
        and opening_word(evaluation.text) not in blocked_opening_words
        and closing_word(evaluation.text) not in blocked_closing_words
    )
    if not qualified:
        return (), None
    selected = max(enumerate(qualified), key=lambda item: (item[1].total_score, -item[0]))[1]
    return qualified, selected


def select_flow_with_fallback(
    evaluations: Sequence[CandidateEvaluation],
    *,
    minimum_score: float,
    minimum_stress: float,
    blocked_trigrams: frozenset[tuple[str, str, str]] = frozenset(),
    blocked_opening_words: frozenset[str] = frozenset(),
    blocked_closing_words: frozenset[str] = frozenset(),
) -> tuple[tuple[CandidateEvaluation, ...], CandidateEvaluation | None, str]:
    """Select strictly, retaining an opening-cap-only last-resort choice."""
    qualified, selected = select_flow_qualified(
        evaluations,
        minimum_score=minimum_score,
        minimum_stress=minimum_stress,
        blocked_trigrams=blocked_trigrams,
        blocked_opening_words=blocked_opening_words,
        blocked_closing_words=blocked_closing_words,
    )
    if selected is not None:
        return qualified, selected, "strict"

    _relaxed, fallback = select_flow_qualified(
        evaluations,
        minimum_score=minimum_score,
        minimum_stress=minimum_stress,
        blocked_trigrams=blocked_trigrams,
        blocked_closing_words=blocked_closing_words,
    )
    if fallback is not None:
        return qualified, fallback, "relaxed_opening_word_cap"
    return qualified, None, "unavailable"
