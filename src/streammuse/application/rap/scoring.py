"""Pure exact-alignment validation and transparent candidate ranking."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from streammuse.application.rap.alignment import align_exact
from streammuse.domain.rap import (
    CandidateEvaluation,
    FlowTemplate,
    ProsodyAnalysis,
    ScoreComponent,
    ScoreWeights,
    SelectionResult,
    materialize_flow,
)
from streammuse.domain.rap.prosody import extract_words, normalize_text


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


def stress_alignment(analysis: ProsodyAnalysis, template: FlowTemplate) -> float:
    """Score weighted distance between lexical and flow-template stress."""
    lexical = [1.0 if item.stress == 1 else 0.5 if item.stress == 2 else 0.0 for item in analysis.syllables]
    errors = [
        abs(actual - slot.target_stress) * (1.0 + slot.target_stress)
        for actual, slot in zip(lexical, template.slots, strict=True)
    ]
    denominator = sum(1.0 + slot.target_stress for slot in template.slots)
    return 1.0 - sum(errors) / denominator


def boundary_fit(analysis: ProsodyAnalysis, template: FlowTemplate) -> float:
    """Reward lexical or punctuation boundaries at configured flow boundaries."""
    targets = [index for index, slot in enumerate(template.slots) if slot.boundary_strength > 0]
    if not targets:
        return 1.0
    word_ends = {index for index, syllable in enumerate(analysis.syllables) if syllable.is_word_end}
    punctuation = set(analysis.punctuation_boundary_after)
    return sum(1.0 if index in punctuation else 0.6 if index in word_ends else 0.0 for index in targets) / len(targets)


def rhyme_quality(tail: tuple[str, ...], anchor: tuple[str, ...] | None) -> float:
    """Score exact or vowel-class rhyme agreement with a frozen anchor."""
    if anchor is None:
        return 0.5
    if tail == anchor and tail:
        return 1.0
    tail_vowels = tuple(re.sub(r"\d$", "", phone) for phone in tail if phone[-1:].isdigit())
    anchor_vowels = tuple(re.sub(r"\d$", "", phone) for phone in anchor if phone[-1:].isdigit())
    return 0.6 if tail_vowels and tail_vowels == anchor_vowels else 0.0


def topic_coverage(analysis: ProsodyAnalysis, topic: str) -> float:
    """Return the fraction of normalized topic content words that occur."""
    topic_tokens = _content_tokens(normalize_text(topic))
    if not topic_tokens:
        return 1.0
    candidate_tokens = set(_content_tokens(analysis.normalized_text))
    return sum(token in candidate_tokens for token in topic_tokens) / len(topic_tokens)


def lexical_continuity(analysis: ProsodyAnalysis, history: Sequence[ProsodyAnalysis | str], topic: str) -> float:
    """Measure shared non-topic content words against the two latest bars."""
    if not history:
        return 0.5
    topic_tokens = set(_content_tokens(normalize_text(topic)))
    candidate_tokens = set(_content_tokens(analysis.normalized_text)) - topic_tokens
    recent_tokens: set[str] = set()
    for item in history[-2:]:
        recent_tokens.update(_content_tokens(_history_normalized_text(item)))
    shared_non_topic_content_tokens = len(candidate_tokens & (recent_tokens - topic_tokens))
    return min(1.0, shared_non_topic_content_tokens / 2)


def novelty(analysis: ProsodyAnalysis, history: Sequence[ProsodyAnalysis | str]) -> float:
    """Return one minus the largest bigram Jaccard overlap in four recent bars."""
    candidate_bigrams = _bigrams(extract_words(analysis.normalized_text))
    similarities = [
        _bigram_jaccard(candidate_bigrams, _bigrams(extract_words(_history_normalized_text(item))))
        for item in history[-4:]
    ]
    return 1.0 - max(similarities, default=0.0)


def evaluate_candidate(
    *,
    candidate_id: str,
    text: str,
    analysis: ProsodyAnalysis,
    template: FlowTemplate,
    topic: str,
    history: Sequence[ProsodyAnalysis | str],
    rhyme_anchors: Mapping[tuple[int, str], tuple[str, ...]],
    weights: ScoreWeights,
    segment_start_bar: int = 0,
    target_bar: int = 0,
) -> CandidateEvaluation:
    """Hard-gate, exactly schedule, then score one analyzed candidate.

    The component scores are deterministic engineering proxies. They do not
    make a claim about human judgments of rap quality.
    """
    rejection_reasons: list[str] = []
    if len(analysis.syllables) != len(template.slots):
        rejection_reasons.append(f"syllable_count:{len(analysis.syllables)}!={len(template.slots)}")
    normalized_history = {_history_normalized_text(item) for item in history}
    if analysis.normalized_text in normalized_history:
        rejection_reasons.append("duplicate_normalized_text")
    if rejection_reasons:
        return CandidateEvaluation(
            candidate_id=candidate_id,
            text=text,
            analysis=analysis,
            valid=False,
            rejection_reasons=tuple(rejection_reasons),
            components=(),
            total_score=None,
            scheduled=(),
        )

    rhyme_group = next((slot.rhyme_group for slot in reversed(template.slots) if slot.rhyme_group is not None), None)
    anchor = rhyme_anchors.get((segment_start_bar, rhyme_group)) if rhyme_group is not None else None
    values = (
        ("stress_alignment", stress_alignment(analysis, template), "weighted_absolute_stress_error"),
        ("boundary_fit", boundary_fit(analysis, template), "punctuation_or_word_end_targets"),
        ("rhyme_quality", rhyme_quality(analysis.end_rhyme_tail, anchor), "exact_tail_or_vowel_class"),
        ("topic_coverage", topic_coverage(analysis, topic), "normalized_topic_content_fraction"),
        ("lexical_continuity", lexical_continuity(analysis, history, topic), "shared_non_topic_content_tokens_over_two"),
        ("novelty", novelty(analysis, history), "one_minus_max_bigram_jaccard"),
    )
    components = tuple(
        ScoreComponent(
            name=name,
            value=value,
            weight=getattr(weights, name),
            contribution=value * getattr(weights, name),
            method=method,
        )
        for name, value, method in values
    )
    return CandidateEvaluation(
        candidate_id=candidate_id,
        text=text,
        analysis=analysis,
        valid=True,
        rejection_reasons=(),
        components=components,
        total_score=sum(item.contribution for item in components),
        scheduled=align_exact(analysis, materialize_flow(template, target_bar)),
    )


def rank_candidates(
    candidates: Sequence[tuple[str, str, ProsodyAnalysis]],
    *,
    template: FlowTemplate,
    topic: str,
    history: Sequence[ProsodyAnalysis | str],
    rhyme_anchors: Mapping[tuple[int, str], tuple[str, ...]],
    weights: ScoreWeights,
    minimum_score: float = 0.0,
    segment_start_bar: int = 0,
    target_bar: int = 0,
) -> SelectionResult:
    """Evaluate every source candidate and select the best passing one.

    Equal totals retain their original candidate order. The function does not
    modify history or anchors; callers establish anchors only when a bar is
    frozen.
    """
    evaluations = tuple(
        evaluate_candidate(
            candidate_id=candidate_id,
            text=text,
            analysis=analysis,
            template=template,
            topic=topic,
            history=history,
            rhyme_anchors=rhyme_anchors,
            weights=weights,
            segment_start_bar=segment_start_bar,
            target_bar=target_bar,
        )
        for candidate_id, text, analysis in candidates
    )
    valid = [(index, item) for index, item in enumerate(evaluations) if item.valid and item.total_score is not None]
    if not valid:
        return SelectionResult(
            evaluations=evaluations,
            selected=None,
            threshold=minimum_score,
            fallback_reason="no_valid_candidates",
        )

    _index, best = max(valid, key=lambda item: (item[1].total_score, -item[0]))
    if best.total_score is None or best.total_score < minimum_score:
        return SelectionResult(
            evaluations=evaluations,
            selected=None,
            threshold=minimum_score,
            fallback_reason="minimum_score_not_met",
        )
    return SelectionResult(evaluations=evaluations, selected=best, threshold=minimum_score, fallback_reason=None)


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in extract_words(text) if token not in _STOPWORDS)


def _history_normalized_text(item: ProsodyAnalysis | str) -> str:
    return item.normalized_text if isinstance(item, ProsodyAnalysis) else normalize_text(item)


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:], strict=False))


def _bigram_jaccard(left: set[tuple[str, str]], right: set[tuple[str, str]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
