"""Tests for exact candidate evaluation and deterministic ranking."""

from __future__ import annotations

import pytest

from streammuse.application.rap.scoring import evaluate_candidate, rank_candidates
from streammuse.domain.rap import FlowProvenance, FlowSlot, FlowTemplate, ProsodyAnalysis, ScoreWeights, Syllable


def _analysis(
    text: str,
    stresses: tuple[int, ...],
    *,
    rhyme_tail: tuple[str, ...] = (),
    punctuation: tuple[int, ...] = (),
) -> ProsodyAnalysis:
    words = text.split()
    assert len(words) == len(stresses)
    return ProsodyAnalysis(
        text=text,
        normalized_text=text.lower(),
        syllables=tuple(
            Syllable(word=word.lower(), index_in_word=0, syllable_count=1, stress=stress)
            for word, stress in zip(words, stresses, strict=True)
        ),
        end_rhyme_tail=rhyme_tail,
        oov_words=(),
        heuristic_words=(),
        punctuation_boundary_after=punctuation,
    )


def _template(
    stresses: tuple[float, ...],
    *,
    boundary_index: int | None = None,
    rhyme_group: str | None = None,
) -> FlowTemplate:
    return FlowTemplate(
        template_id="test-template",
        name="Test template",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=tuple(
            FlowSlot(
                tick_in_bar=index,
                duration_ticks=1,
                target_stress=stress,
                boundary_strength=1 if index == boundary_index else 0,
                rhyme_group=rhyme_group if index == len(stresses) - 1 else None,
            )
            for index, stress in enumerate(stresses)
        ),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )


def test_exact_syllable_count_is_a_hard_gate() -> None:
    result = evaluate_candidate(
        candidate_id="c1",
        text="one two",
        analysis=_analysis("one two", (1, 1)),
        template=_template((1.0, 0.0, 1.0)),
        topic="space",
        history=(),
        rhyme_anchors={},
        weights=ScoreWeights(),
    )

    assert result.valid is False
    assert result.rejection_reasons == ("syllable_count:2!=3",)
    assert result.scheduled == ()
    assert result.components == ()
    assert result.total_score is None


def test_exact_normalized_duplicate_history_is_a_hard_gate() -> None:
    result = evaluate_candidate(
        candidate_id="c1",
        text="SPACE, lights!",
        analysis=_analysis("space lights", (1, 1)),
        template=_template((1.0, 1.0)),
        topic="space",
        history=(_analysis("space lights", (1, 1)),),
        rhyme_anchors={},
        weights=ScoreWeights(),
    )

    assert result.valid is False
    assert result.rejection_reasons == ("duplicate_normalized_text",)
    assert result.scheduled == ()
    assert result.components == ()
    assert result.total_score is None


def test_score_components_are_reconstructable_from_logged_values() -> None:
    result = evaluate_candidate(
        candidate_id="c1",
        text="space lights move",
        analysis=_analysis(
            "space lights move",
            (1, 0, 1),
            rhyme_tail=("AY1", "T"),
            punctuation=(1,),
        ),
        template=_template((1.0, 0.0, 1.0), boundary_index=1, rhyme_group="A"),
        topic="space",
        history=(_analysis("lights city", (1, 1)),),
        rhyme_anchors={(0, "A"): ("AY1", "T")},
        weights=ScoreWeights(),
    )

    assert result.valid is True
    assert result.component("stress_alignment").value == 1.0
    assert result.component("boundary_fit").value == 1.0
    assert result.component("rhyme_quality").value == 1.0
    assert result.component("topic_coverage").value == 1.0
    assert result.component("lexical_continuity").value == 0.5
    assert result.component("novelty").value == 1.0
    assert all(item.contribution == pytest.approx(item.value * item.weight) for item in result.components)
    assert result.total_score == pytest.approx(sum(item.contribution for item in result.components))
    assert len(result.scheduled) == 3


def test_rhyme_uses_segment_scoped_anchor_and_neutral_first_occurrence() -> None:
    analysis = _analysis("night light", (1, 1), rhyme_tail=("AY1", "T"))
    template = _template((1.0, 1.0), rhyme_group="A")

    neutral = evaluate_candidate(
        candidate_id="first",
        text=analysis.text,
        analysis=analysis,
        template=template,
        topic="night",
        history=(),
        rhyme_anchors={(3, "A"): ("IY1", "T")},
        weights=ScoreWeights(),
        segment_start_bar=4,
    )
    matching = evaluate_candidate(
        candidate_id="next",
        text=analysis.text,
        analysis=analysis,
        template=template,
        topic="night",
        history=(),
        rhyme_anchors={(4, "A"): ("AY1", "T")},
        weights=ScoreWeights(),
        segment_start_bar=4,
    )

    assert neutral.component("rhyme_quality").value == 0.5
    assert matching.component("rhyme_quality").value == 1.0


def test_novelty_uses_normalized_word_bigrams_without_content_filtering() -> None:
    result = evaluate_candidate(
        candidate_id="c1",
        text="space in light",
        analysis=_analysis("space in light", (1, 0, 1)),
        template=_template((1.0, 0.0, 1.0)),
        topic="space",
        history=(_analysis("night in light", (1, 0, 1)),),
        rhyme_anchors={},
        weights=ScoreWeights(),
    )

    # {space in, in light} vs {night in, in light}: Jaccard = 1/3.
    assert result.component("novelty").value == pytest.approx(2 / 3)


def test_rank_candidates_uses_source_order_for_equal_scores_and_applies_threshold() -> None:
    template = _template((1.0, 1.0))
    candidates = (
        ("second-in-source", "space light", _analysis("space light", (1, 1))),
        ("first-in-source", "space night", _analysis("space night", (1, 1))),
    )

    selected = rank_candidates(
        candidates,
        template=template,
        topic="space",
        history=(),
        rhyme_anchors={},
        weights=ScoreWeights(),
        minimum_score=0.0,
    )
    below_threshold = rank_candidates(
        candidates,
        template=template,
        topic="space",
        history=(),
        rhyme_anchors={},
        weights=ScoreWeights(),
        minimum_score=1.01,
    )

    assert [item.candidate_id for item in selected.evaluations] == ["second-in-source", "first-in-source"]
    assert selected.selected is not None
    assert selected.selected.candidate_id == "second-in-source"
    assert below_threshold.selected is None
    assert below_threshold.threshold == 1.01
