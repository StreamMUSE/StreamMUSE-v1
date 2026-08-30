"""Tests for deterministic offline rap-song assembly rules."""

from __future__ import annotations

from streammuse.application.rap.offline_song import (
    FLOW_TEMPLATE_CYCLE,
    narrative_focus_for_bar,
    overused_closing_words,
    overused_opening_words,
    select_flow_qualified,
    select_flow_with_fallback,
    template_id_for_bar,
    word_trigrams,
)
from streammuse.domain.rap import (
    CandidateEvaluation,
    ProsodyAnalysis,
    ScoreComponent,
)


def _evaluation(candidate_id: str, *, total: float, stress: float) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id=candidate_id,
        text=candidate_id,
        analysis=ProsodyAnalysis(candidate_id, candidate_id, (), (), (), (), ()),
        valid=True,
        rejection_reasons=(),
        components=(
            ScoreComponent(
                name="stress_alignment",
                value=stress,
                weight=0.30,
                contribution=stress * 0.30,
                method="test",
            ),
        ),
        total_score=total,
        scheduled=(),
    )


def test_template_cycle_repeats_without_changing_bar_zero() -> None:
    assert template_id_for_bar(0) == FLOW_TEMPLATE_CYCLE[0]
    assert template_id_for_bar(len(FLOW_TEMPLATE_CYCLE)) == FLOW_TEMPLATE_CYCLE[0]
    assert [template_id_for_bar(bar) for bar in range(len(FLOW_TEMPLATE_CYCLE))] == list(FLOW_TEMPLATE_CYCLE)


def test_flow_qualified_selection_rejects_higher_total_with_weak_stress() -> None:
    weak = _evaluation("weak", total=0.90, stress=0.59)
    aligned = _evaluation("aligned", total=0.72, stress=0.68)

    qualified, selected = select_flow_qualified(
        (weak, aligned),
        minimum_score=0.30,
        minimum_stress=0.60,
    )

    assert qualified == (aligned,)
    assert selected == aligned


def test_flow_qualified_selection_uses_stable_source_order_for_equal_scores() -> None:
    first = _evaluation("first", total=0.72, stress=0.70)
    second = _evaluation("second", total=0.72, stress=0.80)

    qualified, selected = select_flow_qualified(
        (first, second),
        minimum_score=0.30,
        minimum_stress=0.60,
    )

    assert qualified == (first, second)
    assert selected == first


def test_flow_qualified_selection_returns_none_when_every_candidate_fails_gate() -> None:
    invalid = _evaluation("invalid", total=0.29, stress=0.90)

    qualified, selected = select_flow_qualified(
        (invalid,),
        minimum_score=0.30,
        minimum_stress=0.60,
    )

    assert qualified == ()
    assert selected is None


def test_flow_qualified_selection_rejects_song_level_trigram_reuse() -> None:
    repeated = _evaluation("space dreams ignite through stars", total=0.90, stress=0.80)
    fresh = _evaluation("rocket trails reveal new worlds", total=0.72, stress=0.70)

    qualified, selected = select_flow_qualified(
        (repeated, fresh),
        minimum_score=0.30,
        minimum_stress=0.60,
        blocked_trigrams=word_trigrams("space dreams ignite past earth"),
    )

    assert qualified == (fresh,)
    assert selected == fresh


def test_narrative_focus_advances_across_a_fifty_bar_song() -> None:
    assert "origin" in narrative_focus_for_bar(0, 50)
    assert "obstacle" in narrative_focus_for_bar(20, 50)
    assert "resolution" in narrative_focus_for_bar(49, 50)
    assert narrative_focus_for_bar(0, 50) != narrative_focus_for_bar(1, 50)


def test_selection_rejects_an_opening_word_after_two_song_uses() -> None:
    repeated = _evaluation("neon rockets cross the sky", total=0.90, stress=0.80)
    fresh = _evaluation("pilots chart a distant course", total=0.72, stress=0.70)
    blocked = overused_opening_words(
        ("neon lights reveal the launch", "neon trails cross the dark"),
        maximum_uses=2,
    )

    qualified, selected = select_flow_qualified(
        (repeated, fresh),
        minimum_score=0.30,
        minimum_stress=0.60,
        blocked_opening_words=blocked,
    )

    assert blocked == frozenset({"neon"})
    assert qualified == (fresh,)
    assert selected == fresh


def test_selection_rejects_a_closing_word_after_two_song_uses() -> None:
    repeated = _evaluation("street lights fade then rise", total=0.90, stress=0.80)
    fresh = _evaluation("street lights fade by dawn", total=0.72, stress=0.70)
    blocked = overused_closing_words(
        ("shadows climb and rise", "towers gleam then rise"),
        maximum_uses=2,
    )

    qualified, selected = select_flow_qualified(
        (repeated, fresh),
        minimum_score=0.30,
        minimum_stress=0.60,
        blocked_closing_words=blocked,
    )

    assert blocked == frozenset({"rise"})
    assert qualified == (fresh,)
    assert selected == fresh


def test_fallback_relaxes_only_opening_cap_before_other_diversity_gates() -> None:
    repeated_opening = _evaluation("neon rivers cross silent towns", total=0.80, stress=0.75)
    repeated_trigram = _evaluation("old lights shine through rain", total=0.90, stress=0.80)

    qualified, selected, mode = select_flow_with_fallback(
        (repeated_opening, repeated_trigram),
        minimum_score=0.30,
        minimum_stress=0.60,
        blocked_trigrams=word_trigrams("old lights shine past dawn"),
        blocked_opening_words=frozenset({"neon"}),
    )

    assert qualified == ()
    assert selected == repeated_opening
    assert mode == "relaxed_opening_word_cap"


def test_selection_rejects_non_ascii_text_before_pronunciation() -> None:
    mixed_script = _evaluation("memory stores life's烈焰", total=0.90, stress=0.80)
    ascii_line = _evaluation("memory stores a fading light", total=0.72, stress=0.70)

    qualified, selected = select_flow_qualified(
        (mixed_script, ascii_line),
        minimum_score=0.30,
        minimum_stress=0.60,
    )

    assert qualified == (ascii_line,)
    assert selected == ascii_line
