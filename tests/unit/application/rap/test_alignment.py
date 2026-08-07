"""Tests for deterministic lyric-to-slot alignment."""

import pytest

from streammuse.application.rap.alignment import (
    align_exact,
    align_legacy_flexible,
    align_text_to_slots,
    choose_best_line,
)
from streammuse.domain.rap import ProsodyAnalysis, Syllable
from streammuse.application.rap.rhythm import build_bar_slots
from streammuse.domain.timing import Tempo


def _slots():
    return build_bar_slots(Tempo(bpm=92, ticks_per_beat=4, beats_per_bar=4), "boom_bap", bar=0)


def test_alignment_keeps_syllables_in_order_and_anchors_the_downbeat() -> None:
    line = align_text_to_slots("space travel in the midnight", _slots())

    assert line.overflow_count == 0
    assert len(line.events) == len(line.syllables)
    assert line.events[0].slot.tick == 0
    assert [event.slot.tick for event in line.events] == sorted(event.slot.tick for event in line.events)
    assert [event.syllable.word for event in line.events] == [item.word for item in line.syllables]


def test_overflow_line_has_no_partial_schedule() -> None:
    line = align_text_to_slots("one " * 17, _slots())

    assert line.overflow_count == 1
    assert line.events == ()
    assert line.score < -100


def test_choose_best_line_rejects_overflow_when_a_fitting_candidate_exists() -> None:
    line = choose_best_line(("one two three four", "one " * 17), _slots())

    assert line.text == "one two three four"
    assert line.overflow_count == 0


def test_choose_best_line_preserves_candidate_order_when_scores_tie() -> None:
    line = choose_best_line(("one two", "three four"), _slots())

    assert line.text == "one two"


def test_exact_alignment_assigns_each_analyzed_syllable_to_its_matching_slot() -> None:
    slots = _slots()[:2]
    analysis = ProsodyAnalysis(
        text="one two",
        normalized_text="one two",
        syllables=(
            Syllable(word="one", index_in_word=0, syllable_count=1, stress=1),
            Syllable(word="two", index_in_word=0, syllable_count=1, stress=0),
        ),
        end_rhyme_tail=(),
        oov_words=(),
        heuristic_words=(),
        punctuation_boundary_after=(),
    )

    scheduled = align_exact(analysis, slots)

    assert [(item.syllable.word, item.slot.tick) for item in scheduled] == [("one", 0), ("two", 1)]


def test_exact_alignment_rejects_any_syllable_slot_count_mismatch() -> None:
    analysis = ProsodyAnalysis(
        text="one",
        normalized_text="one",
        syllables=(Syllable(word="one", index_in_word=0, syllable_count=1, stress=1),),
        end_rhyme_tail=(),
        oov_words=(),
        heuristic_words=(),
        punctuation_boundary_after=(),
    )

    with pytest.raises(ValueError, match="exact alignment requires 2 syllables, got 1"):
        align_exact(analysis, _slots()[:2])


def test_legacy_flexible_alignment_remains_the_existing_public_behavior() -> None:
    legacy = align_legacy_flexible("space travel in the midnight", _slots())
    compatibility_wrapper = align_text_to_slots("space travel in the midnight", _slots())

    assert legacy == compatibility_wrapper
