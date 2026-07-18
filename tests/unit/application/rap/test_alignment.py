"""Tests for deterministic lyric-to-slot alignment."""

from streammuse.application.rap.alignment import align_text_to_slots, choose_best_line
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
