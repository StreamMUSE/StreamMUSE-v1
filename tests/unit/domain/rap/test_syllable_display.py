"""Display-only spelling approximation and bounded reuse."""

import pytest

from streammuse.domain.rap.models import Syllable
from streammuse.domain.rap.syllable_display import (
    syllable_display_label,
    written_syllable_fragments,
)


def test_fragment_cache_reuses_words_but_distinguishes_authoritative_counts():
    written_syllable_fragments.cache_clear()
    fragments = written_syllable_fragments("whisper", 2)

    assert fragments == ("whis", "per")
    assert written_syllable_fragments("whisper", 2) is fragments
    assert written_syllable_fragments("whisper", 3) == ("wh", "is", "per")
    info = written_syllable_fragments.cache_info()
    assert info.hits == 1
    assert info.misses == 2
    assert info.maxsize is not None
    for index in range(info.maxsize + 1):
        written_syllable_fragments(f"word{index}", 1)
    assert written_syllable_fragments.cache_info().currsize == info.maxsize


@pytest.mark.parametrize(("word", "count"), [
    ("Whisper", 2), ("can't", 1), ("o'clock", 2), ("o\u2019clock", 2),
    ("zqxtr", 3), ("beautiful", 3), ("fire", 1), ("fire", 2),
])
def test_fragments_preserve_case_apostrophes_and_oov_spelling(word, count):
    fragments = written_syllable_fragments(word, count)

    assert len(fragments) == count
    assert all(fragments)
    assert "".join(fragments) == word


def test_counts_exceeding_letter_count_do_not_lose_cells_or_raise():
    labels = [syllable_display_label(Syllable("a", index, 3, 0)) for index in range(3)]

    assert labels == ["-", "-", "-a"]
    assert "".join(label.strip("-") for label in labels) == "a"


@pytest.mark.parametrize(("index", "count"), [(0, 0), (-1, 2), (2, 2)])
def test_invalid_metadata_falls_back_to_existing_label(index, count):
    syllable = Syllable("word", index, count, 0)

    assert syllable_display_label(syllable) == syllable.label
