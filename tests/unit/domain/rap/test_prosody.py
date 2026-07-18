"""Tests for dependency-free lyric prosody analysis."""

from streammuse.domain.rap import analyse_syllables


def test_analyse_syllables_marks_word_boundaries_and_irregular_counts() -> None:
    syllables = analyse_syllables("rhythm exploration")

    assert [(item.word, item.index_in_word, item.syllable_count, item.stressed) for item in syllables] == [
        ("rhythm", 0, 2, True),
        ("rhythm", 1, 2, False),
        ("exploration", 0, 4, True),
        ("exploration", 1, 4, False),
        ("exploration", 2, 4, False),
        ("exploration", 3, 4, False),
    ]


def test_analyse_syllables_uses_vowel_group_heuristic_for_common_words() -> None:
    syllables = analyse_syllables("space travel")

    assert [(item.word, item.index_in_word, item.syllable_count) for item in syllables] == [
        ("space", 0, 1),
        ("travel", 0, 2),
        ("travel", 1, 2),
    ]


def test_analyse_syllables_handles_silent_e_template_verbs_with_s_suffix() -> None:
    syllables = analyse_syllables("moves makes")

    assert [(item.word, item.syllable_count) for item in syllables] == [
        ("moves", 1),
        ("makes", 1),
    ]


def test_analyse_syllables_distinguishes_silent_and_syllabic_le_endings() -> None:
    counts = {word: len(analyse_syllables(word)) for word in ("while", "whole", "table", "simple", "file")}

    assert counts == {
        "while": 1,
        "whole": 1,
        "table": 2,
        "simple": 2,
        "file": 1,
    }


def test_analyse_syllables_ignores_non_words() -> None:
    assert analyse_syllables("  -- 123 !! ") == ()
