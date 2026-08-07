"""Tests for CMU-dictionary-backed rap prosody analysis."""

from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer, HeuristicProsodyAnalyzer


def test_cmu_analyzer_exposes_stress_and_rhyme_tail() -> None:
    result = CmuProsodyAnalyzer().analyze("moving night")

    assert [syllable.word for syllable in result.syllables] == ["moving", "moving", "night"]
    assert [syllable.stress for syllable in result.syllables] == [1, 0, 1]
    assert result.end_rhyme_tail[-1].startswith("T")
    assert result.heuristic_words == ()


def test_oov_word_is_retained_and_identified_as_heuristic() -> None:
    result = CmuProsodyAnalyzer().analyze("xyzzy beat")

    assert "xyzzy" in result.oov_words
    assert "xyzzy" in result.heuristic_words
    assert any(syllable.word == "xyzzy" for syllable in result.syllables)
    assert all(syllable.analysis_source == "vowel_group_heuristic" for syllable in result.syllables if syllable.word == "xyzzy")


def test_punctuation_boundaries_follow_preceding_syllables_but_ignore_apostrophes() -> None:
    result = CmuProsodyAnalyzer().analyze("don't stop, now!")

    assert [syllable.word for syllable in result.syllables] == ["don't", "stop", "now"]
    assert result.punctuation_boundary_after == (1, 2)


def test_punctuation_boundaries_track_repeated_words_independently() -> None:
    result = CmuProsodyAnalyzer().analyze("moving, moving!")

    assert result.punctuation_boundary_after == (1, 3)


def test_punctuation_boundaries_include_unicode_dashes() -> None:
    result = CmuProsodyAnalyzer().analyze("night—day")

    assert result.punctuation_boundary_after == (0,)


def test_heuristic_analyzer_retains_the_baseline_for_empty_nonword_input() -> None:
    result = HeuristicProsodyAnalyzer().analyze("  -- 123 !! ")

    assert result.normalized_text == ""
    assert result.syllables == ()
    assert result.end_rhyme_tail == ()
    assert result.oov_words == ()
    assert result.heuristic_words == ()
    assert result.punctuation_boundary_after == ()
