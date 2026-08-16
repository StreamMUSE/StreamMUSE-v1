"""Tests for native timing plans in the rap audio protocol comparison."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pytest

from streammuse.experiments.rap_audio_protocols.corpus import load_song_corpus
from streammuse.experiments.rap_audio_protocols.timing import (
    build_fastpitch_phone_plan,
    build_ted_segments,
    moss_token_target,
)


FIXTURE_PATH = Path("tests/fixtures/rap_audio_protocols/two_bar_records.jsonl")
CAMPAIGN_OUTPUT_DIR = Path("output/rap_album_10x50_90bpm_20260816_v4")
TOTAL_FRAMES = round((16 / 3) * 22050 / 256)
OOV_PHONE_LABELS = {
    "gravity's": ("G", "R", "AE1", "V", "AH0", "T", "IY0", "Z"),
    "seabirds": ("S", "IY1", "B", "ER0", "D", "Z"),
    "darkness's": ("D", "AA1", "R", "K", "N", "AH0", "S", "Z"),
    "briny": ("B", "R", "AY1", "N", "IY0"),
    "midnight's": ("M", "IH1", "D", "N", "AY0", "T", "S"),
    "ebon": ("EH1", "B", "AH0", "N"),
    "twilight's": ("T", "W", "AY1", "L", "AY0", "T", "S"),
    "silicon's": ("S", "IH1", "L", "AH0", "K", "AA0", "N", "Z"),
    "ai's": ("AY1", "Z"),
    "binary's": ("B", "AY1", "N", "ER0", "IY0", "Z"),
    "skelebots": ("S", "K", "EH1", "L", "AH0", "B", "AA0", "T", "S"),
    "sync'd": ("S", "IH1", "NG", "K", "T"),
    "cogs": ("K", "AA1", "G", "Z"),
}


def _request():
    corpus = load_song_corpus(FIXTURE_PATH, song_id="01_space_exploration", expected_bars=2)
    return corpus.two_bar_requests()[0]


def _campaign_request(song_id: str, chunk_index: int):
    corpus = load_song_corpus(CAMPAIGN_OUTPUT_DIR / song_id / "chosen_lyrics.jsonl", song_id=song_id)
    return corpus.two_bar_requests()[chunk_index]


def _tokenizer_labels(request, overrides: dict[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    overrides = overrides or {}
    labels: list[str] = ["<blk>"]
    words = list(_word_syllables(request.syllables))
    for word_index, (word, syllables) in enumerate(words):
        if word.lower() in overrides:
            phones = overrides[word.lower()]
        elif all(syllable.phonemes for syllable in syllables):
            phones = tuple(phone for syllable in syllables for phone in syllable.phonemes)
        else:
            phones = OOV_PHONE_LABELS[word.lower()]
        for phone in phones:
            labels.append(phone)
            labels.append("<blk>")
        if word_index + 1 < len(words):
            labels.append(" ")
    return tuple(labels)


def _request_with_replaced_word(request, old_word: str, new_word: str, phoneme_groups: tuple[tuple[str, ...], ...]):
    matching_indices = [index for index, syllable in enumerate(request.syllables) if syllable.word == old_word]
    assert len(matching_indices) == len(phoneme_groups)
    syllables = list(request.syllables)
    for index, phonemes in zip(matching_indices, phoneme_groups):
        syllables[index] = replace(syllables[index], word=new_word, phonemes=phonemes)
    text_start = request.text.lower().index(old_word.lower())
    text_end = text_start + len(old_word)
    return replace(
        request,
        text=request.text[:text_start] + new_word + request.text[text_end:],
        syllables=tuple(syllables),
    )


def _nemo_punctuated_tokenizer_labels(request) -> tuple[str, ...]:
    labels: list[str] = ["<pad>"]
    punctuated_word_indices = {2, 5, 11}
    words = list(_word_syllables(request.syllables))
    for word_index, (_, syllables) in enumerate(words):
        labels.extend(phone for syllable in syllables for phone in syllable.phonemes)
        if word_index in punctuated_word_indices:
            labels.append(",")
        labels.append(" ")
    labels.extend((" ", "<pad>"))
    return tuple(labels)


def _nemo_tokenizer_labels_with_internal_apostrophe(request) -> tuple[str, ...]:
    labels: list[str] = ["<pad>"]
    words = list(_word_syllables(request.syllables))
    for word, syllables in words:
        if all(syllable.phonemes for syllable in syllables):
            phones = tuple(phone for syllable in syllables for phone in syllable.phonemes)
        else:
            phones = OOV_PHONE_LABELS[word.lower()]
        if "'" in word:
            labels.extend(phones[:-1])
            labels.append("'")
            labels.append(phones[-1])
        else:
            labels.extend(phones)
        labels.append(" ")
    return tuple(labels)


def _word_syllables(syllables) -> Iterable[tuple[str, tuple[object, ...]]]:
    start = 0
    while start < len(syllables):
        word = syllables[start].word
        end = start
        while end + 1 < len(syllables) and syllables[end + 1].word == word:
            end += 1
        yield word, syllables[start : end + 1]
        start = end + 1


def test_moss_target_and_ted_segments_follow_the_approved_two_bar_request() -> None:
    request = _request()

    assert moss_token_target(request) == 67

    segments = build_ted_segments(request)

    assert [segment.text_with_spacing for segment in segments] == [
        "Rocket blasts liftoff, silence breaks free, ",
        "Shadows stretch beneath vast cosmic sea,",
    ]
    assert "".join(segment.text_with_spacing for segment in segments) == request.text
    assert "".join(segment.text_with_spacing for segment in segments).split() == request.text.split()
    assert [segment.target_seconds for segment in segments] == pytest.approx([8 / 3, 8 / 3])
    assert sum(segment.target_seconds for segment in segments) == pytest.approx(16 / 3)
    assert all(segment.target_seconds >= 0.02 for segment in segments)


def test_ted_segments_fall_back_to_bar_boundaries_when_phrase_markers_are_missing() -> None:
    request = _request()
    flattened = []
    for syllable in request.syllables:
        flattened.append(replace(syllable, boundary_strength=0))
    request = replace(request, syllables=tuple(flattened))

    segments = build_ted_segments(request)

    assert [segment.text_with_spacing for segment in segments] == [
        "Rocket blasts liftoff, silence breaks free, ",
        "Shadows stretch beneath vast cosmic sea,",
    ]
    assert [segment.target_seconds for segment in segments] == pytest.approx([8 / 3, 8 / 3])


def test_fastpitch_phone_plan_allocates_durations_for_every_token_and_keeps_vowels_on_anchor() -> None:
    request = _request()
    tokenizer_labels = _tokenizer_labels(request)

    plan = build_fastpitch_phone_plan(request, tokenizer_labels)

    blank_or_space = {index for index, label in enumerate(tokenizer_labels) if label == "<blk>" or label == " "}
    vowel_indices = set(plan.vowel_label_indices)
    consonant_indices = set(plan.spoken_label_indices) - vowel_indices

    assert len(plan.duration_frames) == len(tokenizer_labels)
    assert all(isinstance(value, int) and value >= 0 for value in plan.duration_frames)
    assert sum(plan.duration_frames) == TOTAL_FRAMES
    assert all(plan.duration_frames[index] == 0 for index in blank_or_space)
    assert all(plan.duration_frames[index] == 1 for index in consonant_indices)
    assert all(plan.duration_frames[index] >= 1 for index in vowel_indices)
    assert all(index in vowel_indices for index, value in enumerate(plan.duration_frames) if value > 1)
    assert plan.anchor_error_frames[0] == 1
    assert all(abs(value) <= 1 for value in plan.anchor_error_frames)
    assert plan.compressed_consonant_regions == (0,)


def test_fastpitch_phone_plan_ignores_nemo_punctuation_and_padded_spaces() -> None:
    request = _request()
    tokenizer_labels = _nemo_punctuated_tokenizer_labels(request)

    plan = build_fastpitch_phone_plan(request, tokenizer_labels)

    non_spoken_indices = {
        index for index, label in enumerate(tokenizer_labels) if label in {",", " ", "<pad>"}
    }
    assert len(plan.duration_frames) == len(tokenizer_labels)
    assert all(plan.duration_frames[index] == 0 for index in non_spoken_indices)
    assert non_spoken_indices.isdisjoint(plan.spoken_label_indices)
    assert sum(plan.duration_frames) == TOTAL_FRAMES


def test_fastpitch_phone_plan_rejects_phone_token_mismatches() -> None:
    request = _request()
    tokenizer_labels = list(_tokenizer_labels(request))
    tokenizer_labels[tokenizer_labels.index("AE1")] = "AE0"

    with pytest.raises(ValueError, match="tokenizer labels do not align"):
        build_fastpitch_phone_plan(request, tuple(tokenizer_labels))


def test_fastpitch_phone_plan_recovers_single_syllable_where_from_graphemes() -> None:
    request = _request_with_replaced_word(_request(), "blasts", "where", (("W", "EH1", "R"),))
    tokenizer_labels = _tokenizer_labels(request, overrides={"where": tuple("where")})

    plan = build_fastpitch_phone_plan(request, tokenizer_labels)

    where_group = next(
        plan.syllable_phone_groups[index]
        for index, syllable in enumerate(request.syllables)
        if syllable.word == "where"
    )
    assert where_group == ("w", "h", "e", "r", "e")
    assert plan.grapheme_fallback_words == ("where",)
    where_indices = [index for index, label in enumerate(tokenizer_labels) if label in tuple("where")]
    assert any(plan.duration_frames[index] > 1 for index in where_indices if tokenizer_labels[index] == "e")


def test_fastpitch_phone_plan_recovers_multisyllabic_beyond_monotonically() -> None:
    request = _request_with_replaced_word(
        _request(),
        "rocket",
        "beyond",
        (("B", "IH0"), ("Y", "AA1", "N", "D")),
    )
    tokenizer_labels = _tokenizer_labels(request, overrides={"beyond": tuple("beyond")})

    plan = build_fastpitch_phone_plan(request, tokenizer_labels)

    beyond_groups = tuple(
        plan.syllable_phone_groups[index]
        for index, syllable in enumerate(request.syllables)
        if syllable.word == "beyond"
    )
    beyond_label_groups = tuple(
        plan.syllable_label_indices[index]
        for index, syllable in enumerate(request.syllables)
        if syllable.word == "beyond"
    )
    assert beyond_groups == (("b", "e"), ("y", "o", "n", "d"))
    assert beyond_label_groups == ((1, 3), (5, 7, 9, 11))
    assert tuple(
        tuple(tokenizer_labels[label_index] for label_index in label_group)
        for label_group in beyond_label_groups
    ) == beyond_groups
    assert plan.grapheme_fallback_words == ("beyond",)


def test_fastpitch_phone_plan_collapses_contiguous_ai_to_one_vowel_nucleus() -> None:
    request = _request_with_replaced_word(_request(), "blasts", "ai", (("EY1",),))
    tokenizer_labels = _tokenizer_labels(request, overrides={"ai": ("a", "i")})

    plan = build_fastpitch_phone_plan(request, tokenizer_labels)

    ai_group = next(
        plan.syllable_phone_groups[index]
        for index, syllable in enumerate(request.syllables)
        if syllable.word == "ai"
    )
    assert ai_group == ("a", "i")
    assert plan.grapheme_fallback_words == ("ai",)


def test_fastpitch_phone_plan_rejects_graphemes_with_too_few_vowel_nuclei() -> None:
    request = _request_with_replaced_word(
        _request(),
        "rocket",
        "rhythm",
        (("R", "IH1"), ("DH", "AH0", "M")),
    )
    tokenizer_labels = _tokenizer_labels(request, overrides={"rhythm": tuple("rhythm")})

    with pytest.raises(ValueError, match="cannot recover grapheme syllable groups for word 'rhythm'"):
        build_fastpitch_phone_plan(request, tokenizer_labels)


def test_fastpitch_phone_plan_recovers_real_corpus_oov_syllable_groups_from_tokenizer_labels() -> None:
    request = _campaign_request("01_space_exploration", 9)

    plan = build_fastpitch_phone_plan(request, _tokenizer_labels(request))

    gravity_groups = tuple(
        plan.syllable_phone_groups[index]
        for index, syllable in enumerate(request.syllables)
        if syllable.word == "gravity's"
    )
    assert gravity_groups == (
        ("G", "R", "AE1"),
        ("V", "AH0"),
        ("T", "IY0", "Z"),
    )
    assert sum(plan.duration_frames) == TOTAL_FRAMES


def test_fastpitch_phone_plan_ignores_internal_apostrophe_without_splitting_word() -> None:
    request = _campaign_request("01_space_exploration", 9)
    tokenizer_labels = _nemo_tokenizer_labels_with_internal_apostrophe(request)

    plan = build_fastpitch_phone_plan(request, tokenizer_labels)

    apostrophe_index = tokenizer_labels.index("'")
    gravity_groups = tuple(
        plan.syllable_phone_groups[index]
        for index, syllable in enumerate(request.syllables)
        if syllable.word == "gravity's"
    )
    assert gravity_groups == (
        ("G", "R", "AE1"),
        ("V", "AH0"),
        ("T", "IY0", "Z"),
    )
    assert plan.duration_frames[apostrophe_index] == 0
    assert apostrophe_index not in plan.spoken_label_indices


def test_fastpitch_phone_plan_rejects_oov_word_groups_with_the_wrong_vowel_count() -> None:
    request = _campaign_request("01_space_exploration", 9)
    tokenizer_labels = _tokenizer_labels(
        request,
        overrides={"gravity's": ("G", "R", "AE1", "V", "AH0", "T", "Z")},
    )

    with pytest.raises(ValueError, match="cannot recover syllable phone groups"):
        build_fastpitch_phone_plan(request, tokenizer_labels)


def test_campaign_songs_01_through_03_build_timing_plans_without_fastpitch_failures() -> None:
    for song_id in ("01_space_exploration", "02_deep_ocean", "03_artificial_intelligence"):
        corpus = load_song_corpus(CAMPAIGN_OUTPUT_DIR / song_id / "chosen_lyrics.jsonl", song_id=song_id)
        for request in corpus.two_bar_requests():
            labels = _tokenizer_labels(request)
            build_ted_segments(request)
            plan = build_fastpitch_phone_plan(request, labels)

            assert len(plan.duration_frames) == len(labels)
            assert sum(plan.duration_frames) == round(request.duration_seconds * 22050 / 256)
