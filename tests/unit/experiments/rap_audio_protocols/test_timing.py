"""Tests for native timing plans in the rap audio protocol comparison."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from streammuse.experiments.rap_audio_protocols.corpus import load_song_corpus
from streammuse.experiments.rap_audio_protocols.timing import (
    build_fastpitch_phone_plan,
    build_ted_segments,
    moss_token_target,
)


FIXTURE_PATH = Path("tests/fixtures/rap_audio_protocols/two_bar_records.jsonl")
TOTAL_FRAMES = round((16 / 3) * 22050 / 256)


def _request():
    corpus = load_song_corpus(FIXTURE_PATH, song_id="01_space_exploration", expected_bars=2)
    return corpus.two_bar_requests()[0]


def _tokenizer_labels(request) -> tuple[str, ...]:
    labels: list[str] = ["<blk>"]
    last_word = request.syllables[0].word
    for syllable_index, syllable in enumerate(request.syllables):
        if syllable_index > 0:
            labels.append("<blk>")
        labels.extend(syllable.phonemes)
        next_word = request.syllables[syllable_index + 1].word if syllable_index + 1 < len(request.syllables) else None
        if next_word is not None and next_word != last_word:
            labels.append(" ")
        last_word = syllable.word
    labels.append("<blk>")
    return tuple(labels)


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


def test_fastpitch_phone_plan_rejects_phone_token_mismatches() -> None:
    request = _request()
    tokenizer_labels = list(_tokenizer_labels(request))
    tokenizer_labels[tokenizer_labels.index("AE1")] = "AE0"

    with pytest.raises(ValueError, match="tokenizer labels do not align"):
        build_fastpitch_phone_plan(request, tuple(tokenizer_labels))
