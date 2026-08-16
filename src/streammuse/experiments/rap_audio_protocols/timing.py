"""Native timing controls for the rap audio protocol comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass

from streammuse.experiments.rap_audio_protocols.contracts import SyllableTarget, TwoBarRenderRequest


_TOKENS_PER_SECOND = 12.5
_FASTPITCH_SAMPLE_RATE_HZ = 22050
_FASTPITCH_HOP_LENGTH = 256
_MEL_FRAMES_PER_SECOND = _FASTPITCH_SAMPLE_RATE_HZ / _FASTPITCH_HOP_LENGTH
_PADDING_LABELS = frozenset({"", "<blk>", "<blank>", "<eps>", "<pad>"})
_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@dataclass(frozen=True)
class TimedTextSegment:
    text_with_spacing: str
    target_seconds: float


@dataclass(frozen=True)
class FastPitchPhonePlan:
    tokenizer_labels: tuple[str, ...]
    duration_frames: tuple[int, ...]
    spoken_label_indices: tuple[int, ...]
    vowel_label_indices: tuple[int, ...]
    anchor_error_frames: tuple[int, ...]
    compressed_consonant_regions: tuple[int, ...]


@dataclass(frozen=True)
class _WordSpan:
    word: str
    start_syllable_index: int
    end_syllable_index: int
    text_start: int


def moss_token_target(request: TwoBarRenderRequest) -> int:
    """Return the MOSS global token target for a two-bar request."""
    return round(request.duration_seconds * _TOKENS_PER_SECOND)


def build_ted_segments(request: TwoBarRenderRequest) -> tuple[TimedTextSegment, ...]:
    """Collapse a two-bar request into TED text segments with timing targets."""
    words = _word_spans(request)
    internal_boundaries = _ted_boundary_words(request, words)

    segment_word_ranges: list[tuple[int, int]] = []
    start_word_index = 0
    for end_word_index in internal_boundaries:
        segment_word_ranges.append((start_word_index, end_word_index))
        start_word_index = end_word_index + 1
    segment_word_ranges.append((start_word_index, len(words) - 1))

    segments: list[TimedTextSegment] = []
    for segment_index, (start_word_index, end_word_index) in enumerate(segment_word_ranges):
        start_seconds = request.syllables[words[start_word_index].start_syllable_index].target_seconds
        if segment_index + 1 < len(segment_word_ranges):
            next_start_word_index = segment_word_ranges[segment_index + 1][0]
            next_start_seconds = request.syllables[words[next_start_word_index].start_syllable_index].target_seconds
        else:
            next_start_seconds = request.duration_seconds
        text_start = 0 if segment_index == 0 else words[start_word_index].text_start
        text_end = words[end_word_index + 1].text_start if end_word_index + 1 < len(words) else len(request.text)
        segments.append(
            TimedTextSegment(
                text_with_spacing=request.text[text_start:text_end],
                target_seconds=next_start_seconds - start_seconds,
            )
        )
    return tuple(segments)


def build_fastpitch_phone_plan(
    request: TwoBarRenderRequest, tokenizer_labels: tuple[str, ...]
) -> FastPitchPhonePlan:
    """Align tokenizer labels to lexical phones and assign per-label mel durations."""
    spoken_label_indices = tuple(index for index, label in enumerate(tokenizer_labels) if not _is_padding_label(label))
    lexical_phones = tuple(phone for syllable in request.syllables for phone in syllable.phonemes)
    if len(spoken_label_indices) != len(lexical_phones):
        raise ValueError("tokenizer labels do not align with lexical phones")
    for token_index, phone in zip(spoken_label_indices, lexical_phones):
        if tokenizer_labels[token_index] != phone:
            raise ValueError("tokenizer labels do not align with lexical phones")

    duration_frames = [0] * len(tokenizer_labels)
    vowel_label_indices: list[int] = []
    anchor_error_frames: list[int] = []
    compressed_consonant_regions: list[int] = []
    anchor_frames = tuple(round(syllable.target_seconds * _MEL_FRAMES_PER_SECOND) for syllable in request.syllables)
    total_frames = round(request.duration_seconds * _MEL_FRAMES_PER_SECOND)

    phone_cursor = 0
    current_start = 0
    for syllable_index, syllable in enumerate(request.syllables):
        vowel_phone_offset = _vowel_phone_offset(syllable)
        onset_count = vowel_phone_offset
        coda_count = len(syllable.phonemes) - vowel_phone_offset - 1
        label_indices = spoken_label_indices[phone_cursor : phone_cursor + len(syllable.phonemes)]
        phone_cursor += len(syllable.phonemes)

        target_center = anchor_frames[syllable_index]
        left_vowel_frames = max(0, target_center - onset_count - current_start)
        actual_center = current_start + onset_count + left_vowel_frames
        anchor_error = actual_center - target_center
        if anchor_error != 0:
            compressed_consonant_regions.append(syllable_index)

        minimum_end = current_start + onset_count + left_vowel_frames + 1 + coda_count
        if syllable_index + 1 < len(request.syllables):
            next_onset_count = _vowel_phone_offset(request.syllables[syllable_index + 1])
            maximum_end = anchor_frames[syllable_index + 1] - next_onset_count
            right_vowel_frames = max(0, maximum_end - minimum_end)
            if maximum_end < minimum_end:
                compressed_consonant_regions.append(syllable_index)
        else:
            right_vowel_frames = max(0, total_frames - minimum_end)
        vowel_duration = left_vowel_frames + 1 + right_vowel_frames

        for local_index, label_index in enumerate(label_indices):
            if local_index == vowel_phone_offset:
                duration_frames[label_index] = vowel_duration
                vowel_label_indices.append(label_index)
            else:
                duration_frames[label_index] = 1

        anchor_error_frames.append(anchor_error)
        current_start = minimum_end + right_vowel_frames

    return FastPitchPhonePlan(
        tokenizer_labels=tokenizer_labels,
        duration_frames=tuple(duration_frames),
        spoken_label_indices=spoken_label_indices,
        vowel_label_indices=tuple(vowel_label_indices),
        anchor_error_frames=tuple(anchor_error_frames),
        compressed_consonant_regions=tuple(dict.fromkeys(compressed_consonant_regions)),
    )


def _is_padding_label(label: str) -> bool:
    return label in _PADDING_LABELS or label.isspace()


def _vowel_phone_offset(syllable: SyllableTarget) -> int:
    for index, phone in enumerate(syllable.phonemes):
        if phone[-1:].isdigit():
            return index
    raise ValueError(f"syllable {syllable.word!r} has no vowel phone")


def _word_spans(request: TwoBarRenderRequest) -> tuple[_WordSpan, ...]:
    syllable_words: list[tuple[str, int, int]] = []
    start_index = 0
    for index, syllable in enumerate(request.syllables):
        if index > 0 and syllable.index_in_word == 0:
            previous = request.syllables[index - 1]
            syllable_words.append((previous.word, start_index, index - 1))
            start_index = index
    syllable_words.append((request.syllables[-1].word, start_index, len(request.syllables) - 1))

    text_words = list(_WORD_PATTERN.finditer(request.text))
    if len(text_words) != len(syllable_words):
        raise ValueError("request text does not match syllable word boundaries")

    words: list[_WordSpan] = []
    for match, (word, start_syllable_index, end_syllable_index) in zip(text_words, syllable_words):
        if match.group(0).lower() != word.lower():
            raise ValueError("request text does not match syllable surface words")
        words.append(
            _WordSpan(
                word=word,
                start_syllable_index=start_syllable_index,
                end_syllable_index=end_syllable_index,
                text_start=match.start(),
            )
        )
    return tuple(words)


def _ted_boundary_words(request: TwoBarRenderRequest, words: tuple[_WordSpan, ...]) -> tuple[int, ...]:
    boundary_words: list[int] = []
    previous_boundary = -1
    bar_count = request.end_bar - request.start_bar
    ticks_per_bar = 16

    for bar_offset in range(1, bar_count):
        boundary_tick = bar_offset * ticks_per_bar
        next_bar_start = next(
            (
                word_index
                for word_index, word in enumerate(words)
                if request.syllables[word.start_syllable_index].tick_in_chunk >= boundary_tick
            ),
            len(words),
        )
        preferred = next(
            (
                word_index
                for word_index in range(next_bar_start - 1, previous_boundary, -1)
                if request.syllables[words[word_index].end_syllable_index].boundary_strength > 0
            ),
            None,
        )
        boundary_words.append(preferred if preferred is not None else next_bar_start - 1)
        previous_boundary = boundary_words[-1]

    return tuple(boundary_words)
