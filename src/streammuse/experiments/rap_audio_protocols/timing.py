"""Native timing controls for the rap audio protocol comparison."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

from streammuse.experiments.rap_audio_protocols.contracts import SyllableTarget, TwoBarRenderRequest


_TOKENS_PER_SECOND = 12.5
_FASTPITCH_SAMPLE_RATE_HZ = 22050
_FASTPITCH_HOP_LENGTH = 256
_MEL_FRAMES_PER_SECOND = _FASTPITCH_SAMPLE_RATE_HZ / _FASTPITCH_HOP_LENGTH
_PADDING_LABELS = frozenset({"", "<blk>", "<blank>", "<eps>", "<pad>"})
_PUNCTUATION_LABELS = frozenset(string.punctuation)
_WORD_BOUNDARY_PUNCTUATION_LABELS = _PUNCTUATION_LABELS - {"'"}
_GRAPHEME_VOWELS = frozenset("aeiouy")
_CORE_GRAPHEME_VOWELS = _GRAPHEME_VOWELS - {"y"}
_ARPABET_VOWELS = frozenset(
    {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
)
_ARPABET_CONSONANTS = frozenset(
    {
        "B",
        "CH",
        "D",
        "DH",
        "F",
        "G",
        "HH",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
    }
)
_ARPABET_PHONE_PATTERN = re.compile(r"([A-Z]+)([012])?")
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
    syllable_phone_groups: tuple[tuple[str, ...], ...]
    syllable_label_indices: tuple[tuple[int, ...], ...]
    anchor_error_frames: tuple[int, ...]
    compressed_consonant_regions: tuple[int, ...]
    grapheme_fallback_words: tuple[str, ...]
    pronunciation_fallback_words: tuple[str, ...]


@dataclass(frozen=True)
class _WordSpan:
    word: str
    start_syllable_index: int
    end_syllable_index: int
    text_start: int


@dataclass(frozen=True)
class _TokenizerWordGroup:
    labels: tuple[str, ...]
    label_indices: tuple[int, ...]


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
    (
        syllable_phone_groups,
        syllable_label_indices,
        grapheme_fallback_words,
        pronunciation_fallback_words,
    ) = _resolve_syllable_phone_groups(request, tokenizer_labels)
    spoken_label_indices = tuple(index for group in syllable_label_indices for index in group)

    duration_frames = [0] * len(tokenizer_labels)
    vowel_label_indices: list[int] = []
    anchor_error_frames: list[int] = []
    compressed_consonant_regions: list[int] = []
    anchor_frames = tuple(round(syllable.target_seconds * _MEL_FRAMES_PER_SECOND) for syllable in request.syllables)
    total_frames = round(request.duration_seconds * _MEL_FRAMES_PER_SECOND)

    current_start = 0
    for syllable_index, phones in enumerate(syllable_phone_groups):
        vowel_phone_offset = _vowel_phone_offset(phones)
        onset_count = vowel_phone_offset
        coda_count = len(phones) - vowel_phone_offset - 1
        label_indices = syllable_label_indices[syllable_index]

        target_center = anchor_frames[syllable_index]
        left_vowel_frames = max(0, target_center - onset_count - current_start)
        actual_center = current_start + onset_count + left_vowel_frames
        anchor_error = actual_center - target_center
        if anchor_error != 0:
            compressed_consonant_regions.append(syllable_index)

        minimum_end = current_start + onset_count + left_vowel_frames + 1 + coda_count
        if syllable_index + 1 < len(request.syllables):
            next_onset_count = _vowel_phone_offset(syllable_phone_groups[syllable_index + 1])
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
        syllable_phone_groups=syllable_phone_groups,
        syllable_label_indices=syllable_label_indices,
        anchor_error_frames=tuple(anchor_error_frames),
        compressed_consonant_regions=tuple(dict.fromkeys(compressed_consonant_regions)),
        grapheme_fallback_words=grapheme_fallback_words,
        pronunciation_fallback_words=pronunciation_fallback_words,
    )


def _is_padding_label(label: str) -> bool:
    return label in _PADDING_LABELS


def _is_word_boundary_label(label: str) -> bool:
    return bool(label) and (label.isspace() or label in _WORD_BOUNDARY_PUNCTUATION_LABELS)


def _is_punctuation_label(label: str) -> bool:
    return label in _PUNCTUATION_LABELS


def _vowel_phone_offset(phones: tuple[str, ...]) -> int:
    for index, phone in enumerate(phones):
        if phone[-1:].isdigit() or _is_grapheme_vowel(phones, index):
            return index
    raise ValueError("cannot recover syllable phone groups without a vowel phone")


def _is_grapheme_vowel(labels: tuple[str, ...], index: int) -> bool:
    label = labels[index]
    if label not in _GRAPHEME_VOWELS:
        return False
    if label in _CORE_GRAPHEME_VOWELS:
        return True
    previous_is_core_vowel = index > 0 and labels[index - 1] in _CORE_GRAPHEME_VOWELS
    next_is_core_vowel = index + 1 < len(labels) and labels[index + 1] in _CORE_GRAPHEME_VOWELS
    return not previous_is_core_vowel and not next_is_core_vowel


def _resolve_syllable_phone_groups(
    request: TwoBarRenderRequest, tokenizer_labels: tuple[str, ...]
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    words = _word_spans(request)
    tokenizer_words = _tokenizer_word_groups(tokenizer_labels)
    if len(tokenizer_words) != len(words):
        raise ValueError("tokenizer word groups do not align with request words")

    syllable_phone_groups: list[tuple[str, ...]] = []
    syllable_label_indices: list[tuple[int, ...]] = []
    grapheme_fallback_words: list[str] = []
    pronunciation_fallback_words: list[str] = []
    for word_span, tokenizer_word in zip(words, tokenizer_words):
        word_syllables = request.syllables[word_span.start_syllable_index : word_span.end_syllable_index + 1]
        phone_groups, label_groups, used_grapheme_fallback, used_pronunciation_fallback = _resolve_word_phone_groups(
            word_span.word, word_syllables, tokenizer_word
        )
        syllable_phone_groups.extend(phone_groups)
        syllable_label_indices.extend(label_groups)
        if used_grapheme_fallback:
            grapheme_fallback_words.append(word_span.word)
        if used_pronunciation_fallback:
            pronunciation_fallback_words.append(word_span.word)
    return (
        tuple(syllable_phone_groups),
        tuple(syllable_label_indices),
        tuple(grapheme_fallback_words),
        tuple(pronunciation_fallback_words),
    )


def _tokenizer_word_groups(tokenizer_labels: tuple[str, ...]) -> tuple[_TokenizerWordGroup, ...]:
    groups: list[_TokenizerWordGroup] = []
    current_labels: list[str] = []
    current_indices: list[int] = []
    for index, label in enumerate(tokenizer_labels):
        if _is_padding_label(label):
            continue
        if _is_word_boundary_label(label):
            if current_labels:
                groups.append(_TokenizerWordGroup(labels=tuple(current_labels), label_indices=tuple(current_indices)))
                current_labels = []
                current_indices = []
            continue
        if _is_punctuation_label(label):
            continue
        current_labels.append(label)
        current_indices.append(index)
    if current_labels:
        groups.append(_TokenizerWordGroup(labels=tuple(current_labels), label_indices=tuple(current_indices)))
    return tuple(groups)


def _resolve_word_phone_groups(
    word: str,
    syllables: tuple[SyllableTarget, ...],
    tokenizer_word: _TokenizerWordGroup,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[int, ...], ...], bool, bool]:
    lexical_groups = tuple(syllable.phonemes for syllable in syllables)
    lexical_phones_populated = all(lexical_groups)
    if lexical_phones_populated:
        expected = tuple(phone for group in lexical_groups for phone in group)
        if tokenizer_word.labels == expected:
            label_groups: list[tuple[int, ...]] = []
            start = 0
            for group in lexical_groups:
                end = start + len(group)
                label_groups.append(tokenizer_word.label_indices[start:end])
                start = end
            return lexical_groups, tuple(label_groups), False, False

        if _is_pronunciation_fallback_word(tokenizer_word.labels, len(syllables)):
            phone_groups, label_groups = _recover_syllable_phone_groups(word, len(syllables), tokenizer_word)
            return phone_groups, label_groups, False, True

    if _is_grapheme_word(tokenizer_word.labels):
        phone_groups, label_groups = _recover_grapheme_syllable_groups(word, len(syllables), tokenizer_word)
        return phone_groups, label_groups, True, False

    if lexical_phones_populated:
        raise ValueError("tokenizer labels do not align with lexical phones")

    if any(lexical_groups):
        raise ValueError(f"cannot recover syllable phone groups for partially populated word {word!r}")
    phone_groups, label_groups = _recover_syllable_phone_groups(word, len(syllables), tokenizer_word)
    return phone_groups, label_groups, False, False


def _is_grapheme_word(labels: tuple[str, ...]) -> bool:
    return bool(labels) and all(
        len(label) == 1 and label.isascii() and label.islower() and label.isalpha() for label in labels
    )


def _is_pronunciation_fallback_word(labels: tuple[str, ...], syllable_count: int) -> bool:
    if not labels or not all(_is_arpabet_phone_label(label) for label in labels):
        return False
    return sum(_is_stress_marked_arpabet_vowel(label) for label in labels) == syllable_count


def _is_arpabet_phone_label(label: str) -> bool:
    match = _ARPABET_PHONE_PATTERN.fullmatch(label)
    if match is None:
        return False
    base_phone, stress = match.groups()
    if base_phone in _ARPABET_VOWELS:
        return stress is not None
    return base_phone in _ARPABET_CONSONANTS and stress is None


def _is_stress_marked_arpabet_vowel(label: str) -> bool:
    match = _ARPABET_PHONE_PATTERN.fullmatch(label)
    return match is not None and match.group(1) in _ARPABET_VOWELS and match.group(2) is not None


def _recover_grapheme_syllable_groups(
    word: str,
    syllable_count: int,
    tokenizer_word: _TokenizerWordGroup,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[int, ...], ...]]:
    nuclei: list[tuple[int, int]] = []
    index = 0
    while index < len(tokenizer_word.labels):
        if not _is_grapheme_vowel(tokenizer_word.labels, index):
            index += 1
            continue
        start = index
        while index < len(tokenizer_word.labels) and _is_grapheme_vowel(tokenizer_word.labels, index):
            index += 1
        nuclei.append((start, index))

    if len(nuclei) < syllable_count:
        raise ValueError(f"cannot recover grapheme syllable groups for word {word!r}")

    phone_groups: list[tuple[str, ...]] = []
    label_groups: list[tuple[int, ...]] = []
    start = 0
    for syllable_index in range(syllable_count):
        end = nuclei[syllable_index][1] if syllable_index + 1 < syllable_count else len(tokenizer_word.labels)
        phone_groups.append(tokenizer_word.labels[start:end])
        label_groups.append(tokenizer_word.label_indices[start:end])
        start = end
    return tuple(phone_groups), tuple(label_groups)


def _recover_syllable_phone_groups(
    word: str,
    syllable_count: int,
    tokenizer_word: _TokenizerWordGroup,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[int, ...], ...]]:
    vowel_positions = [index for index, phone in enumerate(tokenizer_word.labels) if phone[-1:].isdigit()]
    if len(vowel_positions) != syllable_count:
        raise ValueError(f"cannot recover syllable phone groups for word {word!r}")
    phone_groups: list[tuple[str, ...]] = []
    label_groups: list[tuple[int, ...]] = []
    start = 0
    for syllable_index, vowel_position in enumerate(vowel_positions):
        end = vowel_position + 1 if syllable_index + 1 < len(vowel_positions) else len(tokenizer_word.labels)
        if start > vowel_position or end <= start:
            raise ValueError(f"cannot recover syllable phone groups for word {word!r}")
        phone_groups.append(tokenizer_word.labels[start:end])
        label_groups.append(tokenizer_word.label_indices[start:end])
        start = end
    return tuple(phone_groups), tuple(label_groups)


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
