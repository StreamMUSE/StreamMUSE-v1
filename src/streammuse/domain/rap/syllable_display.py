"""Approximate written fragments for monitoring only, never pronunciation."""

from functools import lru_cache
import re

from streammuse.domain.rap.models import Syllable


_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.IGNORECASE)


@lru_cache(maxsize=4096)
def written_syllable_fragments(word: str, count: int) -> tuple[str, ...]:
    """Keep the original spelling and fit it to the authoritative syllable count.

    Vowel groups suggest boundaries, with intervening consonants shared between
    neighbors (whis/per). When spelling and pronunciation counts disagree, use
    equal character spans instead. This is not phonetic or acoustic alignment.
    Counts exceeding the spelling length leave empty spans, displayed as '-'.
    """
    if count < 1:
        return ()
    if count == 1:
        return (word,)
    vowels = list(_VOWEL_GROUPS.finditer(word))
    if len(vowels) == count:
        boundaries = [
            left.end() + (right.start() - left.end()) // 2
            for left, right in zip(vowels, vowels[1:])
        ]
    else:
        boundaries = [len(word) * index // count for index in range(1, count)]
    edges = [0, *boundaries, len(word)]
    return tuple(word[start:end] for start, end in zip(edges, edges[1:]))


def syllable_display_label(syllable: Syllable) -> str:
    """Hyphenate continuation cells without changing the core syllable label."""
    fragments = written_syllable_fragments(syllable.word, syllable.syllable_count)
    index = syllable.index_in_word
    if not 0 <= index < len(fragments):
        return syllable.label
    fragment = fragments[index]
    if not fragment:
        return "-"
    prefix = "-" if index else ""
    suffix = "-" if index < len(fragments) - 1 else ""
    return f"{prefix}{fragment}{suffix}"
