"""Small, dependency-free prosody approximation for the prototype."""

from __future__ import annotations

import re

from streammuse.domain.rap.models import Syllable


_WORDS = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")

# These words are frequent enough in rap-oriented prompts to justify a precise
# local correction while the prototype deliberately has no pronunciation model.
_IRREGULAR_SYLLABLE_COUNTS = {
    "every": 3,
    "fire": 1,
    "hour": 1,
    "makes": 1,
    "moves": 1,
    "ocean": 2,
    "rhythm": 2,
    "sapphire": 2,
}


def analyse_syllables(text: str) -> tuple[Syllable, ...]:
    """Estimate syllables and word-level stress for a line of English text."""
    syllables: list[Syllable] = []
    for word in _WORDS.findall(text.lower()):
        count = _count_syllables(word)
        syllables.extend(
            Syllable(
                word=word,
                index_in_word=index,
                syllable_count=count,
                stressed=index == 0,
            )
            for index in range(count)
        )
    return tuple(syllables)


def _count_syllables(word: str) -> int:
    if word in _IRREGULAR_SYLLABLE_COUNTS:
        return _IRREGULAR_SYLLABLE_COUNTS[word]

    groups = _VOWEL_GROUPS.findall(word)
    count = len(groups)
    if word.endswith("e") and not _ends_with_syllabic_le(word) and not word.endswith("ye") and count > 1:
        count -= 1
    return max(1, count)


def _ends_with_syllabic_le(word: str) -> bool:
    """Treat `-le` as a syllable only after a consonant, as in `table`."""
    return len(word) > 2 and word.endswith("le") and word[-3] not in "aeiouy"
