"""Deterministic CMU-dictionary prosody adapter for rap text."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

import pronouncing

from streammuse.domain.rap.models import ProsodyAnalysis, Syllable
from streammuse.domain.rap.prosody import analyse_syllables, extract_words, normalize_text

if TYPE_CHECKING:
    from streammuse.application.rap.service import ProsodyAnalyzer


_PUNCTUATION = re.compile(r"[,;:\-?!\.\u2013\u2014]+")
_WORD_MATCHES = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


class HeuristicProsodyAnalyzer:
    """Expose the legacy vowel-group estimate through the analyzer protocol."""

    def analyze(self, text: str) -> ProsodyAnalysis:
        syllables = tuple(
            replace(syllable, analysis_source="vowel_group_heuristic")
            for syllable in analyse_syllables(text)
        )
        return ProsodyAnalysis(
            text=text,
            normalized_text=normalize_text(text),
            syllables=syllables,
            end_rhyme_tail=(),
            oov_words=(),
            heuristic_words=extract_words(text),
            punctuation_boundary_after=punctuation_boundaries(text, syllables),
        )


class CmuProsodyAnalyzer:
    """Analyze text with CMUdict, retaining first-pronunciation provenance."""

    def __init__(self, fallback: ProsodyAnalyzer | None = None) -> None:
        self._fallback = fallback or HeuristicProsodyAnalyzer()

    def analyze(self, text: str) -> ProsodyAnalysis:
        syllables: list[Syllable] = []
        oov_words: list[str] = []
        heuristic_words: list[str] = []
        final_phones: tuple[str, ...] = ()

        for word in extract_words(text):
            pronunciations = pronouncing.phones_for_word(word)
            if not pronunciations:
                fallback = self._fallback.analyze(word)
                syllables.extend(fallback.syllables)
                oov_words.append(word)
                heuristic_words.append(word)
                final_phones = ()
                continue

            phones = tuple(pronunciations[0].split())
            syllables.extend(split_arpabet_syllables(word, phones))
            final_phones = phones

        frozen_syllables = tuple(syllables)
        return ProsodyAnalysis(
            text=text,
            normalized_text=normalize_text(text),
            syllables=frozen_syllables,
            end_rhyme_tail=rhyme_tail_from_last_stressed_vowel(final_phones),
            oov_words=tuple(oov_words),
            heuristic_words=tuple(heuristic_words),
            punctuation_boundary_after=punctuation_boundaries(text, frozen_syllables),
        )


def split_arpabet_syllables(word: str, phones: tuple[str, ...]) -> tuple[Syllable, ...]:
    """Split a first CMU pronunciation on its digit-bearing vowel nuclei."""
    nuclei = [index for index, phone in enumerate(phones) if phone[-1:].isdigit()]
    if not nuclei:
        return HeuristicProsodyAnalyzer().analyze(word).syllables

    syllables: list[Syllable] = []
    start = 0
    for syllable_index, nucleus in enumerate(nuclei):
        end = nuclei[syllable_index + 1] if syllable_index + 1 < len(nuclei) else len(phones)
        syllables.append(
            Syllable(
                word=word,
                index_in_word=syllable_index,
                syllable_count=len(nuclei),
                stress=int(phones[nucleus][-1]),
                phonemes=phones[start:end],
                analysis_source="cmudict_first_pronunciation",
            )
        )
        start = end
    return tuple(syllables)


def rhyme_tail_from_last_stressed_vowel(phones: tuple[str, ...]) -> tuple[str, ...]:
    """Return phones from the final primary or secondary stressed vowel onward."""
    stressed = [index for index, phone in enumerate(phones) if phone.endswith(("1", "2"))]
    vowels = [index for index, phone in enumerate(phones) if phone[-1:].isdigit()]
    start = stressed[-1] if stressed else vowels[-1] if vowels else len(phones)
    return phones[start:]


def punctuation_boundaries(text: str, syllables: tuple[Syllable, ...]) -> tuple[int, ...]:
    """Map supported punctuation runs to the preceding zero-based syllable index."""
    syllable_ends = [index for index, syllable in enumerate(syllables) if syllable.is_word_end]
    word_ends = [
        (match.end(), syllable_ends[word_index])
        for word_index, match in enumerate(_WORD_MATCHES.finditer(text))
        if word_index < len(syllable_ends)
    ]

    boundaries: list[int] = []
    for punctuation in _PUNCTUATION.finditer(text):
        preceding = [index for end, index in word_ends if end <= punctuation.start()]
        if preceding and (not boundaries or boundaries[-1] != preceding[-1]):
            boundaries.append(preceding[-1])
    return tuple(boundaries)
