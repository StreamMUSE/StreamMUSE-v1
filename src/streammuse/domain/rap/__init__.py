"""Domain contracts and prosody analysis for beat-aligned rap text."""

from streammuse.domain.rap.models import (
    AlignedLine,
    BeatSlot,
    CandidateBatch,
    RapPlan,
    ScheduledSyllable,
    Syllable,
)
from streammuse.domain.rap.prosody import analyse_syllables

__all__ = [
    "AlignedLine",
    "BeatSlot",
    "CandidateBatch",
    "RapPlan",
    "ScheduledSyllable",
    "Syllable",
    "analyse_syllables",
]
