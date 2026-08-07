"""Domain contracts and prosody analysis for beat-aligned rap text."""

from streammuse.domain.rap.models import (
    AlignedLine,
    BeatSlot,
    CandidateBatch,
    CandidateRequest,
    ProsodyAnalysis,
    RapPlan,
    ScheduledSyllable,
    Syllable,
)
from streammuse.domain.rap.flow import FlowProvenance, FlowSlot, FlowTemplate, materialize_flow
from streammuse.domain.rap.evaluation import CandidateEvaluation, ScoreComponent, ScoreWeights, SelectionResult
from streammuse.domain.rap.prosody import analyse_syllables, extract_words, normalize_text
from streammuse.domain.rap.scenario import RapScenario, ScenarioSegment

__all__ = [
    "AlignedLine",
    "BeatSlot",
    "CandidateBatch",
    "CandidateRequest",
    "CandidateEvaluation",
    "FlowProvenance",
    "FlowSlot",
    "FlowTemplate",
    "ProsodyAnalysis",
    "RapPlan",
    "RapScenario",
    "ScheduledSyllable",
    "ScenarioSegment",
    "ScoreComponent",
    "ScoreWeights",
    "SelectionResult",
    "Syllable",
    "analyse_syllables",
    "extract_words",
    "materialize_flow",
    "normalize_text",
]
