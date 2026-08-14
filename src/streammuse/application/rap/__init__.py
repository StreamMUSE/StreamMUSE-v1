"""Application services for the rap-alignment prototype."""

from streammuse.application.rap.alignment import align_exact, align_legacy_flexible, align_text_to_slots, choose_best_line
from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher, RapStateProjector
from streammuse.application.rap.realtime import RollingRapController
from streammuse.application.rap.rhythm import available_patterns, build_bar_slots
from streammuse.application.rap.scoring import evaluate_candidate, rank_candidates
from streammuse.application.rap.audio_rendering import (
    FitContext,
    FittedSyllable,
    bar_frame_count,
    bar_start_frame,
    fit_syllable,
    limit_peak,
    mix_at,
    tick_frame_in_bar,
    trim_silence,
)

__all__ = [
    "RollingRapController",
    "RapEventDispatcher",
    "RapEventPublisher",
    "RapStateProjector",
    "align_exact",
    "align_legacy_flexible",
    "align_text_to_slots",
    "available_patterns",
    "build_bar_slots",
    "choose_best_line",
    "evaluate_candidate",
    "rank_candidates",
    "FitContext",
    "FittedSyllable",
    "bar_frame_count",
    "bar_start_frame",
    "fit_syllable",
    "limit_peak",
    "mix_at",
    "tick_frame_in_bar",
    "trim_silence",
]
