"""Application services for the rap-alignment prototype."""

from streammuse.application.rap.alignment import align_exact, align_legacy_flexible, align_text_to_slots, choose_best_line
from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher, RapStateProjector
from streammuse.application.rap.realtime import RollingRapController
from streammuse.application.rap.rhythm import available_patterns, build_bar_slots
from streammuse.application.rap.scoring import evaluate_candidate, rank_candidates

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
]
