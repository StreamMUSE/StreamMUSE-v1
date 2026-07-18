"""Application services for the rap-alignment prototype."""

from streammuse.application.rap.alignment import align_text_to_slots, choose_best_line
from streammuse.application.rap.realtime import RollingRapController
from streammuse.application.rap.rhythm import available_patterns, build_bar_slots

__all__ = [
    "RollingRapController",
    "align_text_to_slots",
    "available_patterns",
    "build_bar_slots",
    "choose_best_line",
]
