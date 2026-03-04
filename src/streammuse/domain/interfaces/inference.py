"""Inference engine protocol."""

from __future__ import annotations

from typing import List, Protocol

from streammuse.domain.interfaces.timing_info import TimingInfo
from streammuse.domain.musical import MusicalEvent


class InferenceEngine(Protocol):
    """
    Protocol for ML inference engines (Stanley, Lekai, or future LLM backend).

    Canonical format is events; duration-based engines (e.g. Stanley) are
    adapted in infrastructure to this interface.
    """

    def generate_accompaniment(
        self,
        melody_events: List[MusicalEvent],
        generation_start_tick: int,
        generation_length_frames: int,
        prompt_length_ticks: int | None = None,
    ) -> tuple[List[MusicalEvent], TimingInfo]:
        """Generate accompaniment for the given melody; returns events and timings."""
        ...

    def inject_history(
        self,
        melody_events: List[MusicalEvent],
        accompaniment_events: List[MusicalEvent],
        injection_length_ticks: int,
    ) -> None:
        """Inject melody and accompaniment history (e.g. for prompt)."""
        ...

    def set_injection_offset(self, offset_ticks: int) -> None:
        """Set the injection offset used for subsequent generation."""
        ...

    def clear_history(self) -> None:
        """Clear internal history and injection state."""
        ...
