"""
Musical sequence: immutable collection of events with tempo.

No framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

from streammuse.domain.musical.events import MusicalEvent
from streammuse.domain.timing import Tempo


@dataclass(frozen=True)
class MusicalSequence:
    """
    Immutable collection of musical events with timing information.

    Represents a complete musical passage (e.g. a phrase or window).
    """

    events: tuple[MusicalEvent, ...]
    tempo: Tempo

    def get_events_in_range(self, start_tick: int, end_tick: int) -> MusicalSequence:
        """Extract events in time range [start_tick, end_tick), returning a new sequence."""
        filtered = tuple(
            e for e in self.events
            if start_tick <= e.tick < end_tick
        )
        return MusicalSequence(events=filtered, tempo=self.tempo)

    def quantize(self, quantization_ticks: int) -> MusicalSequence:
        """Quantize event ticks to a grid (round down to nearest multiple of quantization_ticks)."""
        if quantization_ticks <= 0:
            raise ValueError("quantization_ticks must be positive")
        quantized = tuple(
            MusicalEvent(
                tick=(e.tick // quantization_ticks) * quantization_ticks,
                pitch=e.pitch,
                event_type=e.event_type,
                velocity=e.velocity,
                channel=e.channel,
                program=e.program,
                is_placeholder=e.is_placeholder,
            )
            for e in self.events
        )
        return MusicalSequence(events=quantized, tempo=self.tempo)
