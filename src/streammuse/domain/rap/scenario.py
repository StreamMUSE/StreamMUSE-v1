"""Immutable prescheduled rap scenarios."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSegment:
    """One contiguous sequence of bars sharing a topic and flow template."""

    start_bar: int
    bars: int
    topic: str
    template_id: str
    fallback_lines: tuple[str, ...]


@dataclass(frozen=True)
class RapScenario:
    """A fixed sequence of bar-level rap planning inputs."""

    scenario_id: str
    tempo_bpm: float
    segments: tuple[ScenarioSegment, ...]
    loop: bool = True

    @property
    def total_bars(self) -> int:
        """Return the scheduled duration before looping."""
        return sum(segment.bars for segment in self.segments)

    def segment_for_bar(self, bar: int) -> ScenarioSegment:
        """Return the segment scheduled for an absolute bar index."""
        if bar < 0:
            raise ValueError("bar must not be negative")
        effective = bar % self.total_bars if self.loop else bar
        for segment in self.segments:
            if segment.start_bar <= effective < segment.start_bar + segment.bars:
                return segment
        raise IndexError(f"bar {bar} lies outside scenario {self.scenario_id}")
