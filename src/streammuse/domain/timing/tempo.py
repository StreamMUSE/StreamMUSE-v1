"""
Tempo and musical time.

Immutable value objects for timing. No framework dependencies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tempo:
    """Immutable tempo configuration."""

    bpm: float
    ticks_per_beat: int
    beats_per_bar: int

    def __post_init__(self) -> None:
        if self.bpm <= 0:
            raise ValueError("BPM must be positive")
        if self.ticks_per_beat <= 0:
            raise ValueError("Ticks per beat must be positive")
        if self.beats_per_bar <= 0:
            raise ValueError("Beats per bar must be positive")

    @property
    def seconds_per_tick(self) -> float:
        """Real-time duration of one tick in seconds."""
        return (60.0 / self.bpm) / self.ticks_per_beat

    def tick_to_seconds(self, tick: int) -> float:
        """Convert tick to real-time seconds."""
        return tick * self.seconds_per_tick

    def seconds_to_tick(self, seconds: float) -> int:
        """Convert real-time seconds to tick."""
        return int(seconds / self.seconds_per_tick)

    @property
    def ticks_per_bar(self) -> int:
        """Total ticks per bar."""
        return self.ticks_per_beat * self.beats_per_bar


@dataclass(frozen=True)
class MusicalTime:
    """
    Musical time position (bar, beat, tick within beat).

    Used for display (e.g. "Bar 2, Beat 3") and metronome (first beat of bar
    vs other beats). Created from absolute tick + Tempo via from_tick.
    """

    tick: int
    bar: int
    beat: int
    tick_in_beat: int

    @classmethod
    def from_tick(cls, tick: int, tempo: Tempo) -> "MusicalTime":
        """Convert absolute tick to musical time (bar, beat, tick_in_beat)."""
        ticks_per_bar = tempo.ticks_per_bar
        bar = tick // ticks_per_bar
        tick_in_bar = tick % ticks_per_bar
        beat = tick_in_bar // tempo.ticks_per_beat
        tick_in_beat = tick_in_bar % tempo.ticks_per_beat
        return cls(tick=tick, bar=bar, beat=beat, tick_in_beat=tick_in_beat)
