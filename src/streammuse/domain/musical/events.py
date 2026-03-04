"""
Musical event and note value objects.

Immutable domain types for note_on/note_off events and duration-based notes.
No framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class EventType(Enum):
    """Musical event types (MIDI note on/off)."""

    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"


@dataclass(frozen=True)
class MusicalEvent:
    """
    Immutable musical event — the fundamental unit of musical data.

    All musical data flows through the system as MusicalEvent instances.
    Supports placeholder events (e.g. for missing ticks) via is_placeholder.
    """

    tick: int
    pitch: int
    event_type: EventType
    velocity: int = 100
    channel: int = 0
    program: int = 0  # MIDI program/instrument
    is_placeholder: bool = False

    def __post_init__(self) -> None:
        """Validate event data."""
        if self.tick < 0:
            raise ValueError(f"Invalid tick: {self.tick}")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"Invalid velocity: {self.velocity}")
        if self.is_placeholder:
            if self.pitch != -1:
                raise ValueError(
                    "Placeholder events must have pitch=-1"
                )
        else:
            if not 0 <= self.pitch <= 127:
                raise ValueError(f"Invalid pitch: {self.pitch}")
        if not 0 <= self.channel <= 15:
            raise ValueError(f"Invalid channel: {self.channel}")
        if not 0 <= self.program <= 127:
            raise ValueError(f"Invalid program: {self.program}")


@dataclass(frozen=True)
class Note:
    """
    Duration-based note representation.

    Can be converted to/from a pair of MusicalEvent (note_on, note_off).
    Supports placeholder notes (pitch=-1, is_placeholder=True) for missing ticks.
    """

    pitch: int
    tick: int
    duration: int
    velocity: int = 100
    channel: int = 0
    program: int = 0
    is_placeholder: bool = False

    def __post_init__(self) -> None:
        """Validate note data."""
        if self.tick < 0:
            raise ValueError(f"Invalid tick: {self.tick}")
        if self.duration <= 0:
            raise ValueError(f"Invalid duration: {self.duration} (must be positive)")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"Invalid velocity: {self.velocity}")
        if not 0 <= self.channel <= 15:
            raise ValueError(f"Invalid channel: {self.channel}")
        if not 0 <= self.program <= 127:
            raise ValueError(f"Invalid program: {self.program}")
        if self.is_placeholder:
            if self.pitch != -1:
                raise ValueError("Placeholder notes must have pitch=-1")
        else:
            if not 0 <= self.pitch <= 127:
                raise ValueError(f"Invalid pitch: {self.pitch}")

    def to_events(self) -> List[MusicalEvent]:
        """Convert to note_on and note_off event pair."""
        return [
            MusicalEvent(
                tick=self.tick,
                pitch=self.pitch,
                event_type=EventType.NOTE_ON,
                velocity=self.velocity,
                channel=self.channel,
                program=self.program,
                is_placeholder=self.is_placeholder,
            ),
            MusicalEvent(
                tick=self.tick + self.duration,
                pitch=self.pitch,
                event_type=EventType.NOTE_OFF,
                velocity=0,
                channel=self.channel,
                program=self.program,
                is_placeholder=self.is_placeholder,
            ),
        ]

    @classmethod
    def from_events(cls, note_on: MusicalEvent, note_off: MusicalEvent) -> Note:
        """Create Note from note_on and note_off event pair."""
        if note_on.event_type != EventType.NOTE_ON:
            raise ValueError("First event must be NOTE_ON")
        if note_off.event_type != EventType.NOTE_OFF:
            raise ValueError("Second event must be NOTE_OFF")
        if note_on.pitch != note_off.pitch:
            raise ValueError("Events must have same pitch")
        if note_off.tick <= note_on.tick:
            raise ValueError("note_off tick must be after note_on tick")
        return cls(
            pitch=note_on.pitch,
            tick=note_on.tick,
            duration=note_off.tick - note_on.tick,
            velocity=note_on.velocity,
            channel=note_on.channel,
            program=note_on.program,
            is_placeholder=note_on.is_placeholder,
        )
