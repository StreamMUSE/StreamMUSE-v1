"""
Generic event model with tick-based timing.

All events (musical, text, audio) flow through the same tick-based timing system.
Tick is a core property of all events, enabling fixed-rate systems (BPM, ticks_per_beat)
to work uniformly across different event types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union

from streammuse.domain.musical.events import EventType


class EventKind(str, Enum):
    """Event type kinds."""

    MUSICAL = "musical"
    TEXT_CHUNK = "text_chunk"
    AUDIO_FRAME = "audio_frame"


@dataclass(frozen=True)
class MusicalEventPayload:
    """Payload for musical events (note_on/note_off)."""

    pitch: int
    event_type: EventType
    velocity: int = 100
    channel: int = 0
    program: int = 0  # MIDI program/instrument
    is_placeholder: bool = False

    def __post_init__(self) -> None:
        """Validate payload data."""
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"Invalid velocity: {self.velocity}")
        if self.is_placeholder:
            if self.pitch != -1:
                raise ValueError("Placeholder events must have pitch=-1")
        else:
            if not 0 <= self.pitch <= 127:
                raise ValueError(f"Invalid pitch: {self.pitch}")
        if not 0 <= self.channel <= 15:
            raise ValueError(f"Invalid channel: {self.channel}")
        if not 0 <= self.program <= 127:
            raise ValueError(f"Invalid program: {self.program}")


@dataclass(frozen=True)
class TextChunkPayload:
    """Payload for text chunk events (future LLM support)."""

    text: str
    token_count: int = 0
    is_complete: bool = True  # True if this is a complete chunk, False if streaming


@dataclass(frozen=True)
class AudioFramePayload:
    """Payload for audio frame events (future audio support)."""

    samples: bytes
    sample_rate: int = 44100
    channels: int = 1


# Union type for all payload types
EventPayload = Union[MusicalEventPayload, TextChunkPayload, AudioFramePayload]


@dataclass(frozen=True)
class Event:
    """
    Generic event with tick-based timing.

    All events (musical, text, audio) use the same tick-based timing system.
    Tick is a core property, enabling fixed-rate systems (BPM, ticks_per_beat)
    to work uniformly across different event types.
    """

    tick: int
    kind: EventKind
    payload: EventPayload

    def __post_init__(self) -> None:
        """Validate event data."""
        if self.tick < 0:
            raise ValueError(f"Invalid tick: {self.tick}")

    @property
    def is_musical(self) -> bool:
        """Check if this is a musical event."""
        return self.kind == EventKind.MUSICAL

    @property
    def is_text(self) -> bool:
        """Check if this is a text event."""
        return self.kind == EventKind.TEXT_CHUNK

    @property
    def is_audio(self) -> bool:
        """Check if this is an audio event."""
        return self.kind == EventKind.AUDIO_FRAME
