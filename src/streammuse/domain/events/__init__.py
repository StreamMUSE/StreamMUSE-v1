"""Generic event model supporting multiple event types (musical, text, audio)."""

from streammuse.domain.events.generic import (
    AudioFramePayload,
    Event,
    EventKind,
    MusicalEventPayload,
    TextChunkPayload,
)

__all__ = [
    "Event",
    "EventKind",
    "MusicalEventPayload",
    "TextChunkPayload",
    "AudioFramePayload",
]
