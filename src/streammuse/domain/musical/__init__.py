"""Musical domain: events, notes, converters."""

from streammuse.domain.musical.converters import events_to_notes
from streammuse.domain.musical.events import EventType, MusicalEvent, Note

__all__ = ["EventType", "MusicalEvent", "Note", "events_to_notes"]
