"""Musical domain: events, notes, sequences."""

from streammuse.domain.musical.converters import events_to_notes
from streammuse.domain.musical.events import EventType, MusicalEvent, Note
from streammuse.domain.musical.sequence import MusicalSequence

__all__ = ["EventType", "MusicalEvent", "MusicalSequence", "Note", "events_to_notes"]
