"""Bounded, immutable user directions, not claims about generated or heard lyrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from collections.abc import Mapping

MAX_STORY_EVENTS = 32
MAX_STORY_TEXT_CHARS = 512
MAX_STORY_TEXT_BYTES = 512
MAX_STORY_EVENT_ID_CHARS = 64


class StoryConflict(ValueError):
    """An existing immutable event ID was reused with different content."""


def _text(value: object, name: str, limit: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be nonblank and at most {limit} characters")


@dataclass(frozen=True)
class StoryEvent:
    event_id: str
    sequence: int
    text: str
    received_time: float
    source: str

    def __post_init__(self) -> None:
        _text(self.event_id, "event_id", MAX_STORY_EVENT_ID_CHARS)
        _text(self.text, "story text", MAX_STORY_TEXT_CHARS)
        _text(self.source, "story source", 32)
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("story sequence must be a positive integer")
        if (isinstance(self.received_time, bool) or not isinstance(self.received_time, (int, float))
                or not isfinite(self.received_time) or self.received_time < 0):
            raise ValueError("story received_time must be finite and non-negative")

    def to_payload(self) -> dict[str, object]:
        return {"event_id": self.event_id, "sequence": self.sequence, "text": self.text,
                "received_time": self.received_time, "source": self.source}

    @classmethod
    def from_payload(cls, payload: object) -> StoryEvent:
        if not isinstance(payload, Mapping) or set(payload) != {
            "event_id", "sequence", "text", "received_time", "source"
        }:
            raise ValueError("invalid story event payload")
        return cls(**payload)


@dataclass(frozen=True)
class StoryContext:
    version: int = 0
    events: tuple[StoryEvent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple) or not all(isinstance(e, StoryEvent) for e in self.events):
            raise ValueError("story events must be an immutable tuple")
        if type(self.version) is not int or self.version != len(self.events):
            raise ValueError("story version must equal the event count")
        if len(self.events) > MAX_STORY_EVENTS:
            raise ValueError(f"story is limited to {MAX_STORY_EVENTS} events; reset for a new story")
        if tuple(e.sequence for e in self.events) != tuple(range(1, self.version + 1)):
            raise ValueError("story events must be in contiguous sequence order")
        if len({e.event_id for e in self.events}) != self.version:
            raise ValueError("story event_id values must be unique")
        if sum(len(e.text.encode("utf-8")) for e in self.events) > MAX_STORY_TEXT_BYTES:
            raise ValueError(f"story text budget is {MAX_STORY_TEXT_BYTES} UTF-8 bytes; reset for a new story")

    def with_event(self, event: StoryEvent) -> StoryContext:
        for existing in self.events:
            if existing.event_id == event.event_id:
                if existing.text != event.text or existing.source != event.source:
                    raise StoryConflict("event_id already belongs to a different direction")
                return self
        return StoryContext(self.version + 1, (*self.events, event))

    def to_payload(self) -> dict[str, object]:
        return {"version": self.version, "events": [e.to_payload() for e in self.events]}

    @classmethod
    def from_payload(cls, payload: object) -> StoryContext:
        if (not isinstance(payload, Mapping) or set(payload) != {"version", "events"}
                or not isinstance(payload["events"], list)):
            raise ValueError("invalid story context payload")
        if len(payload["events"]) > MAX_STORY_EVENTS:
            raise ValueError(f"story is limited to {MAX_STORY_EVENTS} events")
        return cls(payload["version"], tuple(StoryEvent.from_payload(e) for e in payload["events"]))
