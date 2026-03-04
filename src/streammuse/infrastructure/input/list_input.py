"""In-memory list input adapter for tests and replay."""

from __future__ import annotations

from typing import Iterator, List

from streammuse.domain.musical import MusicalEvent


class ListInput:
    """
    InputSource that yields events from a fixed list.

    Useful for tests and for replaying a pre-recorded sequence.
    Tick on events is taken from the list; application may overwrite with timestamp-derived tick.
    """

    def __init__(self, events: List[MusicalEvent]) -> None:
        self._events = list(events)
        self._closed = False

    def read_events(self) -> Iterator[MusicalEvent]:
        """Yield each event from the list once."""
        if self._closed:
            return
        for ev in self._events:
            yield ev

    def close(self) -> None:
        """Mark as closed."""
        self._closed = True
