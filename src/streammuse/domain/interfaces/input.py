"""Input source protocol."""

from __future__ import annotations

from typing import Iterator, Protocol

from streammuse.domain.musical import MusicalEvent


class InputSource(Protocol):
    """Protocol for musical input sources (MIDI, keyboard, file, etc.)."""

    def read_events(self) -> Iterator[MusicalEvent]:
        """Read musical events from the source. May block or yield as events arrive."""
        ...

    def close(self) -> None:
        """Release the input source."""
        ...
