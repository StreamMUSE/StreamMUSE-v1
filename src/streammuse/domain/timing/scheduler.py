"""
Playback scheduler: thread-safe schedule of musical events by tick.

No framework dependencies.
"""

from threading import Lock
from typing import Dict, List

from streammuse.domain.musical import MusicalEvent


class PlaybackScheduler:
    """
    Thread-safe scheduler for musical events.

    Manages future events by tick and supports cancellation of stale events
    (e.g. when a new inference response replaces not-yet-played model output).
    """

    def __init__(self) -> None:
        self._schedule: Dict[int, List[MusicalEvent]] = {}
        self._lock = Lock()

    def schedule(self, event: MusicalEvent, tick: int) -> None:
        """Schedule event for playback at the given tick."""
        with self._lock:
            if tick not in self._schedule:
                self._schedule[tick] = []
            self._schedule[tick].append(event)

    def get_events_at_tick(self, tick: int) -> List[MusicalEvent]:
        """Return all events scheduled for this tick and remove them from the schedule."""
        with self._lock:
            return self._schedule.pop(tick, [])

    def clear_future_events(
        self,
        from_tick: int,
        source: str | None = None,
    ) -> None:
        """
        Remove events from from_tick onward.

        If source is provided, only remove events that have a matching
        'source' attribute (e.g. "model"); events without a source attribute
        are kept. If source is None, clear all events in the range.
        """
        with self._lock:
            ticks_to_consider = [t for t in self._schedule.keys() if t >= from_tick]
            for tick in ticks_to_consider:
                if source is None:
                    del self._schedule[tick]
                else:
                    self._schedule[tick] = [
                        e for e in self._schedule[tick]
                        if getattr(e, "source", None) != source
                    ]
                    if not self._schedule[tick]:
                        del self._schedule[tick]
