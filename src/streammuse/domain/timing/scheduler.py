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

    def pop_future_events(
        self,
        from_tick: int,
        source: str | None = None,
    ) -> List[MusicalEvent]:
        """Remove and return events from from_tick onward.

        If source is provided, only events with matching source are removed.
        Returned events are ordered by scheduled tick and insertion order within
        each tick, matching normal playback order before any caller-side sorting.
        """
        removed: List[MusicalEvent] = []
        with self._lock:
            ticks_to_consider = sorted(t for t in self._schedule.keys() if t >= from_tick)
            for tick in ticks_to_consider:
                events = self._schedule.get(tick, [])
                if source is None:
                    removed.extend(events)
                    del self._schedule[tick]
                    continue

                kept: List[MusicalEvent] = []
                for event in events:
                    if event.source == source:
                        removed.append(event)
                    else:
                        kept.append(event)
                if kept:
                    self._schedule[tick] = kept
                else:
                    self._schedule.pop(tick, None)
        return removed

    def clear_future_events(
        self,
        from_tick: int,
        source: str | None = None,
    ) -> None:
        """
        Remove events from from_tick onward.

        If source is provided, only remove events with matching source
        (e.g. "model"). If source is None, clear all events in the range.
        """
        self.pop_future_events(from_tick=from_tick, source=source)
