"""Output sink protocol (extended for Web/CLI)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from streammuse.domain.musical import MusicalEvent


class OutputSink(Protocol):
    """
    Protocol for musical output and UI updates.

    Core: output_event, close. Extended: output_tick, output_stats,
    output_status, output_config for Web/CLI (display, stats, state).
    Simple sinks (e.g. audio) can no-op the extended methods.
    """

    def output_event(self, event: MusicalEvent, source: str) -> None:
        """Output a musical event (e.g. note_on/note_off to audio or WebSocket)."""
        ...

    def output_tick(self, tick: int, bar: int, beat: int) -> None:
        """Notify current tick position (for display / metronome)."""
        ...

    def output_stats(
        self,
        hit_rate: Optional[float] = None,
        avg_backup_level: Optional[float] = None,
        round_trip_ms: Optional[float] = None,
        server_process_ms: Optional[float] = None,
        network_latency_ms: Optional[float] = None,
        total_hits: Optional[int] = None,
        total_ticks: Optional[int] = None,
    ) -> None:
        """Send stats update (e.g. for debug / performance mode)."""
        ...

    def output_status(self, state: str, message: str = "") -> None:
        """Send status (e.g. 'running', 'listening', 'stopped', 'error')."""
        ...

    def output_config(self, config: Dict[str, Any]) -> None:
        """Send config snapshot (e.g. to Web UI)."""
        ...

    def close(self) -> None:
        """Release the output sink."""
        ...
