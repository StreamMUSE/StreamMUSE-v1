"""Composite OutputSink that fans out to multiple sinks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from streammuse.domain.musical import MusicalEvent


@dataclass(frozen=True)
class CompositeOutputSink:
    sinks: List[Any]

    def output_event(self, event: MusicalEvent, source: str) -> None:
        for s in self.sinks:
            s.output_event(event, source)

    def output_tick(self, tick: int, bar: int, beat: int) -> None:
        for s in self.sinks:
            s.output_tick(tick, bar, beat)

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
        for s in self.sinks:
            s.output_stats(
                hit_rate=hit_rate,
                avg_backup_level=avg_backup_level,
                round_trip_ms=round_trip_ms,
                server_process_ms=server_process_ms,
                network_latency_ms=network_latency_ms,
                total_hits=total_hits,
                total_ticks=total_ticks,
            )

    def output_status(self, state: str, message: str = "") -> None:
        for s in self.sinks:
            s.output_status(state, message)

    def output_config(self, config: Dict[str, Any]) -> None:
        for s in self.sinks:
            s.output_config(config)

    def close(self) -> None:
        for s in self.sinks:
            s.close()

