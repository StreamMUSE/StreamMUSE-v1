"""Console OutputSink adapter (debug/CLI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from streammuse.domain.musical import MusicalEvent


@dataclass(frozen=True)
class ConsoleOutputConfig:
    show_events: bool = True
    show_ticks: bool = True
    show_stats: bool = True


class ConsoleOutputSink:
    def __init__(self, config: ConsoleOutputConfig | None = None) -> None:
        self._config = config or ConsoleOutputConfig()

    def output_event(self, event: MusicalEvent, source: str) -> None:
        if not self._config.show_events:
            return
        print(f"[event] source={source} tick={event.tick} type={event.event_type.value} pitch={event.pitch}")

    def output_tick(self, tick: int, bar: int, beat: int) -> None:
        if not self._config.show_ticks:
            return
        print(f"[tick] tick={tick} bar={bar} beat={beat}")

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
        if not self._config.show_stats:
            return
        print(
            "[stats] "
            f"hit_rate={hit_rate} avg_backup_level={avg_backup_level} "
            f"round_trip_ms={round_trip_ms} server_process_ms={server_process_ms} "
            f"network_latency_ms={network_latency_ms} total_hits={total_hits} total_ticks={total_ticks}"
        )

    def output_status(self, state: str, message: str = "") -> None:
        print(f"[status] state={state} message={message}")

    def output_config(self, config: Dict[str, Any]) -> None:
        print(f"[config] {config}")

    def close(self) -> None:
        return

