"""WebSocket OutputSink adapter (queue-based, thread-safe)."""

from __future__ import annotations

import json
import queue
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from streammuse.domain.musical import EventType, MusicalEvent


@dataclass(frozen=True)
class WebSocketOutputConfig:
    include_timestamps: bool = True


class WebSocketOutputSink:
    """
    OutputSink that queues JSON messages for an async WebSocket server to consume.

    Matches the legacy approach in `archive/current_system/output_handlers/websocket_output.py`.
    """

    def __init__(self, config: WebSocketOutputConfig | None = None) -> None:
        self._config = config or WebSocketOutputConfig()
        self.message_queue: queue.Queue[str] = queue.Queue()

    def _enqueue(self, message: Dict[str, Any]) -> None:
        if self._config.include_timestamps:
            message.setdefault("timestamp", time.time())
        self.message_queue.put(json.dumps(message))

    def output_event(self, event: MusicalEvent, source: str) -> None:
        if event.is_placeholder:
            return
        if event.event_type == EventType.NOTE_ON and event.velocity > 0:
            self._enqueue(
                {
                    "type": "note",
                    "event": "on",
                    "pitch": event.pitch,
                    "velocity": event.velocity,
                    "tick": event.tick,
                    "duration": 0,
                    "source": source,
                }
            )
            return
        if event.event_type == EventType.NOTE_OFF:
            self._enqueue(
                {
                    "type": "note",
                    "event": "off",
                    "pitch": event.pitch,
                    "tick": event.tick,
                    "source": source,
                }
            )

    def output_tick(self, tick: int, bar: int, beat: int) -> None:
        self._enqueue({"type": "tick", "tick": tick, "bar": bar, "beat": beat})

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
        msg: Dict[str, Any] = {"type": "stats"}
        if hit_rate is not None:
            msg["hit_rate"] = hit_rate
        if avg_backup_level is not None:
            msg["avg_backup_level"] = avg_backup_level
        if round_trip_ms is not None:
            msg["round_trip_ms"] = round_trip_ms
        if server_process_ms is not None:
            msg["server_process_ms"] = server_process_ms
        if network_latency_ms is not None:
            msg["network_latency_ms"] = network_latency_ms
        if total_hits is not None:
            msg["total_hits"] = total_hits
        if total_ticks is not None:
            msg["total_ticks"] = total_ticks
        self._enqueue(msg)

    def output_status(self, state: str, message: str = "") -> None:
        self._enqueue({"type": "status", "state": state, "message": message})

    def output_config(self, config: Dict[str, Any]) -> None:
        self._enqueue({"type": "config", **config})

    def get_pending_messages(self) -> List[str]:
        messages: List[str] = []
        while True:
            try:
                messages.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return messages

    def close(self) -> None:
        return

