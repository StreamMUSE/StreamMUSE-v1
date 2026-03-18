"""JSON logging output sink."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from streammuse.domain.logging import EventType, InferenceEvent, LogEvent, MetricsCalculator
from streammuse.domain.musical import MusicalEvent


class JsonLoggerOutputSink:
    def __init__(self, session_dir: Path, inference_log_detail: str = "summary") -> None:
        self.session_dir = Path(session_dir)
        self.events_file = self.session_dir / "events.jsonl"
        self.inferences_file = self.session_dir / "inferences.json"
        self.inferences: list[InferenceEvent] = []
        self.metrics_calculator = MetricsCalculator()
        self.inference_counter = 0
        self.inference_log_detail = inference_log_detail

    def output_event(self, event: MusicalEvent, source: str) -> None:
        log_event = LogEvent(
            timestamp=time.time(),
            tick=event.tick,
            event_type=self._get_event_type(event),
            data={
                "pitch": event.pitch,
                "source": source,
                "velocity": getattr(event, "velocity", 80),
            },
        )
        self.metrics_calculator.add_event(log_event)
        self._append_jsonl(log_event.to_dict())

    def _get_event_type(self, event: MusicalEvent) -> EventType:
        event_type_str = event.event_type.value
        if event_type_str == "note_on":
            return EventType.NOTE_ON
        elif event_type_str == "note_off":
            return EventType.NOTE_OFF
        else:
            return EventType.STATUS

    def log_inference(
        self,
        request: Dict[str, Any],
        response: Dict[str, Any],
        latency_ms: float,
        server_process_ms: float,
    ) -> None:
        self.inference_counter += 1
        request_data = dict(request)
        request_data["log_detail"] = self.inference_log_detail
        response_data = dict(response)

        inference = InferenceEvent(
            inference_id=f"inf_{self.inference_counter:03d}",
            timestamp_request=request_data.get("timestamp", time.time()),
            timestamp_response=response_data.get("timestamp", time.time()),
            request_data=request_data,
            response_data=response_data,
            latency_ms=latency_ms,
            server_process_ms=server_process_ms,
        )
        self.inferences.append(inference)
        self.metrics_calculator.add_inference(inference)

    def _append_jsonl(self, obj: Dict[str, Any]) -> None:
        with open(self.events_file, "a") as f:
            f.write(json.dumps(obj) + "\n")

    def save_metrics(self, session_config: Dict[str, Any]) -> None:
        performance_json = self.metrics_calculator.generate_performance_json(session_config)
        stats_csv = self.metrics_calculator.generate_statistics_csv()

        performance_path = self.session_dir / "performance.json"
        with open(performance_path, "w") as f:
            json.dump(performance_json, f, indent=2)

        csv_path = self.session_dir / "statistics.csv"
        with open(csv_path, "w") as f:
            f.write(stats_csv)

    def save_inferences(self) -> None:
        inferences_data = [inf.to_dict() for inf in self.inferences]
        with open(self.inferences_file, "w") as f:
            json.dump(inferences_data, f, indent=2)

    def output_tick(self, tick: int, bar: int, beat: int) -> None:
        pass

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
        pass

    def output_status(self, state: str, message: str = "") -> None:
        pass

    def output_config(self, config: Dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        self.save_inferences()
