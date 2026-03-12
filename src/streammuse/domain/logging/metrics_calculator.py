"""Metrics calculation for logging."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from streammuse.domain.logging.event_types import EventType, InferenceEvent, LogEvent


class MetricsCalculator:
    def __init__(self) -> None:
        self.events: List[LogEvent] = []
        self.inferences: List[InferenceEvent] = []

    def add_event(self, event: LogEvent) -> None:
        self.events.append(event)

    def add_inference(self, inf: InferenceEvent) -> None:
        self.inferences.append(inf)

    def calculate_latency_stats(self) -> Dict[str, float]:
        if not self.inferences:
            return {}

        latencies = [inf.latency_ms for inf in self.inferences]
        sorted_latencies = sorted(latencies)

        latency_stats = {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
        }

        if len(latencies) > 1:
            latency_stats["std"] = statistics.stdev(latencies)

        if len(sorted_latencies) >= 20:
            idx_95 = int(len(sorted_latencies) * 0.95)
            latency_stats["p95"] = sorted_latencies[idx_95]
            idx_99 = int(len(sorted_latencies) * 0.99)
            latency_stats["p99"] = sorted_latencies[idx_99]

        return latency_stats

    def calculate_event_stats(self) -> Dict[str, Any]:
        user_events = sum(1 for e in self.events if "user" in e.data.get("source", ""))
        model_events = sum(1 for e in self.events if "model" in e.data.get("source", ""))
        inference_requests = sum(
            1 for e in self.events if e.event_type == EventType.INFERENCE_REQUEST
        )
        inference_responses = sum(
            1 for e in self.events if e.event_type == EventType.INFERENCE_RESPONSE
        )

        return {
            "total_events": len(self.events),
            "user_input_events": user_events,
            "model_output_events": model_events,
            "inference_requests": inference_requests,
            "successful_responses": min(inference_responses, inference_requests),
            "failed_responses": max(0, inference_requests - inference_responses),
            "hit_rate": 100.0 if inference_requests == 0 else (
                100.0 * min(inference_responses, inference_requests) / inference_requests
            ),
        }

    def calculate_music_stats(self) -> Dict[str, Any]:
        note_ons = [e for e in self.events if e.event_type == EventType.NOTE_ON]
        user_notes = [e for e in note_ons if e.data.get("source") == "user"]
        model_notes = [e for e in note_ons if e.data.get("source") == "model"]

        user_pitches = [e.data.get("pitch", 0) for e in user_notes if e.data.get("pitch")]
        model_pitches = [e.data.get("pitch", 0) for e in model_notes if e.data.get("pitch")]

        user_velocities = [e.data.get("velocity", 80) for e in user_notes]
        model_velocities = [e.data.get("velocity", 80) for e in model_notes]

        stats: Dict[str, Any] = {
            "total_notes_user": len(user_notes),
            "total_notes_model": len(model_notes),
            "total_notes_combined": len(user_notes) + len(model_notes),
        }

        if user_pitches:
            stats["pitch_range_user"] = [min(user_pitches), max(user_pitches)]
            if user_velocities:
                stats["average_velocity_user"] = sum(user_velocities) / len(user_velocities)

        if model_pitches:
            stats["pitch_range_model"] = [min(model_pitches), max(model_pitches)]
            if model_velocities:
                stats["average_velocity_model"] = sum(model_velocities) / len(model_velocities)

        if self.events:
            duration_ticks = self.events[-1].tick - self.events[0].tick if len(self.events) > 1 else 0
            stats["total_duration_ticks"] = duration_ticks

        return stats

    def generate_performance_json(self, session_config: Dict[str, Any]) -> Dict[str, Any]:
        latency_stats = self.calculate_latency_stats()
        event_stats = self.calculate_event_stats()
        music_stats = self.calculate_music_stats()

        server_process_times = [inf.server_process_ms for inf in self.inferences]
        server_stats: Dict[str, float] = {}
        if server_process_times:
            server_stats["mean"] = statistics.mean(server_process_times)
            if len(server_process_times) > 1:
                server_stats["std"] = statistics.stdev(server_process_times)

        return {
            "session_config": session_config,
            "timing_statistics": {
                "latency_ms": latency_stats,
                "server_process_ms": server_stats,
            },
            "event_statistics": event_stats,
            "music_analysis": music_stats,
        }

    def generate_statistics_csv(self) -> str:
        event_stats = self.calculate_event_stats()
        latency_stats = self.calculate_latency_stats()

        lines = ["metric,value"]
        for key, value in event_stats.items():
            lines.append(f"{key},{value}")

        for key, value in latency_stats.items():
            lines.append(f"{key},{value}")

        return "\n".join(lines)
