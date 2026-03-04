"""JSON serialization helpers for HTTP inference adapters."""

from __future__ import annotations

from typing import Any, Dict

from streammuse.domain.interfaces import TimingInfo
from streammuse.domain.musical import EventType, MusicalEvent


def event_to_dict(event: MusicalEvent) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "type": event.event_type.value,
        "pitch": int(event.pitch),
        "tick": int(event.tick),
    }
    # The legacy server only requires type/pitch/tick. Keep these optional for
    # forward/backward compatibility.
    if event.velocity is not None:
        d["velocity"] = int(event.velocity)
    if event.channel is not None:
        d["channel"] = int(event.channel)
    if event.program is not None:
        d["program"] = int(event.program)
    if event.is_placeholder:
        d["is_placeholder"] = True
    return d


def event_from_dict(d: Dict[str, Any]) -> MusicalEvent:
    return MusicalEvent(
        tick=int(d.get("tick", 0)),
        pitch=int(d.get("pitch", -1)),
        event_type=EventType(str(d.get("type"))),
        velocity=int(d.get("velocity", 100)),
        channel=int(d.get("channel", 0)),
        program=int(d.get("program", 0)),
        is_placeholder=bool(d.get("is_placeholder", False)),
    )


def timing_info_from_dict(d: Dict[str, Any]) -> TimingInfo:
    # Server fields are required by our TimingInfo type
    return TimingInfo(
        request_arrival_time=float(d["request_arrival_time"]),
        response_output_time=float(d["response_output_time"]),
        preprocess_start_time=float(d["preprocess_start_time"]),
        inference_start_time=float(d["inference_start_time"]),
        inference_end_time=float(d["inference_end_time"]),
        postprocess_start_time=float(d["postprocess_start_time"]),
        # Optional client-side fields
        round_trip_time=(float(d["round_trip_time"]) if "round_trip_time" in d else None),
        server_processing_duration=(
            float(d["server_processing_duration"]) if "server_processing_duration" in d else None
        ),
        total_network_latency=(float(d["total_network_latency"]) if "total_network_latency" in d else None),
    )

