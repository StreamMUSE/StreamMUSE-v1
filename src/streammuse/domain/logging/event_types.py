"""Event types for logging subsystem."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict


class EventType(str, Enum):
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"
    TICK = "tick"
    STATUS = "status"


@dataclass(frozen=True)
class LogEvent:
    timestamp: float
    tick: int
    event_type: EventType
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "tick": self.tick,
            "type": self.event_type.value,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True)
class InferenceEvent:
    inference_id: str
    timestamp_request: float
    timestamp_response: float
    request_data: Dict[str, Any]
    response_data: Dict[str, Any]
    latency_ms: float
    server_process_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.inference_id,
            "timestamp_request": self.timestamp_request,
            "timestamp_response": self.timestamp_response,
            "request_data": self.request_data,
            "response_data": self.response_data,
            "latency_ms": self.latency_ms,
            "server_process_ms": self.server_process_ms,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
