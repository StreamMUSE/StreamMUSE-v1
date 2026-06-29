"""Structured debug trace events."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TraceStatus = Literal["ok", "skipped", "error"]


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    path: str
    hash: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DebugTraceEvent:
    run_id: str
    runner_kind: str
    scenario: str
    stage: str
    schema_version: int = 1
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_event_id: str | None = None
    status: TraceStatus = "ok"
    wall_time_ns: int = field(default_factory=time.time_ns)
    logical_tick: int | None = None
    logical_beat: float | None = None
    input_refs: list[ArtifactRef] = field(default_factory=list)
    output_refs: list[ArtifactRef] = field(default_factory=list)
    input_hash: str | None = None
    output_hash: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_refs"] = [ref.to_dict() for ref in self.input_refs]
        payload["output_refs"] = [ref.to_dict() for ref in self.output_refs]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DebugTraceEvent":
        data = dict(payload)
        data["input_refs"] = [ArtifactRef(**ref) for ref in data.get("input_refs", [])]
        data["output_refs"] = [ArtifactRef(**ref) for ref in data.get("output_refs", [])]
        return cls(**data)
