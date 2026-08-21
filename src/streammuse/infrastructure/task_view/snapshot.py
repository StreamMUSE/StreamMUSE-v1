"""Server-side projection of task view events into a reconnectable snapshot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from streammuse.domain.tasks import TaskViewEvent
from streammuse.infrastructure.serialization import json_safe


@dataclass(frozen=True)
class TaskViewSnapshot:
    session_id: str
    task: str
    schema_version: int = 1
    event_seq: int = 0
    status: str = "waiting_for_game"
    config: dict[str, Any] = field(default_factory=dict)
    current_turn: dict[str, Any] | None = None
    stage: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(
        default_factory=lambda: {
            "turn_count": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "deadline_miss_count": 0,
        }
    )
    responses: dict[str, Any] = field(
        default_factory=lambda: {"human": None, "llm": None}
    )
    asr: dict[str, Any] | None = None
    speech_output: dict[str, Any] | None = None
    session_result: dict[str, Any] | None = None
    dropped_event_count: int = 0

    def envelope(self, *, now_ms: float | None = None) -> dict[str, Any]:
        payload = asdict(self)
        current_turn = payload.get("current_turn")
        if isinstance(current_turn, dict):
            started_ms = current_turn.pop("started_server_ms", None)
            deadline_ms = current_turn.get("deadline_ms")
            if (
                current_turn.get("active")
                and now_ms is not None
                and isinstance(started_ms, (int, float))
                and isinstance(deadline_ms, (int, float))
            ):
                elapsed_ms = max(0.0, float(now_ms) - float(started_ms))
                current_turn["elapsed_ms_at_snapshot"] = elapsed_ms
                current_turn["remaining_ms_at_snapshot"] = max(
                    0.0, float(deadline_ms) - elapsed_ms
                )
                current_turn["overrun_ms_at_snapshot"] = max(
                    0.0, elapsed_ms - float(deadline_ms)
                )
        return {
            "type": "snapshot",
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "event_seq": self.event_seq,
            "payload": json_safe(payload),
        }


def reduce_task_view_snapshot(
    snapshot: TaskViewSnapshot,
    event: TaskViewEvent,
) -> TaskViewSnapshot:
    """Apply one ordered event; stale, duplicate, or foreign events are ignored."""

    if event.session_id != snapshot.session_id or event.event_seq <= snapshot.event_seq:
        return snapshot

    payload = json_safe(event.payload)
    if not isinstance(payload, dict):
        payload = {}
    updated = replace(snapshot, event_seq=int(event.event_seq))

    if event.type == "session_config":
        return replace(
            updated,
            task=str(payload.get("task") or snapshot.task),
            status="running",
            config=dict(payload),
            stage={
                "stage_index": payload.get("stage_index"),
                "deadline_ms": payload.get("deadline_ms"),
                "stage_count": len(payload.get("challenge_deadline_ms_list") or []),
            },
        )
    if event.type == "turn_attempt_started":
        current = dict(payload)
        current["active"] = True
        return replace(
            updated,
            status="running",
            current_turn=current,
            asr=None if payload.get("actor") == "human" else snapshot.asr,
        )
    if event.type == "asr":
        return replace(updated, asr=dict(payload))
    if event.type == "speech_output":
        return replace(updated, speech_output=dict(payload))
    if event.type == "turn_finished":
        actor = str(payload.get("actor") or "")
        responses = dict(snapshot.responses)
        if actor in responses:
            responses[actor] = dict(payload)
        current = dict(snapshot.current_turn or {})
        current.update(
            {
                "active": False,
                "latency_ms": payload.get("latency_ms"),
                "deadline_missed": bool(payload.get("deadline_missed")),
            }
        )
        stats = payload.get("stats")
        return replace(
            updated,
            current_turn=current,
            responses=responses,
            stats=dict(stats) if isinstance(stats, dict) else snapshot.stats,
        )
    if event.type == "stage_changed":
        return replace(updated, stage=dict(payload))
    if event.type == "session_finished":
        stats = payload.get("stats")
        return replace(
            updated,
            status="finished",
            current_turn={**(snapshot.current_turn or {}), "active": False},
            stats=dict(stats) if isinstance(stats, dict) else snapshot.stats,
            session_result=dict(payload),
        )
    return updated
