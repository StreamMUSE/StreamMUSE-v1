from __future__ import annotations

import json

from streammuse.domain.tasks import TaskViewEvent
from streammuse.infrastructure.task_view import (
    QueueTaskEventSink,
    TaskViewSnapshot,
    reduce_task_view_snapshot,
)


def event(kind: str, seq: int, **payload: object) -> TaskViewEvent:
    return TaskViewEvent(
        type=kind,  # type: ignore[arg-type]
        session_id="session",
        event_seq=seq,
        payload=dict(payload),
    )


def test_reducer_projects_complete_session_and_ignores_stale_events() -> None:
    snapshot = TaskViewSnapshot(session_id="session", task="animal_naming")
    snapshot = reduce_task_view_snapshot(
        snapshot,
        event(
            "session_config",
            1,
            task="animal_naming",
            deadline_mode="challenge",
            deadline_ms=3000.0,
            challenge_deadline_ms_list=[3000.0, 1000.0],
        ),
    )
    snapshot = reduce_task_view_snapshot(
        snapshot,
        event(
            "turn_attempt_started",
            2,
            turn_id=0,
            actor="human",
            prompt="Name one unused animal:",
            display_value=None,
            deadline_ms=3000.0,
            started_server_ms=1000.0,
        ),
    )
    snapshot = reduce_task_view_snapshot(
        snapshot,
        event(
            "asr",
            3,
            turn_id=0,
            raw_transcript="Tiger.",
            canonical_response="tiger",
            parse_status="ok",
        ),
    )
    snapshot = reduce_task_view_snapshot(
        snapshot,
        event(
            "turn_finished",
            4,
            turn_id=0,
            actor="human",
            response="tiger",
            is_valid=True,
            deadline_missed=False,
            stats={"turn_count": 1, "valid_count": 1, "invalid_count": 0, "deadline_miss_count": 0},
        ),
    )
    snapshot = reduce_task_view_snapshot(
        snapshot,
        event("stage_changed", 5, stage_index=1, old_deadline_ms=3000.0, new_deadline_ms=1000.0),
    )
    snapshot = reduce_task_view_snapshot(
        snapshot,
        event("session_finished", 6, stop_reason="completed", winner=None, loser=None, stats=snapshot.stats),
    )

    assert snapshot.status == "finished"
    assert snapshot.responses["human"]["response"] == "tiger"
    assert snapshot.asr["raw_transcript"] == "Tiger."
    assert snapshot.stage["new_deadline_ms"] == 1000.0
    assert reduce_task_view_snapshot(snapshot, event("asr", 5, raw_transcript="stale")) is snapshot


def test_snapshot_envelope_is_json_safe_and_reports_reconnect_elapsed_time() -> None:
    snapshot = TaskViewSnapshot(
        session_id="session",
        task="zip_zap_zop",
        current_turn={
            "active": True,
            "deadline_ms": 1000.0,
            "started_server_ms": 500.0,
            "value": float("nan"),
        },
    )
    envelope = snapshot.envelope(now_ms=750.0)
    assert envelope["payload"]["current_turn"]["elapsed_ms_at_snapshot"] == 250.0
    assert envelope["payload"]["current_turn"]["remaining_ms_at_snapshot"] == 750.0
    assert envelope["payload"]["current_turn"]["value"] is None
    json.dumps(envelope, allow_nan=False)


def test_queue_is_bounded_and_overflow_replaces_backlog_with_latest_snapshot() -> None:
    sink = QueueTaskEventSink(
        session_id="session",
        task="zip_zap_zop",
        capacity=2,
        monotonic=lambda: 1.0,
    )
    subscription, initial = sink.subscribe()
    assert initial["payload"]["status"] == "waiting_for_game"

    for seq in range(1, 6):
        sink.emit(event("asr", seq, raw_transcript=str(seq)))

    pending = sink.drain(subscription, limit=10)
    assert len(pending) <= 2
    assert pending[-1]["event_seq"] == 5
    assert pending[-1]["payload"]["dropped_event_count"] > 0
    assert sink.snapshot.asr["raw_transcript"] == "5"
