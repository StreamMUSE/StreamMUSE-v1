"""Trace comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from streammuse.domain.debug.trace import DebugTraceEvent


def compare_trace_events(
    left: Iterable[DebugTraceEvent],
    right: Iterable[DebugTraceEvent],
) -> dict[str, Any]:
    left_events = _last_event_by_stage(list(left))
    right_events = _last_event_by_stage(list(right))
    right_by_stage = {event.stage: event for event in right_events}
    stage_results: list[dict[str, Any]] = []
    first_mismatch_stage: str | None = None
    matching_stage_count = 0

    for left_event in left_events:
        right_event = right_by_stage.get(left_event.stage)
        hashes_match = right_event is not None and left_event.output_hash == right_event.output_hash
        if hashes_match and first_mismatch_stage is None:
            matching_stage_count += 1
        elif first_mismatch_stage is None:
            first_mismatch_stage = left_event.stage
        stage_results.append(
            {
                "stage": left_event.stage,
                "match": bool(hashes_match),
                "left_hash": left_event.output_hash,
                "right_hash": right_event.output_hash if right_event else None,
                "left_summary": left_event.summary,
                "right_summary": right_event.summary if right_event else None,
                "diff_summary": _diff_summary(left_event, right_event, bool(hashes_match)),
                "left_refs": [ref.to_dict() for ref in left_event.output_refs],
                "right_refs": [ref.to_dict() for ref in right_event.output_refs] if right_event else [],
            }
        )

    return {
        "schema_version": 1,
        "left_runner": left_events[0].runner_kind if left_events else None,
        "right_runner": right_events[0].runner_kind if right_events else None,
        "first_mismatch_stage": first_mismatch_stage,
        "matching_stage_count": matching_stage_count,
        "stage_count": len(stage_results),
        "stages": stage_results,
    }


def _last_event_by_stage(events: list[DebugTraceEvent]) -> list[DebugTraceEvent]:
    order: list[str] = []
    by_stage: dict[str, DebugTraceEvent] = {}
    for event in events:
        if event.stage not in by_stage:
            order.append(event.stage)
        by_stage[event.stage] = event
    return [by_stage[stage] for stage in order]


def _diff_summary(
    left_event: DebugTraceEvent,
    right_event: DebugTraceEvent | None,
    hashes_match: bool,
) -> dict[str, Any]:
    if right_event is None:
        return {
            "status": "missing_right",
            "message": "Realtime simulation did not emit this checkpoint",
            "changed_fields": [],
        }
    if hashes_match:
        return {
            "status": "match",
            "message": "Canonical checkpoint output matched",
            "changed_fields": [],
        }

    changed_fields = _changed_summary_fields(left_event.summary, right_event.summary)
    if changed_fields:
        first = changed_fields[0]
        message = (
            f"{first['field']} changed: "
            f"offline={first['left']}, realtime={first['right']}"
        )
    else:
        message = "Canonical hashes differ; inspect the linked artifacts"
    return {
        "status": "different",
        "message": message,
        "changed_fields": changed_fields,
    }


def _changed_summary_fields(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    fields = [
        "event_count",
        "length",
        "other_length",
        "first_mismatch",
        "kind",
        "status",
        "ready",
        "generated_count",
        "playable_count",
    ]
    changed: list[dict[str, Any]] = []
    for field in fields:
        left_value = left.get(field)
        right_value = right.get(field)
        if left_value != right_value:
            changed.append({"field": field, "left": left_value, "right": right_value})
    return changed


def load_trace(path: Path) -> list[DebugTraceEvent]:
    events: list[DebugTraceEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(DebugTraceEvent.from_dict(json.loads(line)))
    return events
