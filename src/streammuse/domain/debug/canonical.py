"""Canonical summaries and hashes for debugger artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def hash_jsonable(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_event_payloads(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for event in events:
        payloads.append(
            {
                "tick": int(event.get("tick", 0)),
                "pitch": int(event.get("pitch", -1)),
                "type": str(event.get("type", event.get("event_type", ""))),
                "velocity": int(event.get("velocity", 0)),
                "channel": int(event.get("channel", 0)),
                "program": int(event.get("program", 0)),
                "source": str(event.get("source", "")),
            }
        )
    payloads.sort(
        key=lambda e: (
            e["tick"],
            0 if e["type"] == "note_off" else 1,
            e["pitch"],
            e["channel"],
            e["program"],
            e["velocity"],
            e["source"],
        )
    )
    return payloads


def summarize_events(events: Iterable[dict[str, Any]], *, limit: int = 12) -> dict[str, Any]:
    canonical = canonical_event_payloads(events)
    return {
        "event_count": len(canonical),
        "hash": hash_jsonable(canonical),
        "first": canonical[:limit],
        "last": canonical[-limit:] if len(canonical) > limit else canonical,
    }


def summarize_token_sequence(tokens: Iterable[int], *, other: Iterable[int] | None = None) -> dict[str, Any]:
    left = [int(token) for token in tokens]
    summary: dict[str, Any] = {
        "length": len(left),
        "hash": hash_jsonable(left),
        "first_32": left[:32],
    }
    if other is None:
        return summary

    right = [int(token) for token in other]
    first_mismatch = None
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            first_mismatch = {"position": index, "left": left_token, "right": right_token}
            break
    if first_mismatch is None and len(left) != len(right):
        first_mismatch = {
            "position": min(len(left), len(right)),
            "left": left[min(len(left), len(right))] if len(left) > len(right) else None,
            "right": right[min(len(left), len(right))] if len(right) > len(left) else None,
        }
    summary["other_length"] = len(right)
    summary["first_mismatch"] = first_mismatch
    return summary
