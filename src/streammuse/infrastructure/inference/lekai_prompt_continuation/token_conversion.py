"""Shared request/event helpers for prompt-continuation inference.

The real tokenization bridge will live here once the prompt and continuation
models are wired. For now this module keeps the engine boundary explicit and
avoids leaking API payload mutation across backend/engine/model layers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

EventPayload = dict[str, int | str]


@dataclass(frozen=True)
class PromptContinuationRequest:
    """Engine-level generation request detached from the HTTP schema."""

    melody_events: list[EventPayload]
    generation_start_tick: int
    generation_length_frames: int
    generation_interval_ticks: int
    prompt_length_ticks: Optional[int]
    inference_mode: str
    model_name: str
    checkpoint_path: Optional[str]


def copy_event(event: EventPayload) -> EventPayload:
    """Return a normalized copy so downstream engines do not mutate API payloads."""

    copied: EventPayload = {
        "type": str(event.get("type", "")),
        "pitch": int(event["pitch"]),
        "tick": int(event.get("tick", 0)),
    }
    if "velocity" in event:
        copied["velocity"] = int(event["velocity"])
    return copied


def copy_events(events: list[EventPayload]) -> list[EventPayload]:
    return [copy_event(event) for event in events]


def canonical_event_key(event: EventPayload) -> tuple[int, int, str, int]:
    """Return the stable event identity used across server/client diagnostics."""

    event_type = str(event.get("type", ""))
    raw_velocity = event.get("velocity")
    velocity = int(raw_velocity) if raw_velocity is not None else _default_velocity(event_type)
    return (
        int(event.get("tick", 0)),
        int(event.get("pitch", -1)),
        event_type,
        velocity,
    )


def _default_velocity(event_type: str) -> int:
    return 0 if str(event_type) == "note_off" else 100


def event_representation_summary(
    events: list[EventPayload],
    *,
    include_keys: bool = False,
) -> dict[str, object]:
    """Summarize a decoded event representation with a deterministic digest.

    The prompt-continuation server and HTTP client both use this function so we
    can tell whether the model-decoded playable history changed while crossing
    the HTTP/client boundary.
    """

    keys = sorted(canonical_event_key(event) for event in events)
    ticks = [int(event.get("tick", 0)) for event in events if int(event.get("pitch", -1)) != -1]
    digest_payload = json.dumps(keys, separators=(",", ":"), ensure_ascii=True)
    summary: dict[str, object] = {
        "event_count": len(events),
        "note_on_count": sum(1 for event in events if str(event.get("type", "")) == "note_on"),
        "note_off_count": sum(1 for event in events if str(event.get("type", "")) == "note_off"),
        "placeholder_count": sum(1 for event in events if int(event.get("pitch", -1)) == -1),
        "min_tick": min(ticks) if ticks else None,
        "max_tick": max(ticks) if ticks else None,
        "digest": hashlib.sha256(digest_payload.encode("utf-8")).hexdigest(),
        "first_keys": [list(key) for key in keys[:8]],
        "last_keys": [list(key) for key in keys[-8:]],
    }
    if include_keys:
        summary["event_keys"] = [list(key) for key in keys]
    return summary
