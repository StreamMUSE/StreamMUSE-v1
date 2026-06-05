"""Shared request/event helpers for prompt-continuation inference.

The real tokenization bridge will live here once the prompt and continuation
models are wired. For now this module keeps the engine boundary explicit and
avoids leaking API payload mutation across backend/engine/model layers.
"""

from __future__ import annotations

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
