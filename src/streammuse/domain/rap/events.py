"""Canonical events for the terminal-only realtime rap showcase."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RapEventType(str, Enum):
    SESSION_STARTED = "session_started"
    SESSION_STOPPED = "session_stopped"
    BAR_RESERVED = "bar_reserved"
    BAR_PLANNING_STARTED = "bar_planning_started"
    CANDIDATE_BATCH_RECEIVED = "candidate_batch_received"
    CANDIDATE_EVALUATED = "candidate_evaluated"
    GENERATION_FAILED = "generation_failed"
    BAR_REPLACED = "bar_replaced"
    BAR_FROZEN = "bar_frozen"
    FALLBACK_ACTIVATED = "fallback_activated"
    TICK = "tick"
    SYLLABLE_EMITTED = "syllable_emitted"
    PRESENTATION_ERROR = "presentation_error"


@dataclass(frozen=True)
class RapEvent:
    session_id: str
    sequence: int
    event_type: RapEventType
    utc_time: str
    monotonic_ns: int
    bar: int | None
    tick: int | None
    request_id: str | None
    payload: dict[str, Any]
