"""Immutable terminal-monitor state projected from canonical rap events."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Any

from streammuse.domain.rap import RapEvent, RapEventType


@dataclass(frozen=True)
class TerminalRapBarView:
    bar: int
    topic: str | None
    flow: Mapping[str, Any] | None
    text: str | None
    source: str | None
    fallback: bool | None
    fallback_reason: str | None
    frozen: bool
    scheduled_syllables: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class TerminalRapRequestView:
    request_id: str | None
    bar: int | None
    topic: str | None
    required_syllables: int | None
    candidate_count: int | None
    context_lines: tuple[str, ...]
    seed: int | None
    flow: Mapping[str, Any] | None


@dataclass(frozen=True)
class TerminalRapBatchView:
    request_id: str | None
    bar: int | None
    source: str | None
    prompt: tuple[Mapping[str, Any], ...]
    raw_response: str | None
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class TerminalRapCandidateView:
    request_id: str | None
    candidate_id: str | None
    text: str | None
    valid: bool | None
    selected: bool | None
    rejection_reasons: tuple[str, ...]
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class TerminalRapEventView:
    sequence: int
    event_type: RapEventType
    bar: int | None
    tick: int | None
    request_id: str | None
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class TerminalRapViewState:
    session_id: str | None
    session_metadata: Mapping[str, Any]
    current_bar: int | None
    current_tick: int | None
    bars: Mapping[int, TerminalRapBarView]
    latest_request: TerminalRapRequestView | None
    latest_batch: TerminalRapBatchView | None
    candidates: tuple[TerminalRapCandidateView, ...]
    current_syllable: Mapping[str, Any] | None
    last_error: TerminalRapEventView | None
    stopped: bool
    recent_events: tuple[TerminalRapEventView, ...]


class TerminalRapStateProjector:
    """Keep a bounded presentation projection without changing canonical state."""

    def __init__(self, *, history_limit: int = 32, candidate_limit: int | None = None) -> None:
        if history_limit <= 0 or (candidate_limit is not None and candidate_limit <= 0):
            raise ValueError("projector limits must be positive")
        self._lock = Lock()
        self._history_limit = history_limit
        self._candidate_limit = candidate_limit or history_limit
        self._session_id: str | None = None
        self._session_metadata: Mapping[str, Any] = MappingProxyType({})
        self._current_bar: int | None = None
        self._current_tick: int | None = None
        self._bars: OrderedDict[int, TerminalRapBarView] = OrderedDict()
        self._latest_request: TerminalRapRequestView | None = None
        self._active_request_id: str | None = None
        self._latest_batch: TerminalRapBatchView | None = None
        self._candidates: OrderedDict[str, TerminalRapCandidateView] = OrderedDict()
        self._current_syllable: Mapping[str, Any] | None = None
        self._last_error: TerminalRapEventView | None = None
        self._stopped = False
        self._recent_events: deque[TerminalRapEventView] = deque(maxlen=history_limit)

    def __call__(self, event: RapEvent) -> TerminalRapViewState:
        return self.apply(event)

    @property
    def state(self) -> TerminalRapViewState:
        with self._lock:
            return self._snapshot()

    def apply(self, event: RapEvent) -> TerminalRapViewState:
        with self._lock:
            self._session_id = event.session_id
            event_view = _event_view(event)
            self._recent_events.append(event_view)
            kind = event.event_type
            if kind == RapEventType.SESSION_STARTED:
                self._session_metadata = _freeze(event.payload)
                self._stopped = False
            elif kind == RapEventType.SESSION_STOPPED:
                self._stopped = True
            elif kind == RapEventType.BAR_RESERVED:
                self._update_bar(event, frozen=False, ignore_if_frozen=True)
            elif kind == RapEventType.BAR_PLANNING_STARTED:
                self._latest_request = _request_view(event)
                self._active_request_id = event.request_id if isinstance(event.request_id, str) else None
                self._latest_batch = None
                self._candidates.clear()
                self._update_bar(event)
            elif kind == RapEventType.CANDIDATE_BATCH_RECEIVED and self._matches_active_request(event):
                self._latest_batch = _batch_view(event)
            elif kind == RapEventType.CANDIDATE_EVALUATED and self._matches_active_request(event):
                candidate = _candidate_view(event)
                key = candidate.candidate_id or f"sequence-{event.sequence}"
                self._candidates[key] = candidate
                self._candidates.move_to_end(key)
                while len(self._candidates) > self._candidate_limit:
                    self._candidates.popitem(last=False)
            elif kind == RapEventType.BAR_REPLACED and self._matches_active_request(event):
                self._update_bar(event, ignore_if_frozen=True)
            elif kind == RapEventType.BAR_FROZEN:
                self._update_bar(event, frozen=True)
            elif kind == RapEventType.FALLBACK_ACTIVATED:
                self._update_bar(event)
            elif kind == RapEventType.TICK:
                self._current_bar = event.bar
                self._current_tick = event.tick
            elif kind == RapEventType.SYLLABLE_EMITTED:
                self._current_syllable = _freeze({"bar": event.bar, "tick": event.tick, **event.payload})
            elif kind in (RapEventType.GENERATION_FAILED, RapEventType.PRESENTATION_ERROR):
                self._last_error = event_view
            return self._snapshot()

    def _matches_active_request(self, event: RapEvent) -> bool:
        return self._active_request_id is not None and event.request_id == self._active_request_id

    def _update_bar(
        self,
        event: RapEvent,
        *,
        frozen: bool | None = None,
        ignore_if_frozen: bool = False,
    ) -> None:
        if event.bar is None:
            return
        old = self._bars.get(event.bar)
        if ignore_if_frozen and old is not None and old.frozen:
            return
        payload = event.payload
        flow = _mapping_value(payload.get("flow")) if "flow" in payload else (old.flow if old else None)
        if flow is None and old is not None:
            flow = old.flow
        scheduled = _mapping_sequence(payload.get("scheduled_syllables")) if "scheduled_syllables" in payload else None
        if scheduled is None:
            scheduled = old.scheduled_syllables if old else ()
        fallback = old.fallback if old else None
        if "fallback" in payload and isinstance(payload["fallback"], bool):
            fallback = payload["fallback"]
        fallback_reason = old.fallback_reason if old else None
        if "fallback_reason" in payload and (payload["fallback_reason"] is None or isinstance(payload["fallback_reason"], str)):
            fallback_reason = payload["fallback_reason"]
        view = TerminalRapBarView(
            bar=event.bar,
            topic=_string(payload.get("topic")) if "topic" in payload else (old.topic if old else None),
            flow=flow,
            text=_string(payload.get("text")) if "text" in payload else (old.text if old else None),
            source=_string(payload.get("source")) if "source" in payload else (old.source if old else None),
            fallback=fallback,
            fallback_reason=fallback_reason,
            frozen=frozen if frozen is not None else (old.frozen if old else False),
            scheduled_syllables=scheduled,
        )
        self._bars[event.bar] = view
        self._bars.move_to_end(event.bar)
        while len(self._bars) > self._history_limit:
            self._bars.popitem(last=False)

    def _snapshot(self) -> TerminalRapViewState:
        return TerminalRapViewState(
            session_id=self._session_id,
            session_metadata=self._session_metadata,
            current_bar=self._current_bar,
            current_tick=self._current_tick,
            bars=MappingProxyType(dict(self._bars)),
            latest_request=self._latest_request,
            latest_batch=self._latest_batch,
            candidates=tuple(self._candidates.values()),
            current_syllable=self._current_syllable,
            last_error=self._last_error,
            stopped=self._stopped,
            recent_events=tuple(self._recent_events),
        )


def _request_view(event: RapEvent) -> TerminalRapRequestView:
    payload = event.payload
    context = payload.get("context_lines", ())
    return TerminalRapRequestView(
        request_id=event.request_id,
        bar=event.bar,
        topic=_string(payload.get("topic")),
        required_syllables=_integer(payload.get("required_syllables")),
        candidate_count=_integer(payload.get("candidate_count")),
        context_lines=tuple(item for item in context if isinstance(item, str)) if isinstance(context, (list, tuple)) else (),
        seed=_integer(payload.get("seed")),
        flow=_mapping_value(payload.get("flow")),
    )


def _batch_view(event: RapEvent) -> TerminalRapBatchView:
    payload = _freeze(event.payload)
    prompt = event.payload.get("prompt", ())
    return TerminalRapBatchView(
        request_id=event.request_id,
        bar=event.bar,
        source=_string(event.payload.get("source")),
        prompt=_prompt_items(prompt),
        raw_response=_string(event.payload.get("raw_response")),
        payload=payload,
    )


def _candidate_view(event: RapEvent) -> TerminalRapCandidateView:
    payload = _freeze(event.payload)
    reasons = event.payload.get("rejection_reasons", ())
    return TerminalRapCandidateView(
        request_id=event.request_id,
        candidate_id=_string(event.payload.get("candidate_id")),
        text=_string(event.payload.get("text")),
        valid=event.payload.get("valid") if isinstance(event.payload.get("valid"), bool) else None,
        selected=event.payload.get("selected") if isinstance(event.payload.get("selected"), bool) else None,
        rejection_reasons=tuple(item for item in reasons if isinstance(item, str)) if isinstance(reasons, (list, tuple)) else (),
        payload=payload,
    )


def _event_view(event: RapEvent) -> TerminalRapEventView:
    return TerminalRapEventView(event.sequence, event.event_type, event.bar, event.tick, event.request_id, _freeze(event.payload))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    try:
        return deepcopy(value)
    except Exception:
        return repr(value)


def _mapping_value(value: object) -> Mapping[str, Any] | None:
    return _freeze(value) if isinstance(value, Mapping) else None


def _mapping_sequence(value: object) -> tuple[Mapping[str, Any], ...] | None:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, Mapping) for item in value):
        return None
    return tuple(_freeze(item) for item in value)


def _prompt_items(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(_freeze(item) for item in value if isinstance(item, Mapping))


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
