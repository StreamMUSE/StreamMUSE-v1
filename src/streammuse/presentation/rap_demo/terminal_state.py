"""Immutable terminal-monitor state projected from canonical rap events."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Any

from streammuse.application.rap.monitoring_payloads import bounded_chunk_event_payload
from streammuse.domain.rap import RapEvent, RapEventType


_CHUNK_EVENTS = frozenset(
    {
        RapEventType.CHUNK_REQUEST_SUBMITTED,
        RapEventType.CHUNK_REMOTE_COMPLETED,
        RapEventType.CHUNK_REMOTE_REJECTED,
        RapEventType.CHUNK_COMMITTED,
        RapEventType.CHUNK_FALLBACK_ACTIVATED,
    }
)
_COORDINATOR_EVENTS = frozenset(
    {
        RapEventType.AUDIO_RENDER_STARTED,
        RapEventType.AUDIO_RENDER_COMPLETED,
        RapEventType.BAR_AUDIO_READY,
        RapEventType.BAR_AUDIO_COMMITTED,
        RapEventType.PRONUNCIATION_FALLBACK,
        RapEventType.TIMING_PRESSURE,
        RapEventType.FORCED_BAR_FIT,
        RapEventType.SYNTHESIS_FAILED,
    }
)


@dataclass(frozen=True)
class TerminalRapBarView:
    bar: int
    topic: str | None
    flow: Mapping[str, Any] | None
    text: str | None
    source: str | None
    fallback: bool | None
    fallback_reason: str | None
    total_score: float | None
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
    audio: Mapping[str, Any]
    audio_warnings: tuple[Mapping[str, Any], ...]
    recent_events: tuple[TerminalRapEventView, ...]
    remote_chunk: Mapping[str, Any] | None = None
    coordinator_epoch: int | None = None


class TerminalRapStateProjector:
    """Keep a bounded presentation projection without changing canonical state."""

    def __init__(self, *, history_limit: int = 32, candidate_limit: int | None = None) -> None:
        if history_limit <= 0 or (candidate_limit is not None and candidate_limit <= 0):
            raise ValueError("projector limits must be positive")
        self._lock = Lock()
        self._history_limit = history_limit
        self._candidate_limit = candidate_limit or history_limit
        self._session_id: str | None = None
        self._coordinator_epoch: int | None = None
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
        self._audio: dict[str, Any] = {
            "state": "disabled",
            "current_bar": None,
            "render_state": "idle",
            "render_bar": None,
            "queue_depth": 0,
            "buffered_seconds": 0.0,
            "underruns": 0,
            "device": None,
            "recording_path": None,
            "absolute_frame": None,
        }
        self._audio_warnings: deque[Mapping[str, Any]] = deque(maxlen=128)
        self._recent_events: deque[TerminalRapEventView] = deque(maxlen=history_limit)
        self._remote_chunk: Mapping[str, Any] | None = None

    def __call__(self, event: RapEvent) -> TerminalRapViewState:
        return self.apply(event)

    @property
    def state(self) -> TerminalRapViewState:
        with self._lock:
            return self._snapshot()

    def apply(self, event: RapEvent) -> TerminalRapViewState:
        with self._lock:
            self._session_id = event.session_id
            kind = event.event_type
            if self._is_stale_coordinator_event(event):
                return self._snapshot()
            event_view = _event_view(event)
            self._recent_events.append(event_view)
            if kind == RapEventType.SESSION_STARTED:
                self._session_metadata = _freeze({**self._session_metadata, **event.payload})
                self._stopped = False
                self._update_audio(event)
                playback_state = event.payload.get("playback_state")
                if playback_state in {"priming", "running"}:
                    self._audio["state"] = playback_state
            elif kind == RapEventType.SESSION_STOPPED:
                self._stopped = True
                if self._audio["state"] != "disabled":
                    self._audio["state"] = "stopped"
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
            elif kind in _CHUNK_EVENTS:
                self._remote_chunk = _freeze(
                    {
                        "sequence": event.sequence,
                        "event_type": event.event_type.value,
                        "bar": event.bar,
                        "tick": event.tick,
                        "request_id": event.request_id,
                        **bounded_chunk_event_payload(event.payload),
                    }
                )
            elif kind == RapEventType.TICK:
                self._current_bar = event.bar
                self._current_tick = event.tick
            elif kind == RapEventType.SYLLABLE_EMITTED:
                self._current_syllable = _freeze({"bar": event.bar, "tick": event.tick, **event.payload})
            elif kind == RapEventType.AUDIO_RENDER_COMPLETED:
                self._update_audio_render(event, state="rendering")
            elif kind == RapEventType.BAR_AUDIO_READY:
                self._update_audio_render(event, state="ready")
            elif kind == RapEventType.BAR_AUDIO_COMMITTED:
                self._update_audio_render(event, state="committed")
            elif kind in (RapEventType.BAR_PLAYBACK_STARTED, RapEventType.BAR_PLAYBACK_COMPLETED):
                self._update_audio(event, state="running")
            elif kind == RapEventType.STOP_REQUESTED:
                self._update_audio(event, state="stop_requested")
            elif kind == RapEventType.SESSION_RESET:
                self._reset_epoch(event, event_view)
            elif kind in (
                RapEventType.PRONUNCIATION_FALLBACK,
                RapEventType.TIMING_PRESSURE,
                RapEventType.FORCED_BAR_FIT,
                RapEventType.SYNTHESIS_FAILED,
            ):
                self._remember_audio_warning(event)
            elif kind == RapEventType.AUDIO_UNDERRUN:
                self._update_audio(event)
                self._audio["underruns"] += 1
                self._remember_audio_warning(event)
            elif kind == RapEventType.AUDIO_DEVICE_FAILED:
                self._update_audio(event, state="failed")
                self._remember_audio_warning(event)
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
            total_score=_number_value(payload.get("total_score")) if "total_score" in payload else (old.total_score if old else None),
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
            audio=_freeze(self._audio),
            audio_warnings=tuple(self._audio_warnings),
            recent_events=tuple(self._recent_events),
            remote_chunk=self._remote_chunk,
            coordinator_epoch=self._coordinator_epoch,
        )

    def _update_audio(
        self,
        event: RapEvent,
        *,
        state: str | None = None,
        update_current_bar: bool = True,
    ) -> None:
        if state is not None:
            self._audio["state"] = state
        if update_current_bar and event.bar is not None:
            self._audio["current_bar"] = event.bar
        for key in ("queue_depth", "buffered_seconds", "absolute_frame"):
            value = event.payload.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._audio[key] = value
        for source_key, target_key in (
            ("device", "device"),
            ("output_device", "device"),
            ("audio_device", "device"),
            ("recording_path", "recording_path"),
        ):
            value = event.payload.get(source_key)
            if isinstance(value, str):
                self._audio[target_key] = value
        configuration = event.payload.get("audio")
        if isinstance(configuration, Mapping):
            device = configuration.get("audio_device")
            if isinstance(device, str):
                self._audio["device"] = device
            artifacts = configuration.get("artifact_paths")
            if isinstance(artifacts, Mapping) and isinstance(artifacts.get("wav"), str):
                self._audio["recording_path"] = artifacts["wav"]

    def _update_audio_render(self, event: RapEvent, *, state: str) -> None:
        self._update_audio(event, update_current_bar=False)
        self._audio["render_state"] = state
        if event.bar is not None:
            self._audio["render_bar"] = event.bar

    def _remember_audio_warning(self, event: RapEvent) -> None:
        self._audio_warnings.append(_freeze({"bar": event.bar, "tick": event.tick, "type": event.event_type.value, **event.payload}))

    def _reset_epoch(self, event: RapEvent, event_view: TerminalRapEventView) -> None:
        device = self._audio["device"]
        recording_path = self._audio["recording_path"]
        self._coordinator_epoch = _coordinator_epoch(event.payload)
        self._current_bar = None
        self._current_tick = None
        self._bars.clear()
        self._latest_request = None
        self._active_request_id = None
        self._latest_batch = None
        self._candidates.clear()
        self._current_syllable = None
        self._last_error = None
        self._stopped = True
        self._audio = {
            "state": "stopped",
            "current_bar": None,
            "render_state": "idle",
            "render_bar": None,
            "queue_depth": 0,
            "buffered_seconds": 0.0,
            "underruns": 0,
            "device": device,
            "recording_path": recording_path,
            "absolute_frame": None,
        }
        self._audio_warnings.clear()
        self._remote_chunk = None
        self._recent_events.clear()
        self._recent_events.append(event_view)

    def _is_stale_coordinator_event(self, event: RapEvent) -> bool:
        event_epoch = _coordinator_epoch(event.payload)
        return (
            event.event_type in _COORDINATOR_EVENTS
            and self._coordinator_epoch is not None
            and event_epoch is not None
            and event_epoch < self._coordinator_epoch
        )


def _coordinator_epoch(payload: Mapping[str, Any]) -> int | None:
    value = payload.get("coordinator_epoch")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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
    payload = (
        bounded_chunk_event_payload(event.payload)
        if event.event_type in _CHUNK_EVENTS
        else event.payload
    )
    return TerminalRapEventView(
        event.sequence,
        event.event_type,
        event.bar,
        event.tick,
        event.request_id,
        _freeze(payload),
    )


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


def _number_value(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
