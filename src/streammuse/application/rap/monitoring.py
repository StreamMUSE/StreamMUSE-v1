"""Nonblocking publication and ordered dispatch of rap showcase events."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from queue import Empty, SimpleQueue
from threading import Lock, Thread
from time import monotonic_ns
from typing import Any, Callable

from streammuse.domain.rap import RapEvent, RapEventType


_SENTINEL = object()


class _RapEventQueue:
    """Queue facade that lets the dispatcher report sink failures canonically."""

    def __init__(self, publisher: "RapEventPublisher") -> None:
        self._publisher = publisher
        self._queue: SimpleQueue[RapEvent | object] = SimpleQueue()

    def put(self, item: RapEvent | object) -> None:
        self._queue.put(item)

    def get(self) -> RapEvent | object:
        return self._queue.get()

    def get_nowait(self) -> RapEvent | object:
        return self._queue.get_nowait()

    def publish_presentation_error(self, event: RapEvent, sink: Callable[[RapEvent], None], error: Exception) -> None:
        self._publisher._emit_presentation_error(
            bar=event.bar,
            tick=event.tick,
            request_id=event.request_id,
            payload={
                "sink": getattr(sink, "__name__", type(sink).__name__),
                "failed_event_type": event.event_type.value,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )

    def close_publication(self) -> None:
        self._publisher.close()


class RapEventPublisher:
    """Assign event identity once, then enqueue without presentation I/O."""

    def __init__(
        self,
        session_id: str,
        *,
        utc_now: Callable[[], str] | None = None,
        monotonic_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        self.session_id = session_id
        self.queue = _RapEventQueue(self)
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc).isoformat())
        self._monotonic_ns = monotonic_ns
        self._lock = Lock()
        self._sequence = 0
        self._closed = False

    def emit(
        self,
        event_type: RapEventType,
        *,
        bar: int | None = None,
        tick: int | None = None,
        request_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RapEvent:
        return self._emit(
            event_type,
            bar=bar,
            tick=tick,
            request_id=request_id,
            payload=payload,
        )

    def close(self) -> None:
        """Close external publication after all in-flight emitters have queued."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.queue.put(_SENTINEL)

    def _emit_presentation_error(
        self,
        *,
        bar: int | None,
        tick: int | None,
        request_id: str | None,
        payload: dict[str, Any],
    ) -> RapEvent:
        return self._emit(
            RapEventType.PRESENTATION_ERROR,
            bar=bar,
            tick=tick,
            request_id=request_id,
            payload=payload,
            internal=True,
        )

    def _emit(
        self,
        event_type: RapEventType,
        *,
        bar: int | None,
        tick: int | None,
        request_id: str | None,
        payload: dict[str, Any] | None,
        internal: bool = False,
    ) -> RapEvent:
        with self._lock:
            if self._closed and not internal:
                raise RuntimeError("rap event publisher is closed")
            self._sequence += 1
            event = RapEvent(
                session_id=self.session_id,
                sequence=self._sequence,
                event_type=event_type,
                utc_time=self._utc_now(),
                monotonic_ns=self._monotonic_ns(),
                bar=bar,
                tick=tick,
                request_id=request_id,
                payload=deepcopy(payload or {}),
            )
            # Sequence allocation and publication are one critical section so the
            # FIFO queue order is also canonical event order across producers.
            self.queue.put(event)
        return event


class RapEventDispatcher:
    """Fan one queue out to terminal-compatible sinks in sequence order."""

    def __init__(self, queue: Any, *, sinks: tuple[Callable[[RapEvent], None], ...]) -> None:
        self._queue = queue
        self._sinks = list(sinks)
        self._thread = Thread(target=self._run, name="streammuse-rap-events", daemon=True)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def flush_and_close(self) -> None:
        if not self._started:
            return
        close_publication = getattr(self._queue, "close_publication", None)
        if callable(close_publication):
            close_publication()
        else:
            self._queue.put(_SENTINEL)
        self._thread.join()
        self._started = False

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                # Presentation errors raised while draining can land behind the
                # close marker. Drain them before reporting a completed flush.
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except Empty:
                        return
                    if item is not _SENTINEL:
                        self._dispatch(item)

            self._dispatch(item)

    def _dispatch(self, item: RapEvent | object) -> None:
        assert isinstance(item, RapEvent)
        active: list[Callable[[RapEvent], None]] = []
        errors: list[tuple[Callable[[RapEvent], None], Exception]] = []
        for sink in self._sinks:
            try:
                sink(item)
            except Exception as error:
                errors.append((sink, error))
            else:
                active.append(sink)
        self._sinks = active

        if item.event_type == RapEventType.PRESENTATION_ERROR:
            return
        publish_error = getattr(self._queue, "publish_presentation_error", None)
        if callable(publish_error):
            for sink, error in errors:
                publish_error(item, sink, error)


class RapStateProjector:
    """Build a serializable consumer snapshot from canonical events only."""

    def __init__(
        self,
        *,
        max_recent_bars: int = 16,
        max_emitted_syllables: int = 128,
        max_candidates: int = 64,
    ) -> None:
        if min(max_recent_bars, max_emitted_syllables, max_candidates) <= 0:
            raise ValueError("projector limits must be positive")
        self._lock = Lock()
        self._max_recent_bars = max_recent_bars
        self._max_emitted_syllables = max_emitted_syllables
        self._max_candidates = max_candidates
        self._segments: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._state: dict[str, Any] = {
            "session_id": None,
            "last_sequence": 0,
            "current_tick": None,
            "current_segment": None,
            "pending_request": None,
            "candidates": OrderedDict(),
            "frozen_bars": OrderedDict(),
            "emitted_syllables": [],
            "latencies": {
                "generation_latency_ms": self._aggregate(),
                "deadline_slack_ms": self._aggregate(),
                "emission_jitter_ms": self._aggregate(),
            },
            "fallbacks": {"count": 0, "by_reason": {}},
        }

    def __call__(self, event: RapEvent) -> None:
        self.apply(event)

    def apply(self, event: RapEvent) -> None:
        with self._lock:
            self._state["session_id"] = event.session_id
            self._state["last_sequence"] = event.sequence
            self._remember_segment(event)

            if event.event_type == RapEventType.TICK:
                self._state["current_tick"] = event.tick
                if event.bar is not None and str(event.bar) in self._segments:
                    self._state["current_segment"] = deepcopy(self._segments[str(event.bar)])
            elif event.event_type == RapEventType.BAR_PLANNING_STARTED:
                self._state["candidates"].clear()
                self._state["pending_request"] = self._event_state(event)
            elif event.event_type == RapEventType.CANDIDATE_BATCH_RECEIVED:
                if self._state["pending_request"] and self._state["pending_request"]["request_id"] == event.request_id:
                    self._state["pending_request"] = None
                self._add_payload_number(event.payload, "latency_ms", "generation_latency_ms")
                self._add_payload_number(event.payload, "deadline_slack_ms", "deadline_slack_ms")
            elif event.event_type == RapEventType.CANDIDATE_EVALUATED:
                candidate_id = event.payload.get("candidate_id")
                if isinstance(candidate_id, str):
                    self._state["candidates"][candidate_id] = self._event_state(event)
                    self._state["candidates"].move_to_end(candidate_id)
                    self._trim_mapping(self._state["candidates"], self._max_candidates)
            elif event.event_type == RapEventType.BAR_FROZEN and event.bar is not None:
                self._state["frozen_bars"][str(event.bar)] = self._event_state(event)
                self._state["frozen_bars"].move_to_end(str(event.bar))
                self._trim_mapping(self._state["frozen_bars"], self._max_recent_bars)
            elif event.event_type == RapEventType.GENERATION_FAILED:
                if self._state["pending_request"] and self._state["pending_request"]["request_id"] == event.request_id:
                    self._state["pending_request"] = None
            elif event.event_type == RapEventType.FALLBACK_ACTIVATED:
                fallbacks = self._state["fallbacks"]
                fallbacks["count"] += 1
                reason = event.payload.get("fallback_reason")
                if isinstance(reason, str):
                    fallbacks["by_reason"][reason] = fallbacks["by_reason"].get(reason, 0) + 1
            elif event.event_type == RapEventType.SYLLABLE_EMITTED:
                syllable = {"bar": event.bar, "tick": event.tick, **deepcopy(event.payload)}
                self._state["emitted_syllables"].append(syllable)
                del self._state["emitted_syllables"][:-self._max_emitted_syllables]
                self._add_payload_number(event.payload, "jitter_ms", "emission_jitter_ms")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    @staticmethod
    def _aggregate() -> dict[str, int | float | None]:
        return {"count": 0, "total": 0.0, "min": None, "max": None}

    @staticmethod
    def _event_state(event: RapEvent) -> dict[str, Any]:
        return {
            "sequence": event.sequence,
            "bar": event.bar,
            "tick": event.tick,
            "request_id": event.request_id,
            **deepcopy(event.payload),
        }

    def _remember_segment(self, event: RapEvent) -> None:
        if event.bar is None:
            return
        topic = event.payload.get("topic")
        template_id = event.payload.get("template_id")
        if isinstance(topic, str) or isinstance(template_id, str):
            self._segments[str(event.bar)] = {
                "bar": event.bar,
                "topic": topic if isinstance(topic, str) else None,
                "template_id": template_id if isinstance(template_id, str) else None,
            }
            self._segments.move_to_end(str(event.bar))
            self._trim_mapping(self._segments, self._max_recent_bars + 16)

    @staticmethod
    def _trim_mapping(mapping: OrderedDict[str, Any], limit: int) -> None:
        while len(mapping) > limit:
            mapping.popitem(last=False)

    def _add_payload_number(self, payload: dict[str, Any], payload_key: str, aggregate_key: str) -> None:
        value = payload.get(payload_key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        aggregate = self._state["latencies"][aggregate_key]
        aggregate["count"] += 1
        aggregate["total"] += value
        aggregate["min"] = value if aggregate["min"] is None else min(aggregate["min"], value)
        aggregate["max"] = value if aggregate["max"] is None else max(aggregate["max"], value)
