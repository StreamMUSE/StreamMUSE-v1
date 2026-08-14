"""Nonblocking publication and ordered dispatch of rap showcase events."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from queue import Empty, SimpleQueue
from threading import Lock, Thread
from time import monotonic_ns
from typing import Any, Callable

from streammuse.domain.rap import RapEvent, RapEventType, normalize_text


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


class _CumulativeResearchMetrics:
    """Incrementally mirror recorder metrics without retaining the event stream."""

    def __init__(self) -> None:
        self._planned: set[str] = set()
        self._batches: dict[str, int | None] = {}
        self._candidates: set[tuple[str, str]] = set()
        self._errored: set[str] = set()
        self._frozen: set[int] = set()
        self._valid_candidates = 0
        self._parsed_candidates = 0
        self._fallback_bars = 0
        self._deadline_misses = 0
        self._pronunciation_fallbacks = 0
        self._pronunciation_total = 0
        self._repeated_bigrams = 0
        self._generated_bigrams = 0
        self._repetition_window = 4
        self._recent_bigrams: list[set[tuple[str, str]]] = []
        self._latencies: dict[str, list[float]] = {
            "generation_latency_ms": [],
            "deadline_slack_ms": [],
            "emission_jitter_ms": [],
            "synthesis_latency_ms": [],
            "bar_render_latency_ms": [],
            "audio_commit_slack_ms": [],
        }

    def apply(self, event: RapEvent) -> dict[str, Any]:
        request_id = event.request_id if isinstance(event.request_id, str) else None
        payload = event.payload
        if event.event_type == RapEventType.SESSION_STARTED:
            window = payload.get("repetition_window_bars")
            if isinstance(window, int) and not isinstance(window, bool) and window > 0:
                self._repetition_window = window
        elif event.event_type == RapEventType.BAR_PLANNING_STARTED and request_id is not None:
            self._planned.add(request_id)
        elif event.event_type == RapEventType.CANDIDATE_BATCH_RECEIVED and request_id is not None:
            if request_id not in self._batches:
                candidate_count = payload.get("candidate_count")
                declared = (
                    candidate_count
                    if isinstance(candidate_count, int) and not isinstance(candidate_count, bool) and candidate_count >= 0
                    else None
                )
                self._batches[request_id] = declared
                if declared is not None:
                    self._parsed_candidates += declared
                if request_id in self._planned and payload.get("late") is True:
                    self._deadline_misses += 1
                self._remember_number(payload.get("latency_ms"), "generation_latency_ms")
                self._remember_number(payload.get("deadline_slack_ms"), "deadline_slack_ms")
                if request_id in self._planned and payload.get("error_type"):
                    self._errored.add(request_id)
        elif event.event_type == RapEventType.CANDIDATE_EVALUATED and request_id in self._batches:
            candidate_id = payload.get("candidate_id")
            identity = (request_id, candidate_id) if isinstance(candidate_id, str) else None
            if identity is not None and identity not in self._candidates:
                self._candidates.add(identity)
                if self._batches[request_id] is None:
                    self._parsed_candidates += 1
                if payload.get("valid") is True:
                    self._valid_candidates += 1
                self._remember_pronunciations(payload.get("word_analysis_sources"))
        elif event.event_type == RapEventType.GENERATION_FAILED and request_id in self._planned:
            self._errored.add(request_id)
        elif event.event_type == RapEventType.BAR_FROZEN and event.bar is not None and event.bar not in self._frozen:
            self._frozen.add(event.bar)
            if payload.get("fallback") is True:
                self._fallback_bars += 1
            self._remember_repetition(payload.get("text"))
        elif event.event_type == RapEventType.SYLLABLE_EMITTED:
            self._remember_number(payload.get("jitter_ms"), "emission_jitter_ms")
        elif event.event_type == RapEventType.AUDIO_RENDER_COMPLETED:
            self._remember_number(payload.get("synthesis_latency_ms"), "synthesis_latency_ms")
            self._remember_number(payload.get("render_latency_ms"), "bar_render_latency_ms")
        elif event.event_type == RapEventType.BAR_AUDIO_COMMITTED:
            self._remember_number(payload.get("deadline_slack_ms"), "audio_commit_slack_ms")
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "metrics": {
                "candidate_validity": self._ratio(self._valid_candidates, self._parsed_candidates),
                "fallback": self._ratio(self._fallback_bars, len(self._frozen)),
                "deadline_miss": self._ratio(self._deadline_misses, len(self._planned)),
                "generator_error": self._ratio(len(self._errored), len(self._planned)),
                "pronunciation_fallback": self._ratio(
                    self._pronunciation_fallbacks,
                    self._pronunciation_total,
                ),
                "repetition": self._ratio(self._repeated_bigrams, self._generated_bigrams),
            },
            "latencies": {
                name: self._distribution(values)
                for name, values in self._latencies.items()
            },
        }

    def _remember_pronunciations(self, value: object) -> None:
        if not isinstance(value, list):
            return
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("source"), str):
                continue
            self._pronunciation_total += 1
            if item["source"] != "cmudict_first_pronunciation":
                self._pronunciation_fallbacks += 1

    def _remember_repetition(self, value: object) -> None:
        words = normalize_text(value).split() if isinstance(value, str) else []
        bigrams = list(zip(words, words[1:]))
        prior = set().union(*self._recent_bigrams) if self._recent_bigrams else set()
        self._generated_bigrams += len(bigrams)
        self._repeated_bigrams += sum(bigram in prior for bigram in bigrams)
        self._recent_bigrams.append(set(bigrams))
        del self._recent_bigrams[:-self._repetition_window]

    def _remember_number(self, value: object, name: str) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self._latencies[name].append(float(value))

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": numerator / denominator if denominator else None,
        }

    @classmethod
    def _distribution(cls, values: list[float]) -> dict[str, int | float | None]:
        ordered = sorted(values)
        if not ordered:
            return {"count": 0, "p50": None, "p95": None, "max": None}
        return {
            "count": len(ordered),
            "p50": cls._percentile(ordered, 0.50),
            "p95": cls._percentile(ordered, 0.95),
            "max": ordered[-1],
        }

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)


class RapStateProjector:
    """Build a serializable consumer snapshot from canonical events only."""

    def __init__(
        self,
        *,
        max_recent_bars: int = 16,
        max_emitted_syllables: int = 128,
        max_candidates: int = 64,
        max_recent_events: int = 128,
    ) -> None:
        if min(max_recent_bars, max_emitted_syllables, max_candidates, max_recent_events) <= 0:
            raise ValueError("projector limits must be positive")
        self._lock = Lock()
        self._max_recent_bars = max_recent_bars
        self._max_emitted_syllables = max_emitted_syllables
        self._max_candidates = max_candidates
        self._max_recent_events = max_recent_events
        self._segments: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._latest_request_id: str | None = None
        self._research_metrics = _CumulativeResearchMetrics()
        self._state: dict[str, Any] = {
            "session_id": None,
            "last_sequence": 0,
            "current_tick": None,
            "current_playback": None,
            "current_syllable": None,
            "current_segment": None,
            "pending_request": None,
            "latest_request": None,
            "latest_batch": None,
            "last_error": None,
            "session_metadata": {},
            "stopped": False,
            "recent_events": [],
            "candidates": OrderedDict(),
            "bars": OrderedDict(),
            "frozen_bars": OrderedDict(),
            "emitted_syllables": [],
            "latencies": {
                "generation_latency_ms": self._aggregate(),
                "deadline_slack_ms": self._aggregate(),
                "emission_jitter_ms": self._aggregate(),
                "synthesis_latency_ms": self._aggregate(),
                "bar_render_latency_ms": self._aggregate(),
                "audio_commit_slack_ms": self._aggregate(),
            },
            "audio": {
                "state": "disabled",
                "current_bar": None,
                "queue_depth": 0,
                "buffered_seconds": 0.0,
                "underruns": 0,
                "device": None,
                "recording_path": None,
                "absolute_frame": None,
            },
            "audio_warnings": [],
            "fallbacks": {"count": 0, "by_reason": {}},
            "research_metrics": self._research_metrics.snapshot(),
        }

    def __call__(self, event: RapEvent) -> None:
        self.apply(event)

    def apply(self, event: RapEvent) -> None:
        with self._lock:
            self._state["session_id"] = event.session_id
            self._state["last_sequence"] = event.sequence
            self._state["recent_events"].append(self._canonical_event_state(event))
            del self._state["recent_events"][:-self._max_recent_events]
            self._remember_segment(event)

            if event.event_type == RapEventType.SESSION_STARTED:
                self._state["session_metadata"] = deepcopy(event.payload)
                self._state["stopped"] = False
                self._state["last_error"] = None
                if event.payload.get("playback_state") in {"priming", "running"}:
                    self._state["audio"]["state"] = event.payload["playback_state"]
            elif event.event_type == RapEventType.SESSION_STOPPED:
                self._state["stopped"] = True
                if self._state["audio"]["state"] != "disabled":
                    self._state["audio"]["state"] = "stopped"
            elif event.event_type == RapEventType.BAR_RESERVED:
                self._merge_bar(event, frozen=False, ignore_if_frozen=True)
            elif event.event_type == RapEventType.TICK:
                self._state["current_tick"] = event.tick
                self._state["current_playback"] = self._event_state(event)
                if event.bar is not None and str(event.bar) in self._segments:
                    self._state["current_segment"] = deepcopy(self._segments[str(event.bar)])
                    self._prune_past_segments(event.bar)
            elif event.event_type == RapEventType.BAR_PLANNING_STARTED:
                self._state["candidates"].clear()
                self._latest_request_id = event.request_id if isinstance(event.request_id, str) else None
                self._state["pending_request"] = self._event_state(event)
                self._state["latest_request"] = self._event_state(event)
                self._state["latest_batch"] = None
                self._merge_bar(event)
            elif event.event_type == RapEventType.CANDIDATE_BATCH_RECEIVED:
                if self._state["pending_request"] and self._state["pending_request"]["request_id"] == event.request_id:
                    self._state["pending_request"] = None
                if self._latest_request_id is not None and event.request_id == self._latest_request_id:
                    self._state["latest_batch"] = self._event_state(event)
                self._add_payload_number(event.payload, "latency_ms", "generation_latency_ms")
                self._add_payload_number(event.payload, "deadline_slack_ms", "deadline_slack_ms")
            elif event.event_type == RapEventType.CANDIDATE_EVALUATED:
                candidate_id = event.payload.get("candidate_id")
                if (
                    isinstance(candidate_id, str)
                    and self._latest_request_id is not None
                    and event.request_id == self._latest_request_id
                ):
                    self._state["candidates"][candidate_id] = self._event_state(event)
                    self._state["candidates"].move_to_end(candidate_id)
                    self._trim_mapping(self._state["candidates"], self._max_candidates)
            elif (
                event.event_type == RapEventType.BAR_REPLACED
                and self._latest_request_id is not None
                and event.request_id == self._latest_request_id
            ):
                self._merge_bar(event, ignore_if_frozen=True)
            elif event.event_type == RapEventType.BAR_FROZEN and event.bar is not None:
                self._merge_bar(event, frozen=True)
                self._state["frozen_bars"][str(event.bar)] = deepcopy(self._state["bars"][str(event.bar)])
                self._state["frozen_bars"].move_to_end(str(event.bar))
                self._trim_mapping(self._state["frozen_bars"], self._max_recent_bars)
            elif event.event_type == RapEventType.GENERATION_FAILED:
                if self._state["pending_request"] and self._state["pending_request"]["request_id"] == event.request_id:
                    self._state["pending_request"] = None
                self._state["last_error"] = self._canonical_event_state(event)
            elif event.event_type == RapEventType.FALLBACK_ACTIVATED:
                self._merge_bar(event)
                fallbacks = self._state["fallbacks"]
                fallbacks["count"] += 1
                reason = event.payload.get("fallback_reason")
                if isinstance(reason, str):
                    fallbacks["by_reason"][reason] = fallbacks["by_reason"].get(reason, 0) + 1
            elif event.event_type == RapEventType.SYLLABLE_EMITTED:
                syllable = {"bar": event.bar, "tick": event.tick, **deepcopy(event.payload)}
                self._state["current_syllable"] = syllable
                self._state["emitted_syllables"].append(syllable)
                del self._state["emitted_syllables"][:-self._max_emitted_syllables]
                self._add_payload_number(event.payload, "jitter_ms", "emission_jitter_ms")
            elif event.event_type == RapEventType.AUDIO_RENDER_COMPLETED:
                self._update_audio(event, state="rendering")
                self._add_payload_number(event.payload, "synthesis_latency_ms", "synthesis_latency_ms")
                self._add_payload_number(event.payload, "render_latency_ms", "bar_render_latency_ms")
            elif event.event_type == RapEventType.BAR_AUDIO_READY:
                self._update_audio(event, state="ready")
            elif event.event_type == RapEventType.BAR_AUDIO_COMMITTED:
                self._update_audio(event, state="priming")
                self._add_payload_number(event.payload, "render_latency_ms", "bar_render_latency_ms")
                self._add_payload_number(event.payload, "deadline_slack_ms", "audio_commit_slack_ms")
            elif event.event_type == RapEventType.BAR_PLAYBACK_STARTED:
                self._update_audio(event, state="running")
            elif event.event_type == RapEventType.BAR_PLAYBACK_COMPLETED:
                self._update_audio(event, state="running")
            elif event.event_type == RapEventType.STOP_REQUESTED:
                self._update_audio(event, state="stop_requested")
            elif event.event_type == RapEventType.SESSION_RESET:
                self._state["audio"].update(
                    {"state": "stopped", "current_bar": None, "queue_depth": 0, "buffered_seconds": 0.0, "underruns": 0, "absolute_frame": None}
                )
                self._state["audio_warnings"].clear()
            elif event.event_type in (RapEventType.PRONUNCIATION_FALLBACK, RapEventType.TIMING_PRESSURE):
                self._remember_audio_warning(event)
            elif event.event_type == RapEventType.AUDIO_UNDERRUN:
                self._update_audio(event)
                self._state["audio"]["underruns"] += 1
                self._remember_audio_warning(event)
            elif event.event_type == RapEventType.AUDIO_DEVICE_FAILED:
                self._update_audio(event, state="failed")
                self._remember_audio_warning(event)
            elif event.event_type == RapEventType.PRESENTATION_ERROR:
                self._state["last_error"] = self._canonical_event_state(event)
            self._state["research_metrics"] = self._research_metrics.apply(event)

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

    @staticmethod
    def _canonical_event_state(event: RapEvent) -> dict[str, Any]:
        return {
            "session_id": event.session_id,
            "sequence": event.sequence,
            "event_type": event.event_type.value,
            "utc_time": event.utc_time,
            "monotonic_ns": event.monotonic_ns,
            "bar": event.bar,
            "tick": event.tick,
            "request_id": event.request_id,
            "payload": deepcopy(event.payload),
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

    def _merge_bar(
        self,
        event: RapEvent,
        *,
        frozen: bool | None = None,
        ignore_if_frozen: bool = False,
    ) -> None:
        if event.bar is None:
            return
        key = str(event.bar)
        existing = self._state["bars"].get(key)
        if ignore_if_frozen and existing and existing.get("frozen") is True:
            return
        bar_state = deepcopy(existing) if existing else {"bar": event.bar, "frozen": False}
        bar_state["sequence"] = event.sequence
        if event.tick is not None:
            bar_state["tick"] = event.tick
        if event.request_id is not None:
            bar_state["request_id"] = event.request_id
        bar_state.update(deepcopy(event.payload))
        if frozen is not None:
            bar_state["frozen"] = frozen
        self._state["bars"][key] = bar_state
        self._state["bars"].move_to_end(key)
        self._trim_mapping(self._state["bars"], self._max_recent_bars)

    def _prune_past_segments(self, current_bar: int) -> None:
        for bar_key in tuple(self._segments):
            if int(bar_key) < current_bar:
                del self._segments[bar_key]

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

    def _update_audio(self, event: RapEvent, *, state: str | None = None) -> None:
        audio = self._state["audio"]
        payload = event.payload
        if state is not None:
            audio["state"] = state
        if event.bar is not None:
            audio["current_bar"] = event.bar
        for key in ("queue_depth", "buffered_seconds", "absolute_frame"):
            value = payload.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                audio[key] = value
        for source_key, target_key in (("device", "device"), ("output_device", "device"), ("recording_path", "recording_path")):
            value = payload.get(source_key)
            if isinstance(value, str):
                audio[target_key] = value

    def _remember_audio_warning(self, event: RapEvent) -> None:
        warning = {"bar": event.bar, "tick": event.tick, "type": event.event_type.value, **deepcopy(event.payload)}
        self._state["audio_warnings"].append(warning)
        del self._state["audio_warnings"][:-128]
