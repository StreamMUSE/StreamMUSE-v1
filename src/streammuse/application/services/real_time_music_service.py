"""Real-time orchestration service with inference worker."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Protocol

from streammuse.application.services.input_timing import stamp_user_input_event
from streammuse.domain.interfaces import InferenceEngine, InputSource, OutputSink
from streammuse.domain.interfaces.timing_info import TimingInfo
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import MusicalTime, PlaybackScheduler, Tempo


@dataclass(frozen=True)
class RealTimeServiceRuntime:
    session_start_time: float
    timeline_start_time: float


class RealTimeServiceState(str, Enum):
    """Explicit lifecycle states for production, drain, and shutdown."""

    STOPPED = "stopped"
    ACCEPTING_REQUESTS = "accepting_requests"
    DRAINING = "draining"


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    generation_start_tick: int
    melody_events: List[MusicalEvent]
    input_increment_digest: str
    input_cumulative_digest: str

    def __iter__(self):
        """Keep legacy two-value unpacking used by tests and local tools."""
        yield self.generation_start_tick
        yield self.melody_events


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    accompaniment_events: List[MusicalEvent]
    generation_start_tick: int
    response_metadata: dict[str, object] = field(default_factory=dict)

    def __iter__(self):
        """Keep legacy two-value unpacking used by tests and local tools."""
        yield self.accompaniment_events
        yield self.generation_start_tick


_INFERENCE_STOP = object()


EventKey = tuple[int, int, int]


class TickObserver(Protocol):
    """Optional lifecycle hook for features sharing the musical tick clock."""

    def start(self) -> None:
        """Prepare work required before the first musical tick."""

    def on_tick(self, tick: int) -> None:
        """Observe one absolute musical tick."""

    def close(self) -> None:
        """Release any optional resources without raising."""


@dataclass(frozen=True)
class ScheduledModelEvent:
    event: MusicalEvent
    scheduled_tick: int
    logical_tick: int
    policy: str


@dataclass
class ModelSchedulePlan:
    scheduled_events: list[ScheduledModelEvent] = field(default_factory=list)
    trace_rows: list[dict[str, object]] = field(default_factory=list)
    clamped_onset_count: int = 0
    dropped_past_note_count: int = 0
    orphan_note_off_count: int = 0
    forced_note_off_count: int = 0

    @property
    def scheduled_actual_event_count(self) -> int:
        return len(self.scheduled_events)


class RealTimeMusicService:
    """
    Real-time music generation service.

    Orchestrates input ingestion, inference generation, and output playback
    using a tick-based timing system. Runs three threads:
    - Input worker: Reads events from input source
    - Tick loop: Manages timing and schedules playback
    - Inference worker: Triggers generation and processes responses
    """

    def __init__(
        self,
        *,
        input_source: InputSource,
        inference_engine: InferenceEngine,
        output_sink: OutputSink,
        tempo: Tempo,
        scheduler: PlaybackScheduler,
        generation_interval_ticks: int = 2,
        generation_length_frames: int = 20,
        count_in_beats: int = 0,
        input_snap_forward_fraction: float = 0.0,
        tick_observer: TickObserver | None = None,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        drain_timeout_seconds: float = 10.0,
    ) -> None:
        if int(count_in_beats) < 0:
            raise ValueError("count_in_beats must be >= 0")
        if int(generation_interval_ticks) <= 0:
            raise ValueError("generation_interval_ticks must be > 0")
        if float(drain_timeout_seconds) <= 0:
            raise ValueError("drain_timeout_seconds must be > 0")
        self._input = input_source
        self._engine = inference_engine
        self._output = output_sink
        self._tempo = tempo
        self._scheduler = scheduler
        self._generation_interval_ticks = generation_interval_ticks
        self._generation_length_frames = generation_length_frames
        self._count_in_beats = int(count_in_beats)
        self._count_in_ticks = self._count_in_beats * int(self._tempo.ticks_per_beat)
        self._input_snap_forward_fraction = float(input_snap_forward_fraction)
        self._tick_observer = tick_observer
        self._tick_observer_closed = False
        self._now = now
        self._sleep = sleep
        self._monotonic = monotonic
        self._drain_timeout_seconds = float(drain_timeout_seconds)

        self._running = False
        self._state = RealTimeServiceState.STOPPED
        self._state_lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._worker_done = threading.Event()
        self._drain_done = threading.Event()
        self._sentinel_enqueued = False
        self._resources_closed = False
        self._validity_finalized = False
        self._drain_timed_out = False
        self._drain_reason: str | None = None
        self._analysis_end_tick: int | None = None
        self._last_input_note_off_tick: int | None = None
        self._request_cutoff_tick: int | None = None
        self._run_stop_tick: int | None = None
        # ``None`` keeps legacy/unbounded sessions out of the formal request-tick
        # contract. A list (including an empty one) is a frozen, independently
        # derived schedule for a bounded formal run.
        self._planned_generation_start_ticks: list[int] | None = None
        self._actual_generation_start_ticks: list[int] = []
        self._rejected_generation_start_ticks: list[int] = []
        self._initial_generation_history: list[MusicalEvent] = []
        self._last_processed_tick = -1
        self._runtime: RealTimeServiceRuntime | None = None
        self._input_thread: threading.Thread | None = None
        self._tick_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._event_q: queue.Queue[MusicalEvent] = queue.Queue()
        self._melody_history: List[MusicalEvent] = []
        self._melody_history_lock = threading.Lock()
        self._inference_request_queue: queue.Queue[object] = queue.Queue()
        self._inference_response_queue: queue.Queue[object] = queue.Queue()
        self._active_model_note_keys: set[EventKey] = set()

        self._lifecycle_session_id = f"rt-{uuid.uuid4().hex}"
        self._request_counter = 0
        self._request_records: dict[str, dict[str, object]] = {}
        self._lifecycle_lock = threading.RLock()
        self._cumulative_input_events: list[MusicalEvent] = []
        self._in_flight_request_ids: set[str] = set()
        self._operational_counts: dict[str, int] = {
            "stale_request_drops": 0,
            "late_events": 0,
            "clamped_onsets": 0,
            "dropped_model_events": 0,
            "orphan_note_offs": 0,
            "forced_note_offs": 0,
        }
        self._max_lateness_ticks = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def lifecycle_state(self) -> RealTimeServiceState:
        with self._state_lock:
            return self._state

    @staticmethod
    def _canonical_event_payload(events: List[MusicalEvent]) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for event in events:
            row: dict[str, object] = {
                "type": event.event_type.value,
                "pitch": int(event.pitch),
                "tick": int(event.tick),
                "velocity": int(event.velocity),
                "channel": int(event.channel),
                "program": int(event.program),
            }
            # Match infrastructure.inference.serialization.event_to_dict():
            # false placeholders are intentionally omitted on the wire.
            if event.is_placeholder:
                row["is_placeholder"] = True
            payload.append(row)
        return payload

    @classmethod
    def _digest_events(cls, events: List[MusicalEvent]) -> str:
        payload = json.dumps(
            cls._canonical_event_payload(events),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _log_lifecycle(
        self,
        event: str,
        *,
        request: InferenceRequest | None = None,
        **details: object,
    ) -> None:
        row: dict[str, object] = {
            "schema_version": 1,
            "timestamp": float(self._now()),
            "session_id": self._lifecycle_session_id,
            "event": str(event),
            "state": self.lifecycle_state.value,
        }
        if request is not None:
            row.update(
                {
                    "request_id": request.request_id,
                    "generation_start_tick": int(request.generation_start_tick),
                    "input_event_count": len(request.melody_events),
                    "input_increment_digest": request.input_increment_digest,
                    "input_cumulative_digest": request.input_cumulative_digest,
                }
            )
        row.update(details)
        logger = getattr(self._output, "log_request_lifecycle", None)
        if callable(logger):
            try:
                logger(row)
            except Exception as exc:
                self._output.output_status("error", f"Failed to log request lifecycle: {exc}")

    def _can_accept_request(self, generation_start_tick: int) -> bool:
        with self._state_lock:
            # STOPPED is accepted only for direct unit-level tick-loop calls that
            # predate the managed start()/stop() lifecycle.
            accepting = self._state == RealTimeServiceState.ACCEPTING_REQUESTS or (
                self._state == RealTimeServiceState.STOPPED and self._running
            )
            if not accepting:
                return False
            return self._request_cutoff_tick is None or int(generation_start_tick) <= int(
                self._request_cutoff_tick
            )

    def _register_request(
        self,
        generation_start_tick: int,
        melody_events: List[MusicalEvent],
    ) -> InferenceRequest:
        events = list(melody_events)
        with self._lifecycle_lock:
            self._actual_generation_start_ticks.append(int(generation_start_tick))
            self._request_counter += 1
            request_id = f"{self._lifecycle_session_id}:req_{self._request_counter:04d}"
            increment_digest = self._digest_events(events)
            self._cumulative_input_events.extend(events)
            cumulative_digest = self._digest_events(self._cumulative_input_events)
            request = InferenceRequest(
                request_id=request_id,
                generation_start_tick=int(generation_start_tick),
                melody_events=events,
                input_increment_digest=increment_digest,
                input_cumulative_digest=cumulative_digest,
            )
            self._request_records[request_id] = {
                "request_id": request_id,
                "generation_start_tick": int(generation_start_tick),
                "expected": True,
                "enqueued": True,
                "started": False,
                "succeeded": False,
                "failed": False,
                "processed": False,
                "stale_dropped": False,
                "empty_success": False,
                "input_event_count": len(events),
                "input_increment_digest": increment_digest,
                "input_cumulative_digest": cumulative_digest,
                "raw_token_digest": None,
                "token_decode_digest": None,
                "response_metadata_status": "pending",
            }
        self._log_lifecycle("expected", request=request)
        self._log_lifecycle("enqueued", request=request)
        return request

    def _enqueue_inference_request(
        self,
        generation_start_tick: int,
        melody_events: List[MusicalEvent],
    ) -> bool:
        if not self._can_accept_request(generation_start_tick):
            with self._lifecycle_lock:
                self._rejected_generation_start_ticks.append(int(generation_start_tick))
            self._log_lifecycle(
                "enqueue_rejected",
                generation_start_tick=int(generation_start_tick),
                reason=(
                    "request_cutoff"
                    if self._request_cutoff_tick is not None
                    and int(generation_start_tick) > int(self._request_cutoff_tick)
                    else "not_accepting_requests"
                ),
            )
            return False
        request = self._register_request(generation_start_tick, melody_events)
        self._inference_request_queue.put(request)
        return True

    def _normalize_request(self, item: object) -> InferenceRequest:
        if isinstance(item, InferenceRequest):
            return item
        if isinstance(item, tuple) and len(item) == 2:
            generation_start_tick, melody_events = item
            return self._register_request(int(generation_start_tick), list(melody_events))
        raise TypeError(f"Unsupported inference request item: {type(item).__name__}")

    def _normalize_response(self, item: object) -> InferenceResponse:
        if isinstance(item, InferenceResponse):
            return item
        if isinstance(item, tuple) and len(item) == 2:
            accompaniment_events, generation_start_tick = item
            with self._lifecycle_lock:
                candidates = [
                    record
                    for record in self._request_records.values()
                    if int(record["generation_start_tick"]) == int(generation_start_tick)
                    and not bool(record["processed"])
                ]
            request_id = (
                str(candidates[-1]["request_id"])
                if candidates
                else f"{self._lifecycle_session_id}:external_{int(generation_start_tick)}"
            )
            return InferenceResponse(
                request_id=request_id,
                accompaniment_events=list(accompaniment_events),
                generation_start_tick=int(generation_start_tick),
            )
        raise TypeError(f"Unsupported inference response item: {type(item).__name__}")

    def _update_request_record(self, request_id: str, **values: object) -> None:
        with self._lifecycle_lock:
            record = self._request_records.get(request_id)
            if record is not None:
                record.update(values)

    def _consume_response_metadata(self) -> dict[str, object]:
        consumer = getattr(self._engine, "consume_last_response_metadata", None)
        metadata: object = None
        if callable(consumer):
            metadata = consumer()
        elif hasattr(self._engine, "last_response_metadata"):
            metadata = getattr(self._engine, "last_response_metadata")
        if not isinstance(metadata, dict):
            return {"metadata_status": "metadata_unavailable"}

        normalized = dict(metadata)
        raw_tokens = normalized.get("raw_tokens")
        if isinstance(raw_tokens, list):
            structural_tokens = normalized.get("structural_tokens")
            canonical = json.dumps(
                {
                    "raw": raw_tokens,
                    "structural": structural_tokens if isinstance(structural_tokens, list) else [],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            computed_digest = hashlib.sha256(canonical).hexdigest()
            supplied_digest = normalized.get("raw_token_digest")
            normalized["raw_token_count"] = len(raw_tokens)
            normalized["raw_token_digest"] = str(supplied_digest or computed_digest)
            normalized["raw_token_digest_verified"] = (
                supplied_digest is None or str(supplied_digest) == computed_digest
            )
        token_decode_beats = normalized.get("token_decode_beats")
        token_decode_initial = normalized.get("token_decode_initial_active_pitches")
        if isinstance(token_decode_beats, list) and isinstance(token_decode_initial, list):
            canonical_decode = json.dumps(
                {
                    "initial_active_pitches": token_decode_initial,
                    "beats": token_decode_beats,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            computed_decode_digest = hashlib.sha256(canonical_decode).hexdigest()
            normalized["token_decode_digest_verified"] = (
                normalized.get("token_decode_digest") == computed_decode_digest
            )
        part0_tokens = normalized.get("part0_tokens")
        if isinstance(part0_tokens, list):
            canonical_part0 = json.dumps(
                part0_tokens, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            computed_part0_digest = hashlib.sha256(canonical_part0).hexdigest()
            normalized["part0_token_digest_verified"] = (
                normalized.get("part0_token_digest") == computed_part0_digest
            )
        normalized["metadata_status"] = "available"
        return normalized

    def _validate_response_metadata(
        self,
        *,
        request: InferenceRequest,
        metadata: dict[str, object],
        sent_input_increment_digest: str,
        empty_success: bool,
    ) -> dict[str, object]:
        metadata_required = callable(
            getattr(self._engine, "consume_last_response_metadata", None)
        ) or hasattr(self._engine, "last_response_metadata")
        available = metadata.get("metadata_status") == "available"
        if not metadata_required:
            return {
                "metadata_required": False,
                "metadata_contract_valid": True,
                "metadata_status": metadata.get("metadata_status", "metadata_unavailable"),
            }

        expected_session_id = getattr(self._engine, "_session_id", None)
        expected_session_epoch = getattr(self._engine, "_session_epoch", None)
        expected_seed = getattr(self._engine, "_effective_seed", None)
        engine_config = getattr(self._engine, "_config", None)
        expected_model_bpm = getattr(engine_config, "bpm", None)
        required_keys = {
            "request_id",
            "session_id",
            "session_epoch",
            "effective_seed",
            "generation_start_tick",
            "raw_tokens",
            "structural_tokens",
            "raw_token_digest",
            "token_decode_beats",
            "token_decode_initial_active_pitches",
            "token_decode_digest",
            "prompt_token_digest",
            "part0_tokens",
            "part0_token_digest",
            "part0_trace_available",
            "input_increment_digest",
            "input_cumulative_digest",
            "part0_roll_digest",
            "part0_roll_shape",
            "part0_roll_bytes_sha256",
            "part0_roll_start_tick",
            "part0_roll_end_tick",
            "effective_bpm",
            "output_event_digest",
            "empty_success",
            "context_start_tick",
        }
        part0_trace_available = metadata.get("part0_trace_available") is True
        checks: dict[str, bool] = {
            "metadata_available": available,
            "required_keys_present": required_keys.issubset(metadata),
            "request_id_present": bool(metadata.get("request_id")),
            "request_id_matches": metadata.get("request_id") == request.request_id,
            "generation_start_tick_matches": (
                metadata.get("generation_start_tick") is not None
                and int(metadata["generation_start_tick"]) == int(request.generation_start_tick)
            ),
            "input_increment_digest_matches": (
                metadata.get("input_increment_digest") == sent_input_increment_digest
            ),
            "input_cumulative_digest_matches": (
                metadata.get("input_cumulative_digest") == request.input_cumulative_digest
            ),
            "raw_token_digest_verified": bool(metadata.get("raw_token_digest_verified")),
            "token_decode_digest_verified": bool(
                metadata.get("token_decode_digest_verified")
            ),
            "part0_token_digest_verified": bool(
                metadata.get("part0_token_digest_verified")
            ),
            "part0_trace_availability_declared": isinstance(
                metadata.get("part0_trace_available"), bool
            ),
            "part0_trace_unavailable_is_empty": (
                part0_trace_available
                or (
                    metadata.get("part0_roll_digest") is None
                    and metadata.get("part0_roll_shape") == []
                    and metadata.get("part0_roll_bytes_sha256") is None
                    and metadata.get("part0_roll_start_tick") is None
                    and metadata.get("part0_roll_end_tick") is None
                    and metadata.get("context_start_tick") is None
                )
            ),
            "part0_roll_digest_present": (
                not part0_trace_available or bool(metadata.get("part0_roll_digest"))
            ),
            "part0_roll_shape_valid": (
                not part0_trace_available
                or (
                    isinstance(metadata.get("part0_roll_shape"), list)
                    and len(metadata["part0_roll_shape"]) == 3
                    and list(metadata["part0_roll_shape"][:2]) == [2, 88]
                    and int(metadata["part0_roll_shape"][2]) > 0
                )
            ),
            "part0_roll_window_valid": (
                not part0_trace_available
                or (
                    metadata.get("part0_roll_start_tick") is not None
                    and metadata.get("part0_roll_end_tick") is not None
                    and int(metadata["part0_roll_start_tick"])
                    == int(metadata.get("context_start_tick", -1))
                    and int(metadata["part0_roll_end_tick"])
                    > int(metadata["part0_roll_start_tick"])
                )
            ),
            "effective_bpm_matches": (
                expected_model_bpm is None
                or int(metadata.get("effective_bpm", -1)) == int(expected_model_bpm)
            ),
            "empty_success_matches": bool(metadata.get("empty_success")) == bool(empty_success),
            "session_id_matches": (
                expected_session_id is None or metadata.get("session_id") == expected_session_id
            ),
            "session_epoch_matches": (
                expected_session_epoch is None
                or metadata.get("session_epoch") == expected_session_epoch
            ),
            "effective_seed_matches": (
                expected_seed is None or metadata.get("effective_seed") == expected_seed
            ),
        }
        return {
            "metadata_required": True,
            "metadata_contract_valid": all(checks.values()),
            "metadata_status": metadata.get("metadata_status"),
            "metadata_checks": checks,
        }

    def _queue_inference_stop(self) -> None:
        with self._state_lock:
            if self._sentinel_enqueued:
                return
            self._sentinel_enqueued = True
        self._inference_request_queue.put(_INFERENCE_STOP)

    def _begin_draining(self, reason: str) -> None:
        transitioned = False
        with self._state_lock:
            if self._state == RealTimeServiceState.ACCEPTING_REQUESTS:
                self._state = RealTimeServiceState.DRAINING
                self._drain_reason = str(reason)
                transitioned = True
            elif self._state == RealTimeServiceState.STOPPED:
                return
        if transitioned:
            self._log_lifecycle(
                "drain_started",
                reason=str(reason),
                request_cutoff_tick=self._request_cutoff_tick,
                run_stop_tick=self._run_stop_tick,
            )
        self._queue_inference_stop()

    def _record_operational_plan(self, plan: ModelSchedulePlan) -> None:
        late_rows = [
            row
            for row in plan.trace_rows
            if row.get("logical_tick") is not None
            and int(row["logical_tick"]) < int(row.get("current_tick", row["logical_tick"]))
            and row.get("policy") != "forced_note_off"
        ]
        lateness = [
            int(row.get("current_tick", 0)) - int(row["logical_tick"])
            for row in late_rows
        ]
        with self._lifecycle_lock:
            self._operational_counts["late_events"] += len(late_rows)
            self._operational_counts["clamped_onsets"] += int(plan.clamped_onset_count)
            self._operational_counts["dropped_model_events"] += int(plan.dropped_past_note_count)
            self._operational_counts["orphan_note_offs"] += int(plan.orphan_note_off_count)
            self._operational_counts["forced_note_offs"] += int(plan.forced_note_off_count)
            if lateness:
                self._max_lateness_ticks = max(self._max_lateness_ticks, max(lateness))

    def _build_validity_summary(self) -> dict[str, object]:
        with self._lifecycle_lock:
            records = [dict(record) for record in self._request_records.values()]
            operational = dict(self._operational_counts)
            max_lateness = int(self._max_lateness_ticks)
            in_flight = sorted(self._in_flight_request_ids)
            planned_generation_ticks = (
                None
                if self._planned_generation_start_ticks is None
                else list(self._planned_generation_start_ticks)
            )
            actual_generation_ticks = list(self._actual_generation_start_ticks)
            rejected_generation_ticks = list(self._rejected_generation_start_ticks)

        request_tick_gate_enabled = planned_generation_ticks is not None
        planned_tick_counter = Counter(planned_generation_ticks or [])
        actual_tick_counter = Counter(actual_generation_ticks)
        missing_generation_ticks = sorted(
            (planned_tick_counter - actual_tick_counter).elements()
        )
        # A rejected request is an observed scheduling attempt even though the
        # cutoff gate correctly prevented it from becoming a registered request.
        unexpected_generation_ticks = sorted(
            list((actual_tick_counter - planned_tick_counter).elements())
            + rejected_generation_ticks
        )
        duplicate_generation_ticks = sorted(
            tick for tick, count in actual_tick_counter.items() if count > 1
        )
        generation_tick_order_valid = bool(
            not request_tick_gate_enabled
            or actual_generation_ticks == planned_generation_ticks
        )
        request_tick_contract_valid = bool(
            not request_tick_gate_enabled
            or (
                generation_tick_order_valid
                and not missing_generation_ticks
                and not unexpected_generation_ticks
                and not duplicate_generation_ticks
            )
        )

        expected = {str(record["request_id"]) for record in records if record["expected"]}
        succeeded = {str(record["request_id"]) for record in records if record["succeeded"]}
        processed = {str(record["request_id"]) for record in records if record["processed"]}
        failed = {str(record["request_id"]) for record in records if record["failed"]}
        stale = {str(record["request_id"]) for record in records if record["stale_dropped"]}
        metadata_invalid = {
            str(record["request_id"])
            for record in records
            if record.get("metadata_required") and not record.get("metadata_contract_valid")
        }
        pending = expected - processed - failed - stale
        content_valid = bool(
            expected == succeeded
            and expected == processed
            and not failed
            and not stale
            and not metadata_invalid
            and not pending
            and not in_flight
            and not self._drain_timed_out
            and request_tick_contract_valid
        )
        operational_valid = bool(content_valid and not any(operational.values()))
        if request_tick_gate_enabled:
            processed_tick_counter = Counter(
                int(record["generation_start_tick"])
                for record in records
                if record["processed"]
            )
            covered_tick_count = sum(
                (processed_tick_counter & planned_tick_counter).values()
            )
            coverage = (
                1.0
                if not planned_tick_counter
                else covered_tick_count / sum(planned_tick_counter.values())
            )
        else:
            coverage = 1.0 if not expected else len(processed & expected) / len(expected)
        empty_ids = sorted(
            str(record["request_id"])
            for record in records
            if record.get("empty_success")
        )
        return {
            "schema_version": 1,
            "session_id": self._lifecycle_session_id,
            "state": RealTimeServiceState.STOPPED.value,
            "tick_semantics": {
                "analysis_end_tick": self._analysis_end_tick,
                "analysis_end_exclusive": True,
                "last_input_note_off_tick": self._last_input_note_off_tick,
                "request_cutoff_tick": self._request_cutoff_tick,
                "request_cutoff_inclusive": True,
                "run_stop_tick": self._run_stop_tick,
                "run_stop_exclusive": True,
                "last_processed_tick": int(self._last_processed_tick),
            },
            "drain": {
                "reason": self._drain_reason,
                "timeout_seconds": float(self._drain_timeout_seconds),
                "timed_out": bool(self._drain_timed_out),
            },
            "content": {
                "valid": content_valid,
                "expected_request_ids": sorted(expected),
                "succeeded_request_ids": sorted(succeeded),
                "processed_request_ids": sorted(processed),
                "failed_request_ids": sorted(failed),
                "stale_dropped_request_ids": sorted(stale),
                "metadata_invalid_request_ids": sorted(metadata_invalid),
                "pending_at_stop_request_ids": sorted(pending | set(in_flight)),
                "analysis_request_coverage": coverage,
                "request_tick_gate_enabled": request_tick_gate_enabled,
                "request_tick_contract_valid": request_tick_contract_valid,
                "generation_start_tick_order_valid": generation_tick_order_valid,
                "planned_generation_start_ticks": planned_generation_ticks,
                "actual_generation_start_ticks": actual_generation_ticks,
                "missing_generation_start_ticks": missing_generation_ticks,
                "unexpected_generation_start_ticks": unexpected_generation_ticks,
                "duplicate_generation_start_ticks": duplicate_generation_ticks,
                "rejected_generation_start_ticks": sorted(rejected_generation_ticks),
                "input_digest_chain_complete": all(
                    bool(record.get("input_increment_digest"))
                    and bool(record.get("input_cumulative_digest"))
                    for record in records
                ),
                "empty_success": bool(empty_ids),
                "empty_success_request_ids": empty_ids,
                "request_count": len(expected),
            },
            "operational": {
                "valid": operational_valid,
                **operational,
                "max_lateness_ticks": max_lateness,
            },
            "requests": records,
        }

    def _finalize_validity(self) -> None:
        with self._state_lock:
            if self._validity_finalized:
                return
            self._validity_finalized = True
        summary = self._build_validity_summary()
        pending = summary["content"]["pending_at_stop_request_ids"]  # type: ignore[index]
        for request_id in pending:
            self._log_lifecycle("pending_at_stop", request_id=request_id)
        finalizer = getattr(self._output, "finalize_validity", None)
        if callable(finalizer):
            try:
                finalizer(summary)
            except Exception as exc:
                self._output.output_status("error", f"Failed to finalize validity: {exc}")

    def _output_metronome_tick(self, tick: int, bar: int, beat: int) -> None:
        if hasattr(self._output, "output_metronome_tick"):
            self._output.output_metronome_tick(tick, bar, beat)  # type: ignore[union-attr]

    def _timeline_start_time(self) -> float:
        assert self._runtime is not None
        return float(getattr(self._runtime, "timeline_start_time", self._runtime.session_start_time))

    def _sleep_until(self, target_time: float) -> None:
        delay = target_time - self._now()
        if delay > 0:
            self._sleep(delay)

    def _start_tick_observer(self) -> None:
        if self._tick_observer is None or self._tick_observer_closed:
            return
        try:
            self._tick_observer.start()
        except Exception as exc:
            self._output.output_status("error", f"Tick observer startup error: {exc}")
            self._tick_observer_closed = True

    def _close_tick_observer(self) -> None:
        if self._tick_observer is None or self._tick_observer_closed:
            return
        self._tick_observer_closed = True
        try:
            self._tick_observer.close()
        except Exception as exc:
            self._output.output_status("error", f"Tick observer shutdown error: {exc}")

    def _run_count_in(self) -> None:
        """Play metronome-only pre-roll before the musical timeline starts."""
        if self._count_in_ticks <= 0:
            return
        assert self._runtime is not None
        start = self._runtime.session_start_time
        ticks_per_beat = max(1, int(self._tempo.ticks_per_beat))
        beats_per_bar = max(1, int(self._tempo.beats_per_bar))

        self._output.output_status("count_in", f"{self._count_in_beats} beat(s)")
        for elapsed_tick in range(self._count_in_ticks):
            if not self._running or self._stop_requested.is_set():
                return
            target_time = start + self._tempo.tick_to_seconds(elapsed_tick)
            self._sleep_until(target_time)

            count_tick = elapsed_tick - self._count_in_ticks
            beat = (elapsed_tick // ticks_per_beat) % beats_per_bar
            bar = elapsed_tick // (ticks_per_beat * beats_per_bar)
            self._output_metronome_tick(tick=count_tick, bar=bar, beat=beat)

    def _event_to_log_dict(self, event: MusicalEvent) -> dict:
        return {
            "type": event.event_type.value,
            "pitch": int(event.pitch),
            "tick": int(event.tick),
            "velocity": int(event.velocity),
            "channel": int(event.channel),
            "program": int(event.program),
            "source": str(event.source),
            "is_placeholder": bool(event.is_placeholder),
            "backup_level": int(event.backup_level),
        }

    @staticmethod
    def _model_event_key(event: MusicalEvent) -> EventKey:
        return (int(event.pitch), int(event.channel), int(event.program))

    @staticmethod
    def _event_order_key(event: MusicalEvent) -> tuple[int, int, int, int, int]:
        # MIDI-safe ordering: close an active note before opening the same key.
        event_priority = 0 if event.event_type == EventType.NOTE_OFF else 1
        return (int(event.tick), event_priority, int(event.pitch), int(event.channel), int(event.program))

    def _copy_model_event(
        self,
        event: MusicalEvent,
        *,
        tick: int,
        generation_start_tick: int,
        logical_tick: int | None = None,
    ) -> MusicalEvent:
        logical = int(event.tick if logical_tick is None else logical_tick)
        return MusicalEvent(
            tick=int(tick),
            pitch=int(event.pitch),
            event_type=event.event_type,
            velocity=int(event.velocity),
            channel=int(event.channel),
            program=int(event.program),
            is_placeholder=bool(event.is_placeholder),
            source="model",
            backup_level=max(0, logical - int(generation_start_tick)),
        )

    def _make_trace_row(
        self,
        *,
        event: MusicalEvent,
        logical_tick: int,
        scheduled_tick: int | None,
        policy: str,
        generation_start_tick: int,
        current_tick: int,
        action: str,
    ) -> dict[str, object]:
        return {
            "type": event.event_type.value,
            "pitch": int(event.pitch),
            "channel": int(event.channel),
            "program": int(event.program),
            "velocity": int(event.velocity),
            "logical_tick": int(logical_tick),
            "scheduled_tick": (int(scheduled_tick) if scheduled_tick is not None else None),
            "policy": str(policy),
            "action": str(action),
            "generation_start_tick": int(generation_start_tick),
            "current_tick": int(current_tick),
            "key": [int(event.pitch), int(event.channel), int(event.program)],
        }

    def _append_scheduled_model_event(
        self,
        plan: ModelSchedulePlan,
        event: MusicalEvent,
        *,
        scheduled_tick: int,
        logical_tick: int,
        policy: str,
        generation_start_tick: int,
        current_tick: int,
    ) -> None:
        scheduled = self._copy_model_event(
            event,
            tick=scheduled_tick,
            generation_start_tick=generation_start_tick,
            logical_tick=logical_tick,
        )
        plan.scheduled_events.append(
            ScheduledModelEvent(
                event=scheduled,
                scheduled_tick=int(scheduled_tick),
                logical_tick=int(logical_tick),
                policy=str(policy),
            )
        )
        plan.trace_rows.append(
            self._make_trace_row(
                event=event,
                logical_tick=logical_tick,
                scheduled_tick=scheduled_tick,
                policy=policy,
                generation_start_tick=generation_start_tick,
                current_tick=current_tick,
                action="scheduled",
            )
        )

    def _append_dropped_model_event(
        self,
        plan: ModelSchedulePlan,
        event: MusicalEvent,
        *,
        logical_tick: int,
        policy: str,
        generation_start_tick: int,
        current_tick: int,
    ) -> None:
        plan.trace_rows.append(
            self._make_trace_row(
                event=event,
                logical_tick=logical_tick,
                scheduled_tick=None,
                policy=policy,
                generation_start_tick=generation_start_tick,
                current_tick=current_tick,
                action="dropped",
            )
        )

    def _plan_model_events_for_playback(
        self,
        acc_events: List[MusicalEvent],
        *,
        current_tick: int,
        generation_start_tick: int,
        active_model_keys: set[EventKey],
    ) -> ModelSchedulePlan:
        plan = ModelSchedulePlan()
        open_by_key: dict[EventKey, list[MusicalEvent]] = {}
        paired_notes: list[tuple[MusicalEvent, MusicalEvent]] = []
        orphan_note_offs: list[MusicalEvent] = []

        events = sorted(acc_events, key=self._event_order_key)
        for event in events:
            if event.is_placeholder or event.pitch == -1:
                if int(event.tick) < int(current_tick):
                    self._append_dropped_model_event(
                        plan,
                        event,
                        logical_tick=int(event.tick),
                        policy="dropped_late_placeholder",
                        generation_start_tick=generation_start_tick,
                        current_tick=current_tick,
                    )
                else:
                    self._append_scheduled_model_event(
                        plan,
                        event,
                        scheduled_tick=int(event.tick),
                        logical_tick=int(event.tick),
                        policy="future_placeholder",
                        generation_start_tick=generation_start_tick,
                        current_tick=current_tick,
                    )
                continue

            key = self._model_event_key(event)
            if event.event_type == EventType.NOTE_ON and int(event.velocity) > 0:
                open_by_key.setdefault(key, []).append(event)
                continue

            if event.event_type == EventType.NOTE_OFF:
                opened = open_by_key.get(key)
                if opened:
                    note_on = opened.pop(0)
                    if not opened:
                        open_by_key.pop(key, None)
                    paired_notes.append((note_on, event))
                else:
                    orphan_note_offs.append(event)

        open_note_ons = [event for events_for_key in open_by_key.values() for event in events_for_key]

        for note_on, note_off in paired_notes:
            on_tick = int(note_on.tick)
            off_tick = int(note_off.tick)
            if off_tick <= int(current_tick):
                plan.dropped_past_note_count += 1
                self._append_dropped_model_event(
                    plan,
                    note_on,
                    logical_tick=on_tick,
                    policy="dropped_past_note",
                    generation_start_tick=generation_start_tick,
                    current_tick=current_tick,
                )
                self._append_dropped_model_event(
                    plan,
                    note_off,
                    logical_tick=off_tick,
                    policy="dropped_past_note",
                    generation_start_tick=generation_start_tick,
                    current_tick=current_tick,
                )
                continue

            if on_tick < int(current_tick):
                plan.clamped_onset_count += 1
                self._append_scheduled_model_event(
                    plan,
                    note_on,
                    scheduled_tick=int(current_tick),
                    logical_tick=on_tick,
                    policy="clamped_partial_note",
                    generation_start_tick=generation_start_tick,
                    current_tick=current_tick,
                )
                self._append_scheduled_model_event(
                    plan,
                    note_off,
                    scheduled_tick=off_tick,
                    logical_tick=off_tick,
                    policy="clamped_partial_note_off",
                    generation_start_tick=generation_start_tick,
                    current_tick=current_tick,
                )
                continue

            self._append_scheduled_model_event(
                plan,
                note_on,
                scheduled_tick=on_tick,
                logical_tick=on_tick,
                policy="future_note",
                generation_start_tick=generation_start_tick,
                current_tick=current_tick,
            )
            self._append_scheduled_model_event(
                plan,
                note_off,
                scheduled_tick=off_tick,
                logical_tick=off_tick,
                policy="future_note",
                generation_start_tick=generation_start_tick,
                current_tick=current_tick,
            )

        for note_on in open_note_ons:
            on_tick = int(note_on.tick)
            if on_tick < int(current_tick):
                plan.clamped_onset_count += 1
                scheduled_tick = int(current_tick)
                policy = "clamped_open_note"
            else:
                scheduled_tick = on_tick
                policy = "future_open_note"
            self._append_scheduled_model_event(
                plan,
                note_on,
                scheduled_tick=scheduled_tick,
                logical_tick=on_tick,
                policy=policy,
                generation_start_tick=generation_start_tick,
                current_tick=current_tick,
            )

        for note_off in orphan_note_offs:
            off_tick = int(note_off.tick)
            key = self._model_event_key(note_off)
            if key not in active_model_keys:
                plan.orphan_note_off_count += 1
                self._append_dropped_model_event(
                    plan,
                    note_off,
                    logical_tick=off_tick,
                    policy="orphan_note_off",
                    generation_start_tick=generation_start_tick,
                    current_tick=current_tick,
                )
                continue

            if off_tick < int(current_tick):
                scheduled_tick = int(current_tick)
                policy = "late_isolated_note_off"
            elif off_tick == int(current_tick):
                scheduled_tick = off_tick
                policy = "current_isolated_note_off"
            else:
                scheduled_tick = off_tick
                policy = "future_isolated_note_off"
            self._append_scheduled_model_event(
                plan,
                note_off,
                scheduled_tick=scheduled_tick,
                logical_tick=off_tick,
                policy=policy,
                generation_start_tick=generation_start_tick,
                current_tick=current_tick,
            )

        plan.scheduled_events.sort(
            key=lambda scheduled: (
                int(scheduled.scheduled_tick),
                0 if scheduled.event.event_type == EventType.NOTE_OFF else 1,
                int(scheduled.event.pitch),
                int(scheduled.event.channel),
                int(scheduled.event.program),
            )
        )
        return plan

    def _emit_model_schedule_trace(self, rows: list[dict[str, object]]) -> None:
        if not rows or not hasattr(self._output, "log_model_schedule"):
            return
        self._output.log_model_schedule(rows)  # type: ignore[union-attr]

    def _observe_model_output_event(self, event: MusicalEvent) -> None:
        if event.is_placeholder or event.pitch == -1:
            return
        key = self._model_event_key(event)
        if event.event_type == EventType.NOTE_ON and int(event.velocity) > 0:
            self._active_model_note_keys.add(key)
        elif event.event_type == EventType.NOTE_OFF:
            self._active_model_note_keys.discard(key)

    def _output_model_event(self, event: MusicalEvent) -> None:
        self._output.output_event(event, source=event.source)
        self._observe_model_output_event(event)

    def _timing_to_log_dict(self, timing: TimingInfo) -> dict:
        return {
            "request_arrival_time": float(timing.request_arrival_time),
            "response_output_time": float(timing.response_output_time),
            "preprocess_start_time": float(timing.preprocess_start_time),
            "inference_start_time": float(timing.inference_start_time),
            "inference_end_time": float(timing.inference_end_time),
            "postprocess_start_time": float(timing.postprocess_start_time),
            "round_trip_time": (float(timing.round_trip_time) if timing.round_trip_time is not None else None),
            "server_processing_duration": (
                float(timing.server_processing_duration)
                if timing.server_processing_duration is not None
                else None
            ),
            "total_network_latency": (
                float(timing.total_network_latency)
                if timing.total_network_latency is not None
                else None
            ),
        }

    def _build_inference_log_payload(
        self,
        *,
        generation_start_tick: int,
        melody_events: List[MusicalEvent],
        acc_events: List[MusicalEvent],
        timing_info: TimingInfo,
        request_timestamp: float,
        response_timestamp: float,
    ) -> tuple[dict, dict]:
        engine_config = getattr(self._engine, "_config", None)
        detail = getattr(self._output, "inference_log_detail", "summary")

        request_full = {
            "timestamp": request_timestamp,
            "generation_start_tick": int(generation_start_tick),
            "generation_length_frames": int(self._generation_length_frames),
            "generation_interval_ticks": int(self._generation_interval_ticks),
            "prompt_length_ticks": None,
            "model_name": getattr(engine_config, "model_name", None),
            "inference_mode": getattr(engine_config, "inference_mode", None),
            "melody_notes_count": len(melody_events),
            "melody_notes": [self._event_to_log_dict(event) for event in melody_events],
        }
        response_full = {
            "timestamp": response_timestamp,
            "accompaniment_notes_count": len(acc_events),
            "accompaniment": [self._event_to_log_dict(event) for event in acc_events],
            "timings": self._timing_to_log_dict(timing_info),
        }

        if detail == "full":
            return request_full, response_full

        request_summary = {
            "timestamp": request_timestamp,
            "generation_start_tick": int(generation_start_tick),
            "melody_notes_count": len(melody_events),
            "generation_length_frames": int(self._generation_length_frames),
        }
        response_summary = {
            "timestamp": response_timestamp,
            "accompaniment_notes_count": len(acc_events),
        }
        return request_summary, response_summary

    def _input_worker(self) -> None:
        """Read events from input source and add to queues."""
        assert self._runtime is not None
        start = self._timeline_start_time()
        self._sleep_until(start)
        for ev in self._input.read_events():
            if not self._running or self._stop_requested.is_set():
                break
            elapsed = max(0.0, self._now() - start)
            stamped = stamp_user_input_event(
                ev,
                elapsed_seconds=elapsed,
                tempo=self._tempo,
                snap_forward_fraction=self._input_snap_forward_fraction,
            )
            self._event_q.put(stamped)
            # Add to melody history (only note_on/note_off events)
            with self._melody_history_lock:
                self._melody_history.append(stamped)

    # Fraction of a tick to sleep after time-sync before draining the input queue,
    # giving user events generated near the tick boundary time to arrive.
    _INPUT_BUFFER_RATIO: float = 0.1

    def _process_inference_responses(self, *, current_tick: int) -> int:
        """Consume every available response and mark it processed exactly once."""
        processed_count = 0
        while True:
            try:
                item = self._inference_response_queue.get_nowait()
            except queue.Empty:
                break

            try:
                response = self._normalize_response(item)
                acc_events = response.accompaniment_events
                generation_start_tick = response.generation_start_tick
                removed_future = self._scheduler.pop_future_events(
                    from_tick=generation_start_tick,
                    source="model",
                )

                plan = self._plan_model_events_for_playback(
                    acc_events,
                    current_tick=current_tick,
                    generation_start_tick=generation_start_tick,
                    active_model_keys=set(self._active_model_note_keys),
                )

                forced_events: list[ScheduledModelEvent] = []
                for removed in removed_future:
                    if removed.event_type != EventType.NOTE_OFF:
                        continue
                    if removed.is_placeholder or removed.pitch == -1:
                        continue
                    if self._model_event_key(removed) not in self._active_model_note_keys:
                        continue
                    forced = self._copy_model_event(
                        removed,
                        tick=current_tick,
                        generation_start_tick=generation_start_tick,
                        logical_tick=int(removed.tick),
                    )
                    forced_events.append(
                        ScheduledModelEvent(
                            event=forced,
                            scheduled_tick=int(current_tick),
                            logical_tick=int(removed.tick),
                            policy="forced_note_off",
                        )
                    )
                    plan.forced_note_off_count += 1
                    plan.trace_rows.append(
                        self._make_trace_row(
                            event=removed,
                            logical_tick=int(removed.tick),
                            scheduled_tick=int(current_tick),
                            policy="forced_note_off",
                            generation_start_tick=generation_start_tick,
                            current_tick=current_tick,
                            action="scheduled",
                        )
                    )

                self._emit_model_schedule_trace(plan.trace_rows)
                self._record_operational_plan(plan)

                for scheduled in forced_events + plan.scheduled_events:
                    self._scheduler.schedule(scheduled.event, scheduled.scheduled_tick)

                if (
                    plan.clamped_onset_count
                    or plan.dropped_past_note_count
                    or plan.orphan_note_off_count
                    or plan.forced_note_off_count
                ):
                    self._output.output_status(
                        "debug",
                        (
                            "Model scheduling adjusted "
                            f"at tick={current_tick} (generation_start_tick={generation_start_tick}): "
                            f"clamped={plan.clamped_onset_count}, "
                            f"dropped_past={plan.dropped_past_note_count}, "
                            f"orphan_off={plan.orphan_note_off_count}, "
                            f"forced_off={plan.forced_note_off_count}."
                        ),
                    )

                self._update_request_record(response.request_id, processed=True)
                self._log_lifecycle(
                    "processed",
                    request_id=response.request_id,
                    generation_start_tick=int(generation_start_tick),
                    accompaniment_event_count=len(acc_events),
                    empty_success=not acc_events,
                    processed_at_tick=int(current_tick),
                    response_metadata=response.response_metadata,
                )
                processed_count += 1
            finally:
                self._inference_response_queue.task_done()
        return processed_count

    def _drain_pending_work(self, *, current_tick: int) -> None:
        """Drain worker requests and responses, bounded by a hard timeout."""
        self._begin_draining(self._drain_reason or "producer_stopped")
        deadline = self._monotonic() + self._drain_timeout_seconds
        while True:
            self._process_inference_responses(current_tick=current_tick)
            with self._lifecycle_lock:
                in_flight = bool(self._in_flight_request_ids)
            idle = (
                self._worker_done.is_set()
                and self._inference_request_queue.unfinished_tasks == 0
                and self._inference_response_queue.unfinished_tasks == 0
                and not in_flight
            )
            if idle:
                self._drain_done.set()
                self._log_lifecycle("drain_completed", current_tick=int(current_tick))
                return
            if self._monotonic() >= deadline:
                self._drain_timed_out = True
                self._log_lifecycle(
                    "drain_timeout",
                    current_tick=int(current_tick),
                    request_queue_unfinished=int(self._inference_request_queue.unfinished_tasks),
                    response_queue_unfinished=int(self._inference_response_queue.unfinished_tasks),
                    in_flight_request_ids=sorted(self._in_flight_request_ids),
                )
                return
            self._sleep(0.001)

    def _tick_loop(self, *, max_ticks: Optional[int]) -> None:
        """Main tick loop: timing, event playback, and inference triggering."""
        assert self._runtime is not None
        self._run_count_in()
        start = self._timeline_start_time()
        tick = 0
        # Accumulates user events between beat-tail triggers.
        notes_for_next_request: List[MusicalEvent] = []

        managed_lifecycle = self.lifecycle_state != RealTimeServiceState.STOPPED
        while self._running and not self._stop_requested.is_set():
            if max_ticks is not None and tick >= max_ticks:
                break

            # 1. Absolute time sync: sleep until this tick's wall-clock boundary.
            target_time = start + self._tempo.tick_to_seconds(tick)
            self._sleep_until(target_time)

            # 2. Emit tick info.
            mt = MusicalTime.from_tick(tick, self._tempo)
            self._output.output_tick(tick=tick, bar=mt.bar, beat=mt.beat)
            if self._tick_observer is not None and not self._tick_observer_closed:
                try:
                    self._tick_observer.on_tick(tick)
                except Exception as exc:
                    self._output.output_status("error", f"Tick observer error: {exc}")

            # 3. tick=0: fire once with full melody history (covers injected history).
            if tick == 0:
                if self._planned_generation_start_ticks is not None:
                    # Formal runs freeze the initial-history decision in start().
                    # Live input racing the tick thread must not nondeterministically
                    # create an extra generation request at tick zero.
                    notes_for_request = list(self._initial_generation_history)
                else:
                    with self._melody_history_lock:
                        notes_for_request = self._melody_history.copy()
                if notes_for_request:
                    self._enqueue_inference_request(0, notes_for_request)
                if (
                    managed_lifecycle
                    and self._request_cutoff_tick == 0
                    and self.lifecycle_state == RealTimeServiceState.ACCEPTING_REQUESTS
                ):
                    self._begin_draining("request_cutoff_reached")

            # 4. Input buffer window: wait 10% of a tick so events near the
            #    boundary have time to arrive before we drain the queue.
            self._sleep(self._tempo.seconds_per_tick * self._INPUT_BUFFER_RATIO)

            # 5. Drain input events and buffer them for aligned playback.
            user_events_to_play: List[MusicalEvent] = []
            while True:
                try:
                    ev = self._event_q.get_nowait()
                except queue.Empty:
                    break
                user_events_to_play.append(ev)
                notes_for_next_request.append(ev)

            # 6. Process inference responses.
            self._process_inference_responses(current_tick=tick)

            # 7. Play metronome and musical events from the same post-buffer phase.
            self._output_metronome_tick(tick=tick, bar=mt.bar, beat=mt.beat)
            for ev in user_events_to_play:
                self._output.output_event(ev, source="user")
            for ev in sorted(self._scheduler.get_events_at_tick(tick), key=self._event_order_key):
                if ev.source == "model":
                    self._output_model_event(ev)
                else:
                    self._output.output_event(ev, source=ev.source)

            # 8. Beat tail (tick 4n-1, n≥1): always send for next beat, even if no new events.
            ticks_per_beat = self._tempo.ticks_per_beat
            if tick > 0 and (tick % ticks_per_beat) == (ticks_per_beat - 1):
                next_generation_tick = tick + 1
                # Once a managed session enters draining, beat tails remain part
                # of playback but are no longer inference scheduling boundaries.
                # If draining began prematurely, the independent planned schedule
                # below will expose the omitted request as a content failure.
                if (
                    not managed_lifecycle
                    or self.lifecycle_state == RealTimeServiceState.ACCEPTING_REQUESTS
                ):
                    enqueued = self._enqueue_inference_request(
                        next_generation_tick,
                        notes_for_next_request,
                    )
                    notes_for_next_request = []
                    if (
                        managed_lifecycle
                        and enqueued
                        and self._request_cutoff_tick is not None
                        and next_generation_tick == self._request_cutoff_tick
                    ):
                        self._begin_draining("request_cutoff_reached")

            self._last_processed_tick = int(tick)
            tick += 1

        if managed_lifecycle:
            self._begin_draining(
                "stop_requested" if self._stop_requested.is_set() else "run_stop_reached"
            )
            self._drain_pending_work(current_tick=tick)
            with self._state_lock:
                self._state = RealTimeServiceState.STOPPED
                self._running = False
            self._finalize_validity()
        else:
            self._running = False

    def _inference_worker(self) -> None:
        """Background worker that processes inference requests."""
        try:
            while self._running or self.lifecycle_state == RealTimeServiceState.DRAINING:
                try:
                    item = self._inference_request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is _INFERENCE_STOP:
                    self._inference_request_queue.task_done()
                    break

                consumed_requests: list[InferenceRequest] = []
                stop_after_current = False
                try:
                    request = self._normalize_request(item)
                    consumed_requests.append(request)
                    merged_melody_events = list(request.melody_events)

                    # Latest-only behavior is explicit in validity: older requests
                    # are permanently recorded as stale drops, never as successes.
                    while True:
                        try:
                            newer_item = self._inference_request_queue.get_nowait()
                        except queue.Empty:
                            break
                        if newer_item is _INFERENCE_STOP:
                            self._inference_request_queue.task_done()
                            stop_after_current = True
                            break
                        newer_request = self._normalize_request(newer_item)
                        consumed_requests.append(newer_request)
                        if newer_request.melody_events:
                            merged_melody_events.extend(newer_request.melody_events)
                        request = newer_request

                    stale_requests = consumed_requests[:-1]
                    if stale_requests:
                        with self._lifecycle_lock:
                            self._operational_counts["stale_request_drops"] += len(stale_requests)
                        for stale_request in stale_requests:
                            self._update_request_record(
                                stale_request.request_id,
                                stale_dropped=True,
                            )
                            self._log_lifecycle(
                                "stale_dropped",
                                request=stale_request,
                                merged_into_request_id=request.request_id,
                            )
                        self._output.output_status(
                            "debug",
                            (
                                "Latest-only inference dropped "
                                f"{len(stale_requests)} stale request(s); "
                                f"merged {len(merged_melody_events)} melody event(s)."
                            ),
                        )

                    actual_increment_digest = self._digest_events(merged_melody_events)
                    self._update_request_record(
                        request.request_id,
                        started=True,
                        sent_input_event_count=len(merged_melody_events),
                        sent_input_increment_digest=actual_increment_digest,
                    )
                    with self._lifecycle_lock:
                        self._in_flight_request_ids.add(request.request_id)
                    self._log_lifecycle(
                        "started",
                        request=request,
                        sent_input_event_count=len(merged_melody_events),
                        sent_input_increment_digest=actual_increment_digest,
                    )

                    try:
                        request_send_time = self._now()
                        request_id_setter = getattr(self._engine, "set_next_request_id", None)
                        if callable(request_id_setter):
                            request_id_setter(request.request_id)
                        acc_events, timing_info = self._engine.generate_accompaniment(
                            melody_events=merged_melody_events,
                            generation_start_tick=request.generation_start_tick,
                            generation_length_frames=self._generation_length_frames,
                        )
                        response_receive_time = self._now()
                        response_metadata = self._consume_response_metadata()
                        round_trip_time = response_receive_time - request_send_time
                        empty_success = not acc_events
                        metadata_validation = self._validate_response_metadata(
                            request=request,
                            metadata=response_metadata,
                            sent_input_increment_digest=actual_increment_digest,
                            empty_success=empty_success,
                        )
                        self._update_request_record(
                            request.request_id,
                            succeeded=True,
                            empty_success=empty_success,
                            response_metadata_status=response_metadata.get("metadata_status"),
                            raw_token_digest=response_metadata.get("raw_token_digest"),
                            token_decode_digest=response_metadata.get(
                                "token_decode_digest"
                            ),
                            raw_token_count=response_metadata.get("raw_token_count"),
                            output_event_digest=response_metadata.get(
                                "output_event_digest"
                            ),
                            engine_request_id=response_metadata.get("request_id"),
                            session_epoch=response_metadata.get("session_epoch"),
                            effective_seed=response_metadata.get("effective_seed"),
                            effective_bpm=response_metadata.get("effective_bpm"),
                            server_input_increment_digest=response_metadata.get(
                                "input_increment_digest"
                            ),
                            server_input_cumulative_digest=response_metadata.get(
                                "input_cumulative_digest"
                            ),
                            prompt_token_digest=response_metadata.get(
                                "prompt_token_digest"
                            ),
                            part0_token_digest=response_metadata.get(
                                "part0_token_digest"
                            ),
                            part0_roll_digest=response_metadata.get(
                                "part0_roll_digest"
                            ),
                            part0_roll_shape=response_metadata.get(
                                "part0_roll_shape"
                            ),
                            part0_roll_bytes_sha256=response_metadata.get(
                                "part0_roll_bytes_sha256"
                            ),
                            part0_roll_start_tick=response_metadata.get(
                                "part0_roll_start_tick"
                            ),
                            part0_roll_end_tick=response_metadata.get(
                                "part0_roll_end_tick"
                            ),
                            context_start_tick=response_metadata.get(
                                "context_start_tick"
                            ),
                            **metadata_validation,
                        )
                        self._log_lifecycle(
                            "succeeded",
                            request=request,
                            accompaniment_event_count=len(acc_events),
                            empty_success=empty_success,
                            response_metadata=response_metadata,
                            metadata_validation=metadata_validation,
                        )
                        self._inference_response_queue.put(
                            InferenceResponse(
                                request_id=request.request_id,
                                accompaniment_events=list(acc_events),
                                generation_start_tick=request.generation_start_tick,
                                response_metadata=response_metadata,
                            )
                        )

                        server_process_ms = None
                        if timing_info.server_processing_duration is not None:
                            server_process_ms = timing_info.server_processing_duration * 1000
                        elif (
                            timing_info.response_output_time > 0
                            and timing_info.request_arrival_time > 0
                        ):
                            server_process_ms = (
                                timing_info.response_output_time - timing_info.request_arrival_time
                            ) * 1000

                        self._output.output_stats(
                            round_trip_ms=round_trip_time * 1000,
                            server_process_ms=server_process_ms,
                        )

                        if hasattr(self._output, "log_inference"):
                            request_data, response_data = self._build_inference_log_payload(
                                generation_start_tick=request.generation_start_tick,
                                melody_events=merged_melody_events,
                                acc_events=acc_events,
                                timing_info=timing_info,
                                request_timestamp=request_send_time,
                                response_timestamp=response_receive_time,
                            )
                            request_data["lifecycle_request_id"] = request.request_id
                            response_data["response_metadata"] = response_metadata
                            self._output.log_inference(  # type: ignore[union-attr]
                                request=request_data,
                                response=response_data,
                                latency_ms=round_trip_time * 1000,
                                server_process_ms=server_process_ms or 0.0,
                            )
                    except Exception as exc:
                        self._update_request_record(
                            request.request_id,
                            failed=True,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        self._log_lifecycle(
                            "failed",
                            request=request,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            http_error=bool(
                                "http" in type(exc).__name__.lower()
                                or "http" in str(exc).lower()
                            ),
                        )
                        self._output.output_status("error", f"Inference error: {exc}")
                    finally:
                        with self._lifecycle_lock:
                            self._in_flight_request_ids.discard(request.request_id)
                finally:
                    for _ in consumed_requests:
                        self._inference_request_queue.task_done()

                if stop_after_current:
                    break
                if not self._running and self.lifecycle_state == RealTimeServiceState.STOPPED:
                    break
        finally:
            self._worker_done.set()

    def start(
        self,
        *,
        max_ticks: Optional[int] = None,
        run_stop_tick: Optional[int] = None,
        analysis_end_tick: Optional[int] = None,
        last_input_note_off_tick: Optional[int] = None,
        request_cutoff_tick: Optional[int] = None,
        drain_timeout_seconds: Optional[float] = None,
    ) -> None:
        """Start the service with explicit, exclusive run/analysis horizons.

        ``run_stop_tick`` (and its legacy alias ``max_ticks``) is exclusive.
        ``analysis_end_tick`` is also exclusive. ``request_cutoff_tick`` is the
        last allowed generation-start tick (inclusive) and must be beat aligned.
        If analysis_end is supplied without an explicit cutoff, the default is
        ``floor((analysis_end_tick - 1) / ticks_per_beat) * ticks_per_beat``:
        the final beat boundary strictly before the exclusive analysis horizon.

        Supplying either cutoff form enables the formal request-tick contract.
        Formal scheduling is beat-tail based, so ``generation_interval_ticks``
        must equal ``ticks_per_beat``. Its planned generation starts are frozen
        before worker threads start and checked exactly at finalization.
        """
        if self._running:
            return
        if max_ticks is not None and run_stop_tick is not None and int(max_ticks) != int(run_stop_tick):
            raise ValueError("max_ticks and run_stop_tick must match when both are supplied")
        effective_run_stop = run_stop_tick if run_stop_tick is not None else max_ticks
        if effective_run_stop is not None and int(effective_run_stop) < 0:
            raise ValueError("run_stop_tick is exclusive and must be >= 0")
        if analysis_end_tick is not None and int(analysis_end_tick) <= 0:
            raise ValueError("analysis_end_tick is exclusive and must be > 0")
        if last_input_note_off_tick is not None and int(last_input_note_off_tick) < 0:
            raise ValueError("last_input_note_off_tick must be >= 0")
        if drain_timeout_seconds is not None:
            if float(drain_timeout_seconds) <= 0:
                raise ValueError("drain_timeout_seconds must be > 0")
            self._drain_timeout_seconds = float(drain_timeout_seconds)

        ticks_per_beat = int(self._tempo.ticks_per_beat)
        effective_cutoff = request_cutoff_tick
        if effective_cutoff is None and analysis_end_tick is not None:
            effective_cutoff = ((int(analysis_end_tick) - 1) // ticks_per_beat) * ticks_per_beat
        if effective_cutoff is not None:
            if int(self._generation_interval_ticks) != ticks_per_beat:
                raise ValueError(
                    "formal request scheduling requires generation_interval_ticks "
                    "to equal ticks_per_beat"
                )
            if int(effective_cutoff) < 0:
                raise ValueError("request_cutoff_tick must be >= 0")
            if int(effective_cutoff) % ticks_per_beat != 0:
                raise ValueError("request_cutoff_tick must be aligned to a beat boundary")
            if analysis_end_tick is not None and int(effective_cutoff) >= int(analysis_end_tick):
                raise ValueError("request_cutoff_tick must be strictly before analysis_end_tick")
            if effective_run_stop is not None and int(effective_cutoff) >= int(effective_run_stop):
                raise ValueError("request_cutoff_tick must be strictly before run_stop_tick")
        if (
            effective_run_stop is not None
            and last_input_note_off_tick is not None
            and int(last_input_note_off_tick) > int(effective_run_stop)
        ):
            raise ValueError("last_input_note_off_tick must not exceed run_stop_tick")

        self._analysis_end_tick = int(analysis_end_tick) if analysis_end_tick is not None else None
        self._last_input_note_off_tick = (
            int(last_input_note_off_tick) if last_input_note_off_tick is not None else None
        )
        self._request_cutoff_tick = int(effective_cutoff) if effective_cutoff is not None else None
        self._run_stop_tick = int(effective_run_stop) if effective_run_stop is not None else None
        with self._melody_history_lock:
            self._initial_generation_history = list(self._melody_history)
        with self._lifecycle_lock:
            self._actual_generation_start_ticks = []
            self._rejected_generation_start_ticks = []
            if self._request_cutoff_tick is None:
                self._planned_generation_start_ticks = None
            else:
                planned = list(
                    range(
                        ticks_per_beat,
                        self._request_cutoff_tick + 1,
                        ticks_per_beat,
                    )
                )
                if self._initial_generation_history:
                    planned.insert(0, 0)
                self._planned_generation_start_ticks = planned
        self._stop_requested.clear()
        self._worker_done.clear()
        self._drain_done.clear()
        self._sentinel_enqueued = False
        self._validity_finalized = False
        self._drain_timed_out = False
        self._drain_reason = None
        self._last_processed_tick = -1
        self._lifecycle_session_id = f"rt-{uuid.uuid4().hex}"
        with self._state_lock:
            self._state = RealTimeServiceState.ACCEPTING_REQUESTS
            self._running = True
        session_start_time = self._now()
        timeline_start_time = session_start_time + self._tempo.tick_to_seconds(self._count_in_ticks)
        self._runtime = RealTimeServiceRuntime(
            session_start_time=session_start_time,
            timeline_start_time=timeline_start_time,
        )
        self._output.output_status("running", "")
        self._log_lifecycle(
            "session_started",
            analysis_end_tick=self._analysis_end_tick,
            analysis_end_exclusive=True,
            last_input_note_off_tick=self._last_input_note_off_tick,
            request_cutoff_tick=self._request_cutoff_tick,
            request_cutoff_inclusive=True,
            planned_generation_start_ticks=(
                None
                if self._planned_generation_start_ticks is None
                else list(self._planned_generation_start_ticks)
            ),
            run_stop_tick=self._run_stop_tick,
            run_stop_exclusive=True,
            drain_timeout_seconds=self._drain_timeout_seconds,
        )
        self._start_tick_observer()

        self._input_thread = threading.Thread(target=self._input_worker, daemon=True)
        self._tick_thread = threading.Thread(
            target=self._tick_loop,
            kwargs={"max_ticks": self._run_stop_tick},
            daemon=True,
        )
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)

        self._input_thread.start()
        self._inference_thread.start()
        self._tick_thread.start()

    def stop(self) -> None:
        """Idempotently stop production, drain pending work, and close resources."""
        self._stop_requested.set()
        self._close_tick_observer()
        self._begin_draining("explicit_stop")
        try:
            self._input.close()
        except Exception:
            pass

        join_timeout = self._drain_timeout_seconds + 1.0
        current = threading.current_thread()
        if self._tick_thread is not None and self._tick_thread is not current:
            self._tick_thread.join(timeout=join_timeout)

        # A service stopped before its tick thread started still needs a worker
        # sentinel and a response consumer.
        if self.lifecycle_state != RealTimeServiceState.STOPPED:
            self._drain_pending_work(current_tick=max(0, self._last_processed_tick + 1))

        if self._inference_thread is not None and self._inference_thread is not current:
            self._inference_thread.join(timeout=1.0)
            if self._inference_thread.is_alive():
                self._drain_timed_out = True
                self._log_lifecycle("worker_join_timeout")
        if self._input_thread is not None and self._input_thread is not current:
            self._input_thread.join(timeout=1.0)

        with self._state_lock:
            self._state = RealTimeServiceState.STOPPED
            self._running = False
        self._finalize_validity()

        with self._state_lock:
            if self._resources_closed:
                return
            self._resources_closed = True
        try:
            self._output.output_status("stopped", "")
        except Exception:
            pass
        try:
            self._output.close()
        except Exception:
            pass
