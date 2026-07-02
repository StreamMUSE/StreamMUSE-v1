"""Real-time orchestration service with inference worker."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from streammuse.domain.interfaces import InferenceEngine, InputSource, OutputSink
from streammuse.domain.interfaces.timing_info import TimingInfo
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import MusicalTime, PlaybackScheduler, Tempo


@dataclass(frozen=True)
class RealTimeServiceRuntime:
    session_start_time: float
    timeline_start_time: float


EventKey = tuple[int, int, int]


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
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if int(count_in_beats) < 0:
            raise ValueError("count_in_beats must be >= 0")
        self._input = input_source
        self._engine = inference_engine
        self._output = output_sink
        self._tempo = tempo
        self._scheduler = scheduler
        self._generation_interval_ticks = generation_interval_ticks
        self._generation_length_frames = generation_length_frames
        self._count_in_beats = int(count_in_beats)
        self._count_in_ticks = self._count_in_beats * int(self._tempo.ticks_per_beat)
        self._now = now
        self._sleep = sleep

        self._running = False
        self._runtime: RealTimeServiceRuntime | None = None
        self._input_thread: threading.Thread | None = None
        self._tick_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._event_q: queue.Queue[MusicalEvent] = queue.Queue()
        self._melody_history: List[MusicalEvent] = []
        self._melody_history_lock = threading.Lock()
        self._inference_request_queue: queue.Queue[tuple[int, List[MusicalEvent]]] = queue.Queue()
        self._inference_response_queue: queue.Queue[tuple[List[MusicalEvent], int]] = queue.Queue()
        self._active_model_note_keys: set[EventKey] = set()

    @property
    def running(self) -> bool:
        return self._running

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
            if not self._running:
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
            if not self._running:
                break
            elapsed = max(0.0, self._now() - start)
            tick = self._tempo.seconds_to_tick(elapsed)
            # Assign tick at dequeue time per timing design.
            stamped = MusicalEvent(
                tick=tick,
                pitch=ev.pitch,
                event_type=ev.event_type,
                velocity=ev.velocity,
                channel=ev.channel,
                program=ev.program,
                is_placeholder=ev.is_placeholder,
                source="user",  # Mark as user-generated event
            )
            self._event_q.put(stamped)
            # Add to melody history (only note_on/note_off events)
            with self._melody_history_lock:
                self._melody_history.append(stamped)

    # Fraction of a tick to sleep after time-sync before draining the input queue,
    # giving user events generated near the tick boundary time to arrive.
    _INPUT_BUFFER_RATIO: float = 0.1

    def _tick_loop(self, *, max_ticks: Optional[int]) -> None:
        """Main tick loop: timing, event playback, and inference triggering."""
        assert self._runtime is not None
        self._run_count_in()
        start = self._timeline_start_time()
        tick = 0
        # Accumulates user events between beat-tail triggers.
        notes_for_next_request: List[MusicalEvent] = []

        while self._running:
            if max_ticks is not None and tick >= max_ticks:
                self._running = False
                break

            # 1. Absolute time sync: sleep until this tick's wall-clock boundary.
            target_time = start + self._tempo.tick_to_seconds(tick)
            self._sleep_until(target_time)

            # 2. Emit tick info.
            mt = MusicalTime.from_tick(tick, self._tempo)
            self._output.output_tick(tick=tick, bar=mt.bar, beat=mt.beat)

            # 3. tick=0: fire once with full melody history (covers injected history).
            if tick == 0:
                with self._melody_history_lock:
                    notes_for_request = self._melody_history.copy()
                if notes_for_request:
                    self._inference_request_queue.put((0, notes_for_request))

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
            while True:
                try:
                    acc_events, generation_start_tick = self._inference_response_queue.get_nowait()
                except queue.Empty:
                    break

                removed_future = self._scheduler.pop_future_events(
                    from_tick=generation_start_tick,
                    source="model",
                )

                plan = self._plan_model_events_for_playback(
                    acc_events,
                    current_tick=tick,
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
                        tick=tick,
                        generation_start_tick=generation_start_tick,
                        logical_tick=int(removed.tick),
                    )
                    forced_events.append(
                        ScheduledModelEvent(
                            event=forced,
                            scheduled_tick=int(tick),
                            logical_tick=int(removed.tick),
                            policy="forced_note_off",
                        )
                    )
                    plan.forced_note_off_count += 1
                    plan.trace_rows.append(
                        self._make_trace_row(
                            event=removed,
                            logical_tick=int(removed.tick),
                            scheduled_tick=int(tick),
                            policy="forced_note_off",
                            generation_start_tick=generation_start_tick,
                            current_tick=tick,
                            action="scheduled",
                        )
                    )

                self._emit_model_schedule_trace(plan.trace_rows)

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
                            f"at tick={tick} (generation_start_tick={generation_start_tick}): "
                            f"clamped={plan.clamped_onset_count}, "
                            f"dropped_past={plan.dropped_past_note_count}, "
                            f"orphan_off={plan.orphan_note_off_count}, "
                            f"forced_off={plan.forced_note_off_count}."
                        ),
                    )

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
                self._inference_request_queue.put((tick + 1, notes_for_next_request))
                notes_for_next_request = []

            tick += 1

    def _inference_worker(self) -> None:
        """Background worker that processes inference requests."""
        while self._running:
            try:
                generation_start_tick, melody_events = self._inference_request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Latest-only queue behavior: keep newest tick while merging skipped melody increments.
            dropped_request_count = 0
            merged_melody_events = list(melody_events)
            while True:
                try:
                    newer_tick, newer_events = self._inference_request_queue.get_nowait()
                except queue.Empty:
                    break
                generation_start_tick = newer_tick
                dropped_request_count += 1
                if newer_events:
                    merged_melody_events.extend(newer_events)

            if dropped_request_count > 0:
                self._output.output_status(
                    "debug",
                    (
                        "Latest-only inference dropped "
                        f"{dropped_request_count} stale request(s); "
                        f"merged {len(merged_melody_events)} melody event(s)."
                    ),
                )

            if not self._running:
                break

            try:
                request_send_time = self._now()
                # Call inference engine
                acc_events, timing_info = self._engine.generate_accompaniment(
                    melody_events=merged_melody_events,
                    generation_start_tick=generation_start_tick,
                    generation_length_frames=self._generation_length_frames,
                )
                response_receive_time = self._now()

                # Calculate round trip time
                round_trip_time = response_receive_time - request_send_time

                # Put response in queue for tick loop to process
                self._inference_response_queue.put((acc_events, generation_start_tick))

                # Output stats
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

                # Log inference details if the output sink supports it
                if hasattr(self._output, "log_inference"):
                    request_data, response_data = self._build_inference_log_payload(
                        generation_start_tick=generation_start_tick,
                        melody_events=merged_melody_events,
                        acc_events=acc_events,
                        timing_info=timing_info,
                        request_timestamp=request_send_time,
                        response_timestamp=response_receive_time,
                    )
                    self._output.log_inference(  # type: ignore[union-attr]
                        request=request_data,
                        response=response_data,
                        latency_ms=round_trip_time * 1000,
                        server_process_ms=server_process_ms or 0.0,
                    )
            except Exception as e:
                # Log error but continue running
                self._output.output_status("error", f"Inference error: {e}")

    def start(self, *, max_ticks: Optional[int] = None) -> None:
        """Start the service (input worker, tick loop, inference worker)."""
        if self._running:
            return
        self._running = True
        session_start_time = self._now()
        timeline_start_time = session_start_time + self._tempo.tick_to_seconds(self._count_in_ticks)
        self._runtime = RealTimeServiceRuntime(
            session_start_time=session_start_time,
            timeline_start_time=timeline_start_time,
        )
        self._output.output_status("running", "")

        self._input_thread = threading.Thread(target=self._input_worker, daemon=True)
        self._tick_thread = threading.Thread(target=self._tick_loop, kwargs={"max_ticks": max_ticks}, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)

        self._input_thread.start()
        self._tick_thread.start()
        self._inference_thread.start()

    def stop(self) -> None:
        """Stop the service and clean up threads."""
        if not self._running:
            return
        self._running = False

        # Signal inference worker to stop
        try:
            self._inference_request_queue.put((0, []))  # Dummy item to wake worker
        except Exception:
            pass

        try:
            self._input.close()
        except Exception:
            pass
        try:
            self._output.output_status("stopped", "")
            self._output.close()
        except Exception:
            pass

        # Best-effort join.
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=1.0)
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=1.0)
        if self._input_thread is not None:
            self._input_thread.join(timeout=1.0)
