"""Real-time orchestration service with inference worker."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from streammuse.domain.interfaces import InferenceEngine, InputSource, OutputSink
from streammuse.domain.interfaces.timing_info import TimingInfo
from streammuse.domain.musical import MusicalEvent
from streammuse.domain.timing import MusicalTime, PlaybackScheduler, Tempo


@dataclass(frozen=True)
class RealTimeServiceRuntime:
    session_start_time: float


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
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._input = input_source
        self._engine = inference_engine
        self._output = output_sink
        self._tempo = tempo
        self._scheduler = scheduler
        self._generation_interval_ticks = generation_interval_ticks
        self._generation_length_frames = generation_length_frames
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

    @property
    def running(self) -> bool:
        return self._running

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
        start = self._runtime.session_start_time
        for ev in self._input.read_events():
            if not self._running:
                break
            elapsed = self._now() - start
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
        start = self._runtime.session_start_time
        tick = 0
        # Accumulates user events between beat-tail triggers.
        notes_for_next_request: List[MusicalEvent] = []

        while self._running:
            if max_ticks is not None and tick >= max_ticks:
                self._running = False
                break

            # 1. Absolute time sync: sleep until this tick's wall-clock boundary.
            target_time = start + self._tempo.tick_to_seconds(tick)
            delay = target_time - self._now()
            if delay > 0:
                self._sleep(delay)

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

            # 5. Drain input events: emit immediately and buffer for next trigger.
            while True:
                try:
                    ev = self._event_q.get_nowait()
                except queue.Empty:
                    break
                self._output.output_event(ev, source="user")
                notes_for_next_request.append(ev)

            # 6. Process inference responses.
            while True:
                try:
                    acc_events, generation_start_tick = self._inference_response_queue.get_nowait()
                except queue.Empty:
                    break

                self._scheduler.clear_future_events(from_tick=generation_start_tick, source="model")

                late_event_count = 0
                for ev in acc_events:
                    backup_level = max(0, ev.tick - generation_start_tick)
                    ev_model = MusicalEvent(
                        tick=ev.tick,
                        pitch=ev.pitch,
                        event_type=ev.event_type,
                        velocity=ev.velocity,
                        channel=ev.channel,
                        program=ev.program,
                        is_placeholder=ev.is_placeholder,
                        source="model",
                        backup_level=backup_level,
                    )
                    schedule_tick = ev.tick if ev.tick >= tick else tick
                    if ev.tick < tick:
                        late_event_count += 1
                    self._scheduler.schedule(ev_model, schedule_tick)

                if late_event_count > 0:
                    self._output.output_status(
                        "debug",
                        (
                            f"Recovered {late_event_count} late model event(s) "
                            f"at tick={tick} (generation_start_tick={generation_start_tick})."
                        ),
                    )

            # 7. Play scheduled events.
            for ev in self._scheduler.get_events_at_tick(tick):
                self._output.output_event(ev, source=ev.source)

            # 8. Beat tail (tick 4n-1, n≥1): send buffered events for next beat.
            ticks_per_beat = self._tempo.ticks_per_beat
            if tick > 0 and (tick % ticks_per_beat) == (ticks_per_beat - 1):
                if notes_for_next_request:
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
        self._runtime = RealTimeServiceRuntime(session_start_time=self._now())
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

