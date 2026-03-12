"""Real-time orchestration service with inference worker."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from streammuse.domain.interfaces import InferenceEngine, InputSource, OutputSink
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

    def _tick_loop(self, *, max_ticks: Optional[int]) -> None:
        """Main tick loop: timing, event playback, and inference triggering."""
        assert self._runtime is not None
        start = self._runtime.session_start_time
        tick = 0
        last_generation_tick = -self._generation_interval_ticks

        while self._running:
            if max_ticks is not None and tick >= max_ticks:
                self._running = False
                break

            # Target-time sync: sleep until the next tick boundary.
            target_time = start + self._tempo.tick_to_seconds(tick)
            delay = target_time - self._now()
            if delay > 0:
                self._sleep(delay)

            mt = MusicalTime.from_tick(tick, self._tempo)
            self._output.output_tick(tick=tick, bar=mt.bar, beat=mt.beat)

            # Drain input events and emit user events immediately.
            while True:
                try:
                    ev = self._event_q.get_nowait()
                except queue.Empty:
                    break
                self._output.output_event(ev, source="user")

            # Trigger inference at generation intervals.
            if tick - last_generation_tick >= self._generation_interval_ticks:
                with self._melody_history_lock:
                    melody_snapshot = self._melody_history.copy()
                if melody_snapshot:
                    self._inference_request_queue.put((tick, melody_snapshot))
                    last_generation_tick = tick

            # Process inference responses.
            while True:
                try:
                    acc_events, generation_start_tick = self._inference_response_queue.get_nowait()
                except queue.Empty:
                    break

                # Clear stale model events from this generation point onward.
                self._scheduler.clear_future_events(from_tick=generation_start_tick, source="model")

                # Schedule new accompaniment events.
                for ev in acc_events:
                    # Create new event with source="model"
                    ev_with_source = MusicalEvent(
                        tick=ev.tick,
                        pitch=ev.pitch,
                        event_type=ev.event_type,
                        velocity=ev.velocity,
                        channel=ev.channel,
                        program=ev.program,
                        is_placeholder=ev.is_placeholder,
                        source="model",  # Mark as model-generated event
                    )
                    if ev.tick >= tick:  # Only schedule future events
                        self._scheduler.schedule(ev_with_source, ev.tick)

            # Play scheduled events (if any).
            for ev in self._scheduler.get_events_at_tick(tick):
                self._output.output_event(ev, source=ev.source)

            tick += 1

    def _inference_worker(self) -> None:
        """Background worker that processes inference requests."""
        while self._running:
            try:
                generation_start_tick, melody_events = self._inference_request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._running:
                break

            try:
                request_send_time = self._now()
                # Call inference engine
                acc_events, timing_info = self._engine.generate_accompaniment(
                    melody_events=melody_events,
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
                    request_data = {
                        "generation_start_tick": generation_start_tick,
                        "melody_notes_count": len(melody_events),
                        "generation_length_frames": self._generation_length_frames,
                    }
                    response_data = {
                        "accompaniment_notes_count": len(acc_events),
                    }
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

