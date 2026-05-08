"""Realtime client-side orchestration for Lekai prompt-continuation mode."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from streammuse.domain.interfaces import InputSource, OutputSink
from streammuse.domain.musical import MusicalEvent
from streammuse.domain.timing import MusicalTime, PlaybackScheduler, Tempo


class PromptContinuationClient(Protocol):
    def clear_history(self) -> dict[str, Any]: ...

    def start(
        self,
        *,
        melody_events: list[MusicalEvent],
        prompt_length_ticks: int,
        generation_interval_ticks: int,
        observed_until_tick: int,
    ) -> dict[str, Any]: ...

    def append_melody(
        self,
        *,
        melody_events: list[MusicalEvent],
        observed_until_tick: int,
    ) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]: ...

    def playable(self) -> tuple[list[MusicalEvent], dict[str, Any]]: ...


@dataclass(frozen=True)
class PromptContinuationRuntime:
    session_start_time: float


@dataclass(frozen=True)
class _ControlAction:
    kind: str
    melody_events: list[MusicalEvent]
    observed_until_tick: int


class PromptContinuationRealtimeService:
    """Drive prompt-continuation endpoints from a realtime input source.

    Intended first use: frontend/client reads a melody MIDI file via `MidiFileInput`,
    stamps it as if it were user input, and talks to the special backend protocol:
    start prompt generation after the prompt window, keep appending melody progress,
    poll until catch-up is ready, then schedule playable accompaniment.
    """

    _INPUT_BUFFER_RATIO = 0.1

    def __init__(
        self,
        *,
        input_source: InputSource,
        prompt_client: PromptContinuationClient,
        output_sink: OutputSink,
        tempo: Tempo,
        scheduler: PlaybackScheduler,
        prompt_length_ticks: int = 32,
        generation_interval_ticks: int = 4,
        protocol_poll_interval_s: float = 0.05,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if int(prompt_length_ticks) <= 0:
            raise ValueError("prompt_length_ticks must be > 0")
        if int(generation_interval_ticks) <= 0:
            raise ValueError("generation_interval_ticks must be > 0")
        self._input = input_source
        self._client = prompt_client
        self._output = output_sink
        self._tempo = tempo
        self._scheduler = scheduler
        self._prompt_length_ticks = int(prompt_length_ticks)
        self._generation_interval_ticks = int(generation_interval_ticks)
        self._protocol_poll_interval_s = float(protocol_poll_interval_s)
        self._now = now
        self._sleep = sleep

        self._running = False
        self._runtime: PromptContinuationRuntime | None = None
        self._input_thread: threading.Thread | None = None
        self._tick_thread: threading.Thread | None = None
        self._protocol_thread: threading.Thread | None = None

        self._event_q: queue.Queue[MusicalEvent] = queue.Queue()
        self._control_q: queue.Queue[_ControlAction] = queue.Queue()
        self._playable_q: queue.Queue[tuple[list[MusicalEvent], dict[str, Any]]] = queue.Queue()

        self._prompt_events: list[MusicalEvent] = []
        self._pending_append_events: list[MusicalEvent] = []
        self._start_enqueued = False
        self._last_append_observed_tick = 0
        self._scheduled_model_event_keys: set[tuple[int, int, str, int]] = set()
        self._append_generation = 0
        self._last_playable_marker: tuple[int, int, int] | None = None
        self._protocol_started = False
        self._append_sent_after_prompt = False

    @property
    def running(self) -> bool:
        return self._running

    def _input_worker(self) -> None:
        assert self._runtime is not None
        start = self._runtime.session_start_time
        for ev in self._input.read_events():
            if not self._running:
                break
            elapsed = self._now() - start
            tick = self._tempo.seconds_to_tick(elapsed)
            stamped = MusicalEvent(
                tick=tick,
                pitch=ev.pitch,
                event_type=ev.event_type,
                velocity=ev.velocity,
                channel=ev.channel,
                program=ev.program,
                is_placeholder=ev.is_placeholder,
                source="user",
            )
            self._event_q.put(stamped)

    def _protocol_worker(self) -> None:
        try:
            self._client.clear_history()
        except Exception as exc:
            self._output.output_status("error", f"Prompt-continuation clear_history failed: {exc}")

        while self._running:
            try:
                action = self._control_q.get(timeout=self._protocol_poll_interval_s)
            except queue.Empty:
                action = None

            if action is not None:
                try:
                    if action.kind == "start":
                        self._client.start(
                            melody_events=action.melody_events,
                            prompt_length_ticks=self._prompt_length_ticks,
                            generation_interval_ticks=self._generation_interval_ticks,
                            observed_until_tick=action.observed_until_tick,
                        )
                        self._protocol_started = True
                        self._output.output_status("prompt_running", "Prompt-continuation start sent")
                    elif action.kind == "append":
                        self._client.append_melody(
                            melody_events=action.melody_events,
                            observed_until_tick=action.observed_until_tick,
                        )
                        if int(action.observed_until_tick) > self._prompt_length_ticks:
                            self._append_sent_after_prompt = True
                            self._append_generation += 1
                    else:
                        self._output.output_status("error", f"Unknown prompt-continuation action: {action.kind}")
                except Exception as exc:
                    self._output.output_status("error", f"Prompt-continuation {action.kind} failed: {exc}")

            if self._protocol_started and self._append_sent_after_prompt:
                try:
                    status = self._client.status()
                    if status.get("is_failed"):
                        self._output.output_status("error", f"Prompt-continuation failed: {status.get('error')}")
                    elif status.get("is_playback_ready"):
                        marker = (
                            int(self._append_generation),
                            int(status.get("accompaniment_event_count", 0) or 0),
                            int(status.get("continuation_calls", 0) or 0),
                        )
                        if marker != self._last_playable_marker:
                            accompaniment, playable_status = self._client.playable()
                            self._playable_q.put((accompaniment, playable_status))
                            self._last_playable_marker = marker
                            self._output.output_status("ready", "Prompt-continuation accompaniment is playable")
                except Exception as exc:
                    self._output.output_status("error", f"Prompt-continuation status/playable failed: {exc}")
                    self._sleep(self._protocol_poll_interval_s)

    def _drain_user_events(self) -> list[MusicalEvent]:
        drained: list[MusicalEvent] = []
        while True:
            try:
                event = self._event_q.get_nowait()
            except queue.Empty:
                break
            drained.append(event)
            self._output.output_event(event, source="user")
            if int(event.tick) < self._prompt_length_ticks:
                self._prompt_events.append(event)
            else:
                self._pending_append_events.append(event)
        return drained

    def _maybe_enqueue_start(self, observed_until_tick: int) -> None:
        if self._start_enqueued:
            return
        if int(observed_until_tick) < self._prompt_length_ticks:
            return
        self._control_q.put(
            _ControlAction(
                kind="start",
                melody_events=list(self._prompt_events),
                observed_until_tick=self._prompt_length_ticks,
            )
        )
        self._start_enqueued = True
        self._last_append_observed_tick = self._prompt_length_ticks

    def _maybe_enqueue_append(self, observed_until_tick: int) -> None:
        if not self._start_enqueued:
            return
        observed_until_tick = int(observed_until_tick)
        if observed_until_tick <= self._last_append_observed_tick:
            return
        if (observed_until_tick - self._prompt_length_ticks) % self._generation_interval_ticks != 0:
            return
        self._control_q.put(
            _ControlAction(
                kind="append",
                melody_events=list(self._pending_append_events),
                observed_until_tick=observed_until_tick,
            )
        )
        self._pending_append_events = []
        self._last_append_observed_tick = observed_until_tick

    def _schedule_playable(self, accompaniment: list[MusicalEvent], *, current_tick: int) -> None:
        scheduled = 0
        dropped_past = 0
        skipped_duplicate = 0
        for event in accompaniment:
            if int(event.tick) < int(current_tick):
                dropped_past += 1
                continue
            event_key = (
                int(event.tick),
                int(event.pitch),
                str(event.event_type.value),
                int(event.velocity),
            )
            if event_key in self._scheduled_model_event_keys:
                skipped_duplicate += 1
                continue
            model_event = MusicalEvent(
                tick=event.tick,
                pitch=event.pitch,
                event_type=event.event_type,
                velocity=event.velocity,
                channel=event.channel,
                program=event.program,
                is_placeholder=event.is_placeholder,
                source="model",
                backup_level=max(0, int(event.tick) - int(current_tick)),
            )
            self._scheduler.schedule(model_event, int(event.tick))
            self._scheduled_model_event_keys.add(event_key)
            scheduled += 1
        self._output.output_status(
            "ready",
            f"Scheduled {scheduled} playable accompaniment event(s); "
            f"dropped {dropped_past} past event(s); "
            f"skipped {skipped_duplicate} duplicate event(s).",
        )

    def _tick_loop(self, *, max_ticks: int | None) -> None:
        assert self._runtime is not None
        start = self._runtime.session_start_time
        tick = 0
        while self._running:
            if max_ticks is not None and tick >= max_ticks:
                self._running = False
                break

            target_time = start + self._tempo.tick_to_seconds(tick)
            delay = target_time - self._now()
            if delay > 0:
                self._sleep(delay)

            mt = MusicalTime.from_tick(tick, self._tempo)
            self._output.output_tick(tick=tick, bar=mt.bar, beat=mt.beat)
            self._sleep(self._tempo.seconds_per_tick * self._INPUT_BUFFER_RATIO)

            self._drain_user_events()
            observed_until_tick = tick + 1
            self._maybe_enqueue_start(observed_until_tick)
            self._maybe_enqueue_append(observed_until_tick)

            while True:
                try:
                    accompaniment, _status = self._playable_q.get_nowait()
                except queue.Empty:
                    break
                self._schedule_playable(accompaniment, current_tick=tick)

            for event in self._scheduler.get_events_at_tick(tick):
                self._output.output_event(event, source=event.source)

            tick += 1

    def start(self, *, max_ticks: int | None = None) -> None:
        if self._running:
            return
        self._running = True
        self._runtime = PromptContinuationRuntime(session_start_time=self._now())
        self._output.output_status("running", "prompt-continuation")

        self._input_thread = threading.Thread(target=self._input_worker, daemon=True)
        self._tick_thread = threading.Thread(target=self._tick_loop, kwargs={"max_ticks": max_ticks}, daemon=True)
        self._protocol_thread = threading.Thread(target=self._protocol_worker, daemon=True)

        self._input_thread.start()
        self._tick_thread.start()
        self._protocol_thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._input.close()
        except Exception:
            pass
        try:
            self._output.output_status("stopped", "")
            self._output.close()
        except Exception:
            pass

        if self._protocol_thread is not None:
            self._protocol_thread.join(timeout=1.0)
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=1.0)
        if self._input_thread is not None:
            self._input_thread.join(timeout=1.0)
