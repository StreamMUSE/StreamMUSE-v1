"""Single-process background scheduler for prompt-continuation catch-up.

This module keeps realtime coordination separate from HTTP request handling and
model wrappers. It uses one worker thread so model calls are serialized, while
the backend thread can continue accepting melody events.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from streammuse.infrastructure.inference.lekai_prompt_continuation.catchup_state import (
    CatchUpState,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.token_conversion import (
    copy_events,
)

if TYPE_CHECKING:
    from streammuse.infrastructure.inference.lekai_prompt_continuation.continuation_engine import (
        LekaiContinuationEngine,
    )
    from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_engine import (
        LekaiPromptEngine,
    )

EventPayload = dict[str, int | str]
TIMESTEPS_PER_BEAT = 4


class LekaiPromptContinuationScheduler:
    """Coordinate prompt generation and continuation catch-up in one process."""

    def __init__(
        self,
        *,
        prompt_engine: LekaiPromptEngine,
        continuation_engine: LekaiContinuationEngine,
        max_continuation_chunk_beats: int = 1,
    ) -> None:
        if int(max_continuation_chunk_beats) <= 0:
            raise ValueError("max_continuation_chunk_beats must be > 0")
        self._prompt_engine = prompt_engine
        self._continuation_engine = continuation_engine
        self._max_continuation_chunk_beats = int(max_continuation_chunk_beats)
        self._lock = threading.RLock()
        self._melody_condition = threading.Condition(self._lock)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lekai-prompt-continuation",
        )
        self._future: Optional[Future[None]] = None
        self._phase = "idle"
        self._error: Optional[str] = None
        self._melody_history: list[EventPayload] = []
        self._prompt_melody_input: list[EventPayload] = []
        self._prompt_accompaniment_history: list[EventPayload] = []
        self._accompaniment_history: list[EventPayload] = []
        self._pending_melody_events: list[EventPayload] = []
        self._melody_observed_until_tick = 0
        self._catchup_state = CatchUpState()
        self._prompt_length_ticks = 0
        self._generation_interval_ticks = TIMESTEPS_PER_BEAT
        self._inference_mode = "sliding_window"
        self._model_name = "lekai_prompt_continuation"
        self._checkpoint_path: Optional[str] = None
        self._effective_bpm: Optional[int] = None
        self._continuation_calls = 0
        self._last_continuation_event_count = 0
        self._last_continuation_note_on_count = 0
        self._last_continuation_min_tick: Optional[int] = None
        self._last_continuation_max_tick: Optional[int] = None
        self._empty_continuation_output_streak = 0
        self._run_id = 0

    @staticmethod
    def _ticks_to_beats(ticks: int) -> int:
        tick_count = max(0, int(ticks))
        return (tick_count + TIMESTEPS_PER_BEAT - 1) // TIMESTEPS_PER_BEAT

    @staticmethod
    def _max_event_tick(events: list[EventPayload]) -> int:
        if not events:
            return 0
        return max(int(event.get("tick", 0)) for event in events)

    @staticmethod
    def _validate_events_before_boundary(
        events: list[EventPayload],
        *,
        boundary_tick: int,
        label: str,
    ) -> None:
        for event in events:
            event_tick = int(event.get("tick", 0))
            if event_tick >= int(boundary_tick):
                raise ValueError(
                    f"{label} must contain only events with tick < "
                    f"observed_until_tick; got event tick {event_tick} "
                    f"for boundary {int(boundary_tick)}"
                )

    def _set_melody_observed_until(self, observed_until_tick: int) -> None:
        observed_tick = int(observed_until_tick)
        if observed_tick < 0:
            raise ValueError("observed_until_tick must be >= 0")
        if observed_tick < int(self._melody_observed_until_tick):
            raise ValueError(
                "observed_until_tick must be monotonic; "
                f"got {observed_tick} after {self._melody_observed_until_tick}"
            )
        self._melody_observed_until_tick = observed_tick
        self._catchup_state.melody_history_beats = max(
            self._catchup_state.melody_history_beats,
            self._ticks_to_beats(self._melody_observed_until_tick),
        )

    def start(
        self,
        *,
        melody_events: list[EventPayload],
        prompt_length_ticks: int,
        generation_interval_ticks: int,
        inference_mode: str,
        model_name: str,
        checkpoint_path: Optional[str],
        bpm: int,
        observed_until_tick: Optional[int] = None,
    ) -> dict[str, int | bool | str | None]:
        """Start prompt generation in the background.

        `observed_until_tick` is important when the user has played rests near
        the boundary. If omitted, the scheduler falls back to the largest event
        tick or the prompt length.
        """
        if int(prompt_length_ticks) <= 0:
            raise ValueError("prompt_length_ticks must be > 0")
        if int(generation_interval_ticks) <= 0:
            raise ValueError("generation_interval_ticks must be > 0")
        if int(bpm) <= 0:
            raise ValueError("bpm must be > 0")

        with self._lock:
            if self._future is not None and not self._future.done():
                raise RuntimeError("prompt-continuation scheduler is already running")

            self._phase = "prompt_running"
            self._error = None
            self._melody_history = copy_events(melody_events)
            self._prompt_melody_input = copy_events(melody_events)
            self._prompt_accompaniment_history = []
            self._accompaniment_history = []
            self._pending_melody_events = []
            self._melody_observed_until_tick = 0
            self._catchup_state.reset()
            self._prompt_length_ticks = int(prompt_length_ticks)
            self._generation_interval_ticks = int(generation_interval_ticks)
            self._inference_mode = str(inference_mode)
            self._model_name = str(model_name)
            self._checkpoint_path = checkpoint_path
            self._effective_bpm = int(bpm)
            self._continuation_calls = 0
            self._last_continuation_event_count = 0
            self._last_continuation_note_on_count = 0
            self._last_continuation_min_tick = None
            self._last_continuation_max_tick = None
            self._empty_continuation_output_streak = 0
            self._run_id += 1
            run_id = self._run_id

            observed_tick = (
                int(observed_until_tick)
                if observed_until_tick is not None
                else max(self._max_event_tick(self._melody_history), int(prompt_length_ticks))
            )
            if observed_tick < int(prompt_length_ticks):
                raise ValueError("observed_until_tick must cover prompt_length_ticks")
            self._validate_events_before_boundary(
                self._prompt_melody_input,
                boundary_tick=int(prompt_length_ticks),
                label="prompt melody_events",
            )
            self._set_melody_observed_until(observed_tick)
            self._future = self._executor.submit(self._run_prompt_then_catchup, run_id)
            return self.status()

    def append_melody(
        self,
        melody_events: list[EventPayload],
        *,
        observed_until_tick: Optional[int] = None,
    ) -> dict[str, int | bool | str | None]:
        """Append user melody while prompt or continuation is running."""
        with self._lock:
            copied_events = copy_events(melody_events)
            previous_observed_tick = int(self._melody_observed_until_tick)
            observed_tick = (
                int(observed_until_tick)
                if observed_until_tick is not None
                else max(
                    int(self._melody_observed_until_tick),
                    self._max_event_tick(copied_events) + (1 if copied_events else 0),
                )
            )
            self._validate_events_before_boundary(
                copied_events,
                boundary_tick=observed_tick,
                label="append melody_events",
            )
            for event in copied_events:
                event_tick = int(event.get("tick", 0))
                if event_tick < previous_observed_tick:
                    raise ValueError(
                        "append melody_events must have ticks in "
                        "[previous_observed_until_tick, observed_until_tick); "
                        f"got event tick {event_tick} before "
                        f"{previous_observed_tick}"
                    )
            self._melody_history.extend(copy_events(copied_events))
            self._pending_melody_events.extend(copy_events(copied_events))
            self._set_melody_observed_until(observed_tick)
            self._melody_condition.notify_all()
            if (
                self._phase == "ready"
                and (self._future is None or self._future.done())
                and self._catchup_state.beats_needed_for_playback() > 0
            ):
                # Prompt generation can finish before the client sends melody
                # observed after the prompt window. In that case a later append
                # must restart catch-up instead of leaving a stale "ready" phase.
                self._phase = "catchup_running"
                self._future = self._executor.submit(self._run_catchup_loop, self._run_id)
            return self.status()

    def wait(self, timeout: Optional[float] = None) -> dict[str, int | bool | str | None]:
        future: Optional[Future[None]]
        with self._lock:
            future = self._future
        if future is not None:
            future.result(timeout=timeout)
        return self.status()

    def clear(self) -> dict[str, int | bool | str | None]:
        with self._lock:
            self._phase = "idle"
            self._error = None
            self._future = None
            self._run_id += 1
            self._melody_history = []
            self._prompt_melody_input = []
            self._prompt_accompaniment_history = []
            self._accompaniment_history = []
            self._pending_melody_events = []
            self._melody_observed_until_tick = 0
            self._catchup_state.reset()
            self._continuation_calls = 0
            self._last_continuation_event_count = 0
            self._last_continuation_note_on_count = 0
            self._last_continuation_min_tick = None
            self._last_continuation_max_tick = None
            self._empty_continuation_output_streak = 0
            self._effective_bpm = None
            self._melody_condition.notify_all()
            return self.status()

    def drain_and_clear(self) -> dict[str, int | bool | str | None]:
        """Invalidate current work, wait for its worker, then clear all state."""

        with self._lock:
            self._run_id += 1
            future = self._future
            self._melody_condition.notify_all()
        if future is not None:
            try:
                future.result()
            except Exception:
                # A failed retired run must not prevent the next session reset.
                pass
        with self._lock:
            self._phase = "idle"
            self._error = None
            self._future = None
            self._melody_history = []
            self._prompt_melody_input = []
            self._prompt_accompaniment_history = []
            self._accompaniment_history = []
            self._pending_melody_events = []
            self._melody_observed_until_tick = 0
            self._catchup_state.reset()
            self._continuation_calls = 0
            self._last_continuation_event_count = 0
            self._last_continuation_note_on_count = 0
            self._last_continuation_min_tick = None
            self._last_continuation_max_tick = None
            self._empty_continuation_output_streak = 0
            self._effective_bpm = None
            return self.status()

    def shutdown(self) -> None:
        with self._lock:
            self._run_id += 1
            self._melody_condition.notify_all()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def status(self) -> dict[str, int | bool | str | None]:
        with self._lock:
            future_running = self._future is not None and not self._future.done()
            snapshot = self._catchup_state.snapshot()
            return {
                "phase": self._phase,
                "is_running": bool(future_running),
                "is_failed": self._phase == "failed",
                "error": self._error,
                "melody_event_count": len(self._melody_history),
                "accompaniment_event_count": len(self._accompaniment_history),
                "melody_observed_until_tick": int(self._melody_observed_until_tick),
                "pending_melody_event_count": len(self._pending_melody_events),
                "prompt_length_ticks": int(self._prompt_length_ticks),
                "generation_interval_ticks": int(self._generation_interval_ticks),
                "effective_bpm": self._effective_bpm,
                "continuation_calls": int(self._continuation_calls),
                "last_continuation_event_count": int(self._last_continuation_event_count),
                "last_continuation_note_on_count": int(self._last_continuation_note_on_count),
                "last_continuation_min_tick": self._last_continuation_min_tick,
                "last_continuation_max_tick": self._last_continuation_max_tick,
                "empty_continuation_output_streak": int(
                    self._empty_continuation_output_streak
                ),
                **snapshot,
            }

    def playable_accompaniment(self) -> list[EventPayload]:
        with self._lock:
            if not self._catchup_state.is_playback_ready():
                return []
            return copy_events(self._accompaniment_history)

    def raw_accompaniment_history(self) -> list[EventPayload]:
        with self._lock:
            return copy_events(self._accompaniment_history)

    def prompt_accompaniment_history(self) -> list[EventPayload]:
        with self._lock:
            return copy_events(self._prompt_accompaniment_history)

    def _is_current_run(self, run_id: int) -> bool:
        return int(run_id) == int(self._run_id)

    def _run_prompt_then_catchup(self, run_id: int) -> None:
        try:
            with self._lock:
                if not self._is_current_run(run_id):
                    return
                prompt_melody_input = copy_events(self._prompt_melody_input)
                prompt_length_ticks = int(self._prompt_length_ticks)
                effective_bpm = self._effective_bpm

            if effective_bpm is None:
                raise RuntimeError("prompt-continuation session BPM is not initialized")

            prompt_accompaniment = self._prompt_engine.generate_prompt_accompaniment(
                melody_events=prompt_melody_input,
                prompt_start_tick=0,
                prompt_length_ticks=prompt_length_ticks,
                bpm=effective_bpm,
            )
            generated_acc_beats = (
                self._prompt_engine.last_generated_acc_beats()
                if hasattr(self._prompt_engine, "last_generated_acc_beats")
                else self._ticks_to_beats(prompt_length_ticks)
            )
            actual_prompt_length_ticks = min(
                prompt_length_ticks,
                max(0, int(generated_acc_beats) * TIMESTEPS_PER_BEAT),
            )

            with self._lock:
                if not self._is_current_run(run_id):
                    return
                self._prompt_accompaniment_history = copy_events(prompt_accompaniment)
                self._accompaniment_history = copy_events(prompt_accompaniment)
                self._catchup_state.accompaniment_history_beats = max(
                    self._catchup_state.accompaniment_history_beats,
                    self._ticks_to_beats(actual_prompt_length_ticks),
                )
                self._phase = "catchup_running"
                prompt_melody_snapshot = [
                    event
                    for event in copy_events(self._prompt_melody_input)
                    if int(event.get("tick", 0)) < actual_prompt_length_ticks
                ]
                frozen_future_events = [
                    event
                    for event in copy_events(self._prompt_melody_input)
                    if int(event.get("tick", 0)) >= actual_prompt_length_ticks
                ]
                if frozen_future_events:
                    self._pending_melody_events = (
                        frozen_future_events + self._pending_melody_events
                    )

            self._continuation_engine.inject_history(
                melody_events=prompt_melody_snapshot,
                accompaniment_events=prompt_accompaniment,
                injection_length_ticks=actual_prompt_length_ticks,
            )

            self._run_catchup_loop(run_id)
        except Exception as exc:
            with self._lock:
                if self._is_current_run(run_id):
                    self._phase = "failed"
                    self._error = f"{type(exc).__name__}: {exc}"
            raise

    def _run_catchup_loop(self, run_id: int) -> None:
        while True:
            with self._lock:
                if not self._is_current_run(run_id):
                    return
                beats_needed = int(self._catchup_state.beats_needed_for_playback())
                if beats_needed <= 0:
                    self._phase = "ready"
                    return
                chunk_beats = min(self._max_continuation_chunk_beats, beats_needed)
                generation_start_tick = (
                    int(self._catchup_state.accompaniment_history_beats) * TIMESTEPS_PER_BEAT
                )
                while (
                    self._is_current_run(run_id)
                    and int(self._melody_observed_until_tick) < generation_start_tick
                ):
                    self._melody_condition.wait()
                if not self._is_current_run(run_id):
                    return
                melody_increment = [
                    event
                    for event in copy_events(self._pending_melody_events)
                    if int(event.get("tick", 0)) < generation_start_tick
                ]
                self._pending_melody_events = [
                    event
                    for event in self._pending_melody_events
                    if int(event.get("tick", 0)) >= generation_start_tick
                ]
                generation_interval_ticks = int(self._generation_interval_ticks)
                inference_mode = str(self._inference_mode)
                model_name = str(self._model_name)
                checkpoint_path = self._checkpoint_path
                effective_bpm = self._effective_bpm

            if effective_bpm is None:
                raise RuntimeError("prompt-continuation session BPM is not initialized")

            accompaniment, _timings = self._continuation_engine.generate(
                melody_events=melody_increment,
                generation_start_tick=generation_start_tick,
                generation_length_frames=chunk_beats * TIMESTEPS_PER_BEAT,
                generation_interval_ticks=generation_interval_ticks,
                prompt_length_ticks=self._prompt_length_ticks,
                inference_mode=inference_mode,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                bpm=effective_bpm,
            )

            with self._lock:
                if not self._is_current_run(run_id):
                    return
                continuation_event_ticks = [
                    int(event.get("tick", 0)) for event in accompaniment
                ]
                continuation_note_on_count = sum(
                    1
                    for event in accompaniment
                    if str(event.get("type", "")) == "note_on"
                )
                self._accompaniment_history.extend(copy_events(accompaniment))
                self._catchup_state.accept_continuation_beats(chunk_beats)
                self._continuation_calls += 1
                self._last_continuation_event_count = int(len(accompaniment))
                self._last_continuation_note_on_count = int(continuation_note_on_count)
                self._last_continuation_min_tick = (
                    min(continuation_event_ticks) if continuation_event_ticks else None
                )
                self._last_continuation_max_tick = (
                    max(continuation_event_ticks) if continuation_event_ticks else None
                )
                if accompaniment:
                    self._empty_continuation_output_streak = 0
                else:
                    self._empty_continuation_output_streak += 1
