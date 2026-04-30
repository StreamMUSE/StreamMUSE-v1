"""Single-process background scheduler for prompt-continuation catch-up.

This module keeps realtime coordination separate from HTTP request handling and
model wrappers. It uses one worker thread so model calls are serialized, while
the backend thread can continue accepting melody events.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from streammuse.infrastructure.inference.lekai_http_backend import (
    EventPayload,
    TIMESTEPS_PER_BEAT,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.catchup_state import (
    CatchUpState,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.continuation_engine import (
    LekaiContinuationEngine,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_engine import (
    LekaiPromptEngine,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.token_conversion import (
    copy_events,
)


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
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lekai-prompt-continuation",
        )
        self._future: Optional[Future[None]] = None
        self._phase = "idle"
        self._error: Optional[str] = None
        self._melody_history: list[EventPayload] = []
        self._prompt_melody_input: list[EventPayload] = []
        self._accompaniment_history: list[EventPayload] = []
        self._catchup_state = CatchUpState()
        self._prompt_length_ticks = 0
        self._generation_interval_ticks = TIMESTEPS_PER_BEAT
        self._inference_mode = "sliding_window"
        self._model_name = "lekai_prompt_continuation"
        self._checkpoint_path: Optional[str] = None
        self._continuation_calls = 0
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

    def _set_melody_observed_until(self, observed_until_tick: int) -> None:
        self._catchup_state.melody_history_beats = max(
            self._catchup_state.melody_history_beats,
            self._ticks_to_beats(observed_until_tick),
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

        with self._lock:
            if self._future is not None and not self._future.done():
                raise RuntimeError("prompt-continuation scheduler is already running")

            self._phase = "prompt_running"
            self._error = None
            self._melody_history = copy_events(melody_events)
            self._prompt_melody_input = copy_events(melody_events)
            self._accompaniment_history = []
            self._catchup_state.reset()
            self._prompt_length_ticks = int(prompt_length_ticks)
            self._generation_interval_ticks = int(generation_interval_ticks)
            self._inference_mode = str(inference_mode)
            self._model_name = str(model_name)
            self._checkpoint_path = checkpoint_path
            self._continuation_calls = 0
            self._run_id += 1
            run_id = self._run_id

            observed_tick = (
                int(observed_until_tick)
                if observed_until_tick is not None
                else max(self._max_event_tick(self._melody_history), int(prompt_length_ticks))
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
            self._melody_history.extend(copy_events(melody_events))
            observed_tick = (
                int(observed_until_tick)
                if observed_until_tick is not None
                else self._max_event_tick(self._melody_history)
            )
            self._set_melody_observed_until(observed_tick)
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
            self._accompaniment_history = []
            self._catchup_state.reset()
            self._continuation_calls = 0
            return self.status()

    def shutdown(self) -> None:
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
                "prompt_length_ticks": int(self._prompt_length_ticks),
                "generation_interval_ticks": int(self._generation_interval_ticks),
                "continuation_calls": int(self._continuation_calls),
                **snapshot,
            }

    def playable_accompaniment(self) -> list[EventPayload]:
        with self._lock:
            if not self._catchup_state.is_playback_ready():
                return []
            return copy_events(self._accompaniment_history)

    def _is_current_run(self, run_id: int) -> bool:
        return int(run_id) == int(self._run_id)

    def _run_prompt_then_catchup(self, run_id: int) -> None:
        try:
            with self._lock:
                if not self._is_current_run(run_id):
                    return
                prompt_melody_input = copy_events(self._prompt_melody_input)
                prompt_length_ticks = int(self._prompt_length_ticks)

            prompt_accompaniment = self._prompt_engine.generate_prompt_accompaniment(
                melody_events=prompt_melody_input,
                prompt_start_tick=0,
                prompt_length_ticks=prompt_length_ticks,
            )

            with self._lock:
                if not self._is_current_run(run_id):
                    return
                self._accompaniment_history = copy_events(prompt_accompaniment)
                self._catchup_state.accompaniment_history_beats = max(
                    self._catchup_state.accompaniment_history_beats,
                    self._ticks_to_beats(prompt_length_ticks),
                )
                self._phase = "catchup_running"
                melody_snapshot = copy_events(self._melody_history)

            if prompt_accompaniment:
                self._continuation_engine.inject_history(
                    melody_events=melody_snapshot,
                    accompaniment_events=prompt_accompaniment,
                    injection_length_ticks=prompt_length_ticks,
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
                melody_snapshot = copy_events(self._melody_history)
                generation_interval_ticks = int(self._generation_interval_ticks)
                inference_mode = str(self._inference_mode)
                model_name = str(self._model_name)
                checkpoint_path = self._checkpoint_path

            accompaniment, _timings = self._continuation_engine.generate(
                melody_events=melody_snapshot,
                generation_start_tick=generation_start_tick,
                generation_length_frames=chunk_beats * TIMESTEPS_PER_BEAT,
                generation_interval_ticks=generation_interval_ticks,
                prompt_length_ticks=self._prompt_length_ticks,
                inference_mode=inference_mode,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
            )

            with self._lock:
                if not self._is_current_run(run_id):
                    return
                self._accompaniment_history.extend(copy_events(accompaniment))
                self._catchup_state.accept_continuation_beats(chunk_beats)
                self._continuation_calls += 1
