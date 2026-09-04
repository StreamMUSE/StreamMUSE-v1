"""Prompt-extension scheduler variant for prompt-continuation catch-up.

This variant asks the prompt model to generate accompaniment beyond the observed
prompt melody window. The extra prompt-model output is intentional: the prompt
model was trained to produce that next beat, and this variant tests using that
capacity instead of discarding it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from streammuse.infrastructure.inference.lekai_prompt_continuation.scheduler import (
    TIMESTEPS_PER_BEAT,
    LekaiPromptContinuationScheduler,
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


class LekaiPromptExtensionContinuationScheduler(LekaiPromptContinuationScheduler):
    """Prompt-continuation scheduler that keeps prompt-generated extension beats."""

    def __init__(
        self,
        *,
        prompt_engine: LekaiPromptEngine,
        continuation_engine: LekaiContinuationEngine,
        max_continuation_chunk_beats: int = 1,
        prompt_extension_ticks: int = TIMESTEPS_PER_BEAT,
    ) -> None:
        if int(prompt_extension_ticks) < 0:
            raise ValueError("prompt_extension_ticks must be >= 0")
        super().__init__(
            prompt_engine=prompt_engine,
            continuation_engine=continuation_engine,
            max_continuation_chunk_beats=max_continuation_chunk_beats,
        )
        self._prompt_extension_ticks = int(prompt_extension_ticks)

    @property
    def prompt_extension_ticks(self) -> int:
        return int(self._prompt_extension_ticks)

    def status(self) -> dict[str, int | bool | str | None]:
        status = super().status()
        status["prompt_extension_ticks"] = int(self._prompt_extension_ticks)
        return status

    def _run_prompt_then_catchup(self, run_id: int) -> None:
        try:
            with self._lock:
                if not self._is_current_run(run_id):
                    return
                prompt_melody_input = copy_events(self._prompt_melody_input)
                prompt_length_ticks = int(self._prompt_length_ticks)
                prompt_generation_length_ticks = prompt_length_ticks + int(self._prompt_extension_ticks)
                effective_bpm = self._effective_bpm

            if effective_bpm is None:
                raise RuntimeError("prompt-continuation session BPM is not initialized")

            prompt_accompaniment = self._prompt_engine.generate_prompt_accompaniment(
                melody_events=prompt_melody_input,
                prompt_start_tick=0,
                prompt_length_ticks=prompt_generation_length_ticks,
                bpm=effective_bpm,
            )
            generated_acc_beats = (
                self._prompt_engine.last_generated_acc_beats()
                if hasattr(self._prompt_engine, "last_generated_acc_beats")
                else self._ticks_to_beats(prompt_generation_length_ticks)
            )
            actual_prompt_length_ticks = min(
                prompt_generation_length_ticks,
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
                melody_snapshot = copy_events(self._melody_history)
                self._continuation_sent_melody_event_count = len(self._melody_history)

            self._continuation_engine.inject_history(
                melody_events=melody_snapshot,
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
