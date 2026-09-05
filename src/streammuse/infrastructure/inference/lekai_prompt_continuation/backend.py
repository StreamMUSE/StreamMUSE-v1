"""HTTP backend for the Lekai prompt-continuation inference path.

The backend is the request-facing boundary. It should handle API-facing state,
validation-adjacent concerns, and response contracts, while model loading and
generation live in the engine layer.

The first version keeps the public backend contract in place and delegates runtime
behavior to LekaiPromptContinuationEngine until the real prompt and continuation
model wrappers are wired in.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from streammuse.infrastructure.inference.lekai_http_backend import (
    BackendRuntimeConfig,
    EventPayload,
    TimingPayload,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.engine import (
    LekaiPromptContinuationEngine,
)


def _make_prompt_continuation_engine(
    *,
    checkpoint_path: Optional[str],
    prompt_checkpoint_path: Optional[str],
    continuation_checkpoint_path: Optional[str],
) -> LekaiPromptContinuationEngine:
    variant = os.environ.get("LEKAI_PROMPT_CONTINUATION_ENGINE", "standard").strip().lower()
    if variant in {"prompt_extension", "extension", "extend_prompt", "bridge", "prompt_bridge", "extra_beat", "prompt_extra_beat"}:
        from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_extension_engine import (
            LekaiPromptExtensionContinuationEngine,
        )

        return LekaiPromptExtensionContinuationEngine(
            checkpoint_path=checkpoint_path,
            prompt_checkpoint_path=prompt_checkpoint_path,
            continuation_checkpoint_path=continuation_checkpoint_path,
        )
    return LekaiPromptContinuationEngine(
        checkpoint_path=checkpoint_path,
        prompt_checkpoint_path=prompt_checkpoint_path,
        continuation_checkpoint_path=continuation_checkpoint_path,
    )


class LekaiPromptContinuationBackend:
    """Server-side backend for prompt accompaniment plus continuation generation."""

    MODEL_NAME = "lekai_prompt_continuation"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        prompt_checkpoint_path: Optional[str] = None,
        continuation_checkpoint_path: Optional[str] = None,
        engine: Optional[LekaiPromptContinuationEngine] = None,
    ) -> None:
        self._engine = engine or _make_prompt_continuation_engine(
            checkpoint_path=checkpoint_path,
            prompt_checkpoint_path=prompt_checkpoint_path,
            continuation_checkpoint_path=continuation_checkpoint_path,
        )

    def configure(self, config: BackendRuntimeConfig) -> None:
        self._engine.configure(config)

    def runtime_info(self) -> dict[str, str | float | bool | int | None]:
        info = dict(self._engine.runtime_info())
        info["runtime_model_name"] = self.MODEL_NAME
        return info

    def prompt_generation_log(self) -> dict[str, Any]:
        return self._engine.prompt_generation_log()

    def replay_audit(self) -> dict[str, Any]:
        return self._engine.replay_audit()

    def catchup_status(self) -> dict[str, int | bool]:
        return self._engine.catchup_status()

    def start_prompt_catchup(
        self,
        *,
        melody_events: list[EventPayload],
        prompt_length_ticks: int,
        generation_interval_ticks: int,
        inference_mode: str,
        model_name: str,
        checkpoint_path: Optional[str],
        bpm: Optional[int] = None,
        observed_until_tick: Optional[int] = None,
        prompt_selection_mode: Optional[str] = None,
        prompt_batch_candidates: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> dict[str, int | bool | str | None]:
        return self._engine.start_prompt_catchup(
            melody_events=melody_events,
            prompt_length_ticks=prompt_length_ticks,
            generation_interval_ticks=generation_interval_ticks,
            inference_mode=inference_mode,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            bpm=bpm,
            observed_until_tick=observed_until_tick,
            prompt_selection_mode=prompt_selection_mode,
            prompt_batch_candidates=prompt_batch_candidates,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    def append_melody_events(
        self,
        melody_events: list[EventPayload],
        *,
        observed_until_tick: Optional[int] = None,
    ) -> dict[str, int | bool | str | None]:
        return self._engine.append_melody_events(
            melody_events=melody_events,
            observed_until_tick=observed_until_tick,
        )

    def scheduler_status(self) -> dict[str, int | bool | str | None]:
        return self._engine.scheduler_status()

    def wait_for_scheduler(self, timeout: Optional[float] = None) -> dict[str, int | bool | str | None]:
        return self._engine.wait_for_scheduler(timeout=timeout)

    def playable_accompaniment(self) -> list[EventPayload]:
        return self._engine.playable_accompaniment()

    def raw_accompaniment_history(self) -> list[EventPayload]:
        return self._engine.raw_accompaniment_history()

    def prompt_accompaniment_history(self) -> list[EventPayload]:
        return self._engine.prompt_accompaniment_history()

    def generate(
        self,
        melody_events: list[EventPayload],
        generation_start_tick: int,
        generation_length_frames: int,
        generation_interval_ticks: int,
        prompt_length_ticks: Optional[int],
        inference_mode: str,
        model_name: str,
        checkpoint_path: Optional[str],
    ) -> tuple[list[EventPayload], TimingPayload]:
        return self._engine.generate(
            melody_events=melody_events,
            generation_start_tick=generation_start_tick,
            generation_length_frames=generation_length_frames,
            generation_interval_ticks=generation_interval_ticks,
            prompt_length_ticks=prompt_length_ticks,
            inference_mode=inference_mode,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
        )

    def inject_history(
        self,
        melody_events: list[EventPayload],
        accompaniment_events: list[EventPayload],
        injection_length_ticks: int,
    ) -> dict[str, int | bool | str]:
        return self._engine.inject_history(
            melody_events=melody_events,
            accompaniment_events=accompaniment_events,
            injection_length_ticks=injection_length_ticks,
        )

    def clear_history(self) -> dict[str, Any]:
        return self._engine.clear_history()

    def reset_session(self, *, prompt_seed: int, continuation_seed: int) -> dict[str, Any]:
        return self._engine.reset_session(
            prompt_seed=int(prompt_seed),
            continuation_seed=int(continuation_seed),
        )

    def injection_status(self) -> dict[str, bool | int | str]:
        status = dict(self._engine.injection_status())
        status["runtime_model_name"] = self.MODEL_NAME
        return status
