"""Continuation engine for Lekai prompt-continuation."""

from __future__ import annotations

from typing import Any, Optional

from streammuse.infrastructure.inference.lekai_http_backend import (
    BackendRuntimeConfig,
    EventPayload,
    LekaiHttpBackend,
    TimingPayload,
)


class LekaiContinuationEngine:
    """Load and run the continuation model after prompt accompaniment exists."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        backend: Optional[LekaiHttpBackend] = None,
    ) -> None:
        # Compatibility wrapper around the current Lekai runtime. Later this can
        # be replaced by a continuation-specific adapter without changing the
        # request-facing backend.
        self._backend = backend or LekaiHttpBackend(checkpoint_path=checkpoint_path)

    def configure(self, config: BackendRuntimeConfig) -> None:
        self._backend.configure(config)

    def runtime_info(self) -> dict[str, str | float | bool | None]:
        return self._backend.runtime_info()

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
        return self._backend.generate(
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
        return self._backend.inject_history(
            melody_events=melody_events,
            accompaniment_events=accompaniment_events,
            injection_length_ticks=injection_length_ticks,
        )

    def clear_history(self) -> dict[str, Any]:
        return self._backend.clear_history()

    def reset_session(self, seed: int) -> dict[str, Any]:
        return self._backend.reset_session(seed=int(seed))

    def injection_status(self) -> dict[str, bool | int | str]:
        return self._backend.injection_status()
