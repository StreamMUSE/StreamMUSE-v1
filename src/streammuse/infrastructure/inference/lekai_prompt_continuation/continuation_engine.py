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
        self._last_generation_bpm: Optional[int] = None

    def configure(self, config: BackendRuntimeConfig) -> None:
        self._backend.configure(config)

    def set_session_generation_config(
        self,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> None:
        self._backend.set_session_generation_config(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    def runtime_info(self) -> dict[str, str | float | bool | int | None]:
        info = dict(self._backend.runtime_info())
        info["last_generation_bpm"] = self._last_generation_bpm
        return info

    def generation_metadata_snapshot(self) -> list[dict[str, Any]]:
        return self._backend.generation_metadata_snapshot()

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
        bpm: Optional[int] = None,
    ) -> tuple[list[EventPayload], TimingPayload]:
        result = self._backend.generate(
            melody_events=melody_events,
            generation_start_tick=generation_start_tick,
            generation_length_frames=generation_length_frames,
            generation_interval_ticks=generation_interval_ticks,
            prompt_length_ticks=prompt_length_ticks,
            inference_mode=inference_mode,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            bpm=bpm,
        )
        if bpm is not None:
            self._last_generation_bpm = int(bpm)
        else:
            runtime_bpm = self._backend.runtime_info().get("effective_bpm")
            self._last_generation_bpm = (
                int(runtime_bpm) if runtime_bpm is not None else None
            )
        return result

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
        result = self._backend.clear_history()
        self._last_generation_bpm = None
        return result

    def reset_session(self, seed: int) -> dict[str, Any]:
        result = self._backend.reset_session(seed=int(seed))
        self._last_generation_bpm = None
        return result

    def injection_status(self) -> dict[str, bool | int | str]:
        return self._backend.injection_status()
