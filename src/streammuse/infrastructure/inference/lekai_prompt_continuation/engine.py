"""Inference engine for the Lekai prompt-continuation path.

The engine owns model/runtime work. The backend should call this class instead of
loading models or constructing inference inputs directly.
"""

from __future__ import annotations

import copy
import os
import threading
from typing import Any, Optional

from streammuse.infrastructure.inference.lekai_http_backend import (
    BackendRuntimeConfig,
    EventPayload,
    TIMESTEPS_PER_BEAT,
    TimingPayload,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.catchup_state import (
    CatchUpState,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.continuation_engine import (
    LekaiContinuationEngine,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_engine import (
    LekaiPromptEngine,
    resolve_prompt_bpm,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.scheduler import (
    LekaiPromptContinuationScheduler,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.token_conversion import (
    PromptContinuationRequest,
    copy_events,
)
from streammuse.infrastructure.inference.runtime_device import parse_env_bool


class LekaiPromptContinuationEngine:
    """Run the two-stage prompt accompaniment plus continuation inference flow."""

    MODEL_NAME = "lekai_prompt_continuation"

    def __init__(
        self,
        prompt_checkpoint_path: Optional[str] = None,
        continuation_checkpoint_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        prompt_engine: Optional[LekaiPromptEngine] = None,
        continuation_engine: Optional[LekaiContinuationEngine] = None,
    ) -> None:
        resolved_continuation_checkpoint = continuation_checkpoint_path or checkpoint_path
        self._prompt_engine = prompt_engine or LekaiPromptEngine(
            checkpoint_path=prompt_checkpoint_path
        )
        self._continuation_engine = continuation_engine or LekaiContinuationEngine(
            checkpoint_path=resolved_continuation_checkpoint
        )
        self._catchup_state = CatchUpState()
        self._session_gate = threading.RLock()
        self._scheduler = LekaiPromptContinuationScheduler(
            prompt_engine=self._prompt_engine,
            continuation_engine=self._continuation_engine,
        )
        self._validate_required_real_models()

    def _validate_required_real_models(self) -> None:
        if not parse_env_bool(
            os.environ.get("LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS"),
            default=False,
        ):
            return

        prompt_info = self._prompt_engine.runtime_info()
        continuation_info = self._continuation_engine.runtime_info()
        failures = []
        if not bool(prompt_info.get("has_real_model")):
            failures.append(
                "prompt model not loaded"
                + (
                    f" ({prompt_info.get('fallback_reason')})"
                    if prompt_info.get("fallback_reason") is not None
                    else ""
                )
            )
        if not bool(continuation_info.get("has_real_model")):
            failures.append(
                "continuation model not loaded"
                + (
                    f" ({continuation_info.get('fallback_reason')})"
                    if continuation_info.get("fallback_reason") is not None
                    else ""
                )
            )
        if failures:
            raise RuntimeError(
                "LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS=1 but "
                + "; ".join(failures)
            )

    def configure(self, config: BackendRuntimeConfig) -> None:
        self._continuation_engine.configure(config)

    def runtime_info(self) -> dict[str, str | float | bool | int | None]:
        continuation_info = dict(self._continuation_engine.runtime_info())
        info = dict(continuation_info)
        info["continuation_effective_bpm"] = info.get("last_generation_bpm")
        info["runtime_model_name"] = self.MODEL_NAME
        prompt_info = self._prompt_engine.runtime_info()
        info["prompt_mode"] = str(prompt_info["mode"])
        info["prompt_has_real_model"] = bool(prompt_info["has_real_model"])
        info["prompt_checkpoint_path"] = (
            str(prompt_info["checkpoint_path"]) if prompt_info["checkpoint_path"] is not None else None
        )
        info["prompt_fallback_reason"] = (
            str(prompt_info["fallback_reason"]) if prompt_info.get("fallback_reason") is not None else None
        )
        info["prompt_load_time_ms"] = (
            float(prompt_info["load_time_ms"]) if prompt_info.get("load_time_ms") is not None else None
        )
        info["prompt_warmup_time_ms"] = (
            float(prompt_info["warmup_time_ms"]) if prompt_info.get("warmup_time_ms") is not None else None
        )
        info["prompt_warmup_error"] = (
            str(prompt_info["warmup_error"]) if prompt_info.get("warmup_error") is not None else None
        )
        info["prompt_is_warmed_up"] = bool(prompt_info.get("is_warmed_up", False))
        info["prompt_selection_mode"] = str(
            prompt_info.get("selection_mode", "single")
        )
        info["prompt_batch_candidate_count"] = int(
            prompt_info.get("batch_candidate_count", 1)
        )
        info["prompt_batch_candidates"] = int(
            prompt_info.get("batch_candidate_count", 1)
        )
        info["prompt_temperature"] = prompt_info.get("temperature")
        info["prompt_top_p"] = prompt_info.get("top_p")
        info["prompt_top_k"] = prompt_info.get("top_k")
        info["prompt_repetition_penalty"] = prompt_info.get(
            "repetition_penalty"
        )
        info["continuation_temperature"] = continuation_info.get("temperature")
        info["continuation_top_p"] = continuation_info.get("top_p")
        info["continuation_top_k"] = continuation_info.get("top_k")
        info["continuation_repetition_penalty"] = continuation_info.get(
            "repetition_penalty"
        )
        info["prompt_effective_bpm"] = (
            self._prompt_engine.last_effective_bpm()
            if hasattr(self._prompt_engine, "last_effective_bpm")
            else None
        )
        info.update(self._prefixed_catchup_snapshot())
        scheduler_status = self.scheduler_status()
        info["session_effective_bpm"] = scheduler_status.get("effective_bpm")
        info["scheduler_phase"] = str(scheduler_status["phase"])
        info["scheduler_is_running"] = bool(scheduler_status["is_running"])
        info["scheduler_is_failed"] = bool(scheduler_status["is_failed"])
        info["scheduler_error"] = (
            str(scheduler_status["error"]) if scheduler_status["error"] is not None else None
        )
        return info

    def prompt_generation_log(self) -> dict[str, Any]:
        """Return diagnostics for the latest Prompt Model generation."""

        return copy.deepcopy(self._prompt_engine.last_generation_log())

    def replay_audit(self) -> dict[str, Any]:
        """Return read-only evidence for replaying the current history run."""

        with self._session_gate:
            scheduler_status = self.scheduler_status()
            scheduler_is_running = bool(scheduler_status["is_running"])
            scheduler_is_failed = bool(scheduler_status["is_failed"])
            if scheduler_is_failed:
                trace_capture_reason = "scheduler_failed"
            elif scheduler_is_running:
                trace_capture_reason = "scheduler_running"
            else:
                trace_capture_reason = "complete"
            trace_capture_complete = trace_capture_reason == "complete"

            prompt_info = dict(self._prompt_engine.runtime_info())
            continuation_info = dict(self._continuation_engine.runtime_info())
            runtime_info = dict(self.runtime_info())

            prompt_sample_seed = prompt_info.get("sample_seed")
            continuation_sample_seed = continuation_info.get("sample_seed")
            session_id = continuation_info.get("session_id")
            session_epoch = int(continuation_info.get("session_epoch") or 0)
            seeded_session_active = bool(session_id) and session_epoch > 0

            if not session_id:
                seed_provenance_reason = "continuation_session_id_missing"
            elif session_epoch <= 0:
                seed_provenance_reason = "continuation_session_epoch_not_reset"
            elif prompt_sample_seed is None:
                seed_provenance_reason = "prompt_sample_seed_missing"
            elif continuation_sample_seed is None:
                seed_provenance_reason = "continuation_sample_seed_missing"
            else:
                seed_provenance_reason = "complete"

            seed_provenance_complete = seed_provenance_reason == "complete"
            runtime_info.update(
                {
                    "prompt_sample_seed": prompt_sample_seed,
                    "continuation_sample_seed": continuation_sample_seed,
                    "prompt_checkpoint_path": prompt_info.get("checkpoint_path"),
                    "continuation_checkpoint_path": continuation_info.get(
                        "checkpoint_path"
                    ),
                    "continuation_checkpoint_sha256": continuation_info.get(
                        "checkpoint_sha256"
                    ),
                    "continuation_source_sha256": continuation_info.get(
                        "source_sha256"
                    ),
                    "continuation_code_identity": continuation_info.get(
                        "code_identity"
                    ),
                    "session_id": session_id,
                    "session_epoch": session_epoch,
                    "seeded_session_active": seeded_session_active,
                    "seed_provenance_complete": seed_provenance_complete,
                    "seed_provenance_reason": seed_provenance_reason,
                    "scheduler_phase": str(scheduler_status["phase"]),
                    "scheduler_is_running": scheduler_is_running,
                    "scheduler_is_failed": scheduler_is_failed,
                    "scheduler_error": scheduler_status.get("error"),
                    "trace_capture_complete": trace_capture_complete,
                    "trace_capture_reason": trace_capture_reason,
                }
            )
            return {
                "schema_version": 1,
                "trace_capture_complete": trace_capture_complete,
                "trace_capture_reason": trace_capture_reason,
                "runtime_info": runtime_info,
                "prompt_generation_log": self.prompt_generation_log(),
                "continuation_generations": (
                    self._continuation_engine.generation_metadata_snapshot()
                ),
            }

    @staticmethod
    def _ticks_to_beats(ticks: int) -> int:
        tick_count = max(0, int(ticks))
        return (tick_count + TIMESTEPS_PER_BEAT - 1) // TIMESTEPS_PER_BEAT

    @staticmethod
    def _max_event_tick(events: list[EventPayload]) -> int:
        if not events:
            return 0
        return max(int(event.get("tick", 0)) for event in events)

    def _prefixed_catchup_snapshot(self) -> dict[str, int | bool]:
        return {
            f"catchup_{key}": value
            for key, value in self._catchup_state.snapshot().items()
        }

    def catchup_status(self) -> dict[str, int | bool]:
        return self._catchup_state.snapshot()

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
        effective_bpm = resolve_prompt_bpm(bpm)
        with self._session_gate:
            if bool(self.scheduler_status()["is_running"]):
                raise RuntimeError("prompt-continuation scheduler is already running")
            self._prompt_engine.set_session_generation_config(
                prompt_selection_mode=prompt_selection_mode,
                prompt_batch_candidates=prompt_batch_candidates,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            self._continuation_engine.set_session_generation_config(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            return self._scheduler.start(
                melody_events=melody_events,
                prompt_length_ticks=prompt_length_ticks,
                generation_interval_ticks=generation_interval_ticks,
                inference_mode=inference_mode,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                bpm=effective_bpm,
                observed_until_tick=observed_until_tick,
            )

    def append_melody_events(
        self,
        melody_events: list[EventPayload],
        *,
        observed_until_tick: Optional[int] = None,
    ) -> dict[str, int | bool | str | None]:
        with self._session_gate:
            return self._scheduler.append_melody(
                melody_events=melody_events,
                observed_until_tick=observed_until_tick,
            )

    def scheduler_status(self) -> dict[str, int | bool | str | None]:
        return self._scheduler.status()

    def wait_for_scheduler(self, timeout: Optional[float] = None) -> dict[str, int | bool | str | None]:
        return self._scheduler.wait(timeout=timeout)

    def playable_accompaniment(self) -> list[EventPayload]:
        return self._scheduler.playable_accompaniment()

    def raw_accompaniment_history(self) -> list[EventPayload]:
        return self._scheduler.raw_accompaniment_history()

    def prompt_accompaniment_history(self) -> list[EventPayload]:
        return self._scheduler.prompt_accompaniment_history()

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
        request = PromptContinuationRequest(
            melody_events=copy_events(melody_events),
            generation_start_tick=int(generation_start_tick),
            generation_length_frames=int(generation_length_frames),
            generation_interval_ticks=int(generation_interval_ticks),
            prompt_length_ticks=(int(prompt_length_ticks) if prompt_length_ticks is not None else None),
            inference_mode=str(inference_mode),
            model_name=str(model_name),
            checkpoint_path=checkpoint_path,
        )
        with self._session_gate:
            return self._generate_from_request(request)

    def _generate_from_request(
        self,
        request: PromptContinuationRequest,
    ) -> tuple[list[EventPayload], TimingPayload]:
        prompt_accompaniment: list[EventPayload] = []
        observed_melody_end_tick = max(
            int(request.generation_start_tick),
            self._max_event_tick(request.melody_events),
        )
        self._catchup_state.melody_history_beats = max(
            self._catchup_state.melody_history_beats,
            self._ticks_to_beats(observed_melody_end_tick),
        )

        if request.prompt_length_ticks is not None and request.prompt_length_ticks > 0:
            prompt_accompaniment = self._prompt_engine.generate_prompt_accompaniment(
                melody_events=request.melody_events,
                prompt_start_tick=0,
                prompt_length_ticks=request.prompt_length_ticks,
            )
            if prompt_accompaniment:
                prompt_beats = self._ticks_to_beats(request.prompt_length_ticks)
                self._catchup_state.accompaniment_history_beats = max(
                    self._catchup_state.accompaniment_history_beats,
                    prompt_beats,
                )
                self._continuation_engine.inject_history(
                    melody_events=request.melody_events,
                    accompaniment_events=prompt_accompaniment,
                    injection_length_ticks=request.prompt_length_ticks,
                )

        accompaniment, timings = self._continuation_engine.generate(
            melody_events=request.melody_events,
            generation_start_tick=request.generation_start_tick,
            generation_length_frames=request.generation_length_frames,
            generation_interval_ticks=request.generation_interval_ticks,
            prompt_length_ticks=request.prompt_length_ticks,
            inference_mode=request.inference_mode,
            model_name=request.model_name,
            checkpoint_path=request.checkpoint_path,
        )
        if request.generation_length_frames > 0:
            continuation_end_tick = int(request.generation_start_tick) + int(request.generation_length_frames)
            self._catchup_state.accompaniment_history_beats = max(
                self._catchup_state.accompaniment_history_beats,
                self._ticks_to_beats(continuation_end_tick),
            )
        return accompaniment, timings

    def inject_history(
        self,
        melody_events: list[EventPayload],
        accompaniment_events: list[EventPayload],
        injection_length_ticks: int,
    ) -> dict[str, int | bool | str]:
        with self._session_gate:
            injection_beats = self._ticks_to_beats(injection_length_ticks)
            self._catchup_state.set_history_lengths(
                melody_beats=injection_beats,
                accompaniment_beats=injection_beats,
            )
            result = self._continuation_engine.inject_history(
                melody_events=copy_events(melody_events),
                accompaniment_events=copy_events(accompaniment_events),
                injection_length_ticks=injection_length_ticks,
            )
            result.update(self._prefixed_catchup_snapshot())
            return result

    def clear_history(self) -> dict[str, Any]:
        with self._session_gate:
            self._catchup_state.reset()
            self._scheduler.clear()
            result = self._continuation_engine.clear_history()
            result.update(self._prefixed_catchup_snapshot())
            return result

    def reset_session(self, *, prompt_seed: int, continuation_seed: int) -> dict[str, Any]:
        """Atomically retire P+C work and create independently seeded state."""

        with self._session_gate:
            scheduler_status = self._scheduler.drain_and_clear()
            self._catchup_state.reset()
            actual_prompt_seed = self._prompt_engine.reset_session(int(prompt_seed))
            continuation = self._continuation_engine.reset_session(int(continuation_seed))
            return {
                "success": True,
                "prompt_seed": int(actual_prompt_seed),
                "continuation_effective_seed": int(continuation["effective_seed"]),
                "session_id": str(continuation["session_id"]),
                "session_epoch": int(continuation["session_epoch"]),
                "pending_boundary_generations": int(
                    continuation.get("pending_boundary_generations", 0)
                ),
                "scheduler_phase": str(scheduler_status["phase"]),
                "scheduler_is_running": bool(scheduler_status["is_running"]),
            }

    def injection_status(self) -> dict[str, bool | int | str]:
        status = dict(self._continuation_engine.injection_status())
        status["runtime_model_name"] = self.MODEL_NAME
        status.update(self._prefixed_catchup_snapshot())
        scheduler_status = self.scheduler_status()
        status["scheduler_phase"] = str(scheduler_status["phase"])
        status["scheduler_is_running"] = bool(scheduler_status["is_running"])
        status["scheduler_is_failed"] = bool(scheduler_status["is_failed"])
        return status
