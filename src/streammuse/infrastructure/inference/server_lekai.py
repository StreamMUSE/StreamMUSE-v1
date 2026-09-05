from __future__ import annotations

import os
import secrets
import uuid
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from streammuse.infrastructure.inference.lekai_http_backend import (
    EventPayload,
    LekaiHttpBackend,
    SessionStateError,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation import (
    LekaiPromptContinuationBackend,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.token_conversion import (
    event_representation_summary,
)


class MelodyNoteEvent(BaseModel):
    type: str
    pitch: int
    tick: int
    velocity: int = 100
    channel: int = 0
    program: int = 0
    is_placeholder: bool = False


class AccompanimentNoteEvent(BaseModel):
    type: str
    pitch: int
    tick: int
    velocity: Optional[int] = None


class InferenceRequest(BaseModel):
    melody_notes: List[MelodyNoteEvent]
    generation_start_tick: int
    generation_length_frames: int = Field(gt=0)
    generation_interval_ticks: int = Field(gt=0)
    prompt_length_ticks: Optional[int] = None
    inference_mode: str = "sliding_window"
    model_name: str = "lekai"
    checkpoint_path: Optional[str] = None
    client_request_send_time: Optional[float] = None
    bpm: Optional[int] = None
    input_file: Optional[str] = None  # 输入文件名，用于日志记录
    session_id: Optional[str] = None
    session_epoch: Optional[int] = None
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class Timings(BaseModel):
    request_arrival_time: float
    response_output_time: float
    preprocess_start_time: float
    inference_start_time: float
    inference_end_time: float
    postprocess_start_time: float


class AccompanimentResponse(BaseModel):
    accompaniment: List[AccompanimentNoteEvent]
    timings: Timings
    generation_start_tick: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResetSessionRequest(BaseModel):
    seed: int


class ResetSessionResponse(BaseModel):
    success: bool
    session_id: str
    session_epoch: int
    effective_seed: int
    pending_boundary_generations: int


class PromptContinuationResetSessionRequest(BaseModel):
    prompt_seed: int
    continuation_seed: int


class PromptContinuationResetSessionResponse(BaseModel):
    success: bool
    prompt_seed: int
    continuation_effective_seed: int
    session_id: str
    session_epoch: int
    pending_boundary_generations: int
    scheduler_phase: str
    scheduler_is_running: bool


class PromptContinuationSessionInitializeRequest(BaseModel):
    prompt_seed: Optional[int] = Field(default=None, ge=0, lt=2**63)
    continuation_seed: Optional[int] = Field(default=None, ge=0, lt=2**63)


class PromptContinuationSessionInitializeResponse(BaseModel):
    success: bool
    prompt_requested_seed: int
    prompt_effective_seed: int
    continuation_requested_seed: int
    continuation_effective_seed: int
    prompt_seed_source: str
    continuation_seed_source: str
    session_id: str
    session_epoch: int
    pending_boundary_generations: int
    scheduler_phase: str
    scheduler_is_running: bool


class DirectInjectionRequest(BaseModel):
    melody_notes: List[MelodyNoteEvent]
    accompaniment_notes: List[AccompanimentNoteEvent]
    injection_length_ticks: int = Field(ge=0)


class DirectInjectionResponse(BaseModel):
    success: bool
    message: str
    melody_notes_injected: int
    accompaniment_notes_injected: int
    injection_length_ticks: int


class ClearHistoryResponse(BaseModel):
    success: bool
    message: str
    melody_history: List[dict[str, Any]] = Field(default_factory=list)
    accompaniment_history: List[dict[str, Any]] = Field(default_factory=list)


class InjectionStatusResponse(BaseModel):
    is_injected: bool
    injection_length_ticks: int
    runtime_model_name: str
    runtime_inference_mode: str
    generation_interval_ticks: Optional[int] = None
    generation_length_frames: Optional[int] = None
    prompt_length_ticks: Optional[int] = None
    ticks_per_beat: int


class RuntimeInfoResponse(BaseModel):
    mode: str
    has_real_model: bool
    resolved_device: str
    resolved_dtype: str
    checkpoint_path: Optional[str] = None
    checkpoint_format: Optional[str] = None
    fallback_reason: Optional[str] = None
    load_time_ms: Optional[float] = None
    warmup_time_ms: Optional[float] = None
    use_cache: bool
    runtime_model_name: str
    runtime_inference_mode: str
    generation_interval_ticks: Optional[int] = None
    generation_length_frames: Optional[int] = None
    prompt_length_ticks: Optional[int] = None
    ticks_per_beat: int
    checkpoint_sha256: Optional[str] = None
    source_sha256: str
    code_identity: str
    effective_bpm: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    prompt_context_beats: int
    history_retention_ticks: int
    max_generation_length_frames: Optional[int] = None
    max_prompt_ticks: Optional[int] = None
    time_signature_index: int
    sample_seed: int
    session_id: Optional[str] = None
    session_epoch: int
    accepting_requests: bool
    pending_boundary_generations: int
    boundary_generation_order: str


class PromptContinuationStartRequest(BaseModel):
    melody_notes: List[MelodyNoteEvent]
    prompt_length_ticks: int = Field(gt=0)
    generation_interval_ticks: int = Field(gt=0)
    observed_until_tick: Optional[int] = Field(default=None, ge=0)
    inference_mode: str = "sliding_window"
    model_name: str = "lekai_prompt_continuation"
    checkpoint_path: Optional[str] = None
    bpm: Optional[int] = Field(default=None, gt=0)


class PromptContinuationAppendMelodyRequest(BaseModel):
    melody_notes: List[MelodyNoteEvent]
    observed_until_tick: Optional[int] = Field(default=None, ge=0)


class PromptContinuationStatusResponse(BaseModel):
    phase: str
    is_running: bool
    is_failed: bool
    error: Optional[str] = None
    melody_event_count: int
    accompaniment_event_count: int
    prompt_length_ticks: int
    generation_interval_ticks: int
    effective_bpm: Optional[int] = None
    continuation_calls: int
    last_continuation_event_count: int = 0
    last_continuation_note_on_count: int = 0
    last_continuation_min_tick: Optional[int] = None
    last_continuation_max_tick: Optional[int] = None
    empty_continuation_output_streak: int = 0
    melody_history_beats: int
    accompaniment_history_beats: int
    playable_lookahead_beats: int
    target_playable_accompaniment_beats: int
    beats_needed_for_playback: int
    is_history_aligned: bool
    is_playback_ready: bool


class PromptContinuationPlayableResponse(BaseModel):
    accompaniment: List[AccompanimentNoteEvent]
    status: PromptContinuationStatusResponse
    representation: dict[str, Any] = Field(default_factory=dict)


class PromptContinuationRawHistoryResponse(BaseModel):
    accompaniment: List[AccompanimentNoteEvent]
    status: PromptContinuationStatusResponse
    representation: dict[str, Any] = Field(default_factory=dict)


class PromptContinuationReplayAuditResponse(BaseModel):
    schema_version: int
    trace_capture_complete: bool
    trace_capture_reason: str
    runtime_info: dict[str, Any]
    prompt_generation_log: dict[str, Any]
    continuation_generations: List[dict[str, Any]]


LEKAI_MODEL_NAME = "lekai"
LEKAI_PROMPT_CONTINUATION_MODEL_NAME = "lekai_prompt_continuation"
_LEKAI_FIXED_BEAT_MODELS = {
    LEKAI_MODEL_NAME,
    LEKAI_PROMPT_CONTINUATION_MODEL_NAME,
}


def _debug_reset_enabled() -> bool:
    return os.environ.get("LEKAI_ENABLE_DEBUG_RESET", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _new_session_seed() -> int:
    return secrets.randbits(63)


def _validate_lekai_constraints(model_name: str, generation_length_frames: int) -> None:
    if model_name not in _LEKAI_FIXED_BEAT_MODELS:
        return
    if generation_length_frames % 4 != 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "generation_length_frames must be a multiple of 4 "
                f"(model uses fixed 4-timesteps-per-beat tokenization, got {generation_length_frames})"
            ),
        )


app = FastAPI(title="StreamMUSE Lekai Inference Server")
_ENV_CHECKPOINT_PATH = os.environ.get("LEKAI_CHECKPOINT_PATH")
_ENV_PROMPT_CONTINUATION_CHECKPOINT_PATH = os.environ.get(
    "LEKAI_PROMPT_CONTINUATION_CHECKPOINT_PATH"
)
_ENV_PROMPT_CHECKPOINT_PATH = os.environ.get("LEKAI_PROMPT_CHECKPOINT_PATH")
_ENV_CONTINUATION_CHECKPOINT_PATH = os.environ.get(
    "LEKAI_CONTINUATION_CHECKPOINT_PATH"
)
backend = LekaiHttpBackend(checkpoint_path=_ENV_CHECKPOINT_PATH)
prompt_continuation_backend = LekaiPromptContinuationBackend(
    checkpoint_path=_ENV_PROMPT_CONTINUATION_CHECKPOINT_PATH,
    prompt_checkpoint_path=_ENV_PROMPT_CHECKPOINT_PATH,
    continuation_checkpoint_path=_ENV_CONTINUATION_CHECKPOINT_PATH,
)


def _melody_payload(notes: List[MelodyNoteEvent]) -> List[EventPayload]:
    return [
        {
            "type": note.type,
            "pitch": int(note.pitch),
            "tick": int(note.tick),
            "velocity": int(note.velocity),
            "channel": int(note.channel),
            "program": int(note.program),
            **({"is_placeholder": True} if note.is_placeholder else {}),
        }
        for note in notes
    ]


def _accompaniment_response_events(
    events: list[EventPayload],
) -> list[AccompanimentNoteEvent]:
    return [
        AccompanimentNoteEvent(
            type=str(event["type"]),
            pitch=int(event["pitch"]),
            tick=int(event["tick"]),
            velocity=(
                int(event["velocity"])
                if "velocity" in event and event["velocity"] is not None
                else (0 if str(event.get("type", "")) == "note_off" else 100)
            ),
        )
        for event in events
    ]


def _event_representation(events: list[EventPayload]) -> dict[str, Any]:
    return dict(event_representation_summary(events))


def _scheduler_status_response(
    status: dict[str, int | bool | str | None],
) -> PromptContinuationStatusResponse:
    return PromptContinuationStatusResponse(
        phase=str(status["phase"]),
        is_running=bool(status["is_running"]),
        is_failed=bool(status["is_failed"]),
        error=(str(status["error"]) if status.get("error") is not None else None),
        melody_event_count=int(status["melody_event_count"]),
        accompaniment_event_count=int(status["accompaniment_event_count"]),
        prompt_length_ticks=int(status["prompt_length_ticks"]),
        generation_interval_ticks=int(status["generation_interval_ticks"]),
        effective_bpm=(
            int(status["effective_bpm"])
            if status.get("effective_bpm") is not None
            else None
        ),
        continuation_calls=int(status["continuation_calls"]),
        last_continuation_event_count=int(
            status.get("last_continuation_event_count", 0) or 0
        ),
        last_continuation_note_on_count=int(
            status.get("last_continuation_note_on_count", 0) or 0
        ),
        last_continuation_min_tick=(
            int(status["last_continuation_min_tick"])
            if status.get("last_continuation_min_tick") is not None
            else None
        ),
        last_continuation_max_tick=(
            int(status["last_continuation_max_tick"])
            if status.get("last_continuation_max_tick") is not None
            else None
        ),
        empty_continuation_output_streak=int(
            status.get("empty_continuation_output_streak", 0) or 0
        ),
        melody_history_beats=int(status["melody_history_beats"]),
        accompaniment_history_beats=int(status["accompaniment_history_beats"]),
        playable_lookahead_beats=int(status["playable_lookahead_beats"]),
        target_playable_accompaniment_beats=int(
            status["target_playable_accompaniment_beats"]
        ),
        beats_needed_for_playback=int(status["beats_needed_for_playback"]),
        is_history_aligned=bool(status["is_history_aligned"]),
        is_playback_ready=bool(status["is_playback_ready"]),
    )


@app.post("/generate_accompaniment", response_model=AccompanimentResponse)
async def generate_accompaniment(request: InferenceRequest) -> AccompanimentResponse:
    _validate_lekai_constraints(
        model_name=request.model_name,
        generation_length_frames=request.generation_length_frames,
    )

    melody_payload = _melody_payload(request.melody_notes)

    try:
        if request.model_name == LEKAI_PROMPT_CONTINUATION_MODEL_NAME:
            accompaniment, timings = prompt_continuation_backend.generate(
                melody_events=melody_payload,
                generation_start_tick=int(request.generation_start_tick),
                generation_length_frames=int(request.generation_length_frames),
                generation_interval_ticks=int(request.generation_interval_ticks),
                prompt_length_ticks=(
                    int(request.prompt_length_ticks)
                    if request.prompt_length_ticks is not None
                    else None
                ),
                inference_mode=request.inference_mode,
                model_name=request.model_name,
                checkpoint_path=request.checkpoint_path,
            )
            metadata: dict[str, Any] = {}
        else:
            accompaniment, timings = backend.generate(
                melody_events=melody_payload,
                generation_start_tick=int(request.generation_start_tick),
                generation_length_frames=int(request.generation_length_frames),
                generation_interval_ticks=int(request.generation_interval_ticks),
                prompt_length_ticks=(
                    int(request.prompt_length_ticks)
                    if request.prompt_length_ticks is not None
                    else None
                ),
                inference_mode=request.inference_mode,
                model_name=request.model_name,
                checkpoint_path=request.checkpoint_path,
                bpm=request.bpm,
                input_file=request.input_file,
                session_id=request.session_id,
                session_epoch=request.session_epoch,
                request_id=request.request_id,
            )
            metadata = backend.consume_generation_metadata(request.request_id)
    except SessionStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    response_events = _accompaniment_response_events(accompaniment)

    return AccompanimentResponse(
        accompaniment=response_events,
        timings=Timings(**timings),
        generation_start_tick=int(request.generation_start_tick),
        metadata=metadata,
    )


@app.post(
    "/prompt_continuation/start",
    response_model=PromptContinuationStatusResponse,
)
async def prompt_continuation_start(
    request: PromptContinuationStartRequest,
) -> PromptContinuationStatusResponse:
    if request.model_name != LEKAI_PROMPT_CONTINUATION_MODEL_NAME:
        raise HTTPException(
            status_code=422,
            detail=f"model_name must be {LEKAI_PROMPT_CONTINUATION_MODEL_NAME}",
        )
    try:
        status = prompt_continuation_backend.start_prompt_catchup(
            melody_events=_melody_payload(request.melody_notes),
            prompt_length_ticks=int(request.prompt_length_ticks),
            generation_interval_ticks=int(request.generation_interval_ticks),
            inference_mode=request.inference_mode,
            model_name=request.model_name,
            checkpoint_path=request.checkpoint_path,
            bpm=request.bpm,
            observed_until_tick=(
                int(request.observed_until_tick)
                if request.observed_until_tick is not None
                else None
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _scheduler_status_response(status)


@app.post(
    "/prompt_continuation/append_melody",
    response_model=PromptContinuationStatusResponse,
)
async def prompt_continuation_append_melody(
    request: PromptContinuationAppendMelodyRequest,
) -> PromptContinuationStatusResponse:
    status = prompt_continuation_backend.append_melody_events(
        melody_events=_melody_payload(request.melody_notes),
        observed_until_tick=(
            int(request.observed_until_tick)
            if request.observed_until_tick is not None
            else None
        ),
    )
    return _scheduler_status_response(status)


@app.get(
    "/prompt_continuation/status",
    response_model=PromptContinuationStatusResponse,
)
async def prompt_continuation_status() -> PromptContinuationStatusResponse:
    return _scheduler_status_response(
        prompt_continuation_backend.scheduler_status()
    )


@app.get("/prompt_continuation/runtime_info")
async def prompt_continuation_runtime_info() -> dict[str, object]:
    return dict(prompt_continuation_backend.runtime_info())


@app.get("/prompt_continuation/prompt_generation_log")
async def prompt_continuation_prompt_generation_log() -> dict[str, Any]:
    return prompt_continuation_backend.prompt_generation_log()


@app.get(
    "/prompt_continuation/replay_audit",
    response_model=PromptContinuationReplayAuditResponse,
)
async def prompt_continuation_replay_audit() -> PromptContinuationReplayAuditResponse:
    return PromptContinuationReplayAuditResponse(
        **prompt_continuation_backend.replay_audit()
    )


@app.post(
    "/prompt_continuation/session/initialize",
    response_model=PromptContinuationSessionInitializeResponse,
)
async def prompt_continuation_initialize_session(
    request: PromptContinuationSessionInitializeRequest,
) -> PromptContinuationSessionInitializeResponse:
    if (request.prompt_seed is None) != (request.continuation_seed is None):
        raise HTTPException(
            status_code=422,
            detail=(
                "prompt_seed and continuation_seed must either both be supplied "
                "or both be omitted"
            ),
        )
    prompt_seed_source = "requested" if request.prompt_seed is not None else "system"
    continuation_seed_source = prompt_seed_source
    if request.prompt_seed is None:
        requested_prompt_seed = _new_session_seed()
        requested_continuation_seed = requested_prompt_seed
    else:
        assert request.continuation_seed is not None
        requested_prompt_seed = int(request.prompt_seed)
        requested_continuation_seed = int(request.continuation_seed)
    result = prompt_continuation_backend.reset_session(
        prompt_seed=requested_prompt_seed,
        continuation_seed=requested_continuation_seed,
    )
    return PromptContinuationSessionInitializeResponse(
        success=bool(result["success"]),
        prompt_requested_seed=requested_prompt_seed,
        prompt_effective_seed=int(result["prompt_seed"]),
        continuation_requested_seed=requested_continuation_seed,
        continuation_effective_seed=int(result["continuation_effective_seed"]),
        prompt_seed_source=prompt_seed_source,
        continuation_seed_source=continuation_seed_source,
        session_id=str(result["session_id"]),
        session_epoch=int(result["session_epoch"]),
        pending_boundary_generations=int(
            result.get("pending_boundary_generations", 0)
        ),
        scheduler_phase=str(result["scheduler_phase"]),
        scheduler_is_running=bool(result["scheduler_is_running"]),
    )


@app.get(
    "/prompt_continuation/playable",
    response_model=PromptContinuationPlayableResponse,
)
async def prompt_continuation_playable() -> PromptContinuationPlayableResponse:
    status = prompt_continuation_backend.scheduler_status()
    accompaniment = prompt_continuation_backend.playable_accompaniment()
    return PromptContinuationPlayableResponse(
        accompaniment=_accompaniment_response_events(accompaniment),
        status=_scheduler_status_response(status),
        representation=_event_representation(accompaniment),
    )


@app.get(
    "/prompt_continuation/raw_history",
    response_model=PromptContinuationRawHistoryResponse,
)
async def prompt_continuation_raw_history() -> PromptContinuationRawHistoryResponse:
    status = prompt_continuation_backend.scheduler_status()
    accompaniment = prompt_continuation_backend.raw_accompaniment_history()
    return PromptContinuationRawHistoryResponse(
        accompaniment=_accompaniment_response_events(accompaniment),
        status=_scheduler_status_response(status),
        representation=_event_representation(accompaniment),
    )


@app.get(
    "/prompt_continuation/prompt_history",
    response_model=PromptContinuationRawHistoryResponse,
)
async def prompt_continuation_prompt_history() -> PromptContinuationRawHistoryResponse:
    status = prompt_continuation_backend.scheduler_status()
    accompaniment = prompt_continuation_backend.prompt_accompaniment_history()
    return PromptContinuationRawHistoryResponse(
        accompaniment=_accompaniment_response_events(accompaniment),
        status=_scheduler_status_response(status),
        representation=_event_representation(accompaniment),
    )


@app.post("/debug/reset_session", response_model=ResetSessionResponse)
async def reset_session(request: ResetSessionRequest) -> ResetSessionResponse:
    if not _debug_reset_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "debug session reset is disabled; set "
                "LEKAI_ENABLE_DEBUG_RESET=true only on a dedicated local server"
            ),
        )
    result = backend.reset_session(seed=int(request.seed))
    return ResetSessionResponse(**result)


@app.post(
    "/prompt_continuation/debug/reset_session",
    response_model=PromptContinuationResetSessionResponse,
)
async def prompt_continuation_reset_session(
    request: PromptContinuationResetSessionRequest,
) -> PromptContinuationResetSessionResponse:
    if not _debug_reset_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "debug session reset is disabled; set "
                "LEKAI_ENABLE_DEBUG_RESET=true only on a dedicated local server"
            ),
        )
    result = prompt_continuation_backend.reset_session(
        prompt_seed=int(request.prompt_seed),
        continuation_seed=int(request.continuation_seed),
    )
    return PromptContinuationResetSessionResponse(**result)


@app.post("/inject_notes", response_model=DirectInjectionResponse)
async def inject_notes(request: DirectInjectionRequest) -> DirectInjectionResponse:
    melody_payload = _melody_payload(request.melody_notes)
    accompaniment_payload: List[EventPayload] = [
        {
            "type": note.type,
            "pitch": int(note.pitch),
            "tick": int(note.tick),
            **({"velocity": int(note.velocity)} if note.velocity is not None else {}),
        }
        for note in request.accompaniment_notes
    ]

    result = backend.inject_history(
        melody_events=melody_payload,
        accompaniment_events=accompaniment_payload,
        injection_length_ticks=int(request.injection_length_ticks),
    )
    prompt_continuation_backend.inject_history(
        melody_events=melody_payload,
        accompaniment_events=accompaniment_payload,
        injection_length_ticks=int(request.injection_length_ticks),
    )

    return DirectInjectionResponse(
        success=bool(result["success"]),
        message=str(result["message"]),
        melody_notes_injected=int(result["melody_notes_injected"]),
        accompaniment_notes_injected=int(result["accompaniment_notes_injected"]),
        injection_length_ticks=int(result["injection_length_ticks"]),
    )


@app.post("/clear_history", response_model=ClearHistoryResponse)
async def clear_history() -> ClearHistoryResponse:
    result = backend.clear_history()
    prompt_continuation_backend.clear_history()
    return ClearHistoryResponse(
        success=bool(result["success"]),
        message=str(result["message"]),
        melody_history=list(result.get("melody_history", [])),
        accompaniment_history=list(result.get("accompaniment_history", [])),
    )


@app.get("/injection_status", response_model=InjectionStatusResponse)
async def injection_status() -> InjectionStatusResponse:
    result = backend.injection_status()
    return InjectionStatusResponse(
        is_injected=bool(result["is_injected"]),
        injection_length_ticks=int(result["injection_length_ticks"]),
        runtime_model_name=str(result["runtime_model_name"]),
        runtime_inference_mode=str(result["runtime_inference_mode"]),
        generation_interval_ticks=result["generation_interval_ticks"],
        generation_length_frames=result["generation_length_frames"],
        prompt_length_ticks=result["prompt_length_ticks"],
        ticks_per_beat=int(result["ticks_per_beat"]),
    )


@app.get("/runtime_info", response_model=RuntimeInfoResponse)
async def runtime_info() -> RuntimeInfoResponse:
    result = backend.runtime_info()
    return RuntimeInfoResponse(
        mode=str(result["mode"]),
        has_real_model=bool(result["has_real_model"]),
        resolved_device=str(result["resolved_device"]),
        resolved_dtype=str(result["resolved_dtype"]),
        checkpoint_path=(str(result["checkpoint_path"]) if result["checkpoint_path"] is not None else None),
        checkpoint_format=(str(result["checkpoint_format"]) if result["checkpoint_format"] is not None else None),
        fallback_reason=(str(result["fallback_reason"]) if result["fallback_reason"] is not None else None),
        load_time_ms=(float(result["load_time_ms"]) if result["load_time_ms"] is not None else None),
        warmup_time_ms=(float(result["warmup_time_ms"]) if result["warmup_time_ms"] is not None else None),
        use_cache=bool(result["use_cache"]),
        runtime_model_name=str(result["runtime_model_name"]),
        runtime_inference_mode=str(result["runtime_inference_mode"]),
        generation_interval_ticks=result["generation_interval_ticks"],
        generation_length_frames=result["generation_length_frames"],
        prompt_length_ticks=result["prompt_length_ticks"],
        ticks_per_beat=int(result["ticks_per_beat"]),
        checkpoint_sha256=result["checkpoint_sha256"],
        source_sha256=str(result["source_sha256"]),
        code_identity=str(result["code_identity"]),
        effective_bpm=int(result["effective_bpm"]),
        temperature=float(result["temperature"]),
        top_k=int(result["top_k"]),
        top_p=float(result["top_p"]),
        repetition_penalty=float(result["repetition_penalty"]),
        prompt_context_beats=int(result["prompt_context_beats"]),
        history_retention_ticks=int(result["history_retention_ticks"]),
        max_generation_length_frames=result["max_generation_length_frames"],
        max_prompt_ticks=result["max_prompt_ticks"],
        time_signature_index=int(result["time_signature_index"]),
        sample_seed=int(result["sample_seed"]),
        session_id=result["session_id"],
        session_epoch=int(result["session_epoch"]),
        accepting_requests=bool(result["accepting_requests"]),
        pending_boundary_generations=int(result["pending_boundary_generations"]),
        boundary_generation_order=str(result["boundary_generation_order"]),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Bug #7 fix: 添加启动入口
def main() -> None:
    import uvicorn

    host = os.environ.get("LEKAI_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("LEKAI_SERVER_PORT", "8000"))

    if _ENV_CHECKPOINT_PATH:
        print(f"[LekaiServer] LEKAI_CHECKPOINT_PATH={_ENV_CHECKPOINT_PATH}")
    else:
        print("[LekaiServer] LEKAI_CHECKPOINT_PATH not set")

    info = backend.runtime_info()
    print(f"[LekaiServer] Inference mode: {info['mode']}")
    print(
        "[LekaiServer] Runtime: "
        f"device={info['resolved_device']}, "
        f"dtype={info['resolved_dtype']}, "
        f"checkpoint_format={info['checkpoint_format']}, "
        f"use_cache={info['use_cache']}"
    )
    if info["fallback_reason"]:
        print(f"[LekaiServer] Fallback reason: {info['fallback_reason']}")

    print(f"Listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
