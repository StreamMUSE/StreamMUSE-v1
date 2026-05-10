from __future__ import annotations

import os
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from streammuse.infrastructure.inference.lekai_http_backend import EventPayload, LekaiHttpBackend
from streammuse.infrastructure.inference.lekai_prompt_continuation import LekaiPromptContinuationBackend


class MelodyNoteEvent(BaseModel):
    type: str
    pitch: int
    tick: int


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


class PromptContinuationStartRequest(BaseModel):
    melody_notes: List[MelodyNoteEvent]
    prompt_length_ticks: int = Field(gt=0)
    generation_interval_ticks: int = Field(gt=0)
    observed_until_tick: Optional[int] = Field(default=None, ge=0)
    inference_mode: str = "sliding_window"
    model_name: str = "lekai_prompt_continuation"
    checkpoint_path: Optional[str] = None


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
    continuation_calls: int
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


class PromptContinuationRawHistoryResponse(BaseModel):
    accompaniment: List[AccompanimentNoteEvent]
    status: PromptContinuationStatusResponse


LEKAI_MODEL_NAME = "lekai"
LEKAI_PROMPT_CONTINUATION_MODEL_NAME = "lekai_prompt_continuation"
_LEKAI_FIXED_BEAT_MODELS = {LEKAI_MODEL_NAME, LEKAI_PROMPT_CONTINUATION_MODEL_NAME}


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
_ENV_PROMPT_CONTINUATION_CHECKPOINT_PATH = os.environ.get("LEKAI_PROMPT_CONTINUATION_CHECKPOINT_PATH")
_ENV_PROMPT_CHECKPOINT_PATH = os.environ.get("LEKAI_PROMPT_CHECKPOINT_PATH")
_ENV_CONTINUATION_CHECKPOINT_PATH = os.environ.get("LEKAI_CONTINUATION_CHECKPOINT_PATH")
backend = LekaiHttpBackend(checkpoint_path=_ENV_CHECKPOINT_PATH)
prompt_continuation_backend = LekaiPromptContinuationBackend(
    checkpoint_path=_ENV_PROMPT_CONTINUATION_CHECKPOINT_PATH,
    prompt_checkpoint_path=_ENV_PROMPT_CHECKPOINT_PATH,
    continuation_checkpoint_path=_ENV_CONTINUATION_CHECKPOINT_PATH,
)


def _select_backend(model_name: str):
    if model_name == LEKAI_PROMPT_CONTINUATION_MODEL_NAME:
        return prompt_continuation_backend
    return backend


def _melody_payload(notes: List[MelodyNoteEvent]) -> List[EventPayload]:
    return [
        {"type": note.type, "pitch": int(note.pitch), "tick": int(note.tick)}
        for note in notes
    ]


def _accompaniment_response_events(events: list[EventPayload]) -> list[AccompanimentNoteEvent]:
    return [
        AccompanimentNoteEvent(
            type=str(event["type"]),
            pitch=int(event["pitch"]),
            tick=int(event["tick"]),
            velocity=(int(event["velocity"]) if "velocity" in event else None),
        )
        for event in events
    ]


def _scheduler_status_response(status: dict[str, int | bool | str | None]) -> PromptContinuationStatusResponse:
    return PromptContinuationStatusResponse(
        phase=str(status["phase"]),
        is_running=bool(status["is_running"]),
        is_failed=bool(status["is_failed"]),
        error=(str(status["error"]) if status["error"] is not None else None),
        melody_event_count=int(status["melody_event_count"]),
        accompaniment_event_count=int(status["accompaniment_event_count"]),
        prompt_length_ticks=int(status["prompt_length_ticks"]),
        generation_interval_ticks=int(status["generation_interval_ticks"]),
        continuation_calls=int(status["continuation_calls"]),
        melody_history_beats=int(status["melody_history_beats"]),
        accompaniment_history_beats=int(status["accompaniment_history_beats"]),
        playable_lookahead_beats=int(status["playable_lookahead_beats"]),
        target_playable_accompaniment_beats=int(status["target_playable_accompaniment_beats"]),
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

    selected_backend = _select_backend(request.model_name)
    accompaniment, timings = selected_backend.generate(
        melody_events=melody_payload,
        generation_start_tick=int(request.generation_start_tick),
        generation_length_frames=int(request.generation_length_frames),
        generation_interval_ticks=int(request.generation_interval_ticks),
        prompt_length_ticks=(int(request.prompt_length_ticks) if request.prompt_length_ticks is not None else None),
        inference_mode=request.inference_mode,
        model_name=request.model_name,
        checkpoint_path=request.checkpoint_path,
        bpm=request.bpm,
        input_file=request.input_file,
    )

    response_events = _accompaniment_response_events(accompaniment)

    return AccompanimentResponse(
        accompaniment=response_events,
        timings=Timings(**timings),
        generation_start_tick=int(request.generation_start_tick),
    )


@app.post("/prompt_continuation/start", response_model=PromptContinuationStatusResponse)
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
            observed_until_tick=(
                int(request.observed_until_tick) if request.observed_until_tick is not None else None
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _scheduler_status_response(status)


@app.post("/prompt_continuation/append_melody", response_model=PromptContinuationStatusResponse)
async def prompt_continuation_append_melody(
    request: PromptContinuationAppendMelodyRequest,
) -> PromptContinuationStatusResponse:
    status = prompt_continuation_backend.append_melody_events(
        melody_events=_melody_payload(request.melody_notes),
        observed_until_tick=(
            int(request.observed_until_tick) if request.observed_until_tick is not None else None
        ),
    )
    return _scheduler_status_response(status)


@app.get("/prompt_continuation/status", response_model=PromptContinuationStatusResponse)
async def prompt_continuation_status() -> PromptContinuationStatusResponse:
    return _scheduler_status_response(prompt_continuation_backend.scheduler_status())


@app.get("/prompt_continuation/runtime_info")
async def prompt_continuation_runtime_info() -> dict[str, object]:
    return dict(prompt_continuation_backend.runtime_info())


@app.get("/prompt_continuation/playable", response_model=PromptContinuationPlayableResponse)
async def prompt_continuation_playable() -> PromptContinuationPlayableResponse:
    status = prompt_continuation_backend.scheduler_status()
    return PromptContinuationPlayableResponse(
        accompaniment=_accompaniment_response_events(
            prompt_continuation_backend.playable_accompaniment()
        ),
        status=_scheduler_status_response(status),
    )


@app.get("/prompt_continuation/raw_history", response_model=PromptContinuationRawHistoryResponse)
async def prompt_continuation_raw_history() -> PromptContinuationRawHistoryResponse:
    status = prompt_continuation_backend.scheduler_status()
    return PromptContinuationRawHistoryResponse(
        accompaniment=_accompaniment_response_events(
            prompt_continuation_backend.raw_accompaniment_history()
        ),
        status=_scheduler_status_response(status),
    )


@app.get("/prompt_continuation/prompt_history", response_model=PromptContinuationRawHistoryResponse)
async def prompt_continuation_prompt_history() -> PromptContinuationRawHistoryResponse:
    status = prompt_continuation_backend.scheduler_status()
    return PromptContinuationRawHistoryResponse(
        accompaniment=_accompaniment_response_events(
            prompt_continuation_backend.prompt_accompaniment_history()
        ),
        status=_scheduler_status_response(status),
    )


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
