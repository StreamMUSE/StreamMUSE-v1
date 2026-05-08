from __future__ import annotations

import os
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from streammuse.infrastructure.inference.lekai_http_backend import EventPayload, LekaiHttpBackend


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


def _validate_lekai_constraints(model_name: str, generation_length_frames: int) -> None:
    if model_name != "lekai":
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
backend = LekaiHttpBackend(checkpoint_path=_ENV_CHECKPOINT_PATH)


@app.post("/generate_accompaniment", response_model=AccompanimentResponse)
async def generate_accompaniment(request: InferenceRequest) -> AccompanimentResponse:
    _validate_lekai_constraints(
        model_name=request.model_name,
        generation_length_frames=request.generation_length_frames,
    )

    melody_payload: List[EventPayload] = [
        {"type": note.type, "pitch": int(note.pitch), "tick": int(note.tick)}
        for note in request.melody_notes
    ]

    accompaniment, timings = backend.generate(
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

    response_events = [
        AccompanimentNoteEvent(
            type=str(event["type"]),
            pitch=int(event["pitch"]),
            tick=int(event["tick"]),
            velocity=(int(event["velocity"]) if "velocity" in event else None),
        )
        for event in accompaniment
    ]

    return AccompanimentResponse(
        accompaniment=response_events,
        timings=Timings(**timings),
        generation_start_tick=int(request.generation_start_tick),
    )


@app.post("/inject_notes", response_model=DirectInjectionResponse)
async def inject_notes(request: DirectInjectionRequest) -> DirectInjectionResponse:
    melody_payload: List[EventPayload] = [
        {"type": note.type, "pitch": int(note.pitch), "tick": int(note.tick)}
        for note in request.melody_notes
    ]
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
