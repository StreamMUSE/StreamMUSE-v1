"""
Fake HTTP inference server for testing the CLI without a real model.

Echoes melody_notes back as accompaniment so the full client pipeline can be
exercised (input → service → HTTP → response → scheduling → output).

REMOVE THIS SCRIPT once the real inference server is implemented and used.
Run from repo root: uv run python scripts/fake_inference_server.py
Serves at http://localhost:8000 by default (matches CLI default --server-url).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


# Request/response shapes must match what HttpInferenceClient sends/expects.

class MelodyNoteEvent(BaseModel):
    type: str  # "note_on" | "note_off"
    pitch: int
    tick: int
    velocity: int | None = None
    channel: int | None = None
    program: int | None = None
    is_placeholder: bool | None = None


class GenerateRequest(BaseModel):
    melody_notes: List[MelodyNoteEvent]
    generation_start_tick: int
    client_request_send_time: float | None = None
    generation_length_frames: int | None = None
    prompt_length_ticks: int | None = None


class Timings(BaseModel):
    request_arrival_time: float
    response_output_time: float
    preprocess_start_time: float
    inference_start_time: float
    inference_end_time: float
    postprocess_start_time: float


def echo_melody_as_accompaniment(melody_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return melody events as accompaniment (same events, so client can schedule them)."""
    out = []
    for n in melody_notes:
        # Drop None values so client's event_from_dict uses defaults (velocity=100, etc.)
        out.append({k: v for k, v in dict(n).items() if v is not None})
    return out


app = FastAPI(title="StreamMUSE Fake Inference Server")


@app.post("/generate_accompaniment")
async def generate_accompaniment(request: GenerateRequest) -> Dict[str, Any]:
    request_arrival_time = time.perf_counter()

    melody_dicts = [n.model_dump() for n in request.melody_notes]
    preprocess_start_time = time.perf_counter()

    inference_start_time = time.perf_counter()
    time.sleep(0.01)  # Simulate minimal work
    accompaniment = echo_melody_as_accompaniment(melody_dicts)
    inference_end_time = time.perf_counter()

    postprocess_start_time = time.perf_counter()
    response_output_time = time.perf_counter()

    return {
        "accompaniment": accompaniment,
        "timings": {
            "request_arrival_time": request_arrival_time,
            "response_output_time": response_output_time,
            "preprocess_start_time": preprocess_start_time,
            "inference_start_time": inference_start_time,
            "inference_end_time": inference_end_time,
            "postprocess_start_time": postprocess_start_time,
        },
        "generation_start_tick": request.generation_start_tick,
    }


@app.post("/inject_notes")
async def inject_notes(request: dict) -> Dict[str, Any]:
    """Stub for client inject_history."""
    return {
        "success": True,
        "message": "Fake server: inject ignored",
        "melody_notes_injected": len(request.get("melody_notes", [])),
        "accompaniment_notes_injected": len(request.get("accompaniment_notes", [])),
        "injection_length_ticks": request.get("injection_length_ticks", 0),
    }


@app.get("/injection_status")
async def injection_status() -> Dict[str, Any]:
    """Stub for client get_injection_status."""
    return {"is_injected": False, "injection_length_ticks": 0, "injection_file_path": None}


@app.post("/clear_history")
async def clear_history() -> Dict[str, Any]:
    """Stub for client clear_history."""
    return {"message": "Fake server: history cleared"}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "server": "fake"}


def main() -> None:
    host = "0.0.0.0"
    port = 8000
    print("StreamMUSE Fake Inference Server (echo melody as accompaniment)")
    print("REMOVE this script once the real server is in place.")
    print(f"Listening on http://{host}:{port}")
    print("Start the CLI with: uv run streammuse-cli (default URL points here)")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
