"""
Fake Server for StreamMUSE Web UI Demo

This is a mock server that echoes melody notes back as accompaniment.
It mirrors the real server.py API but requires no ML dependencies.

Usage:
    python fake_server.py
    # Runs on http://localhost:8001
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import time


class MelodyNoteEvent(BaseModel):
    pitch: int
    tick: int
    duration: int


class InferenceRequest(BaseModel):
    melody_notes: list[MelodyNoteEvent]
    generation_start_tick: int
    client_request_send_time: float = 0.0
    generation_length_frames: int = None


class AccompanimentNoteEvent(BaseModel):
    pitch: int
    tick: int
    duration: int
    program: int


class Timings(BaseModel):
    request_arrival_time: float
    response_output_time: float
    preprocess_start_time: float
    inference_start_time: float
    inference_end_time: float
    postprocess_start_time: float


class AccompanimentResponse(BaseModel):
    accompaniment: list[AccompanimentNoteEvent]
    timings: Timings
    generation_start_tick: int


class InjectionRequest(BaseModel):
    injection_file_path: str
    injection_length_ticks: int


class InjectionResponse(BaseModel):
    success: bool
    message: str
    injection_length_ticks: int
    melody_notes_injected: int
    accompaniment_notes_injected: int


class DirectInjectionRequest(BaseModel):
    melody_notes: list[MelodyNoteEvent]
    accompaniment_notes: list[AccompanimentNoteEvent]
    injection_length_ticks: int


class DirectInjectionResponse(BaseModel):
    success: bool
    message: str
    melody_notes_injected: int
    accompaniment_notes_injected: int
    injection_length_ticks: int


app = FastAPI(title="StreamMUSE Fake Server (Demo)")

melody_history = []
accompaniment_history = []
injection_state = {
    "is_injected": False,
    "injection_length_ticks": 0,
    "injection_file_path": None,
}


def generate_echo_accompaniment(
    melody_notes: list[dict],
    generation_start_tick: int,
    generation_length_frames: int = None
) -> list[dict]:
    """
    Echo melody notes back as accompaniment, 8 ticks later and 1 octave down.
    """
    accompaniment = []
    for note in melody_notes:
        accompaniment.append({
            "pitch": max(21, note["pitch"] - 12),  # 1 octave down
            "tick": note["tick"] + 8,  # 8 ticks later
            "duration": note["duration"],
            "program": 1
        })
    return accompaniment


@app.post("/generate_accompaniment", response_model=AccompanimentResponse)
async def generate_accompaniment(request: InferenceRequest):
    """
    Generate accompaniment by echoing melody (fake inference).
    Stateless - only echoes the notes in this request.
    """
    request_arrival_time = time.perf_counter()
    preprocess_start_time = time.perf_counter()
    
    melody_notes_dicts = [note.dict() for note in request.melody_notes]
    
    inference_start_time = time.perf_counter()
    
    time.sleep(0.01)
    
    accompaniment_dicts = generate_echo_accompaniment(
        melody_notes_dicts,
        request.generation_start_tick,
        request.generation_length_frames
    )
    
    inference_end_time = time.perf_counter()
    postprocess_start_time = time.perf_counter()
    
    response_output_time = time.perf_counter()
    
    return AccompanimentResponse(
        accompaniment=accompaniment_dicts,
        timings=Timings(
            request_arrival_time=request_arrival_time,
            response_output_time=response_output_time,
            preprocess_start_time=preprocess_start_time,
            inference_start_time=inference_start_time,
            inference_end_time=inference_end_time,
            postprocess_start_time=postprocess_start_time,
        ),
        generation_start_tick=request.generation_start_tick,
    )


@app.post("/inject_music", response_model=InjectionResponse)
async def inject_music(request: InjectionRequest):
    """
    Fake injection endpoint - just acknowledges the request.
    """
    global injection_state
    
    injection_state = {
        "is_injected": True,
        "injection_length_ticks": request.injection_length_ticks,
        "injection_file_path": request.injection_file_path,
    }
    
    return InjectionResponse(
        success=True,
        message="Fake injection successful",
        injection_length_ticks=request.injection_length_ticks,
        melody_notes_injected=0,
        accompaniment_notes_injected=0,
    )


@app.post("/inject_notes", response_model=DirectInjectionResponse)
async def inject_notes(request: DirectInjectionRequest):
    """
    Fake direct injection endpoint.
    """
    global injection_state, melody_history, accompaniment_history
    
    melody_history = [note.dict() for note in request.melody_notes]
    accompaniment_history = [note.dict() for note in request.accompaniment_notes]
    
    injection_state = {
        "is_injected": True,
        "injection_length_ticks": request.injection_length_ticks,
        "injection_file_path": None,
    }
    
    return DirectInjectionResponse(
        success=True,
        message="Fake injection successful",
        melody_notes_injected=len(request.melody_notes),
        accompaniment_notes_injected=len(request.accompaniment_notes),
        injection_length_ticks=request.injection_length_ticks,
    )


@app.get("/injection_status")
async def get_injection_status():
    """
    Get current injection status.
    """
    return injection_state


@app.post("/clear_history")
async def clear_history():
    """
    Clear history and injection state.
    """
    global injection_state, melody_history, accompaniment_history
    
    melody_history = []
    accompaniment_history = []
    injection_state = {
        "is_injected": False,
        "injection_length_ticks": 0,
        "injection_file_path": None,
    }
    
    return {"message": "History cleared successfully."}


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "server": "fake"}


if __name__ == "__main__":
    print("Starting StreamMUSE Fake Server...")
    print("This server echoes melody notes as accompaniment (no ML required)")
    print("API available at http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
