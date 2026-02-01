"""
StreamMUSE Server - Lekai Version

This is the server side for the StreamMUSE end to end system.
Uses event-stream protocol (note_on/note_off) instead of duration-based notes.
"""

import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import time
from contextlib import asynccontextmanager
from app.inference_engines.transformer_engine_lekai import InferenceEngineLekai


# ============================================================================
# Pydantic Models - Event-Stream Protocol
# ============================================================================


class MelodyNoteEvent(BaseModel):
    """
    Event-stream format for melody notes.
    Uses type (note_on/note_off) instead of duration.
    """

    type: str  # "note_on" or "note_off"
    pitch: int
    tick: int


class InferenceRequest(BaseModel):
    """Request model for accompaniment generation."""

    melody_notes: list[MelodyNoteEvent]
    generation_start_tick: int
    client_request_send_time: Optional[float] = None
    generation_length_frames: Optional[int] = None


class AccompanimentNoteEvent(BaseModel):
    """
    Event-stream format for accompaniment notes.
    Uses type (note_on/note_off) instead of duration.
    """

    type: str  # "note_on" or "note_off"
    pitch: int
    tick: int


class Timings(BaseModel):
    """Timing information for performance analysis."""

    request_arrival_time: float
    response_output_time: float
    preprocess_start_time: float
    inference_start_time: float
    inference_end_time: float
    postprocess_start_time: float


class AccompanimentResponse(BaseModel):
    """Response model for accompaniment generation."""

    accompaniment: list[AccompanimentNoteEvent]
    timings: Timings
    generation_start_tick: int


# For injection requests
class InjectionRequest(BaseModel):
    """Request model for music injection."""

    injection_file_path: str
    injection_length_ticks: int


class InjectionResponse(BaseModel):
    """Response model for music injection."""

    success: bool
    message: str
    injection_length_ticks: int
    melody_notes_injected: int
    accompaniment_notes_injected: int


# ============================================================================
# Global State
# ============================================================================

inference_engine: Optional[InferenceEngineLekai] = None
injection_state = {
    "is_injected": False,
    "injection_length_ticks": 0,
    "injection_file_path": None,
}


# ============================================================================
# Helper Functions
# ============================================================================


def notes_to_events(notes: list) -> list:
    """
    Convert duration-based notes to event-stream format.

    Input: [{"pitch": 60, "tick": 0, "duration": 4}, ...]
    Output: [{"type": "note_on", "pitch": 60, "tick": 0},
             {"type": "note_off", "pitch": 60, "tick": 4}, ...]
    """
    events = []
    for n in notes:
        pitch = n["pitch"]
        tick = n["tick"]
        duration = n.get("duration", 1)
        events.append({"type": "note_on", "pitch": pitch, "tick": tick})
        events.append({"type": "note_off", "pitch": pitch, "tick": tick + duration})
    # Sort by tick, note_off before note_on if same tick
    events.sort(key=lambda e: (e["tick"], 0 if e["type"] == "note_off" else 1))
    return events


# ============================================================================
# Lifespan & App Initialization
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler: load model on startup.
    """
    global inference_engine

    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if not checkpoint_path:
        print("Warning: CHECKPOINT_PATH environment variable not set.")
        print("Please set it to the path of your model checkpoint.")
        yield
        return

    try:
        print(f"Loading Lekai inference engine from: {checkpoint_path}")

        inference_engine = InferenceEngineLekai(
            checkpoint_path=checkpoint_path,
            model_size="llama",
            inference_mode="sliding_window",
        )

        print("✓ Inference engine loaded successfully.")

    except Exception as e:
        print(f"✗ Failed to load inference engine: {e}")
        import traceback

        traceback.print_exc()

    yield
    # Cleanup on shutdown (if needed)


app = FastAPI(title="StreamMUSE Inference Server (Lekai)", lifespan=lifespan)


# ============================================================================
# Endpoints
# ============================================================================


@app.post("/inject_music", response_model=InjectionResponse)
async def inject_music(request: InjectionRequest):
    """
    Inject music into the inference engine's history.
    Converts duration-based notes from MIDI files to event-stream format.
    """
    global injection_state

    if not inference_engine:
        return JSONResponse(
            status_code=503, content={"error": "Inference engine not loaded"}
        )

    try:
        # Check if melody file exists
        melody_file_path = request.injection_file_path
        if not os.path.exists(melody_file_path):
            return InjectionResponse(
                success=False,
                message=f"Injection file not found: {melody_file_path}",
                injection_length_ticks=0,
                melody_notes_injected=0,
                accompaniment_notes_injected=0,
            )

        # Derive accompaniment file path
        accompaniment_file_path = melody_file_path.replace("/mel/", "/acc/")

        if accompaniment_file_path == melody_file_path:
            return InjectionResponse(
                success=False,
                message=f"无法推导伴奏文件路径，旋律文件路径应包含 '/mel/' 目录: {melody_file_path}",
                injection_length_ticks=0,
                melody_notes_injected=0,
                accompaniment_notes_injected=0,
            )

        if not os.path.exists(accompaniment_file_path):
            return InjectionResponse(
                success=False,
                message=f"Accompaniment file not found: {accompaniment_file_path}",
                injection_length_ticks=0,
                melody_notes_injected=0,
                accompaniment_notes_injected=0,
            )

        # Clear existing history
        inference_engine.clear_history()

        # Set injection offset
        inference_engine.set_injection_offset(request.injection_length_ticks)

        # Read MIDI files (duration-based notes)
        from app.inference_engines.transformer_engine_lekai import midi_to_note

        melody_notes, _, _ = midi_to_note(
            melody_file_path, beat_div=inference_engine.ticks_per_beat
        )
        accompaniment_notes, _, _ = midi_to_note(
            accompaniment_file_path, beat_div=inference_engine.ticks_per_beat
        )

        # Filter notes within injection length
        melody_notes = [
            n for n in melody_notes if n["tick"] < request.injection_length_ticks
        ]
        accompaniment_notes = [
            n for n in accompaniment_notes if n["tick"] < request.injection_length_ticks
        ]

        # Convert duration-based notes to event-stream format for melody
        melody_events = notes_to_events(melody_notes)

        # Update engine history with events (for melody)
        inference_engine.melody_event_history.extend(melody_events)

        # For accompaniment, engine still uses duration-based history internally
        inference_engine.accompaniment_history.extend(accompaniment_notes)

        # Initialize active pitches based on injection events
        # Find pitches that are still "on" at injection_length_ticks
        active_at_end = set()
        for e in melody_events:
            if e["type"] == "note_on":
                active_at_end.add(e["pitch"])
            elif e["type"] == "note_off":
                active_at_end.discard(e["pitch"])
        inference_engine._active_melody_pitches = active_at_end

        # Update injection state
        injection_state = {
            "is_injected": True,
            "injection_length_ticks": request.injection_length_ticks,
            "injection_file_path": request.injection_file_path,
        }

        print(
            f"✓ 注入完成: {len(melody_events)} 个旋律事件, {len(accompaniment_notes)} 个伴奏音符"
        )

        return InjectionResponse(
            success=True,
            message="Music injected successfully",
            injection_length_ticks=request.injection_length_ticks,
            melody_notes_injected=len(melody_events),
            accompaniment_notes_injected=len(accompaniment_notes),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return InjectionResponse(
            success=False,
            message=f"Error injecting music: {str(e)}",
            injection_length_ticks=0,
            melody_notes_injected=0,
            accompaniment_notes_injected=0,
        )


@app.get("/injection_status")
async def get_injection_status():
    """Get current injection status."""
    return injection_state


@app.post("/clear_history")
async def clear_history():
    """Clear history and injection state."""
    global injection_state

    if inference_engine:
        inference_engine.clear_history()

    injection_state = {
        "is_injected": False,
        "injection_length_ticks": 0,
        "injection_file_path": None,
    }

    return {"success": True, "message": "History cleared"}


@app.post("/generate_accompaniment", response_model=AccompanimentResponse)
async def generate_accompaniment(request: InferenceRequest):
    """
    Receive list of note events from client (event-stream format).
    Returns generated list of accompaniment events with timing info.
    """
    request_arrival_time = time.perf_counter()

    if not inference_engine:
        return JSONResponse(
            status_code=503, content={"error": "Inference engine not loaded"}
        )

    # Convert Pydantic models to dicts (already in event-stream format)
    melody_notes_dicts = [note.model_dump() for note in request.melody_notes]

    # Call inference engine
    (
        accompaniment_events,
        preprocess_start_time,
        inference_start_time,
        inference_end_time,
        postprocess_start_time,
    ) = inference_engine.generate_accompaniment(
        melody_notes_dicts,
        generation_start_tick=request.generation_start_tick,
    )

    response_output_time = time.perf_counter()

    return AccompanimentResponse(
        accompaniment=accompaniment_events,
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_loaded": inference_engine is not None,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8988)
