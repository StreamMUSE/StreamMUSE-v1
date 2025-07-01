"""
This is the server side for the StreamMUSE end to end system.
"""

import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import time
from contextlib import asynccontextmanager

from app.inference_engines.transformer_engine import TransformerInferenceEngine

class MelodyNoteEvent(BaseModel):
    pitch: int
    tick: int
    duration: int

class InferenceRequest(BaseModel):
    melody_notes: list[MelodyNoteEvent]
    generation_start_tick: int
    client_request_send_time: float

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

# app = FastAPI(title='StreamMUSE Inference Server')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler: 在应用启动时加载模型
    """
    checkpoint_path = os.getenv('CHECKPOINT_PATH')
    if not checkpoint_path:
        print('Fatal Error: CHECKPOINT_PATH environment variable is not set')
        print("Please run the server like: CHECKPOINT_PATH=path/to/model.ckpt uvicorn ...")
        exit()

    # Get model parameters from environment variables with defaults
    try:
        model_max_seq_len_frames = int(os.getenv('MODEL_MAX_SEQ_LEN_FRAMES', 96))
        generation_length_frames = int(os.getenv('GENERATION_LENGTH_FRAMES', 20))
    except ValueError:
        print("Fatal Error: Invalid integer value for model parameters in environment variables.")
        exit()

    global inference_engine
    try:
        print(f"Loading model from {checkpoint_path}...")
        print(f"Using Model Max Sequence Length (Frames): {model_max_seq_len_frames}")
        print(f"Using Generation Length (Frames): {generation_length_frames}")
        inference_engine = TransformerInferenceEngine(
            checkpoint_path=checkpoint_path,
            model_max_seq_len_frames=model_max_seq_len_frames,
            generation_length_frames=generation_length_frames
        )
        print("Inference engine loaded successfully.")
    except FileNotFoundError as e:
        print(f"Fatal Error: {e}")
        exit()
    yield
    # 这里可以添加关闭/清理逻辑（可选）

app = FastAPI(title='StreamMUSE Inference Server', lifespan=lifespan)

@app.post('/generate_accompaniment', response_model=AccompanimentResponse)
async def generate_accompaniment(request: InferenceRequest):
    """
    Receive list of note events from client
    Returns generated list of accompaniment events with timing info.
    """
    request_arrival_time = time.perf_counter()

    from fastapi.responses import JSONResponse
    if not inference_engine:
        return JSONResponse(status_code=503, content={"error": "Inference engine not loaded"})
    
    melody_notes_dicts = [note.dict() for note in request.melody_notes]
    
    accompaniment_dicts, preprocess_start_time, inference_start_time, inference_end_time, postprocess_start_time = inference_engine.generate_accompaniment(
        melody_notes_dicts,
        generation_start_tick=request.generation_start_tick
    )
    
    response_output_time = time.perf_counter()

    return AccompanimentResponse(
        accompaniment=accompaniment_dicts,
        timings=Timings(
            request_arrival_time=request_arrival_time,
            response_output_time=response_output_time,
            preprocess_start_time=preprocess_start_time,
            inference_start_time=inference_start_time,
            inference_end_time=inference_end_time,
            postprocess_start_time=postprocess_start_time
        ),
        generation_start_tick=request.generation_start_tick
    )

@app.post('/clear_history')
async def clear_history():
    """
    Clears the inference engine's history.
    """
    if inference_engine:
        inference_engine.clear_history()
        return {"message": "History cleared successfully."}
    return JSONResponse(status_code=503, content={"error": "Inference engine not loaded"})