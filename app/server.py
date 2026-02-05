"""
This is the server side for the StreamMUSE end to end system.
"""

import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import time
from contextlib import asynccontextmanager

# from app.inference_engines.transformer_engine import TransformerInferenceEngine
from app.inference_engines.transformer_engine_stanley import InferenceEngineStanley
from app.inference_engines.transformer_engine_lekai import InferenceEngineLekai


class MelodyNoteEvent(BaseModel):
    type: str
    pitch: int
    tick: int


class InferenceRequest(BaseModel):
    melody_notes: list[MelodyNoteEvent]
    generation_start_tick: int
    client_request_send_time: float
    generation_length_frames: Optional[int] = None
    prompt_length_ticks: Optional[int] = None
    # Analysis parameters
    inference_interval_ticks: Optional[int] = None
    tempo: Optional[float] = None
    assumed_network_latency_ms: Optional[float] = None


class AccompanimentNoteEvent(BaseModel):
    type: str
    pitch: int
    tick: int


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


# For injection requests (old file-based approach - deprecated)
class InjectionRequest(BaseModel):
    injection_file_path: str
    injection_length_ticks: int


class InjectionResponse(BaseModel):
    success: bool
    message: str
    injection_length_ticks: int
    melody_notes_injected: int
    accompaniment_notes_injected: int


# For direct note injection (new client-side approach)
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


# app = FastAPI(title='StreamMUSE Inference Server')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler: 在应用启动时加载模型
    """
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if not checkpoint_path:
        print("Fatal Error: CHECKPOINT_PATH environment variable is not set")
        print("Please run the server like: CHECKPOINT_PATH=path/to/model.ckpt uvicorn ...")
        exit()

    # Get model parameters from environment variables with defaults
    try:
        model_max_seq_len_frames = int(os.getenv("MODEL_MAX_SEQ_LEN_FRAMES", 96))
        generation_length_frames = int(os.getenv("GENERATION_LENGTH_FRAMES", 20))
        model_size = os.getenv("MODEL_SIZE", "0.12B")
    except ValueError:
        print("Fatal Error: Invalid integer value for model parameters in environment variables.")
        exit()

    # 验证 model_size 是否有效
    valid_model_sizes = ["small", "0.12B", "0.25B", "0.5B", "llama"]
    if model_size not in valid_model_sizes:
        print(f"Fatal Error: Invalid MODEL_SIZE '{model_size}'. Valid options: {valid_model_sizes}")
        exit()

    engine_type = os.getenv("ENGINE_TYPE", "stanley")
    inference_mode = os.getenv("INFERENCE_MODE", "sliding_window")

    global inference_engine, injection_state
    try:
        print(f"Loading model from {checkpoint_path}...")
        print(f"Using Engine Type: {engine_type}")
        print(f"Using Model Max Sequence Length (Frames): {model_max_seq_len_frames}")
        print(f"Using Generation Length (Frames): {generation_length_frames}")

        if engine_type == "lekai":
            inference_engine = InferenceEngineLekai(
                checkpoint_path=checkpoint_path,
                model_size=model_size,
                model_max_seq_len_frames=model_max_seq_len_frames,
                generation_length_frames=generation_length_frames,
                inference_mode=inference_mode,
            )
        else:
            inference_engine = InferenceEngineStanley(
                checkpoint_path=checkpoint_path,
                model_size=model_size,
                model_max_seq_len_frames=model_max_seq_len_frames,
                generation_length_frames=generation_length_frames,
            )
        print("Inference engine loaded successfully.")

        # 初始化注入状态
        injection_state = {
            "is_injected": False,
            "injection_length_ticks": 0,
            "injection_file_path": None,
        }

    except FileNotFoundError as e:
        print(f"Fatal Error: {e}")
        exit()
    yield
    # 这里可以添加关闭/清理逻辑（可选）


app = FastAPI(title="StreamMUSE Inference Server", lifespan=lifespan)


# 添加注入端点
@app.post("/inject_music", response_model=InjectionResponse)
async def inject_music(request: InjectionRequest):
    """
    注入一段音乐到推理引擎的历史中

    DEPRECATED: This endpoint requires MIDI files to be on the server.
    Use /inject_notes instead for client-side prompting.
    This endpoint is kept for backward compatibility.
    """
    global injection_state

    if not inference_engine:
        return JSONResponse(status_code=503, content={"error": "Inference engine not loaded"})

    try:
        # 检查文件是否存在
        melody_file_path = request.injection_file_path
        if not os.path.exists(melody_file_path):
            return InjectionResponse(
                success=False,
                message=f"Injection file not found: {melody_file_path}",
                injection_length_ticks=0,
                melody_notes_injected=0,
                accompaniment_notes_injected=0,
            )

        # 自动推导伴奏文件路径
        # 比如将 input/mel/001.mid 转换为 input/acc/001.mid
        accompaniment_file_path = melody_file_path.replace("/mel/", "/acc/")

        # 验证路径替换是否成功
        if accompaniment_file_path == melody_file_path:
            return InjectionResponse(
                success=False,
                message=f"无法推导伴奏文件路径，旋律文件路径应包含 '/mel/' 目录: {melody_file_path}",
                injection_length_ticks=0,
                melody_notes_injected=0,
                accompaniment_notes_injected=0,
            )

        # 检查伴奏文件是否存在
        if not os.path.exists(accompaniment_file_path):
            return InjectionResponse(
                success=False,
                message=f"Accompaniment file not found: {accompaniment_file_path}",
                injection_length_ticks=0,
                melody_notes_injected=0,
                accompaniment_notes_injected=0,
            )

        # 清除现有历史
        inference_engine.clear_history()

        # 设置注入偏移
        inference_engine.set_injection_offset(request.injection_length_ticks)

        # 读取注入文件
        from app.midi_input_script import midi_to_note

        # 读取旋律和伴奏
        melody_notes, _, _ = midi_to_note(melody_file_path, max_tick=request.injection_length_ticks)

        accompaniment_notes, _, _ = midi_to_note(accompaniment_file_path, max_tick=request.injection_length_ticks)

        # 过滤只保留指定长度内的音符
        melody_notes = [n for n in melody_notes if n["tick"] < request.injection_length_ticks]
        accompaniment_notes = [n for n in accompaniment_notes if n["tick"] < request.injection_length_ticks]

        # 注入到引擎历史中
        inference_engine.melody_history.extend(melody_notes)
        inference_engine.accompaniment_history.extend(accompaniment_notes)

        # 更新注入状态
        injection_state = {
            "is_injected": True,
            "injection_length_ticks": request.injection_length_ticks,
            "injection_file_path": request.injection_file_path,
        }

        print(f"注入完成: {len(melody_notes)} 个旋律音符, {len(accompaniment_notes)} 个伴奏音符")

        return InjectionResponse(
            success=True,
            message="Music injected successfully",
            injection_length_ticks=request.injection_length_ticks,
            melody_notes_injected=len(melody_notes),
            accompaniment_notes_injected=len(accompaniment_notes),
        )

    except Exception as e:
        return InjectionResponse(
            success=False,
            message=f"Error injecting music: {str(e)}",
            injection_length_ticks=0,
            melody_notes_injected=0,
            accompaniment_notes_injected=0,
        )


# 新的客户端侧注入端点
@app.post("/D", response_model=DirectInjectionResponse)
async def inject_notes(request: DirectInjectionRequest):
    """
    Inject notes directly into inference engine history.
    Client-side handles all file I/O and prompt selection.
    This is the new preferred method - server never touches disk.
    """
    if not inference_engine:
        return JSONResponse(
            status_code=503, content={"error": "Inference engine not loaded"}
        )

    try:
        # Convert Pydantic models to dicts
        melody_notes_dicts = [note.dict() for note in request.melody_notes]
        accompaniment_notes_dicts = [note.dict() for note in request.accompaniment_notes]

        # Clear existing history
        inference_engine.clear_history()

        # Direct injection
        inference_engine.melody_history.extend(melody_notes_dicts)
        inference_engine.accompaniment_history.extend(accompaniment_notes_dicts)

        # Set offset
        inference_engine.set_injection_offset(request.injection_length_ticks)

        print(f"Injected {len(melody_notes_dicts)} melody notes and {len(accompaniment_notes_dicts)} accompaniment notes")

        return DirectInjectionResponse(
            success=True,
            message="Notes injected successfully",
            melody_notes_injected=len(melody_notes_dicts),
            accompaniment_notes_injected=len(accompaniment_notes_dicts),
            injection_length_ticks=request.injection_length_ticks
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return DirectInjectionResponse(
            success=False,
            message=f"Error injecting notes: {str(e)}",
            melody_notes_injected=0,
            accompaniment_notes_injected=0,
            injection_length_ticks=0
        )


# 添加获取注入状态的端点
@app.get("/injection_status")
async def get_injection_status():
    """
    获取当前注入状态
    """
    global injection_state
    return injection_state


# Global state for accumulated latency tracking
accumulated_latency_ms = 0.0


def calculate_musical_timing_analysis(
    inference_duration_ms: float,
    generation_length_frames: int,
    inference_interval_ticks: Optional[int] = None,
    tempo: Optional[float] = None,
    assumed_network_latency_ms: Optional[float] = None,
):
    """
    Calculate compact timing analysis for real-time requests.

    Args:
        inference_duration_ms: Actual inference time in milliseconds
        generation_length_frames: Number of frames generated
        inference_interval_ticks: Specific interval to analyze (if provided)
        tempo: BPM for musical timing calculation
        assumed_network_latency_ms: Additional network latency to include

    Returns:
        String with one-line timing analysis
    """
    global accumulated_latency_ms

    # Check if we have all parameters for full analysis
    has_full_params = all(
        [inference_interval_ticks is not None, tempo is not None, assumed_network_latency_ms is not None]
    )

    if not has_full_params:
        # Show basic info and what's missing
        missing = []
        if inference_interval_ticks is None:
            missing.append("inference_interval_ticks")
        if tempo is None:
            missing.append("tempo")
        if assumed_network_latency_ms is None:
            missing.append("assumed_network_latency_ms")

        return (
            f"TIMING: Inference={inference_duration_ms:.1f}ms, "
            f"Gen={generation_length_frames}f | "
            f"Missing for full analysis: {', '.join(missing)}"
        )

    # Full analysis with all parameters
    tick_duration_ms = (60000 / tempo) / 4  # 4 ticks per beat
    target_musical_time = tick_duration_ms * inference_interval_ticks
    total_latency = inference_duration_ms + assumed_network_latency_ms
    difference = total_latency - target_musical_time

    # Update accumulated latency (doesn't go below 0)
    accumulated_latency_ms = max(0, accumulated_latency_ms + difference)

    status = "EARLY" if difference < 0 else "LATE" if difference > 0 else "EXACT"

    return (
        f"TIMING: Inf={inference_duration_ms:.1f}ms + Net={assumed_network_latency_ms:.1f}ms = "
        f"{total_latency:.1f}ms vs Target={target_musical_time:.1f}ms "
        f"({difference:+.1f}ms {status}) | "
        f"Accumulated={accumulated_latency_ms:.1f}ms | "
        f"Interval={inference_interval_ticks}ticks @ {tempo:.0f}BPM"
    )


@app.post("/generate_accompaniment", response_model=AccompanimentResponse)
async def generate_accompaniment(request: InferenceRequest):
    """
    Receive list of note events from client
    Returns generated list of accompaniment events with timing info.
    """
    request_arrival_time = time.perf_counter()

    if not inference_engine:
        return JSONResponse(status_code=503, content={"error": "Inference engine not loaded"})

    melody_notes_dicts = [note.dict() for note in request.melody_notes]

    # Use request-specific generation length or fall back to server default
    generation_length = request.generation_length_frames or inference_engine.generation_length_frames

    accompaniment_dicts, preprocess_start_time, inference_start_time, inference_end_time, postprocess_start_time = (
        inference_engine.generate_accompaniment(
            melody_notes_dicts,
            generation_start_tick=request.generation_start_tick,
            generation_length_frames=generation_length,
            prompt_length_ticks=request.prompt_length_ticks,
        )
    )
    print(f"Using generation length (frames): {request.generation_length_frames}")

    response_output_time = time.perf_counter()

    # Calculate and display musical timing analysis
    inference_duration_ms = (inference_end_time - inference_start_time) * 1000
    timing_analysis = calculate_musical_timing_analysis(
        inference_duration_ms,
        generation_length,
        inference_interval_ticks=request.inference_interval_ticks,
        tempo=request.tempo,
        assumed_network_latency_ms=request.assumed_network_latency_ms,
    )
    print(timing_analysis)

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


@app.post("/clear_history")
async def clear_history():
    """
    清除历史和注入状态
    """
    global injection_state, accumulated_latency_ms

    if inference_engine:
        inference_engine.clear_history()
        inference_engine.set_injection_offset(0)  # 重置注入偏移
        accumulated_latency_ms = 0.0  # Reset accumulated latency
        injection_state = {
            "is_injected": False,
            "injection_length_ticks": 0,
            "injection_file_path": None,
        }
        return {"message": "History, injection state, and accumulated latency cleared successfully."}
    return JSONResponse(status_code=503, content={"error": "Inference engine not loaded"})
