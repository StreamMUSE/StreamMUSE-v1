"""Typed configuration models for the application layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class TempoConfig:
    bpm: float = 120.0
    ticks_per_beat: int = 4
    beats_per_bar: int = 4


InputType = Literal["midi_device", "keyboard", "midi_file", "list", "queue"]
OutputType = Literal["audio", "midi_file", "console", "websocket", "composite", "json_log", "session"]
InferenceLogDetail = Literal["summary", "full"]
InferenceType = Literal["http", "stanley"]
ModelName = Literal["stanley", "lekai", "lekai_prompt_continuation"]


@dataclass(frozen=True)
class InputConfig:
    type: InputType = "midi_device"
    midi_device_name: Optional[str] = None
    midi_file_path: Optional[str] = None
    midi_file_delay_ticks: int = 0
    midi_file_trim_leading_rest: bool = False
    injection_file: Optional[str] = None
    injection_length_ticks: int = 0
    injection_acc_file: Optional[str] = None


@dataclass(frozen=True)
class OutputConfig:
    type: OutputType = "console"
    midi_out_port: Optional[str] = None
    midi_file_output_path: Optional[str] = None
    inference_log_detail: InferenceLogDetail = "summary"


@dataclass(frozen=True)
class InferenceConfig:
    type: InferenceType = "http"
    server_generate_url: str = "http://localhost:8000/generate_accompaniment"
    timeout_s: float = 30.0
    model_name: ModelName = "stanley"
    inference_mode: str = "sliding_window"
    checkpoint_path: Optional[str] = None
    model_size: str = "0.12B"
    model_max_seq_len_frames: int = 96
    generation_length_frames: int = 20
    generation_interval_ticks: int = 2  # How often to trigger generation
    prompt_length_ticks: int = 32  # First 8 beats at 4 ticks/beat for prompt-continuation


@dataclass(frozen=True)
class ApplicationConfig:
    tempo: TempoConfig = TempoConfig()
    input: InputConfig = InputConfig()
    output: OutputConfig = OutputConfig()
    inference: InferenceConfig = InferenceConfig()
