"""Typed configuration models for the application layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True)
class TempoConfig:
    bpm: float = 120.0
    ticks_per_beat: int = 4
    beats_per_bar: int = 4


InputType = Literal["midi_device", "keyboard", "midi_file", "list"]
OutputType = Literal["audio", "midi_file", "console", "websocket", "composite", "json_log", "session"]
InferenceLogDetail = Literal["summary", "full"]
SessionArtifactTier = Literal["normal", "debug"]
InferenceType = Literal["http", "stanley"]
ModelName = Literal["stanley", "lekai"]
ContinuationMode = Literal["standard", "prompt_continuation"]
PromptSelectionMode = Literal[
    "single",
    "batch_first",
    "rule_s",
    "rule_s_v3",
    "rule_s_if_else",
]
RapPattern = Literal["boom_bap", "straight_8", "trap_sparse"]
RapGenerator = Literal["phrase_bank", "local_chat"]


@dataclass(frozen=True)
class InputConfig:
    type: InputType = "midi_device"
    midi_device_name: Optional[str] = None
    midi_file_path: Optional[str] = None
    midi_file_delay_ticks: int = 0
    midi_file_trim_leading_rest: bool = False
    midi_file_source_tick_mode: bool = False
    injection_file: Optional[str] = None
    injection_length_ticks: int = 0
    injection_acc_file: Optional[str] = None


@dataclass(frozen=True)
class OutputConfig:
    type: OutputType = "console"
    midi_out_port: Optional[str] = None
    midi_file_output_path: Optional[str] = None
    inference_log_detail: InferenceLogDetail = "summary"
    session_artifact_tier: SessionArtifactTier = "debug"
    metronome_enabled: bool = False
    metronome_port: Optional[str] = None
    metronome_channel: int = 9
    close_active_notes_on_finalize: bool = True


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
    prompt_length_ticks: int = 32
    # Model-conditioning tempo can intentionally differ from wall-clock
    # playback tempo (for example, robustness RT playback at 60 BPM while the
    # Lekai prompt remains conditioned at its trained 120 BPM).
    model_condition_bpm: Optional[int] = None
    prompt_selection_mode: Optional[PromptSelectionMode] = None
    prompt_batch_candidates: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None


@dataclass(frozen=True)
class RapConfig:
    """Optional rolling beat-aligned text layer for the real-time CLI."""

    topic: Optional[str] = None
    pattern: RapPattern = "boom_bap"
    generator: RapGenerator = "phrase_bank"
    lookahead_bars: int = 2
    candidate_count: int = 12
    model_url: str = "http://localhost:8000/v1"
    model: str = "local-model"
    timeout_s: float = 5.0


@dataclass(frozen=True)
class ApplicationConfig:
    tempo: TempoConfig = TempoConfig()
    input: InputConfig = InputConfig()
    output: OutputConfig = OutputConfig()
    inference: InferenceConfig = InferenceConfig()
    rap: RapConfig = field(default_factory=RapConfig)
    continuation_mode: ContinuationMode = "standard"
    count_in_beats: int = 0
    input_snap_forward_fraction: float = 0.4
    input_quantization_trace_enabled: bool = False
