"""Parse CLI arguments and environment variables into ApplicationConfig."""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional

from streammuse.application.services.input_timing import clamp_snap_forward_fraction
from streammuse.application.config import (
    ApplicationConfig,
    InferenceConfig,
    InputConfig,
    OutputConfig,
    RapConfig,
    TempoConfig,
)
from streammuse.application.rap.rhythm import available_patterns


PROMPT_SELECTION_MODES = (
    "single",
    "batch_first",
    "rule_s",
    "rule_s_v3",
    "rule_s_if_else",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="StreamMUSE: Real-time AI music generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Tempo configuration
    parser.add_argument("--tempo", type=float, default=120.0, help="BPM (beats per minute)")
    parser.add_argument("--ticks-per-beat", type=int, default=4, help="Ticks per beat")
    parser.add_argument("--beats-per-bar", type=int, default=4, help="Beats per bar")

    # Input configuration
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["midi_device", "keyboard", "midi_file", "list"],
        default="midi_device",
        help="Input source type",
    )
    parser.add_argument("--midi-device-name", type=str, default=None, help="MIDI input device name")
    parser.add_argument("--midi-file-path", type=str, default=None, help="Path to MIDI file (for simulation)")
    parser.add_argument("--midi-file-delay-ticks", type=int, default=0, help="Delay before starting MIDI file playback")
    parser.add_argument(
        "--midi-file-trim-leading-rest",
        action="store_true",
        help="Start MIDI-file playback from the first retained note instead of preserving leading silence",
    )
    parser.add_argument(
        "--midi-file-source-tick-mode",
        action="store_true",
        help=(
            "Synchronously replay pre-quantized MIDI ticks in prompt-continuation "
            "mode instead of quantizing file events from wall-clock arrival"
        ),
    )
    parser.add_argument(
        "--injection-file",
        type=str,
        default=None,
        help="Path to melody MIDI file to inject as prompt",
    )
    parser.add_argument(
        "--injection-length",
        type=int,
        default=0,
        help="Number of ticks to inject (e.g., 16 for 4 beats)",
    )
    parser.add_argument(
        "--inject-acc-file",
        type=str,
        default=None,
        help="Optional accompaniment MIDI path (default: replace '/mel/' with '/acc/')",
    )

    # Output configuration
    parser.add_argument(
        "--output-type",
        type=str,
        choices=["audio", "console", "midi_file", "websocket", "composite", "json_log", "session"],
        default="console",
        help=(
            "Output sink type. console/audio/websocket/session/composite also auto-record "
            "a combined MIDI in log dir; json_log does not."
        ),
    )
    parser.add_argument("--web-host", type=str, default="127.0.0.1", help="Host/interface for the web viewer to bind (use 0.0.0.0 to allow LAN access)")
    parser.add_argument("--web-port", type=int, default=8001, help="Port for the web viewer")
    parser.add_argument("--midi-out-port", type=str, default=None, help="MIDI output port name (for audio output)")
    parser.add_argument("--midi-file-output-path", type=str, default=None, help="Path to save MIDI file output")
    parser.add_argument(
        "--close-active-notes-on-finalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Close notes still active at the final observed tick when exporting MIDI",
    )
    parser.add_argument("--enable-metronome", action="store_true", help="Play MIDI metronome clicks aligned with playback")
    parser.add_argument("--metronome-port", type=str, default=None, help="MIDI output port for metronome clicks")
    parser.add_argument("--metronome-channel", type=int, default=9, help="MIDI channel for metronome clicks")

    # Logging configuration
    parser.add_argument("--log-dir", type=str, default="logs", help="Base directory for session logs")
    parser.add_argument(
        "--inference-log-detail",
        type=str,
        choices=["summary", "full"],
        default="summary",
        help="Inference logging detail level (full can significantly increase log size)",
    )
    parser.add_argument(
        "--session-artifact-tier",
        type=str,
        choices=["normal", "debug"],
        default="debug",
        help="Session artifact tier: normal keeps core MIDI/trace files; debug keeps full JSON/log diagnostics",
    )
    parser.add_argument("--enable-performance-tracking", action="store_true", help="Enable detailed performance metrics calculation")
    parser.add_argument(
        "--log-input-quantization",
        action="store_true",
        help="Write input_quantization_trace.jsonl in the session directory",
    )

    # Inference configuration
    parser.add_argument(
        "--inference-type",
        type=str,
        choices=["http", "stanley"],
        default="http",
        help="Inference engine type",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000/generate_accompaniment",
        help="HTTP inference server URL (recommended production path)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        choices=["stanley", "lekai"],
        default="stanley",
        help="Model backend selected on HTTP server",
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=["sliding_window", "stateful"],
        default="sliding_window",
        help="Inference mode hint passed to server",
    )
    parser.add_argument("--timeout-s", type=float, default=30.0, help="HTTP request timeout (seconds)")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to model checkpoint (for Stanley engine)")
    parser.add_argument("--model-size", type=str, default="0.12B", help="Model size (for Stanley engine)")
    parser.add_argument("--model-max-seq-len-frames", type=int, default=96, help="Model max sequence length (frames)")
    parser.add_argument("--generation-length-frames", type=int, default=20, help="Frames to generate per request")
    parser.add_argument("--generation-interval-ticks", type=int, default=2, help="Ticks between generation requests")
    parser.add_argument(
        "--prompt-length-ticks",
        type=int,
        default=32,
        help="Prompt window length for prompt-continuation mode",
    )
    parser.add_argument(
        "--model-condition-bpm",
        type=int,
        default=None,
        help=(
            "Optional BPM token sent to the HTTP model, independent of wall-clock "
            "--tempo; defaults to the playback tempo for backward compatibility"
        ),
    )
    parser.add_argument(
        "--prompt-selection-mode",
        choices=PROMPT_SELECTION_MODES,
        default=None,
        help="Prompt candidate selection mode; defaults to the backend setting",
    )
    parser.add_argument(
        "--prompt-batch-candidates",
        type=int,
        default=None,
        help="Prompt candidates generated per batch; defaults to the backend setting",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Shared Prompt and Continuation sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Shared Prompt and Continuation nucleus-sampling threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Shared Prompt and Continuation top-k sampling limit",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Shared Prompt and Continuation repetition penalty",
    )

    # Optional beat-aligned rap text layer
    parser.add_argument("--rap-topic", type=str, default=None, help="Enable live rap text for this topic")
    parser.add_argument("--rap-pattern", choices=available_patterns(), default="boom_bap", help="Rap rhythm preset")
    parser.add_argument(
        "--rap-generator",
        choices=("phrase_bank", "local_chat"),
        default="phrase_bank",
        help="Lyric candidate source",
    )
    parser.add_argument("--rap-lookahead-bars", type=int, default=2, help="Future bars kept ready")
    parser.add_argument("--rap-candidate-count", type=int, default=12, help="Candidates scored per lyric request")
    parser.add_argument("--rap-model-url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible rap model URL")
    parser.add_argument("--rap-model", type=str, default="local-model", help="Rap model name")
    parser.add_argument("--rap-timeout-s", type=float, default=5.0, help="Rap model request timeout in seconds")

    # Runtime options
    parser.add_argument(
        "--continuation-mode",
        choices=("standard", "prompt_continuation"),
        default="standard",
        help="Realtime continuation flow to run",
    )
    parser.add_argument(
        "--count-in-beats",
        type=int,
        default=0,
        help="Number of beats to click before accepting input and sending inference requests",
    )
    parser.add_argument(
        "--input-snap-forward-fraction",
        type=float,
        default=0.4,
        help=(
            "Fraction of a tick near the end of each tick that realtime input "
            "snaps forward to the next tick"
        ),
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Legacy alias for exclusive --run-stop-tick",
    )
    parser.add_argument(
        "--analysis-end-tick",
        type=int,
        default=None,
        help="Exclusive clean-reference analysis horizon",
    )
    parser.add_argument(
        "--last-input-note-off-tick",
        type=int,
        default=None,
        help="Last source melody note-off tick recorded in validity artifacts",
    )
    parser.add_argument(
        "--request-cutoff-tick",
        type=int,
        default=None,
        help="Inclusive, beat-aligned last generation-start tick",
    )
    parser.add_argument(
        "--run-stop-tick",
        type=int,
        default=None,
        help="Exclusive playback/drain horizon",
    )
    parser.add_argument(
        "--tail-beats",
        type=int,
        default=None,
        help=(
            "Derive run-stop from ceil(last-input-note-off/beat)+tail; "
            "requires --last-input-note-off-tick"
        ),
    )
    parser.add_argument(
        "--drain-timeout-s",
        type=float,
        default=10.0,
        help="Hard timeout for queued/in-flight inference drain",
    )

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> ApplicationConfig:
    """Convert parsed arguments to ApplicationConfig."""
    # Tempo config
    tempo = TempoConfig(
        bpm=float(args.tempo),
        ticks_per_beat=int(args.ticks_per_beat),
        beats_per_bar=int(args.beats_per_bar),
    )

    # Input config
    input_config = InputConfig(
        type=args.input_mode,  # type: ignore
        midi_device_name=args.midi_device_name,
        midi_file_path=args.midi_file_path,
        midi_file_delay_ticks=int(args.midi_file_delay_ticks),
        midi_file_trim_leading_rest=bool(
            getattr(args, "midi_file_trim_leading_rest", False)
        ),
        midi_file_source_tick_mode=bool(
            getattr(args, "midi_file_source_tick_mode", False)
        ),
        injection_file=getattr(args, "injection_file", None),
        injection_length_ticks=int(getattr(args, "injection_length", 0) or 0),
        injection_acc_file=getattr(args, "inject_acc_file", None),
    )

    # Output config
    output_config = OutputConfig(
        type=args.output_type,  # type: ignore
        midi_out_port=args.midi_out_port,
        midi_file_output_path=args.midi_file_output_path,
        close_active_notes_on_finalize=bool(
            getattr(args, "close_active_notes_on_finalize", True)
        ),
        inference_log_detail=getattr(args, "inference_log_detail", "summary"),  # type: ignore
        session_artifact_tier=getattr(args, "session_artifact_tier", "debug"),  # type: ignore
        metronome_enabled=bool(getattr(args, "enable_metronome", False)),
        metronome_port=getattr(args, "metronome_port", None),
        metronome_channel=int(getattr(args, "metronome_channel", 9)),
    )

    prompt_selection_mode = getattr(args, "prompt_selection_mode", None)
    prompt_batch_candidates = getattr(args, "prompt_batch_candidates", None)
    temperature = getattr(args, "temperature", None)
    top_p = getattr(args, "top_p", None)
    top_k = getattr(args, "top_k", None)
    repetition_penalty = getattr(args, "repetition_penalty", None)
    if (
        prompt_selection_mode is not None
        and prompt_selection_mode not in PROMPT_SELECTION_MODES
    ):
        raise ValueError(
            "prompt_selection_mode must be one of: "
            + ", ".join(PROMPT_SELECTION_MODES)
        )
    if prompt_batch_candidates is not None and prompt_batch_candidates < 1:
        raise ValueError("prompt_batch_candidates must be >= 1")
    if (
        prompt_selection_mode not in (None, "single")
        and prompt_batch_candidates is not None
        and prompt_batch_candidates < 2
    ):
        raise ValueError(
            "prompt_batch_candidates must be >= 2 for non-single selection modes"
        )
    if temperature is not None and (
        not math.isfinite(temperature) or temperature < 0
    ):
        raise ValueError("temperature must be >= 0")
    if top_p is not None and (
        not math.isfinite(top_p) or not 0 <= top_p <= 1
    ):
        raise ValueError("top_p must be between 0 and 1")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be >= 0")
    if repetition_penalty is not None and (
        not math.isfinite(repetition_penalty) or repetition_penalty <= 0
    ):
        raise ValueError("repetition_penalty must be > 0")

    # Inference config
    inference_config = InferenceConfig(
        type=args.inference_type,  # type: ignore
        server_generate_url=args.server_url,
        timeout_s=float(args.timeout_s),
        model_name=getattr(args, "model_name", "stanley"),  # type: ignore
        inference_mode=getattr(args, "inference_mode", "sliding_window"),
        checkpoint_path=args.checkpoint_path,
        model_size=args.model_size,
        model_max_seq_len_frames=int(args.model_max_seq_len_frames),
        generation_length_frames=int(args.generation_length_frames),
        generation_interval_ticks=int(args.generation_interval_ticks),
        prompt_length_ticks=max(1, int(getattr(args, "prompt_length_ticks", 32))),
        model_condition_bpm=(
            int(args.model_condition_bpm)
            if getattr(args, "model_condition_bpm", None) is not None
            else None
        ),
        prompt_selection_mode=prompt_selection_mode,
        prompt_batch_candidates=prompt_batch_candidates,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    raw_topic = getattr(args, "rap_topic", None)
    rap_topic = str(raw_topic).strip() if raw_topic is not None else None
    rap_config = RapConfig(
        topic=rap_topic or None,
        pattern=getattr(args, "rap_pattern", "boom_bap"),  # type: ignore[arg-type]
        generator=getattr(args, "rap_generator", "phrase_bank"),  # type: ignore[arg-type]
        lookahead_bars=max(1, int(getattr(args, "rap_lookahead_bars", 2) or 0)),
        candidate_count=max(1, int(getattr(args, "rap_candidate_count", 12) or 0)),
        model_url=str(getattr(args, "rap_model_url", "http://localhost:8000/v1")),
        model=str(getattr(args, "rap_model", "local-model")),
        timeout_s=max(0.1, float(getattr(args, "rap_timeout_s", 5.0) or 0.0)),
    )

    continuation_mode = getattr(args, "continuation_mode", "standard")
    if continuation_mode == "prompt_continuation" and rap_config.topic:
        raise ValueError("rap cannot be combined with prompt_continuation mode")
    if input_config.midi_file_source_tick_mode:
        if input_config.type != "midi_file":
            raise ValueError(
                "midi_file_source_tick_mode requires input_mode=midi_file"
            )
        if continuation_mode != "prompt_continuation":
            raise ValueError(
                "midi_file_source_tick_mode requires prompt_continuation mode"
            )

    return ApplicationConfig(
        tempo=tempo,
        input=input_config,
        output=output_config,
        inference=inference_config,
        rap=rap_config,
        continuation_mode=continuation_mode,  # type: ignore[arg-type]
        count_in_beats=max(0, int(getattr(args, "count_in_beats", 0) or 0)),
        input_snap_forward_fraction=clamp_snap_forward_fraction(
            float(getattr(args, "input_snap_forward_fraction", 0.4))
        ),
        input_quantization_trace_enabled=bool(
            getattr(args, "log_input_quantization", False)
        ),
    )


def env_to_config() -> Optional[ApplicationConfig]:
    """Load configuration from environment variables (optional)."""
    # For now, return None (can be extended later)
    return None
