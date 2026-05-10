"""Parse CLI arguments and environment variables into ApplicationConfig."""

from __future__ import annotations

import argparse
import os
from typing import Optional

from streammuse.application.config import (
    ApplicationConfig,
    InferenceConfig,
    InputConfig,
    OutputConfig,
    TempoConfig,
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
        help="For MIDI-file simulation, emit from the first retained note instead of waiting through leading rests",
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
    parser.add_argument("--midi-out-port", type=str, default=None, help="MIDI output port name (for audio output)")
    parser.add_argument("--midi-file-output-path", type=str, default=None, help="Path to save MIDI file output")

    # Logging configuration
    parser.add_argument("--log-dir", type=str, default="logs", help="Base directory for session logs")
    parser.add_argument(
        "--inference-log-detail",
        type=str,
        choices=["summary", "full"],
        default="summary",
        help="Inference logging detail level (full can significantly increase log size)",
    )
    parser.add_argument("--enable-performance-tracking", action="store_true", help="Enable detailed performance metrics calculation")

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
        choices=["stanley", "lekai", "lekai_prompt_continuation"],
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
        help="Prompt window length for lekai_prompt_continuation (8 beats at 4 ticks/beat)",
    )

    # Runtime options
    parser.add_argument("--max-ticks", type=int, default=None, help="Maximum ticks to run (for testing)")

    # Web UI options (used by `streammuse-web`; ignored by the CLI entry point)
    parser.add_argument("--web-host", type=str, default="127.0.0.1", help="Web UI bind host (streammuse-web only)")
    parser.add_argument("--web-port", type=int, default=8001, help="Web UI port (streammuse-web only)")
    parser.add_argument("--prompts-dir", type=str, default="prompts", help="Prompt library directory (streammuse-web only)")

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
        midi_file_trim_leading_rest=bool(getattr(args, "midi_file_trim_leading_rest", False)),
        injection_file=getattr(args, "injection_file", None),
        injection_length_ticks=int(getattr(args, "injection_length", 0) or 0),
        injection_acc_file=getattr(args, "inject_acc_file", None),
    )

    # Output config
    output_config = OutputConfig(
        type=args.output_type,  # type: ignore
        midi_out_port=args.midi_out_port,
        midi_file_output_path=args.midi_file_output_path,
        inference_log_detail=getattr(args, "inference_log_detail", "summary"),  # type: ignore
    )

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
        prompt_length_ticks=int(getattr(args, "prompt_length_ticks", 32)),
    )

    return ApplicationConfig(
        tempo=tempo,
        input=input_config,
        output=output_config,
        inference=inference_config,
    )


def env_to_config() -> Optional[ApplicationConfig]:
    """Load configuration from environment variables (optional)."""
    # For now, return None (can be extended later)
    return None
