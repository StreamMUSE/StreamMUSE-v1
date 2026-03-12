"""CLI entry point for StreamMUSE."""

from __future__ import annotations

import atexit
import signal
import sys

from streammuse.application.config import ApplicationConfig
from streammuse.application.factories import (
    InferenceEngineFactory,
    InputSourceFactory,
    OutputSinkFactory,
)
from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.logging import SessionManager
from streammuse.domain.timing import PlaybackScheduler, Tempo
from streammuse.infrastructure.output import SessionLoggerOutputSink
from streammuse.presentation.cli.config_parser import args_to_config, env_to_config, parse_args


def main() -> int:
    """Main CLI entry point."""
    args = parse_args()

    config = env_to_config()
    if config is None:
        config = args_to_config(args)
    else:
        config = args_to_config(args)

    session_manager = None
    session_config: dict[str, object] = {}
    if config.output.type in ["json_log", "session", "composite"]:
        session_manager = SessionManager(args.log_dir)
        session_manager.create_session_directory()
        session_config = {
            "tempo_bpm": config.tempo.bpm,
            "ticks_per_beat": config.tempo.ticks_per_beat,
            "beats_per_bar": config.tempo.beats_per_bar,
            "input_type": config.input.type,
            "output_type": config.output.type,
            "inference_type": config.inference.type,
            "generation_interval_ticks": config.inference.generation_interval_ticks,
            "generation_length_frames": config.inference.generation_length_frames,
        }
        session_manager.save_config(session_config)

    input_source = InputSourceFactory.create(config)
    output_sink = OutputSinkFactory.create(config, session_manager)
    inference_engine = InferenceEngineFactory.create(config)

    tempo = Tempo(
        bpm=config.tempo.bpm,
        ticks_per_beat=config.tempo.ticks_per_beat,
        beats_per_bar=config.tempo.beats_per_bar,
    )
    scheduler = PlaybackScheduler()

    service = RealTimeMusicService(
        input_source=input_source,
        inference_engine=inference_engine,
        output_sink=output_sink,
        tempo=tempo,
        scheduler=scheduler,
        generation_interval_ticks=config.inference.generation_interval_ticks,
        generation_length_frames=config.inference.generation_length_frames,
    )

    def cleanup() -> None:
        output_sink.close()
        if session_manager and isinstance(output_sink, SessionLoggerOutputSink):
            output_sink.save_metrics(session_config)
            session_manager.save_summary(
                {
                    "status": "completed",
                    "session_id": session_manager.get_session_id(),
                }
            )

    atexit.register(cleanup)

    def signal_handler(sig, frame):
        print("\nShutting down...")
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Starting StreamMUSE...")
    print(f"  Tempo: {tempo.bpm} BPM, {tempo.ticks_per_beat} ticks/beat, {tempo.beats_per_bar} beats/bar")
    print(f"  Input: {config.input.type}")
    print(f"  Output: {config.output.type}")
    print(f"  Inference: {config.inference.type}")
    print(f"  Generation interval: {config.inference.generation_interval_ticks} ticks")
    print(f"  Generation length: {config.inference.generation_length_frames} frames")
    if session_manager:
        print(f"  Logging: {session_manager.get_session_dir()}")
    print("\nPress Ctrl+C to stop\n")

    try:
        service.start(max_ticks=args.max_ticks)
        while service.running:
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
