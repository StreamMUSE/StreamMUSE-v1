"""CLI entry point for StreamMUSE."""

from __future__ import annotations

import signal
import sys

from streammuse.application.config import ApplicationConfig
from streammuse.application.factories import (
    InferenceEngineFactory,
    InputSourceFactory,
    OutputSinkFactory,
)
from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.timing import PlaybackScheduler, Tempo
from streammuse.presentation.cli.config_parser import args_to_config, env_to_config, parse_args


def main() -> int:
    """Main CLI entry point."""
    # Parse arguments
    args = parse_args()

    # Load config from env (optional) and override with CLI args
    config = env_to_config()
    if config is None:
        config = args_to_config(args)
    else:
        # Merge env config with CLI args (CLI takes precedence)
        config = args_to_config(args)

    # Create factories and build dependencies
    input_source = InputSourceFactory.create(config)
    output_sink = OutputSinkFactory.create(config)
    inference_engine = InferenceEngineFactory.create(config)

    # Create tempo and scheduler
    tempo = Tempo(
        bpm=config.tempo.bpm,
        ticks_per_beat=config.tempo.ticks_per_beat,
        beats_per_bar=config.tempo.beats_per_bar,
    )
    scheduler = PlaybackScheduler()

    # Create service
    service = RealTimeMusicService(
        input_source=input_source,
        inference_engine=inference_engine,
        output_sink=output_sink,
        tempo=tempo,
        scheduler=scheduler,
        generation_interval_ticks=config.inference.generation_interval_ticks,
        generation_length_frames=config.inference.generation_length_frames,
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...")
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start service
    print("Starting StreamMUSE...")
    print(f"  Tempo: {tempo.bpm} BPM, {tempo.ticks_per_beat} ticks/beat, {tempo.beats_per_bar} beats/bar")
    print(f"  Input: {config.input.type}")
    print(f"  Output: {config.output.type}")
    print(f"  Inference: {config.inference.type}")
    print(f"  Generation interval: {config.inference.generation_interval_ticks} ticks")
    print(f"  Generation length: {config.inference.generation_length_frames} frames")
    print("\nPress Ctrl+C to stop\n")

    try:
        service.start(max_ticks=args.max_ticks)
        # Wait for service to finish (or be interrupted)
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
