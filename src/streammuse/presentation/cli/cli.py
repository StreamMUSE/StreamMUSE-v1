"""CLI entry point for StreamMUSE."""

from __future__ import annotations

import atexit
import os
import signal
import sys

from streammuse.application.config import ApplicationConfig
from streammuse.application.factories import (
    InferenceEngineFactory,
    InputSourceFactory,
    OutputSinkFactory,
)
from streammuse.application.runtime import RuntimeSessionBuilder
from streammuse.application.runtime import builder as runtime_builder_module
from streammuse.application.services.prompt_continuation_realtime_service import (
    PromptContinuationRealtimeService,
)
from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.interfaces import InferenceEngine
from streammuse.domain.musical import EventType, MusicalEvent, Note
from streammuse.infrastructure.inference.prompt_continuation_http_client import (
    PromptContinuationHttpClient,
)
from streammuse.infrastructure.input.midi_file import MidiFileInput
from streammuse.presentation.cli.config_parser import args_to_config, env_to_config, parse_args


class _CliInjectionFailed(RuntimeError):
    pass


def _notes_to_musical_events(notes: list[dict[str, int]], velocity: int = 80) -> list[MusicalEvent]:
    """Convert duration notes to note_on/note_off events via domain Note.to_events()."""
    events: list[MusicalEvent] = []
    for raw in notes:
        note = Note(
            pitch=int(raw["pitch"]),
            tick=int(raw["tick"]),
            duration=max(1, int(raw.get("duration", 1))),
            velocity=int(velocity),
        )
        events.extend(note.to_events())

    events.sort(key=lambda e: (e.tick, 0 if e.event_type == EventType.NOTE_OFF else 1))
    return events


def _perform_injection(inference_engine: InferenceEngine, config: ApplicationConfig) -> int:
    """Inject prompt notes into server history. Returns injected tick length on success, else 0."""
    injection_file = config.input.injection_file
    injection_length = int(config.input.injection_length_ticks)
    if not injection_file:
        return 0

    acc_file = config.input.injection_acc_file
    if acc_file is None:
        acc_candidate = injection_file.replace("/mel/", "/acc/")
        if acc_candidate == injection_file:
            print(f"Warning: Cannot derive acc path from {injection_file}")
            acc_file = None
        else:
            acc_file = acc_candidate

    print(f"Injecting from: {injection_file} (first {injection_length} ticks)")
    try:
        mel_notes, _resolution, _max_tick = MidiFileInput._midi_to_notes(
            midi_path=injection_file,
            beat_div=config.tempo.ticks_per_beat,
            min_pitch=0,
            max_pitch=127,
            program=None,
            max_tick=injection_length,
        )
        mel_events = _notes_to_musical_events(mel_notes)
        mel_note_count = len(mel_notes)

        acc_events: list[MusicalEvent] = []
        acc_note_count = 0
        if acc_file and os.path.exists(acc_file):
            acc_notes, _resolution, _max_tick = MidiFileInput._midi_to_notes(
                midi_path=acc_file,
                beat_div=config.tempo.ticks_per_beat,
                min_pitch=0,
                max_pitch=127,
                program=None,
                max_tick=injection_length,
            )
            acc_events = _notes_to_musical_events(acc_notes)
            acc_note_count = len(acc_notes)
            print(f"  Loaded accompaniment: {acc_note_count} notes")
        elif acc_file:
            print(f"Warning: Accompaniment file not found, melody-only injection: {acc_file}")

        inference_engine.clear_history()
        inference_engine.inject_history(
            melody_events=mel_events,
            accompaniment_events=acc_events,
            injection_length_ticks=injection_length,
        )

        print(f"✓ Injected: {mel_note_count} melody notes, {acc_note_count} acc notes")
        return injection_length
    except Exception as exc:
        print(f"✗ Injection failed: {exc}")
        return 0


def main() -> int:
    """Main CLI entry point."""
    args = parse_args()

    config = env_to_config()
    if config is None:
        config = args_to_config(args)
    else:
        config = args_to_config(args)

    if config.input.injection_file:
        if config.continuation_mode != "standard":
            print("Error: --injection-file is only supported with --continuation-mode standard")
            return 1
        if config.input.type != "midi_file":
            print("Error: --injection-file is only supported with --input-mode midi_file")
            return 1
        if config.input.injection_length_ticks <= 0:
            print("Error: --injection-length must be positive")
            return 1
        if not os.path.exists(config.input.injection_file):
            print(f"Error: Injection file not found: {config.input.injection_file}")
            return 1

    def _before_input_create(inference_engine, _prompt_client, output_sink) -> None:
        if config.input.injection_file and inference_engine is not None:
            injected = _perform_injection(inference_engine, config)
            if injected == 0:
                try:
                    output_sink.close()
                except Exception:
                    pass
                raise _CliInjectionFailed()

    _sync_runtime_builder_patch_surface()
    try:
        runtime = RuntimeSessionBuilder(
            config=config,
            log_dir=args.log_dir,
            before_input_create=_before_input_create,
        ).build_cli()
    except _CliInjectionFailed:
        return 1

    def cleanup() -> None:
        try:
            runtime.cleanup()
        except Exception as exc:
            print(f"Warning: Failed to clean runtime session: {exc}")

    atexit.register(cleanup)

    def signal_handler(sig, frame):
        print("\nShutting down...")
        runtime.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Starting StreamMUSE...")
    print(
        "  Tempo: "
        f"{config.tempo.bpm} BPM, {config.tempo.ticks_per_beat} ticks/beat, "
        f"{config.tempo.beats_per_bar} beats/bar"
    )
    print(f"  Input: {config.input.type}")
    print(f"  Output: {config.output.type}")
    print(
        "  Metronome: "
        f"{'enabled' if config.output.metronome_enabled else 'disabled'}"
        + (f" ({config.output.metronome_port})" if config.output.metronome_port else "")
    )
    print(f"  Count-in: {config.count_in_beats} beat(s)")
    print(f"  Continuation mode: {config.continuation_mode}")
    print(f"  Inference: {config.inference.type}")
    if config.continuation_mode == "prompt_continuation":
        print(f"  Prompt length: {config.inference.prompt_length_ticks} ticks")
    print(f"  Generation interval: {config.inference.generation_interval_ticks} ticks")
    print(f"  Generation length: {config.inference.generation_length_frames} frames")
    print(f"  Logging: {runtime.session_dir}")
    print("\nPress Ctrl+C to stop\n")

    try:
        runtime.start(max_ticks=args.max_ticks)
        while runtime.running:
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()

    return 0


def _sync_runtime_builder_patch_surface() -> None:
    """Preserve CLI-level monkeypatch seams while delegating assembly to runtime."""
    runtime_builder_module.InputSourceFactory = InputSourceFactory
    runtime_builder_module.OutputSinkFactory = OutputSinkFactory
    runtime_builder_module.InferenceEngineFactory = InferenceEngineFactory
    runtime_builder_module.RealTimeMusicService = RealTimeMusicService
    runtime_builder_module.PromptContinuationRealtimeService = PromptContinuationRealtimeService
    runtime_builder_module.PromptContinuationHttpClient = PromptContinuationHttpClient


if __name__ == "__main__":
    sys.exit(main())
