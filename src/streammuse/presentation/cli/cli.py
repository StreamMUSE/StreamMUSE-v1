"""CLI entry point for StreamMUSE."""

from __future__ import annotations

import atexit
import json
import os
import signal
import sys

from streammuse.application.config import ApplicationConfig
from streammuse.application.factories import (
    InferenceEngineFactory,
    InputSourceFactory,
    OutputSinkFactory,
)
from streammuse.domain.interfaces import InferenceEngine
from streammuse.domain.musical import EventType, MusicalEvent, Note
from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.logging import SessionManager
from streammuse.domain.timing import PlaybackScheduler, Tempo
from streammuse.infrastructure.input.midi_file import MidiFileInput
from streammuse.presentation.cli.config_parser import args_to_config, env_to_config, parse_args


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
        if config.input.type != "midi_file":
            print("Error: --injection-file is only supported with --input-mode midi_file")
            return 1
        if config.input.injection_length_ticks <= 0:
            print("Error: --injection-length must be positive")
            return 1
        if not os.path.exists(config.input.injection_file):
            print(f"Error: Injection file not found: {config.input.injection_file}")
            return 1

    session_manager = None
    session_config: dict[str, object] = {}
    if config.output.type != "midi_file":
        session_manager = SessionManager(args.log_dir)
        session_manager.create_session_directory()
        session_config = {
            "tempo_bpm": config.tempo.bpm,
            "ticks_per_beat": config.tempo.ticks_per_beat,
            "beats_per_bar": config.tempo.beats_per_bar,
            "input_type": config.input.type,
            "output_type": config.output.type,
            "metronome_enabled": config.output.metronome_enabled,
            "metronome_port": config.output.metronome_port,
            "metronome_channel": config.output.metronome_channel,
            "inference_type": config.inference.type,
            "generation_interval_ticks": config.inference.generation_interval_ticks,
            "generation_length_frames": config.inference.generation_length_frames,
        }
        session_manager.save_config(session_config)

    output_sink = OutputSinkFactory.create(config, session_manager)
    inference_engine = InferenceEngineFactory.create(config)

    if config.input.injection_file:
        injected = _perform_injection(inference_engine, config)
        if injected == 0:
            try:
                output_sink.close()
            except Exception:
                pass
            return 1

    input_source = InputSourceFactory.create(config)

    if session_manager:
        output_sink.output_config(session_config)

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

    def _save_history_logs(history_payload: object) -> None:
        if session_manager is None:
            return
        if not isinstance(history_payload, dict):
            print("Warning: clear_history returned unexpected payload; skipping history log files")
            return

        melody_history = history_payload.get("melody_history", [])
        accompaniment_history = history_payload.get("accompaniment_history", [])

        session_dir = session_manager.get_session_dir()
        melody_path = session_dir / "melody_history.json"
        accompaniment_path = session_dir / "accompaniment_history.json"

        with open(melody_path, "w") as f:
            json.dump(melody_history if isinstance(melody_history, list) else [], f, indent=2)
        with open(accompaniment_path, "w") as f:
            json.dump(accompaniment_history if isinstance(accompaniment_history, list) else [], f, indent=2)

    def cleanup() -> None:
        try:
            history_payload = inference_engine.clear_history()
            if session_manager:
                _save_history_logs(history_payload)
        except Exception as exc:
            print(f"Warning: Failed to clear inference history: {exc}")

        try:
            output_sink.close()
        except Exception as exc:
            print(f"Warning: Failed to close output sink cleanly: {exc}")

        if session_manager:
            try:
                session_manager.save_summary(
                    {
                        "status": "completed",
                        "session_id": session_manager.get_session_id(),
                    }
                )
            except Exception as exc:
                print(f"Warning: Failed to save session summary: {exc}")

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
    print(
        "  Metronome: "
        f"{'enabled' if config.output.metronome_enabled else 'disabled'}"
        + (f" ({config.output.metronome_port})" if config.output.metronome_port else "")
    )
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
