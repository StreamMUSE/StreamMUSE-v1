"""CLI entry point for StreamMUSE."""

from __future__ import annotations

import atexit
import os
import signal
import sys

from streammuse.application.config import ApplicationConfig
from streammuse.application.runtime import RuntimeSessionBuilder
from streammuse.domain.interfaces import InferenceEngine
from streammuse.application.services.input_timing import effective_input_snap_forward_fraction
from streammuse.domain.musical import EventType, MusicalEvent, Note
from streammuse.application.rap.realtime import RollingRapController
from streammuse.application.rap.rhythm import LEGACY_PATTERN_TEMPLATE_IDS
from streammuse.domain.rap import RapScenario, ScenarioSegment, ScoreWeights
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.input.midi_file import MidiFileInput
from streammuse.infrastructure.inference.local_chat_client import LocalChatModelClient, LocalChatModelClientConfig
from streammuse.infrastructure.rap.generators import LocalChatCandidateGenerator, PhraseBankGenerator
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES
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


def _build_rap_controller(config: ApplicationConfig, tempo: Tempo) -> RollingRapController | None:
    """Assemble the optional text layer without widening output sink contracts."""
    rap = config.rap
    if not rap.topic:
        return None

    template_id = LEGACY_PATTERN_TEMPLATE_IDS[rap.pattern]
    fallback_line = {
        "boom_bap": "space dreams rise while bright stars cross dark night",
        "straight_8": "deep sea winds move while moon lights guide ships",
        "trap_sparse": "code sparks grow as quick hands shape new sound",
    }[rap.pattern]
    scenario = RapScenario(
        scenario_id="streammuse_cli_compatibility",
        tempo_bpm=tempo.bpm,
        segments=(ScenarioSegment(0, 1, rap.topic, template_id, (fallback_line,)),),
        loop=True,
    )
    analyzer = CmuProsodyAnalyzer()
    fallback_catalog = PrevalidatedFallbackCatalog.build(scenario, BUILTIN_TEMPLATES, analyzer)
    primary = PhraseBankGenerator()
    stop_primary = None
    close_primary = None
    if rap.generator == "local_chat":
        client = LocalChatModelClient(
            LocalChatModelClientConfig(
                base_url=rap.model_url,
                model=rap.model,
                timeout_s=rap.timeout_s,
            )
        )
        primary = LocalChatCandidateGenerator(client)
        stop_primary = client.stop_accepting_and_abort
        close_primary = client.close

    def emit(event) -> None:
        suffix = "*" if event.syllable.stressed else ""
        print(
            f"[RAP B{event.slot.bar + 1} {event.slot.beat + 1}.{event.slot.tick_in_beat + 1}] "
            f"{event.syllable.label}{suffix}",
            flush=True,
        )

    return RollingRapController(
        tempo=tempo,
        scenario=scenario,
        templates=BUILTIN_TEMPLATES,
        fallback_catalog=fallback_catalog,
        analyzer=analyzer,
        weights=ScoreWeights(),
        publisher=None,
        primary_generator=primary,
        candidate_count=rap.candidate_count,
        lookahead_bars=rap.lookahead_bars,
        minimum_score=0.0,
        seed=0,
        emit=emit,
        stop_primary=stop_primary,
        close_primary=close_primary,
    )


def main() -> int:
    """Main CLI entry point."""
    args = parse_args()

    config = env_to_config()
    if config is None:
        config = args_to_config(args)
    else:
        config = args_to_config(args)

    input_snap_forward_fraction = effective_input_snap_forward_fraction(
        config.input.type,
        config.input_snap_forward_fraction,
    )

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

    tempo = Tempo(
        bpm=config.tempo.bpm,
        ticks_per_beat=config.tempo.ticks_per_beat,
        beats_per_bar=config.tempo.beats_per_bar,
    )

    def _before_input_create(inference_engine, _prompt_client, output_sink) -> None:
        if not config.input.injection_file:
            return
        if inference_engine is None:
            output_sink.close()
            raise RuntimeError(
                "--injection-file is not supported with prompt_continuation mode"
            )
        if _perform_injection(inference_engine, config) == 0:
            output_sink.close()
            raise RuntimeError("injection failed")

    try:
        runtime = RuntimeSessionBuilder(
            config=config,
            log_dir=args.log_dir,
            before_input_create=_before_input_create,
            tick_observer_factory=lambda built_tempo: _build_rap_controller(
                config, built_tempo
            ),
        ).build_cli()
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    session_manager = runtime.session_manager
    atexit.register(runtime.cleanup)

    def signal_handler(sig, frame):
        print("\nShutting down...")
        runtime.stop()
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
    print(f"  Count-in: {config.count_in_beats} beat(s)")
    print(f"  Input snap-forward: {input_snap_forward_fraction:.2f} tick fraction")
    print(f"  Inference: {config.inference.type}")
    print(f"  Generation interval: {config.inference.generation_interval_ticks} ticks")
    print(f"  Generation length: {config.inference.generation_length_frames} frames")
    if config.rap.topic:
        print(
            f"  Rap: {config.rap.pattern} ({config.rap.generator}, "
            f"{config.rap.lookahead_bars} bar lookahead)"
        )
    if session_manager:
        print(f"  Logging: {session_manager.get_session_dir()}")
        print(f"  Session artifact tier: {config.output.session_artifact_tier}")
    print("\nPress Ctrl+C to stop\n")

    def _optional_int_arg(name: str) -> int | None:
        value = getattr(args, name, None)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    def _optional_float_arg(name: str) -> float | None:
        value = getattr(args, name, None)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    analysis_end_tick = _optional_int_arg("analysis_end_tick")
    last_input_note_off_tick = _optional_int_arg("last_input_note_off_tick")
    request_cutoff_tick = _optional_int_arg("request_cutoff_tick")
    run_stop_tick = _optional_int_arg("run_stop_tick")
    legacy_max_ticks = _optional_int_arg("max_ticks")
    tail_beats = _optional_int_arg("tail_beats")
    drain_timeout_seconds = _optional_float_arg("drain_timeout_s")

    if run_stop_tick is None:
        run_stop_tick = legacy_max_ticks
    elif legacy_max_ticks is not None and run_stop_tick != legacy_max_ticks:
        raise ValueError("--max-ticks and --run-stop-tick must match when both are supplied")

    if tail_beats is not None:
        if tail_beats < 0:
            raise ValueError("--tail-beats must be >= 0")
        if last_input_note_off_tick is None:
            raise ValueError("--tail-beats requires --last-input-note-off-tick")
        ticks_per_beat = int(tempo.ticks_per_beat)
        rounded_input_end = (
            (last_input_note_off_tick + ticks_per_beat - 1) // ticks_per_beat
        ) * ticks_per_beat
        derived_run_stop = rounded_input_end + tail_beats * ticks_per_beat
        if run_stop_tick is not None and run_stop_tick != derived_run_stop:
            raise ValueError(
                "explicit run-stop does not match ceil(last-input-note-off/beat)+tail"
            )
        run_stop_tick = derived_run_stop

    try:
        runtime.start(
            run_stop_tick=run_stop_tick,
            analysis_end_tick=analysis_end_tick,
            last_input_note_off_tick=last_input_note_off_tick,
            request_cutoff_tick=request_cutoff_tick,
            drain_timeout_seconds=drain_timeout_seconds,
        )
        while runtime.running:
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
