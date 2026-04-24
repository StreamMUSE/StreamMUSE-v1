from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import mido

from streammuse.domain.musical import EventType, MusicalEvent, Note
from streammuse.infrastructure.inference.http_client import HttpInferenceClient, HttpInferenceClientConfig
from streammuse.infrastructure.input.midi_file import MidiFileInput
from streammuse.infrastructure.output.midi_file import MidiFileOutputConfig, MidiFileOutputSink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fake realtime runner: incrementally call HTTP inference without wall-clock playback drops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--midi-file-path", type=str, required=True, help="Melody MIDI path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8001/generate_accompaniment",
        help="Lekai generate endpoint",
    )
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--ticks-per-beat", type=int, default=4)
    parser.add_argument("--generation-interval-ticks", type=int, default=4)
    parser.add_argument("--generation-length-frames", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=None, help="If unset, use melody max_tick + tail")
    parser.add_argument("--tail-ticks", type=int, default=16, help="Extra tail ticks when --max-ticks is unset")
    parser.add_argument("--start-from-first-note", action="store_true", help="Start tick loop from first melody note tick")
    return parser.parse_args()


def notes_to_events(notes: list[dict[str, int]]) -> list[MusicalEvent]:
    events: list[MusicalEvent] = []
    for raw in notes:
        note = Note(
            pitch=int(raw["pitch"]),
            tick=int(raw["tick"]),
            duration=max(1, int(raw["duration"])),
            velocity=64,
            channel=0,
            program=0,
            is_placeholder=False,
        )
        events.extend(note.to_events())
    events.sort(key=lambda e: (e.tick, 0 if e.event_type == EventType.NOTE_OFF else 1))
    return events


def build_tick_index(events: list[MusicalEvent]) -> dict[int, list[MusicalEvent]]:
    by_tick: dict[int, list[MusicalEvent]] = defaultdict(list)
    for event in events:
        by_tick[int(event.tick)].append(event)
    return by_tick


def export_midi(
    output_path: Path,
    *,
    bpm: float,
    ticks_per_beat: int,
    melody_events: list[MusicalEvent],
    model_events: list[MusicalEvent],
) -> None:
    sink = MidiFileOutputSink(
        MidiFileOutputConfig(
            bpm=float(bpm),
            ticks_per_beat=int(ticks_per_beat),
            output_path=str(output_path),
            user_program=0,
            model_program=0,
            user_track_name="Melody",
            model_track_name="Accompaniment",
        )
    )

    for event in sorted(melody_events, key=lambda e: (e.tick, 0 if e.event_type == EventType.NOTE_OFF else 1)):
        sink.output_event(event, source="user")

    for event in sorted(model_events, key=lambda e: (e.tick, 0 if e.event_type == EventType.NOTE_OFF else 1)):
        sink.output_event(event, source="model")

    sink.close()


def main() -> int:
    args = parse_args()

    if int(args.generation_length_frames) % 4 != 0:
        raise ValueError("--generation-length-frames must be a multiple of 4 for lekai")

    midi_file = Path(args.midi_file_path).expanduser().resolve()
    if not midi_file.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_file}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract BPM from MIDI file for accurate model conditioning.
    midi_bpm: Optional[int] = None
    try:
        mid_raw = mido.MidiFile(str(midi_file))
        for track in mid_raw.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    midi_bpm = round(60_000_000 / msg.tempo)
                    break
            if midi_bpm is not None:
                break
    except Exception:
        pass

    notes, _res, max_tick = MidiFileInput._midi_to_notes(
        str(midi_file),
        beat_div=int(args.ticks_per_beat),
        min_pitch=0,
        max_pitch=127,
        program=None,
        max_tick=None,
    )
    melody_events = notes_to_events(notes)
    melody_by_tick = build_tick_index(melody_events)

    if args.max_ticks is None:
        max_ticks = int(max_tick) + int(args.tail_ticks)
    else:
        max_ticks = int(args.max_ticks)

    first_tick = min((n["tick"] for n in notes), default=0)
    start_tick = int(first_tick) if args.start_from_first_note else 0
    if max_ticks <= start_tick:
        raise ValueError(f"max_ticks ({max_ticks}) must be > start tick ({start_tick})")

    client = HttpInferenceClient(
        HttpInferenceClientConfig(
            generate_url=str(args.server_url),
            timeout_s=float(args.timeout_s),
            model_name="lekai",
            inference_mode="sliding_window",
            generation_interval_ticks=int(args.generation_interval_ticks),
            checkpoint_path=None,
            bpm=midi_bpm,
        )
    )
    if midi_bpm is not None:
        print(f"[fake-rt] detected BPM={midi_bpm} from MIDI file")

    try:
        client.clear_history()
    except Exception:
        # Keep going; some environments may not have prior history.
        pass

    model_events: list[MusicalEvent] = []
    pending_new_events: list[MusicalEvent] = []
    request_count = 0
    response_event_total = 0
    response_nonzero_count = 0
    last_generation_tick = -int(args.generation_interval_ticks)

    for tick in range(start_tick, max_ticks):
        tick_events = melody_by_tick.get(tick, [])
        if tick_events:
            pending_new_events.extend(tick_events)

        if tick - last_generation_tick >= int(args.generation_interval_ticks) and pending_new_events:
            generation_start_tick = tick
            # Match runtime behavior: if request is triggered at beat tail, shift start to next beat.
            if (tick % int(args.ticks_per_beat)) == (int(args.ticks_per_beat) - 1):
                generation_start_tick = tick + 1

            accompaniment, _timing = client.generate_accompaniment(
                melody_events=list(pending_new_events),
                generation_start_tick=int(generation_start_tick),
                generation_length_frames=int(args.generation_length_frames),
            )

            request_count += 1
            response_event_total += len(accompaniment)
            if accompaniment:
                response_nonzero_count += 1

            # Match runtime replacement semantics but without dropping late events by wall-clock.
            model_events = [ev for ev in model_events if ev.tick < generation_start_tick]
            model_events.extend(accompaniment)

            pending_new_events = []
            last_generation_tick = tick

    combined_path = output_dir / f"{midi_file.stem}_fake_realtime_combined.mid"
    export_midi(
        combined_path,
        bpm=float(args.tempo),
        ticks_per_beat=int(args.ticks_per_beat),
        melody_events=[ev for ev in melody_events if start_tick <= ev.tick < max_ticks],
        model_events=model_events,
    )

    summary = {
        "midi_file_path": str(midi_file),
        "server_url": str(args.server_url),
        "tempo": float(args.tempo),
        "midi_bpm": midi_bpm,
        "ticks_per_beat": int(args.ticks_per_beat),
        "generation_interval_ticks": int(args.generation_interval_ticks),
        "generation_length_frames": int(args.generation_length_frames),
        "start_tick": int(start_tick),
        "max_ticks": int(max_ticks),
        "source_note_count": len(notes),
        "source_first_tick": int(first_tick),
        "source_max_tick": int(max_tick),
        "request_count": int(request_count),
        "response_event_total": int(response_event_total),
        "response_nonzero_count": int(response_nonzero_count),
        "model_event_count": int(len(model_events)),
        "model_note_on_count": int(sum(1 for e in model_events if e.event_type == EventType.NOTE_ON)),
        "output_midi": str(combined_path),
    }

    summary_path = output_dir / f"{midi_file.stem}_fake_realtime_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[fake-rt] source={midi_file}")
    print(f"[fake-rt] requests={request_count}, nonzero_responses={response_nonzero_count}, response_events={response_event_total}")
    print(f"[fake-rt] model_events={len(model_events)}, model_note_on={summary['model_note_on_count']}")
    print(f"[fake-rt] output_midi={combined_path}")
    print(f"[fake-rt] summary={summary_path}")

    client.clear_history()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())