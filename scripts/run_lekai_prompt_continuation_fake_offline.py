from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from streammuse.domain.musical import EventType, MusicalEvent, Note
from streammuse.infrastructure.inference.serialization import event_from_dict, event_to_dict
from streammuse.infrastructure.input.midi_file import MidiFileInput
from streammuse.infrastructure.output.midi_file import MidiFileOutputConfig, MidiFileOutputSink

DEFAULT_RT_MELODY = Path(
    "/data/home/yuanxin/RT-accompanimentV2/user_midi_recording_data/"
    "aligned/periodic_20260409-114946_6217163/dataset_gt_melody.mid"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fake-offline driver for the Lekai prompt-continuation HTTP path. "
            "It reads a real melody MIDI, starts prompt generation from the first prompt window, "
            "keeps appending melody progress, then polls playable accompaniment."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--midi-file-path", type=str, default=str(DEFAULT_RT_MELODY), help="Melody MIDI path")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Server base URL. /generate_accompaniment and /prompt_continuation/... paths are also accepted.",
    )
    parser.add_argument("--output-dir", type=str, default="local_tmp/lekai_prompt_continuation_fake_offline")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.1)
    parser.add_argument("--max-wait-s", type=float, default=60.0)
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--ticks-per-beat", type=int, default=4)
    parser.add_argument("--prompt-length-ticks", type=int, default=32, help="8 beats at 4 ticks/beat")
    parser.add_argument("--generation-interval-ticks", type=int, default=4)
    parser.add_argument("--append-interval-ticks", type=int, default=4)
    parser.add_argument(
        "--append-beats",
        type=int,
        default=3,
        help="If --append-until-tick is unset, append this many beats after the prompt window.",
    )
    parser.add_argument("--append-until-tick", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="lekai_prompt_continuation")
    parser.add_argument("--inference-mode", type=str, default="sliding_window")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--no-clear-history", action="store_true", help="Do not POST /clear_history before starting")
    parser.add_argument("--dry-run", action="store_true", help="Only print split summary; do not call HTTP server")
    return parser.parse_args()


def normalise_base_url(server_url: str) -> str:
    """Return a base URL from either a base or known endpoint URL."""
    raw = str(server_url).rstrip("/")
    parsed = urlsplit(raw)
    known_suffixes = (
        "/generate_accompaniment",
        "/prompt_continuation/start",
        "/prompt_continuation/append_melody",
        "/prompt_continuation/status",
        "/prompt_continuation/playable",
    )
    path = parsed.path.rstrip("/")
    for suffix in known_suffixes:
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))


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
    events.sort(key=lambda e: (int(e.tick), 0 if e.event_type == EventType.NOTE_OFF else 1, int(e.pitch)))
    return events


def load_midi_events(midi_file: Path, *, ticks_per_beat: int) -> tuple[list[MusicalEvent], int, int, int]:
    notes, resolution, max_tick = MidiFileInput._midi_to_notes(
        str(midi_file),
        beat_div=int(ticks_per_beat),
        min_pitch=0,
        max_pitch=127,
        program=None,
        max_tick=None,
    )
    return notes_to_events(notes), int(resolution), int(max_tick), len(notes)


def event_payloads(events: Iterable[MusicalEvent]) -> list[dict[str, Any]]:
    # The server accepts type/pitch/tick for melody; preserving velocity makes logs easier to audit.
    return [event_to_dict(event) for event in events]


def split_prompt_and_append_chunks(
    events: list[MusicalEvent],
    *,
    prompt_length_ticks: int,
    append_until_tick: int,
    append_interval_ticks: int,
) -> tuple[list[MusicalEvent], list[dict[str, Any]]]:
    """Split events into prompt input and observed melody append chunks.

    Empty chunks are kept because observed_until_tick advances through rests.
    """
    if int(prompt_length_ticks) <= 0:
        raise ValueError("prompt_length_ticks must be > 0")
    if int(append_interval_ticks) <= 0:
        raise ValueError("append_interval_ticks must be > 0")
    if int(append_until_tick) < int(prompt_length_ticks):
        raise ValueError("append_until_tick must be >= prompt_length_ticks")

    prompt_events = [event for event in events if int(event.tick) < int(prompt_length_ticks)]
    chunks: list[dict[str, Any]] = []
    start_tick = int(prompt_length_ticks)
    while start_tick < int(append_until_tick):
        end_tick = min(start_tick + int(append_interval_ticks), int(append_until_tick))
        chunk_events = [event for event in events if start_tick <= int(event.tick) < end_tick]
        chunks.append(
            {
                "start_tick": int(start_tick),
                "end_tick": int(end_tick),
                "observed_until_tick": int(end_tick),
                "events": chunk_events,
            }
        )
        start_tick = end_tick
    return prompt_events, chunks


def request_json(method: str, url: str, *, timeout_s: float, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if method == "GET":
        response = requests.get(url, timeout=float(timeout_s))
    elif method == "POST":
        response = requests.post(url, json=payload, timeout=float(timeout_s))
    else:
        raise ValueError(f"unsupported method: {method}")
    response.raise_for_status()
    return response.json()


def wait_for_terminal_status(
    *,
    base_url: str,
    timeout_s: float,
    poll_interval_s: float,
    max_wait_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.monotonic() + float(max_wait_s)
    polls: list[dict[str, Any]] = []
    while True:
        status = request_json(
            "GET",
            f"{base_url}/prompt_continuation/status",
            timeout_s=float(timeout_s),
        )
        polls.append(status)
        if status.get("is_playback_ready") or status.get("is_failed") or not status.get("is_running"):
            return status, polls
        if time.monotonic() >= deadline:
            raise TimeoutError(f"prompt-continuation did not finish within {max_wait_s}s; last_status={status}")
        time.sleep(float(poll_interval_s))


def export_midi(
    output_path: Path,
    *,
    bpm: float,
    ticks_per_beat: int,
    melody_events: list[MusicalEvent],
    accompaniment_events: list[MusicalEvent],
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
    for event in sorted(melody_events, key=lambda e: (int(e.tick), 0 if e.event_type == EventType.NOTE_OFF else 1)):
        sink.output_event(event, source="user")
    for event in sorted(accompaniment_events, key=lambda e: (int(e.tick), 0 if e.event_type == EventType.NOTE_OFF else 1)):
        sink.output_event(event, source="model")
    sink.close()


def main() -> int:
    args = parse_args()
    midi_file = Path(args.midi_file_path).expanduser().resolve()
    if not midi_file.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_file}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = normalise_base_url(args.server_url)
    melody_events, resolution, max_tick, note_count = load_midi_events(
        midi_file,
        ticks_per_beat=int(args.ticks_per_beat),
    )
    append_until_tick = (
        int(args.append_until_tick)
        if args.append_until_tick is not None
        else int(args.prompt_length_ticks) + int(args.append_beats) * int(args.ticks_per_beat)
    )
    append_until_tick = min(max(int(args.prompt_length_ticks), append_until_tick), int(max_tick))

    prompt_events, append_chunks = split_prompt_and_append_chunks(
        melody_events,
        prompt_length_ticks=int(args.prompt_length_ticks),
        append_until_tick=int(append_until_tick),
        append_interval_ticks=int(args.append_interval_ticks),
    )

    split_summary = {
        "midi_file_path": str(midi_file),
        "server_base_url": base_url,
        "midi_resolution": int(resolution),
        "ticks_per_beat": int(args.ticks_per_beat),
        "source_note_count": int(note_count),
        "source_event_count": int(len(melody_events)),
        "source_max_tick": int(max_tick),
        "prompt_length_ticks": int(args.prompt_length_ticks),
        "prompt_event_count": int(len(prompt_events)),
        "append_until_tick": int(append_until_tick),
        "append_interval_ticks": int(args.append_interval_ticks),
        "append_chunk_count": int(len(append_chunks)),
        "append_event_count": int(sum(len(chunk["events"]) for chunk in append_chunks)),
    }

    if args.dry_run:
        print(json.dumps(split_summary, indent=2))
        return 0

    if not args.no_clear_history:
        request_json("POST", f"{base_url}/clear_history", timeout_s=float(args.timeout_s), payload={})

    timeline: list[dict[str, Any]] = []
    start_payload: dict[str, Any] = {
        "melody_notes": event_payloads(prompt_events),
        "prompt_length_ticks": int(args.prompt_length_ticks),
        "generation_interval_ticks": int(args.generation_interval_ticks),
        "observed_until_tick": int(args.prompt_length_ticks),
        "inference_mode": str(args.inference_mode),
        "model_name": str(args.model_name),
    }
    if args.checkpoint_path:
        start_payload["checkpoint_path"] = str(args.checkpoint_path)

    start_status = request_json(
        "POST",
        f"{base_url}/prompt_continuation/start",
        timeout_s=float(args.timeout_s),
        payload=start_payload,
    )
    timeline.append({"event": "start", "status": start_status})

    for chunk in append_chunks:
        append_payload = {
            "melody_notes": event_payloads(chunk["events"]),
            "observed_until_tick": int(chunk["observed_until_tick"]),
        }
        status = request_json(
            "POST",
            f"{base_url}/prompt_continuation/append_melody",
            timeout_s=float(args.timeout_s),
            payload=append_payload,
        )
        timeline.append(
            {
                "event": "append_melody",
                "start_tick": int(chunk["start_tick"]),
                "end_tick": int(chunk["end_tick"]),
                "event_count": int(len(chunk["events"])),
                "status": status,
            }
        )

    final_status, polls = wait_for_terminal_status(
        base_url=base_url,
        timeout_s=float(args.timeout_s),
        poll_interval_s=float(args.poll_interval_s),
        max_wait_s=float(args.max_wait_s),
    )
    playable = request_json(
        "GET",
        f"{base_url}/prompt_continuation/playable",
        timeout_s=float(args.timeout_s),
    )
    accompaniment_events = [event_from_dict(event) for event in playable.get("accompaniment", [])]

    output_midi = output_dir / f"{midi_file.stem}_prompt_continuation_fake_offline.mid"
    export_midi(
        output_midi,
        bpm=float(args.tempo),
        ticks_per_beat=int(args.ticks_per_beat),
        melody_events=[event for event in melody_events if int(event.tick) < int(append_until_tick)],
        accompaniment_events=accompaniment_events,
    )

    summary = {
        **split_summary,
        "final_status": final_status,
        "poll_count": int(len(polls)),
        "timeline": timeline,
        "playable_accompaniment_event_count": int(len(accompaniment_events)),
        "playable_accompaniment_note_on_count": int(
            sum(1 for event in accompaniment_events if event.event_type == EventType.NOTE_ON)
        ),
        "output_midi": str(output_midi),
    }
    summary_path = output_dir / f"{midi_file.stem}_prompt_continuation_fake_offline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[prompt-cont-fake] source={midi_file}")
    print(
        "[prompt-cont-fake] "
        f"phase={final_status.get('phase')} ready={final_status.get('is_playback_ready')} "
        f"continuation_calls={final_status.get('continuation_calls')} "
        f"melody_beats={final_status.get('melody_history_beats')} "
        f"acc_beats={final_status.get('accompaniment_history_beats')}"
    )
    print(f"[prompt-cont-fake] playable_events={len(accompaniment_events)}")
    print(f"[prompt-cont-fake] output_midi={output_midi}")
    print(f"[prompt-cont-fake] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
