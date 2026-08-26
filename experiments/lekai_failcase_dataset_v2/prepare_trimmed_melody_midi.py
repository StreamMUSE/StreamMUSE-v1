#!/usr/bin/env python3
"""Export selected NPZ Melody tracks to MIDI, then remove leading MIDI ticks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import mido

from scripts.prepare_and_compare_lekai_prompt_alignment import (
    export_npz_melody_midi,
)


MANIFEST_FIELDS = [
    "order",
    "style",
    "id",
    "title",
    "source_npz",
    "source_melody_midi",
    "trimmed_melody_midi",
    "bpm",
    "ticks_per_beat",
    "first_note_tick_original",
    "offset_ticks",
    "source_end_tick",
    "output_end_tick",
    "source_note_count",
    "trimmed_note_count",
    "source_sha256",
    "output_sha256",
    "exact_tick_shift_verified",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_tracks(
    midi: mido.MidiFile,
) -> list[list[tuple[int, mido.Message | mido.MetaMessage]]]:
    tracks: list[list[tuple[int, mido.Message | mido.MetaMessage]]] = []
    for track in midi.tracks:
        absolute_tick = 0
        events: list[tuple[int, mido.Message | mido.MetaMessage]] = []
        for message in track:
            absolute_tick += int(message.time)
            events.append((absolute_tick, message))
        tracks.append(events)
    return tracks


def _is_positive_note_on(message: mido.Message | mido.MetaMessage) -> bool:
    return message.type == "note_on" and int(getattr(message, "velocity", 0)) > 0


def _first_note_tick(
    tracks: list[list[tuple[int, mido.Message | mido.MetaMessage]]],
) -> int:
    ticks = [
        tick
        for track in tracks
        for tick, message in track
        if _is_positive_note_on(message)
    ]
    if not ticks:
        raise ValueError("Source Melody MIDI has no velocity>0 note_on")
    return min(ticks)


def _note_count(
    tracks: list[list[tuple[int, mido.Message | mido.MetaMessage]]],
) -> int:
    return sum(
        1
        for track in tracks
        for _tick, message in track
        if _is_positive_note_on(message)
    )


def _end_tick(
    tracks: list[list[tuple[int, mido.Message | mido.MetaMessage]]],
) -> int:
    return max((track[-1][0] if track else 0 for track in tracks), default=0)


def _first_bpm(
    tracks: list[list[tuple[int, mido.Message | mido.MetaMessage]]],
) -> float:
    tempo_events = sorted(
        (tick, track_index, event_index, int(message.tempo))
        for track_index, track in enumerate(tracks)
        for event_index, (tick, message) in enumerate(track)
        if message.type == "set_tempo"
    )
    if not tempo_events:
        return 120.0
    return float(mido.tempo2bpm(tempo_events[0][3]))


def _message_without_time(
    message: mido.Message | mido.MetaMessage,
) -> mido.Message | mido.MetaMessage:
    return message.copy(time=0)


def verify_exact_tick_shift(
    source_path: Path,
    output_path: Path,
    *,
    cutoff_tick: int,
) -> dict[str, Any]:
    source = mido.MidiFile(str(source_path))
    output = mido.MidiFile(str(output_path))
    if source.type != output.type:
        raise RuntimeError("MIDI type changed during leading trim")
    if int(source.ticks_per_beat) != int(output.ticks_per_beat):
        raise RuntimeError("ticks_per_beat changed during leading trim")

    source_tracks = _absolute_tracks(source)
    output_tracks = _absolute_tracks(output)
    if len(source_tracks) != len(output_tracks):
        raise RuntimeError("MIDI track count changed during leading trim")

    for track_index, (source_track, output_track) in enumerate(
        zip(source_tracks, output_tracks)
    ):
        if len(source_track) != len(output_track):
            raise RuntimeError(
                f"MIDI message count changed in track {track_index} during leading trim"
            )
        for event_index, (
            (source_tick, source_message),
            (output_tick, output_message),
        ) in enumerate(zip(source_track, output_track)):
            if _message_without_time(source_message) != _message_without_time(
                output_message
            ):
                raise RuntimeError(
                    f"MIDI message payload changed at track {track_index}, "
                    f"event {event_index}"
                )
            expected_tick = max(0, int(source_tick) - int(cutoff_tick))
            if int(output_tick) != expected_tick:
                raise RuntimeError(
                    f"MIDI tick shift mismatch at track {track_index}, "
                    f"event {event_index}: expected {expected_tick}, got {output_tick}"
                )

    output_first_note_tick = _first_note_tick(output_tracks)
    if output_first_note_tick != 0:
        raise RuntimeError(
            f"Trimmed Melody first note_on must be at tick 0, got {output_first_note_tick}"
        )
    source_note_count = _note_count(source_tracks)
    output_note_count = _note_count(output_tracks)
    if source_note_count != output_note_count:
        raise RuntimeError("Melody note count changed during leading trim")

    source_end_tick = _end_tick(source_tracks)
    output_end_tick = _end_tick(output_tracks)
    expected_end_tick = max(0, source_end_tick - int(cutoff_tick))
    if output_end_tick != expected_end_tick:
        raise RuntimeError(
            f"MIDI end tick mismatch: expected {expected_end_tick}, got {output_end_tick}"
        )

    return {
        "bpm": _first_bpm(source_tracks),
        "ticks_per_beat": int(source.ticks_per_beat),
        "first_note_tick_original": int(cutoff_tick),
        "offset_ticks": int(cutoff_tick),
        "source_end_tick": source_end_tick,
        "output_end_tick": output_end_tick,
        "source_note_count": source_note_count,
        "trimmed_note_count": output_note_count,
        "exact_tick_shift_verified": True,
    }


def trim_melody_midi_leading_ticks(
    source_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Shift one physical Melody MIDI so its first positive note_on is at tick 0."""

    source = mido.MidiFile(str(source_path))
    source_tracks = _absolute_tracks(source)
    cutoff_tick = _first_note_tick(source_tracks)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if cutoff_tick == 0:
        shutil.copy2(source_path, output_path)
    else:
        output = mido.MidiFile(
            type=source.type,
            ticks_per_beat=source.ticks_per_beat,
            charset=source.charset,
            clip=source.clip,
        )
        for source_track in source_tracks:
            output_track = mido.MidiTrack()
            previous_output_tick = 0
            for source_tick, message in source_track:
                output_tick = max(0, int(source_tick) - cutoff_tick)
                output_track.append(
                    message.copy(time=output_tick - previous_output_tick)
                )
                previous_output_tick = output_tick
            output.tracks.append(output_track)
        output.save(str(output_path))

    return verify_exact_tick_shift(
        source_path,
        output_path,
        cutoff_tick=cutoff_tick,
    )


def load_selection(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Selection line {line_number} is not a JSON object")
            if "id" not in record:
                raise ValueError(f"Selection line {line_number} has no id")
            records.append(record)
    if not records:
        raise ValueError(f"Selection is empty: {path}")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Selection contains duplicate ids")
    return records


def prepare_record(
    record: dict[str, Any],
    *,
    npz_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    piece_id = str(record["id"])
    source_npz = npz_root / f"{piece_id}.npz"
    if not source_npz.is_file():
        raise FileNotFoundError(f"Source NPZ does not exist for {piece_id}: {source_npz}")

    source_midi = output_root / "source_melody_midi" / f"{piece_id}.mid"
    trimmed_midi = output_root / "trimmed_melody_midi" / f"{piece_id}.mid"
    export_npz_melody_midi(source_npz, source_midi)
    if not source_midi.is_file():
        raise RuntimeError(f"Existing exporter did not write source MIDI: {source_midi}")

    trim_report = trim_melody_midi_leading_ticks(source_midi, trimmed_midi)
    return {
        "order": record.get("order", ""),
        "style": record.get("style", ""),
        "id": piece_id,
        "title": record.get("title", ""),
        "source_npz": str(source_npz.resolve()),
        "source_melody_midi": str(source_midi.resolve()),
        "trimmed_melody_midi": str(trimmed_midi.resolve()),
        **trim_report,
        "source_sha256": sha256_file(source_midi),
        "output_sha256": sha256_file(trimmed_midi),
    }


def prepare_selection(
    selection_jsonl: Path,
    *,
    npz_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    records = load_selection(selection_jsonl)
    rows = [
        prepare_record(record, npz_root=npz_root, output_root=output_root)
        for record in records
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-jsonl", type=Path, required=True)
    parser.add_argument("--npz-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = prepare_selection(
        args.selection_jsonl,
        npz_root=args.npz_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "count": len(rows),
                "output_root": str(args.output_root.resolve()),
                "all_exact_tick_shift_verified": all(
                    row["exact_tick_shift_verified"] for row in rows
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
