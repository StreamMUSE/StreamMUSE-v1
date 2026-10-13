#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mido
import numpy as np

from scripts.midi_to_npz import midi_pair_to_npz
from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter


MODEL_STEPS_PER_BEAT = 4


@dataclass(frozen=True)
class PieceSpec:
    piece_id: str
    melody_file: str
    accompaniment_file: str


DEFAULT_PIECES = (
    PieceSpec("43038", "14-6___43038_mel.mid", "14-6___43038_acc.mid"),
    PieceSpec("3410421", "17-7__3410421_mel.mid", "17-7__3410421_acc.mid"),
    PieceSpec("2581346", "18-7___2581346_mel.mid", "18-7___2581346_acc.mid"),
)


@dataclass
class _EventRef:
    track_index: int
    event_index: int
    absolute_tick: int
    message: mido.Message | mido.MetaMessage
    new_absolute_tick: int


@dataclass
class _NoteRef:
    on: _EventRef
    off: _EventRef
    pitch: int
    channel: int

    @property
    def start_tick(self) -> int:
        return self.on.absolute_tick

    @property
    def end_tick(self) -> int:
        return self.off.absolute_tick


def _read_midi_events(mid: mido.MidiFile) -> tuple[list[list[_EventRef]], list[_NoteRef]]:
    tracks: list[list[_EventRef]] = []
    notes: list[_NoteRef] = []
    for track_index, track in enumerate(mid.tracks):
        absolute_tick = 0
        events: list[_EventRef] = []
        active: dict[tuple[int, int], list[_EventRef]] = {}
        for event_index, message in enumerate(track):
            absolute_tick += int(message.time)
            event = _EventRef(
                track_index=track_index,
                event_index=event_index,
                absolute_tick=absolute_tick,
                message=message.copy(time=0),
                new_absolute_tick=absolute_tick,
            )
            events.append(event)
            if not hasattr(message, "channel") or not hasattr(message, "note"):
                continue
            key = (int(message.channel), int(message.note))
            if message.type == "note_on" and int(message.velocity) > 0:
                active.setdefault(key, []).append(event)
            elif message.type == "note_off" or (
                message.type == "note_on" and int(message.velocity) == 0
            ):
                pending = active.get(key)
                if pending:
                    on = pending.pop(0)
                    notes.append(
                        _NoteRef(
                            on=on,
                            off=event,
                            pitch=key[1],
                            channel=key[0],
                        )
                    )
        tracks.append(events)
    return tracks, notes


def _event_priority(event: _EventRef) -> int:
    message = event.message
    if message.type == "note_off" or (
        message.type == "note_on" and int(getattr(message, "velocity", 0)) == 0
    ):
        return 0
    if message.type == "note_on":
        return 2
    return 1


def _write_midi_events(
    source: mido.MidiFile,
    tracks: list[list[_EventRef]],
    output_midi: Path,
) -> None:
    output = mido.MidiFile(type=source.type, ticks_per_beat=source.ticks_per_beat)
    for events in tracks:
        ordered = sorted(
            events,
            key=lambda event: (
                int(event.new_absolute_tick),
                _event_priority(event),
                int(event.event_index),
            ),
        )
        track = mido.MidiTrack()
        previous_tick = 0
        for event in ordered:
            absolute_tick = int(event.new_absolute_tick)
            track.append(event.message.copy(time=absolute_tick - previous_tick))
            previous_tick = absolute_tick
        output.tracks.append(track)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    output.save(str(output_midi))


def _model_note_counter(notes: Iterable[dict]) -> Counter[tuple[int, int, int, int]]:
    return Counter(
        (
            int(note["pitch"]),
            int(note["tick"]),
            int(note["duration"]),
            int(note.get("program", 0)),
        )
        for note in notes
        if 21 <= int(note["pitch"]) <= 108
    )


def _model_shape_counter(notes: Iterable[dict]) -> Counter[tuple[int, int, int]]:
    return Counter(
        (
            int(note["pitch"]),
            int(note["duration"]),
            int(note.get("program", 0)),
        )
        for note in notes
        if 21 <= int(note["pitch"]) <= 108
    )


def _pianoroll(notes: list[dict], converter: MidiConverter, max_tick: int) -> np.ndarray:
    return converter.notes_to_pianoroll(notes, max_tick=max(1, int(max_tick)))


def perturb_onsets_on_model_grid(
    input_midi: Path,
    output_midi: Path,
    *,
    selection_rate: float,
    shift_steps: int,
    seed: int,
    model_steps_per_beat: int = MODEL_STEPS_PER_BEAT,
) -> dict:
    if not 0.0 < selection_rate <= 1.0:
        raise ValueError("selection_rate must be in (0, 1]")
    if shift_steps <= 0:
        raise ValueError("shift_steps must be positive")
    if model_steps_per_beat <= 0:
        raise ValueError("model_steps_per_beat must be positive")

    mid = mido.MidiFile(str(input_midi))
    source_ticks_per_beat = int(mid.ticks_per_beat)
    tracks, all_notes = _read_midi_events(mid)
    candidates = [
        note
        for note in all_notes
        if 21 <= int(note.pitch) <= 108
        and int(note.channel) != 9
        and int(note.end_tick) > int(note.start_tick)
    ]
    candidates.sort(
        key=lambda note: (
            note.start_tick,
            note.pitch,
            note.end_tick,
            note.on.track_index,
            note.on.event_index,
        )
    )
    if not candidates:
        raise ValueError(f"no model-range Melody notes found in {input_midi}")

    selected_count = max(1, int(round(len(candidates) * float(selection_rate))))
    selected_count = min(selected_count, len(candidates))
    rng = random.Random(int(seed))

    converter = MidiConverter(ticks_per_beat=model_steps_per_beat)
    baseline_pm, _ = converter.load_midi(str(input_midi))
    baseline_notes, baseline_max_tick = converter.midi_to_notes(baseline_pm, filter_drums=True)
    baseline_counter = _model_note_counter(baseline_notes)
    baseline_shape_counter = _model_shape_counter(baseline_notes)

    planned_intervals: dict[tuple[int, int, int], list[tuple[int, int, _NoteRef]]] = {}
    for note in candidates:
        key = (note.on.track_index, note.channel, note.pitch)
        planned_intervals.setdefault(key, []).append((note.start_tick, note.end_tick, note))

    shuffled_candidates = list(candidates)
    rng.shuffle(shuffled_candidates)
    displacement_ms: list[float] = []
    intended_delta_steps: Counter[int] = Counter()
    selected_notes = 0
    for note in shuffled_candidates:
        start_step = int(round(note.start_tick / source_ticks_per_beat * model_steps_per_beat))
        end_step = int(round(note.end_tick / source_ticks_per_beat * model_steps_per_beat))
        duration_steps = max(1, end_step - start_step)
        key = (note.on.track_index, note.channel, note.pitch)
        intervals = planned_intervals[key]
        current_index = next(index for index, item in enumerate(intervals) if item[2] is note)
        allowed: list[tuple[int, int, int]] = []
        for direction in (-1, 1):
            delta_steps = direction * int(shift_steps)
            target_start_step = start_step + delta_steps
            target_end_step = target_start_step + duration_steps
            if target_start_step < 0:
                continue
            target_start_tick = int(round(target_start_step / model_steps_per_beat * source_ticks_per_beat))
            target_end_tick = int(round(target_end_step / model_steps_per_beat * source_ticks_per_beat))
            previous_end = intervals[current_index - 1][1] if current_index > 0 else None
            next_start = intervals[current_index + 1][0] if current_index + 1 < len(intervals) else None
            if previous_end is not None and target_start_tick < previous_end:
                continue
            if next_start is not None and target_end_tick > next_start:
                continue
            allowed.append((delta_steps, target_start_tick, target_end_tick))
        if not allowed:
            continue

        delta_steps, target_start_tick, target_end_tick = rng.choice(allowed)
        note.on.new_absolute_tick = target_start_tick
        note.off.new_absolute_tick = target_end_tick
        intervals[current_index] = (target_start_tick, target_end_tick, note)
        tempo = float(baseline_pm.get_tempo_changes()[1][0])
        displacement_ms.append(
            (target_start_tick - note.start_tick) / source_ticks_per_beat * 60000.0 / tempo
        )
        intended_delta_steps[delta_steps] += 1
        selected_notes += 1
        if selected_notes >= selected_count:
            break

    if selected_notes != selected_count:
        raise RuntimeError(
            f"only {selected_notes}/{selected_count} notes can be shifted without new same-pitch overlap"
        )

    _write_midi_events(mid, tracks, output_midi)

    reloaded, _ = converter.load_midi(str(output_midi))
    perturbed_notes, perturbed_max_tick = converter.midi_to_notes(reloaded, filter_drums=True)
    perturbed_counter = _model_note_counter(perturbed_notes)
    perturbed_shape_counter = _model_shape_counter(perturbed_notes)
    if sum(baseline_counter.values()) != sum(perturbed_counter.values()):
        raise RuntimeError("model-range note count changed after perturbation")
    if baseline_shape_counter != perturbed_shape_counter:
        raise RuntimeError("pitch/program/quantized-duration content changed during onset-only perturbation")

    unchanged_records = sum((baseline_counter & perturbed_counter).values())
    effective_changed_records = sum(baseline_counter.values()) - unchanged_records
    max_tick = max(int(baseline_max_tick), int(perturbed_max_tick), 1)
    baseline_pr = _pianoroll(baseline_notes, converter, max_tick)
    perturbed_pr = _pianoroll(perturbed_notes, converter, max_tick)
    changed_cells = int(np.count_nonzero(baseline_pr != perturbed_pr))
    if effective_changed_records <= 0 or changed_cells <= 0:
        raise RuntimeError("perturbation disappeared after Lekai 4tpb conversion")

    bpm = float(reloaded.get_tempo_changes()[1][0])
    return {
        "input_midi": str(input_midi),
        "output_midi": str(output_midi),
        "seed": int(seed),
        "selection_rate_requested": float(selection_rate),
        "shift_steps": int(shift_steps),
        "model_steps_per_beat": int(model_steps_per_beat),
        "bpm": bpm,
        "model_step_ms_at_initial_bpm": 60000.0 / bpm / model_steps_per_beat,
        "model_note_count": int(sum(baseline_counter.values())),
        "selected_note_count": int(selected_count),
        "selected_note_rate": float(selected_count / len(candidates)),
        "effective_changed_note_records": int(effective_changed_records),
        "effective_changed_note_rate": float(effective_changed_records / sum(baseline_counter.values())),
        "pianoroll_changed_cells": changed_cells,
        "pianoroll_equal": False,
        "quantized_note_shapes_preserved": True,
        "intended_delta_step_counts": {
            str(delta): int(count) for delta, count in sorted(intended_delta_steps.items())
        },
        "actual_displacement_ms_min": min(displacement_ms),
        "actual_displacement_ms_max": max(displacement_ms),
        "actual_displacement_ms_mean_abs": float(np.mean(np.abs(displacement_ms))),
    }


def _parse_number_list(raw: str, cast) -> list:
    values = [cast(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("list argument must not be empty")
    return values


def validate_prepared_npz(rows: list[dict]) -> None:
    baselines = {row["piece_id"]: row for row in rows if row["condition"] == "baseline"}
    for row in rows:
        if row["condition"] == "baseline":
            continue
        baseline = np.load(baselines[row["piece_id"]]["npz"], allow_pickle=True)
        perturbed = np.load(row["npz"], allow_pickle=True)
        baseline_metadata = baseline["metadata"].item()
        perturbed_metadata = perturbed["metadata"].item()
        if baseline_metadata != perturbed_metadata:
            raise RuntimeError(f"NPZ metadata changed for {row['condition']}")

        melody_changed_cells = 0
        accompaniment_changed_cells = 0
        for measure_index in range(int(baseline_metadata["num_measures"])):
            key = f"measure_{measure_index}"
            baseline_measure = baseline[key]
            perturbed_measure = perturbed[key]
            if baseline_measure.shape != perturbed_measure.shape:
                raise RuntimeError(f"NPZ measure shape changed for {row['condition']}:{key}")
            melody_changed_cells += int(
                np.count_nonzero(baseline_measure[:2] != perturbed_measure[:2])
            )
            accompaniment_changed_cells += int(
                np.count_nonzero(baseline_measure[2:] != perturbed_measure[2:])
            )
        if melody_changed_cells <= 0:
            raise RuntimeError(f"NPZ Melody is unchanged for {row['condition']}")
        if accompaniment_changed_cells != 0:
            raise RuntimeError(f"NPZ accompaniment changed for {row['condition']}")
        row["npz_metadata_equal_to_baseline"] = True
        row["npz_melody_changed_cells"] = melody_changed_cells
        row["npz_accompaniment_changed_cells"] = accompaniment_changed_cells


def prepare_dataset(
    input_root: Path,
    output_root: Path,
    *,
    pieces: Iterable[PieceSpec],
    selection_rates: Iterable[float],
    seeds: Iterable[int],
    shift_steps: int,
) -> list[dict]:
    melody_root = input_root / "mel"
    accompaniment_root = input_root / "acc"
    midi_root = output_root / "midi"
    npz_root = output_root / "npz"
    npz_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for piece in pieces:
        melody_path = melody_root / piece.melody_file
        accompaniment_path = accompaniment_root / piece.accompaniment_file
        if not melody_path.is_file():
            raise FileNotFoundError(melody_path)
        if not accompaniment_path.is_file():
            raise FileNotFoundError(accompaniment_path)

        piece_midi_root = midi_root / piece.piece_id
        piece_midi_root.mkdir(parents=True, exist_ok=True)
        baseline_midi = piece_midi_root / f"{piece.piece_id}__baseline.mid"
        gt_acc_midi = piece_midi_root / f"{piece.piece_id}__gt_acc.mid"
        shutil.copy2(melody_path, baseline_midi)
        shutil.copy2(accompaniment_path, gt_acc_midi)
        baseline_npz = npz_root / f"{piece.piece_id}__baseline.npz"
        if not midi_pair_to_npz(str(baseline_midi), str(gt_acc_midi), str(baseline_npz)):
            raise RuntimeError(f"failed to create baseline NPZ for {piece.piece_id}")
        rows.append(
            {
                "piece_id": piece.piece_id,
                "condition": "baseline",
                "seed": None,
                "selection_rate_requested": 0.0,
                "shift_steps": 0,
                "melody_midi": str(baseline_midi),
                "gt_accompaniment_midi": str(gt_acc_midi),
                "npz": str(baseline_npz),
            }
        )

        for selection_rate in selection_rates:
            percentage = int(round(float(selection_rate) * 100))
            for seed in seeds:
                condition = f"onset_tick{shift_steps}_pct{percentage:02d}_seed{int(seed)}"
                perturbed_midi = piece_midi_root / f"{piece.piece_id}__{condition}.mid"
                perturbation = perturb_onsets_on_model_grid(
                    melody_path,
                    perturbed_midi,
                    selection_rate=float(selection_rate),
                    shift_steps=int(shift_steps),
                    seed=int(seed),
                )
                output_npz = npz_root / f"{piece.piece_id}__{condition}.npz"
                if not midi_pair_to_npz(str(perturbed_midi), str(gt_acc_midi), str(output_npz)):
                    raise RuntimeError(f"failed to create perturbed NPZ: {condition}")
                rows.append(
                    {
                        "piece_id": piece.piece_id,
                        "condition": condition,
                        "melody_midi": str(perturbed_midi),
                        "gt_accompaniment_midi": str(gt_acc_midi),
                        "npz": str(output_npz),
                        **perturbation,
                    }
                )

    validate_prepared_npz(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_json = output_root / "manifest.json"
    manifest_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare effective 4tpb Melody timing perturbations")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/home/yuanxin/RT-accompanimentV2/formal_pop909_sanity_20260710/inputs/scored19"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-rates", default="0.10,0.25,0.50")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--shift-steps", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = prepare_dataset(
        args.input_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
        pieces=DEFAULT_PIECES,
        selection_rates=_parse_number_list(args.selection_rates, float),
        seeds=_parse_number_list(args.seeds, int),
        shift_steps=int(args.shift_steps),
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.expanduser().resolve()),
                "sample_count": len(rows),
                "baseline_count": sum(row["condition"] == "baseline" for row in rows),
                "perturbed_count": sum(row["condition"] != "baseline" for row in rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
