#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pretty_midi


def _accompaniment_signature(
    midi_path: Path,
    steps_per_beat: int = 4,
    min_start_step: int = 0,
) -> tuple[Counter, set]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    tempi = midi.get_tempo_changes()[1]
    bpm = float(tempi[0]) if len(tempi) else 120.0
    instruments = [
        instrument
        for instrument in midi.instruments
        if "accompan" in instrument.name.lower()
    ]
    if not instruments and len(midi.instruments) >= 2:
        instruments = [midi.instruments[1]]
    if not instruments:
        instruments = []

    records: Counter[tuple[int, int, int]] = Counter()
    cells: set[tuple[int, int, int]] = set()
    for instrument in instruments:
        for note in instrument.notes:
            start = int(round(float(note.start) * bpm / 60.0 * steps_per_beat))
            end = int(round(float(note.end) * bpm / 60.0 * steps_per_beat))
            end = max(start + 1, end)
            if start < int(min_start_step):
                continue
            pitch = int(note.pitch)
            records[(pitch, start, end)] += 1
            cells.add((1, pitch, start))
            cells.update((0, pitch, step) for step in range(start, end))
    return records, cells


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _beat_comparison(
    baseline_cells: set[tuple[int, int, int]],
    output_cells: set[tuple[int, int, int]],
    *,
    start_beat: int,
    total_steps: int,
    steps_per_beat: int = 4,
) -> tuple[int, int, int | None]:
    total_beats = max(start_beat, (int(total_steps) + steps_per_beat - 1) // steps_per_beat)
    exact_beats = 0
    first_different_beat = None
    for beat in range(int(start_beat), total_beats):
        start = beat * steps_per_beat
        end = start + steps_per_beat
        baseline_beat = {cell for cell in baseline_cells if start <= cell[2] < end}
        output_beat = {cell for cell in output_cells if start <= cell[2] < end}
        if baseline_beat == output_beat:
            exact_beats += 1
        elif first_different_beat is None:
            first_different_beat = beat
    return exact_beats, max(0, total_beats - int(start_beat)), first_different_beat


def _first_melody_changed_step(baseline_npz: Path, perturbed_npz: Path) -> int | None:
    if not baseline_npz.is_file() or not perturbed_npz.is_file():
        return None
    baseline = np.load(baseline_npz, allow_pickle=True)
    perturbed = np.load(perturbed_npz, allow_pickle=True)
    metadata = baseline["metadata"].item()
    offset = 0
    for measure_index in range(int(metadata["num_measures"])):
        key = f"measure_{measure_index}"
        difference = np.any(baseline[key][:2] != perturbed[key][:2], axis=(0, 1))
        changed = np.flatnonzero(difference)
        if len(changed):
            return offset + int(changed[0])
        offset += int(baseline[key].shape[2])
    return None


def analyze(
    prepared_manifest: Path,
    inference_manifest: Path,
    output_dir: Path,
    *,
    ignore_prefix_beats: int = 0,
) -> tuple[list[dict], dict]:
    prepared_rows = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    inference = json.loads(inference_manifest.read_text(encoding="utf-8"))
    item_by_npz = {
        str(item["file_name"]): item
        for item in inference["items"]
    }
    baseline_by_piece = {
        row["piece_id"]: row for row in prepared_rows if row["condition"] == "baseline"
    }
    signature_cache: dict[Path, tuple[Counter, set]] = {}
    min_start_step = int(ignore_prefix_beats) * 4

    def signature(path: Path):
        if path not in signature_cache:
            signature_cache[path] = _accompaniment_signature(
                path,
                min_start_step=min_start_step,
            )
        return signature_cache[path]

    output_dir.mkdir(parents=True, exist_ok=True)
    listening_root = output_dir / "listening_by_piece"
    rows: list[dict] = []
    for prepared in prepared_rows:
        npz_name = Path(prepared["npz"]).name
        inference_item = item_by_npz.get(npz_name)
        generated_midi = Path(inference_item["gen_midi_path"]) if inference_item else None
        if generated_midi is None or not generated_midi.is_file():
            raise FileNotFoundError(f"missing generated MIDI for {npz_name}")
        piece_id = str(prepared["piece_id"])
        baseline_name = Path(baseline_by_piece[piece_id]["npz"]).name
        baseline_item = item_by_npz[baseline_name]
        baseline_midi = Path(baseline_item["gen_midi_path"])
        baseline_records, baseline_cells = signature(baseline_midi)
        output_records, output_cells = signature(generated_midi)
        common_records = sum((baseline_records & output_records).values())
        union_records = sum((baseline_records | output_records).values())
        total_steps = int(inference_item["metadata"]["total_steps"])
        exact_beats, compared_beats, first_different_beat = _beat_comparison(
            baseline_cells,
            output_cells,
            start_beat=int(ignore_prefix_beats),
            total_steps=total_steps,
        )
        first_input_changed_step = _first_melody_changed_step(
            Path(baseline_by_piece[piece_id]["npz"]),
            Path(prepared["npz"]),
        )

        listening_piece = listening_root / piece_id
        listening_piece.mkdir(parents=True, exist_ok=True)
        listening_name = f"{piece_id}__{prepared['condition']}.mid"
        shutil.copy2(generated_midi, listening_piece / listening_name)

        row = {
            **prepared,
            "generated_midi": str(generated_midi),
            "listening_midi": str(Path("listening_by_piece") / piece_id / listening_name),
            "baseline_generated_midi": str(baseline_midi),
            "baseline_acc_note_count": int(sum(baseline_records.values())),
            "output_acc_note_count": int(sum(output_records.values())),
            "output_empty": not output_records,
            "acc_exact_match_to_baseline": output_records == baseline_records,
            "acc_note_record_jaccard": 1.0 if union_records == 0 else common_records / union_records,
            "acc_pianoroll_cell_jaccard": _jaccard(baseline_cells, output_cells),
            "continuation_compared_beats": compared_beats,
            "continuation_exact_beat_count": exact_beats,
            "continuation_exact_beat_rate": 1.0 if compared_beats == 0 else exact_beats / compared_beats,
            "first_different_acc_beat": first_different_beat,
            "first_changed_melody_step": first_input_changed_step,
            "first_changed_melody_beat": (
                None if first_input_changed_step is None else first_input_changed_step // 4
            ),
        }
        rows.append(row)

    perturbed = [row for row in rows if row["condition"] != "baseline"]
    grouped: list[dict] = []
    group_keys = sorted(
        {
            (row["piece_id"], float(row["selection_rate_requested"]))
            for row in perturbed
        }
    )
    for piece_id, selection_rate in group_keys:
        group = [
            row
            for row in perturbed
            if row["piece_id"] == piece_id
            and float(row["selection_rate_requested"]) == selection_rate
        ]
        grouped.append(
            {
                "piece_id": piece_id,
                "selection_rate_requested": selection_rate,
                "sample_count": len(group),
                "exact_match_count": sum(row["acc_exact_match_to_baseline"] for row in group),
                "empty_output_count": sum(row["output_empty"] for row in group),
                "mean_note_record_jaccard": sum(row["acc_note_record_jaccard"] for row in group) / len(group),
                "mean_pianoroll_cell_jaccard": sum(row["acc_pianoroll_cell_jaccard"] for row in group) / len(group),
                "mean_continuation_exact_beat_rate": sum(
                    row["continuation_exact_beat_rate"] for row in group
                ) / len(group),
                "first_changed_melody_beats": [row["first_changed_melody_beat"] for row in group],
                "first_different_acc_beats": [row["first_different_acc_beat"] for row in group],
                "output_acc_note_counts": [row["output_acc_note_count"] for row in group],
            }
        )

    overall_by_rate: list[dict] = []
    for selection_rate in sorted({float(row["selection_rate_requested"]) for row in perturbed}):
        group = [
            row
            for row in perturbed
            if float(row["selection_rate_requested"]) == selection_rate
        ]
        overall_by_rate.append(
            {
                "selection_rate_requested": selection_rate,
                "sample_count": len(group),
                "piece_count": len({row["piece_id"] for row in group}),
                "exact_match_count": sum(row["acc_exact_match_to_baseline"] for row in group),
                "empty_output_count": sum(row["output_empty"] for row in group),
                "mean_note_record_jaccard": sum(
                    row["acc_note_record_jaccard"] for row in group
                ) / len(group),
                "mean_pianoroll_cell_jaccard": sum(
                    row["acc_pianoroll_cell_jaccard"] for row in group
                ) / len(group),
                "mean_continuation_exact_beat_rate": sum(
                    row["continuation_exact_beat_rate"] for row in group
                ) / len(group),
            }
        )

    summary = {
        "prepared_manifest": str(prepared_manifest),
        "inference_manifest": str(inference_manifest),
        "total_outputs": len(rows),
        "perturbed_outputs": len(perturbed),
        "ignore_prefix_beats": int(ignore_prefix_beats),
        "perturbed_exact_match_count": sum(
            row["acc_exact_match_to_baseline"] for row in perturbed
        ),
        "perturbed_empty_output_count": sum(row["output_empty"] for row in perturbed),
        "baseline_acc_note_counts": {
            row["piece_id"]: row["output_acc_note_count"]
            for row in rows
            if row["condition"] == "baseline"
        },
        "groups": grouped,
        "overall_by_selection_rate": overall_by_rate,
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Task 1 offline MIDI outputs")
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ignore-prefix-beats", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = sorted(args.inference_dir.glob("*_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"expected exactly one inference manifest, found {len(manifests)}")
    rows, summary = analyze(
        args.prepared_manifest.resolve(),
        manifests[0].resolve(),
        args.output_dir.resolve(),
        ignore_prefix_beats=int(args.ignore_prefix_beats),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_condition.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "per_condition.csv", rows)
    rating_rows = [
        {
            "piece_id": row["piece_id"],
            "condition": row["condition"],
            "selection_rate_requested": row["selection_rate_requested"],
            "seed": row.get("seed"),
            "listening_midi": row["listening_midi"],
            "melody_accompaniment_fit_rating": "",
            "musical_quality_rating": "",
            "noticeable_degradation": "",
            "comments": "",
        }
        for row in rows
    ]
    _write_csv(args.output_dir / "ratings.csv", rating_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
