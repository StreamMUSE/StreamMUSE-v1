#!/usr/bin/env python3
"""Analyze greedy versus stochastic Lekai continuation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean, median, pstdev

import pretty_midi


STEPS_PER_BEAT = 4
BEATS_PER_BAR = 4


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _accompaniment_notes(path: Path):
    midi = pretty_midi.PrettyMIDI(str(path))
    for instrument in midi.instruments:
        if instrument.name.strip().lower() == "accompaniment":
            return midi, instrument.notes
    return midi, []


def _entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _max_run(values: list[tuple]) -> int:
    if not values:
        return 0
    best = current = 1
    for previous, value in zip(values, values[1:]):
        current = current + 1 if value == previous else 1
        best = max(best, current)
    return best


def _max_empty_run(values: list[tuple]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if not value else 0
        best = max(best, current)
    return best


def _max_identical_nonempty_run(values: list[tuple]) -> int:
    best = current = 0
    previous = None
    for value in values:
        if not value:
            previous = None
            current = 0
        elif value == previous:
            current += 1
        else:
            previous = value
            current = 1
        best = max(best, current)
    return best


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _output_features(path: Path, bpm: float, total_steps: int, prefix_beats: int) -> dict:
    midi, notes = _accompaniment_notes(path)
    beat_seconds = 60.0 / bpm
    total_beats = total_steps / STEPS_PER_BEAT
    evaluation_beats = max(0.0, total_beats - prefix_beats)
    selected = [note for note in notes if note.start / beat_seconds >= prefix_beats]
    piece_end_time = midi.get_end_time()
    piece_end_note_count = int(
        sum(abs(note.end - piece_end_time) <= 1e-3 for note in selected)
    )

    note_records = Counter()
    cells: set[tuple[int, int]] = set()
    durations = []
    for note in selected:
        onset = int(round(note.start / beat_seconds * STEPS_PER_BEAT))
        end = max(onset + 1, int(round(note.end / beat_seconds * STEPS_PER_BEAT)))
        duration = end - onset
        note_records[(onset, note.pitch, duration)] += 1
        durations.append(duration / STEPS_PER_BEAT)
        for step in range(onset, end):
            cells.add((step, note.pitch))

    bar_count = int(evaluation_beats // BEATS_PER_BAR)
    bar_signatures = []
    prefix_step = prefix_beats * STEPS_PER_BEAT
    bar_steps = BEATS_PER_BAR * STEPS_PER_BEAT
    for bar_index in range(bar_count):
        start = prefix_step + bar_index * bar_steps
        end = start + bar_steps
        signature = tuple(
            sorted(
                (onset - start, pitch, duration)
                for (onset, pitch, duration), count in note_records.items()
                for _ in range(count)
                if start <= onset < end
            )
        )
        bar_signatures.append(signature)

    adjacent_pairs = max(0, len(bar_signatures) - 1)
    adjacent_repeats = sum(a == b for a, b in zip(bar_signatures, bar_signatures[1:]))
    adjacent_nonempty_pairs = [
        (a, b) for a, b in zip(bar_signatures, bar_signatures[1:]) if a and b
    ]
    pitch_class_counts = Counter(note.pitch % 12 for note in selected)
    nonempty_bars = [signature for signature in bar_signatures if signature]
    return {
        "output_empty": not selected,
        "acc_note_count": len(selected),
        "acc_note_density_per_beat": len(selected) / evaluation_beats if evaluation_beats else 0.0,
        "pitch_class_entropy": _entropy([note.pitch % 12 for note in selected]),
        "dominant_pitch_class_rate": (
            max(pitch_class_counts.values()) / len(selected) if selected else 0.0
        ),
        "mean_duration_beats": mean(durations) if durations else 0.0,
        "median_duration_beats": median(durations) if durations else 0.0,
        "p90_duration_beats": _percentile(durations, 0.9),
        "long_note_rate_gt_8_beats": (
            sum(duration > 8.0 for duration in durations) / len(durations)
            if durations
            else 0.0
        ),
        "piece_end_note_count": piece_end_note_count,
        "piece_end_note_rate": piece_end_note_count / len(selected) if selected else 0.0,
        "evaluated_bar_count": bar_count,
        "empty_bar_rate": (
            sum(not signature for signature in bar_signatures) / bar_count if bar_count else 0.0
        ),
        "unique_bar_ratio": (
            len(set(bar_signatures)) / bar_count if bar_count else 0.0
        ),
        "unique_nonempty_bar_ratio": (
            len(set(nonempty_bars)) / len(nonempty_bars) if nonempty_bars else 0.0
        ),
        "adjacent_exact_bar_repeat_rate": (
            adjacent_repeats / adjacent_pairs if adjacent_pairs else 0.0
        ),
        "adjacent_exact_nonempty_bar_repeat_rate": (
            sum(a == b for a, b in adjacent_nonempty_pairs) / len(adjacent_nonempty_pairs)
            if adjacent_nonempty_pairs
            else 0.0
        ),
        "max_identical_bar_run": _max_run(bar_signatures),
        "max_identical_nonempty_bar_run": _max_identical_nonempty_run(bar_signatures),
        "max_empty_bar_run": _max_empty_run(bar_signatures),
        "_note_records": note_records,
        "_cells": cells,
    }


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _record_jaccard(left: Counter, right: Counter) -> float:
    union = sum((left | right).values())
    return sum((left & right).values()) / union if union else 1.0


def _run_identity(manifest_path: Path) -> tuple[str, int]:
    parts = manifest_path.parts
    if "greedy" in parts:
        return "greedy", 0
    if "sampling" in parts:
        seed_part = next(part for part in parts if part.startswith("seed"))
        return "sampling", int(seed_part.removeprefix("seed"))
    raise ValueError(f"cannot infer run identity from {manifest_path}")


def _summary(rows: list[dict]) -> list[dict]:
    metric_names = [
        "acc_note_density_per_beat",
        "pitch_class_entropy",
        "dominant_pitch_class_rate",
        "mean_duration_beats",
        "median_duration_beats",
        "p90_duration_beats",
        "long_note_rate_gt_8_beats",
        "piece_end_note_rate",
        "empty_bar_rate",
        "unique_bar_ratio",
        "unique_nonempty_bar_ratio",
        "adjacent_exact_bar_repeat_rate",
        "adjacent_exact_nonempty_bar_repeat_rate",
        "max_identical_bar_run",
        "max_identical_nonempty_bar_run",
        "max_empty_bar_run",
    ]
    summaries = []
    for mode in ("greedy", "sampling"):
        selected = [row for row in rows if row["mode"] == mode]
        summary = {
            "mode": mode,
            "sample_count": len(selected),
            "piece_count": len({row["piece_id"] for row in selected}),
            "empty_output_count": sum(row["output_empty"] for row in selected),
        }
        for name in metric_names:
            values = [float(row[name]) for row in selected]
            summary[f"mean_{name}"] = mean(values) if values else 0.0
            summary[f"std_{name}"] = pstdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ignore-prefix-beats", type=int, default=8)
    args = parser.parse_args()

    manifest_paths = sorted(args.inference_root.glob("*/*/*_manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"no inference manifests below {args.inference_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    listening_root = args.output_dir / "listening_by_piece"
    internal_rows = []
    manifests = []
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mode, seed = _run_identity(manifest_path)
        manifests.append(
            {
                "path": str(manifest_path),
                "mode": mode,
                "seed": seed,
                "selection_fingerprint_sha1": manifest["selection_fingerprint_sha1"],
                "generation": manifest["generation"],
            }
        )
        for item in manifest["items"]:
            piece_id = item["file_name"].split("__", 1)[0]
            generated_midi = Path(item["gen_midi_path"])
            if not generated_midi.is_file():
                raise FileNotFoundError(generated_midi)
            features = _output_features(
                generated_midi,
                bpm=float(item["metadata"]["bpm"]),
                total_steps=int(item["metadata"]["total_steps"]),
                prefix_beats=args.ignore_prefix_beats,
            )
            listening_dir = listening_root / piece_id
            listening_dir.mkdir(parents=True, exist_ok=True)
            listening_name = f"{piece_id}__{mode}__seed{seed}.mid"
            shutil.copy2(generated_midi, listening_dir / listening_name)
            internal_rows.append(
                {
                    "piece_id": piece_id,
                    "mode": mode,
                    "seed": seed,
                    "source_midi": str(generated_midi),
                    "listening_midi": str(
                        Path("listening_by_piece") / piece_id / listening_name
                    ),
                    **features,
                }
            )

    fingerprints = {entry["selection_fingerprint_sha1"] for entry in manifests}
    if len(fingerprints) != 1:
        raise RuntimeError(f"dataset selection changed between runs: {sorted(fingerprints)}")

    greedy_by_piece = {
        row["piece_id"]: row for row in internal_rows if row["mode"] == "greedy"
    }
    for row in internal_rows:
        baseline = greedy_by_piece[row["piece_id"]]
        row["note_record_jaccard_to_greedy"] = _record_jaccard(
            row["_note_records"], baseline["_note_records"]
        )
        row["pianoroll_cell_jaccard_to_greedy"] = _jaccard(
            row["_cells"], baseline["_cells"]
        )

    pairwise = []
    for piece_id in sorted(greedy_by_piece):
        sampling_rows = [
            row
            for row in internal_rows
            if row["piece_id"] == piece_id and row["mode"] == "sampling"
        ]
        pairs = list(combinations(sampling_rows, 2))
        pairwise.append(
            {
                "piece_id": piece_id,
                "pair_count": len(pairs),
                "mean_sampling_pair_note_jaccard": mean(
                    _record_jaccard(a["_note_records"], b["_note_records"])
                    for a, b in pairs
                ),
                "mean_sampling_pair_cell_jaccard": mean(
                    _jaccard(a["_cells"], b["_cells"]) for a, b in pairs
                ),
            }
        )

    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in internal_rows
    ]
    rating_rows = [
        {
            "piece_id": row["piece_id"],
            "mode": row["mode"],
            "seed": row["seed"],
            "listening_midi": row["listening_midi"],
            "variety_rating_1_low_5_high": "",
            "musical_quality_rating_1_low_5_high": "",
            "melody_accompaniment_fit_1_low_5_high": "",
            "stability_rating_1_low_5_high": "",
            "comments": "",
        }
        for row in sorted(public_rows, key=lambda value: (value["piece_id"], value["mode"], value["seed"]))
    ]
    summary = {
        "ignore_prefix_beats": args.ignore_prefix_beats,
        "selection_fingerprint_sha1": next(iter(fingerprints)),
        "runs": manifests,
        "overall": _summary(public_rows),
        "sampling_pairwise_diversity": pairwise,
    }
    _write_csv(args.output_dir / "per_output.csv", public_rows)
    _write_csv(args.output_dir / "ratings.csv", rating_rows)
    (args.output_dir / "per_output.json").write_text(
        json.dumps(public_rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
