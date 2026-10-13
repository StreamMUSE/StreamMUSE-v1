#!/usr/bin/env python
"""Validate structure and fixed-prefix invariants of a Q1 result bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pretty_midi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accompaniment_prefix_events(
    path: Path, prefix_beats: int, *, clip_note_end: bool
) -> list[tuple]:
    midi = pretty_midi.PrettyMIDI(str(path))
    _, tempi = midi.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) else 120.0
    prefix_end = prefix_beats * 60.0 / tempo
    instruments = [
        instrument
        for instrument in midi.instruments
        if instrument.name.strip().lower() == "accompaniment"
    ]
    if not instruments and len(midi.instruments) >= 2:
        instruments = [midi.instruments[1]]
    events = []
    for instrument in instruments:
        for note in instrument.notes:
            if note.start < prefix_end - 1e-7:
                note_end = min(float(note.end), prefix_end) if clip_note_end else float(note.end)
                events.append(
                    (
                        int(note.pitch),
                        round(float(note.start), 6),
                        round(note_end, 6),
                        int(note.velocity),
                    )
                )
    return sorted(events)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    prefix_beats = int(config["prompt_prefix_beats"])
    prompt_rows = read_csv(run_dir / "prompt_manifest.csv")
    continuation_rows = read_csv(run_dir / "continuation_manifest.csv")
    mapping_rows = read_csv(run_dir / "blind_mapping_private.csv")

    expected_prompts = len(config["npz_ids"]) * len(config["prompt_seeds"])
    expected_continuations = expected_prompts * len(config["continuation_seeds"])
    assert len(prompt_rows) == expected_prompts
    assert len(continuation_rows) == expected_continuations
    assert len(mapping_rows) == expected_continuations

    blind_midi = sorted((run_dir / "blind_listening").glob("*.mid"))
    assert len(blind_midi) == expected_continuations
    for row in mapping_rows:
        blind_path = run_dir / "blind_listening" / f"{row['blind_id']}.mid"
        source_path = run_dir / row["output_midi"]
        assert file_sha256(blind_path) == file_sha256(source_path), row["blind_id"]

    prompt_by_key = {
        (row["piece_id"], row["prompt_seed"]): row for row in prompt_rows
    }
    continuations_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in continuation_rows:
        continuations_by_key[(row["piece_id"], row["prompt_seed"])].append(row)

    empty_prompt_count = 0
    checked_groups = 0
    boundary_note_off_variation_groups = 0
    for key, prompt_row in prompt_by_key.items():
        prompt_events = accompaniment_prefix_events(
            run_dir / prompt_row["prompt_midi"],
            prefix_beats,
            clip_note_end=True,
        )
        prompt_events_unclipped = accompaniment_prefix_events(
            run_dir / prompt_row["prompt_midi"],
            prefix_beats,
            clip_note_end=False,
        )
        if not prompt_events:
            empty_prompt_count += 1
        rows = continuations_by_key[key]
        assert len(rows) == len(config["continuation_seeds"]), key
        output_events = [
            accompaniment_prefix_events(
                run_dir / row["output_midi"],
                prefix_beats,
                clip_note_end=True,
            )
            for row in rows
        ]
        output_events_unclipped = [
            accompaniment_prefix_events(
                run_dir / row["output_midi"],
                prefix_beats,
                clip_note_end=False,
            )
            for row in rows
        ]
        assert all(events == prompt_events for events in output_events), key
        if any(events != prompt_events_unclipped for events in output_events_unclipped):
            boundary_note_off_variation_groups += 1
        checked_groups += 1

    rating_sheets = sorted((run_dir / "ratings").glob("*.csv"))
    for path in rating_sheets:
        rows = read_csv(path)
        assert len(rows) == expected_continuations, path
        assert all(not row["overall_score_1_to_5"] for row in rows), path

    report = {
        "valid": True,
        "piece_count": len(config["npz_ids"]),
        "prompt_count": len(prompt_rows),
        "continuation_count": len(continuation_rows),
        "blind_midi_count": len(blind_midi),
        "fixed_prefix_groups_checked": checked_groups,
        "prompt_prefix_beats": prefix_beats,
        "empty_prompt_count": empty_prompt_count,
        "rating_sheet_count": len(rating_sheets),
        "boundary_note_off_variation_groups": boundary_note_off_variation_groups,
        "prefix_comparison_note_ends_clipped_at_boundary": True,
        "blind_files_match_source_sha256": True,
        "all_continuation_prefixes_match_prompt_midi": True,
    }
    report_path = run_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
