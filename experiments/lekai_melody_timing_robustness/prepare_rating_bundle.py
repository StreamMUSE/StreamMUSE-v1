#!/usr/bin/env python3
"""Create a shallow, short-named Task 1 listening and rating bundle."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


PIECE_ORDER = ["43038", "3410421", "2581346"]
CONDITION_PATTERN = re.compile(r"^onset_tick1_pct(10|25|50)_seed([0-2])$")


def _condition_name(condition: str) -> tuple[int, str]:
    if condition == "baseline":
        return 0, "base"
    match = CONDITION_PATTERN.fullmatch(condition)
    if not match:
        raise ValueError(f"unexpected condition: {condition}")
    rate = int(match.group(1))
    seed = int(match.group(2))
    offset = {10: 1, 25: 4, 50: 7}[rate]
    return offset + seed, f"p{rate}_s{seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_ratings = args.analysis_dir / "ratings.csv"
    with source_ratings.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    rows_by_piece: dict[str, list[dict]] = {piece_id: [] for piece_id in PIECE_ORDER}
    for row in source_rows:
        piece_id = row["piece_id"]
        if piece_id not in rows_by_piece:
            raise ValueError(f"unexpected piece: {piece_id}")
        condition_order, short_condition = _condition_name(row["condition"])
        row["condition_order"] = condition_order
        row["short_condition"] = short_condition
        rows_by_piece[piece_id].append(row)

    output_rows = []
    for piece_index, piece_id in enumerate(PIECE_ORDER, start=1):
        piece_rows = sorted(rows_by_piece[piece_id], key=lambda row: row["condition_order"])
        if len(piece_rows) != 10:
            raise ValueError(f"expected 10 conditions for {piece_id}, found {len(piece_rows)}")
        piece_dir = args.output_root / f"{piece_index:02d}_{piece_id}"
        piece_dir.mkdir(parents=True, exist_ok=True)
        for row in piece_rows:
            source = args.analysis_dir / row["listening_midi"]
            if not source.is_file():
                raise FileNotFoundError(source)
            filename = f"{row['condition_order']:02d}_{row['short_condition']}.mid"
            destination = piece_dir / filename
            shutil.copy2(source, destination)
            output_rows.append(
                {
                    "order": len(output_rows) + 1,
                    "piece_id": piece_id,
                    "condition": row["short_condition"],
                    "midi": str(destination.relative_to(args.output_root)),
                    "musical_quality_1_low_5_high": row.get(
                        "musical_quality_rating", ""
                    ),
                    "melody_accompaniment_fit_1_low_5_high": row.get(
                        "melody_accompaniment_fit_rating", ""
                    ),
                    "degradation_vs_base_1_none_5_severe": row.get(
                        "noticeable_degradation", ""
                    ),
                    "comments": row.get("comments", ""),
                }
            )

    rating_path = args.output_root / "ratings.csv"
    with rating_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Prepared {len(output_rows)} MIDI files: {args.output_root}")


if __name__ == "__main__":
    main()

