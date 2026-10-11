#!/usr/bin/env python3
"""Create a shallow, short-named Task 2 listening and rating bundle."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


PIECE_ORDER = ["43038", "3410421", "2581346"]


def _condition(mode: str, seed: int) -> tuple[int, str]:
    if mode == "greedy" and seed == 0:
        return 0, "greedy"
    if mode == "sampling" and 0 <= seed <= 4:
        return seed + 1, f"sample{seed}"
    raise ValueError(f"unexpected mode/seed: {mode}/{seed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    with (args.analysis_dir / "ratings.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        source_rows = list(csv.DictReader(handle))

    rows_by_piece: dict[str, list[dict]] = {piece_id: [] for piece_id in PIECE_ORDER}
    for row in source_rows:
        piece_id = row["piece_id"]
        if piece_id not in rows_by_piece:
            raise ValueError(f"unexpected piece: {piece_id}")
        order, short_condition = _condition(row["mode"], int(row["seed"]))
        row["condition_order"] = order
        row["short_condition"] = short_condition
        rows_by_piece[piece_id].append(row)

    output_rows = []
    for piece_index, piece_id in enumerate(PIECE_ORDER, start=1):
        piece_rows = sorted(rows_by_piece[piece_id], key=lambda row: row["condition_order"])
        if len(piece_rows) != 6:
            raise ValueError(f"expected 6 outputs for {piece_id}, found {len(piece_rows)}")
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
                    "decoder": row["mode"],
                    "seed": row["seed"],
                    "midi": str(destination.relative_to(args.output_root)),
                    "variety_1_low_5_high": row.get(
                        "variety_rating_1_low_5_high",
                        row.get("monotony_rating_1_low_5_high", ""),
                    ),
                    "musical_quality_1_low_5_high": row.get(
                        "musical_quality_rating_1_low_5_high", ""
                    ),
                    "melody_accompaniment_fit_1_low_5_high": row.get(
                        "melody_accompaniment_fit_1_low_5_high", ""
                    ),
                    "stability_1_low_5_high": row.get(
                        "stability_rating_1_low_5_high",
                        row.get("instability_rating_1_low_5_high", ""),
                    ),
                    "comments": row.get("comments", ""),
                }
            )

    with (args.output_root / "ratings.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Prepared {len(output_rows)} MIDI files: {args.output_root}")


if __name__ == "__main__":
    main()
