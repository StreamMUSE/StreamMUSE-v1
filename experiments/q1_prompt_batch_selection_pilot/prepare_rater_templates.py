#!/usr/bin/env python
"""Create independent blank rating sheets for an existing blind bundle."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rater-ids", default="rater_01,rater_02,rater_03")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    source_path = run_dir / "blind_ratings.csv"
    with source_path.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise RuntimeError(f"empty rating template: {source_path}")

    ratings_dir = run_dir / "ratings"
    ratings_dir.mkdir(exist_ok=True)
    rater_ids = [item.strip() for item in args.rater_ids.split(",") if item.strip()]
    for rater_index, rater_id in enumerate(rater_ids):
        target = ratings_dir / f"{rater_id}.csv"
        if target.exists() and not args.force:
            raise FileExistsError(f"{target} already exists; use --force to replace it")
        ordered_source_rows = source_rows.copy()
        random.Random(args.seed + rater_index).shuffle(ordered_source_rows)
        rows = [
            {
                "rater_id": rater_id,
                **{
                    field: row[field] if field == "blind_id" else ""
                    for field in row
                },
            }
            for row in ordered_source_rows
        ]
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    guide = """# Blind Rating Guide

- Rate every MIDI in `blind_listening/` independently, following your CSV row order.
- Do not open `blind_mapping_private.csv` before all ratings are complete.
- `overall_score_1_to_5`: overall downstream accompaniment quality.
- `melody_fit_1_to_5`: fit between melody and accompaniment.
- `accompaniment_quality_1_to_5`: accompaniment quality by itself.
- Use `0` only for a complete generation failure.
- Decimal scores are allowed. Prefer exact numbers instead of `3.x` or ranges.
- Record failure mode or repeated-pattern observations in the final two columns.
"""
    (ratings_dir / "RATING_GUIDE.md").write_text(guide, encoding="utf-8")
    print(f"Created {len(rater_ids)} rating sheets in {ratings_dir}")


if __name__ == "__main__":
    main()
