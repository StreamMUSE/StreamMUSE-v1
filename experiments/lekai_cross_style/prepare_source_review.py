#!/usr/bin/env python3
"""Create the manual source-arrangement review sheet for frozen Task 3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--source-gt-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    prepared = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
    rows = []
    for order, item in enumerate(prepared, start=1):
        source_gt = args.source_gt_root / item["style"] / f"{item['piece_id']}__source_gt.mid"
        if not source_gt.is_file():
            raise FileNotFoundError(source_gt)
        rows.append(
            {
                "order": order,
                "assigned_style": item["style"],
                "piece_id": item["piece_id"],
                "title": item["title"],
                "creator": item["creator"],
                "metadata_evidence": item["primary_evidence"],
                "source_gt_midi": str(source_gt.relative_to(args.output_csv.parent)),
                "arrangement_ok_0_no_1_yes": "",
                "melody_accompaniment_tracks_ok_0_no_1_yes": "",
                "comments": "",
            }
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} source review rows: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
