#!/usr/bin/env python3
"""Materialize and checksum the frozen Task 3 NPZ input set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--npz-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    with args.shortlist.open("r", encoding="utf-8-sig", newline="") as handle:
        shortlist = list(csv.DictReader(handle))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected_pairs = [
        (style, str(piece_id))
        for style, piece_ids in selection.items()
        for piece_id in piece_ids
    ]
    if len(selected_pairs) != len(set(selected_pairs)):
        raise ValueError("duplicate style/piece pair in selection")
    piece_ids = [piece_id for _, piece_id in selected_pairs]
    if len(piece_ids) != len(set(piece_ids)):
        raise ValueError("a piece may not appear in more than one style")

    rows_by_pair = {(row["style"], row["piece_id"]): row for row in shortlist}
    missing = [pair for pair in selected_pairs if pair not in rows_by_pair]
    if missing:
        raise ValueError(f"selection is absent from shortlist: {missing}")

    input_dir = args.output_root / "input_npz"
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for style, piece_id in selected_pairs:
        source = args.npz_root / f"{piece_id}.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = input_dir / f"{style}__{piece_id}.npz"
        shutil.copy2(source, destination)
        candidate = rows_by_pair[(style, piece_id)]
        manifest.append(
            {
                "style": style,
                "piece_id": piece_id,
                "title": candidate["title"],
                "creator": candidate["creator"],
                "primary_evidence": candidate["primary_evidence"],
                "bpm": float(candidate["bpm"]),
                "num_measures": int(candidate["num_measures"]),
                "melody_density_per_beat": float(candidate["melody_density_per_beat"]),
                "accompaniment_density_per_beat": float(
                    candidate["accompaniment_density_per_beat"]
                ),
                "source_npz": str(source.resolve()),
                "prepared_npz": str(destination.resolve()),
                "sha256": _sha256(destination),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_csv(args.output_root / "manifest.csv", manifest)
    (args.output_root / "indices.json").write_text(
        json.dumps(list(range(len(manifest)))), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "piece_count": len(manifest),
                "styles": {style: len(piece_ids) for style, piece_ids in selection.items()},
                "input_dir": str(input_dir),
            }
        )
    )


if __name__ == "__main__":
    main()

