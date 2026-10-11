#!/usr/bin/env python3
"""Copy the frozen clean NPZ inputs used by the Task 2 decoding comparison."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--piece-ids", default="2581346,3410421,43038")
    args = parser.parse_args()

    piece_ids = [value.strip() for value in args.piece_ids.split(",") if value.strip()]
    if not piece_ids:
        raise ValueError("piece-ids must not be empty")

    input_dir = args.output_root / "input_npz"
    input_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for piece_id in piece_ids:
        source = args.source_root / f"{piece_id}__baseline.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = input_dir / source.name
        shutil.copy2(source, destination)
        records.append(
            {
                "piece_id": piece_id,
                "file_name": destination.name,
                "source": str(source.resolve()),
                "sha256": _sha256(destination),
            }
        )

    (args.output_root / "input_manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (args.output_root / "indices.json").write_text(
        json.dumps(list(range(len(records)))), encoding="utf-8"
    )
    print(json.dumps({"piece_count": len(records), "input_dir": str(input_dir)}))


if __name__ == "__main__":
    main()

