#!/usr/bin/env python3
"""Remove only the prefix before the first Melody onset from selected NPZ files."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


CHANNEL_COUNT = 4
PIANO_PITCH_COUNT = 88
MEASURE_WIDTH = 16
STEPS_PER_BEAT = 4
MELODY_ONSET_CHANNEL = 1
CHANNEL_ORDER = ["mel_sustain", "mel_onset", "acc_sustain", "acc_onset"]
MEASURE_KEY_PATTERN = re.compile(r"measure_(\d+)")
MANIFEST_FIELDS = [
    "order",
    "style",
    "id",
    "title",
    "source",
    "output",
    "offset_steps",
    "source_total_steps",
    "output_total_steps",
    "source_sha256",
    "output_sha256",
    "exact_suffix_verified",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_metadata(archive: np.lib.npyio.NpzFile, path: Path) -> dict[str, Any]:
    if "metadata" not in archive.files:
        raise ValueError(f"NPZ metadata is missing: {path}")
    try:
        metadata = archive["metadata"].item()
    except ValueError as exc:
        raise ValueError(f"NPZ metadata must be a scalar dict: {path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"NPZ metadata must be a dict: {path}")
    return copy.deepcopy(metadata)


def _ordered_measure_keys(archive: np.lib.npyio.NpzFile, path: Path) -> list[str]:
    indexed_keys: list[tuple[int, str]] = []
    for key in archive.files:
        match = MEASURE_KEY_PATTERN.fullmatch(key)
        if match:
            indexed_keys.append((int(match.group(1)), key))
    indexed_keys.sort()
    if not indexed_keys:
        raise ValueError(f"NPZ contains no measure arrays: {path}")
    indices = [index for index, _ in indexed_keys]
    if indices != list(range(len(indices))):
        raise ValueError(f"NPZ measure keys are not contiguous from measure_0: {path}")
    return [key for _, key in indexed_keys]


def load_npz_roll(
    path: Path,
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    """Load and validate one 4-channel NPZ without changing its timeline."""

    with np.load(path, allow_pickle=True) as archive:
        metadata = _load_metadata(archive, path)
        measure_keys = _ordered_measure_keys(archive, path)
        measures = [np.array(archive[key], copy=True) for key in measure_keys]
        extras = {
            key: np.array(archive[key], copy=True)
            for key in archive.files
            if key != "metadata" and not MEASURE_KEY_PATTERN.fullmatch(key)
        }

    for index, measure in enumerate(measures):
        if measure.ndim != 3 or measure.shape[:2] != (
            CHANNEL_COUNT,
            PIANO_PITCH_COUNT,
        ):
            raise ValueError(
                f"{path} measure_{index} must have shape (4, 88, width), "
                f"got {measure.shape}"
            )
        width = int(measure.shape[2])
        is_last = index == len(measures) - 1
        if not is_last and width != MEASURE_WIDTH:
            raise ValueError(
                f"{path} non-final measure_{index} width must be {MEASURE_WIDTH}, "
                f"got {width}"
            )
        if is_last and not 1 <= width <= MEASURE_WIDTH:
            raise ValueError(
                f"{path} final measure width must be in [1, {MEASURE_WIDTH}], got {width}"
            )

    full_roll = np.concatenate(measures, axis=2)
    return metadata, full_roll, extras


def first_melody_onset_step(full_roll: np.ndarray) -> int:
    onset_steps = np.flatnonzero(np.any(full_roll[MELODY_ONSET_CHANNEL] != 0, axis=0))
    if onset_steps.size == 0:
        raise ValueError("NPZ has no Melody onset in channel 1")
    return int(onset_steps[0])


def split_without_padding(full_roll: np.ndarray) -> list[np.ndarray]:
    """Split a nonempty roll into 16-step measures without adding tail padding."""

    total_steps = int(full_roll.shape[2])
    if total_steps <= 0:
        raise ValueError("Cannot split an empty piano roll")
    return [
        full_roll[:, :, start : start + MEASURE_WIDTH]
        for start in range(0, total_steps, MEASURE_WIDTH)
    ]


def _write_trimmed_npz(
    output_path: Path,
    *,
    metadata: dict[str, Any],
    trimmed_roll: np.ndarray,
    extras: dict[str, np.ndarray],
    offset_steps: int,
    source_total_steps: int,
) -> None:
    measures = split_without_padding(trimmed_roll)
    output_metadata = copy.deepcopy(metadata)
    output_metadata["num_measures"] = len(measures)
    output_metadata["total_length"] = int(trimmed_roll.shape[2])
    output_metadata["leading_trim_provenance"] = {
        "operation": "remove_prefix_before_first_melody_onset",
        "channel_order": CHANNEL_ORDER,
        "melody_onset_channel": MELODY_ONSET_CHANNEL,
        "offset_steps": int(offset_steps),
        "source_total_steps": int(source_total_steps),
        "output_total_steps": int(trimmed_roll.shape[2]),
        "steps_per_beat": STEPS_PER_BEAT,
        "measure_width_steps": MEASURE_WIDTH,
        "exact_source_suffix": True,
    }

    payload: dict[str, Any] = {"metadata": output_metadata}
    payload.update(extras)
    payload.update({f"measure_{index}": measure for index, measure in enumerate(measures)})
    np.savez_compressed(output_path, **payload)


def _metadata_lengths_match(metadata: dict[str, Any], full_roll: np.ndarray) -> bool:
    expected_measures = (int(full_roll.shape[2]) + MEASURE_WIDTH - 1) // MEASURE_WIDTH
    try:
        return (
            int(metadata["num_measures"]) == expected_measures
            and int(metadata["total_length"]) == int(full_roll.shape[2])
        )
    except (KeyError, TypeError, ValueError):
        return False


def resolve_source_path(record: dict[str, Any], npz_root: Path | None) -> Path:
    piece_id = str(record["id"])
    if npz_root is not None:
        return npz_root / f"{piece_id}.npz"

    npz_record = record.get("npz")
    if isinstance(npz_record, dict) and npz_record.get("path"):
        return Path(str(npz_record["path"]))
    if record.get("npz_path"):
        return Path(str(record["npz_path"]))
    raise ValueError(f"Selection record {piece_id} has no NPZ path")


def prepare_record(
    record: dict[str, Any],
    *,
    output_root: Path,
    npz_root: Path | None = None,
) -> dict[str, Any]:
    piece_id = str(record["id"])
    source_path = resolve_source_path(record, npz_root)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source NPZ does not exist for {piece_id}: {source_path}")

    output_dir = output_root / "input_npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{piece_id}.npz"
    if source_path.resolve() == output_path.resolve():
        raise ValueError(f"Source and output paths must differ: {source_path}")

    metadata, source_full_roll, extras = load_npz_roll(source_path)
    offset_steps = first_melody_onset_step(source_full_roll)
    expected_roll = source_full_roll[:, :, offset_steps:]

    if offset_steps == 0 and _metadata_lengths_match(metadata, source_full_roll):
        shutil.copy2(source_path, output_path)
    else:
        _write_trimmed_npz(
            output_path,
            metadata=metadata,
            trimmed_roll=expected_roll,
            extras=extras,
            offset_steps=offset_steps,
            source_total_steps=int(source_full_roll.shape[2]),
        )

    output_metadata, output_full_roll, _ = load_npz_roll(output_path)
    exact_suffix_verified = np.array_equal(output_full_roll, expected_roll)
    if not exact_suffix_verified:
        raise RuntimeError(f"Output is not the exact source suffix for {piece_id}")
    if int(output_metadata.get("num_measures", -1)) != (
        int(output_full_roll.shape[2]) + MEASURE_WIDTH - 1
    ) // MEASURE_WIDTH:
        raise RuntimeError(f"Output metadata.num_measures is incorrect for {piece_id}")
    if int(output_metadata.get("total_length", -1)) != int(output_full_roll.shape[2]):
        raise RuntimeError(f"Output metadata.total_length is incorrect for {piece_id}")

    return {
        "order": record.get("order", ""),
        "style": record.get("style", ""),
        "id": piece_id,
        "title": record.get("title", ""),
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "offset_steps": offset_steps,
        "source_total_steps": int(source_full_roll.shape[2]),
        "output_total_steps": int(output_full_roll.shape[2]),
        "source_sha256": sha256_file(source_path),
        "output_sha256": sha256_file(output_path),
        "exact_suffix_verified": True,
    }


def load_selection(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Selection line {line_number} is not a JSON object")
            if "id" not in record:
                raise ValueError(f"Selection line {line_number} has no id")
            records.append(record)
    if not records:
        raise ValueError(f"Selection is empty: {path}")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Selection contains duplicate ids")
    return records


def prepare_selection(
    selection_jsonl: Path,
    *,
    output_root: Path,
    npz_root: Path | None = None,
) -> list[dict[str, Any]]:
    records = load_selection(selection_jsonl)
    rows = [
        prepare_record(record, output_root=output_root, npz_root=npz_root)
        for record in records
    ]

    manifest_json = output_root / "manifest.json"
    manifest_csv = output_root / "manifest.csv"
    manifest_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--npz-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = prepare_selection(
        args.selection_jsonl,
        output_root=args.output_root,
        npz_root=args.npz_root,
    )
    print(
        json.dumps(
            {
                "count": len(rows),
                "output_root": str(args.output_root.resolve()),
                "all_exact_suffix_verified": all(
                    row["exact_suffix_verified"] for row in rows
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
