#!/usr/bin/env python3
"""Shift source NPZ timelines to match a leading-rest-trimmed MIDI cohort."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import mido
import numpy as np


CHANNEL_COUNT = 4
PIANO_PITCH_COUNT = 88
LOWEST_MIDI_PITCH = 21
MEASURE_STEPS = 16
STEPS_PER_BEAT = 4
TRACK_CHANNELS = {"Melody": (0, 1), "Accompaniment": (2, 3)}
SCHEMA_VERSION = "streammuse.matched_eval.trimmed_npz_cohort.v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MEASURE_KEY_PATTERN = re.compile(r"measure_(\d+)")
AUDIT_FIELDS = (
    "sample_index",
    "piece_id",
    "source_npz",
    "source_npz_sha256",
    "output_npz",
    "output_npz_sha256",
    "leading_trim_beats",
    "cutoff_steps",
    "source_valid_steps",
    "source_serialized_steps",
    "output_valid_steps",
    "output_serialized_steps",
    "output_num_measures",
    "rehydrated_onset_count",
    "midi_npz_equal",
)


@dataclass(frozen=True)
class LoadedNpz:
    metadata: dict[str, Any]
    roll: np.ndarray
    valid_steps: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{context} must be an integer")
    if parsed < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return parsed


def cutoff_steps_from_beats(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a non-negative number")
    try:
        beats = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{context} must be a non-negative number") from exc
    if not beats.is_finite() or beats < 0:
        raise ValueError(f"{context} must be a non-negative finite number")
    steps = beats * STEPS_PER_BEAT
    if steps != steps.to_integral_value():
        raise ValueError(f"{context} * {STEPS_PER_BEAT} must be an exact integer")
    return int(steps)


def _strict_sha256(value: Any, context: str) -> str:
    candidate = str(value or "").strip()
    if not SHA256_PATTERN.fullmatch(candidate):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return candidate


def _resolve_reference(value: Any, manifest_path: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{context} not found: {resolved}")
    return resolved


def _validate_artifact(
    row: Mapping[str, Any],
    path_field: str,
    hash_field: str,
    manifest_path: Path,
    context: str,
) -> Path:
    path = _resolve_reference(row.get(path_field), manifest_path, f"{context}.{path_field}")
    expected = _strict_sha256(row.get(hash_field), f"{context}.{hash_field}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{context}.{path_field} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path


def _validated_artifacts(
    row: Mapping[str, Any], manifest_path: Path, context: str
) -> dict[str, Path]:
    paths = {
        field: _validate_artifact(row, field, f"{field}_sha256", manifest_path, context)
        for field in ("source_npz", "melody_midi", "gt_midi")
    }
    for hash_field in row:
        if not hash_field.endswith("_sha256"):
            continue
        path_field = hash_field[: -len("_sha256")]
        if path_field in row and path_field not in paths:
            paths[path_field] = _validate_artifact(
                row, path_field, hash_field, manifest_path, context
            )
    return paths


def _load_npz(path: Path) -> LoadedNpz:
    try:
        with np.load(path, allow_pickle=True) as archive:
            if "metadata" not in archive.files:
                raise ValueError(f"{path}: metadata is missing")
            try:
                metadata = archive["metadata"].item()
            except (ValueError, AttributeError) as exc:
                raise ValueError(f"{path}: metadata must be a scalar object") from exc
            if not isinstance(metadata, Mapping):
                raise ValueError(f"{path}: metadata must be an object")
            metadata = copy.deepcopy(dict(metadata))
            num_measures = _strict_int(
                metadata.get("num_measures"), f"{path}: metadata.num_measures", minimum=1
            )
            expected_keys = {f"measure_{index}" for index in range(num_measures)}
            actual_keys = {
                key for key in archive.files if MEASURE_KEY_PATTERN.fullmatch(key)
            }
            if actual_keys != expected_keys:
                raise ValueError(
                    f"{path}: measure keys mismatch; expected={sorted(expected_keys)}, "
                    f"actual={sorted(actual_keys)}"
                )
            extras = set(archive.files) - expected_keys - {"metadata"}
            if extras:
                raise ValueError(f"{path}: unexpected NPZ keys: {sorted(extras)}")
            measures = []
            for index in range(num_measures):
                measure = np.asarray(archive[f"measure_{index}"])
                if measure.shape != (
                    CHANNEL_COUNT,
                    PIANO_PITCH_COUNT,
                    MEASURE_STEPS,
                ):
                    raise ValueError(
                        f"{path}: measure_{index} must have shape (4, 88, 16), "
                        f"got {measure.shape}"
                    )
                if measure.dtype.kind not in "biuf" or not np.all(np.isfinite(measure)):
                    raise ValueError(f"{path}: measure_{index} must contain finite numbers")
                if not np.all((measure == 0) | (measure == 1)):
                    raise ValueError(f"{path}: measure_{index} must be binary")
                measures.append(np.array(measure, copy=True))
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(str(path)):
            raise
        raise ValueError(f"cannot load NPZ {path}: {exc}") from exc

    roll = np.concatenate(measures, axis=2)
    valid_steps = _strict_int(
        metadata.get("total_length"), f"{path}: metadata.total_length", minimum=1
    )
    if valid_steps > roll.shape[2]:
        raise ValueError(f"{path}: metadata.total_length exceeds serialized length")
    valid_measure_count = (valid_steps + MEASURE_STEPS - 1) // MEASURE_STEPS
    expected_valid_measures = list(range(valid_measure_count))
    if metadata.get("valid_measures") != expected_valid_measures:
        raise ValueError(
            f"{path}: metadata.valid_measures must equal {expected_valid_measures}"
        )
    if metadata.get("num_channels") not in (None, CHANNEL_COUNT):
        raise ValueError(f"{path}: metadata.num_channels must be 4")
    if metadata.get("resolution") not in (None, MEASURE_STEPS):
        raise ValueError(f"{path}: metadata.resolution must be 16")
    if metadata.get("steps_per_beat") not in (None, STEPS_PER_BEAT):
        raise ValueError(f"{path}: metadata.steps_per_beat must be 4")
    if np.any(roll[:, :, valid_steps:] != 0):
        raise ValueError(f"{path}: serialized tail beyond total_length is not all-zero")
    return LoadedNpz(metadata=metadata, roll=roll, valid_steps=valid_steps)


def _shift_roll(
    loaded: LoadedNpz, cutoff_steps: int
) -> tuple[np.ndarray, int, int]:
    if cutoff_steps >= loaded.valid_steps:
        raise ValueError(
            f"cutoff {cutoff_steps} leaves no valid NPZ timeline "
            f"from {loaded.valid_steps} steps"
        )
    output_valid_steps = loaded.valid_steps - cutoff_steps
    shifted = np.array(
        loaded.roll[:, :, cutoff_steps : loaded.valid_steps], copy=True
    )
    rehydrated = 0
    if cutoff_steps > 0:
        for sustain_channel, onset_channel in ((0, 1), (2, 3)):
            crossing = (
                (loaded.roll[sustain_channel, :, cutoff_steps] != 0)
                & (loaded.roll[sustain_channel, :, cutoff_steps - 1] != 0)
                & (loaded.roll[onset_channel, :, cutoff_steps] == 0)
            )
            rehydrated += int(np.count_nonzero(crossing))
            shifted[onset_channel, crossing, 0] = 1

    serialized_steps = (
        (output_valid_steps + MEASURE_STEPS - 1) // MEASURE_STEPS
    ) * MEASURE_STEPS
    output = np.zeros(
        (CHANNEL_COUNT, PIANO_PITCH_COUNT, serialized_steps), dtype=shifted.dtype
    )
    output[:, :, :output_valid_steps] = shifted
    return output, output_valid_steps, rehydrated


def _track_name(track: mido.MidiTrack, context: str) -> str:
    names = [str(message.name).strip() for message in track if message.type == "track_name"]
    if len(names) > 1:
        raise ValueError(f"{context}: a MIDI track has multiple track names")
    return names[0] if names else ""


def _track_notes(
    track: mido.MidiTrack, *, ticks_per_step: int, context: str
) -> list[tuple[int, int, int]]:
    absolute_tick = 0
    active: dict[tuple[int, int], int] = {}
    notes: list[tuple[int, int, int]] = []
    for message in track:
        absolute_tick += int(message.time)
        if message.type == "note_on" and int(message.velocity) > 0:
            if int(message.channel) == 9:
                raise ValueError(f"{context}: drum-channel notes are not supported")
            key = (int(message.channel), int(message.note))
            if key in active:
                raise ValueError(f"{context}: overlapping same-pitch notes are ambiguous")
            active[key] = absolute_tick
            continue
        if message.type not in {"note_off", "note_on"}:
            continue
        key = (int(message.channel), int(message.note))
        if key not in active:
            raise ValueError(f"{context}: unmatched note-off at tick {absolute_tick}")
        onset_tick = active.pop(key)
        if absolute_tick <= onset_tick:
            raise ValueError(f"{context}: non-positive note duration")
        if onset_tick % ticks_per_step or absolute_tick % ticks_per_step:
            raise ValueError(f"{context}: notes must lie on the four-steps-per-beat grid")
        notes.append(
            (int(message.note), onset_tick // ticks_per_step, absolute_tick // ticks_per_step)
        )
    if active:
        raise ValueError(f"{context}: notes without note-off events")
    return notes


def midi_roll_from_named_gt(path: Path, *, valid_steps: int) -> np.ndarray:
    try:
        midi = mido.MidiFile(path)
    except (OSError, EOFError, ValueError) as exc:
        raise ValueError(f"cannot parse GT MIDI {path}: {exc}") from exc
    ticks_per_beat = int(midi.ticks_per_beat)
    if ticks_per_beat <= 0 or ticks_per_beat % STEPS_PER_BEAT:
        raise ValueError(f"{path}: ticks_per_beat must be positive and divisible by 4")
    ticks_per_step = ticks_per_beat // STEPS_PER_BEAT
    named_tracks: dict[str, mido.MidiTrack] = {}
    for index, track in enumerate(midi.tracks):
        name = _track_name(track, f"{path}: track {index}")
        has_notes = any(message.type in {"note_on", "note_off"} for message in track)
        if name in TRACK_CHANNELS:
            if name in named_tracks:
                raise ValueError(f"{path}: duplicate track named {name}")
            named_tracks[name] = track
        elif has_notes:
            raise ValueError(f"{path}: note events outside Melody/Accompaniment tracks")
    if set(named_tracks) != set(TRACK_CHANNELS):
        raise ValueError(
            f"{path}: expected exactly Melody and Accompaniment tracks, "
            f"got {sorted(named_tracks)}"
        )

    roll = np.zeros((CHANNEL_COUNT, PIANO_PITCH_COUNT, valid_steps), dtype=np.uint8)
    for name, (sustain_channel, onset_channel) in TRACK_CHANNELS.items():
        notes = _track_notes(
            named_tracks[name], ticks_per_step=ticks_per_step, context=f"{path}: {name}"
        )
        for pitch, onset, offset in notes:
            if not LOWEST_MIDI_PITCH <= pitch < LOWEST_MIDI_PITCH + PIANO_PITCH_COUNT:
                raise ValueError(f"{path}: {name} pitch {pitch} is outside MIDI 21..108")
            if onset < 0 or onset >= valid_steps or offset > valid_steps:
                raise ValueError(f"{path}: {name} note exceeds valid duration {valid_steps}")
            pitch_index = pitch - LOWEST_MIDI_PITCH
            if (
                np.any(roll[sustain_channel, pitch_index, onset:offset])
                or roll[onset_channel, pitch_index, onset]
            ):
                raise ValueError(f"{path}: overlapping {name} pitch {pitch} is ambiguous")
            roll[sustain_channel, pitch_index, onset:offset] = 1
            roll[onset_channel, pitch_index, onset] = 1
    return roll


def _validate_midi_equality(output_roll: np.ndarray, midi_roll: np.ndarray, valid_steps: int) -> None:
    effective = (output_roll[:, :, :valid_steps] != 0).astype(np.uint8)
    if not np.array_equal(effective, midi_roll):
        differences = np.argwhere(effective != midi_roll)
        first = tuple(int(value) for value in differences[0])
        raise ValueError(
            f"MIDI/NPZ mismatch at channel,pitch,step={first}; "
            f"{len(differences)} differing cells"
        )
    if np.any(output_roll[:, :, valid_steps:] != 0):
        raise ValueError("output NPZ has nonzero data beyond its valid duration")


def _relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _safe_piece_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not safe:
        raise ValueError(f"piece_id has no safe path representation: {value!r}")
    return safe


def _prepare_output_dir(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_npz(path: Path, metadata: Mapping[str, Any], roll: np.ndarray) -> None:
    payload: dict[str, Any] = {"metadata": dict(metadata)}
    for index, start in enumerate(range(0, roll.shape[2], MEASURE_STEPS)):
        payload[f"measure_{index}"] = roll[:, :, start : start + MEASURE_STEPS]
    np.savez_compressed(path, **payload)


def build_cohort(*, cohort_manifest: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = cohort_manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cohort manifest not found: {manifest_path}")
    source_manifest_hash = file_sha256(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("cohort manifest root must be an object")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("cohort manifest must contain a non-empty samples list")
    preprocessing = payload.get("leading_rest_preprocessing")
    if isinstance(preprocessing, Mapping) and preprocessing.get("source_npz_shifted") is True:
        raise ValueError("source manifest says source_npz is already shifted")
    destination = _prepare_output_dir(output_dir)

    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen_piece_ids: set[str] = set()
    digits = max(2, len(str(len(samples))))
    for sample_index, raw_row in enumerate(samples):
        context = f"samples[{sample_index}]"
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{context} must be an object")
        row = dict(raw_row)
        piece_id = str(row.get("piece_id") or "").strip()
        if not piece_id:
            raise ValueError(f"{context}.piece_id is required")
        if piece_id in seen_piece_ids:
            raise ValueError(f"duplicate piece_id: {piece_id}")
        seen_piece_ids.add(piece_id)
        artifacts = _validated_artifacts(row, manifest_path, context)
        cutoff_steps = cutoff_steps_from_beats(
            row.get("leading_trim_beats"), f"{context}.leading_trim_beats"
        )
        source = _load_npz(artifacts["source_npz"])
        output_roll, output_valid_steps, rehydrated = _shift_roll(
            source, cutoff_steps
        )
        midi_roll = midi_roll_from_named_gt(
            artifacts["gt_midi"], valid_steps=output_valid_steps
        )
        _validate_midi_equality(output_roll, midi_roll, output_valid_steps)

        piece_dir = destination / (
            f"{sample_index + 1:0{digits}d}_{_safe_piece_name(piece_id)}"
        )
        piece_dir.mkdir()
        output_npz = piece_dir / "source.npz"
        output_num_measures = output_roll.shape[2] // MEASURE_STEPS
        output_metadata = copy.deepcopy(source.metadata)
        output_metadata.update(
            {
                "num_measures": output_num_measures,
                "total_length": output_valid_steps,
                "valid_measures": list(range(output_num_measures)),
                "num_channels": CHANNEL_COUNT,
                "resolution": MEASURE_STEPS,
                "steps_per_beat": STEPS_PER_BEAT,
                "trimmed_eval_npz_provenance": {
                    "schema_version": SCHEMA_VERSION,
                    "source_manifest_sha256": source_manifest_hash,
                    "source_sample_index": sample_index,
                    "source_npz_sha256": file_sha256(artifacts["source_npz"]),
                    "leading_trim_beats": row["leading_trim_beats"],
                    "cutoff_steps": cutoff_steps,
                    "source_valid_steps": source.valid_steps,
                    "source_serialized_steps": int(source.roll.shape[2]),
                    "output_valid_steps": output_valid_steps,
                    "output_serialized_steps": int(output_roll.shape[2]),
                    "tail_padding_steps": int(output_roll.shape[2] - output_valid_steps),
                    "rehydrated_onset_count": rehydrated,
                    "gt_midi_sha256": file_sha256(artifacts["gt_midi"]),
                    "midi_npz_equal": True,
                },
            }
        )
        _write_npz(output_npz, output_metadata, output_roll)
        written = _load_npz(output_npz)
        if written.metadata != output_metadata or not np.array_equal(written.roll, output_roll):
            raise RuntimeError(f"written output NPZ failed round-trip validation: {output_npz}")
        _validate_midi_equality(written.roll, midi_roll, written.valid_steps)
        output_hash = file_sha256(output_npz)
        if file_sha256(output_npz) != output_hash:
            raise RuntimeError(f"output NPZ hash changed after writing: {output_npz}")

        output_row = dict(row)
        for field, path in artifacts.items():
            output_row[field] = _relative_path(path, destination)
        output_row.update(
            {
                "source_npz": _relative_path(output_npz, destination),
                "source_npz_sha256": output_hash,
                "num_measures": output_num_measures,
                "num_steps": output_valid_steps,
                "total_length": output_valid_steps,
                "valid_measures": list(range(output_num_measures)),
                "midi_artifacts_shifted_only": False,
                "trimmed_eval_npz_provenance": output_metadata[
                    "trimmed_eval_npz_provenance"
                ],
            }
        )
        output_rows.append(output_row)
        audit_rows.append(
            {
                "sample_index": sample_index,
                "piece_id": piece_id,
                "source_npz": str(artifacts["source_npz"]),
                "source_npz_sha256": file_sha256(artifacts["source_npz"]),
                "output_npz": str(output_npz),
                "output_npz_sha256": output_hash,
                "leading_trim_beats": row["leading_trim_beats"],
                "cutoff_steps": cutoff_steps,
                "source_valid_steps": source.valid_steps,
                "source_serialized_steps": source.roll.shape[2],
                "output_valid_steps": output_valid_steps,
                "output_serialized_steps": output_roll.shape[2],
                "output_num_measures": output_num_measures,
                "rehydrated_onset_count": rehydrated,
                "midi_npz_equal": True,
            }
        )

    output_manifest = copy.deepcopy(dict(payload))
    output_manifest["samples"] = output_rows
    output_manifest["trimmed_eval_npz_cohort"] = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": source_manifest_hash,
        "sample_count": len(output_rows),
        "steps_per_beat": STEPS_PER_BEAT,
        "measure_steps": MEASURE_STEPS,
        "channel_order": ["mel_sus", "mel_on", "acc_sus", "acc_on"],
        "valid_duration_field": "metadata.total_length",
        "tail_padding_policy": "zero_pad_to_16_steps_only_after_valid_duration",
        "midi_validation": "exact_named_tracks_over_valid_duration",
    }
    output_manifest_path = destination / "cohort_manifest.json"
    output_manifest_path.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (destination / "cohort_manifest.json.sha256").write_text(
        file_sha256(output_manifest_path) + "\n", encoding="ascii"
    )
    with (destination / "npz_trim_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AUDIT_FIELDS))
        writer.writeheader()
        writer.writerows(audit_rows)
    return output_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_cohort(
        cohort_manifest=args.cohort_manifest,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "sample_count": len(result["samples"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
