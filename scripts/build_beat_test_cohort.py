#!/usr/bin/env python3
"""Build the checkpoint-aligned BEAT held-out evaluation cohort.

This intentionally reproduces the legacy 192,788-file, seed-42, 90/10 split
used by the current Prompt/Continuation checkpoints. It is not the newer BEAT
v3 169,283-piece 80/10/10 split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import mido
import numpy as np


SCHEMA_VERSION = 1
SPLIT_SEED = 42
TEST_RATIO = 0.10
EXPECTED_TOTAL_FILES = 192_788
EXPECTED_TEST_FILES = 19_278
EXPECTED_TEST_LIST_SHA256 = (
    "f0366d936d815f775d398e4658ab393a55762a6cc176357693a9a1b4e389733e"
)
DEFAULT_COUNT = 40
DEFAULT_SELECTION_SEED = 20_260_901
DEFAULT_DATA_DIR = Path(
    "/data/home/yuanxin/data/allxml_npz_dual_track_optimized_no_underscore"
)
DEFAULT_OUTPUT_DIR = Path("output/metrics/beat_test_cohort_v1")
EXPECTED_FIRST_FIVE = (
    "5509144",
    "6123939",
    "5563373",
    "3810536",
    "5820238",
)
STEPS_PER_BEAT = 4
MIDI_TICKS_PER_BEAT = 480
MIDI_VELOCITY = 80


@dataclass(frozen=True)
class SplitContract:
    total_files: int = EXPECTED_TOTAL_FILES
    test_files: int = EXPECTED_TEST_FILES
    split_seed: int = SPLIT_SEED
    test_ratio: float = TEST_RATIO
    test_list_sha256: str = EXPECTED_TEST_LIST_SHA256


@dataclass(frozen=True, order=True)
class NoteEvent:
    onset_step: int
    pitch: int
    offset_step: int


@dataclass
class CandidateInspection:
    eligible: bool
    reasons: list[str]
    metadata: dict[str, Any] | None
    full_roll: np.ndarray | None
    stats: dict[str, Any]
    error: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _scalar(value.reshape(-1)[0])
        return [_scalar(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalise_cache_entry(entry: Any) -> str:
    try:
        value = os.fspath(entry)
    except TypeError as exc:
        raise ValueError(f"invalid cache data_files entry: {entry!r}") from exc
    if not isinstance(value, str) or not value.endswith(".npz"):
        raise ValueError(f"invalid cache data_files entry: {entry!r}")
    return value.replace("\\", "/")


def load_cache_entries(data_dir: Path) -> tuple[list[str], Path]:
    cache_path = data_dir / ".lengths_cache.pkl"
    if not cache_path.is_file():
        raise FileNotFoundError(f"legacy length cache not found: {cache_path}")
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, dict) or "data_files" not in cache:
        raise ValueError(f"unexpected legacy cache format: {cache_path}")
    entries = [_normalise_cache_entry(item) for item in cache["data_files"]]
    if len(entries) != len(set(entries)):
        raise ValueError("legacy cache data_files contains duplicates")
    return entries, cache_path


def legacy_test_entries(
    data_files: Sequence[str], contract: SplitContract = SplitContract()
) -> tuple[list[str], str]:
    """Reproduce the checkpoint training code's NumPy legacy RNG split."""
    if len(data_files) != contract.total_files:
        raise ValueError(
            f"legacy corpus size mismatch: expected {contract.total_files}, "
            f"got {len(data_files)}"
        )
    rng = np.random.RandomState(contract.split_seed)
    indices = np.arange(len(data_files))
    rng.shuffle(indices)
    test_size = int(len(data_files) * contract.test_ratio)
    if test_size != contract.test_files:
        raise ValueError(
            f"legacy test size mismatch: expected {contract.test_files}, got {test_size}"
        )
    test_entries = [data_files[int(index)] for index in indices[-test_size:]]
    payload = "\n".join(test_entries)
    actual_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if actual_hash != contract.test_list_sha256:
        raise ValueError(
            "legacy test-list SHA256 mismatch: "
            f"expected {contract.test_list_sha256}, got {actual_hash}"
        )
    return test_entries, payload


def resolve_npz(data_dir: Path, entry: str) -> Path:
    raw = Path(entry)
    for candidate in (raw, data_dir / raw, data_dir / raw.name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve cached NPZ entry: {entry}")


def _time_signature_values(value: Any) -> list[Any]:
    scalar = _scalar(value)
    return scalar if isinstance(scalar, list) else [scalar]


def inspect_candidate(path: Path) -> CandidateInspection:
    reasons: list[str] = []
    stats: dict[str, Any] = {
        "channel_count": "",
        "time_signature_idx": "",
        "num_measures": "",
        "measure_widths": "",
        "total_steps": "",
        "mel_onsets_0_32": "",
        "acc_onsets_0_32": "",
        "mel_onsets_32_128": "",
        "acc_onsets_32_128": "",
    }
    try:
        with np.load(path, allow_pickle=True) as data:
            if "metadata" not in data:
                reasons.append("metadata_missing")
                return CandidateInspection(False, reasons, None, None, stats)
            metadata = data["metadata"].item()
            if not isinstance(metadata, dict):
                reasons.append("metadata_not_mapping")
                return CandidateInspection(False, reasons, None, None, stats)

            ts_values = _time_signature_values(metadata.get("time_signature_idx"))
            stats["time_signature_idx"] = json.dumps(ts_values, separators=(",", ":"))
            if not ts_values or any(value != 0 for value in ts_values):
                reasons.append("time_signature_not_all_4_4_idx0")

            try:
                num_measures = int(_scalar(metadata.get("num_measures", 0)))
            except (TypeError, ValueError):
                num_measures = 0
            stats["num_measures"] = num_measures
            if num_measures <= 0:
                reasons.append("num_measures_invalid")
                return CandidateInspection(False, reasons, metadata, None, stats)

            measures: list[np.ndarray] = []
            widths: list[int] = []
            channel_counts: list[int] = []
            pitch_bins: list[int] = []
            for measure_index in range(num_measures):
                key = f"measure_{measure_index}"
                if key not in data:
                    reasons.append(f"measure_missing:{measure_index}")
                    continue
                measure = np.asarray(data[key])
                if measure.ndim != 3:
                    reasons.append(f"measure_rank_not_3:{measure_index}")
                    continue
                channel_counts.append(int(measure.shape[0]))
                pitch_bins.append(int(measure.shape[1]))
                widths.append(int(measure.shape[2]))
                measures.append(measure)

            stats["channel_count"] = json.dumps(sorted(set(channel_counts)))
            stats["measure_widths"] = json.dumps(sorted(set(widths)))
            if len(measures) != num_measures:
                return CandidateInspection(False, reasons, metadata, None, stats)
            structure_valid = True
            if any(count != 4 for count in channel_counts):
                reasons.append("channel_count_not_4")
                structure_valid = False
            if any(width != 16 for width in widths):
                reasons.append("measure_width_not_16")
                structure_valid = False
            if any(count != 88 for count in pitch_bins):
                reasons.append("pitch_bins_not_88")
                structure_valid = False
            metadata_channels = _scalar(metadata.get("num_channels"))
            if metadata_channels is not None and metadata_channels != 4:
                reasons.append("metadata_num_channels_not_4")
            if not structure_valid:
                return CandidateInspection(False, reasons, metadata, None, stats)

            full_roll = np.concatenate(measures, axis=2)
            total_steps = int(full_roll.shape[2])
            stats["total_steps"] = total_steps
            if total_steps < 128:
                reasons.append("total_steps_lt_128")

            windows = {
                "mel_onsets_0_32": full_roll[1, :, 0:32],
                "acc_onsets_0_32": full_roll[3, :, 0:32],
                "mel_onsets_32_128": full_roll[1, :, 32:128],
                "acc_onsets_32_128": full_roll[3, :, 32:128],
            }
            for label, region in windows.items():
                count = int(np.count_nonzero(region))
                stats[label] = count
                if count == 0:
                    reasons.append(f"{label}_empty")
            return CandidateInspection(
                not reasons,
                reasons,
                metadata,
                full_roll if not reasons else None,
                stats,
            )
    except Exception as exc:  # audit malformed candidates rather than hiding them
        return CandidateInspection(
            False,
            [f"npz_load_error:{type(exc).__name__}"],
            None,
            None,
            stats,
            error=str(exc),
        )


def pianoroll_note_events(track_roll: np.ndarray) -> list[NoteEvent]:
    if track_roll.ndim != 3 or track_roll.shape[0] != 2 or track_roll.shape[1] != 88:
        raise ValueError(f"expected (2, 88, T) track roll, got {track_roll.shape}")
    sustain, onset = track_roll
    events: list[NoteEvent] = []
    for pitch_index in range(88):
        for onset_step_raw in np.flatnonzero(onset[pitch_index] > 0):
            onset_step = int(onset_step_raw)
            offset_step = onset_step + 1
            while (
                offset_step < sustain.shape[1]
                and sustain[pitch_index, offset_step] > 0
            ):
                offset_step += 1
            events.append(NoteEvent(onset_step, pitch_index + 21, offset_step))
    return sorted(events)


def melody_input_sha256(events: Sequence[NoteEvent]) -> str:
    payload = {
        "bpm": 120,
        "notes": [
            [event.onset_step, event.offset_step, event.pitch, MIDI_VELOCITY]
            for event in events
        ],
        "schema_version": 1,
        "steps_per_beat": STEPS_PER_BEAT,
        "track_name": "Melody",
    }
    return canonical_sha256(payload)


def _midi_track(name: str, notes: Sequence[NoteEvent], *, include_timing: bool) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    if include_timing:
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
        track.append(
            mido.MetaMessage(
                "time_signature",
                numerator=4,
                denominator=4,
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=0,
            )
        )
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    step_ticks = MIDI_TICKS_PER_BEAT // STEPS_PER_BEAT
    timed: list[tuple[int, int, int, str]] = []
    for note in notes:
        timed.append((note.onset_step * step_ticks, 1, note.pitch, "note_on"))
        timed.append((note.offset_step * step_ticks, 0, note.pitch, "note_off"))
    previous_tick = 0
    for absolute_tick, _priority, pitch, event_type in sorted(timed):
        delta = absolute_tick - previous_tick
        velocity = MIDI_VELOCITY if event_type == "note_on" else 0
        track.append(
            mido.Message(
                event_type, channel=0, note=pitch, velocity=velocity, time=delta
            )
        )
        previous_tick = absolute_tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def write_midi(path: Path, tracks: Sequence[tuple[str, Sequence[NoteEvent]]]) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=MIDI_TICKS_PER_BEAT)
    for index, (name, notes) in enumerate(tracks):
        midi.tracks.append(_midi_track(name, notes, include_timing=index == 0))
    midi.save(path)


AUDIT_FIELDS = [
    "selection_rank",
    "test_position",
    "cache_entry",
    "piece_id",
    "source_path",
    "eligible",
    "selected_order",
    "exclusion_reasons",
    "error",
    "channel_count",
    "time_signature_idx",
    "num_measures",
    "measure_widths",
    "total_steps",
    "mel_onsets_0_32",
    "acc_onsets_0_32",
    "mel_onsets_32_128",
    "acc_onsets_32_128",
]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _generated_readme(
    *, count: int, selection_seed: int, test_hash: str, cache_hash: str
) -> str:
    return f"""# Checkpoint-aligned BEAT test cohort

This cohort uses the legacy 192,788-NPZ corpus and the exact seed-42 90/10
held-out split used by the current Prompt and Continuation checkpoints. It is
not the newer BEAT v3 169,283-piece 80/10/10 split.

## Frozen split and selection

- Source order: `.lengths_cache.pkl[\"data_files\"]`
- Split RNG: NumPy legacy `RandomState`, seed `{SPLIT_SEED}`
- Test partition: final `floor(N * {TEST_RATIO})` shuffled indices
- Test files: `{EXPECTED_TEST_FILES}`
- Test-list SHA256: `{test_hash}` (UTF-8 entries joined by `\\n`, no final newline)
- Length-cache SHA256: `{cache_hash}`
- Selection seed: `{selection_seed}`
- Selected pieces: `{count}`

The selection traverses one seeded permutation of the full test list and keeps
eligible candidates. Changing `--count` therefore preserves the prefix; the
default run is checked against the existing five-piece order.

## Eligibility contract

No genre, complexity, title, composer, or human filtering is applied.

1. Every measure is a `(4, 88, 16)` piano roll.
2. Every `metadata.time_signature_idx` value is `0`, the confirmed 4/4 index.
3. Total length is at least 128 steps at 4 steps/beat.
4. Melody onset channel 1 and accompaniment onset channel 3 are each non-empty
   in both `[0, 32)` and `[32, 128)`.

`candidate_audit.csv` contains every candidate inspected before the cohort was
filled and all exclusion reasons. Each numbered directory contains the exact
copied `source.npz`, `melody_120bpm.mid`, and `gt_120bpm.mid`. MIDI is generated
directly from the piano roll at fixed 120 BPM with tracks named `Melody` and
`Accompaniment`. `canonical_melody_input_sha256` hashes canonical note events,
not MIDI container bytes.
"""


def build_cohort(
    *,
    data_dir: Path,
    output_dir: Path,
    count: int = DEFAULT_COUNT,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    contract: SplitContract = SplitContract(),
    expected_first_ids: Sequence[str] = EXPECTED_FIRST_FIVE,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    _prepare_output_dir(output_dir)

    data_files, cache_path = load_cache_entries(data_dir)
    test_entries, test_payload = legacy_test_entries(data_files, contract)
    (output_dir / "test_split.txt").write_bytes(test_payload.encode("utf-8"))
    (output_dir / "test_split.txt.sha256").write_text(
        f"{contract.test_list_sha256}  test_split.txt\n", encoding="ascii"
    )

    selection_order = np.random.RandomState(selection_seed).permutation(
        len(test_entries)
    )
    selected: list[tuple[str, Path, int, int]] = []
    audit_rows: list[dict[str, Any]] = []
    for selection_rank, test_position_raw in enumerate(selection_order, start=1):
        test_position = int(test_position_raw)
        entry = test_entries[test_position]
        source_path = ""
        piece_id = Path(entry).stem
        try:
            source = resolve_npz(data_dir, entry)
            source_path = str(source)
            inspection = inspect_candidate(source)
        except Exception as exc:
            inspection = CandidateInspection(
                False,
                [f"source_error:{type(exc).__name__}"],
                None,
                None,
                {},
                str(exc),
            )
            source = Path(entry)
        selected_order = ""
        if inspection.eligible:
            selected_order = len(selected) + 1
            selected.append((piece_id, source, selection_rank, test_position))
        audit_rows.append(
            {
                "selection_rank": selection_rank,
                "test_position": test_position,
                "cache_entry": entry,
                "piece_id": piece_id,
                "source_path": source_path,
                "eligible": inspection.eligible,
                "selected_order": selected_order,
                "exclusion_reasons": ";".join(inspection.reasons),
                "error": inspection.error,
                **{field: inspection.stats.get(field, "") for field in AUDIT_FIELDS[9:]},
            }
        )
        if len(selected) == count:
            break

    _write_csv(output_dir / "candidate_audit.csv", audit_rows, AUDIT_FIELDS)
    if len(selected) != count:
        raise RuntimeError(
            f"only {len(selected)} eligible pieces found after {len(audit_rows)} candidates"
        )

    actual_ids = [item[0] for item in selected]
    required_prefix = list(expected_first_ids[: min(len(expected_first_ids), count)])
    if required_prefix and actual_ids[: len(required_prefix)] != required_prefix:
        raise ValueError(
            "selection compatibility mismatch: expected prefix "
            f"{required_prefix}, got {actual_ids[:len(required_prefix)]}"
        )

    cache_hash = sha256_file(cache_path)
    manifest_rows: list[dict[str, Any]] = []
    digits = max(2, len(str(count)))
    for order, (piece_id, source, selection_rank, test_position) in enumerate(
        selected, start=1
    ):
        piece_dir = output_dir / f"{order:0{digits}d}_{piece_id}"
        piece_dir.mkdir()
        source_copy = piece_dir / "source.npz"
        melody_path = piece_dir / "melody_120bpm.mid"
        gt_path = piece_dir / "gt_120bpm.mid"
        shutil.copy2(source, source_copy)
        inspection = inspect_candidate(source_copy)
        if not inspection.eligible or inspection.full_roll is None:
            raise RuntimeError(f"selected source failed re-validation: {source}")
        full_roll = inspection.full_roll
        melody_notes = pianoroll_note_events(full_roll[:2])
        accompaniment_notes = pianoroll_note_events(full_roll[2:])
        write_midi(melody_path, [("Melody", melody_notes)])
        write_midi(
            gt_path,
            [("Melody", melody_notes), ("Accompaniment", accompaniment_notes)],
        )
        source_hash = sha256_file(source)
        copied_hash = sha256_file(source_copy)
        if source_hash != copied_hash:
            raise RuntimeError(f"source copy hash mismatch: {source}")
        manifest_rows.append(
            {
                "order": order,
                "piece_id": piece_id,
                "selection_rank": selection_rank,
                "test_position": test_position,
                "source_npz": str(source_copy.relative_to(output_dir)).replace("\\", "/"),
                "melody_midi": str(melody_path.relative_to(output_dir)).replace("\\", "/"),
                "gt_midi": str(gt_path.relative_to(output_dir)).replace("\\", "/"),
                "num_measures": inspection.stats["num_measures"],
                "num_steps": inspection.stats["total_steps"],
                "melody_note_count": len(melody_notes),
                "accompaniment_note_count": len(accompaniment_notes),
                "source_npz_sha256": copied_hash,
                "melody_midi_sha256": sha256_file(melody_path),
                "gt_midi_sha256": sha256_file(gt_path),
                "canonical_melody_input_sha256": melody_input_sha256(melody_notes),
            }
        )

    split_record = {
        "cache_path": str(cache_path),
        "cache_sha256": cache_hash,
        "corpus_files": len(data_files),
        "split_seed": contract.split_seed,
        "test_ratio": contract.test_ratio,
        "test_files": len(test_entries),
        "test_list_sha256": contract.test_list_sha256,
        "test_list_serialization": "UTF-8 newline-joined cache entries; no final newline",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_contract": "checkpoint-aligned legacy BEAT 192788 NPZ",
        "split": split_record,
        "selection": {
            "count": count,
            "seed": selection_seed,
            "inspected_candidates": len(audit_rows),
            "expected_first_five": list(EXPECTED_FIRST_FIVE),
        },
        "samples": manifest_rows,
    }
    manifest_json = output_dir / "cohort_manifest.json"
    manifest_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_hash = sha256_file(manifest_json)
    (output_dir / "cohort_manifest.json.sha256").write_text(
        f"{manifest_hash}  cohort_manifest.json\n", encoding="ascii"
    )
    _write_csv(
        output_dir / "cohort_manifest.csv",
        manifest_rows,
        list(manifest_rows[0]),
    )
    (output_dir / "README.md").write_text(
        _generated_readme(
            count=count,
            selection_seed=selection_seed,
            test_hash=contract.test_list_sha256,
            cache_hash=cache_hash,
        ),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_cohort(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        count=args.count,
        selection_seed=args.selection_seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "selected": len(manifest["samples"]),
                "test_list_sha256": manifest["split"]["test_list_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
