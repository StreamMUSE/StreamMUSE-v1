"""Convert paired melody/accompaniment MIDI files to NPZ format for PianoLLaMA.

Usage:
    uv run python scripts/midi_to_npz.py \
        --mel-dir prompts/inputs_lekai/mel \
        --acc-dir prompts/inputs_lekai/acc \
        --output-dir prompts/inputs_lekai/npz

NPZ format (expected by PianoDataset):
    metadata: dict with time_signature_idx, bpm, num_measures, is_continuation
    measure_0, measure_1, ...: np.ndarray of shape (4, 88, timesteps_per_measure)
        channels [0:2] = melody (sustain, onset)
        channels [2:4] = accompaniment (sustain, onset)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter


# Time signature (numerator, denominator) → index used by PianoDataset.
# 2/2 is stored as 9; at load time PianoDataset maps 9 → 4.
TIME_SIG_MAP = {
    (2, 4): 2,
    (3, 4): 3,
    (4, 4): 4,
    (6, 8): 6,
    (2, 2): 9,
}

TICKS_PER_BEAT = 4  # Must match model's tokenization resolution
SUMMARY_SCHEMA_VERSION = "streammuse.midi_to_npz_summary.v1"


def midi_pair_to_npz(
    mel_path: str,
    acc_path: str,
    output_path: str,
    ticks_per_beat: int = TICKS_PER_BEAT,
    trim_leading_rest: bool = False,
) -> bool:
    """Convert one (melody, accompaniment) MIDI pair to a single NPZ file."""
    converter = MidiConverter(ticks_per_beat=ticks_per_beat)

    # --- Load melody ---
    mel_pm, mel_meta = converter.load_midi(mel_path)
    if mel_pm is None:
        print(f"  [SKIP] Cannot load melody: {mel_path}")
        return False

    mel_notes, mel_max_tick = converter.midi_to_notes(mel_pm)
    if not mel_notes:
        print(f"  [SKIP] Melody has no notes: {mel_path}")
        return False

    # --- Load accompaniment ---
    acc_pm, acc_meta = converter.load_midi(acc_path)
    if acc_pm is None:
        print(f"  [SKIP] Cannot load accompaniment: {acc_path}")
        return False

    acc_notes, acc_max_tick = converter.midi_to_notes(acc_pm)

    leading_offset = min((int(note["tick"]) for note in mel_notes), default=0)
    if trim_leading_rest and leading_offset > 0:
        def shift_notes(notes: list[dict]) -> list[dict]:
            shifted = []
            for note in notes:
                original_start = int(note["tick"])
                original_end = original_start + int(note["duration"])
                if original_end <= leading_offset:
                    continue
                copied = dict(note)
                copied["tick"] = max(0, original_start - leading_offset)
                copied["duration"] = max(
                    1,
                    original_end - leading_offset - int(copied["tick"]),
                )
                shifted.append(copied)
            return shifted

        mel_notes = shift_notes(mel_notes)
        acc_notes = shift_notes(acc_notes)
        mel_max_tick = max(
            (int(note["tick"]) + int(note["duration"]) for note in mel_notes),
            default=0,
        )
        acc_max_tick = max(
            (int(note["tick"]) + int(note["duration"]) for note in acc_notes),
            default=0,
        )

    # --- Determine metadata ---
    bpm = mel_meta["bpm"]
    time_sig = mel_meta["time_sig"]  # (numerator, denominator)

    time_sig_idx = TIME_SIG_MAP.get(time_sig)
    if time_sig_idx is None:
        print(f"  [WARN] Unknown time signature {time_sig}, defaulting to 4/4")
        time_sig_idx = 4
        time_sig = (4, 4)

    beats_per_measure = time_sig[0]
    ticks_per_measure = beats_per_measure * ticks_per_beat

    # --- Build pianorolls ---
    max_tick = max(mel_max_tick, acc_max_tick)
    # Pad to full measure boundary
    if max_tick % ticks_per_measure != 0:
        max_tick = ((max_tick // ticks_per_measure) + 1) * ticks_per_measure

    mel_pr = converter.notes_to_pianoroll(mel_notes, max_tick=max_tick)  # (2, 88, T)
    acc_pr = converter.notes_to_pianoroll(acc_notes, max_tick=max_tick)  # (2, 88, T)

    num_measures = max_tick // ticks_per_measure

    # --- Build save dict ---
    save_dict: dict = {
        "metadata": {
            "time_signature_idx": time_sig_idx,
            "bpm": int(round(bpm)),
            "num_measures": num_measures,
            "is_continuation": False,
            "trim_leading_rest": bool(trim_leading_rest),
            "leading_offset_ticks": int(leading_offset if trim_leading_rest else 0),
        },
    }

    for i in range(num_measures):
        start = i * ticks_per_measure
        end = start + ticks_per_measure

        # (4, 88, ticks_per_measure): [mel_sustain, mel_onset, acc_sustain, acc_onset]
        measure = np.zeros((4, 88, ticks_per_measure), dtype=np.float32)
        measure[0] = mel_pr[0, :, start:end]  # melody sustain
        measure[1] = mel_pr[1, :, start:end]  # melody onset
        measure[2] = acc_pr[0, :, start:end]  # acc sustain
        measure[3] = acc_pr[1, :, start:end]  # acc onset

        save_dict[f"measure_{i}"] = measure

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **save_dict)
    return True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def _write_canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))


def load_npz_melody_roll(npz_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Concatenate all NPZ measures and return melody channels (2, 88, T)."""
    with np.load(npz_path, allow_pickle=True) as archive:
        raw_metadata = archive["metadata"]
        metadata = raw_metadata.item() if isinstance(raw_metadata, np.ndarray) else raw_metadata
        num_measures = int(metadata["num_measures"])
        expected_keys = {f"measure_{index}" for index in range(num_measures)}
        actual_keys = {key for key in archive.files if key.startswith("measure_")}
        if actual_keys != expected_keys:
            raise ValueError(
                f"{npz_path}: measure keys mismatch; "
                f"expected={sorted(expected_keys)}, actual={sorted(actual_keys)}"
            )
        measures = []
        for index in range(num_measures):
            measure = np.asarray(archive[f"measure_{index}"])
            if measure.ndim != 3 or measure.shape[:2] != (4, 88):
                raise ValueError(f"{npz_path}: invalid measure_{index} shape {measure.shape}")
            measures.append(measure[0:2])
    if not measures:
        return np.zeros((2, 88, 0), dtype=np.uint8), dict(metadata)
    return (np.concatenate(measures, axis=2) > 0).astype(np.uint8), dict(metadata)


def verify_midi_npz_roll(
    midi_path: str | Path,
    npz_path: str | Path,
    *,
    ticks_per_beat: int = TICKS_PER_BEAT,
    expected_horizon: int | None = None,
) -> dict[str, Any]:
    """Static Phase-0 gate: MIDI melody must equal NPZ channels 0:2."""
    converter = MidiConverter(ticks_per_beat=ticks_per_beat)
    pm, _metadata = converter.load_midi(str(midi_path))
    if pm is None:
        raise ValueError(f"cannot load MIDI for roll verification: {midi_path}")
    notes, max_note_tick = converter.midi_to_notes(pm)
    npz_roll, npz_metadata = load_npz_melody_roll(npz_path)
    horizon = int(npz_roll.shape[2])
    if expected_horizon is not None and horizon != int(expected_horizon):
        raise AssertionError(
            f"validation horizon mismatch for {midi_path}: "
            f"expected {expected_horizon}, got {horizon}"
        )
    if max_note_tick > horizon:
        raise AssertionError(
            f"MIDI note end {max_note_tick} exceeds NPZ validation horizon {horizon}"
        )
    midi_roll = converter.notes_to_pianoroll(notes, max_tick=horizon).astype(np.uint8)
    if midi_roll.shape != npz_roll.shape:
        raise AssertionError(f"roll shape mismatch: MIDI {midi_roll.shape}, NPZ {npz_roll.shape}")
    diff = np.argwhere(midi_roll != npz_roll)
    if diff.size:
        first = tuple(int(value) for value in diff[0])
        raise AssertionError(
            f"MIDI/NPZ roll mismatch at {first}; {len(diff)} differing cells: {midi_path}"
        )
    return {
        "differing_cells": 0,
        "horizon_ticks": horizon,
        "midi_max_note_tick": int(max_note_tick),
        "npz_metadata": npz_metadata,
        "shape": list(midi_roll.shape),
    }


def _resolve_resource(manifest_path: Path, resource: dict[str, Any]) -> Path:
    raw_path = resource.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"invalid manifest resource: {resource!r}")
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError(f"manifest resource must be relative: {path}")
    return (manifest_path.parent / path).resolve()


def _assert_hash(path: Path, resource: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    expected = resource.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"manifest lacks a frozen SHA256 for {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch for {path}: expected {expected}, got {actual}"
        )


def _stem_set(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == suffix
    }


def _ensure_exact_set(label: str, actual: set[str], expected: set[str]) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} stem set mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def convert_expected_manifest(
    manifest_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    mel_dir: str | Path | None = None,
    acc_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    ticks_per_beat: int = TICKS_PER_BEAT,
    allow_existing: bool = False,
    update_manifest: bool = True,
) -> dict[str, Any]:
    """Fail-closed conversion of every entry in a perturbation manifest."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    exact_stems_raw = manifest.get("exact_stems")
    if not isinstance(entries, list) or not isinstance(exact_stems_raw, list):
        raise ValueError("manifest must contain entries and exact_stems lists")
    exact_stems = {str(stem) for stem in exact_stems_raw}
    entry_stems = [str(entry.get("stem")) for entry in entries]
    if len(entry_stems) != len(set(entry_stems)) or set(entry_stems) != exact_stems:
        raise ValueError("manifest entries are not a one-to-one mapping of exact_stems")
    if int(manifest.get("input_count", -1)) != len(entries):
        raise ValueError("manifest input_count does not equal entries length")

    resolved: list[tuple[dict[str, Any], Path, Path, Path]] = []
    for entry in entries:
        melody = _resolve_resource(manifest_path, entry["output_midi"])
        accompaniment = _resolve_resource(manifest_path, entry["acc_copy"])
        npz_path = _resolve_resource(manifest_path, entry["npz"])
        stem = str(entry["stem"])
        if {melody.stem, accompaniment.stem, npz_path.stem} != {stem}:
            raise ValueError(f"resource stem mismatch for {stem}")
        _assert_hash(melody, entry["output_midi"], "melody")
        _assert_hash(accompaniment, entry["acc_copy"], "accompaniment")
        resolved.append((entry, melody, accompaniment, npz_path))

    melody_dirs = {item[1].parent for item in resolved}
    accompaniment_dirs = {item[2].parent for item in resolved}
    npz_dirs = {item[3].parent for item in resolved}
    if len(melody_dirs) != 1 or len(accompaniment_dirs) != 1 or len(npz_dirs) != 1:
        raise ValueError("strict manifest requires one melody, accompaniment, and NPZ directory")
    actual_mel_dir = Path(mel_dir).resolve() if mel_dir else next(iter(melody_dirs))
    actual_acc_dir = Path(acc_dir).resolve() if acc_dir else next(iter(accompaniment_dirs))
    actual_output_dir = Path(output_dir).resolve() if output_dir else next(iter(npz_dirs))
    if actual_mel_dir not in melody_dirs or actual_acc_dir not in accompaniment_dirs:
        raise ValueError("CLI MIDI directories disagree with manifest resources")
    if actual_output_dir not in npz_dirs:
        raise ValueError("CLI output directory disagrees with manifest resources")

    _ensure_exact_set("melody", _stem_set(actual_mel_dir, ".mid"), exact_stems)
    _ensure_exact_set("accompaniment", _stem_set(actual_acc_dir, ".mid"), exact_stems)
    actual_output_dir.mkdir(parents=True, exist_ok=True)
    existing = _stem_set(actual_output_dir, ".npz")
    if existing and not allow_existing:
        raise FileExistsError(
            f"strict conversion requires a fresh NPZ directory; found {sorted(existing)}"
        )
    if existing:
        _ensure_exact_set("existing NPZ", existing, exact_stems)


    result_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for entry, melody, accompaniment, npz_path in sorted(
        resolved, key=lambda item: item[0]["stem"]
    ):
        try:
            if npz_path.exists() and allow_existing:
                expected_hash = entry["npz"].get("sha256")
                if not expected_hash or sha256_file(npz_path) != expected_hash:
                    raise ValueError(f"existing NPZ hash is missing or mismatched: {npz_path}")
                status = "verified"
            else:
                ok = midi_pair_to_npz(
                    str(melody),
                    str(accompaniment),
                    str(npz_path),
                    ticks_per_beat=ticks_per_beat,
                )
                if not ok:
                    raise ValueError("converter returned skip")
                status = "converted"
            roll_gate = verify_midi_npz_roll(
                melody,
                npz_path,
                ticks_per_beat=ticks_per_beat,
                expected_horizon=int(entry["validation_horizon_ticks"]),
            )
            npz_hash = sha256_file(npz_path)
            entry["npz"]["sha256"] = npz_hash
            result_rows.append(
                {
                    "acc_sha256": sha256_file(accompaniment),
                    "melody_sha256": sha256_file(melody),
                    "npz_sha256": npz_hash,
                    "roll_gate": roll_gate,
                    "status": status,
                    "stem": entry["stem"],
                }
            )
        except Exception as exc:
            errors.append({"error": f"{type(exc).__name__}: {exc}", "stem": entry["stem"]})

    produced_stems = _stem_set(actual_output_dir, ".npz")
    exact_output_set = produced_stems == exact_stems
    if not exact_output_set:
        errors.append(
            {
                "error": (
                    f"NPZ exact stem mismatch; missing={sorted(exact_stems-produced_stems)}, "
                    f"extra={sorted(produced_stems-exact_stems)}"
                ),
                "stem": "<campaign>",
            }
        )

    summary_path = (
        Path(summary_path).resolve()
        if summary_path
        else actual_output_dir.parent / "midi_to_npz_summary.json"
    )
    summary = {
        "converted": len(result_rows),
        "errors": errors,
        "exact_stem_set": exact_output_set,
        "expected": len(entries),
        "expected_stems": sorted(exact_stems),
        "manifest_path": os.path.relpath(manifest_path, summary_path.parent),
        "results": result_rows,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "skipped": len(errors),
        "status": "ok" if not errors and len(result_rows) == len(entries) else "failed",
        "ticks_per_beat": int(ticks_per_beat),
    }

    if not errors and len(result_rows) == len(entries) and update_manifest:
        _write_canonical_json(manifest_path, manifest)
        manifest_hash = sha256_file(manifest_path)
        manifest_path.with_name(f"{manifest_path.name}.sha256").write_text(
            f"{manifest_hash}  {manifest_path.name}\n", encoding="ascii"
        )
        summary["updated_manifest_sha256"] = manifest_hash
    _write_canonical_json(summary_path, summary)
    if errors or len(result_rows) != len(entries):
        raise RuntimeError(
            f"strict conversion failed ({len(result_rows)}/{len(entries)}); see {summary_path}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert paired MIDI files to NPZ for PianoLLaMA")
    parser.add_argument("--mel-dir", type=Path, help="Directory containing melody MIDI files")
    parser.add_argument("--acc-dir", type=Path, help="Directory containing accompaniment MIDI files")
    parser.add_argument("--output-dir", type=Path, help="Directory for NPZ files")
    parser.add_argument("--ticks-per-beat", type=int, default=TICKS_PER_BEAT)
    parser.add_argument(
        "--trim-leading-rest",
        action="store_true",
        help="Shift both tracks by the Melody first-note tick and discard earlier accompaniment",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--no-update-manifest", action="store_true")
    args = parser.parse_args()

    if args.strict:
        if args.expected_manifest is None:
            parser.error("--strict requires --expected-manifest")
        summary = convert_expected_manifest(
            args.expected_manifest,
            summary_path=args.summary_json,
            mel_dir=args.mel_dir,
            acc_dir=args.acc_dir,
            output_dir=args.output_dir,
            ticks_per_beat=args.ticks_per_beat,
            allow_existing=args.allow_existing,
            update_manifest=not args.no_update_manifest,
        )
        print(json.dumps({"converted": summary["converted"], "status": "ok"}, sort_keys=True))
        return

    if not args.mel_dir or not args.acc_dir or not args.output_dir:
        parser.error("legacy mode requires --mel-dir, --acc-dir, and --output-dir")
    os.makedirs(args.output_dir, exist_ok=True)

    mel_files = sorted(f for f in os.listdir(args.mel_dir) if f.endswith(".mid"))

    converted = 0
    skipped = 0

    for mel_file in mel_files:
        acc_file = mel_file  # Same filename in acc dir
        mel_path = os.path.join(args.mel_dir, mel_file)
        acc_path = os.path.join(args.acc_dir, acc_file)

        if not os.path.exists(acc_path):
            print(f"  [SKIP] No matching acc file for {mel_file}")
            skipped += 1
            continue

        stem = os.path.splitext(mel_file)[0]
        output_path = os.path.join(args.output_dir, f"{stem}.npz")

        print(f"Converting {mel_file} ...", end=" ")
        ok = midi_pair_to_npz(
            mel_path,
            acc_path,
            output_path,
            ticks_per_beat=args.ticks_per_beat,
            trim_leading_rest=args.trim_leading_rest,
        )
        if ok:
            print("OK")
            converted += 1
        else:
            skipped += 1

    print(f"\nDone: {converted} converted, {skipped} skipped")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
