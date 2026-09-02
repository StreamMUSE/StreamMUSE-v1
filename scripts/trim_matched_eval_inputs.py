#!/usr/bin/env python3
"""Prepare leading-rest-trimmed MIDI inputs for matched system evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido

STEPS_PER_BEAT = 4
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
AUDIT_FIELDS = (
    "order",
    "piece_id",
    "source_melody_midi",
    "source_melody_midi_sha256",
    "trimmed_melody_midi",
    "trimmed_melody_midi_sha256",
    "source_gt_midi",
    "source_gt_midi_sha256",
    "trimmed_gt_midi",
    "trimmed_gt_midi_sha256",
    "ticks_per_beat",
    "first_melody_note_tick_before",
    "first_melody_note_tick_after",
    "leading_trim_ticks",
    "leading_trim_beats",
    "midi_artifacts_shifted_only",
    "note_count",
    "gt_dropped_pre_cutoff_note_count",
    "gt_rehydrated_cross_cutoff_note_count",
    "changed",
    "canonical_melody_sha256_before",
    "canonical_melody_sha256_after",
)


@dataclass(frozen=True, order=True)
class MidiNote:
    onset_tick: int
    pitch: int
    offset_tick: int
    velocity: int
    channel: int


@dataclass(frozen=True)
class PreparedSample:
    row: dict[str, Any]
    order: int
    piece_id: str
    melody_path: Path
    melody_sha256: str
    midi: mido.MidiFile
    notes: tuple[MidiNote, ...]
    canonical_sha256: str
    gt_path: Path
    gt_sha256: str
    gt_midi: mido.MidiFile


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_sha256(value: Any, context: str) -> str:
    candidate = str(value or "").strip().lower()
    if not SHA256_PATTERN.fullmatch(candidate):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return candidate


def _resolve_reference(value: Any, manifest_path: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{context} not found: {path}")
    return path


def _track_name(track: mido.MidiTrack) -> str | None:
    for message in track:
        if message.type == "track_name":
            return str(message.name)
    return None


def _melody_track(midi: mido.MidiFile, context: str) -> mido.MidiTrack:
    tracks = [
        track
        for track in midi.tracks
        if (_track_name(track) or "").strip().casefold() == "melody"
    ]
    if len(tracks) != 1:
        raise ValueError(f"{context} must contain exactly one track named Melody")
    return tracks[0]


def _extract_melody_notes(midi: mido.MidiFile, context: str) -> tuple[MidiNote, ...]:
    track = _melody_track(midi, context)
    active: dict[tuple[int, int], list[tuple[int, int]]] = {}
    notes: list[MidiNote] = []
    absolute_tick = 0
    for message in track:
        absolute_tick += int(message.time)
        if message.type == "note_on" and int(message.velocity) > 0:
            key = (int(message.channel), int(message.note))
            active.setdefault(key, []).append((absolute_tick, int(message.velocity)))
            continue
        if message.type not in {"note_off", "note_on"}:
            continue
        key = (int(message.channel), int(message.note))
        starts = active.get(key)
        if not starts:
            raise ValueError(f"{context} has an unmatched note-off at tick {absolute_tick}")
        onset_tick, velocity = starts.pop(0)
        if not starts:
            active.pop(key)
        if absolute_tick <= onset_tick:
            raise ValueError(f"{context} has a non-positive note duration")
        notes.append(
            MidiNote(
                onset_tick=onset_tick,
                pitch=key[1],
                offset_tick=absolute_tick,
                velocity=velocity,
                channel=key[0],
            )
        )
    if active:
        raise ValueError(f"{context} has melody notes without note-off events")
    if not notes:
        raise ValueError(f"{context} has no Melody note-on events")
    return tuple(sorted(notes))


def canonical_melody_sha256(
    notes: Sequence[MidiNote], *, ticks_per_beat: int
) -> str:
    if ticks_per_beat <= 0 or ticks_per_beat % STEPS_PER_BEAT:
        raise ValueError("MIDI ticks_per_beat must be divisible by 4")
    ticks_per_step = ticks_per_beat // STEPS_PER_BEAT
    canonical_notes: list[list[int]] = []
    for note in notes:
        if note.onset_tick % ticks_per_step or note.offset_tick % ticks_per_step:
            raise ValueError("Melody notes must lie on the four-steps-per-beat grid")
        canonical_notes.append(
            [
                note.onset_tick // ticks_per_step,
                note.offset_tick // ticks_per_step,
                note.pitch,
                note.velocity,
            ]
        )
    return _canonical_sha256(
        {
            "bpm": 120,
            "notes": canonical_notes,
            "schema_version": 1,
            "steps_per_beat": STEPS_PER_BEAT,
            "track_name": "Melody",
        }
    )


def _validate_referenced_artifact(
    row: Mapping[str, Any],
    *,
    path_field: str,
    hash_field: str,
    manifest_path: Path,
    context: str,
) -> Path:
    path = _resolve_reference(row.get(path_field), manifest_path, f"{context}.{path_field}")
    expected_hash = _strict_sha256(row.get(hash_field), f"{context}.{hash_field}")
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{context}.{path_field} SHA-256 mismatch: expected {expected_hash}, "
            f"got {actual_hash}"
        )
    return path


def load_source_manifest(path: Path) -> tuple[dict[str, Any], list[PreparedSample]]:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cohort manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("cohort manifest root must be an object")
    rows = payload.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cohort manifest must contain a non-empty samples list")

    prepared: list[PreparedSample] = []
    seen_piece_ids: set[str] = set()
    for index, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise TypeError(f"samples[{index - 1}] must be an object")
        row = dict(raw_row)
        context = f"samples[{index - 1}]"
        piece_id = str(row.get("piece_id") or "").strip()
        if not piece_id:
            raise ValueError(f"{context}.piece_id is required")
        if piece_id in seen_piece_ids:
            raise ValueError(f"duplicate piece_id: {piece_id}")
        seen_piece_ids.add(piece_id)
        order = int(row.get("order", index))
        melody_path = _validate_referenced_artifact(
            row,
            path_field="melody_midi",
            hash_field="melody_midi_sha256",
            manifest_path=manifest_path,
            context=context,
        )
        _validate_referenced_artifact(
            row,
            path_field="source_npz",
            hash_field="source_npz_sha256",
            manifest_path=manifest_path,
            context=context,
        )
        gt_path = _validate_referenced_artifact(
            row,
            path_field="gt_midi",
            hash_field="gt_midi_sha256",
            manifest_path=manifest_path,
            context=context,
        )
        midi = mido.MidiFile(melody_path)
        notes = _extract_melody_notes(midi, str(melody_path))
        gt_midi = mido.MidiFile(gt_path)
        if gt_midi.ticks_per_beat != midi.ticks_per_beat:
            raise ValueError(
                f"{context}.gt_midi ticks_per_beat differs from melody_midi"
            )
        gt_melody_notes = _extract_melody_notes(gt_midi, str(gt_path))
        if gt_melody_notes != notes:
            raise ValueError(
                f"{context}.gt_midi Melody track differs from melody_midi"
            )
        canonical_hash = canonical_melody_sha256(
            notes, ticks_per_beat=midi.ticks_per_beat
        )
        declared_canonical = row.get("canonical_melody_input_sha256")
        if declared_canonical is not None:
            expected_canonical = _strict_sha256(
                declared_canonical, f"{context}.canonical_melody_input_sha256"
            )
            if canonical_hash != expected_canonical:
                raise ValueError(
                    f"{context}.canonical_melody_input_sha256 mismatch: "
                    f"expected {expected_canonical}, got {canonical_hash}"
                )
        prepared.append(
            PreparedSample(
                row=row,
                order=order,
                piece_id=piece_id,
                melody_path=melody_path,
                melody_sha256=file_sha256(melody_path),
                midi=midi,
                notes=notes,
                canonical_sha256=canonical_hash,
                gt_path=gt_path,
                gt_sha256=file_sha256(gt_path),
                gt_midi=gt_midi,
            )
        )
    return payload, prepared


def _safe_piece_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not safe:
        raise ValueError(f"piece_id has no safe path representation: {value!r}")
    return safe


def _trim_midi(
    midi: mido.MidiFile, *, offset_ticks: int
) -> tuple[mido.MidiFile, dict[str, int]]:
    trimmed = mido.MidiFile(
        type=midi.type,
        ticks_per_beat=midi.ticks_per_beat,
        clip=midi.clip,
    )
    dropped_notes = 0
    rehydrated_notes = 0
    for source_track in midi.tracks:
        absolute_messages: list[tuple[int, int, mido.Message | mido.MetaMessage]] = []
        absolute_tick = 0
        for index, message in enumerate(source_track):
            absolute_tick += int(message.time)
            absolute_messages.append((absolute_tick, index, message))

        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
        note_pairs: list[tuple[int, int, int, int]] = []
        for absolute_tick, index, message in absolute_messages:
            if message.type == "note_on" and int(message.velocity) > 0:
                key = (int(message.channel), int(message.note))
                active.setdefault(key, []).append((index, absolute_tick))
                continue
            if message.type not in {"note_off", "note_on"}:
                continue
            key = (int(message.channel), int(message.note))
            starts = active.get(key)
            if not starts:
                raise ValueError("MIDI contains an unmatched note-off")
            start_index, onset_tick = starts.pop(0)
            if not starts:
                active.pop(key)
            note_pairs.append((start_index, index, onset_tick, absolute_tick))
        if active:
            raise ValueError("MIDI contains notes without note-off events")

        dropped_indices: set[int] = set()
        overridden_ticks: dict[int, int] = {}
        for start_index, end_index, onset_tick, offset_tick in note_pairs:
            if offset_tick <= offset_ticks:
                dropped_indices.update((start_index, end_index))
                dropped_notes += 1
            elif onset_tick < offset_ticks:
                overridden_ticks[start_index] = 0
                overridden_ticks[end_index] = offset_tick - offset_ticks
                rehydrated_notes += 1
            else:
                overridden_ticks[start_index] = onset_tick - offset_ticks
                overridden_ticks[end_index] = offset_tick - offset_ticks

        shifted_messages: list[
            tuple[int, int, mido.Message | mido.MetaMessage]
        ] = []
        for absolute_tick, index, message in absolute_messages:
            if index in dropped_indices:
                continue
            shifted_tick = overridden_ticks.get(
                index, max(0, absolute_tick - offset_ticks)
            )
            shifted_messages.append((shifted_tick, index, message))

        target_track = mido.MidiTrack()
        previous_tick = 0
        for shifted_tick, _index, message in sorted(shifted_messages):
            target_track.append(message.copy(time=shifted_tick - previous_tick))
            previous_tick = shifted_tick
        trimmed.tracks.append(target_track)
    return trimmed, {
        "dropped_pre_cutoff_note_count": dropped_notes,
        "rehydrated_cross_cutoff_note_count": rehydrated_notes,
    }


def _relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prepare_output_dir(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _generated_readme(source_manifest: Path, source_hash: str, count: int) -> str:
    return f"""# Leading-rest-trimmed matched evaluation inputs

This directory was prepared from `{source_manifest}` (SHA256 `{source_hash}`).
It contains {count} Melody MIDI inputs for the matched StreamMUSE system runner.

For each sample, the earliest effective note-on in the track named `Melody`
defines one cutoff. The same cutoff is applied to both Melody and GT MIDI.
Notes ending at or before the cutoff are removed; notes crossing the cutoff are
rehydrated with a note-on at tick 0 and retain their remaining duration. Other
notes preserve duration, pitch, velocity, channel, and relative timing.

`cohort_manifest.json` is the runner input. `trim_audit.csv` records source and
output hashes, the exact tick/beat shift, note count, rehydration/drop counts,
and canonical Melody hash before and after trimming. `source_npz` remains the
original hash-validated artifact and is referenced relative to this directory.

The source NPZ is intentionally not shifted. The offline NPZ control therefore
retains its original time origin and must not be treated as tick-exact aligned
with these trimmed MIDI trials without applying the same cutoff separately.

This is an offline cohort preparation step. It does not change production MIDI
input, quantization, scheduling, inference, or offline model code.
"""


def prepare_trimmed_cohort(*, cohort_manifest: Path, output_dir: Path) -> dict[str, Any]:
    source_manifest = cohort_manifest.expanduser().resolve()
    payload, samples = load_source_manifest(source_manifest)
    destination = _prepare_output_dir(output_dir)
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    digits = max(2, len(str(len(samples))))

    for sample in samples:
        first_tick = min(note.onset_tick for note in sample.notes)
        piece_dir = destination / (
            f"{sample.order:0{digits}d}_{_safe_piece_name(sample.piece_id)}"
        )
        piece_dir.mkdir()
        trimmed_path = piece_dir / "melody_120bpm.mid"
        trimmed_gt_path = piece_dir / "gt_120bpm.mid"
        if first_tick == 0:
            shutil.copy2(sample.melody_path, trimmed_path)
            shutil.copy2(sample.gt_path, trimmed_gt_path)
            gt_trim_stats = {
                "dropped_pre_cutoff_note_count": 0,
                "rehydrated_cross_cutoff_note_count": 0,
            }
        else:
            trimmed_melody, _melody_trim_stats = _trim_midi(
                sample.midi, offset_ticks=first_tick
            )
            trimmed_gt, gt_trim_stats = _trim_midi(
                sample.gt_midi, offset_ticks=first_tick
            )
            trimmed_melody.save(trimmed_path)
            trimmed_gt.save(trimmed_gt_path)

        trimmed_midi = mido.MidiFile(trimmed_path)
        trimmed_notes = _extract_melody_notes(trimmed_midi, str(trimmed_path))
        expected_notes = tuple(
            MidiNote(
                onset_tick=note.onset_tick - first_tick,
                pitch=note.pitch,
                offset_tick=note.offset_tick - first_tick,
                velocity=note.velocity,
                channel=note.channel,
            )
            for note in sample.notes
        )
        if trimmed_notes != expected_notes:
            raise RuntimeError(f"trimmed Melody notes differ for {sample.piece_id}")
        trimmed_gt_midi = mido.MidiFile(trimmed_gt_path)
        if _extract_melody_notes(trimmed_gt_midi, str(trimmed_gt_path)) != expected_notes:
            raise RuntimeError(
                f"trimmed GT Melody notes differ for {sample.piece_id}"
            )
        trimmed_canonical = canonical_melody_sha256(
            trimmed_notes, ticks_per_beat=trimmed_midi.ticks_per_beat
        )
        trimmed_hash = file_sha256(trimmed_path)
        trimmed_gt_hash = file_sha256(trimmed_gt_path)

        output_row = dict(sample.row)
        original_source_npz = _resolve_reference(
            sample.row["source_npz"],
            source_manifest,
            f"{sample.piece_id}.source_npz",
        )
        output_row["source_npz"] = _relative_path(original_source_npz, destination)
        output_row.update(
            {
                "melody_midi": _relative_path(trimmed_path, destination),
                "melody_midi_sha256": trimmed_hash,
                "gt_midi": _relative_path(trimmed_gt_path, destination),
                "gt_midi_sha256": trimmed_gt_hash,
                "canonical_melody_input_sha256": trimmed_canonical,
                "original_melody_midi": _relative_path(
                    sample.melody_path, destination
                ),
                "original_melody_midi_sha256": sample.melody_sha256,
                "original_canonical_melody_input_sha256": sample.canonical_sha256,
                "original_gt_midi": _relative_path(sample.gt_path, destination),
                "original_gt_midi_sha256": sample.gt_sha256,
                "leading_trim_ticks": first_tick,
                "leading_trim_beats": first_tick / trimmed_midi.ticks_per_beat,
                "midi_artifacts_shifted_only": True,
            }
        )
        output_rows.append(output_row)
        audit_rows.append(
            {
                "order": sample.order,
                "piece_id": sample.piece_id,
                "source_melody_midi": str(sample.melody_path),
                "source_melody_midi_sha256": sample.melody_sha256,
                "trimmed_melody_midi": str(trimmed_path),
                "trimmed_melody_midi_sha256": trimmed_hash,
                "source_gt_midi": str(sample.gt_path),
                "source_gt_midi_sha256": sample.gt_sha256,
                "trimmed_gt_midi": str(trimmed_gt_path),
                "trimmed_gt_midi_sha256": trimmed_gt_hash,
                "ticks_per_beat": trimmed_midi.ticks_per_beat,
                "first_melody_note_tick_before": first_tick,
                "first_melody_note_tick_after": min(
                    note.onset_tick for note in trimmed_notes
                ),
                "leading_trim_ticks": first_tick,
                "leading_trim_beats": first_tick / trimmed_midi.ticks_per_beat,
                "midi_artifacts_shifted_only": True,
                "note_count": len(trimmed_notes),
                "gt_dropped_pre_cutoff_note_count": gt_trim_stats[
                    "dropped_pre_cutoff_note_count"
                ],
                "gt_rehydrated_cross_cutoff_note_count": gt_trim_stats[
                    "rehydrated_cross_cutoff_note_count"
                ],
                "changed": first_tick > 0,
                "canonical_melody_sha256_before": sample.canonical_sha256,
                "canonical_melody_sha256_after": trimmed_canonical,
            }
        )

    source_hash = file_sha256(source_manifest)
    output_manifest = dict(payload)
    output_manifest["samples"] = output_rows
    output_manifest["leading_rest_preprocessing"] = {
        "schema_version": "streammuse.matched_eval.leading_rest_trim.v1",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_hash,
        "anchor": "earliest effective note_on in the exact Melody track",
        "output_first_melody_note_tick": 0,
        "steps_per_beat_contract": STEPS_PER_BEAT,
        "sample_count": len(output_rows),
        "shifted_artifacts": ["melody_midi", "gt_midi"],
        "source_npz_shifted": False,
        "midi_artifacts_shifted_only": True,
    }
    manifest_path = destination / "cohort_manifest.json"
    _write_json(manifest_path, output_manifest)
    manifest_hash = file_sha256(manifest_path)
    (destination / "cohort_manifest.json.sha256").write_text(
        f"{manifest_hash}  cohort_manifest.json\n", encoding="ascii"
    )
    with (destination / "trim_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AUDIT_FIELDS))
        writer.writeheader()
        writer.writerows(audit_rows)
    (destination / "README.md").write_text(
        _generated_readme(source_manifest, source_hash, len(output_rows)),
        encoding="utf-8",
    )
    return output_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim leading rest from matched cohort Melody MIDI inputs."
    )
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = prepare_trimmed_cohort(
        cohort_manifest=args.cohort_manifest,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "sample_count": len(payload["samples"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
