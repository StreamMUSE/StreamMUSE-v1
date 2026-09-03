#!/usr/bin/env python3
"""Build checkpoint-aligned offline GT NPZ files from trimmed cohort MIDI."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido
import numpy as np
import pretty_midi

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from streammuse.infrastructure.inference.lekai_continuation_model.config import (  # noqa: E402
    ModelConfig,
)
from streammuse.infrastructure.inference.lekai_continuation_model.my_tokenizer import (  # noqa: E402
    PianoMusicTokenizer,
)
from streammuse.infrastructure.inference.lekai_model.MidiConverter import (  # noqa: E402
    MidiConverter,
)


SCHEMA_VERSION = "streammuse.matched_offline_gt_npz.v1"
AUDIT_SCHEMA_VERSION = "streammuse.matched_offline_gt_npz.audit.v1"
STEPS_PER_BEAT = 4
WINDOW_BEATS = 32
WINDOW_STEPS = WINDOW_BEATS * STEPS_PER_BEAT
MEASURE_STEPS = 16
WINDOW_MEASURES = 8
BPM = 120.0
TIME_SIGNATURE = "4/4"
TIME_SIGNATURE_INDEX = 0
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TRACK_NAMES = ("Melody", "Accompaniment")
AUDIT_FIELDS = (
    "order",
    "piece_id",
    "source_manifest_sha256",
    "source_melody_midi",
    "source_melody_midi_sha256",
    "source_gt_midi",
    "source_gt_midi_sha256",
    "source_npz",
    "source_npz_sha256",
    "roll_shape",
    "measure_shape",
    "source_melody_note_count",
    "source_melody_onset_count",
    "npz_melody_note_count",
    "npz_melody_onset_count",
    "source_accompaniment_note_count",
    "source_accompaniment_onset_count",
    "npz_accompaniment_note_count",
    "npz_accompaniment_onset_count",
    "melody_midi_npz_roll_exact",
    "accompaniment_midi_npz_roll_exact",
    "melody_tokenizer_roundtrip_exact",
    "accompaniment_tokenizer_roundtrip_exact",
    "melody_postjoin_geometry_exact",
    "accompaniment_postjoin_geometry_exact",
    "reserved_patch_token_collisions",
    "git_commit",
    "git_dirty",
    "script_sha256",
)


@dataclass(frozen=True)
class RawNote:
    onset_tick: int
    pitch: int
    offset_tick: int


@dataclass(frozen=True)
class SourceSample:
    row: dict[str, Any]
    order: int
    piece_id: str
    melody_path: Path
    melody_sha256: str
    gt_path: Path
    gt_sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json_bytes(value))


def _strict_sha256(value: Any, context: str) -> str:
    candidate = str(value or "").strip().lower()
    if not SHA256_PATTERN.fullmatch(candidate):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return candidate


def _resolve_reference(value: Any, manifest: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{context} not found: {path}")
    return path


def _safe_piece_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not safe:
        raise ValueError(f"piece_id has no safe path representation: {value!r}")
    return safe


def _verify_reference(
    row: Mapping[str, Any], manifest: Path, path_field: str, hash_field: str
) -> tuple[Path, str]:
    path = _resolve_reference(row.get(path_field), manifest, path_field)
    expected = _strict_sha256(row.get(hash_field), hash_field)
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{path_field} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path, actual


def load_cohort_manifest(path: Path) -> tuple[dict[str, Any], list[SourceSample]]:
    manifest = path.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"cohort manifest not found: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("cohort manifest root must be an object")
    rows = payload.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cohort manifest must contain a non-empty samples list")

    samples: list[SourceSample] = []
    piece_ids: set[str] = set()
    orders: set[int] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"samples[{index}] must be an object")
        row = dict(raw)
        piece_id = str(row.get("piece_id") or "").strip()
        if not piece_id:
            raise ValueError(f"samples[{index}].piece_id is required")
        if piece_id in piece_ids:
            raise ValueError(f"duplicate piece_id: {piece_id}")
        piece_ids.add(piece_id)
        order = int(row.get("order", index + 1))
        if order <= 0 or order in orders:
            raise ValueError(f"samples[{index}].order must be positive and unique")
        orders.add(order)
        melody_path, melody_hash = _verify_reference(
            row, manifest, "melody_midi", "melody_midi_sha256"
        )
        gt_path, gt_hash = _verify_reference(
            row, manifest, "gt_midi", "gt_midi_sha256"
        )
        samples.append(
            SourceSample(
                row=row,
                order=order,
                piece_id=piece_id,
                melody_path=melody_path,
                melody_sha256=melody_hash,
                gt_path=gt_path,
                gt_sha256=gt_hash,
            )
        )
    return payload, sorted(samples, key=lambda sample: (sample.order, sample.piece_id))


def _track_name(track: mido.MidiTrack) -> str:
    names = [str(message.name) for message in track if message.type == "track_name"]
    if len(names) > 1:
        raise ValueError("a MIDI track contains multiple track_name messages")
    return names[0].strip() if names else ""


def _physical_track_notes(
    midi: mido.MidiFile, *, expected_names: Sequence[str], context: str
) -> dict[str, tuple[RawNote, ...]]:
    physical: dict[str, tuple[RawNote, ...]] = {}
    for track in midi.tracks:
        absolute_tick = 0
        active: dict[tuple[int, int], list[int]] = {}
        notes: list[RawNote] = []
        uses_drum_channel = False
        for message in track:
            absolute_tick += int(message.time)
            if message.type == "note_on" and int(message.velocity) > 0:
                if int(message.channel) == 9:
                    uses_drum_channel = True
                active.setdefault((int(message.channel), int(message.note)), []).append(
                    absolute_tick
                )
            elif message.type in {"note_off", "note_on"}:
                key = (int(message.channel), int(message.note))
                starts = active.get(key)
                if not starts:
                    raise ValueError(f"{context} has an unmatched note-off")
                onset = starts.pop(0)
                if not starts:
                    active.pop(key)
                if absolute_tick <= onset:
                    raise ValueError(f"{context} has a non-positive note duration")
                notes.append(RawNote(onset, key[1], absolute_tick))
        if active:
            raise ValueError(f"{context} has notes without note-off events")
        if not notes:
            continue
        name = _track_name(track)
        if uses_drum_channel:
            raise ValueError(f"{context} physical track {name!r} is a drum track")
        if name in physical:
            raise ValueError(f"{context} contains duplicate physical track {name!r}")
        physical[name] = tuple(
            sorted(notes, key=lambda note: (note.onset_tick, note.pitch, note.offset_tick))
        )

    if set(physical) != set(expected_names) or len(physical) != len(expected_names):
        raise ValueError(
            f"{context} physical non-drum tracks must be exactly "
            f"{list(expected_names)}, got {sorted(physical)}"
        )
    return physical


def _validate_timing(path: Path, midi: mido.MidiFile) -> None:
    tempo_events: list[tuple[int, int]] = []
    signature_events: list[tuple[int, int, int]] = []
    for track in midi.tracks:
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            if message.type == "set_tempo":
                tempo_events.append((absolute_tick, int(message.tempo)))
            elif message.type == "time_signature":
                signature_events.append(
                    (absolute_tick, int(message.numerator), int(message.denominator))
                )
    expected_tempo = mido.bpm2tempo(BPM)
    if not tempo_events or tempo_events[0][0] != 0 or any(
        tempo != expected_tempo for _tick, tempo in tempo_events
    ):
        raise ValueError(f"{path} must contain only 120 BPM tempo events from tick 0")
    if not signature_events or signature_events[0][0] != 0 or any(
        (numerator, denominator) != (4, 4)
        for _tick, numerator, denominator in signature_events
    ):
        raise ValueError(f"{path} must contain only 4/4 time signatures from tick 0")
    if midi.ticks_per_beat <= 0 or midi.ticks_per_beat % STEPS_PER_BEAT:
        raise ValueError(f"{path} ticks_per_beat must be positive and divisible by 4")


def _canonical_melody_sha256(notes: Sequence[RawNote], ticks_per_beat: int) -> str:
    ticks_per_step = ticks_per_beat // STEPS_PER_BEAT
    canonical: list[list[int]] = []
    for note in notes:
        if note.onset_tick % ticks_per_step or note.offset_tick % ticks_per_step:
            raise ValueError("Melody notes must lie exactly on the four-steps-per-beat grid")
        canonical.append(
            [
                note.onset_tick // ticks_per_step,
                note.offset_tick // ticks_per_step,
                note.pitch,
                80,
            ]
        )
    value = {
        "bpm": 120,
        "notes": canonical,
        "schema_version": 1,
        "steps_per_beat": STEPS_PER_BEAT,
        "track_name": "Melody",
    }
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select_instrument(
    pm: pretty_midi.PrettyMIDI, name: str, *, expected_names: Sequence[str], context: str
) -> pretty_midi.Instrument:
    physical = [instrument for instrument in pm.instruments if instrument.notes]
    if any(instrument.is_drum for instrument in physical):
        raise ValueError(f"{context} contains a physical drum instrument")
    if {instrument.name for instrument in physical} != set(expected_names) or len(
        physical
    ) != len(expected_names):
        raise ValueError(
            f"{context} physical instrument names must be exactly "
            f"{list(expected_names)}, got {sorted(instrument.name for instrument in physical)}"
        )
    return next(instrument for instrument in physical if instrument.name == name)


def _quantize_named_track(
    pm: pretty_midi.PrettyMIDI,
    instrument: pretty_midi.Instrument,
    converter: MidiConverter,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    isolated = copy.deepcopy(pm)
    isolated.instruments = [copy.deepcopy(instrument)]
    notes, _max_tick = converter.midi_to_notes(isolated, filter_drums=True)
    if any(not 21 <= int(note["pitch"]) <= 108 for note in notes):
        raise ValueError(f"track {instrument.name!r} contains pitch outside piano range")
    clipped: list[dict[str, Any]] = []
    for note in notes:
        start = int(note["tick"])
        end = start + int(note["duration"])
        if end <= 0 or start >= WINDOW_STEPS:
            continue
        start = max(0, start)
        end = min(WINDOW_STEPS, end)
        if end <= start:
            continue
        copied = dict(note)
        copied["tick"] = start
        copied["duration"] = end - start
        clipped.append(copied)
    roll = converter.notes_to_pianoroll(clipped, max_tick=WINDOW_STEPS).astype(
        np.uint8, copy=False
    )
    return roll, clipped


def _roll_geometry(roll: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    sustain, onset = roll
    geometry: list[tuple[int, int, int]] = []
    for pitch_index in range(88):
        for start_raw in np.flatnonzero(onset[pitch_index] > 0):
            start = int(start_raw)
            end = start + 1
            while end < roll.shape[2] and sustain[pitch_index, end] > 0:
                end += 1
            geometry.append((pitch_index + 21, start, end))
    return tuple(sorted(geometry))


def _clip_geometry(
    geometry: Sequence[tuple[int, int, int]], start: int, end: int
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        sorted(
            (pitch, max(onset, start) - start, min(offset, end) - start)
            for pitch, onset, offset in geometry
            if offset > start and onset < end
        )
    )


def _raw_step_geometry(
    notes: Sequence[RawNote], ticks_per_beat: int
) -> tuple[tuple[int, int, int], ...]:
    ticks_per_step = ticks_per_beat // STEPS_PER_BEAT
    geometry: list[tuple[int, int, int]] = []
    for note in notes:
        if note.onset_tick % ticks_per_step or note.offset_tick % ticks_per_step:
            raise ValueError("GT notes must lie exactly on the four-steps-per-beat grid")
        onset = note.onset_tick // ticks_per_step
        offset = note.offset_tick // ticks_per_step
        if offset <= 0 or onset >= WINDOW_STEPS:
            continue
        geometry.append(
            (note.pitch, max(0, onset), min(WINDOW_STEPS, offset))
        )
    return tuple(sorted(geometry))


def _tokenizer_roundtrip(
    measures: list[np.ndarray], metadata: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, int]:
    tokenizer = PianoMusicTokenizer(config=ModelConfig())
    collisions = 0
    special_ids = np.asarray(tokenizer._codec.special_token_ids, dtype=np.int64)
    for measure in measures:
        for start in range(0, MEASURE_STEPS, STEPS_PER_BEAT):
            beat = measure[:, :, start : start + STEPS_PER_BEAT]
            for track_roll in (beat[:2], beat[2:]):
                raw = tokenizer._codec.image_to_patch_tokens(
                    track_roll, strict_mode=False
                )
                collisions += int(np.isin(raw, special_ids).sum())
    if collisions:
        raise ValueError(
            "source roll cannot round-trip exactly through the offline tokenizer: "
            f"{collisions} reserved patch-token collision(s)"
        )
    prepared = tokenizer.build_generation_schedule(
        measures,
        metadata,
        gt_prefix_beats=8,
        timesteps_per_beat=STEPS_PER_BEAT,
    )
    melody = tokenizer.decode_beats_to_pianoroll(
        prepared["mel_beats"], tokenizer.vocab.track_marker_mel
    ).astype(np.uint8)
    accompaniment = tokenizer.decode_beats_to_pianoroll(
        prepared["acc_beats_gt"], tokenizer.vocab.track_marker_acc
    ).astype(np.uint8)
    return melody, accompaniment, collisions


def _git_identity() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        dirty = True
    return {"git_commit": commit, "git_dirty": dirty}


def _copy_preserved_relative_midi(
    raw_reference: Any, source: Path, staging: Path, context: str
) -> None:
    reference = Path(str(raw_reference))
    if reference.is_absolute():
        return
    target = (staging / reference).resolve()
    if not target.is_relative_to(staging.resolve()):
        raise ValueError(f"{context} relative path must remain inside output directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and file_sha256(target) != file_sha256(source):
        raise ValueError(f"{context} collides with a different output file")
    if not target.exists():
        shutil.copy2(source, target)


def _write_npz(path: Path, metadata: dict[str, Any], measures: list[np.ndarray]) -> None:
    payload: dict[str, Any] = {"metadata": metadata}
    payload.update({f"measure_{index}": measure for index, measure in enumerate(measures)})
    np.savez_compressed(path, **payload)


def _read_npz_roll(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        metadata = payload["metadata"].item()
        measures = [np.asarray(payload[f"measure_{index}"]) for index in range(8)]
    return metadata, np.concatenate(measures, axis=2)


def _prepare_output_target(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output path is not a directory: {output}")
        if next(output.iterdir(), None) is not None:
            raise FileExistsError(f"output directory is not empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def prepare_matched_offline_gt_npz(
    *, cohort_manifest: Path, output_dir: Path
) -> dict[str, Any]:
    source_manifest = cohort_manifest.expanduser().resolve()
    destination = _prepare_output_target(output_dir)
    source_payload, samples = load_cohort_manifest(source_manifest)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    source_manifest_hash = file_sha256(source_manifest)
    script_hash = file_sha256(Path(__file__).resolve())
    code = _git_identity()
    converter = MidiConverter(ticks_per_beat=STEPS_PER_BEAT)
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    digits = max(2, len(str(len(samples))))

    try:
        for sample in samples:
            melody_mido = mido.MidiFile(sample.melody_path)
            gt_mido = mido.MidiFile(sample.gt_path)
            _validate_timing(sample.melody_path, melody_mido)
            _validate_timing(sample.gt_path, gt_mido)
            melody_raw = _physical_track_notes(
                melody_mido, expected_names=("Melody",), context=str(sample.melody_path)
            )["Melody"]
            gt_raw = _physical_track_notes(
                gt_mido, expected_names=TRACK_NAMES, context=str(sample.gt_path)
            )
            if melody_raw != gt_raw["Melody"]:
                raise ValueError(f"{sample.piece_id} GT Melody differs from melody_midi")
            if min(note.onset_tick for note in melody_raw) != 0:
                raise ValueError(
                    f"{sample.piece_id} canonical trimmed Melody must start at tick 0"
                )
            declared_canonical = sample.row.get("canonical_melody_input_sha256")
            canonical = _canonical_melody_sha256(
                melody_raw, melody_mido.ticks_per_beat
            )
            if declared_canonical is not None and canonical != _strict_sha256(
                declared_canonical, "canonical_melody_input_sha256"
            ):
                raise ValueError(
                    f"{sample.piece_id} canonical_melody_input_sha256 mismatch"
                )

            melody_pm = pretty_midi.PrettyMIDI(str(sample.melody_path))
            gt_pm = pretty_midi.PrettyMIDI(str(sample.gt_path))
            melody_instrument = _select_instrument(
                melody_pm,
                "Melody",
                expected_names=("Melody",),
                context=str(sample.melody_path),
            )
            gt_melody_instrument = _select_instrument(
                gt_pm,
                "Melody",
                expected_names=TRACK_NAMES,
                context=str(sample.gt_path),
            )
            gt_acc_instrument = _select_instrument(
                gt_pm,
                "Accompaniment",
                expected_names=TRACK_NAMES,
                context=str(sample.gt_path),
            )
            melody_roll, _ = _quantize_named_track(
                melody_pm, melody_instrument, converter
            )
            gt_melody_roll, _ = _quantize_named_track(
                gt_pm, gt_melody_instrument, converter
            )
            acc_roll, _ = _quantize_named_track(gt_pm, gt_acc_instrument, converter)
            if not np.array_equal(melody_roll, gt_melody_roll):
                raise ValueError(
                    f"{sample.piece_id} GT and input Melody differ after quantization"
                )

            source_roll = np.concatenate([melody_roll, acc_roll], axis=0)
            source_geometry = {
                "Melody": _raw_step_geometry(
                    gt_raw["Melody"], gt_mido.ticks_per_beat
                ),
                "Accompaniment": _raw_step_geometry(
                    gt_raw["Accompaniment"], gt_mido.ticks_per_beat
                ),
            }
            roll_geometry = {
                "Melody": _roll_geometry(melody_roll),
                "Accompaniment": _roll_geometry(acc_roll),
            }
            if source_geometry != roll_geometry:
                raise ValueError(
                    f"{sample.piece_id} source MIDI geometry is not exactly representable "
                    "by the four-channel roll"
                )

            measures = [
                source_roll[:, :, start : start + MEASURE_STEPS].copy()
                for start in range(0, WINDOW_STEPS, MEASURE_STEPS)
            ]
            metadata = {
                "time_signature": TIME_SIGNATURE,
                "time_signature_idx": TIME_SIGNATURE_INDEX,
                "bpm": BPM,
                "num_measures": WINDOW_MEASURES,
                "num_channels": 4,
                "resolution": MEASURE_STEPS,
                "total_length": WINDOW_STEPS,
                "is_continuation": False,
                "steps_per_beat": STEPS_PER_BEAT,
                "provenance": {
                    "schema_version": SCHEMA_VERSION,
                    "source_cohort_manifest_sha256": source_manifest_hash,
                    "source_melody_midi_sha256": sample.melody_sha256,
                    "source_gt_midi_sha256": sample.gt_sha256,
                    "quantizer": (
                        "streammuse.infrastructure.inference.lekai_model."
                        "MidiConverter(ticks_per_beat=4)"
                    ),
                    "script_sha256": script_hash,
                },
            }
            decoded_melody, decoded_acc, collisions = _tokenizer_roundtrip(
                measures, metadata
            )
            melody_token_exact = np.array_equal(decoded_melody, melody_roll)
            acc_token_exact = np.array_equal(decoded_acc, acc_roll)
            if not melody_token_exact or not acc_token_exact:
                raise ValueError(
                    f"{sample.piece_id} offline tokenizer round-trip is not exact"
                )
            postjoin_source = {
                name: _clip_geometry(geometry, 8 * STEPS_PER_BEAT, WINDOW_STEPS)
                for name, geometry in source_geometry.items()
            }
            postjoin_decoded = {
                "Melody": _clip_geometry(
                    _roll_geometry(decoded_melody), 8 * STEPS_PER_BEAT, WINDOW_STEPS
                ),
                "Accompaniment": _clip_geometry(
                    _roll_geometry(decoded_acc), 8 * STEPS_PER_BEAT, WINDOW_STEPS
                ),
            }
            if postjoin_source != postjoin_decoded:
                raise ValueError(
                    f"{sample.piece_id} post-join tokenizer geometry is not exact"
                )

            piece_dir = staging / (
                f"{sample.order:0{digits}d}_{_safe_piece_id(sample.piece_id)}"
            )
            piece_dir.mkdir()
            npz_path = piece_dir / "source.npz"
            _write_npz(npz_path, metadata, measures)
            loaded_metadata, loaded_roll = _read_npz_roll(npz_path)
            if loaded_metadata != metadata or loaded_roll.shape != (4, 88, 128):
                raise RuntimeError(f"{sample.piece_id} saved NPZ metadata/shape mismatch")
            mel_npz_exact = np.array_equal(loaded_roll[:2], melody_roll)
            acc_npz_exact = np.array_equal(loaded_roll[2:], acc_roll)
            if not mel_npz_exact or not acc_npz_exact:
                raise RuntimeError(f"{sample.piece_id} saved NPZ roll mismatch")
            npz_geometry = {
                "Melody": _roll_geometry(loaded_roll[:2]),
                "Accompaniment": _roll_geometry(loaded_roll[2:]),
            }

            _copy_preserved_relative_midi(
                sample.row["melody_midi"],
                sample.melody_path,
                staging,
                "melody_midi",
            )
            _copy_preserved_relative_midi(
                sample.row["gt_midi"], sample.gt_path, staging, "gt_midi"
            )
            output_row = dict(sample.row)
            output_row["source_npz"] = npz_path.relative_to(staging).as_posix()
            output_row["source_npz_sha256"] = file_sha256(npz_path)
            rows.append(output_row)
            audit_rows.append(
                {
                    "order": sample.order,
                    "piece_id": sample.piece_id,
                    "source_manifest_sha256": source_manifest_hash,
                    "source_melody_midi": str(sample.melody_path),
                    "source_melody_midi_sha256": sample.melody_sha256,
                    "source_gt_midi": str(sample.gt_path),
                    "source_gt_midi_sha256": sample.gt_sha256,
                    "source_npz": output_row["source_npz"],
                    "source_npz_sha256": output_row["source_npz_sha256"],
                    "roll_shape": json.dumps(list(loaded_roll.shape)),
                    "measure_shape": json.dumps(list(measures[0].shape)),
                    "source_melody_note_count": len(source_geometry["Melody"]),
                    "source_melody_onset_count": len(source_geometry["Melody"]),
                    "npz_melody_note_count": len(npz_geometry["Melody"]),
                    "npz_melody_onset_count": int(loaded_roll[1].sum()),
                    "source_accompaniment_note_count": len(
                        source_geometry["Accompaniment"]
                    ),
                    "source_accompaniment_onset_count": len(
                        source_geometry["Accompaniment"]
                    ),
                    "npz_accompaniment_note_count": len(
                        npz_geometry["Accompaniment"]
                    ),
                    "npz_accompaniment_onset_count": int(loaded_roll[3].sum()),
                    "melody_midi_npz_roll_exact": mel_npz_exact,
                    "accompaniment_midi_npz_roll_exact": acc_npz_exact,
                    "melody_tokenizer_roundtrip_exact": melody_token_exact,
                    "accompaniment_tokenizer_roundtrip_exact": acc_token_exact,
                    "melody_postjoin_geometry_exact": (
                        postjoin_source["Melody"] == postjoin_decoded["Melody"]
                    ),
                    "accompaniment_postjoin_geometry_exact": (
                        postjoin_source["Accompaniment"]
                        == postjoin_decoded["Accompaniment"]
                    ),
                    "reserved_patch_token_collisions": collisions,
                    **code,
                    "script_sha256": script_hash,
                }
            )

        output_manifest = dict(source_payload)
        output_manifest["samples"] = rows
        output_manifest["offline_gt_npz_preparation"] = {
            "schema_version": SCHEMA_VERSION,
            "source_cohort_manifest": str(source_manifest),
            "source_cohort_manifest_sha256": source_manifest_hash,
            "sample_count": len(rows),
            "window_beats": [0, WINDOW_BEATS],
            "window_steps": [0, WINDOW_STEPS],
            "steps_per_beat": STEPS_PER_BEAT,
            "channel_order": ["mel_sus", "mel_on", "acc_sus", "acc_on"],
            "exact_roll_gate": True,
            "exact_tokenizer_roundtrip_gate": True,
            "exact_postjoin_geometry_gate": True,
            "script_sha256": script_hash,
            **code,
        }
        manifest_path = staging / "cohort_manifest.json"
        _write_json(manifest_path, output_manifest)
        manifest_hash = file_sha256(manifest_path)
        (staging / "cohort_manifest.json.sha256").write_text(
            f"{manifest_hash}  cohort_manifest.json\n", encoding="ascii"
        )
        with (staging / "audit.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(AUDIT_FIELDS))
            writer.writeheader()
            writer.writerows(audit_rows)
        _write_json(
            staging / "audit.json",
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "source_cohort_manifest": str(source_manifest),
                "source_cohort_manifest_sha256": source_manifest_hash,
                "sample_count": len(audit_rows),
                "code": {**code, "script_sha256": script_hash},
                "samples": audit_rows,
            },
        )

        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
        return output_manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare_matched_offline_gt_npz(
        cohort_manifest=args.cohort_manifest, output_dir=args.output_dir
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "sample_count": len(result["samples"]),
                "status": "complete",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
