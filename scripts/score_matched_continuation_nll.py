from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pretty_midi
import torch
import torch.nn.functional as F

from streammuse.infrastructure.inference.lekai_continuation_model.inference_adapter import (
    PianoContinuationAdapter,
)
from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter


SCHEMA_VERSION = "streammuse.matched_continuation_nll.v1"
STEPS_PER_BEAT = 4
BEATS_PER_MEASURE = 4
POST_JOIN_BEATS = 24
HORIZON_TICKS = POST_JOIN_BEATS * STEPS_PER_BEAT
MEASURE_TICKS = BEATS_PER_MEASURE * STEPS_PER_BEAT
NUM_MEASURES = HORIZON_TICKS // MEASURE_TICKS
BPM = 120
TIME_SIGNATURE_INDEX = 0
MAX_SEQUENCE_LENGTH = 2048
IGNORE_INDEX = -100
SHA256_RE = re.compile(r"[0-9a-f]{64}")
TIME_TOLERANCE_SECONDS = 1e-6
TEMPO_TOLERANCE = 1e-6


class ContractError(ValueError):
    pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("common-valid manifest root must be an object")
    return payload


def _require_text(row: dict[str, Any], field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{context}: {field} must be a non-empty string")
    return value


def _canonical_key(piece_id: Any, seed: Any, *, context: str) -> tuple[str, str]:
    if isinstance(piece_id, bool) or not isinstance(piece_id, (str, int)):
        raise ContractError(f"{context}: piece_id must be a string or integer")
    if isinstance(seed, bool) or not isinstance(seed, (str, int)):
        raise ContractError(f"{context}: seed must be a string or integer")
    piece_text = str(piece_id)
    seed_text = str(seed)
    if not piece_text or not seed_text:
        raise ContractError(f"{context}: piece_id and seed must be non-empty")
    return piece_text, seed_text


def _safe_manifest_path(manifest_path: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ContractError(f"manifest MIDI path must be relative: {relative_path}")
    root = manifest_path.parent.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"manifest MIDI path escapes its directory: {relative_path}") from exc
    if not resolved.is_file():
        raise ContractError(f"manifest MIDI path is not a file: {relative_path}")
    return resolved


def _selected_system_ids(
    manifest_system_ids: list[Any], requested_system_ids: Iterable[str] | None
) -> list[str]:
    if not manifest_system_ids or not all(
        isinstance(value, str) and value for value in manifest_system_ids
    ):
        raise ContractError("manifest system_ids must be a non-empty string list")
    system_ids = list(manifest_system_ids)
    if len(set(system_ids)) != len(system_ids):
        raise ContractError("manifest system_ids contains duplicates")
    if requested_system_ids is None:
        return system_ids
    requested = list(requested_system_ids)
    if not requested:
        raise ContractError("--system-id selection cannot be empty")
    if len(set(requested)) != len(requested):
        raise ContractError("--system-id contains duplicates")
    unknown = [system_id for system_id in requested if system_id not in system_ids]
    if unknown:
        raise ContractError(f"unknown --system-id values: {unknown}")
    return requested


def validate_manifest(
    manifest_path: Path, requested_system_ids: Iterable[str] | None = None
) -> tuple[dict[str, Any], list[str], dict[str, list[dict[str, Any]]]]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ContractError(f"common-valid manifest not found: {manifest_path}")
    payload = _load_json(manifest_path)
    if payload.get("schema_version") != 1:
        raise ContractError("manifest schema_version must be 1")
    if payload.get("key_fields") != ["piece_id", "seed"]:
        raise ContractError("manifest key_fields must be ['piece_id', 'seed']")
    system_ids = payload.get("system_ids")
    if not isinstance(system_ids, list):
        raise ContractError("manifest system_ids must be a list")
    selected = _selected_system_ids(system_ids, requested_system_ids)

    common_keys_raw = payload.get("common_valid_keys")
    if not isinstance(common_keys_raw, list) or not common_keys_raw:
        raise ContractError("manifest common_valid_keys must be a non-empty list")
    expected_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(common_keys_raw):
        if not isinstance(row, dict):
            raise ContractError(f"common_valid_keys[{index}] must be an object")
        key = _canonical_key(
            row.get("piece_id"), row.get("seed"), context=f"common_valid_keys[{index}]"
        )
        if key in expected_keys:
            raise ContractError(f"duplicate common-valid key: {key}")
        expected_keys.add(key)
    declared_count = payload.get("common_valid_key_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        raise ContractError("common_valid_key_count must be an integer")
    if declared_count != len(expected_keys):
        raise ContractError(
            "common_valid_key_count does not match common_valid_keys: "
            f"{declared_count!r} != {len(expected_keys)}"
        )

    trials = payload.get("trials")
    if not isinstance(trials, list):
        raise ContractError("manifest trials must be a list")
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_by_system: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for index, row in enumerate(trials):
        context = f"trials[{index}]"
        if not isinstance(row, dict):
            raise ContractError(f"{context} must be an object")
        system_id = _require_text(row, "system_id", context=context)
        if system_id not in system_ids:
            raise ContractError(f"{context}: undeclared system_id {system_id!r}")
        key = _canonical_key(row.get("piece_id"), row.get("seed"), context=context)
        if key in seen_by_system[system_id]:
            raise ContractError(f"{context}: duplicate key {key} for {system_id}")
        seen_by_system[system_id].add(key)
        basename = _require_text(row, "basename", context=context)
        relative_path = _require_text(row, "common_generated_midi", context=context)
        expected_hash = _require_text(row, "generated_sha256", context=context)
        if SHA256_RE.fullmatch(expected_hash) is None:
            raise ContractError(f"{context}: generated_sha256 must be lowercase SHA256")
        if Path(relative_path).name != basename:
            raise ContractError(
                f"{context}: basename {basename!r} does not match path {relative_path!r}"
            )
        normalized = dict(row)
        normalized["_key"] = key
        by_system[system_id].append(normalized)

    for system_id in system_ids:
        actual_keys = seen_by_system.get(system_id, set())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ContractError(
                f"system {system_id!r} does not match common-valid key grid; "
                f"missing={missing}, extra={extra}"
            )

    for system_id in selected:
        by_system[system_id].sort(key=lambda row: row["_key"])
    return payload, selected, dict(by_system)


def _track_only_midi(instrument: pretty_midi.Instrument) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    copied = pretty_midi.Instrument(
        program=instrument.program,
        is_drum=instrument.is_drum,
        name=instrument.name,
    )
    copied.notes = [
        pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=note.start,
            end=note.end,
        )
        for note in instrument.notes
    ]
    midi.instruments.append(copied)
    return midi


def _validate_midi_timing(midi: pretty_midi.PrettyMIDI, path: Path) -> None:
    tempo_times, tempi = midi.get_tempo_changes()
    if len(tempi) == 0 or len(tempo_times) != len(tempi):
        raise ContractError(f"{path}: missing or invalid tempo map")
    if abs(float(tempo_times[0])) > TIME_TOLERANCE_SECONDS:
        raise ContractError(f"{path}: tempo map must begin at time zero")
    if any(abs(float(tempo) - BPM) > TEMPO_TOLERANCE for tempo in tempi):
        raise ContractError(f"{path}: MIDI must use constant {BPM} BPM")
    if not midi.time_signature_changes:
        raise ContractError(f"{path}: MIDI must explicitly declare 4/4")
    first = midi.time_signature_changes[0]
    if abs(float(first.time)) > TIME_TOLERANCE_SECONDS:
        raise ContractError(f"{path}: time signature map must begin at time zero")
    if any(
        change.numerator != 4 or change.denominator != 4
        for change in midi.time_signature_changes
    ):
        raise ContractError(f"{path}: MIDI must use 4/4 throughout")
    if midi.get_end_time() > 12.0 + TIME_TOLERANCE_SECONDS:
        raise ContractError(f"{path}: MIDI exceeds the fixed 12.0-second horizon")


def midi_to_measures(
    midi_path: Path,
    *,
    midi_loader: Callable[[str], pretty_midi.PrettyMIDI] = pretty_midi.PrettyMIDI,
    converter_factory: Callable[..., MidiConverter] = MidiConverter,
) -> list[np.ndarray]:
    try:
        midi = midi_loader(str(midi_path))
    except Exception as exc:
        raise ContractError(f"cannot parse MIDI {midi_path}: {exc}") from exc
    _validate_midi_timing(midi, midi_path)
    if len(midi.instruments) != 2:
        raise ContractError(
            f"{midi_path}: expected exactly two physical instrument tracks, "
            f"found {len(midi.instruments)}"
        )
    names = [instrument.name for instrument in midi.instruments]
    if names != ["Melody", "Accompaniment"]:
        raise ContractError(
            f"{midi_path}: tracks must be named and ordered exactly "
            f"['Melody', 'Accompaniment']; found {names!r}"
        )
    if any(instrument.is_drum for instrument in midi.instruments):
        raise ContractError(f"{midi_path}: Melody and Accompaniment must be non-drum tracks")

    for instrument in midi.instruments:
        for note in instrument.notes:
            values = (float(note.start), float(note.end))
            if not all(math.isfinite(value) for value in values):
                raise ContractError(f"{midi_path}: note has non-finite timing")
            if note.start < 0 or note.end <= note.start:
                raise ContractError(f"{midi_path}: note has invalid timing")
            if note.end > 12.0 + TIME_TOLERANCE_SECONDS:
                raise ContractError(f"{midi_path}: note exceeds the fixed 12.0-second horizon")
            if not 21 <= int(note.pitch) <= 108:
                raise ContractError(f"{midi_path}: note pitch {note.pitch} is outside piano range")

    converter = converter_factory(ticks_per_beat=STEPS_PER_BEAT)
    rolls: list[np.ndarray] = []
    for instrument in midi.instruments:
        notes, _ = converter.midi_to_notes(_track_only_midi(instrument))
        for note in notes:
            start = int(note["tick"])
            end = start + int(note["duration"])
            if start < 0 or start >= HORIZON_TICKS or end > HORIZON_TICKS:
                raise ContractError(
                    f"{midi_path}: quantized note [{start}, {end}) is outside "
                    f"the 96-tick horizon"
                )
        roll = np.asarray(converter.notes_to_pianoroll(notes, max_tick=HORIZON_TICKS))
        if roll.shape != (2, 88, HORIZON_TICKS):
            raise ContractError(f"{midi_path}: unexpected quantized roll shape {roll.shape}")
        rolls.append((roll > 0).astype(np.uint8))

    combined = np.concatenate([rolls[0], rolls[1]], axis=0)
    measures = [
        combined[:, :, index * MEASURE_TICKS : (index + 1) * MEASURE_TICKS].copy()
        for index in range(NUM_MEASURES)
    ]
    if len(measures) != NUM_MEASURES or any(
        measure.shape != (4, 88, MEASURE_TICKS) for measure in measures
    ):
        raise ContractError(f"{midi_path}: failed to construct six 16-step measures")
    return measures


def build_scoring_sequence(
    tokenizer: Any, measures: list[np.ndarray]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    metadata = {
        "time_signature_idx": TIME_SIGNATURE_INDEX,
        "bpm": BPM,
        "num_measures": NUM_MEASURES,
        "is_continuation": True,
    }
    input_ids, labels = tokenizer.build_training_sequence(
        measures,
        metadata,
        add_bos=True,
        timesteps_per_beat=STEPS_PER_BEAT,
        include_melody_loss=False,
    )
    input_ids = torch.as_tensor(input_ids, dtype=torch.long).reshape(-1)
    labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    if input_ids.shape != labels.shape:
        raise ContractError("tokenizer input_ids and labels have different lengths")
    sequence_length = int(input_ids.numel())
    if sequence_length < 2:
        raise ContractError("tokenizer sequence must contain at least two tokens")
    if sequence_length > MAX_SEQUENCE_LENGTH:
        raise ContractError(
            f"tokenizer sequence length {sequence_length} exceeds {MAX_SEQUENCE_LENGTH}"
        )
    pad_token_id = int(tokenizer.vocab.pad_token_id)
    if pad_token_id != 258:
        raise ContractError(f"continuation tokenizer pad_token_id must be 258, got {pad_token_id}")
    labels = labels.clone()
    labels[labels == pad_token_id] = IGNORE_INDEX
    total_tokens = int((labels[1:] != IGNORE_INDEX).sum().item())
    if total_tokens <= 0:
        raise ContractError("tokenizer produced no causal accompaniment targets")
    return input_ids, labels, total_tokens


def score_sequence(
    model: Any,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str,
) -> tuple[float, float, int]:
    model_inputs = input_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(model_inputs)
    with torch.no_grad():
        outputs = model(input_ids=model_inputs, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else None
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ContractError("continuation model did not return rank-3 logits")
    if logits.shape[:2] != model_inputs.shape:
        raise ContractError(
            f"model logits shape {tuple(logits.shape)} does not match inputs "
            f"{tuple(model_inputs.shape)}"
        )
    shifted_labels = labels[1:].unsqueeze(0).to(device)
    total_tokens = int((shifted_labels != IGNORE_INDEX).sum().item())
    if total_tokens <= 0:
        raise ContractError("causal shift leaves no accompaniment targets")
    shifted_logits = logits[:, :-1, :].float()
    total_nll_tensor = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    total_nll = float(total_nll_tensor.item())
    avg_nll = total_nll / total_tokens
    if (
        not math.isfinite(total_nll)
        or not math.isfinite(avg_nll)
        or total_nll < 0
        or avg_nll < 0
    ):
        raise ContractError("continuation NLL is non-finite or negative")
    return total_nll, avg_nll, total_tokens


def _git_code_identity(repository_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repository_root, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        status = "unavailable"
    script_path = Path(__file__).resolve()
    return {
        "repository_root": str(repository_root.resolve()),
        "git_commit": commit,
        "git_dirty": bool(status),
        "script_path": str(script_path),
        "script_sha256": file_sha256(script_path),
    }


def _metric_contract() -> dict[str, Any]:
    return {
        "metric": "raw continuation-model teacher-forced NLL",
        "aggregation": "none",
        "evaluator_scope": "cross-system generated-music quality; not online latency",
        "scope": "source beats [8, 32), shifted to MIDI beat 0",
        "source_beat_start_inclusive": 8,
        "source_beat_end_exclusive": 32,
        "midi_window_shifted_to_zero": True,
        "post_join_beats": POST_JOIN_BEATS,
        "model_ticks": HORIZON_TICKS,
        "steps_per_beat": STEPS_PER_BEAT,
        "beats_per_measure": BEATS_PER_MEASURE,
        "num_measures": NUM_MEASURES,
        "bpm": BPM,
        "time_signature": "4/4",
        "time_signature_idx": TIME_SIGNATURE_INDEX,
        "channel_order": ["mel_sus", "mel_on", "acc_sus", "acc_on"],
        "sequence_builder": "lekai_continuation_model.PianoMusicTokenizer.build_training_sequence",
        "is_continuation": True,
        "pad_token_id": 258,
        "ignore_index": IGNORE_INDEX,
        "causal_shift": "logits[:, :-1] versus labels[1:]",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "cross_entropy_dtype": "float32",
    }


def score_manifest(
    *,
    common_valid_manifest: Path,
    continuation_checkpoint: Path,
    system_ids: Iterable[str] | None,
    device: str,
    fp16: bool,
    adapter_loader: Callable[..., Any] = PianoContinuationAdapter.from_checkpoint,
    midi_loader: Callable[[str], pretty_midi.PrettyMIDI] = pretty_midi.PrettyMIDI,
    converter_factory: Callable[..., MidiConverter] = MidiConverter,
    code_identity_loader: Callable[[Path], dict[str, Any]] = _git_code_identity,
) -> dict[str, Any]:
    manifest_path = common_valid_manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise ContractError(f"common-valid manifest not found: {manifest_path}")
    manifest_sha256 = file_sha256(manifest_path)
    _manifest, selected, by_system = validate_manifest(manifest_path, system_ids)
    checkpoint_path = continuation_checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ContractError(f"continuation checkpoint not found: {checkpoint_path}")
    checkpoint_sha256 = file_sha256(checkpoint_path)

    prepared: dict[str, list[tuple[dict[str, Any], Path, str, list[np.ndarray]]]] = {}
    for system_id in selected:
        prepared[system_id] = []
        for row in by_system[system_id]:
            relative_path = row["common_generated_midi"]
            midi_path = _safe_manifest_path(manifest_path, relative_path)
            actual_hash = file_sha256(midi_path)
            if actual_hash != row["generated_sha256"]:
                raise ContractError(
                    f"{relative_path}: generated SHA256 mismatch; "
                    f"expected {row['generated_sha256']}, got {actual_hash}"
                )
            measures = midi_to_measures(
                midi_path,
                midi_loader=midi_loader,
                converter_factory=converter_factory,
            )
            if file_sha256(midi_path) != actual_hash:
                raise ContractError(f"{relative_path}: MIDI changed while being read")
            prepared[system_id].append((row, midi_path, actual_hash, measures))

    dtype = torch.float16 if fp16 else torch.float32
    adapter = adapter_loader(
        str(checkpoint_path), device=device, dtype=dtype, use_cache=False
    )
    model = adapter.model
    tokenizer = adapter.tokenizer
    if hasattr(model, "eval"):
        model.eval()

    output_systems: dict[str, Any] = {}
    total_trials = 0
    for system_id in selected:
        scored_trials = []
        for row, midi_path, actual_hash, measures in prepared[system_id]:
            input_ids, labels, expected_tokens = build_scoring_sequence(tokenizer, measures)
            total_nll, avg_nll, total_tokens = score_sequence(
                model, input_ids, labels, device=device
            )
            if total_tokens != expected_tokens:
                raise ContractError("causal target count changed between tokenization and scoring")
            scored_trials.append(
                {
                    "system_id": system_id,
                    "piece_id": row["piece_id"],
                    "seed": row["seed"],
                    "basename": row["basename"],
                    "generated_midi": row["common_generated_midi"],
                    "generated_sha256": actual_hash,
                    "sequence_tokens": int(input_ids.numel()),
                    "total_tokens": total_tokens,
                    "total_nll": total_nll,
                    "avg_nll": avg_nll,
                }
            )
        output_systems[system_id] = {
            "counts": {
                "trials": len(scored_trials),
                "scored": len(scored_trials),
                "errors": 0,
            },
            "errors": [],
            "trials": scored_trials,
        }
        total_trials += len(scored_trials)

    if file_sha256(manifest_path) != manifest_sha256:
        raise ContractError("common-valid manifest changed while scoring")
    if file_sha256(checkpoint_path) != checkpoint_sha256:
        raise ContractError("continuation checkpoint changed while scoring")

    repository_root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": SCHEMA_VERSION,
        "common_valid_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "code_identity": code_identity_loader(repository_root),
        "metric_contract": _metric_contract(),
        "system_ids": selected,
        "counts": {
            "systems": len(selected),
            "trials": total_trials,
            "scored": total_trials,
            "errors": 0,
        },
        "errors": [],
        "systems": output_systems,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score matched generated MIDI with raw continuation-model NLL."
    )
    parser.add_argument("--common-valid-manifest", type=Path, required=True)
    parser.add_argument("--continuation-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--system-id",
        action="append",
        dest="system_ids",
        help="System to score; repeat to select multiple systems. Default: all.",
    )
    parser.add_argument("--device", default="cuda")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--fp16", dest="fp16", action="store_true")
    precision.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = score_manifest(
            common_valid_manifest=args.common_valid_manifest,
            continuation_checkpoint=args.continuation_checkpoint,
            system_ids=args.system_ids,
            device=args.device,
            fp16=args.fp16,
        )
        atomic_write_json(args.output_json, payload)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
