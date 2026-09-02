#!/usr/bin/env python
"""Run matched offline BEAT evaluation with an eight-beat GT ACC prefix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import mido
import pretty_midi
import torch

from streammuse.infrastructure.inference.lekai_continuation_model.inference_adapter import (
    PianoContinuationAdapter,
)
from streammuse.infrastructure.inference.lekai_continuation_model.Token2Midi import (
    MidiConverter,
)


SYSTEM_ID = "beat_gt_offline"
WINDOW_BEATS = 32
PREFIX_BEATS = 8
GENERATED_BEATS = WINDOW_BEATS - PREFIX_BEATS
STEPS_PER_BEAT = 4
MEASURE_STEPS = 16
WINDOW_MEASURES = 8
WINDOW_STEPS = WINDOW_BEATS * STEPS_PER_BEAT
POSTJOIN_STEPS = GENERATED_BEATS * STEPS_PER_BEAT
TIME_SIGNATURE = "4/4"
TIME_SIGNATURE_INDEX = 0
BPM = 120.0
TEMPERATURE = 1.05
TOP_P = 0.98
TOP_K = 0
REPETITION_PENALTY = 1.0
CSV_FIELDS = (
    "piece_id",
    "seed",
    "system_id",
    "status",
    "failure_reason",
    "trial_dir",
    "source_npz",
    "source_npz_sha256",
    "full_generated_midi",
    "full_gt_midi",
    "postjoin_generated_midi",
    "postjoin_gt_midi",
    "trial_manifest",
)


@dataclass(frozen=True)
class CohortSample:
    order: int
    piece_id: str
    source_npz: Path
    source_npz_sha256: str


@dataclass(frozen=True)
class PreparedWindow:
    measures: list[np.ndarray]
    metadata: dict[str, Any]
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class BatchResult:
    exit_code: int
    complete: int
    failed: int
    csv_path: Path
    summary_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the continuation checkpoint on cohort source NPZ files with "
            "eight beats of GT accompaniment context."
        )
    )
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--continuation-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--limit-pieces",
        type=int,
        default=0,
        help="Optional smoke limit; 0 runs every cohort sample.",
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique integers")
    return seeds


def prepare_output_root(path: Path) -> Path:
    """Create a new output root or accept an existing empty directory only."""
    output_root = path.expanduser().resolve()
    if output_root.exists():
        if not output_root.is_dir():
            raise FileExistsError(f"output root is not a directory: {output_root}")
        if next(output_root.iterdir(), None) is not None:
            raise FileExistsError(f"output root is not empty: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def load_cohort_manifest(path: Path) -> list[CohortSample]:
    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cohort manifest must contain a non-empty top-level samples list")
    samples: list[CohortSample] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"samples[{index - 1}] must be an object")
        piece_id = str(row.get("piece_id", "")).strip()
        source_value = str(row.get("source_npz", "")).strip()
        expected_hash = str(row.get("source_npz_sha256", "")).strip().lower()
        if not piece_id or not source_value or len(expected_hash) != 64:
            raise ValueError(
                f"samples[{index - 1}] requires piece_id, source_npz, "
                "and source_npz_sha256"
            )
        if piece_id in seen:
            raise ValueError(f"duplicate piece_id in cohort manifest: {piece_id}")
        seen.add(piece_id)
        source_path = Path(source_value)
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        samples.append(
            CohortSample(
                order=int(row.get("order", index)),
                piece_id=piece_id,
                source_npz=source_path.resolve(),
                source_npz_sha256=expected_hash,
            )
        )
    return samples


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _scalar(value.reshape(-1)[0])
        return [_scalar(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_time_signature_index(value: Any) -> None:
    normalised = _scalar(value)
    values = normalised if isinstance(normalised, list) else [normalised]
    if not values or any(item != TIME_SIGNATURE_INDEX for item in values):
        raise ValueError(
            "source NPZ time_signature_idx must be exactly the checkpoint-aligned "
            "4/4 index 0"
        )


def load_prepared_window(sample: CohortSample) -> tuple[PreparedWindow, str]:
    if not sample.source_npz.is_file():
        raise FileNotFoundError(f"source NPZ not found: {sample.source_npz}")
    actual_hash = file_sha256(sample.source_npz)
    if actual_hash != sample.source_npz_sha256:
        raise ValueError(
            f"source NPZ hash mismatch: expected {sample.source_npz_sha256}, "
            f"got {actual_hash}"
        )
    with np.load(sample.source_npz, allow_pickle=True) as payload:
        if "metadata" not in payload:
            raise ValueError("source NPZ is missing metadata")
        source_metadata = payload["metadata"].item()
        if not isinstance(source_metadata, dict):
            raise ValueError("source NPZ metadata must be a mapping")
        source_time_signature = _scalar(source_metadata.get("time_signature"))
        if source_time_signature != TIME_SIGNATURE:
            raise ValueError(
                "source NPZ time_signature must be exactly the checkpoint-aligned "
                'value "4/4"'
            )
        _validate_time_signature_index(source_metadata.get("time_signature_idx"))
        if "num_channels" in source_metadata and int(
            _scalar(source_metadata["num_channels"])
        ) != 4:
            raise ValueError("source NPZ metadata num_channels must be 4")
        try:
            num_measures = int(_scalar(source_metadata["num_measures"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("source NPZ metadata num_measures is invalid") from exc
        if num_measures < WINDOW_MEASURES:
            raise ValueError(
                f"source NPZ is too short: requires {WINDOW_MEASURES} measures, "
                f"got {num_measures}"
            )
        measures = []
        for measure_index in range(WINDOW_MEASURES):
            key = f"measure_{measure_index}"
            if key not in payload:
                raise ValueError(f"source NPZ is missing {key}")
            measure = np.asarray(payload[key])
            if measure.shape != (4, 88, MEASURE_STEPS):
                raise ValueError(
                    f"{key} shape must be (4, 88, {MEASURE_STEPS}), "
                    f"got {measure.shape}"
                )
            measures.append(measure.copy())

    metadata = dict(source_metadata)
    metadata.update(
        {
            "bpm": BPM,
            "num_channels": 4,
            "num_measures": WINDOW_MEASURES,
            "time_signature": TIME_SIGNATURE,
            "time_signature_idx": TIME_SIGNATURE_INDEX,
        }
    )
    return PreparedWindow(measures, metadata, dict(source_metadata)), actual_hash


def validate_schedule(prepared: dict[str, Any]) -> dict[str, int]:
    actions = [step.action for step in prepared["schedule"]]
    accompaniment_actions = [
        action for action in actions if action in {"inject_gt", "generate"}
    ]
    expected = ["inject_gt"] * PREFIX_BEATS + ["generate"] * GENERATED_BEATS
    if accompaniment_actions != expected:
        raise ValueError(
            "generation schedule must cover beats 0..31 with 8 GT injections "
            "followed by 24 generations"
        )
    if len(prepared["mel_beats"]) != WINDOW_BEATS:
        raise ValueError("generation schedule must contain 32 melody beats")
    if len(prepared["acc_beats_gt"]) != WINDOW_BEATS:
        raise ValueError("generation schedule must contain 32 GT accompaniment beats")
    return {
        "window_beats": WINDOW_BEATS,
        "gt_inject_beats": accompaniment_actions.count("inject_gt"),
        "generated_beats": accompaniment_actions.count("generate"),
        "first_beat": 0,
        "last_beat_inclusive": WINDOW_BEATS - 1,
    }


def _validate_decoded_roll(
    tokenizer: Any,
    beats: list[Any],
    *,
    track_marker_id: int,
    expected_steps: int,
    label: str,
) -> None:
    roll = tokenizer.decode_beats_to_pianoroll(
        beats, track_marker_id=track_marker_id
    )
    if roll.shape != (2, 88, expected_steps):
        raise ValueError(
            f"{label} decoded shape must be (2, 88, {expected_steps}), got {roll.shape}"
        )


def _validate_midi(path: Path, *, expected_steps: int) -> dict[str, Any]:
    midi = pretty_midi.PrettyMIDI(str(path))
    expected_names = ["Melody", "Accompaniment"]
    physical_names = []
    for track in mido.MidiFile(path).tracks:
        name = next(
            (
                message.name
                for message in track
                if message.is_meta and message.type == "track_name"
            ),
            "",
        )
        if name in expected_names:
            physical_names.append(name)
    if physical_names != expected_names:
        raise ValueError(
            f"{path.name} physical track names must be {expected_names}, "
            f"got {physical_names}"
        )
    max_seconds = expected_steps * 60.0 / BPM / STEPS_PER_BEAT
    note_count = 0
    track_note_counts: dict[str, int] = {name: 0 for name in expected_names}
    latest_end = 0.0
    for instrument in midi.instruments:
        track_note_counts[instrument.name] = len(instrument.notes)
        for note in instrument.notes:
            note_count += 1
            latest_end = max(latest_end, note.end)
            if note.start < -1e-7 or note.end <= note.start:
                raise ValueError(f"{path.name} contains an invalid note interval")
            if note.end > max_seconds + 1e-6:
                raise ValueError(f"{path.name} contains notes outside its requested window")
    return {
        "track_names": physical_names,
        "track_note_counts": track_note_counts,
        "note_count": note_count,
        "latest_end_seconds": latest_end,
        "window_steps": expected_steps,
    }


def ensure_named_midi_tracks(path: Path, *, bpm: float = BPM) -> None:
    """Ensure empty logical tracks survive pretty_midi serialization."""
    source = pretty_midi.PrettyMIDI(str(path))
    expected_names = ("Melody", "Accompaniment")
    by_name: dict[str, pretty_midi.Instrument] = {}
    for instrument in source.instruments:
        if instrument.name not in expected_names or instrument.name in by_name:
            raise ValueError(f"unexpected MIDI track layout in {path.name}")
        by_name[instrument.name] = instrument
    rebuilt = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for name in expected_names:
        instrument = by_name.get(name)
        if instrument is None:
            instrument = pretty_midi.Instrument(program=0, name=name)
        if not instrument.notes and not instrument.control_changes:
            instrument.control_changes.append(
                pretty_midi.ControlChange(number=123, value=0, time=0.0)
            )
        rebuilt.instruments.append(instrument)
    rebuilt.write(str(path))


def crop_midi_window(
    source_path: Path,
    output_path: Path,
    *,
    start_step: int,
    end_step: int,
    bpm: float = BPM,
) -> None:
    """Crop a generated MIDI in time, clipping boundary-crossing notes."""
    if not 0 <= start_step < end_step:
        raise ValueError("MIDI crop requires 0 <= start_step < end_step")
    source = pretty_midi.PrettyMIDI(str(source_path))
    if [instrument.name for instrument in source.instruments] != [
        "Melody",
        "Accompaniment",
    ]:
        raise ValueError("source MIDI must contain Melody and Accompaniment tracks")
    seconds_per_step = 60.0 / bpm / STEPS_PER_BEAT
    start_seconds = start_step * seconds_per_step
    end_seconds = end_step * seconds_per_step
    cropped = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for source_instrument in source.instruments:
        target = pretty_midi.Instrument(
            program=source_instrument.program,
            is_drum=source_instrument.is_drum,
            name=source_instrument.name,
        )
        for note in source_instrument.notes:
            if note.end <= start_seconds or note.start >= end_seconds:
                continue
            clipped_start = max(note.start, start_seconds) - start_seconds
            clipped_end = min(note.end, end_seconds) - start_seconds
            if clipped_end > clipped_start:
                target.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=clipped_start,
                        end=clipped_end,
                    )
                )
        if not target.notes:
            # pretty_midi omits completely event-free instruments on write.
            target.control_changes.append(
                pretty_midi.ControlChange(number=123, value=0, time=0.0)
            )
        cropped.instruments.append(target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.write(str(output_path))


def _git_identity(repository_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=repository_root, text=True
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        dirty = True
    return {"git_commit": commit, "git_dirty": dirty}


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"continuation checkpoint not found: {resolved}")
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _failure_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def run_batch(
    args: argparse.Namespace,
    *,
    adapter_loader: Callable[..., Any] = PianoContinuationAdapter.from_checkpoint,
) -> BatchResult:
    repository_root = Path(__file__).resolve().parents[1]
    output_root = prepare_output_root(args.output_root)
    manifest_path = args.cohort_manifest.expanduser().resolve()
    samples = load_cohort_manifest(manifest_path)
    if args.limit_pieces < 0:
        raise ValueError("--limit-pieces cannot be negative")
    if args.limit_pieces:
        samples = samples[: args.limit_pieces]
    seeds = parse_seeds(args.seeds)
    checkpoint = _identity(args.continuation_checkpoint)
    code = _git_identity(repository_root)
    frozen_settings = {
        "system_id": SYSTEM_ID,
        "cohort_manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
        },
        "continuation_checkpoint": checkpoint,
        "code": {
            **code,
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
        },
        "conditioning": {
            "bpm": BPM,
            "steps_per_beat": STEPS_PER_BEAT,
            "time_signature": TIME_SIGNATURE,
            "time_signature_index": TIME_SIGNATURE_INDEX,
        },
        "window": {
            "beats": [0, WINDOW_BEATS],
            "steps": [0, WINDOW_STEPS],
            "gt_accompaniment_prefix_beats": [0, PREFIX_BEATS],
            "generated_accompaniment_beats": [PREFIX_BEATS, WINDOW_BEATS],
            "postjoin_outputs_shifted_to_zero": True,
        },
        "decoding": {
            "seeds": seeds,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
        },
        "track_semantics": {
            "track_0": "GT melody",
            "track_1_full_generated": (
                "GT accompaniment on beats [0,8), generated accompaniment on [8,32)"
            ),
            "track_1_gt": "GT accompaniment",
        },
    }
    _atomic_write_json(output_root / "frozen_settings.json", frozen_settings)

    dtype = torch.float16 if args.fp16 else torch.float32
    csv_path = output_root / "run_manifest.csv"
    try:
        adapter = adapter_loader(
            checkpoint["path"], device=args.device, dtype=dtype, use_cache=True
        )
    except Exception as exc:
        rows = []
        reason = _failure_reason(exc)
        for sample in samples:
            for seed in seeds:
                trial_dir = (
                    output_root / f"{sample.order:03d}_{sample.piece_id}" / f"seed{seed}"
                )
                trial_dir.mkdir(parents=True, exist_ok=True)
                trial_manifest_path = trial_dir / "trial_manifest.json"
                row = {
                    "piece_id": sample.piece_id,
                    "seed": seed,
                    "system_id": SYSTEM_ID,
                    "status": "failed",
                    "failure_reason": reason,
                    "trial_dir": _relative(trial_dir, output_root),
                    "source_npz": str(sample.source_npz),
                    "source_npz_sha256": "",
                    "full_generated_midi": "",
                    "full_gt_midi": "",
                    "postjoin_generated_midi": "",
                    "postjoin_gt_midi": "",
                    "trial_manifest": _relative(trial_manifest_path, output_root),
                }
                _atomic_write_json(
                    trial_manifest_path,
                    {
                        "piece_id": sample.piece_id,
                        "seed": seed,
                        "system_id": SYSTEM_ID,
                        "status": "failed",
                        "failure_reason": reason,
                        "source_npz": str(sample.source_npz),
                        "source_npz_expected_sha256": sample.source_npz_sha256,
                        "frozen_settings": frozen_settings,
                    },
                )
                rows.append(row)
                _atomic_write_csv(csv_path, rows)
        summary_path = output_root / "run_summary.json"
        _atomic_write_json(
            summary_path,
            {
                "system_id": SYSTEM_ID,
                "status": "completed_with_failures",
                "planned_trials": len(rows),
                "complete_trials": 0,
                "failed_trials": len(rows),
                "model_load_attempts": 1,
                "model_loads": 0,
                "run_manifest_csv": csv_path.name,
                "frozen_settings": frozen_settings,
            },
        )
        return BatchResult(1, 0, len(rows), csv_path, summary_path)
    tokenizer = adapter.tokenizer
    converter = MidiConverter(tokenizer)
    rows: list[dict[str, Any]] = []
    started_batch = time.perf_counter()

    for sample in samples:
        for seed in seeds:
            trial_dir = output_root / f"{sample.order:03d}_{sample.piece_id}" / f"seed{seed}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_manifest_path = trial_dir / "trial_manifest.json"
            row = {
                "piece_id": sample.piece_id,
                "seed": seed,
                "system_id": SYSTEM_ID,
                "status": "failed",
                "failure_reason": "trial did not complete",
                "trial_dir": _relative(trial_dir, output_root),
                "source_npz": str(sample.source_npz),
                "source_npz_sha256": "",
                "full_generated_midi": "",
                "full_gt_midi": "",
                "postjoin_generated_midi": "",
                "postjoin_gt_midi": "",
                "trial_manifest": _relative(trial_manifest_path, output_root),
            }
            trial_record: dict[str, Any] = {
                "piece_id": sample.piece_id,
                "seed": seed,
                "system_id": SYSTEM_ID,
                "status": "failed",
                "failure_reason": row["failure_reason"],
                "source_npz": str(sample.source_npz),
                "source_npz_expected_sha256": sample.source_npz_sha256,
                "frozen_settings": frozen_settings,
            }
            started_trial = time.perf_counter()
            try:
                window, source_hash = load_prepared_window(sample)
                prepared = tokenizer.build_generation_schedule(
                    window.measures,
                    window.metadata,
                    gt_prefix_beats=PREFIX_BEATS,
                    timesteps_per_beat=STEPS_PER_BEAT,
                )
                schedule_counts = validate_schedule(prepared)
                generator = torch.Generator(device=args.device)
                generator.manual_seed(seed)
                acc_beats, generated_tokens = adapter.wrapper.generate_accompaniment(
                    initial_tokens=prepared["initial_tokens"],
                    schedule=prepared["schedule"],
                    vocab=prepared.get("vocab", tokenizer.vocab),
                    device=args.device,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    verbose=False,
                    generator=generator,
                )
                if len(acc_beats) != WINDOW_BEATS:
                    raise ValueError(
                        f"model returned {len(acc_beats)} accompaniment beats; "
                        f"expected {WINDOW_BEATS}"
                    )
                _validate_decoded_roll(
                    tokenizer,
                    prepared["mel_beats"],
                    track_marker_id=tokenizer.vocab.track_marker_mel,
                    expected_steps=WINDOW_STEPS,
                    label="full melody",
                )
                _validate_decoded_roll(
                    tokenizer,
                    acc_beats,
                    track_marker_id=tokenizer.vocab.track_marker_acc,
                    expected_steps=WINDOW_STEPS,
                    label="full generated accompaniment",
                )
                _validate_decoded_roll(
                    tokenizer,
                    prepared["acc_beats_gt"],
                    track_marker_id=tokenizer.vocab.track_marker_acc,
                    expected_steps=WINDOW_STEPS,
                    label="full GT accompaniment",
                )

                output_paths = {
                    "full_generated_midi": trial_dir / "full_generated.mid",
                    "full_gt_midi": trial_dir / "full_gt.mid",
                    "postjoin_generated_midi": trial_dir / "postjoin_generated.mid",
                    "postjoin_gt_midi": trial_dir / "postjoin_gt.mid",
                }
                converter.beats_to_midi(
                    prepared["mel_beats"],
                    acc_beats,
                    tempo=BPM,
                    save_path=str(output_paths["full_generated_midi"]),
                )
                converter.beats_to_midi(
                    prepared["mel_beats"],
                    prepared["acc_beats_gt"],
                    tempo=BPM,
                    save_path=str(output_paths["full_gt_midi"]),
                )
                ensure_named_midi_tracks(output_paths["full_generated_midi"])
                ensure_named_midi_tracks(output_paths["full_gt_midi"])
                crop_midi_window(
                    output_paths["full_generated_midi"],
                    output_paths["postjoin_generated_midi"],
                    start_step=PREFIX_BEATS * STEPS_PER_BEAT,
                    end_step=WINDOW_STEPS,
                )
                crop_midi_window(
                    output_paths["full_gt_midi"],
                    output_paths["postjoin_gt_midi"],
                    start_step=PREFIX_BEATS * STEPS_PER_BEAT,
                    end_step=WINDOW_STEPS,
                )
                midi_validation = {
                    key: _validate_midi(
                        value,
                        expected_steps=(
                            WINDOW_STEPS if key.startswith("full_") else POSTJOIN_STEPS
                        ),
                    )
                    for key, value in output_paths.items()
                }
                output_records = {
                    key: {
                        "path": _relative(value, output_root),
                        "sha256": file_sha256(value),
                        **midi_validation[key],
                    }
                    for key, value in output_paths.items()
                }
                row.update(
                    {
                        "status": "complete",
                        "failure_reason": "",
                        "source_npz_sha256": source_hash,
                        **{
                            key: output_records[key]["path"] for key in output_paths
                        },
                    }
                )
                trial_record.update(
                    {
                        "status": "complete",
                        "failure_reason": "",
                        "source_npz_sha256": source_hash,
                        "source_metadata": window.source_metadata,
                        "effective_metadata": window.metadata,
                        "schedule": schedule_counts,
                        "rng": {
                            "type": "torch.Generator",
                            "device": args.device,
                            "seed": int(generator.initial_seed()),
                            "new_generator_per_trial": True,
                        },
                        "generated_token_count": int(generated_tokens.numel()),
                        "outputs": output_records,
                    }
                )
            except Exception as exc:
                reason = _failure_reason(exc)
                row["failure_reason"] = reason
                trial_record["failure_reason"] = reason
            trial_record["elapsed_seconds"] = time.perf_counter() - started_trial
            _atomic_write_json(trial_manifest_path, trial_record)
            rows.append(row)
            _atomic_write_csv(csv_path, rows)

    complete = sum(row["status"] == "complete" for row in rows)
    failed = len(rows) - complete
    summary_path = output_root / "run_summary.json"
    _atomic_write_json(
        summary_path,
        {
            "system_id": SYSTEM_ID,
            "status": "complete" if failed == 0 else "completed_with_failures",
            "planned_trials": len(rows),
            "complete_trials": complete,
            "failed_trials": failed,
            "elapsed_seconds": time.perf_counter() - started_batch,
            "model_load_attempts": 1,
            "model_loads": 1,
            "run_manifest_csv": csv_path.name,
            "frozen_settings": frozen_settings,
        },
    )
    return BatchResult(
        exit_code=1 if failed else 0,
        complete=complete,
        failed=failed,
        csv_path=csv_path,
        summary_path=summary_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_batch(args)
    print(json.dumps(asdict(result), indent=2, default=str))
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
