#!/usr/bin/env python3
"""Pre-freeze and build the 24-clip blinded robustness listening package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Mapping

import mido
import numpy as np
from scipy.signal import resample_poly

from streammuse.experiments.melody_robustness import (
    SEEDS,
    build_listening_selection_manifest,
    build_run_schedule,
    file_sha256,
    read_jsonl,
    validate_campaign_config,
    validate_listening_selection_manifest,
    validate_staged_input_manifest,
    write_canonical_json,
)
from streammuse.experiments.robustness_metrics import Roll, load_midi_roll, write_roll_midi


RENDER_BPM = 120
CLIP_SECONDS = 25
CLIP_BEATS = 50  # 25 seconds at 120 BPM
CLIP_TICKS = CLIP_BEATS * 4
RENDER_SAMPLE_RATE = 44100
FIXED_SYNTH_GAIN = 0.5
GAIN_POLICY = "fixed_pair_gain_with_true_peak_protection_only"
TRUE_PEAK_LIMIT_DBTP = -0.1
TRUE_PEAK_OVERSAMPLE = 4
TRUE_PEAK_IMPLEMENTATION = "scipy_resample_poly_4x_kaiser_8.6"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _song(entry: Mapping[str, Any]) -> str:
    return str(entry.get("song", entry.get("source_stem")))


def _song_analysis_horizons(entries: list[dict[str, Any]]) -> dict[str, int]:
    """Return the single frozen clean analysis horizon for each song."""
    grouped: dict[str, set[int]] = {}
    for entry in entries:
        song = _song(entry)
        raw = entry.get("analysis_end_tick")
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(
                f"{song}: input manifest requires a positive integer analysis_end_tick"
            )
        grouped.setdefault(song, set()).add(raw)
    inconsistent = {song: sorted(values) for song, values in grouped.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"analysis horizon differs across variants: {inconsistent}")
    return {song: next(iter(values)) for song, values in grouped.items()}


def _validate_excerpt(
    *, song: str, start_beat: Any, end_beat: Any, analysis_end_tick: int
) -> None:
    if (
        isinstance(start_beat, bool)
        or not isinstance(start_beat, int)
        or isinstance(end_beat, bool)
        or not isinstance(end_beat, int)
    ):
        raise ValueError(f"{song}: excerpt beats must be integers")
    if start_beat < 0:
        raise ValueError(f"{song}: excerpt start must be non-negative")
    if end_beat != start_beat + CLIP_BEATS:
        raise ValueError(
            f"{song}: excerpt end must equal start + {CLIP_BEATS} model beats"
        )
    if end_beat * 4 > analysis_end_tick:
        raise ValueError(
            f"{song}: excerpt [{start_beat}, {end_beat}) beats exceeds "
            f"analysis_end_tick={analysis_end_tick}"
        )


def _path(entry: Mapping[str, Any], base: Path, *names: str) -> Path:
    containers = [entry]
    for nested in ("paths", "artifacts", "files"):
        if isinstance(entry.get(nested), Mapping):
            containers.append(entry[nested])  # type: ignore[arg-type]
    for container in containers:
        for name in names:
            value = container.get(name)
            if isinstance(value, str):
                candidate = Path(value)
                return (candidate if candidate.is_absolute() else base / candidate).resolve()
            if isinstance(value, Mapping) and value.get("path"):
                candidate = Path(str(value["path"]))
                return (candidate if candidate.is_absolute() else base / candidate).resolve()
    raise KeyError(f"missing artifact path {names}")


def _validated_build_inputs(
    args: argparse.Namespace,
    selection_path: Path,
    selection: dict[str, Any],
) -> tuple[dict[str, Any], str, list[dict[str, Any]], Path, list[dict[str, Any]]]:
    config_path = Path(args.config).resolve()
    config_sha = file_sha256(config_path)
    if config_sha != args.config_sha256:
        raise RuntimeError("campaign config hash mismatch")
    config = _read_json(config_path)
    validate_campaign_config(config)
    checkpoint_path = Path(config["checkpoint"]["path"]).resolve()
    if file_sha256(checkpoint_path) != str(config["checkpoint"]["sha256"]):
        raise RuntimeError("checkpoint hash mismatch with frozen campaign config")

    selection_sha = file_sha256(selection_path)
    if selection_sha != config["listening"].get("selection_manifest_sha256"):
        raise RuntimeError("listening selection hash does not match frozen campaign config")
    configured_selection = Path(
        config["listening"].get("selection_manifest_path", "")
    ).resolve()
    if configured_selection != selection_path.resolve():
        raise RuntimeError("listening selection path does not match frozen campaign config")
    for field in ("render_bpm", "clip_seconds", "clip_count", "gain_policy"):
        if selection.get(field) != config["listening"].get(field):
            raise RuntimeError(f"listening selection {field} differs from campaign config")

    manifest_path = Path(selection["input_manifest_path"]).resolve()
    configured_manifest = Path(config["input_manifest"]["path"]).resolve()
    if manifest_path != configured_manifest:
        raise RuntimeError("selection and campaign config reference different input manifests")
    manifest_sha = file_sha256(manifest_path)
    if manifest_sha != selection.get("input_manifest_sha256"):
        raise RuntimeError("selection input manifest hash mismatch")
    if manifest_sha != config["input_manifest"].get("sha256"):
        raise RuntimeError("campaign input manifest hash mismatch")
    input_manifest = _read_json(manifest_path)
    entries = validate_staged_input_manifest(
        input_manifest, manifest_path=manifest_path, verify_files=True
    )
    validate_listening_selection_manifest(
        selection,
        input_manifest,
        manifest_path=manifest_path,
        verify_files=True,
    )

    schedule_path = Path(args.schedule).resolve()
    if file_sha256(schedule_path) != args.schedule_sha256:
        raise RuntimeError("run schedule hash mismatch")
    schedule = read_jsonl(schedule_path)
    if schedule != build_run_schedule(input_manifest, config):
        raise RuntimeError(
            "run schedule is not the deterministic 160-row schedule rebuilt from "
            "the frozen campaign config and input manifest"
        )
    return config, config_sha, schedule, manifest_path, entries


def _validated_campaign_binding(
    args: argparse.Namespace, config: Mapping[str, Any], config_sha: str
) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    binding_path = output_root / "campaign_binding.json"
    expected = {
        "schema_version": "streammuse.melody_robustness.campaign_binding.v1",
        "qualification": False,
        "campaign_config_path": str(Path(args.config).resolve()),
        "campaign_config_sha256": config_sha,
        "run_schedule_path": str(Path(args.schedule).resolve()),
        "run_schedule_sha256": args.schedule_sha256,
        "input_manifest_path": str(Path(config["input_manifest"]["path"]).resolve()),
        "input_manifest_sha256": str(config["input_manifest"]["sha256"]),
        "checkpoint_path": str(Path(config["checkpoint"]["path"]).resolve()),
        "checkpoint_sha256": str(config["checkpoint"]["sha256"]),
        "code_identity": str(config["code_identity"]),
    }
    if not binding_path.is_file() or _read_json(binding_path) != expected:
        raise RuntimeError("model output root is not bound to this exact formal campaign")
    return {**expected, "campaign_binding_sha256": file_sha256(binding_path)}


def freeze_selection(args: argparse.Namespace) -> None:
    input_path = Path(args.input_manifest).resolve()
    manifest = _read_json(input_path)
    entries = validate_staged_input_manifest(
        manifest, manifest_path=input_path, verify_files=True
    )
    songs = sorted({_song(entry) for entry in entries})
    horizons = _song_analysis_horizons(entries)
    pseed = int(args.perturb_seed)
    sseed = int(args.sample_seed)
    blind_order_seed = int(args.blind_order_seed)
    if pseed not in {int(seed) for seed in SEEDS["perturb"]}:
        raise ValueError(
            f"perturb seed {pseed} is outside the frozen contract: {SEEDS['perturb']}"
        )
    if sseed not in {int(seed) for seed in SEEDS["sample"]}:
        raise ValueError(
            f"sample seed {sseed} is outside the frozen contract: {SEEDS['sample']}"
        )
    if blind_order_seed != int(SEEDS["blind_order"]):
        raise ValueError(
            "blind-order seed must match the frozen contract: "
            f"{SEEDS['blind_order']}"
        )
    if isinstance(args.excerpt_start_beat, bool) or not isinstance(args.excerpt_start_beat, int):
        raise ValueError("--excerpt-start-beat must be an integer")
    starts = {song: args.excerpt_start_beat for song in songs}
    if args.excerpt_starts_json:
        supplied = _read_json(Path(args.excerpt_starts_json))
        unknown = set(map(str, supplied)) - set(songs)
        if unknown:
            raise ValueError(f"excerpt starts name unknown songs: {sorted(unknown)}")
        for song, beat in supplied.items():
            if isinstance(beat, bool) or not isinstance(beat, int):
                raise ValueError(f"{song}: excerpt start must be an integer")
            starts[str(song)] = beat
    for song in songs:
        _validate_excerpt(
            song=song,
            start_beat=starts[song],
            end_beat=starts[song] + CLIP_BEATS,
            analysis_end_tick=horizons[song],
        )
    payload = build_listening_selection_manifest(
        manifest,
        manifest_path=input_path,
        perturb_seed=pseed,
        sample_seed=sseed,
        blind_order_seed=blind_order_seed,
        excerpt_starts=starts,
        verify_files=True,
    )
    digest = write_canonical_json(Path(args.output), payload)
    print(json.dumps({"path": str(Path(args.output).resolve()), "sha256": digest, "clips": 24}))


def _verified_attempt(
    output_root: Path,
    row: Mapping[str, Any],
    expected_binding: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], set[Path]]:
    run_id = str(row["run_id"])
    run_dir = output_root / "runs" / run_id
    pointer = run_dir / "latest_verdict.json"
    if not pointer.is_file():
        raise RuntimeError(f"listening source has no latest verdict: {run_id}")
    verdict = _read_json(pointer)
    if verdict.get("run_id") != run_id or verdict.get("pipeline") != row.get("pipeline"):
        raise RuntimeError(f"listening verdict identity mismatch: {run_id}")
    if verdict.get("content_valid") is not True or verdict.get("operational_valid") is not True:
        raise RuntimeError(f"listening source run is not content/operational valid: {run_id}")
    if expected_binding is not None:
        for field in (
            "campaign_config_sha256",
            "run_schedule_sha256",
            "input_manifest_sha256",
            "checkpoint_sha256",
            "code_identity",
            "campaign_binding_sha256",
        ):
            if verdict.get(field) != expected_binding.get(field):
                raise RuntimeError(f"listening verdict {field} campaign binding mismatch: {run_id}")
    attempt_id = str(verdict.get("attempt_id", ""))
    if re.fullmatch(r"attempt-[0-9]{3}", attempt_id) is None:
        raise RuntimeError(f"invalid listening source attempt ID: {run_id}/{attempt_id}")
    attempt = (run_dir / attempt_id).resolve()
    if not attempt.is_dir() or not attempt.is_relative_to(run_dir.resolve()):
        raise RuntimeError(f"listening source attempt escapes run directory: {run_id}")
    immutable = attempt / "verdict.json"
    if not immutable.is_file() or _read_json(immutable) != verdict:
        raise RuntimeError(f"mutable/immutable verdict mismatch: {run_id}")
    raw_index = verdict.get("artifact_index")
    if not isinstance(raw_index, list) or not raw_index:
        raise RuntimeError(f"listening source has no artifact index: {run_id}")
    indexed: set[Path] = set()
    for record in raw_index:
        if not isinstance(record, Mapping):
            raise RuntimeError(f"malformed artifact index: {run_id}")
        relative = record.get("path")
        digest = record.get("sha256")
        size = record.get("size")
        if not isinstance(relative, str) or not relative or not isinstance(digest, str):
            raise RuntimeError(f"malformed artifact index record: {run_id}")
        path = (attempt / relative).resolve()
        if path in indexed or not path.is_relative_to(attempt):
            raise RuntimeError(f"duplicate/escaping indexed artifact: {run_id}/{relative}")
        if (
            not path.is_file()
            or file_sha256(path) != digest
            or isinstance(size, bool)
            or not isinstance(size, int)
            or path.stat().st_size != size
        ):
            raise RuntimeError(f"missing/corrupt indexed artifact: {run_id}/{relative}")
        indexed.add(path)
    return attempt, verdict, indexed


def _single(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {matches}")
    return matches[0]


def _find_run(
    schedule: list[dict[str, Any]], clip: Mapping[str, Any]
) -> dict[str, Any]:
    candidates = [
        row for row in schedule
        if row["pipeline"] == "rt" and row["song"] == clip["song"]
        and row["condition"] == clip["condition"]
        and row["sample_seed"] == clip["sample_seed"]
        and row.get("perturb_seed") == clip.get("perturb_seed")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"listening selector matched {len(candidates)} runs: {clip}")
    return candidates[0]


def _slice(roll: Roll, start: int, length: int = CLIP_TICKS) -> Roll:
    return Roll(
        end_tick=length,
        sustain=frozenset((tick - start, pitch) for tick, pitch in roll.sustain if start <= tick < start + length),
        onsets=frozenset((tick - start, pitch) for tick, pitch in roll.onsets if start <= tick < start + length),
    )


def _union(left: Roll, right: Roll) -> Roll:
    return Roll(
        end_tick=max(left.end_tick, right.end_tick),
        sustain=frozenset(left.sustain | right.sustain),
        onsets=frozenset(left.onsets | right.onsets),
    )


def _render(
    midi_path: Path, wav_path: Path, *, soundfont: Path, fluidsynth: str,
    sample_rate: int, gain: float,
) -> dict[str, Any]:
    command = [
        fluidsynth, "-ni", str(soundfont), str(midi_path),
        "-F", str(wav_path), "-r", str(sample_rate), "-g", str(gain),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"fluidsynth failed ({result.returncode}): {result.stderr}")
    return {"command": command, "stdout": result.stdout, "stderr": result.stderr}


def _measure_true_peak(samples: np.ndarray) -> float:
    """Measure inter-sample peak with a frozen 4x polyphase reconstruction.

    ``samples`` is frame-major, float64 PCM normalized so digital full scale is
    1.0.  Oversampling each channel together on the time axis exposes peaks
    between the stored samples; using an explicit Kaiser window makes the
    measurement deterministic across all package builds using the frozen
    SciPy environment.
    """
    if samples.ndim != 2:
        raise ValueError("true-peak input must be frame-major [frames, channels]")
    if samples.size == 0:
        return 0.0
    reconstructed = resample_poly(
        samples,
        TRUE_PEAK_OVERSAMPLE,
        1,
        axis=0,
        window=("kaiser", 8.6),
        padtype="constant",
    )
    return float(np.max(np.abs(reconstructed), initial=0.0))


def _dbtp(linear_peak: float) -> float | None:
    return None if linear_peak <= 0.0 else float(20.0 * math.log10(linear_peak))


def _quantize_pcm16(samples: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(samples * 32768.0), -32768, 32767).astype("<i2")


def _fix_wav(path: Path, *, sample_rate: int, seconds: int) -> dict[str, Any]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        original_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if width != 2 or original_rate != sample_rate:
        raise RuntimeError(f"render must be 16-bit/{sample_rate}Hz, got {width * 8}/{original_rate}")
    raw = np.frombuffer(frames, dtype="<i2")
    if raw.size % channels:
        raise RuntimeError("rendered PCM does not contain complete channel frames")
    samples = raw.astype(np.float64).reshape(-1, channels) / 32768.0
    target_frames = sample_rate * seconds
    if samples.shape[0] < target_frames:
        samples = np.pad(samples, ((0, target_frames - samples.shape[0]), (0, 0)))
    elif samples.shape[0] > target_frames:
        samples = samples[:target_frames]

    sample_peak_before = int(np.max(np.abs(raw.astype(np.int32)), initial=0))
    true_peak_before = _measure_true_peak(samples)
    limit_linear = float(10.0 ** (TRUE_PEAK_LIMIT_DBTP / 20.0))
    scale = min(1.0, limit_linear / true_peak_before) if true_peak_before > 0 else 1.0
    protected = scale < 1.0
    quantized = _quantize_pcm16(samples * scale)
    normalized_after = quantized.astype(np.float64) / 32768.0
    true_peak_after = _measure_true_peak(normalized_after)
    # Quantization can move a value by half an LSB.  A second attenuation pass
    # keeps the recorded post-quantization WAV itself beneath the frozen limit.
    if true_peak_after > limit_linear:
        correction = (limit_linear / true_peak_after) * (1.0 - 1e-9)
        scale *= correction
        protected = True
        quantized = _quantize_pcm16(samples * scale)
        normalized_after = quantized.astype(np.float64) / 32768.0
        true_peak_after = _measure_true_peak(normalized_after)
    if true_peak_after > limit_linear + 1e-9:
        raise RuntimeError(
            f"true-peak protection failed: {_dbtp(true_peak_after)} dBTP exceeds "
            f"{TRUE_PEAK_LIMIT_DBTP} dBTP"
        )
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(channels)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(quantized.reshape(-1).tobytes())
    return {
        "channels": channels, "sample_width_bits": 16, "sample_rate": sample_rate,
        "frames": sample_rate * seconds, "seconds": seconds,
        "sample_peak_before": sample_peak_before,
        "sample_peak_after": int(
            np.max(np.abs(quantized.astype(np.int32)), initial=0)
        ),
        "true_peak_linear_before": true_peak_before,
        "true_peak_dbtp_before": _dbtp(true_peak_before),
        "true_peak_linear_after": true_peak_after,
        "true_peak_dbtp_after": _dbtp(true_peak_after),
        "true_peak_limit_dbtp": TRUE_PEAK_LIMIT_DBTP,
        "true_peak_protection_applied": protected,
        "applied_gain_scale": scale,
        "peak_measurement": TRUE_PEAK_IMPLEMENTATION,
        "true_peak_oversample_factor": TRUE_PEAK_OVERSAMPLE,
        "true_peak_verified": True,
        "silent": not bool(np.any(quantized)),
    }


def build_package(args: argparse.Namespace) -> None:
    selection_path = Path(args.selection).resolve()
    selection = _read_json(selection_path)
    if selection.get("schema_version") != "streammuse.melody_robustness.listening_selection.v1":
        raise ValueError("unsupported listening selection schema")
    if selection.get("frozen_before_formal") is not True:
        raise ValueError("listening selection must be frozen before formal outputs")
    if (
        selection.get("clip_count") != 24
        or selection.get("clip_seconds") != CLIP_SECONDS
        or selection.get("render_bpm") != RENDER_BPM
        or selection.get("gain_policy") != GAIN_POLICY
        or len(selection.get("clips", [])) != 24
    ):
        raise ValueError("selection must contain exactly 24 clips")
    config, config_sha, schedule, manifest_path, entries = _validated_build_inputs(
        args, selection_path, selection
    )
    output_root = Path(args.output_root).resolve()
    package = Path(args.package_dir).resolve()
    horizons = _song_analysis_horizons(entries)
    if selection.get("analysis_horizons_ticks") != horizons:
        raise RuntimeError(
            "selection analysis horizons do not match its hash-pinned input manifest"
        )
    for clip in selection["clips"]:
        song = str(clip.get("song"))
        if song not in horizons:
            raise ValueError(f"selection names unknown song: {song}")
        if clip.get("analysis_end_tick") != horizons[song]:
            raise RuntimeError(f"{song}: selection analysis horizon drifted after freeze")
        _validate_excerpt(
            song=song,
            start_beat=clip.get("excerpt_start_model_beat"),
            end_beat=clip.get("excerpt_end_model_beat"),
            analysis_end_tick=horizons[song],
        )
    campaign_binding = _validated_campaign_binding(args, config, config_sha)
    manifest_dir = manifest_path.parent
    entry_index = {
        (_song(entry), str(entry["condition"]), entry.get("perturb_seed")): entry
        for entry in entries
    }
    soundfont = Path(args.soundfont).resolve() if args.soundfont else None
    if args.sample_rate != RENDER_SAMPLE_RATE:
        raise ValueError(f"accepted listening render requires {RENDER_SAMPLE_RATE} Hz")
    if args.gain != FIXED_SYNTH_GAIN:
        raise ValueError(f"accepted listening render requires fixed synth gain {FIXED_SYNTH_GAIN}")
    if not args.midi_only and (soundfont is None or not soundfont.is_file()):
        raise FileNotFoundError("--soundfont is required for an accepted WAV package")
    synth_version = None
    if not args.midi_only:
        version_result = subprocess.run([args.fluidsynth, "--version"], text=True, capture_output=True, check=False)
        synth_version = (version_result.stdout or version_result.stderr).strip()
        if version_result.returncode:
            raise RuntimeError(f"cannot identify fluidsynth: {synth_version}")
    verified_runs: dict[str, tuple[Path, dict[str, Any], set[Path]]] = {}
    for clip in selection["clips"]:
        if clip.get("anchor_kind") == "known_bad_harmonic_m2":
            continue
        row = _find_run(schedule, clip)
        verified_runs.setdefault(
            str(row["run_id"]),
            _verified_attempt(output_root, row, campaign_binding),
        )
    midi_dir = package / "blind" / "midi"
    wav_dir = package / "blind" / "wav"
    midi_dir.mkdir(parents=True, exist_ok=False)
    wav_dir.mkdir(parents=True, exist_ok=False)
    key_rows = []
    render_rows = []
    for clip in selection["clips"]:
        sample_id = str(clip["sample_id"])
        semantic_index = int(clip["semantic_index"])
        start_tick = int(clip["excerpt_start_model_beat"]) * 4
        if clip.get("anchor_kind") == "known_bad_harmonic_m2":
            source_path = Path(args.controls_root).resolve() / clip["song"] / "harmonic_m2.mid"
            accompaniment = load_midi_roll(source_path, end_tick=start_tick + CLIP_TICKS)
            entry = entry_index[(clip["song"], "sham", None)]
            melody_path = _path(entry, manifest_dir, "output_midi", "melody_midi")
            melody = load_midi_roll(melody_path, end_tick=start_tick + CLIP_TICKS)
            rendered_roll = _union(_slice(melody, start_tick), _slice(accompaniment, start_tick))
            source_run_id = None
        else:
            row = _find_run(schedule, clip)
            attempt, _verdict, indexed_artifacts = verified_runs[str(row["run_id"])]
            if clip["pipeline"] == "rt_theoretical":
                acc_path = _single(attempt, "theoretical_model.mid")
                if acc_path.resolve() not in indexed_artifacts:
                    raise RuntimeError(f"theoretical MIDI is not verdict-indexed: {row['run_id']}")
                accompaniment = load_midi_roll(acc_path, end_tick=start_tick + CLIP_TICKS)
                rendered_roll = _slice(accompaniment, start_tick)
            else:
                acc_path = _single(attempt, "combined.mid")
                if acc_path.resolve() not in indexed_artifacts:
                    raise RuntimeError(f"combined MIDI is not verdict-indexed: {row['run_id']}")
                accompaniment = load_midi_roll(
                    acc_path,
                    end_tick=start_tick + CLIP_TICKS,
                    track_name_contains="Accompaniment",
                )
                entry = entry_index[(clip["song"], clip["condition"], clip.get("perturb_seed"))]
                melody_path = _path(entry, manifest_dir, "output_midi", "melody_midi")
                melody = load_midi_roll(melody_path, end_tick=start_tick + CLIP_TICKS)
                rendered_roll = _union(_slice(melody, start_tick), _slice(accompaniment, start_tick))
            source_path = acc_path
            source_run_id = row["run_id"]
        midi_path = midi_dir / f"{sample_id}.mid"
        write_roll_midi(rendered_roll, midi_path, bpm=RENDER_BPM)
        audio_info = None
        render_info = None
        wav_path = wav_dir / f"{sample_id}.wav"
        if not args.midi_only:
            render_info = _render(
                midi_path, wav_path, soundfont=soundfont, fluidsynth=args.fluidsynth,
                sample_rate=args.sample_rate, gain=args.gain,
            )
            audio_info = _fix_wav(wav_path, sample_rate=args.sample_rate, seconds=CLIP_SECONDS)
        key_rows.append(
            {
                "sample_id": sample_id, "semantic_index": semantic_index,
                "block": clip["block"], "song": clip["song"],
                "condition": clip["condition"], "pipeline": clip["pipeline"],
                "perturb_seed": clip.get("perturb_seed"), "sample_seed": clip["sample_seed"],
                "source_run_id": source_run_id, "source_path": str(source_path),
                "source_sha256": file_sha256(source_path),
                "duplicate_semantic_index": clip.get("duplicate_semantic_index"),
            }
        )
        render_rows.append(
            {
                "sample_id": sample_id, "midi": str(midi_path.relative_to(package)),
                "midi_sha256": file_sha256(midi_path),
                "wav": str(wav_path.relative_to(package)) if wav_path.exists() else None,
                "wav_sha256": file_sha256(wav_path) if wav_path.exists() else None,
                "audio": audio_info, "render": render_info,
            }
        )
    # Repeated trials are literal byte copies of their frozen source clips;
    # this removes synthesizer nondeterminism as a possible consistency cue.
    source_sample_by_semantic = {
        int(row["semantic_index"]): str(row["sample_id"])
        for row in key_rows if row.get("duplicate_semantic_index") is None
    }
    render_by_sample = {str(row["sample_id"]): row for row in render_rows}
    for key_row in key_rows:
        duplicate_index = key_row.get("duplicate_semantic_index")
        if duplicate_index is None:
            continue
        source_id = source_sample_by_semantic[int(duplicate_index)]
        target_id = str(key_row["sample_id"])
        source_render = render_by_sample[source_id]
        target_render = render_by_sample[target_id]
        source_midi = package / source_render["midi"]
        target_midi = package / target_render["midi"]
        shutil.copyfile(source_midi, target_midi)
        target_render["midi_sha256"] = file_sha256(target_midi)
        if not args.midi_only:
            source_wav = package / source_render["wav"]
            target_wav = package / target_render["wav"]
            shutil.copyfile(source_wav, target_wav)
            target_render["wav_sha256"] = file_sha256(target_wav)
            target_render["audio"] = dict(source_render["audio"])
            target_render["render"] = {"byte_copy_of": source_id}
    with (package / "blind" / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "overall_quality_1_to_5", "flaw_note"])
        writer.writeheader()
        for clip in selection["clips"]:
            writer.writerow({"sample_id": clip["sample_id"], "overall_quality_1_to_5": "", "flaw_note": ""})
    write_canonical_json(package / "private_key.json", {
        "campaign_config_sha256": config_sha,
        "run_schedule_sha256": args.schedule_sha256,
        "campaign_binding_sha256": campaign_binding["campaign_binding_sha256"],
        "selection_sha256": file_sha256(selection_path), "clips": key_rows,
        "unblind_only_after_scores_sealed": True,
    })
    private_key_sha = file_sha256(package / "private_key.json")
    write_canonical_json(package / "render_manifest.json", {
        "campaign_config_sha256": config_sha,
        "run_schedule_sha256": args.schedule_sha256,
        "campaign_binding_sha256": campaign_binding["campaign_binding_sha256"],
        "input_manifest_sha256": config["input_manifest"]["sha256"],
        "selection_path": str(selection_path), "selection_sha256": file_sha256(selection_path),
        "private_key_sha256": private_key_sha,
        "render_bpm": RENDER_BPM, "sample_rate": args.sample_rate, "bit_depth": 16,
        "gain": args.gain, "gain_policy": selection["gain_policy"],
        "true_peak_requirement": "genuine_inter_sample_true_peak_measurement_and_protection",
        "peak_implementation": TRUE_PEAK_IMPLEMENTATION,
        "soundfont_path": str(soundfont) if soundfont else None,
        "soundfont_sha256": file_sha256(soundfont) if soundfont else None,
        "synth": args.fluidsynth, "synth_version": synth_version,
        "clips": render_rows,
    })
    audit = audit_package_dir(package, require_wav=not args.midi_only)
    write_canonical_json(package / "package_audit.json", audit)
    if not audit["valid"]:
        raise RuntimeError(f"listening package audit failed: {audit['errors']}")


def audit_package_dir(package: Path, *, require_wav: bool = True) -> dict[str, Any]:
    package = package.resolve()
    errors: list[str] = []
    render = _read_json(package / "render_manifest.json")
    if render.get("render_bpm") != RENDER_BPM:
        errors.append(f"render BPM must be {RENDER_BPM}")
    if render.get("sample_rate") != RENDER_SAMPLE_RATE:
        errors.append(f"render sample rate must be {RENDER_SAMPLE_RATE}")
    if render.get("bit_depth") != 16:
        errors.append("render bit depth must be 16")
    if render.get("gain") != FIXED_SYNTH_GAIN:
        errors.append(f"render synth gain must be fixed at {FIXED_SYNTH_GAIN}")
    if render.get("gain_policy") != GAIN_POLICY:
        errors.append("render gain/peak policy does not match the frozen policy")
    if require_wav and render.get("peak_implementation") != TRUE_PEAK_IMPLEMENTATION:
        errors.append(
            "accepted WAV package requires the frozen 4x inter-sample true-peak "
            f"implementation {TRUE_PEAK_IMPLEMENTATION}"
        )
    clips = render.get("clips", [])
    if len(clips) != 24:
        errors.append(f"expected 24 render rows, got {len(clips)}")
    sample_ids = [str(row.get("sample_id")) for row in clips]
    if len(set(sample_ids)) != 24:
        errors.append("sample IDs are not one-to-one")
    key = _read_json(package / "private_key.json")
    key_sha = file_sha256(package / "private_key.json")
    if render.get("private_key_sha256") != key_sha:
        errors.append("private key hash mismatch")
    if key.get("selection_sha256") != render.get("selection_sha256"):
        errors.append("private key and render manifest selection hashes differ")
    if key.get("campaign_config_sha256") != render.get("campaign_config_sha256"):
        errors.append("private key and render manifest campaign hashes differ")
    if key.get("run_schedule_sha256") != render.get("run_schedule_sha256"):
        errors.append("private key and render manifest schedule hashes differ")
    if key.get("campaign_binding_sha256") != render.get("campaign_binding_sha256"):
        errors.append("private key and render manifest campaign binding hashes differ")
    selection_path_raw = render.get("selection_path")
    if not isinstance(selection_path_raw, str) or not Path(selection_path_raw).is_file():
        errors.append("selection path is missing from the render manifest")
    elif file_sha256(selection_path_raw) != render.get("selection_sha256"):
        errors.append("selection file hash mismatch")
    if {row["sample_id"] for row in key.get("clips", [])} != set(sample_ids):
        errors.append("blind key and render manifest do not map one-to-one")
    duplicate_groups: dict[int, list[dict[str, Any]]] = {}
    for row in key.get("clips", []):
        identity = int(
            row["duplicate_semantic_index"]
            if row.get("duplicate_semantic_index") is not None
            else row["semantic_index"]
        )
        duplicate_groups.setdefault(identity, []).append(row)
    repeated = [rows for rows in duplicate_groups.values() if len(rows) == 2]
    if len(repeated) != 2:
        errors.append(f"expected two repeated trials, found {len(repeated)}")
    render_by_id = {row["sample_id"]: row for row in clips}
    for pair in repeated:
        hashes = {
            render_by_id[row["sample_id"]].get("wav_sha256")
            or render_by_id[row["sample_id"]].get("midi_sha256")
            for row in pair
        }
        if len(hashes) != 1:
            errors.append(f"repeat trial audio mismatch: {[row['sample_id'] for row in pair]}")
    for row in clips:
        raw_midi = Path(str(row.get("midi", "")))
        midi = (package / raw_midi).resolve()
        if raw_midi.is_absolute() or not midi.is_relative_to(package):
            errors.append(f"MIDI path escapes package: {row.get('sample_id')}")
            continue
        if not midi.is_file() or file_sha256(midi) != row["midi_sha256"]:
            errors.append(f"missing/corrupt MIDI: {row.get('sample_id')}")
        else:
            try:
                midi_file = mido.MidiFile(midi)
                tempos = [
                    message.tempo
                    for track in midi_file.tracks
                    for message in track
                    if message.type == "set_tempo"
                ]
                expected_tempo = int(mido.bpm2tempo(RENDER_BPM))
                if not tempos or any(int(tempo) != expected_tempo for tempo in tempos):
                    errors.append(f"MIDI tempo is not fixed at {RENDER_BPM} BPM: {row['sample_id']}")
            except Exception as exc:
                errors.append(f"unreadable MIDI for tempo audit: {row.get('sample_id')}: {exc}")
        if require_wav:
            if not row.get("wav"):
                errors.append(f"missing WAV path: {row.get('sample_id')}")
                continue
            raw_wav = Path(str(row["wav"]))
            wav = (package / raw_wav).resolve()
            if raw_wav.is_absolute() or not wav.is_relative_to(package):
                errors.append(f"WAV path escapes package: {row.get('sample_id')}")
                continue
            if not wav.is_file() or file_sha256(wav) != row["wav_sha256"]:
                errors.append(f"missing/corrupt WAV: {row.get('sample_id')}")
                continue
            with wave.open(str(wav), "rb") as source:
                wav_rate = source.getframerate()
                wav_channels = source.getnchannels()
                wav_width = source.getsampwidth()
                wav_frame_count = source.getnframes()
                wav_frames = source.readframes(wav_frame_count)
                if wav_rate != RENDER_SAMPLE_RATE:
                    errors.append(f"sample-rate mismatch: {row['sample_id']}")
                if wav_channels not in {1, 2}:
                    errors.append(f"channel mismatch: {row['sample_id']}")
                if wav_width != 2:
                    errors.append(f"bit-depth mismatch: {row['sample_id']}")
                expected_frames = RENDER_SAMPLE_RATE * CLIP_SECONDS
                if wav_frame_count != expected_frames:
                    errors.append(f"duration mismatch: {row['sample_id']}")
            measured_true_peak_dbtp: float | None = None
            measured_silent: bool | None = None
            if wav_width == 2 and wav_channels in {1, 2}:
                wav_pcm = np.frombuffer(wav_frames, dtype="<i2")
                if wav_pcm.size % wav_channels:
                    errors.append(f"incomplete PCM channel frame: {row['sample_id']}")
                else:
                    measured_silent = not bool(np.any(wav_pcm))
                    measured_linear = _measure_true_peak(
                        wav_pcm.astype(np.float64).reshape(-1, wav_channels) / 32768.0
                    )
                    measured_true_peak_dbtp = _dbtp(measured_linear)
                    if (
                        measured_true_peak_dbtp is not None
                        and measured_true_peak_dbtp > TRUE_PEAK_LIMIT_DBTP
                    ):
                        errors.append(
                            f"independently measured true peak exceeds "
                            f"{TRUE_PEAK_LIMIT_DBTP} dBTP: {row['sample_id']} "
                            f"({measured_true_peak_dbtp} dBTP)"
                        )
                    if measured_silent:
                        errors.append(f"WAV contains no nonzero samples: {row['sample_id']}")
            audio = row.get("audio")
            if not isinstance(audio, Mapping):
                errors.append(f"missing audio audit metadata: {row['sample_id']}")
            else:
                if audio.get("silent") is not False:
                    errors.append(f"silent listening clip: {row['sample_id']}")
                if audio.get("true_peak_verified") is not True:
                    errors.append(
                        f"true peak is unverified (sample peak is insufficient): {row['sample_id']}"
                    )
                else:
                    measured = audio.get("true_peak_dbtp_after")
                    if (
                        isinstance(measured, bool)
                        or not isinstance(measured, (int, float))
                        or float(measured) > TRUE_PEAK_LIMIT_DBTP
                    ):
                        errors.append(
                            f"true peak exceeds {TRUE_PEAK_LIMIT_DBTP} dBTP or lacks a "
                            f"numeric measurement: {row['sample_id']}"
                        )
                    elif (
                        measured_true_peak_dbtp is None
                        or abs(float(measured) - measured_true_peak_dbtp) > 1e-7
                    ):
                        errors.append(
                            f"recorded true-peak metadata does not match independent WAV "
                            f"measurement: {row['sample_id']}"
                        )
                if audio.get("peak_measurement") != TRUE_PEAK_IMPLEMENTATION:
                    errors.append(f"true-peak implementation drift: {row['sample_id']}")
                if audio.get("true_peak_oversample_factor") != TRUE_PEAK_OVERSAMPLE:
                    errors.append(f"true-peak oversample factor drift: {row['sample_id']}")
                if audio.get("true_peak_limit_dbtp") != TRUE_PEAK_LIMIT_DBTP:
                    errors.append(f"true-peak limit drift: {row['sample_id']}")
                if measured_silent is not None and bool(audio.get("silent")) != measured_silent:
                    errors.append(f"recorded silence flag does not match WAV: {row['sample_id']}")
    valid = not errors
    return {
        "valid": valid,
        "accepted_final": valid and require_wav,
        "errors": errors,
        "clip_count": len(clips),
        "repeat_trial_count": len(repeated),
        "campaign_config_sha256": render.get("campaign_config_sha256"),
        "run_schedule_sha256": render.get("run_schedule_sha256"),
        "campaign_binding_sha256": render.get("campaign_binding_sha256"),
        "selection_sha256": render.get("selection_sha256"),
        "private_key_sha256": key_sha,
        "render_manifest_sha256": file_sha256(package / "render_manifest.json"),
        "true_peak_limitation": (
            None if not require_wav or all(
                isinstance(row.get("audio"), Mapping)
                and row["audio"].get("true_peak_verified") is True
                for row in clips
            )
            else "one or more clips lacks a verified post-quantization true-peak measurement"
        ),
    }


def audit_package(args: argparse.Namespace) -> None:
    result = audit_package_dir(Path(args.package_dir).resolve(), require_wav=not args.allow_midi_only)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(2)


def seal_scores(args: argparse.Namespace) -> None:
    package = Path(args.package_dir).resolve()
    package_audit_path = package / "package_audit.json"
    package_audit = _read_json(package_audit_path)
    if package_audit.get("accepted_final") is not True:
        raise RuntimeError("scores can only be sealed for an accepted final listening package")
    scores = package / "blind" / "scores.csv"
    rows = list(csv.DictReader(scores.open("r", encoding="utf-8")))
    if len(rows) != 24:
        raise ValueError("scores.csv must contain 24 rows")
    for row in rows:
        try:
            score = int(row["overall_quality_1_to_5"])
        except Exception as exc:
            raise ValueError(f"missing/invalid score for {row.get('sample_id')}") from exc
        if score not in {1, 2, 3, 4, 5}:
            raise ValueError(f"score outside 1..5 for {row['sample_id']}")
    key_path = package / "private_key.json"
    key = _read_json(key_path)
    expected_ids = {str(row["sample_id"]) for row in key.get("clips", [])}
    score_ids = [str(row.get("sample_id")) for row in rows]
    if len(set(score_ids)) != 24 or set(score_ids) != expected_ids:
        raise ValueError("scores.csv sample IDs do not match the hash-pinned private key")
    payload = {
        "scores_path": str(scores), "scores_sha256": file_sha256(scores),
        "private_key_sha256": file_sha256(key_path),
        "package_audit_sha256": file_sha256(package_audit_path),
        "campaign_config_sha256": package_audit.get("campaign_config_sha256"),
        "run_schedule_sha256": package_audit.get("run_schedule_sha256"),
        "campaign_binding_sha256": package_audit.get("campaign_binding_sha256"),
        "selection_sha256": package_audit.get("selection_sha256"),
        "sealed_before_unblinding": True, "post_unblinding_followup_separate": True,
    }
    write_canonical_json(package / "sealed_scores.json", payload)


def unblind(args: argparse.Namespace) -> None:
    package = Path(args.package_dir).resolve()
    sealed_path = package / "sealed_scores.json"
    sealed = _read_json(sealed_path)
    scores_path = Path(sealed["scores_path"]).resolve()
    if not scores_path.is_relative_to(package) or scores_path != (package / "blind" / "scores.csv"):
        raise RuntimeError("sealed score path escapes or differs from the package score sheet")
    if file_sha256(scores_path) != sealed["scores_sha256"]:
        raise RuntimeError("scores changed after sealing")
    scores = {row["sample_id"]: row for row in csv.DictReader(scores_path.open("r", encoding="utf-8"))}
    key_path = package / "private_key.json"
    if file_sha256(key_path) != sealed.get("private_key_sha256"):
        raise RuntimeError("private key changed after score sealing")
    key = _read_json(key_path)
    rows = [{**record, **scores[record["sample_id"]]} for record in key["clips"]]
    write_canonical_json(package / "unblinded_scores.json", {
        "sealed_scores_sha256": file_sha256(sealed_path),
        "private_key_sha256": file_sha256(key_path),
        "campaign_config_sha256": sealed.get("campaign_config_sha256"),
        "run_schedule_sha256": sealed.get("run_schedule_sha256"),
        "campaign_binding_sha256": sealed.get("campaign_binding_sha256"),
        "selection_sha256": sealed.get("selection_sha256"),
        "single_listener": True, "interpretation": "exploratory_qualitative_judgement",
        "rows": rows,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-selection")
    freeze.add_argument("--input-manifest", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--perturb-seed", type=int, default=SEEDS["perturb"][0])
    freeze.add_argument("--sample-seed", type=int, default=SEEDS["sample"][0])
    freeze.add_argument("--blind-order-seed", type=int, default=SEEDS["blind_order"])
    freeze.add_argument("--excerpt-start-beat", type=int, default=0)
    freeze.add_argument("--excerpt-starts-json")
    freeze.set_defaults(func=freeze_selection)
    build = sub.add_parser("build")
    build.add_argument("--config", required=True)
    build.add_argument("--config-sha256", required=True)
    build.add_argument("--selection", required=True)
    build.add_argument("--schedule", required=True)
    build.add_argument("--schedule-sha256", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--controls-root", required=True)
    build.add_argument("--package-dir", required=True)
    build.add_argument("--soundfont")
    build.add_argument("--fluidsynth", default="fluidsynth")
    build.add_argument("--sample-rate", type=int, default=RENDER_SAMPLE_RATE)
    build.add_argument("--gain", type=float, default=FIXED_SYNTH_GAIN)
    build.add_argument("--midi-only", action="store_true", help="development only; not an accepted final package")
    build.set_defaults(func=build_package)
    audit = sub.add_parser("audit")
    audit.add_argument("--package-dir", required=True)
    audit.add_argument("--allow-midi-only", action="store_true")
    audit.set_defaults(func=audit_package)
    seal = sub.add_parser("seal-scores")
    seal.add_argument("--package-dir", required=True)
    seal.set_defaults(func=seal_scores)
    reveal = sub.add_parser("unblind")
    reveal.add_argument("--package-dir", required=True)
    reveal.set_defaults(func=unblind)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
