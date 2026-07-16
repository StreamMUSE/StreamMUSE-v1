from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from streammuse.infrastructure.inference.lekai_model.PianoDataset import PianoDataset
from streammuse.infrastructure.inference.lekai_model.Token2Midi import tokens_to_midi
from streammuse.infrastructure.inference.lekai_model.config import ModelConfig
from streammuse.infrastructure.inference.lekai_model.inference import load_model, save_gt_midi
from streammuse.infrastructure.inference.runtime_device import dtype_to_name, resolve_device, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lekai offline generation from NPZ dataset.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--npz-dir", type=str, required=True, help="Path to NPZ directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated MIDI files")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16"])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--condition-idx", type=str, default=None, help="Dataset index or 'all'")
    selection.add_argument("--condition-stem", type=str, default=None, help="Exact NPZ filename stem")
    selection.add_argument(
        "--condition-path",
        type=str,
        default=None,
        help="Exact NPZ path (absolute or relative to --npz-dir)",
    )
    parser.add_argument("--gt-prefix-beats", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--delay-beats", type=int, default=-1)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Per-run sampling seed. A fresh device-local Generator is created for every item.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Stable manifest run id (single item only)")
    parser.add_argument(
        "--bpm",
        type=int,
        default=None,
        help="Override conditioning BPM (default: read from NPZ metadata). "
        "Used to pin BPM token identical to the realtime side.",
    )
    parser.add_argument("--cache-lengths", action="store_true", help="Use dataset length cache if available")
    parser.add_argument(
        "--expected-dataset-size",
        type=int,
        default=None,
        help="Fail closed unless mode=all discovers exactly this many NPZ inputs",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-midi", type=str, default=None, help="Source melody MIDI for a single item")
    source.add_argument(
        "--source-midi-dir",
        type=str,
        default=None,
        help="Directory containing source melody MIDIs named <npz-stem>.mid",
    )
    parser.add_argument(
        "--require-source-midi",
        action="store_true",
        help="Fail if a selected input cannot be paired with a source MIDI",
    )
    return parser.parse_args()


def resolve_indices(spec: str, dataset_size: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(dataset_size))
    idx = int(spec)
    if idx < 0 or idx >= dataset_size:
        raise ValueError(f"condition-idx out of range: {idx} (dataset size: {dataset_size})")
    return [idx]


def resolve_selection(
    dataset: PianoDataset,
    *,
    condition_idx: str | None,
    condition_stem: str | None,
    condition_path: str | None,
) -> list[int]:
    """Resolve one exact manifest selector, retaining index 0 as the legacy default."""
    if condition_stem is not None:
        return [dataset.index_for_stem(condition_stem)]
    if condition_path is not None:
        return [dataset.index_for_path(condition_path)]
    return resolve_indices(condition_idx if condition_idx is not None else "0", len(dataset))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_json(path: Path, payload: object) -> None:
    """Write stable JSON so its SHA is meaningful across equivalent reruns."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_part0_roundtrip(
    *, npz_path: Path, part0_beats: object, dataset: PianoDataset
) -> dict[str, object]:
    """Decode the exact part0 beat tokens and compare them to NPZ channels 0:2."""
    if not isinstance(part0_beats, list) or not part0_beats:
        raise RuntimeError("model result did not expose non-empty part0_beats")
    with np.load(npz_path, allow_pickle=True) as payload:
        metadata = payload["metadata"].item()
        measures = [
            np.asarray(payload[f"measure_{index}"][:2], dtype=np.uint8)
            for index in range(int(metadata["num_measures"]))
        ]
    expected = (
        np.concatenate(measures, axis=2)
        if measures
        else np.zeros((2, 88, 0), dtype=np.uint8)
    )
    decoded_beats: list[np.ndarray] = []
    serialized_beats: list[list[int]] = []
    for raw in part0_beats:
        values = raw.detach().cpu().reshape(-1).tolist() if isinstance(raw, torch.Tensor) else list(raw)
        tokens = [int(value) for value in values]
        serialized_beats.append(tokens)
        if tokens == [int(dataset.bar_token)] or (tokens and set(tokens) == {173}):
            continue
        matrix = dataset.tokenizer.decompress_tokens(
            np.asarray(tokens, dtype=np.int64), end_marker_id=170
        )
        image = dataset.tokenizer.patch_tokens_to_image(matrix)
        if tuple(image.shape) != (2, 88, 4):
            raise RuntimeError(f"part0 decoded beat has unexpected shape: {image.shape}")
        decoded_beats.append(np.asarray(image > 0, dtype=np.uint8))
    decoded = (
        np.concatenate(decoded_beats, axis=2)
        if decoded_beats
        else np.zeros((2, 88, 0), dtype=np.uint8)
    )
    differing = (
        int(np.count_nonzero(expected != decoded))
        if expected.shape == decoded.shape
        else -1
    )
    result: dict[str, object] = {
        "valid": expected.shape == decoded.shape and differing == 0,
        "expected_shape": [int(value) for value in expected.shape],
        "decoded_shape": [int(value) for value in decoded.shape],
        "differing_cells": differing,
        "expected_roll_sha256": hashlib.sha256(expected.tobytes(order="C")).hexdigest(),
        "decoded_roll_sha256": hashlib.sha256(decoded.tobytes(order="C")).hexdigest(),
        "part0_beat_tokens_sha256": canonical_digest(serialized_beats),
        "bar_token": int(dataset.bar_token),
        "pad_marker": 173,
        "part0_end_marker": 170,
    }
    if result["valid"] is not True:
        raise RuntimeError(f"part0 encode/decode gate failed: {result}")
    return result


def make_sampling_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create the run-owned RNG immediately before generation."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _safe_artifact_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not normalized:
        raise ValueError(f"run-id has no filesystem-safe characters: {value!r}")
    return normalized


def _source_midi_for(
    *,
    args: argparse.Namespace,
    npz_path: Path,
    selected_count: int,
) -> Path | None:
    if args.source_midi is not None:
        if selected_count != 1:
            raise ValueError("--source-midi can only be used with one selected NPZ")
        candidate = Path(args.source_midi).expanduser().resolve()
    elif args.source_midi_dir is not None:
        candidate = Path(args.source_midi_dir).expanduser().resolve() / f"{npz_path.stem}.mid"
    else:
        # The repository and campaign staging layout both use sibling npz/ and mel/ dirs.
        candidate = npz_path.parent.parent / "mel" / f"{npz_path.stem}.mid"

    if candidate.is_file():
        return candidate.resolve()
    if args.require_source_midi:
        raise FileNotFoundError(f"Source MIDI not found for {npz_path.name}: {candidate}")
    return None


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    npz_dir = Path(args.npz_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not npz_dir.exists():
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")

    model_config = ModelConfig()
    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    checkpoint_sha256 = sha256_file(checkpoint)

    print(f"[offline] device={device}, dtype={dtype_to_name(dtype)}")
    print(f"[offline] checkpoint={checkpoint}")
    print(f"[offline] npz_dir={npz_dir}")

    model = load_model(
        model_path=str(checkpoint),
        model_config=model_config,
        device=device,
        dtype=dtype,
        use_cache=True,
    )

    dataset = PianoDataset(
        str(npz_dir),
        config=model_config,
        cache_lengths=bool(args.cache_lengths),
        mode="all",
    )

    if args.expected_dataset_size is not None and len(dataset) != args.expected_dataset_size:
        raise RuntimeError(
            f"Dataset size mismatch: expected {args.expected_dataset_size}, found {len(dataset)}"
        )

    indices = resolve_selection(
        dataset,
        condition_idx=args.condition_idx,
        condition_stem=args.condition_stem,
        condition_path=args.condition_path,
    )
    if args.run_id is not None and len(indices) != 1:
        raise ValueError("--run-id requires exactly one selected NPZ")
    print(f"[offline] total_items={len(dataset)}, selected={len(indices)}")

    total_start = time.perf_counter()
    per_song_ms: list[float] = []

    for rank, idx in enumerate(indices, start=1):
        song_start = time.perf_counter()
        npz_path = (Path(dataset.root_dir) / dataset.data_files[idx]).resolve()
        source_midi = _source_midi_for(args=args, npz_path=npz_path, selected_count=len(indices))
        npz_sha256 = sha256_file(npz_path)
        source_midi_sha256 = sha256_file(source_midi) if source_midi is not None else None

        # Construct after model loading and immediately before the run, so initialization or
        # warmup cannot consume the experiment RNG.  The generator device exactly matches the
        # model/logits device rather than relying on process-global torch.manual_seed().
        model_parameter = next(model.parameters())
        generator = make_sampling_generator(model_parameter.device, int(args.seed))
        result = model.generate_accompaniment(
            dataset,
            condition_idx=idx,
            delay_beats=int(args.delay_beats),
            gt_prefix_beats=int(args.gt_prefix_beats),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            device=device,
            verbose=False,
            bpm_override=args.bpm,
            generator=generator,
        )
        part0_gate = verify_part0_roundtrip(
            npz_path=npz_path,
            part0_beats=result.get("part0_beats"),
            dataset=dataset,
        )
        actual_npz_path = Path(str(result["GT_path"])).resolve()
        if actual_npz_path != npz_path:
            raise RuntimeError(
                f"Model used unexpected NPZ: selected {npz_path}, returned {actual_npz_path}"
            )
        if sha256_file(npz_path) != npz_sha256:
            raise RuntimeError(f"NPZ changed while generation was running: {npz_path}")
        if source_midi is not None and sha256_file(source_midi) != source_midi_sha256:
            raise RuntimeError(f"Source MIDI changed while generation was running: {source_midi}")
        elapsed_ms = (time.perf_counter() - song_start) * 1000
        per_song_ms.append(elapsed_ms)

        stem = Path(str(result.get("GT_path", f"item_{idx}.npz"))).stem
        artifact_id = _safe_artifact_id(args.run_id) if args.run_id else f"{idx:03d}_{stem}"
        effective_run_id = args.run_id or artifact_id
        generated_path = output_dir / f"{artifact_id}_generated.mid"
        gt_path = output_dir / f"{artifact_id}_gt.mid"
        token_trace_path = output_dir / f"{artifact_id}_tokens.json"
        run_config_path = output_dir / f"{artifact_id}_run_config.json"

        tokens_to_midi(result_dict=result, save_path=str(generated_path), velocity=80)
        save_gt_midi(save_path=str(gt_path), gt_path=str(result["GT_path"]), velocity=80)

        generated_sequence = result["generated_sequence"]
        if isinstance(generated_sequence, torch.Tensor):
            full_tokens = generated_sequence.reshape(-1).tolist()
        else:
            full_tokens = list(generated_sequence)
        sampled_tokens = [int(token) for token in result.get("sampled_token_trace", [])]
        token_trace = {
            "schema_version": 1,
            "run_id": effective_run_id,
            "seed": int(generator.initial_seed()),
            "sampled_tokens": sampled_tokens,
            "full_interleaved_sequence": [int(token) for token in full_tokens],
            "part1_beats": result.get("part1_beats", []),
            "part0_beat_tokens_sha256": part0_gate["part0_beat_tokens_sha256"],
        }
        write_canonical_json(token_trace_path, token_trace)

        source_midi_record = None
        if source_midi is not None:
            source_midi_record = {
                "path": str(source_midi),
                "sha256": source_midi_sha256,
            }

        run_config = {
            "schema_version": 1,
            "run_id": effective_run_id,
            "pipeline": "offline",
            "input": {
                "dataset_index": idx,
                "npz_path": str(npz_path),
                "npz_stem": stem,
                "npz_sha256": npz_sha256,
                "source_midi": source_midi_record,
                "part0_roundtrip": part0_gate,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha256,
            },
            "sampling": {
                "seed": int(generator.initial_seed()),
                "temperature": float(args.temperature),
                "top_k": int(args.top_k),
                "top_p": float(args.top_p),
                "repetition_penalty": float(args.repetition_penalty),
                "gt_prefix_beats": int(args.gt_prefix_beats),
                "delay_beats": int(args.delay_beats),
                "bpm_override": args.bpm,
            },
            "runtime": {
                "device": str(model_parameter.device),
                "dtype": dtype_to_name(model_parameter.dtype),
            },
            "outputs": {
                "generated_midi": generated_path.name,
                "generated_midi_sha256": sha256_file(generated_path),
                "ground_truth_midi": gt_path.name,
                "ground_truth_midi_sha256": sha256_file(gt_path),
                "token_trace": token_trace_path.name,
                "token_trace_sha256": sha256_file(token_trace_path),
                "sampled_token_count": len(sampled_tokens),
                "full_token_count": len(full_tokens),
            },
        }
        write_canonical_json(run_config_path, run_config)

        print(
            f"[offline] ({rank}/{len(indices)}) idx={idx} -> "
            f"gen={generated_path.name}, gt={gt_path.name}, "
            f"run_config={run_config_path.name}, time_ms={elapsed_ms:.1f}"
        )

    total_ms = (time.perf_counter() - total_start) * 1000
    if per_song_ms:
        avg_ms = sum(per_song_ms) / len(per_song_ms)
        print(
            f"[offline] completed={len(per_song_ms)}, total_ms={total_ms:.1f}, "
            f"avg_ms={avg_ms:.1f}, min_ms={min(per_song_ms):.1f}, max_ms={max(per_song_ms):.1f}"
        )


if __name__ == "__main__":
    main()
