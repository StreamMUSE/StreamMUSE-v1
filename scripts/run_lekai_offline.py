from __future__ import annotations

import argparse
import time
from pathlib import Path

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
    parser.add_argument("--condition-idx", type=str, default="0", help="Dataset index or 'all'")
    parser.add_argument("--gt-prefix-beats", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument(
        "--bpm",
        type=int,
        default=None,
        help="Override conditioning BPM (default: read from NPZ metadata). "
        "Used to pin BPM token identical to the realtime side.",
    )
    parser.add_argument("--cache-lengths", action="store_true", help="Use dataset length cache if available")
    return parser.parse_args()


def resolve_indices(spec: str, dataset_size: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(dataset_size))
    idx = int(spec)
    if idx < 0 or idx >= dataset_size:
        raise ValueError(f"condition-idx out of range: {idx} (dataset size: {dataset_size})")
    return [idx]


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
        mode="train",
    )

    indices = resolve_indices(args.condition_idx, len(dataset))
    print(f"[offline] total_items={len(dataset)}, selected={len(indices)}")

    total_start = time.perf_counter()
    per_song_ms: list[float] = []

    for rank, idx in enumerate(indices, start=1):
        song_start = time.perf_counter()
        result = model.generate_accompaniment(
            dataset,
            condition_idx=idx,
            gt_prefix_beats=int(args.gt_prefix_beats),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            device=device,
            verbose=False,
            bpm_override=args.bpm,
        )
        elapsed_ms = (time.perf_counter() - song_start) * 1000
        per_song_ms.append(elapsed_ms)

        stem = Path(str(result.get("GT_path", f"item_{idx}.npz"))).stem
        generated_path = output_dir / f"{idx:03d}_{stem}_generated.mid"
        gt_path = output_dir / f"{idx:03d}_{stem}_gt.mid"

        tokens_to_midi(result_dict=result, save_path=str(generated_path), velocity=80)
        save_gt_midi(save_path=str(gt_path), gt_path=str(result["GT_path"]), velocity=80)

        print(
            f"[offline] ({rank}/{len(indices)}) idx={idx} -> "
            f"gen={generated_path.name}, gt={gt_path.name}, time_ms={elapsed_ms:.1f}"
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
