#!/usr/bin/env python3
"""Audit and organize frozen Task 3 cross-style continuation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean

from experiments.lekai_sampling_monotony.analyze_task2_outputs import _output_features


EXPECTED_RUNS = {
    ("greedy", 0): {
        "gt_prefix_beats": 8,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
    },
    **{
        ("sampling", seed): {
            "gt_prefix_beats": 8,
            "temperature": 1.1,
            "top_k": 0,
            "top_p": 0.95,
        }
        for seed in (0, 1, 2)
    },
}
EXPECTED_COMMON_GENERATION = {
    "repetition_penalty": 1.0,
    "use_fp16": True,
    "export_gt_midi": True,
}


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate_manifest(
    manifest: dict,
    decoder: str,
    seed: int,
    expected_fingerprint: str,
) -> None:
    if int(manifest["seed"]) != seed:
        raise RuntimeError(f"seed mismatch: {manifest['seed']} != {seed}")
    if manifest["selection_fingerprint_sha1"] != expected_fingerprint:
        raise RuntimeError(
            "selection fingerprint mismatch: "
            f"{manifest['selection_fingerprint_sha1']} != {expected_fingerprint}"
        )
    if len(manifest["items"]) != 20:
        raise RuntimeError(f"expected 20 pieces, found {len(manifest['items'])}")
    generation = manifest["generation"]
    expected_generation = {
        **EXPECTED_RUNS[(decoder, seed)],
        **EXPECTED_COMMON_GENERATION,
    }
    for key, expected in expected_generation.items():
        if generation[key] != expected:
            raise RuntimeError(
                f"{decoder}/seed{seed} {key}: {generation[key]} != {expected}"
            )


def _aggregate_by_style_and_decoder(rows: list[dict]) -> list[dict]:
    metric_names = [
        "acc_note_density_per_beat",
        "pitch_class_entropy",
        "dominant_pitch_class_rate",
        "median_duration_beats",
        "long_note_rate_gt_8_beats",
        "piece_end_note_rate",
        "empty_bar_rate",
        "unique_nonempty_bar_ratio",
        "adjacent_exact_nonempty_bar_repeat_rate",
        "max_empty_bar_run",
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["style"]), str(row["decoder"]))].append(row)
    summaries = []
    for (style, decoder), group in grouped.items():
        summary = {
            "style": style,
            "decoder": decoder,
            "sample_count": len(group),
            "piece_count": len({row["piece_id"] for row in group}),
            "seed_count": len({row["seed"] for row in group}),
            "empty_output_count": sum(bool(row["output_empty"]) for row in group),
        }
        for metric in metric_names:
            summary[f"mean_{metric}"] = mean(float(row[metric]) for row in group)
        summaries.append(summary)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ignore-prefix-beats", type=int, default=8)
    parser.add_argument("--expected-fingerprint", required=True)
    args = parser.parse_args()

    prepared_rows = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
    prepared_by_name = {
        Path(row["prepared_npz"]).name: row for row in prepared_rows
    }
    style_order = list(dict.fromkeys(row["style"] for row in prepared_rows))
    piece_order_by_style = {
        style: [row["piece_id"] for row in prepared_rows if row["style"] == style]
        for style in style_order
    }

    manifest_paths = sorted(args.inference_root.glob("*/seed*/*_manifest.json"))
    if len(manifest_paths) != len(EXPECTED_RUNS):
        raise RuntimeError(
            f"expected {len(EXPECTED_RUNS)} inference manifests, "
            f"found {len(manifest_paths)}"
        )

    manifests = []
    output_rows = []
    source_gt_by_piece: dict[str, Path] = {}
    seen_runs = set()
    for manifest_path in manifest_paths:
        decoder = manifest_path.parent.parent.name
        seed = int(manifest_path.parent.name.removeprefix("seed"))
        seen_runs.add((decoder, seed))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(manifest, decoder, seed, args.expected_fingerprint)
        manifests.append(
            {
                "path": str(manifest_path),
                "decoder": decoder,
                "seed": seed,
                "selection_fingerprint_sha1": manifest["selection_fingerprint_sha1"],
                "generation": manifest["generation"],
                "checkpoint": manifest["ckpt"],
            }
        )
        for item in manifest["items"]:
            prepared = prepared_by_name.get(str(item["file_name"]))
            if prepared is None:
                raise RuntimeError(f"unexpected inference item: {item['file_name']}")
            generated_midi = Path(item["gen_midi_path"])
            gt_midi = Path(item["gt_midi_path"])
            if not generated_midi.is_file():
                raise FileNotFoundError(generated_midi)
            if not gt_midi.is_file():
                raise FileNotFoundError(gt_midi)
            source_gt_by_piece.setdefault(str(prepared["piece_id"]), gt_midi)
            features = _output_features(
                generated_midi,
                bpm=float(item["metadata"]["bpm"]),
                total_steps=int(item["metadata"]["total_steps"]),
                prefix_beats=args.ignore_prefix_beats,
            )
            output_rows.append(
                {
                    **prepared,
                    "decoder": decoder,
                    "seed": seed,
                    "source_midi": str(generated_midi),
                    **{
                        key: value
                        for key, value in features.items()
                        if not key.startswith("_")
                    },
                }
            )
    if seen_runs != set(EXPECTED_RUNS):
        raise RuntimeError(
            f"runs {sorted(seen_runs)} != {sorted(EXPECTED_RUNS)}"
        )
    if len(output_rows) != 80:
        raise RuntimeError(f"expected 80 outputs, found {len(output_rows)}")

    ratings = []
    listening_root = args.output_dir / "listening_by_style"
    for style_index, style in enumerate(style_order, start=1):
        for piece_index, piece_id in enumerate(piece_order_by_style[style], start=1):
            prepared = next(row for row in prepared_rows if row["piece_id"] == piece_id)
            piece_dir = (
                listening_root
                / f"{style_index:02d}_{style}"
                / f"{piece_index:02d}_{piece_id}"
            )
            piece_dir.mkdir(parents=True, exist_ok=True)
            source_gt = piece_dir / "00_source_gt.mid"
            shutil.copy2(source_gt_by_piece[str(piece_id)], source_gt)
            for row in sorted(
                (item for item in output_rows if item["piece_id"] == piece_id),
                key=lambda item: (
                    0 if item["decoder"] == "greedy" else 1,
                    int(item["seed"]),
                ),
            ):
                decoder = str(row["decoder"])
                seed = int(row["seed"])
                condition_index = 1 if decoder == "greedy" else seed + 2
                condition_name = (
                    "greedy" if decoder == "greedy" else f"sampling_seed{seed}"
                )
                generated = piece_dir / f"{condition_index:02d}_{condition_name}.mid"
                shutil.copy2(row["source_midi"], generated)
                relative_generated = generated.relative_to(args.output_dir)
                relative_gt = source_gt.relative_to(args.output_dir)
                row["listening_midi"] = str(relative_generated)
                row["listening_source_gt"] = str(relative_gt)
                ratings.append(
                    {
                        "order": len(ratings) + 1,
                        "style": style,
                        "piece_id": piece_id,
                        "decoder": decoder,
                        "seed": seed,
                        "source_gt_midi": str(relative_gt),
                        "midi": str(relative_generated),
                        "musical_quality": "",
                        "accompaniment_quality": "",
                        "melody_accompaniment_fit": "",
                        "variety": "",
                        "stability": "",
                        "comments": "",
                    }
                )

    aggregate = _aggregate_by_style_and_decoder(output_rows)
    source_gt_by_piece = {
        row["piece_id"]: row["source_gt_midi"] for row in ratings
    }
    listening_index = [
        {
            "order": row["order"],
            "style": row["style"],
            "piece_id": row["piece_id"],
            "decoder": row["decoder"],
            "seed": row["seed"],
            "source_gt_midi": row["source_gt_midi"],
            "midi": row["midi"],
        }
        for row in ratings
    ]
    piece_metadata = [
        {
            "piece_order": index,
            "style": row["style"],
            "piece_id": row["piece_id"],
            "title": row["title"],
            "creator": row["creator"],
            "classification_evidence": row["primary_evidence"],
            "bpm": row["bpm"],
            "num_measures": row["num_measures"],
            "melody_density_per_beat": row["melody_density_per_beat"],
            "accompaniment_density_per_beat": row[
                "accompaniment_density_per_beat"
            ],
            "source_gt_midi": source_gt_by_piece[row["piece_id"]],
        }
        for index, row in enumerate(prepared_rows, start=1)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_output.csv", output_rows)
    _write_csv(
        args.output_dir / "ratings.csv",
        ratings,
        fieldnames=[
            "order",
            "style",
            "piece_id",
            "decoder",
            "seed",
            "musical_quality",
            "accompaniment_quality",
            "melody_accompaniment_fit",
            "variety",
            "stability",
            "comments",
        ],
    )
    _write_csv(args.output_dir / "listening_index.csv", listening_index)
    _write_csv(args.output_dir / "piece_metadata.csv", piece_metadata)
    _write_csv(
        args.output_dir / "_technical" / "input_provenance.csv",
        prepared_rows,
    )
    (args.output_dir / "per_output.json").write_text(
        json.dumps(output_rows, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "output_count": len(output_rows),
                "piece_count": len(prepared_rows),
                "style_count": len(style_order),
                "ignore_prefix_beats": args.ignore_prefix_beats,
                "aggregate_by_style_and_decoder": aggregate,
                "runs": manifests,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_technical" / "input_provenance.json").write_text(
        json.dumps(prepared_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_count": len(output_rows), "aggregate": aggregate}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
