#!/usr/bin/env python3
"""Analyze singleton paired Task 1 v2 greedy and nucleus runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from experiments.lekai_melody_timing_robustness.analyze_task1_outputs import analyze


EXPECTED = {
    ("greedy", 0): {"temperature": 1.0, "top_k": 1, "top_p": 1.0},
    ("sampling", 0): {"temperature": 1.1, "top_k": 0, "top_p": 0.95},
    ("sampling", 1): {"temperature": 1.1, "top_k": 0, "top_p": 0.95},
    ("sampling", 2): {"temperature": 1.1, "top_k": 0, "top_p": 0.95},
}


def _write_csv(
    path: Path,
    rows: list[dict],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    if fieldnames is None:
        fieldnames = list(rows[0])
        fieldnames.extend(
            key
            for row in rows[1:]
            for key in row
            if key not in fieldnames
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate_generation(manifest: dict, decoder: str, seed: int) -> None:
    if int(manifest["seed"]) != seed:
        raise RuntimeError(f"seed mismatch for {decoder}/seed{seed}")
    generation = manifest["generation"]
    for key, expected in EXPECTED[(decoder, seed)].items():
        if generation[key] != expected:
            raise RuntimeError(
                f"{decoder}/seed{seed} {key}: {generation[key]} != {expected}"
            )
    if generation["repetition_penalty"] != 1.0:
        raise RuntimeError(f"unexpected repetition penalty for {decoder}/seed{seed}")
    if int(generation["gt_prefix_beats"]) != 8:
        raise RuntimeError(f"unexpected GT prefix for {decoder}/seed{seed}")


def _merge_singleton_manifests(
    inference_root: Path,
    output_dir: Path,
) -> dict[tuple[str, int], Path]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for manifest_path in sorted(inference_root.rglob("*_manifest.json")):
        relative = manifest_path.relative_to(inference_root)
        decoder = relative.parts[0]
        seed = int(relative.parts[1].removeprefix("seed"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_generation(manifest, decoder, seed)
        if len(manifest["items"]) != 1:
            raise RuntimeError(f"singleton run has {len(manifest['items'])} items")
        grouped[(decoder, seed)].append(manifest)

    if set(grouped) != set(EXPECTED):
        raise RuntimeError(f"run groups {sorted(grouped)} != expected {sorted(EXPECTED)}")

    merged_root = output_dir / "_technical" / "merged_manifests"
    merged_root.mkdir(parents=True, exist_ok=True)
    merged_paths = {}
    for key, manifests in grouped.items():
        if len(manifests) != 12:
            raise RuntimeError(f"expected 12 singleton runs for {key}, found {len(manifests)}")
        decoder, seed = key
        merged = dict(manifests[0])
        merged["num_samples"] = 12
        merged["items"] = [item for manifest in manifests for item in manifest["items"]]
        merged["singleton_manifest_count"] = len(manifests)
        path = merged_root / f"{decoder}__seed{seed}.json"
        path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        merged_paths[key] = path
    return merged_paths


def _summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "baseline":
            continue
        groups[(row["decoder"], float(row["selection_rate_requested"]))].append(row)

    summary = []
    for (decoder, rate), group in sorted(groups.items()):
        summary.append(
            {
                "decoder": decoder,
                "selection_rate_requested": rate,
                "sample_count": len(group),
                "piece_count": len({row["piece_id"] for row in group}),
                "decoder_seed_count": len({row["decoder_seed"] for row in group}),
                "empty_output_count": sum(bool(row["output_empty"]) for row in group),
                "exact_match_count": sum(
                    bool(row["acc_exact_match_to_baseline"]) for row in group
                ),
                "mean_note_record_jaccard": sum(
                    float(row["acc_note_record_jaccard"]) for row in group
                )
                / len(group),
                "mean_pianoroll_cell_jaccard": sum(
                    float(row["acc_pianoroll_cell_jaccard"]) for row in group
                )
                / len(group),
                "mean_continuation_exact_beat_rate": sum(
                    float(row["continuation_exact_beat_rate"]) for row in group
                )
                / len(group),
            }
        )
    return summary


def _prepare_blind_bundle(rows: list[dict], output_dir: Path) -> None:
    rng = random.Random(20260806)
    by_piece: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_piece[str(row["piece_id"])].append(row)

    rating_rows = []
    key_rows = []
    order = 0
    listening_root = output_dir / "_technical" / "blind_listening"
    for piece_index, piece_id in enumerate(sorted(by_piece), start=1):
        piece_rows = list(by_piece[piece_id])
        rng.shuffle(piece_rows)
        piece_dir = listening_root / f"{piece_index:02d}_{piece_id}"
        piece_dir.mkdir(parents=True, exist_ok=True)
        for blind_index, row in enumerate(piece_rows, start=1):
            order += 1
            filename = f"{blind_index:02d}.mid"
            destination = piece_dir / filename
            shutil.copy2(row["generated_midi"], destination)
            relative = destination.relative_to(output_dir)
            rating_rows.append(
                {
                    "order": order,
                    "piece_id": piece_id,
                    "midi": str(relative),
                    "variety_1_low_5_high": "",
                    "accompaniment_quality_1_low_5_high": "",
                    "musical_quality_1_low_5_high": "",
                    "melody_accompaniment_fit_1_low_5_high": "",
                    "stability_1_low_5_high": "",
                    "comments": "",
                }
            )
            key_rows.append(
                {
                    "order": order,
                    "piece_id": piece_id,
                    "midi": str(relative),
                    "decoder": row["decoder"],
                    "decoder_seed": row["decoder_seed"],
                    "condition": row["condition"],
                    "selection_rate_requested": row["selection_rate_requested"],
                }
            )
    _write_csv(
        output_dir / "_technical" / "blind_ratings.csv",
        rating_rows,
        fieldnames=[
            "order",
            "piece_id",
            "midi",
            "variety_1_low_5_high",
            "accompaniment_quality_1_low_5_high",
            "musical_quality_1_low_5_high",
            "melody_accompaniment_fit_1_low_5_high",
            "stability_1_low_5_high",
            "comments",
        ],
    )
    _write_csv(output_dir / "_technical" / "listening_key.csv", key_rows)


def _condition_sort_key(row: dict) -> int:
    if row["condition"] == "baseline":
        return 0
    return 1 if float(row["selection_rate_requested"]) == 0.10 else 2


def _prepare_labeled_bundle(rows: list[dict], output_dir: Path) -> None:
    by_piece: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_piece[str(row["piece_id"])].append(row)

    rating_rows = []
    order = 0
    listening_root = output_dir / "listening_by_decoder"
    for piece_index, piece_id in enumerate(sorted(by_piece), start=1):
        piece_dir = listening_root / f"{piece_index:02d}_{piece_id}"
        piece_rows = sorted(
            by_piece[piece_id],
            key=lambda row: (
                0 if row["decoder"] == "greedy" else 1,
                int(row["decoder_seed"]),
                _condition_sort_key(row),
            ),
        )
        for row in piece_rows:
            order += 1
            decoder = str(row["decoder"])
            decoder_seed = int(row["decoder_seed"])
            decoder_dir = piece_dir / f"{decoder}_seed{decoder_seed}"
            decoder_dir.mkdir(parents=True, exist_ok=True)
            condition_index = _condition_sort_key(row) + 1
            condition_label = {
                1: "clean",
                2: "perturb10pct",
                3: "perturb50pct",
            }[condition_index]
            destination = decoder_dir / f"{condition_index:02d}_{condition_label}.mid"
            shutil.copy2(row["generated_midi"], destination)
            relative = destination.relative_to(output_dir)
            rating_rows.append(
                {
                    "order": order,
                    "piece_id": piece_id,
                    "decoder": decoder,
                    "decoder_seed": decoder_seed,
                    "condition": condition_label,
                    "midi": str(relative),
                    "variety_1_low_5_high": "",
                    "accompaniment_quality_1_low_5_high": "",
                    "musical_quality_1_low_5_high": "",
                    "melody_accompaniment_fit_1_low_5_high": "",
                    "stability_1_low_5_high": "",
                    "comments": "",
                }
            )
    _write_csv(
        output_dir / "ratings.csv",
        rating_rows,
        fieldnames=[
            "order",
            "piece_id",
            "decoder",
            "decoder_seed",
            "condition",
            "midi",
            "variety_1_low_5_high",
            "accompaniment_quality_1_low_5_high",
            "musical_quality_1_low_5_high",
            "melody_accompaniment_fit_1_low_5_high",
            "stability_1_low_5_high",
            "comments",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ignore-prefix-beats", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_paths = _merge_singleton_manifests(args.inference_root, args.output_dir)
    all_rows = []
    run_summaries = []
    for (decoder, seed), manifest_path in sorted(merged_paths.items()):
        run_dir = args.output_dir / "_technical" / "by_run" / f"{decoder}__seed{seed}"
        rows, summary = analyze(
            args.prepared_manifest,
            manifest_path,
            run_dir,
            ignore_prefix_beats=args.ignore_prefix_beats,
        )
        for row in rows:
            row["decoder"] = decoder
            row["decoder_seed"] = seed
        summary["decoder"] = decoder
        summary["decoder_seed"] = seed
        all_rows.extend(rows)
        run_summaries.append(summary)

    aggregate = _summarize(all_rows)
    _write_csv(args.output_dir / "per_condition.csv", all_rows)
    (args.output_dir / "per_condition.json").write_text(
        json.dumps(all_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "total_outputs": len(all_rows),
                "run_count": len(run_summaries),
                "ignore_prefix_beats": args.ignore_prefix_beats,
                "aggregate_by_decoder_and_rate": aggregate,
                "runs": run_summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _prepare_blind_bundle(all_rows, args.output_dir)
    _prepare_labeled_bundle(all_rows, args.output_dir)
    print(json.dumps({"total_outputs": len(all_rows), "aggregate": aggregate}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
