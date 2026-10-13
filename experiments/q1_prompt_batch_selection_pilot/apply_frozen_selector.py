#!/usr/bin/env python
"""Apply and freeze held-out Prompt selectors before listening ratings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pretty_midi


DEFAULT_PIECES = "5817083,332891,43038,2581346"
DISCOVERY_PIECE_IDS = ["3410421", "5472153", "5889045"]
DISCOVERY_PIECE_CENTERED_SPEARMAN = {
    "prompt_ppl_numeric": -0.45,
    "acc_pitch_range": 0.47699162299076214,
    "acc_pitch_class_entropy": 0.4666666666666666,
    "acc_average_voice_number": 0.4958158260214505,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subset-dir", required=True)
    parser.add_argument("--piece-ids", default=DEFAULT_PIECES)
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Explicitly replace an existing frozen selector decision.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accompaniment_features(path: Path, beats: int = 8) -> dict[str, float]:
    midi = pretty_midi.PrettyMIDI(str(path))
    instruments = [
        instrument
        for instrument in midi.instruments
        if instrument.name.strip().lower() == "accompaniment"
    ]
    if not instruments and len(midi.instruments) >= 2:
        instruments = [midi.instruments[1]]
    notes = [note for instrument in instruments for note in instrument.notes]
    if not notes:
        return {
            "acc_note_count": 0,
            "acc_pitch_range": 0.0,
            "acc_pitch_class_entropy": 0.0,
            "acc_average_voice_number": 0.0,
        }

    pitches = np.array([note.pitch for note in notes], dtype=float)
    chroma = np.zeros(12, dtype=float)
    for note in notes:
        chroma[note.pitch % 12] += max(note.end - note.start, 1e-6)
    chroma /= chroma.sum()
    nonzero = chroma[chroma > 0]

    _, tempi = midi.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) else 120.0
    duration = beats * 60.0 / tempo
    boundaries = sorted(
        {0.0, duration}
        | {
            min(duration, max(0.0, time))
            for note in notes
            for time in (note.start, note.end)
            if 0.0 <= time <= duration
        }
    )
    voice_area = 0.0
    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = (start + end) / 2
        active = sum(note.start <= midpoint < note.end for note in notes)
        voice_area += active * (end - start)
    return {
        "acc_note_count": len(notes),
        "acc_pitch_range": float(pitches.max() - pitches.min()),
        "acc_pitch_class_entropy": float(-np.sum(nonzero * np.log2(nonzero))),
        "acc_average_voice_number": voice_area / duration,
    }


def normalized_average_rank(
    candidates: list[dict],
    candidate: dict,
    key: str,
) -> float:
    value = candidate[key]
    lower_count = sum(other[key] < value for other in candidates)
    equal_count = sum(other[key] == value for other in candidates)
    average_rank = 1.0 + lower_count + (equal_count - 1) / 2.0
    return average_rank / len(candidates)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    subset_dir = Path(args.subset_dir)
    if not subset_dir.is_absolute():
        subset_dir = run_dir / subset_dir
    subset_dir = subset_dir.resolve()
    private_dir = subset_dir / "_private"
    private_dir.mkdir(exist_ok=True)
    decision_path = private_dir / "selector_decisions.csv"
    commit_path = private_dir / "selector_commit.json"
    if (
        decision_path.exists() or commit_path.exists()
    ) and not args.overwrite_existing:
        raise FileExistsError("selector decision is already frozen")
    if args.overwrite_existing:
        for rating_path in (subset_dir / "ratings").glob("*.csv"):
            if any(
                row.get("overall_score_1_to_5", "").strip()
                for row in read_csv(rating_path)
            ):
                raise RuntimeError(
                    "cannot overwrite selector decisions after ratings exist"
                )

    piece_ids = [item.strip() for item in args.piece_ids.split(",") if item.strip()]
    prompt_rows = read_csv(run_dir / "prompt_manifest.csv")
    subset_mapping = read_csv(subset_dir / "selection_mapping_private.csv")
    subset_keys = {
        (row["piece_id"], row["prompt_seed"]) for row in subset_mapping
    }

    decisions = []
    selector_summary = []
    for piece_id in piece_ids:
        candidates = [row for row in prompt_rows if row["piece_id"] == piece_id]
        if len(candidates) != 3:
            raise RuntimeError(f"{piece_id}: expected 3 Prompt candidates")
        candidate_stats = []
        for row in candidates:
            features = accompaniment_features(run_dir / row["prompt_midi"])
            candidate_stats.append(
                {
                    **row,
                    **features,
                    "is_empty_prompt": features["acc_note_count"] == 0,
                    "prompt_ppl_numeric": float(row["prompt_ppl"]),
                }
            )
        eligible = [row for row in candidate_stats if not row["is_empty_prompt"]]
        pool = eligible or candidate_stats
        raw_ppl_selected = min(
            candidate_stats, key=lambda row: row["prompt_ppl_numeric"]
        )
        filtered_ppl_selected = min(
            pool, key=lambda row: row["prompt_ppl_numeric"]
        )
        for row in pool:
            row["rule_weighted_rank_score"] = sum(
                correlation
                * normalized_average_rank(pool, row, metric)
                for metric, correlation in DISCOVERY_PIECE_CENTERED_SPEARMAN.items()
            )
        rule_selected = min(
            pool,
            key=lambda row: (
                -row["rule_weighted_rank_score"],
                row["prompt_ppl_numeric"],
            ),
        )

        seed_sets = {
            prompt_seed: {
                row["continuation_seed"]
                for row in subset_mapping
                if row["piece_id"] == piece_id
                and row["prompt_seed"] == prompt_seed
            }
            for prompt_seed in ("0", "1", "2")
        }
        if len({tuple(sorted(seeds)) for seeds in seed_sets.values()}) != 1:
            raise RuntimeError(f"{piece_id}: continuation seeds are not paired")
        if any(len(seeds) != 2 for seeds in seed_sets.values()):
            raise RuntimeError(f"{piece_id}: expected 2 continuation seeds per Prompt")

        for selected in (raw_ppl_selected, filtered_ppl_selected, rule_selected):
            if (piece_id, selected["prompt_seed"]) not in subset_keys:
                raise RuntimeError(
                    f"selected Prompt missing from listening subset: "
                    f"{piece_id}/{selected['prompt_seed']}"
                )
        for row in candidate_stats:
            decisions.append(
                {
                    "piece_id": piece_id,
                    "prompt_seed": row["prompt_seed"],
                    "prompt_index": row["prompt_index"],
                    "acc_note_count": row["acc_note_count"],
                    "acc_pitch_range": row["acc_pitch_range"],
                    "acc_pitch_class_entropy": row["acc_pitch_class_entropy"],
                    "acc_average_voice_number": row["acc_average_voice_number"],
                    "is_empty_prompt": int(row["is_empty_prompt"]),
                    "prompt_ppl": row["prompt_ppl"],
                    "eligible_after_empty_filter": int(not row["is_empty_prompt"]),
                    "rule_weighted_rank_score": (
                        row.get("rule_weighted_rank_score", math.nan)
                    ),
                    "selected_by_raw_min_ppl": int(
                        row["prompt_seed"] == raw_ppl_selected["prompt_seed"]
                    ),
                    "selected_by_empty_filtered_min_ppl": int(
                        row["prompt_seed"] == filtered_ppl_selected["prompt_seed"]
                    ),
                    "selected_by_music_rule": int(
                        row["prompt_seed"] == rule_selected["prompt_seed"]
                    ),
                }
            )
        selector_summary.append(
            {
                "piece_id": piece_id,
                "raw_min_ppl_prompt_seed": raw_ppl_selected["prompt_seed"],
                "empty_filtered_min_ppl_prompt_seed": filtered_ppl_selected[
                    "prompt_seed"
                ],
                "music_rule_prompt_seed": rule_selected["prompt_seed"],
                "music_rule_differs_from_empty_filtered_ppl": (
                    rule_selected["prompt_seed"]
                    != filtered_ppl_selected["prompt_seed"]
                ),
                "paired_continuation_seeds": sorted(
                    next(iter(seed_sets.values()))
                ),
                "all_candidates_empty": not eligible,
            }
        )

    fields = list(decisions[0])
    write_csv(decision_path, fields, decisions)
    commit = {
        "rule_id": "empty_filter_discovery_spearman_weighted_rank_v1",
        "frozen_before_heldout_ratings": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "discovery_piece_ids": DISCOVERY_PIECE_IDS,
        "weight_source": "piece-centered Spearman correlation with overall_mean",
        "metric_weights": DISCOVERY_PIECE_CENTERED_SPEARMAN,
        "heldout_piece_ids": piece_ids,
        "piece_count": len(piece_ids),
        "candidate_prompt_count": len(decisions),
        "selected_prompt_count": len(selector_summary),
        "rule": [
            "Reject Prompt candidates with zero accompaniment notes.",
            "Convert each eligible candidate metric to its normalized average "
            "rank within the same piece.",
            "Multiply ranks by piece-centered Spearman correlations estimated "
            "only from the 3-piece discovery set and sum them.",
            "Select the maximum weighted rank score; break ties using lower PPL.",
        ],
        "baselines": [
            "Random Prompt selection.",
            "Raw minimum PPL, including empty Prompt candidates.",
            "Empty-filtered minimum PPL.",
            "Human oracle after rating.",
        ],
        "continuation_seeds_paired_across_prompts_within_piece": True,
        "prompt_manifest_sha256": sha256(run_dir / "prompt_manifest.csv"),
        "heldout_plan_sha256": sha256(subset_dir / "HELDOUT_EVALUATION_PLAN.md"),
        "selector_decisions_sha256": sha256(decision_path),
        "selector_choices": selector_summary,
        "music_rule_differs_from_empty_filtered_ppl_piece_count": sum(
            row["music_rule_differs_from_empty_filtered_ppl"]
            for row in selector_summary
        ),
    }
    commit_path.write_text(
        json.dumps(commit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Frozen {len(selector_summary)} held-out selector decisions in {private_dir}"
    )


if __name__ == "__main__":
    main()
