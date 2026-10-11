#!/usr/bin/env python
"""Prepare a balanced new-piece subset and import compatible prior ratings."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_NEW_PIECES = "5817083,332891,43038,2581346"
PIECE_SEED_PAIRS = [
    ("0", "1"),
    ("2", "3"),
    ("0", "2"),
    ("1", "3"),
]
SCORE_FIELDS = [
    "overall_score_1_to_5",
    "melody_fit_1_to_5",
    "accompaniment_quality_1_to_5",
    "failure_reason",
    "comments",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--discovery-run-dir",
        required=True,
        help="Prior pilot used to discover rules; its pieces and ratings are excluded.",
    )
    parser.add_argument("--output-name", default="single_rater_new4_rule_eval24")
    parser.add_argument("--new-piece-ids", default=DEFAULT_NEW_PIECES)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["piece_id"], row["prompt_seed"], row["continuation_seed"]


def select_balanced_rows(
    continuation_rows: list[dict[str, str]],
    piece_ids: list[str],
    seed: int,
) -> list[dict[str, str]]:
    by_key = {sample_key(row): row for row in continuation_rows}
    selected_by_piece: dict[str, list[dict[str, str]]] = defaultdict(list)
    # Every Prompt for a piece uses the same continuation seed pair. Across
    # four pieces, each continuation seed still appears exactly six times.
    for piece_index, piece_id in enumerate(piece_ids):
        for prompt_seed in ("0", "1", "2"):
            for continuation_seed in PIECE_SEED_PAIRS[piece_index]:
                key = (piece_id, prompt_seed, continuation_seed)
                if key not in by_key:
                    raise RuntimeError(f"missing continuation: {key}")
                selected_by_piece[piece_id].append(by_key[key])

    rng = random.Random(seed)
    for piece_id in piece_ids:
        rng.shuffle(selected_by_piece[piece_id])

    ordered: list[dict[str, str]] = []
    previous_piece: str | None = None
    for round_index in range(6):
        piece_order = piece_ids.copy()
        for _ in range(100):
            rng.shuffle(piece_order)
            if piece_order[0] != previous_piece:
                break
        for piece_id in piece_order:
            ordered.append(selected_by_piece[piece_id][round_index])
        previous_piece = piece_order[-1]
    assert len(ordered) == 24
    assert all(
        left["piece_id"] != right["piece_id"]
        for left, right in zip(ordered, ordered[1:])
    )
    return ordered


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    discovery_dir = Path(args.discovery_run_dir).resolve()
    output_dir = run_dir / args.output_name
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    listening_dir = output_dir / "blind_listening"
    ratings_dir = output_dir / "ratings"
    listening_dir.mkdir(parents=True)
    ratings_dir.mkdir()

    continuation_rows = read_csv(run_dir / "continuation_manifest.csv")
    current_mapping = read_csv(run_dir / "blind_mapping_private.csv")
    current_by_key = {sample_key(row): row for row in current_mapping}
    piece_ids = [
        item.strip() for item in args.new_piece_ids.split(",") if item.strip()
    ]
    discovery_piece_ids = {
        row["piece_id"]
        for row in read_csv(discovery_dir / "continuation_manifest.csv")
    }
    overlap = discovery_piece_ids.intersection(piece_ids)
    if overlap:
        raise RuntimeError(
            f"held-out pieces overlap discovery set: {sorted(overlap)}"
        )
    selected = select_balanced_rows(continuation_rows, piece_ids, args.seed)

    public_rows = []
    private_rows = []
    for row in selected:
        mapped = current_by_key[sample_key(row)]
        blind_id = mapped["blind_id"]
        source = run_dir / "blind_listening" / f"{blind_id}.mid"
        target = listening_dir / f"{blind_id}.mid"
        shutil.copy2(source, target)
        public_rows.append(
            {
                "rater_id": "user_top7_new4",
                "blind_id": blind_id,
                **{field: "" for field in SCORE_FIELDS},
            }
        )
        private_rows.append(mapped)

    rating_fields = ["rater_id", "blind_id", *SCORE_FIELDS]
    write_csv(ratings_dir / "new4_to_rate.csv", rating_fields, public_rows)
    write_csv(
        output_dir / "selection_mapping_private.csv",
        list(private_rows[0]),
        private_rows,
    )

    selection_summary = {
        "evaluation_role": "held_out_rule_validation",
        "discovery_piece_ids_excluded": sorted(discovery_piece_ids),
        "new_piece_ids": piece_ids,
        "new_samples_to_rate": len(public_rows),
        "prior_ratings_imported": 0,
        "evaluation_rating_count_after_completion": len(public_rows),
        "prompts_per_new_piece": 3,
        "continuations_per_prompt": 2,
        "continuation_seeds_paired_across_prompts_within_piece": True,
        "piece_continuation_seed_pairs": {
            piece_id: list(PIECE_SEED_PAIRS[index])
            for index, piece_id in enumerate(piece_ids)
        },
        "continuation_seed_counts": {
            seed: sum(row["continuation_seed"] == seed for row in selected)
            for seed in ("0", "1", "2", "3")
        },
        "adjacent_same_piece_count": sum(
            left["piece_id"] == right["piece_id"]
            for left, right in zip(selected, selected[1:])
        ),
        "discovery_evaluation_piece_overlap": [],
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(selection_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    guide = """# Single-Rater Q1 Expansion

Only `ratings/new4_to_rate.csv` is used in this held-out evaluation.

- Follow the CSV row order and open the matching MIDI in `blind_listening/`.
- The order contains one sample from each of the four pieces per four-row block.
- No adjacent rows come from the same piece.
- Within each piece, all three Prompt candidates use the same two continuation
  seeds so selector comparisons are paired.
- The prior 3-piece pilot is the rule-discovery set and is excluded from all
  held-out statistics.
- Do not open `selection_mapping_private.csv` before rating is complete.
- Use exact numeric scores; use `0` only for complete generation failure.

After rating, analyze only the 24 held-out ratings:

```bash
uv run --isolated --no-project \
  --with pretty-midi --with pandas --with scipy \
  python experiments/q1_prompt_batch_selection_pilot/analyze_ratings.py \
  --run-dir remote_results/q1_prompt_batch_selection_top7_20260724 \
  --ratings-dir single_rater_new4_rule_eval24/ratings
```
"""
    (output_dir / "README.md").write_text(guide, encoding="utf-8")
    heldout_plan = """# Held-Out Rule Evaluation Plan

## Dataset separation

- The prior 3-piece pilot is the discovery set.
- This 4-piece bundle is the held-out evaluation set.
- Discovery ratings are excluded from all held-out statistics.

## Primary question

Does sampling `N=3` Prompt candidates and selecting one with the frozen Rule-S
produce better continuation quality than ordinary `N=1` Prompt sampling?

- `N=1` is estimated by the mean score of the three exchangeable Prompt
  candidates in each piece.
- `N=3 + Rule-S` is the score of the candidate selected before held-out rating.
- The scientific variable is the candidate count and selector, not whether the
  Prompt candidates are generated sequentially or in one GPU batch.

## Frozen rule-based selector

For the three Prompt candidates generated from the same Melody:

1. Reject candidates with zero accompaniment notes.
2. Convert each remaining candidate metric to its normalized average rank
   within the same piece.
3. Compute the discovery-correlation weighted score:
   `-0.450 * rank(PPL) + 0.477 * rank(Pitch Range)
   + 0.467 * rank(Pitch-Class Entropy)
   + 0.496 * rank(Average Voice Number)`.
4. Select the highest score; break ties using lower PPL.

The weights are piece-centered Spearman correlations estimated from the prior
3-piece discovery set. No threshold, sign, or weight is fitted from held-out
ratings.

## Baselines

- `N=1` random Prompt sampling (primary baseline).
- Raw minimum PPL, including empty candidates.
- Empty-filtered minimum PPL.
- Human oracle after rating.

## Paired evaluation

Within each piece, all three Prompt candidates use the same two continuation
seeds. Each Prompt score is the mean of those two paired continuation ratings.

Report selector mean score, gain over random, regret to oracle, and the number
of pieces where each selector chooses the human-best Prompt.
"""
    (output_dir / "HELDOUT_EVALUATION_PLAN.md").write_text(
        heldout_plan, encoding="utf-8"
    )
    print(json.dumps(selection_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
