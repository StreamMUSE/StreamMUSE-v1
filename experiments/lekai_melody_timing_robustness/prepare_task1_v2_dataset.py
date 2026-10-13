#!/usr/bin/env python3
"""Prepare the frozen four-piece Task 1 v2 paired perturbation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.lekai_melody_timing_robustness.prepare_task1_dataset import (
    PieceSpec,
    prepare_dataset,
)


FROZEN_PIECES = (
    PieceSpec("5889045", "09-7__5889045_mel.mid", "09-7__5889045_acc.mid"),
    PieceSpec("5817083", "12-6__5817083_mel.mid", "12-6__5817083_acc.mid"),
    PieceSpec("332891", "13-6__332891_mel.mid", "13-6__332891_acc.mid"),
    PieceSpec("5472153", "15-7__5472153_mel.mid", "15-7__5472153_acc.mid"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "/data/home/yuanxin/RT-accompanimentV2/"
            "formal_pop909_sanity_20260710/inputs/scored19"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows = prepare_dataset(
        args.input_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
        pieces=FROZEN_PIECES,
        selection_rates=(0.10, 0.50),
        seeds=(0,),
        shift_steps=1,
    )
    summary = {
        "piece_ids": [piece.piece_id for piece in FROZEN_PIECES],
        "conditions": ["baseline", "onset_tick1_pct10_seed0", "onset_tick1_pct50_seed0"],
        "sample_count": len(rows),
        "baseline_count": sum(row["condition"] == "baseline" for row in rows),
        "perturbed_count": sum(row["condition"] != "baseline" for row in rows),
        "perturbation": {
            "model_steps_per_beat": 4,
            "shift_steps": 1,
            "selection_rates": [0.10, 0.50],
            "perturbation_seed": 0,
        },
    }
    (args.output_root / "frozen_design.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
