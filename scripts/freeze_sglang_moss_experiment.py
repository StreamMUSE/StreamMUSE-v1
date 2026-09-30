#!/usr/bin/env python3
"""Freeze the corpus, runtime pins, schedule, statistics, and RAP A/B gates."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from streammuse.experiments.sglang_moss_acceptance import (
    ExperimentDesignError,
    freeze_experiment_manifest,
    load_bounded_json,
    write_immutable_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a reproducible SGLang-Omni MOSS A/B design"
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--implementation-id", required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--pins", required=True)
    parser.add_argument("--schedule-seed", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--cohort-a-enabled", action="store_true")
    parser.add_argument("--cohort-a-evidence", default="not_supported")
    parser.add_argument("--speaker-similarity-hard-gate", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        corpus = load_bounded_json(args.corpus)
        pins = load_bounded_json(args.pins)
        if not isinstance(corpus, list) or not all(
            isinstance(item, Mapping) for item in corpus
        ):
            raise ExperimentDesignError("corpus JSON must be an array of objects")
        if not isinstance(pins, Mapping):
            raise ExperimentDesignError("pins JSON must be an object")
        manifest = freeze_experiment_manifest(
            experiment_id=args.experiment_id,
            implementation_id=args.implementation_id,
            created_at_utc=args.created_at_utc,
            corpus=corpus,
            pins=pins,
            schedule_seed=args.schedule_seed,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
            cohort_a_enabled=args.cohort_a_enabled,
            cohort_a_evidence=args.cohort_a_evidence,
            speaker_similarity_hard_gate=args.speaker_similarity_hard_gate,
        )
        write_immutable_json(Path(args.output), manifest)
    except (ExperimentDesignError, OSError, ValueError) as exc:
        print(f"experiment freeze failed: {exc}", file=sys.stderr)
        return 2
    print(f"experiment_manifest_sha256={manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
