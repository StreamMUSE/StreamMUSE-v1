#!/usr/bin/env python3
"""Freeze, bind artifact roots for, and evaluate the SGLang-Omni MOSS A/B campaign.

Subcommands
-----------
freeze        Freeze the corpus, runtime pins, schedule, statistics, and RAP A/B gates.
prepare-root  Create one unused, immutable-identity artifact root for a frozen A/B run.
evaluate      Evaluate immutable A/B evidence against the frozen gates.

Exit codes: 0 success (evaluate: promotion-candidate), 1 evaluate decision
blocked/experimental, 2 malformed or untrusted input.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from streammuse.experiments.sglang_moss_acceptance import (
    AcceptanceDataError,
    ExperimentDesignError,
    evaluate_acceptance,
    freeze_experiment_manifest,
    load_bounded_json,
    load_bounded_jsonl,
    prepare_artifact_root,
    write_immutable_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SGLang-Omni MOSS A/B campaign tooling")
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser(
        "freeze", help="Freeze a reproducible SGLang-Omni MOSS A/B design"
    )
    freeze.add_argument("--experiment-id", required=True)
    freeze.add_argument("--implementation-id", required=True)
    freeze.add_argument("--created-at-utc", required=True)
    freeze.add_argument("--corpus", required=True)
    freeze.add_argument("--pins", required=True)
    freeze.add_argument("--schedule-seed", type=int, required=True)
    freeze.add_argument("--bootstrap-seed", type=int, required=True)
    freeze.add_argument("--bootstrap-resamples", type=int, default=10_000)
    freeze.add_argument("--cohort-a-enabled", action="store_true")
    freeze.add_argument("--cohort-a-evidence", default="not_supported")
    freeze.add_argument("--speaker-similarity-hard-gate", action="store_true")
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(handler=_freeze)

    prepare_root = commands.add_parser(
        "prepare-root",
        help="Prepare a fresh artifact root bound to one frozen A/B coordinate",
    )
    prepare_root.add_argument("--manifest", required=True)
    prepare_root.add_argument("--root", required=True)
    prepare_root.add_argument("--cohort", required=True)
    prepare_root.add_argument("--phase", choices=("qualification", "final"), required=True)
    prepare_root.add_argument("--block-id", required=True)
    prepare_root.add_argument("--backend", choices=("baseline", "candidate"), required=True)
    prepare_root.set_defaults(handler=_prepare_root)

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a frozen SGLang-Omni MOSS A/B campaign"
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--rows", required=True)
    evaluate.add_argument("--artifact-roots", required=True)
    evaluate.add_argument("--recovery-probes", required=True)
    evaluate.add_argument("--blind-scores", required=True)
    evaluate.add_argument("--mac-evidence")
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


def _freeze(args: argparse.Namespace) -> int:
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


def _prepare_root(args: argparse.Namespace) -> int:
    try:
        manifest = load_bounded_json(args.manifest)
        if not isinstance(manifest, Mapping):
            raise AcceptanceDataError("experiment manifest must be an object")
        root_manifest = prepare_artifact_root(
            args.root,
            manifest,
            cohort=args.cohort,
            phase=args.phase,
            block_id=args.block_id,
            backend=args.backend,
        )
    except (AcceptanceDataError, ExperimentDesignError, OSError, ValueError) as exc:
        print(f"artifact root preparation failed: {exc}", file=sys.stderr)
        return 2
    print(f"root_manifest_sha256={root_manifest['root_manifest_sha256']}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    try:
        manifest = load_bounded_json(args.manifest)
        if not isinstance(manifest, Mapping):
            raise AcceptanceDataError("experiment manifest must be an object")
        rows = load_bounded_jsonl(args.rows)
        roots = load_bounded_jsonl(args.artifact_roots)
        probes = load_bounded_jsonl(args.recovery_probes)
        scores = load_bounded_jsonl(args.blind_scores)
        mac = load_bounded_json(args.mac_evidence) if args.mac_evidence else None
        if mac is not None and not isinstance(mac, Mapping):
            raise AcceptanceDataError("Mac E2E evidence must be an object")
        report = evaluate_acceptance(
            manifest,
            rows,
            artifact_roots=roots,
            recovery_probes=probes,
            blind_scores=scores,
            mac_evidence=mac,
        )
        write_immutable_json(Path(args.output), report)
    except (AcceptanceDataError, ExperimentDesignError, OSError, ValueError) as exc:
        print(f"acceptance evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(f"decision={report['decision']}")
    print(f"report_sha256={report['report_sha256']}")
    return 0 if report["decision"] == "promotion-candidate" else 1


if __name__ == "__main__":
    raise SystemExit(main())
