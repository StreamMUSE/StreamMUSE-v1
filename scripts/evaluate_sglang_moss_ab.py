#!/usr/bin/env python3
"""Evaluate immutable SGLang-Omni MOSS A/B evidence against frozen gates."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from streammuse.experiments.sglang_moss_acceptance import (
    AcceptanceDataError,
    ExperimentDesignError,
    evaluate_acceptance,
    load_bounded_json,
    load_bounded_jsonl,
    write_immutable_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen SGLang-Omni MOSS A/B campaign"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--artifact-roots", required=True)
    parser.add_argument("--recovery-probes", required=True)
    parser.add_argument("--blind-scores", required=True)
    parser.add_argument("--mac-evidence")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
