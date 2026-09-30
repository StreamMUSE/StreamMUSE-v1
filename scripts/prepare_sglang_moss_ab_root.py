#!/usr/bin/env python3
"""Create one unused, immutable-identity artifact root for a frozen A/B run."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence

from streammuse.experiments.sglang_moss_acceptance import (
    AcceptanceDataError,
    ExperimentDesignError,
    load_bounded_json,
    prepare_artifact_root,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a fresh artifact root bound to one frozen A/B coordinate"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--phase", choices=("qualification", "final"), required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--backend", choices=("baseline", "candidate"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
