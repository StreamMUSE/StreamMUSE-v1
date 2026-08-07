"""Opt-in CLI for anonymous structural MCFlow template extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from streammuse.infrastructure.rap.mcflow import extract_mcflow_directory, write_extracted_templates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract anonymous MCFlow flow templates")
    parser.add_argument("--mcflow-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-quantization-error-ticks", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.mcflow_dir.is_dir():
        print("error: mcflow directory must be an existing directory", file=sys.stderr)
        return 2
    try:
        extraction = extract_mcflow_directory(
            args.mcflow_dir,
            max_quantization_error_ticks=args.max_quantization_error_ticks,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_extracted_templates(extraction, args.output)
    aggregate = extraction.to_dict()["aggregate"]
    print(
        f"parsed_files={aggregate['parsed_files']} "
        f"accepted_templates={aggregate['accepted_templates']} "
        f"rejected_measures={aggregate['rejected_measures']} "
        f"output={args.output}"
    )
    return 0 if extraction.templates else 1


if __name__ == "__main__":
    raise SystemExit(main())
