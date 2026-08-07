"""Build a local, density-balanced anonymous MCFlow sample catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from streammuse.infrastructure.rap.mcflow import extract_mcflow_directory
from streammuse.infrastructure.rap.sample_catalog import select_sample_templates, write_sample_catalog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local anonymous MCFlow sample catalog")
    parser.add_argument("--mcflow-dir", required=True, type=Path)
    parser.add_argument("--catalog-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--per-bucket", type=int, default=10)
    parser.add_argument("--max-quantization-error-ticks", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    if not args.mcflow_dir.is_dir():
        print("error: mcflow directory must be an existing directory", file=sys.stderr)
        return 2
    try:
        extraction = extract_mcflow_directory(
            args.mcflow_dir,
            max_quantization_error_ticks=args.max_quantization_error_ticks,
        )
    except OSError:
        print("error: unable to read MCFlow input", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        selection = select_sample_templates(extraction.templates, per_bucket=args.per_bucket)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        write_sample_catalog(extraction, selection, args.catalog_output, args.report_output)
    except (OSError, ValueError):
        print("error: unable to write sample catalog", file=sys.stderr)
        return 2

    report = selection.report
    print(
        f"parsed_files={extraction.parsed_files} "
        f"accepted_templates={len(extraction.templates)} "
        f"rejected_measures={len(extraction.rejections)} "
        f"structurally_unique_templates={report.structurally_unique_templates} "
        f"duplicates_removed={report.duplicates_removed} "
        f"out_of_range_templates={report.out_of_range_templates} "
        f"selected_templates={report.selected_templates} "
        f"catalog_output={args.catalog_output} "
        f"report_output={args.report_output}"
    )
    return 1 if not selection.templates or any(count.available == 0 for count in report.band_counts.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
