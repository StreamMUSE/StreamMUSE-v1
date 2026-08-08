#!/usr/bin/env python3
"""Regenerate deterministic summaries from a recorded rap research session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from streammuse.infrastructure.rap.recorder import (
    derive_bar_rows,
    derive_summary,
    read_events,
    validate_session_manifest,
    write_bar_csv,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate summary and per-bar artifacts from canonical rap events."
    )
    parser.add_argument("session_dir", type=Path, help="Recorded rap session directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_dir = args.session_dir
    manifest = _read_manifest(session_dir / "session.json")
    events = read_events(session_dir / "events.jsonl")

    summary = derive_summary(
        events,
        expected_manifest_window=manifest["repetition_window_bars"],
    )
    bar_rows = derive_bar_rows(events)
    write_json(session_dir / "summary.regenerated.json", summary)
    write_bar_csv(session_dir / "bars.regenerated.csv", bar_rows)
    return 0


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("session manifest must be a JSON object")
    return validate_session_manifest(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
