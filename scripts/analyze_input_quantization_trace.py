#!/usr/bin/env python3
"""Summarize StreamMUSE input quantization sidecar traces."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TRACE_NAME = "input_quantization_trace.jsonl"
EPSILON = 1e-9


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def discover_trace_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            paths.update(candidate.resolve() for candidate in path.rglob(TRACE_NAME))
        else:
            raise FileNotFoundError(path)
    if not paths:
        raise FileNotFoundError(f"no {TRACE_NAME} files found")
    return sorted(paths)


def load_trace(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        required = {
            "bpm",
            "event_type",
            "ticks_per_beat",
            "signed_error_ticks",
            "signed_error_ms",
            "snapped_forward",
        }
        missing = required.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{path}:{line_number}: missing fields: {names}")
        rows.append(row)
    return rows


def summarize(inputs: Iterable[Path]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float, int, str], list[dict[str, object]]] = (
        defaultdict(list)
    )
    trace_paths: dict[tuple[str, str, float, int, str], str] = {}
    for path in discover_trace_paths(inputs):
        for row in load_trace(path):
            key = (
                str(path.parent),
                str(row.get("service", "unknown")),
                float(row["bpm"]),
                int(row["ticks_per_beat"]),
                str(row["event_type"]),
            )
            grouped[key].append(row)
            trace_paths[key] = str(path)

    summaries: list[dict[str, object]] = []
    for key in sorted(grouped):
        session_path, service, bpm, ticks_per_beat, event_type = key
        rows = grouped[key]
        signed_ticks = [float(row["signed_error_ticks"]) for row in rows]
        signed_ms = [float(row["signed_error_ms"]) for row in rows]
        absolute_ticks = [abs(value) for value in signed_ticks]
        absolute_ms = [abs(value) for value in signed_ms]
        count = len(rows)
        summaries.append(
            {
                "session": Path(session_path).name,
                "session_path": session_path,
                "trace_path": trace_paths[key],
                "service": service,
                "event_type": event_type,
                "bpm": bpm,
                "ticks_per_beat": ticks_per_beat,
                "event_count": count,
                "mean_signed_error_ticks": sum(signed_ticks) / count,
                "p50_signed_error_ticks": percentile(signed_ticks, 0.50),
                "p95_signed_error_ticks": percentile(signed_ticks, 0.95),
                "mean_signed_error_ms": sum(signed_ms) / count,
                "p50_signed_error_ms": percentile(signed_ms, 0.50),
                "p95_signed_error_ms": percentile(signed_ms, 0.95),
                "mean_absolute_error_ticks": sum(absolute_ticks) / count,
                "p50_absolute_error_ticks": percentile(absolute_ticks, 0.50),
                "p95_absolute_error_ticks": percentile(absolute_ticks, 0.95),
                "mean_absolute_error_ms": sum(absolute_ms) / count,
                "p50_absolute_error_ms": percentile(absolute_ms, 0.50),
                "p95_absolute_error_ms": percentile(absolute_ms, 0.95),
                "quantized_earlier_rate": sum(
                    value < -EPSILON for value in signed_ticks
                )
                / count,
                "quantized_later_rate": sum(
                    value > EPSILON for value in signed_ticks
                )
                / count,
                "on_grid_rate": sum(abs(value) <= EPSILON for value in signed_ticks) / count,
                "snap_rate": sum(bool(row["snapped_forward"]) for row in rows) / count,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows: list[dict[str, object]]) -> None:
    print(
        "| Session | Service | Event | BPM | N | Signed mean (ms) | "
        "Abs. p50/p95 (ms) | Quantized earlier | Quantized later | Snap |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['session']} | {row['service']} | {row['event_type']} | "
            f"{float(row['bpm']):g} | "
            f"{row['event_count']} | {float(row['mean_signed_error_ms']):.3f} | "
            f"{float(row['p50_absolute_error_ms']):.3f}/"
            f"{float(row['p95_absolute_error_ms']):.3f} | "
            f"{float(row['quantized_earlier_rate']):.3f} | "
            f"{float(row['quantized_later_rate']):.3f} | "
            f"{float(row['snap_rate']):.3f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=f"Trace file or directory recursively containing {TRACE_NAME}",
    )
    parser.add_argument("--csv-out", type=Path, help="Optional summary CSV path")
    parser.add_argument("--json-out", type=Path, help="Optional summary JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = summarize(args.inputs)
    print_markdown(rows)
    if args.csv_out:
        write_csv(args.csv_out, rows)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
