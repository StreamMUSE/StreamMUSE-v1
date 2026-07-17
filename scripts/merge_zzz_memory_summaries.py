#!/usr/bin/env python3
"""Merge ZipZapZop memory sweep summaries under one matrix root."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge zzz memory sweep summaries")
    parser.add_argument("--root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("sweep_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        config_path = summary_path.parent / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        for row in payload.get("summary", []):
            merged = dict(row)
            merged["summary_path"] = str(summary_path)
            merged["sweep_root"] = str(summary_path.parent)
            merged["effective_max_tokens"] = config.get("effective_max_tokens")
            merged["pilot"] = config.get("pilot")
            rows.append(merged)
        for record in payload.get("runs", []):
            if record.get("status") == "error":
                failures.append({"summary_path": str(summary_path), **record})

    formal_rows = [row for row in rows if not bool(row.get("pilot"))]
    write_json(root / "matrix_summary.json", {"summary": rows, "failures": failures})
    write_csv(root / "matrix_summary.csv", rows)
    (root / "matrix_summary.md").write_text(render_markdown(rows, failures), encoding="utf-8")
    write_json(root / "matrix_summary_formal.json", {"summary": formal_rows, "failures": failures})
    write_csv(root / "matrix_summary_formal.csv", formal_rows)
    (root / "matrix_summary_formal.md").write_text(
        render_markdown(formal_rows, failures, title="ZipZapZop Formal Memory Matrix Summary"),
        encoding="utf-8",
    )
    print(f"merged {len(rows)} summary rows under {root}")
    return 0 if not failures else 1


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "apc",
        "spec_decode",
        "temperature",
        "config",
        "oracle",
        "n_runs",
        "turns",
        "strict_accuracy_mean",
        "strict_accuracy_std",
        "normalized_accuracy_mean",
        "normalized_accuracy_std",
        "first_error_turn_min",
        "accuracy_before_first_error_mean",
        "post_error_accuracy_mean",
        "latency_median_ms",
        "latency_p95_ms",
        "latency_max_ms",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "tokens_per_second_median",
        "effective_max_tokens",
        "pilot",
        "summary_path",
        "sweep_root",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def render_markdown(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    title: str = "ZipZapZop Memory Matrix Summary",
) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- summary rows: {len(rows)}")
    lines.append(f"- failed run cells: {len(failures)}")
    lines.append("")
    if failures:
        lines.append("## Failures")
        lines.append("")
        for failure in failures:
            lines.append(
                f"- {failure.get('summary_path')} cell={failure.get('cell_order_index')} "
                f"config={failure.get('config')} error={failure.get('error')}"
            )
        lines.append("")
    lines.append("## Summary")
    lines.append("")
    headers = ["model", "apc", "spec", "temp", "config", "oracle", "strict", "norm", "median_ms", "p95_ms"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model")),
                    str(row.get("apc")),
                    str(row.get("spec_decode")),
                    fmt(row.get("temperature")),
                    str(row.get("config")),
                    str(row.get("oracle")),
                    fmt(row.get("strict_accuracy_mean")),
                    fmt(row.get("normalized_accuracy_mean")),
                    fmt(row.get("latency_median_ms")),
                    fmt(row.get("latency_p95_ms")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
