from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_analyzer_summarizes_session_error_and_rates(tmp_path: Path) -> None:
    session = tmp_path / "session-1"
    session.mkdir()
    trace_path = session / "input_quantization_trace.jsonl"
    rows = [
        {
            "service": "standard",
            "event_type": "note_on",
            "bpm": 120.0,
            "ticks_per_beat": 4,
            "signed_error_ticks": -0.2,
            "signed_error_ms": -25.0,
            "snapped_forward": False,
        },
        {
            "service": "standard",
            "event_type": "note_on",
            "bpm": 120.0,
            "ticks_per_beat": 4,
            "signed_error_ticks": 0.0,
            "signed_error_ms": 0.0,
            "snapped_forward": False,
        },
        {
            "service": "standard",
            "event_type": "note_on",
            "bpm": 120.0,
            "ticks_per_beat": 4,
            "signed_error_ticks": 0.4,
            "signed_error_ms": 50.0,
            "snapped_forward": True,
        },
        {
            "service": "standard",
            "event_type": "note_off",
            "bpm": 120.0,
            "ticks_per_beat": 4,
            "signed_error_ticks": 0.2,
            "signed_error_ms": 25.0,
            "snapped_forward": False,
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    csv_path = tmp_path / "summary.csv"
    script = Path(__file__).parents[2] / "scripts" / "analyze_input_quantization_trace.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(tmp_path),
            "--csv-out",
            str(csv_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "session-1" in completed.stdout
    with csv_path.open(encoding="utf-8", newline="") as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == 2
    summary = next(row for row in summaries if row["event_type"] == "note_on")
    assert summary["event_count"] == "3"
    assert float(summary["mean_signed_error_ms"]) == pytest.approx(25.0 / 3.0)
    assert float(summary["p50_absolute_error_ms"]) == pytest.approx(25.0)
    assert float(summary["p95_absolute_error_ms"]) == pytest.approx(47.5)
    assert float(summary["quantized_earlier_rate"]) == pytest.approx(1.0 / 3.0)
    assert float(summary["quantized_later_rate"]) == pytest.approx(1.0 / 3.0)
    assert float(summary["snap_rate"]) == pytest.approx(1.0 / 3.0)
