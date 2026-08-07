"""Tests for the local anonymous MCFlow sample-catalog CLI."""

from __future__ import annotations

import json
from pathlib import Path


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "rap" / "mcflow_minimal.rap"


def test_cli_writes_underfilled_catalog_and_aggregate_only_report(tmp_path: Path, capsys, load_script) -> None:
    """Catches a missing-band result that is not persisted for local inspection."""
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    (source_dir / "private.rap").write_bytes(FIXTURE.read_bytes())
    catalog = tmp_path / "out" / "catalog.json"
    report = tmp_path / "out" / "report.json"
    script = load_script("build_mcflow_sample_catalog")

    assert script.main(
        [
            "--mcflow-dir",
            str(source_dir),
            "--catalog-output",
            str(catalog),
            "--report-output",
            str(report),
            "--per-bucket",
            "2",
        ]
    ) == 1

    rendered = capsys.readouterr().out
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert catalog.exists()
    assert "parsed_files=1" in rendered
    assert "private.rap" not in rendered
    assert str(catalog) in rendered
    assert str(report) in rendered
    assert report_payload["bands"]["medium"]["available"] == 0
    assert "source_hash" not in json.dumps(report_payload)


def test_cli_returns_two_for_invalid_input_without_echoing_source_path(tmp_path: Path, capsys, load_script) -> None:
    """Catches invalid argument handling that leaks a caller's local source location."""
    script = load_script("build_mcflow_sample_catalog")
    private_source = tmp_path / "private-source"
    private_source.mkdir()

    assert script.main(
        [
            "--mcflow-dir",
            str(private_source),
            "--catalog-output",
            str(tmp_path / "catalog.json"),
            "--report-output",
            str(tmp_path / "report.json"),
            "--per-bucket",
            "0",
        ]
    ) == 2

    rendered = capsys.readouterr().err
    assert "per_bucket" in rendered
    assert str(private_source) not in rendered
