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


def test_cli_writes_inspectable_artifacts_for_an_entirely_empty_selection(tmp_path: Path, capsys, load_script) -> None:
    """Catches an empty local corpus exiting before it persists inspectable evidence."""
    source_dir = tmp_path / "empty-inputs"
    source_dir.mkdir()
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
        ]
    ) == 1

    catalog_payload = json.loads(catalog.read_text(encoding="utf-8"))
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert catalog_payload["templates"] == []
    assert report_payload["selected_templates"] == 0
    assert "selected_templates=0" in capsys.readouterr().out


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


def test_cli_returns_two_for_input_read_failure_without_echoing_source_path(tmp_path: Path, capsys, load_script, monkeypatch) -> None:
    """Catches user-input read failures escaping the opt-in CLI with a source path."""
    script = load_script("build_mcflow_sample_catalog")
    private_source = tmp_path / "private-source"
    private_source.mkdir()

    def fail_read(*args, **kwargs):
        raise OSError(f"cannot read {private_source}")

    monkeypatch.setattr(script, "extract_mcflow_directory", fail_read)

    assert script.main(
        [
            "--mcflow-dir",
            str(private_source),
            "--catalog-output",
            str(tmp_path / "catalog.json"),
            "--report-output",
            str(tmp_path / "report.json"),
        ]
    ) == 2

    rendered = capsys.readouterr().err
    assert "unable to read MCFlow input" in rendered
    assert str(private_source) not in rendered


def test_cli_rejects_equal_resolved_destinations_without_writing(tmp_path: Path, capsys, load_script) -> None:
    """Catches a catalog output path that aliases the aggregate report output path."""
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    (source_dir / "private.rap").write_bytes(FIXTURE.read_bytes())
    catalog = tmp_path / "out" / "catalog.json"
    report_alias = tmp_path / "out" / "." / "catalog.json"
    script = load_script("build_mcflow_sample_catalog")

    assert script.main(
        [
            "--mcflow-dir",
            str(source_dir),
            "--catalog-output",
            str(catalog),
            "--report-output",
            str(report_alias),
        ]
    ) == 2

    assert not catalog.exists()
    assert "unable to write sample catalog" in capsys.readouterr().err
