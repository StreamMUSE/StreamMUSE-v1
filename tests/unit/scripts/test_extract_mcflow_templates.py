"""Tests for the opt-in anonymous MCFlow extraction command."""

from __future__ import annotations

import json
from pathlib import Path


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "rap" / "mcflow_minimal.rap"


def test_cli_writes_anonymous_catalog_and_aggregate_summary(tmp_path: Path, capsys, load_script) -> None:
    """Catches a CLI that leaks file names or fails to persist usable output."""
    source_dir = tmp_path / "corpus"
    source_dir.mkdir()
    (source_dir / "private.rap").write_bytes(FIXTURE.read_bytes())
    output = tmp_path / "catalog.json"
    script = load_script("extract_mcflow_templates")

    assert script.main(["--mcflow-dir", str(source_dir), "--output", str(output)]) == 0

    rendered = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "accepted_templates=2" in rendered
    assert "private.rap" not in rendered
    assert "private.rap" not in json.dumps(payload)
    assert payload["aggregate"]["accepted_templates"] == 2


def test_cli_returns_nonzero_for_invalid_directory_without_echoing_it(tmp_path: Path, capsys, load_script) -> None:
    """Catches invalid input handling that exposes a caller's local path."""
    script = load_script("extract_mcflow_templates")
    missing = tmp_path / "private-source"

    assert script.main(["--mcflow-dir", str(missing), "--output", str(tmp_path / "out.json")]) == 2

    rendered = capsys.readouterr().err
    assert "mcflow directory must be an existing directory" in rendered
    assert str(missing) not in rendered
