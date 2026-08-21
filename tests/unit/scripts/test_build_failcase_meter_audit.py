import csv
from pathlib import Path

import numpy as np
import pytest

from experiments.lekai_failcase_analysis.build_meter_audit import build_meter_audit


def _write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["category", "target_id", "title", "record_id", "source_npz"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _manifest_row(piece, source_npz):
    return {
        "category": "test_style",
        "target_id": piece,
        "title": piece.replace("_", " ").title(),
        "record_id": f"record-{piece}",
        "source_npz": source_npz,
    }


def test_build_meter_audit_classifies_metadata(tmp_path):
    manifest = tmp_path / "metadata" / "selection_manifest.csv"
    manifest.parent.mkdir()
    for name, metadata in [
        ("four.npz", {"time_signature": "4/4", "time_signature_idx": 99}),
        ("three.npz", {"time_signature": "3/4", "time_signature_idx": 0}),
        ("unknown.npz", {"time_signature_idx": 0}),
    ]:
        np.savez(tmp_path / name, metadata=metadata)
    _write_manifest(
        manifest,
        [
            _manifest_row("piece_four", "four.npz"),
            _manifest_row("piece_three", "three.npz"),
            _manifest_row("piece_unknown", "unknown.npz"),
        ],
    )

    output = tmp_path / "audit.csv"
    rows = build_meter_audit(manifest, tmp_path, output)

    assert [row["meter_status"] for row in rows] == [
        "include_4_4",
        "hold_non_4_4",
        "hold_unknown",
    ]
    assert rows[0]["npz_time_signature_idx"] == 99
    assert rows[0]["meter_source"] == "npz_metadata"
    assert rows[1]["meter_hold_reason"] == "meter_not_4_4:3/4"
    assert rows[2]["meter_hold_reason"] == "meter_unknown"


def test_build_meter_audit_rejects_duplicate_piece(tmp_path):
    np.savez(tmp_path / "source.npz", metadata={"time_signature": "4/4"})
    manifest = tmp_path / "selection_manifest.csv"
    row = _manifest_row("duplicate", "source.npz")
    _write_manifest(manifest, [row, row])

    with pytest.raises(ValueError, match="duplicate piece"):
        build_meter_audit(manifest, tmp_path, tmp_path / "audit.csv")


def test_build_meter_audit_rejects_missing_npz(tmp_path):
    manifest = tmp_path / "selection_manifest.csv"
    _write_manifest(manifest, [_manifest_row("missing", "missing.npz")])

    with pytest.raises(ValueError, match="source NPZ does not exist"):
        build_meter_audit(manifest, tmp_path, tmp_path / "audit.csv")


def test_build_meter_audit_rejects_non_dict_metadata(tmp_path):
    np.savez(tmp_path / "source.npz", metadata="not-a-dict")
    manifest = tmp_path / "selection_manifest.csv"
    _write_manifest(manifest, [_manifest_row("bad_metadata", "source.npz")])

    with pytest.raises(ValueError, match="metadata is not a dict"):
        build_meter_audit(manifest, tmp_path, tmp_path / "audit.csv")
