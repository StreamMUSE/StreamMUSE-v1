import csv
from pathlib import Path

import numpy as np
import pytest

from experiments.lekai_failcase_analysis.build_meter_audit import (
    AUDIT_FIELDS,
    build_meter_audit,
    main,
)


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


def _write_musicxml(path, time_signatures, namespace=None):
    xmlns = f' xmlns="{namespace}"' if namespace else ""
    times = "".join(
        f"<measure number=\"{index}\"><attributes><time>"
        f"<beats>{beats}</beats><beat-type>{beat_type}</beat-type>"
        f"</time></attributes></measure>"
        for index, (beats, beat_type) in enumerate(time_signatures, start=1)
    )
    path.write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<score-partwise version="4.0"{xmlns}><part id="P1">{times}</part>'
        f"</score-partwise>",
        encoding="utf-8",
    )


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
    assert rows[0]["meter_source"] == "npz_metadata_first_signature_only"
    assert rows[0]["xml_status"] == "not_requested"
    assert rows[0]["xml_time_signatures"] == ""
    assert rows[0]["xml_time_signature_event_count"] == 0
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


@pytest.mark.parametrize(
    ("time_signatures", "expected_signatures", "expected_count", "expected_status"),
    [
        ([('4', '4'), ('4', '4')], "4/4", 2, "include_4_4"),
        (
            [('5', '4'), ('2', '4'), ('4', '4'), ('4', '4')],
            "2/4|4/4|5/4",
            4,
            "hold_non_4_4",
        ),
        ([('3', '4')], "3/4", 1, "hold_non_4_4"),
    ],
)
def test_build_meter_audit_uses_all_musicxml_time_events(
    tmp_path,
    time_signatures,
    expected_signatures,
    expected_count,
    expected_status,
):
    np.savez(tmp_path / "source.npz", metadata={"time_signature": "4/4"})
    manifest = tmp_path / "selection_manifest.csv"
    row = _manifest_row("piece", "source.npz")
    row["record_id"] = "142159"
    _write_manifest(manifest, [row])
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _write_musicxml(xml_dir / "142159.musicxml", time_signatures)

    rows = build_meter_audit(
        manifest,
        tmp_path,
        tmp_path / "audit.csv",
        musicxml_dir=xml_dir,
    )

    assert rows[0]["xml_time_signatures"] == expected_signatures
    assert rows[0]["xml_time_signature_event_count"] == expected_count
    assert rows[0]["xml_status"] == "parsed"
    assert rows[0]["meter_source"] == "musicxml_all_time_signatures"
    assert rows[0]["meter_status"] == expected_status
    assert rows[0]["xml_path"] == str(xml_dir / "142159.musicxml")


def test_build_meter_audit_supports_namespaced_musicxml(tmp_path):
    np.savez(tmp_path / "source.npz", metadata={"time_signature": "3/4"})
    manifest = tmp_path / "selection_manifest.csv"
    row = _manifest_row("namespaced", "source.npz")
    row["record_id"] = "42"
    _write_manifest(manifest, [row])
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _write_musicxml(
        xml_dir / "42.musicxml",
        [("4", "4")],
        namespace="http://www.musicxml.org/ns/musicxml",
    )

    rows = build_meter_audit(
        manifest,
        tmp_path,
        tmp_path / "audit.csv",
        musicxml_dir=xml_dir,
    )

    assert rows[0]["xml_time_signatures"] == "4/4"
    assert rows[0]["meter_status"] == "include_4_4"


@pytest.mark.parametrize(
    ("xml_content", "expected_xml_status", "expected_reason"),
    [
        (
            '<score-partwise><part id="P1"><measure number="1"/></part></score-partwise>',
            "no_time_signature_events",
            "musicxml_no_time_signature_events",
        ),
        ("<score-partwise>", "parse_error", "musicxml_parse_error"),
    ],
)
def test_build_meter_audit_holds_unknown_invalid_musicxml(
    tmp_path,
    xml_content,
    expected_xml_status,
    expected_reason,
):
    np.savez(tmp_path / "source.npz", metadata={"time_signature": "4/4"})
    manifest = tmp_path / "selection_manifest.csv"
    row = _manifest_row("invalid_xml", "source.npz")
    row["record_id"] = "77"
    _write_manifest(manifest, [row])
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    (xml_dir / "77.musicxml").write_text(xml_content, encoding="utf-8")

    rows = build_meter_audit(
        manifest,
        tmp_path,
        tmp_path / "audit.csv",
        musicxml_dir=xml_dir,
    )

    assert rows[0]["meter_status"] == "hold_unknown"
    assert rows[0]["xml_status"] == expected_xml_status
    assert rows[0]["meter_hold_reason"] == expected_reason


def test_build_meter_audit_holds_unknown_when_requested_musicxml_is_missing(tmp_path):
    np.savez(tmp_path / "source.npz", metadata={"time_signature": "4/4"})
    manifest = tmp_path / "selection_manifest.csv"
    row = _manifest_row("missing_xml", "source.npz")
    row["record_id"] = "88"
    _write_manifest(manifest, [row])
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()

    rows = build_meter_audit(
        manifest,
        tmp_path,
        tmp_path / "audit.csv",
        musicxml_dir=xml_dir,
    )

    assert rows[0]["meter_status"] == "hold_unknown"
    assert rows[0]["xml_status"] == "missing"
    assert rows[0]["meter_hold_reason"] == "musicxml_missing"


def test_build_meter_audit_missing_xml_directory_argument_falls_back_to_npz(tmp_path):
    np.savez(tmp_path / "source.npz", metadata={"time_signature": "4/4"})
    manifest = tmp_path / "selection_manifest.csv"
    _write_manifest(manifest, [_manifest_row("fallback", "source.npz")])

    rows = build_meter_audit(manifest, tmp_path, tmp_path / "audit.csv")

    assert rows[0]["meter_status"] == "include_4_4"
    assert rows[0]["meter_source"] == "npz_metadata_first_signature_only"
    assert rows[0]["xml_status"] == "not_requested"


def test_main_accepts_musicxml_dir_and_writes_stable_columns(tmp_path):
    np.savez(tmp_path / "source.npz", metadata={"time_signature": "3/4"})
    manifest = tmp_path / "selection_manifest.csv"
    row = _manifest_row("cli_piece", "source.npz")
    row["record_id"] = "99"
    _write_manifest(manifest, [row])
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _write_musicxml(xml_dir / "99.musicxml", [("4", "4")])
    output = tmp_path / "audit.csv"

    assert main(
        [
            "--selection-manifest",
            str(manifest),
            "--bundle-root",
            str(tmp_path),
            "--musicxml-dir",
            str(xml_dir),
            "--output",
            str(output),
        ]
    ) == 0

    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == AUDIT_FIELDS
    assert rows[0]["meter_status"] == "include_4_4"
    assert rows[0]["xml_time_signature_event_count"] == "1"
