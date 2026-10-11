"""Build a meter audit from MusicXML or source NPZ metadata."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


AUDIT_FIELDS = (
    "style",
    "piece",
    "title",
    "record_id",
    "source_npz",
    "npz_time_signature",
    "npz_time_signature_idx",
    "xml_time_signatures",
    "xml_time_signature_event_count",
    "xml_path",
    "xml_status",
    "meter_source",
    "meter_status",
    "meter_hold_reason",
)


def _load_metadata(npz_path: Path, piece: str) -> dict[object, object]:
    if not npz_path.is_file():
        raise ValueError(f"source NPZ does not exist for piece {piece!r}: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as archive:
        if "metadata" not in archive.files:
            raise ValueError(f"source NPZ metadata missing for piece {piece!r}: {npz_path}")
        raw_metadata = archive["metadata"]
        try:
            metadata = raw_metadata.item()
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"source NPZ metadata is not a scalar dict for piece {piece!r}: {npz_path}"
            ) from exc

    if not isinstance(metadata, dict):
        raise ValueError(f"source NPZ metadata is not a dict for piece {piece!r}: {npz_path}")
    return metadata


def _local_name(tag: str) -> str:
    """Return an XML tag name without its optional namespace."""

    return tag.rsplit("}", 1)[-1]


def _read_musicxml_time_signatures(xml_path: Path) -> tuple[list[str], int]:
    """Read every complete MusicXML ``time`` event in document order."""

    root = ET.parse(xml_path).getroot()
    signatures: list[str] = []
    for element in root.iter():
        if _local_name(element.tag) != "time":
            continue

        beats = [
            (child.text or "").strip()
            for child in element
            if _local_name(child.tag) == "beats" and (child.text or "").strip()
        ]
        beat_types = [
            (child.text or "").strip()
            for child in element
            if _local_name(child.tag) == "beat-type" and (child.text or "").strip()
        ]
        if not beats or len(beats) != len(beat_types):
            continue
        signatures.append(
            "+".join(
                f"{beat}/{beat_type}" for beat, beat_type in zip(beats, beat_types)
            )
        )

    return sorted(set(signatures)), len(signatures)


def _classify_npz_signature(signature: str) -> tuple[str, str]:
    if signature == "4/4":
        return "include_4_4", ""
    if signature:
        return "hold_non_4_4", f"meter_not_4_4:{signature}"
    return "hold_unknown", "meter_unknown"


def build_meter_audit(
    selection_manifest: Path,
    bundle_root: Path,
    output_path: Path,
    musicxml_dir: Path | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_pieces: set[str] = set()

    with selection_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"category", "target_id", "title", "record_id", "source_npz"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"selection manifest missing columns: {', '.join(missing)}")

        for row_number, source_row in enumerate(reader, start=2):
            piece = (source_row["target_id"] or "").strip()
            if piece in seen_pieces:
                raise ValueError(
                    f"duplicate piece in selection manifest at row {row_number}: {piece!r}"
                )
            seen_pieces.add(piece)

            source_npz = (source_row["source_npz"] or "").strip()
            npz_path = Path(source_npz)
            if not npz_path.is_absolute():
                npz_path = bundle_root / npz_path
            metadata = _load_metadata(npz_path, piece)

            raw_signature = metadata.get("time_signature", "")
            signature = (
                raw_signature
                if isinstance(raw_signature, str)
                else "" if raw_signature is None else str(raw_signature)
            )
            xml_signatures: list[str] = []
            xml_event_count = 0
            xml_path_text = ""
            xml_status = "not_requested"

            if musicxml_dir is None:
                meter_source = "npz_metadata_first_signature_only"
                meter_status, hold_reason = _classify_npz_signature(signature)
            else:
                meter_source = "musicxml_all_time_signatures"
                record_id = (source_row["record_id"] or "").strip()
                if not record_id:
                    xml_status = "record_id_missing"
                    meter_status = "hold_unknown"
                    hold_reason = "musicxml_record_id_missing"
                else:
                    xml_path = musicxml_dir / f"{record_id}.musicxml"
                    xml_path_text = str(xml_path)
                    if not xml_path.is_file():
                        xml_status = "missing"
                        meter_status = "hold_unknown"
                        hold_reason = "musicxml_missing"
                    else:
                        try:
                            xml_signatures, xml_event_count = _read_musicxml_time_signatures(
                                xml_path
                            )
                        except (ET.ParseError, OSError):
                            xml_status = "parse_error"
                            meter_status = "hold_unknown"
                            hold_reason = "musicxml_parse_error"
                        else:
                            if xml_event_count == 0:
                                xml_status = "no_time_signature_events"
                                meter_status = "hold_unknown"
                                hold_reason = "musicxml_no_time_signature_events"
                            elif xml_signatures == ["4/4"]:
                                xml_status = "parsed"
                                meter_status = "include_4_4"
                                hold_reason = ""
                            else:
                                xml_status = "parsed"
                                meter_status = "hold_non_4_4"
                                hold_reason = (
                                    "musicxml_not_constant_4_4:"
                                    + "|".join(xml_signatures)
                                )

            signature_idx = metadata.get("time_signature_idx", "")
            if signature_idx is None:
                signature_idx = ""
            rows.append(
                {
                    "style": (source_row["category"] or "").strip(),
                    "piece": piece,
                    "title": (source_row["title"] or "").strip(),
                    "record_id": (source_row["record_id"] or "").strip(),
                    "source_npz": source_npz,
                    "npz_time_signature": signature,
                    "npz_time_signature_idx": signature_idx,
                    "xml_time_signatures": "|".join(xml_signatures),
                    "xml_time_signature_event_count": xml_event_count,
                    "xml_path": xml_path_text,
                    "xml_status": xml_status,
                    "meter_source": meter_source,
                    "meter_status": meter_status,
                    "meter_hold_reason": hold_reason,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--musicxml-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle_root = args.bundle_root or args.selection_manifest.parent.parent
    build_meter_audit(
        args.selection_manifest,
        bundle_root,
        args.output,
        musicxml_dir=args.musicxml_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
