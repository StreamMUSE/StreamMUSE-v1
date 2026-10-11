#!/usr/bin/env python3
"""Build auditable metadata-derived style candidates for Task 3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(character for character in value if not unicodedata.combining(character))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def _flatten(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_flatten(item))
        return result
    return [str(value)]


def _metadata_text(record: dict) -> tuple[list[str], list[str]]:
    titles = _flatten(record.get("movement_title"))
    creators = _flatten(record.get("creators"))
    for credit in record.get("credits") or []:
        credit_type = _normalize(str(credit.get("type") or ""))
        words = _flatten(credit.get("words"))
        if credit_type in {"title", "subtitle", "movement title"}:
            titles.extend(words)
        elif credit_type in {"composer", "creator", "artist", "arranger", "lyricist"}:
            creators.extend(words)
    return list(dict.fromkeys(titles)), list(dict.fromkeys(creators))


def _term_hits(text: str, terms: list[str]) -> list[str]:
    padded = f" {_normalize(text)} "
    hits = []
    for term in terms:
        normalized = _normalize(term)
        if normalized and f" {normalized} " in padded:
            hits.append(term)
    return hits


def _is_single_piano(record: dict) -> bool:
    if int(record.get("n_parts") or 0) != 1:
        return False
    part_text = " ".join(
        str(value or "")
        for part in record.get("parts") or []
        for value in (part.get("name"), part.get("instrument"), part.get("abbr"))
    ).casefold()
    return "piano" in part_text or "keyboard" in part_text or "pno" in part_text


def _npz_stats(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as payload:
        metadata = payload["metadata"].item()
        measure_keys = sorted(
            (key for key in payload.files if key.startswith("measure_")),
            key=lambda key: int(key.split("_")[-1]),
        )
        melody_onsets = 0
        accompaniment_onsets = 0
        measured_steps = 0
        shape_valid = True
        for key in measure_keys:
            measure = payload[key]
            measured_steps += int(measure.shape[-1]) if measure.ndim == 3 else 0
            if measure.ndim != 3 or measure.shape[:2] != (4, 88):
                shape_valid = False
                continue
            melody_onsets += int(np.count_nonzero(measure[1]))
            accompaniment_onsets += int(np.count_nonzero(measure[3]))
    total_steps = int(metadata.get("total_length") or measured_steps)
    total_beats = total_steps / 4.0
    return {
        "time_signature": str(metadata.get("time_signature") or ""),
        "bpm": float(metadata.get("bpm") or 0.0),
        "num_measures": int(metadata.get("num_measures") or len(measure_keys)),
        "num_channels": int(metadata.get("num_channels") or 0),
        "shape_valid": shape_valid,
        "melody_note_count": melody_onsets,
        "accompaniment_note_count": accompaniment_onsets,
        "melody_density_per_beat": melody_onsets / total_beats if total_beats else 0.0,
        "accompaniment_density_per_beat": accompaniment_onsets / total_beats if total_beats else 0.0,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--xml-root", type=Path, required=True)
    parser.add_argument("--npz-root", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shortlist-per-style", type=int, default=30)
    parser.add_argument("--max-per-evidence", type=int, default=3)
    parser.add_argument("--max-npz-audit-per-style", type=int, default=500)
    args = parser.parse_args()

    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    raw_candidates = []
    counters = Counter()
    with args.metadata.open("r", encoding="utf-8") as handle:
        for line in handle:
            counters["metadata_rows"] += 1
            record = json.loads(line)
            if not record.get("has_npz") or not _is_single_piano(record):
                continue
            counters["eligible_single_piano"] += 1
            titles, creators = _metadata_text(record)
            title_text = " | ".join(titles)
            creator_text = " | ".join(creators)
            matches = []
            for style, rule in rules.items():
                creator_hits = _term_hits(creator_text, rule.get("creator_terms", []))
                title_hits = _term_hits(title_text, rule.get("title_terms", []))
                if creator_hits or title_hits:
                    matches.append((style, creator_hits, title_hits))
            if not matches:
                continue
            if len(matches) > 1:
                counters["ambiguous_multi_style"] += 1
                continue
            style, creator_hits, title_hits = matches[0]
            piece_id = str(record["id"])
            xml_path = args.xml_root / f"{piece_id}.musicxml"
            npz_path = args.npz_root / f"{piece_id}.npz"
            confidence = 3 * bool(creator_hits) + 2 * bool(title_hits)
            primary_evidence = (creator_hits or title_hits)[0]
            raw_candidates.append(
                {
                    "piece_id": piece_id,
                    "style": style,
                    "confidence": confidence,
                    "primary_evidence": primary_evidence,
                    "creator_hits": " | ".join(creator_hits),
                    "title_hits": " | ".join(title_hits),
                    "title": title_text,
                    "creator": creator_text,
                    "xml_exists": xml_path.is_file(),
                    "npz_exists": npz_path.is_file(),
                    "xml_path": str(xml_path),
                    "npz_path": str(npz_path),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "all_metadata_matches.csv", raw_candidates)

    by_style = defaultdict(list)
    for row in raw_candidates:
        by_style[row["style"]].append(row)

    shortlist = []
    audit_counts = Counter()
    for style in rules:
        candidates = sorted(
            by_style[style],
            key=lambda row: (
                -row["confidence"],
                hashlib.sha1(row["piece_id"].encode()).hexdigest(),
            ),
        )
        evidence_counts = Counter()
        audited = 0
        for row in candidates:
            if len([item for item in shortlist if item["style"] == style]) >= args.shortlist_per_style:
                break
            evidence_key = _normalize(row["primary_evidence"])
            if evidence_counts[evidence_key] >= args.max_per_evidence:
                continue
            if not row["xml_exists"] or not row["npz_exists"]:
                audit_counts[f"{style}:missing_file"] += 1
                continue
            if audited >= args.max_npz_audit_per_style:
                break
            audited += 1
            try:
                stats = _npz_stats(Path(row["npz_path"]))
            except Exception:
                audit_counts[f"{style}:npz_error"] += 1
                continue
            if not stats["shape_valid"] or stats["num_channels"] != 4:
                audit_counts[f"{style}:invalid_shape"] += 1
                continue
            if stats["time_signature"] != "4/4":
                audit_counts[f"{style}:not_4_4"] += 1
                continue
            if not 32 <= stats["num_measures"] <= 160:
                audit_counts[f"{style}:length_filter"] += 1
                continue
            if stats["melody_note_count"] == 0 or stats["accompaniment_note_count"] == 0:
                audit_counts[f"{style}:empty_track"] += 1
                continue
            evidence_counts[evidence_key] += 1
            shortlist.append(
                {
                    **row,
                    **stats,
                    "manual_keep": "",
                    "manual_style": "",
                    "manual_comments": "",
                }
            )

    _write_csv(args.output_dir / "shortlist_for_manual_review.csv", shortlist)
    summary = {
        **counters,
        "raw_matches_by_style": {
            style: len(by_style[style]) for style in rules
        },
        "shortlist_by_style": Counter(row["style"] for row in shortlist),
        "audit_rejections": audit_counts,
        "rules": rules,
        "filters": {
            "single_piano_part": True,
            "has_npz": True,
            "npz_shape": "(4, 88, T)",
            "time_signature": "4/4",
            "measure_range": [32, 160],
            "both_tracks_nonempty": True,
            "max_per_primary_evidence": args.max_per_evidence,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=dict), encoding="utf-8"
    )
    print(json.dumps(summary, default=dict))


if __name__ == "__main__":
    main()
