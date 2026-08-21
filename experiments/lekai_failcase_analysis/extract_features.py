"""Extract frozen Melody and selected-Prompt features for fail-case analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

if __package__:
    from experiments.lekai_failcase_analysis.music_features import (
        GRID_STEP_BEATS,
        PREFIX_BEATS,
        TrackSelectionError,
        load_prompt_accompaniment,
        load_source_melody,
        melody_features,
        prompt_features,
        relation_features,
    )
else:  # Direct execution by repository-relative script path.
    from music_features import (  # type: ignore[no-redef]
        GRID_STEP_BEATS,
        PREFIX_BEATS,
        TrackSelectionError,
        load_prompt_accompaniment,
        load_source_melody,
        melody_features,
        prompt_features,
        relation_features,
    )


KEY_FIELDS = ("style", "piece", "seed", "condition")
LABEL_FIELDS = (
    "npz_time_signature", "meter_status", "score_raw", "fail_type_raw",
    "quality_floor", "quality_mean", "issue_any", "severe_fail", "label_status",
    "include_main", "exclusion_reason", "comments",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _index_unique(rows: Iterable[dict[str, str]], fields: tuple[str, ...], name: str) -> dict[tuple[str, ...], dict[str, str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple((row.get(field) or "").strip() for field in fields)
        if key in result:
            raise ValueError(f"duplicate {name} key: {key}")
        result[key] = row
    return result


def _relative_or_absolute(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _float_or_nan(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty feature table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _csv_value(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _aggregate_piece_labels(rows: list[dict[str, str]]) -> dict[str, object]:
    first = rows[0]
    for field in ("style", "piece", "npz_time_signature", "meter_status", "include_main", "exclusion_reason"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"piece label field is inconsistent for {first['piece']}: {field}")
    rated = [row for row in rows if row["label_status"] == "rated"]
    severe = [_float_or_nan(row["severe_fail"] == "True") for row in rated]
    issues = [_float_or_nan(row["issue_any"] == "True") for row in rated]
    floors = [_float_or_nan(row["quality_floor"]) for row in rated]
    means = [_float_or_nan(row["quality_mean"]) for row in rated]
    return {
        "style": first["style"],
        "piece": first["piece"],
        "npz_time_signature": first["npz_time_signature"],
        "meter_status": first["meter_status"],
        "include_main": first["include_main"],
        "exclusion_reason": first["exclusion_reason"],
        "run_count": len(rows),
        "rated_run_count": len(rated),
        "severe_fail_rate": float(sum(severe) / len(severe)) if severe else float("nan"),
        "issue_any_rate": float(sum(issues) / len(issues)) if issues else float("nan"),
        "quality_floor_mean": float(sum(floors) / len(floors)) if floors else float("nan"),
        "quality_floor_min": min(floors) if floors else float("nan"),
        "quality_mean_mean": float(sum(means) / len(means)) if means else float("nan"),
    }


def extract_feature_tables(bundle_dir: Path, labels_runs: Path, output_dir: Path) -> dict[str, object]:
    label_rows = _read_csv(labels_runs)
    manifest_rows = _read_csv(bundle_dir / "run_manifest.csv")
    required_labels = set(KEY_FIELDS + LABEL_FIELDS)
    required_manifest = set(KEY_FIELDS + ("status", "prompt_midi"))
    if not label_rows or not required_labels <= set(label_rows[0]):
        raise ValueError(f"labels_runs missing columns: {sorted(required_labels - set(label_rows[0] if label_rows else {}))}")
    if not manifest_rows or not required_manifest <= set(manifest_rows[0]):
        raise ValueError(f"run_manifest missing columns: {sorted(required_manifest - set(manifest_rows[0] if manifest_rows else {}))}")

    labels = _index_unique(label_rows, KEY_FIELDS, "labels")
    manifest = _index_unique(manifest_rows, KEY_FIELDS, "manifest")
    if set(labels) != set(manifest):
        labels_only = sorted(set(labels) - set(manifest))[:5]
        manifest_only = sorted(set(manifest) - set(labels))[:5]
        raise ValueError(f"labels/manifest key mismatch; labels_only={labels_only}, manifest_only={manifest_only}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in label_rows:
        grouped[(row["style"], row["piece"])].append(row)

    status_names = (
        "source_ok", "missing_source_midi", "missing_melody_track",
        "ambiguous_melody_track", "invalid_source_midi", "prompt_ok",
        "empty_prompt", "missing_prompt_midi", "missing_accompaniment_track",
        "ambiguous_accompaniment_track", "invalid_prompt_midi",
        "relation_source_unavailable",
    )
    audit_status = Counter({name: 0 for name in status_names})
    main_audit_status = Counter({name: 0 for name in status_names})
    undefined_counts = Counter()
    melody_rows: list[dict[str, object]] = []
    melody_notes: dict[tuple[str, str], object] = {}
    for style, piece in sorted(grouped):
        labels_for_piece = grouped[(style, piece)]
        row = _aggregate_piece_labels(labels_for_piece)
        source_path = bundle_dir / "listening_by_style" / style / piece / "00_source_melody.mid"
        row["source_midi"] = source_path.relative_to(bundle_dir).as_posix()
        try:
            track_name, notes = load_source_melody(source_path)
            row["melody_track_name"] = track_name
            row["melody_extraction_status"] = "ok"
            features = melody_features(notes)
            row.update(features)
            melody_notes[(style, piece)] = notes
            audit_status["source_ok"] += 1
            if row["include_main"] == "True":
                main_audit_status["source_ok"] += 1
            _count_undefined(undefined_counts, "melody", features)
        except FileNotFoundError:
            row["melody_track_name"] = ""
            row["melody_extraction_status"] = "missing_source_midi"
            audit_status["missing_source_midi"] += 1
            if row["include_main"] == "True":
                main_audit_status["missing_source_midi"] += 1
        except TrackSelectionError as exc:
            row["melody_track_name"] = ""
            row["melody_extraction_status"] = exc.status
            audit_status[exc.status] += 1
            if row["include_main"] == "True":
                main_audit_status[exc.status] += 1
        except Exception as exc:
            row["melody_track_name"] = ""
            row["melody_extraction_status"] = f"invalid_source_midi:{type(exc).__name__}"
            audit_status["invalid_source_midi"] += 1
            if row["include_main"] == "True":
                main_audit_status["invalid_source_midi"] += 1
        melody_rows.append(row)

    prompt_rows: list[dict[str, object]] = []
    for key in sorted(labels):
        label = labels[key]
        manifest_row = manifest[key]
        style, piece, seed, condition = key
        row: dict[str, object] = {field: label[field] for field in KEY_FIELDS + LABEL_FIELDS}
        row["manifest_status"] = manifest_row["status"]
        prompt_path = _relative_or_absolute(bundle_dir, manifest_row["prompt_midi"])
        row["prompt_midi"] = (
            prompt_path.relative_to(bundle_dir).as_posix()
            if prompt_path.is_relative_to(bundle_dir) else str(prompt_path)
        )
        source_notes = melody_notes.get((style, piece))
        try:
            track_name, accompaniment, has_accompaniment = load_prompt_accompaniment(prompt_path)
            row["accompaniment_track_name"] = track_name
            row["prompt_has_accompaniment"] = int(has_accompaniment)
            row["prompt_extraction_status"] = "ok" if has_accompaniment else "empty_prompt"
            features = prompt_features(accompaniment)
            row.update(features)
            _count_undefined(undefined_counts, "prompt", features)
            if source_notes is None:
                row["relation_extraction_status"] = "source_unavailable"
                audit_status["relation_source_unavailable"] += 1
                if label["include_main"] == "True":
                    main_audit_status["relation_source_unavailable"] += 1
            else:
                relations = relation_features(source_notes, accompaniment)
                row.update(relations)
                row["relation_extraction_status"] = "ok"
                _count_undefined(undefined_counts, "relation", relations)
            prompt_status = "prompt_ok" if has_accompaniment else "empty_prompt"
            audit_status[prompt_status] += 1
            if label["include_main"] == "True":
                main_audit_status[prompt_status] += 1
        except FileNotFoundError:
            row["accompaniment_track_name"] = ""
            row["prompt_has_accompaniment"] = ""
            row["prompt_extraction_status"] = "missing_prompt_midi"
            row["relation_extraction_status"] = "prompt_unavailable"
            audit_status["missing_prompt_midi"] += 1
            if label["include_main"] == "True":
                main_audit_status["missing_prompt_midi"] += 1
        except TrackSelectionError as exc:
            row["accompaniment_track_name"] = ""
            row["prompt_has_accompaniment"] = ""
            row["prompt_extraction_status"] = exc.status
            row["relation_extraction_status"] = "prompt_unavailable"
            audit_status[exc.status] += 1
            if label["include_main"] == "True":
                main_audit_status[exc.status] += 1
        except Exception as exc:
            row["accompaniment_track_name"] = ""
            row["prompt_has_accompaniment"] = ""
            row["prompt_extraction_status"] = f"invalid_prompt_midi:{type(exc).__name__}"
            row["relation_extraction_status"] = "prompt_unavailable"
            audit_status["invalid_prompt_midi"] += 1
            if label["include_main"] == "True":
                main_audit_status["invalid_prompt_midi"] += 1
        prompt_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "melody_features.csv", melody_rows)
    _write_csv(output_dir / "prompt_features.csv", prompt_rows)
    main_melodies = [row for row in melody_rows if row["include_main"] == "True"]
    main_prompts = [row for row in prompt_rows if row["include_main"] == "True"]
    audit: dict[str, object] = {
        "bundle_dir": str(bundle_dir.resolve()),
        "labels_runs": str(labels_runs.resolve()),
        "quantization": {
            "beat_coordinate": "PrettyMIDI.time_to_tick(seconds) / resolution",
            "grid_step_beats": GRID_STEP_BEATS,
            "steps_per_beat": int(round(1.0 / GRID_STEP_BEATS)),
            "rounding": "python_round_bankers_matches_MidiConverter",
            "minimum_duration_steps": 1,
            "timeline_origin_shifted": False,
            "prefix_interval_beats": [0.0, PREFIX_BEATS],
        },
        "row_counts": {
            "labels_runs": len(label_rows),
            "manifest_runs": len(manifest_rows),
            "melody_features": len(melody_rows),
            "prompt_features": len(prompt_rows),
            "main_melody_features": len(main_melodies),
            "main_prompt_features": len(main_prompts),
        },
        "track_and_file_status_counts": dict(sorted(audit_status.items())),
        "main_track_and_file_status_counts": dict(sorted(main_audit_status.items())),
        "undefined_metric_counts": dict(sorted(undefined_counts.items())),
        "ppl": "unavailable_in_v1_requires_raw_prompt_token_log_probabilities",
    }
    with (output_dir / "feature_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return audit


def _count_undefined(counter: Counter[str], group: str, values: dict[str, float]) -> None:
    for name, value in values.items():
        if isinstance(value, float) and math.isnan(value):
            counter[f"{group}.{name}"] += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--labels-runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = extract_feature_tables(args.bundle_dir, args.labels_runs, args.output_dir)
    print(json.dumps(audit["row_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
