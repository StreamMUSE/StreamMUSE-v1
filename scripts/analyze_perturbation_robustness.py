#!/usr/bin/env python3
"""Analyze melody-perturbation sensitivity, quality, controls, and interactions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from streammuse.experiments.melody_robustness import (
    build_run_schedule,
    file_sha256,
    read_jsonl,
    validate_campaign_config,
    validate_staged_input_manifest,
    write_canonical_json,
    write_jsonl,
)
from streammuse.experiments.robustness_metrics import (
    Roll,
    bootstrap_song_mean,
    coverage_metrics,
    dissonance_metrics,
    load_midi_roll,
    rhythmic_metrics,
    sensitivity_metrics,
    transform_roll,
    write_roll_midi,
)


CONDITIONS = ("sham", "pitch", "onset", "both", "high")
COVERAGE_ENDPOINTS = (
    "onsets_per_beat",
    "active_pitch_ticks_per_beat",
    "empty_beat_ratio",
)
OPERATIONAL_COUNT_ENDPOINTS = (
    "stale_request_drops",
    "late_events",
    "clamped_onsets",
    "dropped_model_events",
    "orphan_note_offs",
    "forced_note_offs",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _nested_path(entry: Mapping[str, Any], *keys: str) -> str:
    containers = [entry]
    for name in ("paths", "artifacts", "files"):
        if isinstance(entry.get(name), Mapping):
            containers.append(entry[name])  # type: ignore[arg-type]
    for container in containers:
        for key in keys:
            value = container.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, Mapping) and value.get("path"):
                return str(value["path"])
    raise KeyError(f"missing path field {keys}")


def _resolve(entry: Mapping[str, Any], base: Path, *keys: str) -> Path:
    path = Path(_nested_path(entry, *keys))
    return (path if path.is_absolute() else base / path).resolve()


def _single(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} under {root}, found {matches}")
    return matches[0]


def _validated_campaign_inputs(
    config_path: Path,
    schedule_path: Path,
    *,
    expected_config_sha256: str | None,
    expected_schedule_sha256: str | None,
    allow_unpinned_schedule: bool,
) -> tuple[dict[str, Any], str, dict[str, Any], Path, list[dict[str, Any]], str]:
    """Load the three frozen campaign inputs and reject provenance drift.

    The schedule comparison is intentionally structural and order-sensitive.
    Merely presenting another valid 160-row schedule (or recomputing its hash
    after editing it) cannot make it part of this campaign.
    """
    config_path = config_path.resolve()
    schedule_path = schedule_path.resolve()
    config = _read_json(config_path)
    validate_campaign_config(config)
    config_sha = file_sha256(config_path)
    if expected_config_sha256 and config_sha != expected_config_sha256:
        raise RuntimeError("campaign config hash mismatch")
    checkpoint_path = Path(config["checkpoint"]["path"]).resolve()
    if file_sha256(checkpoint_path) != str(config["checkpoint"]["sha256"]):
        raise RuntimeError("checkpoint hash mismatch with frozen campaign config")

    input_manifest_path = Path(config["input_manifest"]["path"]).resolve()
    configured_manifest_sha = str(config["input_manifest"]["sha256"])
    actual_manifest_sha = file_sha256(input_manifest_path)
    if actual_manifest_sha != configured_manifest_sha:
        raise RuntimeError(
            "input manifest hash mismatch: frozen campaign config does not match file"
        )
    input_manifest = _read_json(input_manifest_path)
    validate_staged_input_manifest(
        input_manifest, manifest_path=input_manifest_path, verify_files=True
    )

    schedule_sha = file_sha256(schedule_path)
    if expected_schedule_sha256 is None:
        if not allow_unpinned_schedule:
            raise RuntimeError(
                "run schedule SHA-256 is required; pass --schedule-sha256 for a formal "
                "analysis or explicitly opt into --allow-unpinned-schedule for development"
            )
    elif schedule_sha != expected_schedule_sha256:
        raise RuntimeError("run schedule hash mismatch")
    schedule = read_jsonl(schedule_path)
    expected_schedule = build_run_schedule(input_manifest, config)
    if schedule != expected_schedule:
        raise RuntimeError(
            "run schedule is not the deterministic schedule rebuilt from the frozen "
            "campaign config and input manifest"
        )
    return (
        config,
        config_sha,
        input_manifest,
        input_manifest_path,
        schedule,
        schedule_sha,
    )


def _attempt_dir(output_root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    run_dir = output_root / "runs" / run_id
    verdict = _read_json(run_dir / "latest_verdict.json")
    return run_dir / str(verdict["attempt_id"]), verdict


def _validated_output_binding(
    output_root: Path,
    *,
    config_path: Path,
    schedule_path: Path,
    config: Mapping[str, Any],
    config_sha: str,
    schedule_sha: str,
    allow_missing_development_binding: bool = False,
) -> dict[str, Any]:
    path = output_root / "campaign_binding.json"
    expected = {
        "schema_version": "streammuse.melody_robustness.campaign_binding.v1",
        "qualification": False,
        "campaign_config_path": str(config_path.resolve()),
        "campaign_config_sha256": config_sha,
        "run_schedule_path": str(schedule_path.resolve()),
        "run_schedule_sha256": schedule_sha,
        "input_manifest_path": str(Path(config["input_manifest"]["path"]).resolve()),
        "input_manifest_sha256": str(config["input_manifest"]["sha256"]),
        "checkpoint_path": str(Path(config["checkpoint"]["path"]).resolve()),
        "checkpoint_sha256": str(config["checkpoint"]["sha256"]),
        "code_identity": str(config["code_identity"]),
    }
    if not path.is_file():
        if allow_missing_development_binding:
            return {
                **expected,
                "campaign_binding_sha256": None,
                "development_unbound_output_root": True,
            }
        raise RuntimeError("model output root has no formal campaign binding")
    if _read_json(path) != expected:
        raise RuntimeError("model output root is not bound to this exact formal campaign")
    return {**expected, "campaign_binding_sha256": file_sha256(path)}


def _output_paths(
    row: Mapping[str, Any], output_root: Path
) -> list[tuple[str, Path, str | None, dict[str, Any]]]:
    attempt, verdict = _attempt_dir(output_root, str(row["run_id"]))
    if row["pipeline"] == "offline":
        return [("offline", _single(attempt, f"{row['run_id']}_generated.mid"), None, verdict)]
    return [
        ("rt_theoretical", _single(attempt, "theoretical_model.mid"), None, verdict),
        ("rt_combined", _single(attempt, "combined.mid"), "Accompaniment", verdict),
    ]


def _metric_row(
    row: Mapping[str, Any], output_kind: str, accompaniment_path: Path,
    track_filter: str | None, verdict: Mapping[str, Any], manifest_dir: Path,
    config_sha: str,
) -> tuple[dict[str, Any], dict[str, Roll]]:
    entry = row["input"]
    end = int(entry["analysis_end_tick"])
    clean_path = _resolve(entry, manifest_dir, "source_midi", "clean_midi", "clean_source_midi")
    actual_path = _resolve(entry, manifest_dir, "output_midi", "melody_midi")
    clean = load_midi_roll(clean_path, end_tick=end)
    actual = load_midi_roll(actual_path, end_tick=end)
    accompaniment = load_midi_roll(
        accompaniment_path, end_tick=end, track_name_contains=track_filter
    )
    intended = dissonance_metrics(clean, accompaniment)
    actual_quality = dissonance_metrics(actual, accompaniment)
    metric = {
        "campaign_config_sha256": config_sha,
        "run_id": row["run_id"], "pipeline": output_kind,
        "song": row["song"], "condition": row["condition"],
        "perturb_seed": row.get("perturb_seed"), "sample_seed": row["sample_seed"],
        "content_valid": bool(verdict.get("content_valid")),
        "operational_valid": bool(verdict.get("operational_valid")),
        "analysis_end_tick": end,
        "paths": {
            "clean_melody": str(clean_path), "actual_melody": str(actual_path),
            "accompaniment": str(accompaniment_path),
        },
        "hashes": {
            "clean_melody": file_sha256(clean_path), "actual_melody": file_sha256(actual_path),
            "accompaniment": file_sha256(accompaniment_path),
        },
        "quality": {"D_intended": intended, "D_actual": actual_quality},
        "coverage": coverage_metrics(accompaniment, end_tick=end),
        "rhythm_actual": rhythmic_metrics(actual, accompaniment),
    }
    return metric, {"clean": clean, "actual": actual, "acc": accompaniment}


def _paired_rows(
    metrics: list[dict[str, Any]], rolls: Mapping[tuple[str, str], dict[str, Roll]],
    config_sha: str,
) -> list[dict[str, Any]]:
    sham: dict[tuple[str, str, int], dict[str, Any]] = {}
    for metric in metrics:
        if metric["condition"] == "sham":
            sham[(metric["song"], metric["pipeline"], metric["sample_seed"])] = metric
    result: list[dict[str, Any]] = []
    for metric in metrics:
        if metric["condition"] == "sham":
            continue
        key = (metric["song"], metric["pipeline"], metric["sample_seed"])
        baseline = sham.get(key)
        base_rolls = rolls.get((baseline["run_id"], baseline["pipeline"])) if baseline else None
        cond_rolls = rolls[(metric["run_id"], metric["pipeline"])]
        valid = bool(
            baseline and metric["content_valid"] and baseline["content_valid"]
        )
        cond_d = metric["quality"]["D_intended"]["D_micro"]
        sham_d = baseline["quality"]["D_intended"]["D_micro"] if baseline else None
        intended_effect = cond_d - sham_d if valid and cond_d is not None and sham_d is not None else None
        cond_actual_d = metric["quality"]["D_actual"]["D_micro"]
        sham_actual_d = (
            baseline["quality"]["D_actual"]["D_micro"] if baseline else None
        )
        actual_joint_treatment_effect = (
            cond_actual_d - sham_actual_d
            if valid and cond_actual_d is not None and sham_actual_d is not None
            else None
        )
        direct = adaptation = None
        cross = None
        if valid and base_rolls is not None:
            cross = dissonance_metrics(cond_rolls["actual"], base_rolls["acc"])
            cross_d = cross["D_micro"]
            sham_clean_d = baseline["quality"]["D_intended"]["D_micro"]
            if cross_d is not None and sham_clean_d is not None:
                direct = cross_d - sham_clean_d
            if cond_actual_d is not None and cross_d is not None:
                adaptation = cond_actual_d - cross_d
        sensitivity = (
            sensitivity_metrics(cond_rolls["acc"], base_rolls["acc"])
            if valid and base_rolls is not None else None
        )
        result.append(
            {
                "campaign_config_sha256": config_sha,
                "song": metric["song"], "pipeline": metric["pipeline"],
                "condition": metric["condition"], "perturb_seed": metric["perturb_seed"],
                "sample_seed": metric["sample_seed"], "run_id": metric["run_id"],
                "sham_run_id": baseline["run_id"] if baseline else None,
                "pair_valid": valid,
                "na_reason": None if intended_effect is not None else "invalid_pair_or_D_NA",
                "intended_fidelity_effect": intended_effect,
                # This endpoint deliberately has an explicit estimand name.  It is
                # the joint treatment effect, not the cross-reference adaptation
                # contrast below.
                "D_actual_cond": cond_actual_d,
                "D_actual_sham": sham_actual_d,
                "D_actual_cond_minus_sham": actual_joint_treatment_effect,
                "D_actual_joint_treatment_na_reason": (
                    None
                    if actual_joint_treatment_effect is not None
                    else "invalid_pair_or_D_actual_NA"
                ),
                "direct_melody_effect": direct,
                "adaptation_effect": adaptation,
                "sham_acc_vs_dirty": cross,
                "coverage_delta": {
                    key: metric["coverage"][key] - baseline["coverage"][key]
                    if valid and baseline else None
                    for key in COVERAGE_ENDPOINTS
                },
                **{
                    f"coverage_delta_{coverage_key}": (
                        metric["coverage"][coverage_key]
                        - baseline["coverage"][coverage_key]
                        if valid and baseline else None
                    )
                    for coverage_key in COVERAGE_ENDPOINTS
                },
                "sensitivity": sensitivity,
            }
        )
    return result


def _song_effects(
    pairs: Iterable[Mapping[str, Any]], *, pipeline: str, condition: str,
    endpoint: str = "intended_fidelity_effect",
    songs: Iterable[str] | None = None,
) -> dict[str, float | None]:
    grouped: dict[str, list[float | None]] = defaultdict(list)
    for row in pairs:
        if row["pipeline"] == pipeline and row["condition"] == condition:
            grouped[str(row["song"])].append(row.get(endpoint))
    expected_per_song = _expected_rows_per_song(condition)
    result: dict[str, float | None] = {}
    for song in sorted(set(grouped) | set(songs or [])):
        values = grouped.get(song, [])
        result[song] = (
            statistics.fmean(float(value) for value in values if value is not None)
            if len(values) == expected_per_song and all(value is not None for value in values)
            else None
        )
    return result


def _expected_rows_per_song(condition: str) -> int:
    """Return the frozen number of model rows in one pipeline/song block."""
    if condition in {"sham", "high"}:
        return 2
    if condition in {"pitch", "onset", "both"}:
        return 4
    raise ValueError(f"unknown condition: {condition}")


def _complete_song_means(
    rows: Iterable[Mapping[str, Any]],
    *,
    condition: str,
    value_getter: Callable[[Mapping[str, Any]], float | None],
    songs: Iterable[str],
) -> dict[str, float | None]:
    """Equal-weight complete-block means with no complete-case deletion."""
    grouped: dict[str, list[float | None]] = defaultdict(list)
    for row in rows:
        grouped[str(row["song"])].append(value_getter(row))
    expected = _expected_rows_per_song(condition)
    result: dict[str, float | None] = {}
    for song in sorted(set(grouped) | {str(song) for song in songs}):
        values = grouped.get(song, [])
        result[song] = (
            statistics.fmean(float(value) for value in values if value is not None)
            if len(values) == expected and all(value is not None for value in values)
            else None
        )
    return result


def _numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _delta(left: Any, right: Any, *, valid: bool) -> float | None:
    left_value = _numeric(left)
    right_value = _numeric(right)
    if not valid or left_value is None or right_value is None:
        return None
    return left_value - right_value


def _operational_rows(
    metrics: Iterable[Mapping[str, Any]],
    config_sha: str,
    *,
    expected_schedule: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build within-run raw ``combined - theoretical`` RT endpoints.

    The two MIDI products come from the same model request trace.  Keeping this
    raw within-run delta separate is essential: subtracting sham later answers
    a different, treatment-interaction question.
    """
    indexed: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    inferred: dict[str, Mapping[str, Any]] = {}
    for metric in metrics:
        pipeline = str(metric.get("pipeline"))
        if pipeline not in {"rt_theoretical", "rt_combined"}:
            continue
        run_id = str(metric["run_id"])
        if pipeline in indexed[run_id]:
            raise RuntimeError(f"duplicate {pipeline} metric for {run_id}")
        indexed[run_id][pipeline] = metric
        inferred.setdefault(run_id, metric)

    expected: dict[str, Mapping[str, Any]] = {}
    if expected_schedule is not None:
        for row in expected_schedule:
            if row.get("pipeline") == "rt":
                expected[str(row["run_id"])] = row
    else:
        expected = inferred

    rows: list[dict[str, Any]] = []
    for run_id in sorted(set(expected) | set(indexed)):
        arms = indexed.get(run_id, {})
        theoretical = arms.get("rt_theoretical")
        combined = arms.get("rt_combined")
        source = expected.get(run_id) or theoretical or combined
        assert source is not None
        complete = theoretical is not None and combined is not None
        content_valid = bool(
            complete
            and theoretical.get("content_valid") is True
            and combined.get("content_valid") is True
        )
        operational_valid = bool(
            complete
            and theoretical.get("operational_valid") is True
            and combined.get("operational_valid") is True
        )

        def quality(metric: Mapping[str, Any] | None, endpoint: str) -> Any:
            return (
                metric.get("quality", {}).get(endpoint, {}).get("D_micro")
                if metric is not None else None
            )

        raw_d_intended = _delta(
            quality(combined, "D_intended"),
            quality(theoretical, "D_intended"),
            valid=content_valid,
        )
        raw_d_actual = _delta(
            quality(combined, "D_actual"),
            quality(theoretical, "D_actual"),
            valid=content_valid,
        )
        coverage_delta = {
            endpoint: _delta(
                combined.get("coverage", {}).get(endpoint) if combined else None,
                theoretical.get("coverage", {}).get(endpoint) if theoretical else None,
                valid=content_valid,
            )
            for endpoint in COVERAGE_ENDPOINTS
        }
        row = {
            "campaign_config_sha256": config_sha,
            "run_id": run_id,
            "song": source["song"],
            "condition": source["condition"],
            "perturb_seed": source.get("perturb_seed"),
            "sample_seed": source["sample_seed"],
            "complete_theoretical_combined_pair": complete,
            "content_pair_valid": content_valid,
            "operational_valid": operational_valid,
            "na_reason": (
                None
                if raw_d_intended is not None
                else "missing_output_or_content_invalid_or_D_intended_NA"
            ),
            "D_intended_theoretical": quality(theoretical, "D_intended"),
            "D_intended_combined": quality(combined, "D_intended"),
            "D_intended_combined_minus_theoretical": raw_d_intended,
            "D_actual_theoretical": quality(theoretical, "D_actual"),
            "D_actual_combined": quality(combined, "D_actual"),
            "D_actual_combined_minus_theoretical": raw_d_actual,
            "coverage_combined_minus_theoretical": coverage_delta,
            **{
                f"coverage_{endpoint}_combined_minus_theoretical": value
                for endpoint, value in coverage_delta.items()
            },
        }
        rows.append(row)
    return rows


def _paired_operational_rows(
    raw_rows: Iterable[Mapping[str, Any]], config_sha: str
) -> list[dict[str, Any]]:
    """Pair each raw operational delta with same-song/sample-seed sham."""
    materialized = list(raw_rows)
    sham = {
        (str(row["song"]), int(row["sample_seed"])): row
        for row in materialized
        if row["condition"] == "sham"
    }
    result: list[dict[str, Any]] = []
    for row in materialized:
        if row["condition"] == "sham":
            continue
        baseline = sham.get((str(row["song"]), int(row["sample_seed"])))
        valid = bool(
            baseline
            and row.get("content_pair_valid") is True
            and baseline.get("content_pair_valid") is True
        )
        intended = _delta(
            row.get("D_intended_combined_minus_theoretical"),
            baseline.get("D_intended_combined_minus_theoretical") if baseline else None,
            valid=valid,
        )
        actual = _delta(
            row.get("D_actual_combined_minus_theoretical"),
            baseline.get("D_actual_combined_minus_theoretical") if baseline else None,
            valid=valid,
        )
        coverage = {
            endpoint: _delta(
                row.get(f"coverage_{endpoint}_combined_minus_theoretical"),
                baseline.get(f"coverage_{endpoint}_combined_minus_theoretical")
                if baseline else None,
                valid=valid,
            )
            for endpoint in COVERAGE_ENDPOINTS
        }
        result.append(
            {
                "campaign_config_sha256": config_sha,
                "song": row["song"],
                "condition": row["condition"],
                "perturb_seed": row.get("perturb_seed"),
                "sample_seed": row["sample_seed"],
                "run_id": row["run_id"],
                "sham_run_id": baseline["run_id"] if baseline else None,
                "pair_valid": valid,
                "na_reason": (
                    None
                    if intended is not None
                    else "missing_sham_or_content_invalid_or_D_intended_NA"
                ),
                "D_intended_operational_treatment_interaction": intended,
                "D_actual_operational_treatment_interaction": actual,
                "coverage_operational_treatment_interaction": coverage,
                **{
                    f"coverage_{endpoint}_operational_treatment_interaction": value
                    for endpoint, value in coverage.items()
                },
            }
        )
    return result


def _endpoint_summary(
    per_song: Mapping[str, float | None], *, seed: int
) -> dict[str, Any]:
    return {
        "per_song": dict(per_song),
        "bootstrap": bootstrap_song_mean(per_song, seed=seed),
    }


def _coverage_summary(
    metrics: Iterable[Mapping[str, Any]],
    pairs: Iterable[Mapping[str, Any]],
    *,
    expected_songs: Iterable[str],
    seed: int,
    config_sha: str,
) -> dict[str, Any]:
    """Report absolute coverage and paired condition-minus-sham coverage."""
    metric_rows = list(metrics)
    pair_rows = list(pairs)
    songs = {str(song) for song in expected_songs}
    absolute: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for pipeline in ("offline", "rt_theoretical", "rt_combined"):
        absolute[pipeline] = {}
        paired[pipeline] = {}
        for condition in CONDITIONS:
            selected_metrics = [
                row for row in metric_rows
                if row["pipeline"] == pipeline and row["condition"] == condition
            ]
            absolute[pipeline][condition] = {}
            for endpoint in COVERAGE_ENDPOINTS:
                per_song = _complete_song_means(
                    selected_metrics,
                    condition=condition,
                    value_getter=lambda row, key=endpoint: (
                        _numeric(row.get("coverage", {}).get(key))
                        if row.get("content_valid") is True else None
                    ),
                    songs=songs,
                )
                absolute[pipeline][condition][endpoint] = _endpoint_summary(
                    per_song, seed=seed
                )
            if condition == "sham":
                continue
            paired[pipeline][condition] = {}
            for endpoint in COVERAGE_ENDPOINTS:
                per_song = _song_effects(
                    pair_rows,
                    pipeline=pipeline,
                    condition=condition,
                    endpoint=f"coverage_delta_{endpoint}",
                    songs=songs,
                )
                paired[pipeline][condition][endpoint] = _endpoint_summary(
                    per_song, seed=seed
                )
    return {
        "campaign_config_sha256": config_sha,
        "semantics": {
            "absolute": "condition coverage; complete equal-weight song blocks",
            "paired_delta_vs_sham": (
                "coverage(condition)-coverage(sham), paired by song and sample seed"
            ),
            "bootstrap_unit": "complete_song_block",
            "incomplete_block_policy": "NA_without_complete_case_deletion",
        },
        "endpoints": list(COVERAGE_ENDPOINTS),
        "absolute": absolute,
        "paired_delta_vs_sham": paired,
    }


def _operational_endpoint_summary(
    raw_rows: Iterable[Mapping[str, Any]],
    treatment_rows: Iterable[Mapping[str, Any]],
    *,
    expected_songs: Iterable[str],
    seed: int,
    config_sha: str,
) -> dict[str, Any]:
    raw_materialized = list(raw_rows)
    treatment_materialized = list(treatment_rows)
    songs = {str(song) for song in expected_songs}
    raw_summary: dict[str, Any] = {}
    treatment_summary: dict[str, Any] = {}
    raw_fields = {
        "D_intended": "D_intended_combined_minus_theoretical",
        "D_actual": "D_actual_combined_minus_theoretical",
        **{
            f"coverage.{endpoint}": f"coverage_{endpoint}_combined_minus_theoretical"
            for endpoint in COVERAGE_ENDPOINTS
        },
    }
    interaction_fields = {
        "D_intended": "D_intended_operational_treatment_interaction",
        "D_actual": "D_actual_operational_treatment_interaction",
        **{
            f"coverage.{endpoint}": (
                f"coverage_{endpoint}_operational_treatment_interaction"
            )
            for endpoint in COVERAGE_ENDPOINTS
        },
    }
    for condition in CONDITIONS:
        selected = [row for row in raw_materialized if row["condition"] == condition]
        raw_summary[condition] = {}
        for endpoint, field in raw_fields.items():
            per_song = _complete_song_means(
                selected,
                condition=condition,
                value_getter=lambda row, key=field: _numeric(row.get(key)),
                songs=songs,
            )
            raw_summary[condition][endpoint] = _endpoint_summary(per_song, seed=seed)
        if condition == "sham":
            continue
        selected_treatment = [
            row for row in treatment_materialized if row["condition"] == condition
        ]
        treatment_summary[condition] = {}
        for endpoint, field in interaction_fields.items():
            per_song = _complete_song_means(
                selected_treatment,
                condition=condition,
                value_getter=lambda row, key=field: _numeric(row.get(key)),
                songs=songs,
            )
            treatment_summary[condition][endpoint] = _endpoint_summary(
                per_song, seed=seed
            )
    return {
        "campaign_config_sha256": config_sha,
        "semantics": {
            "raw_operational_delta": (
                "rt_combined minus rt_theoretical within the same RT run; "
                "not sham-adjusted"
            ),
            "treatment_interaction_vs_sham": (
                "raw_operational_delta(condition) minus "
                "raw_operational_delta(sham), paired by song and sample seed"
            ),
            "content_failure_policy": "endpoint_NA; operational_invalid_runs_retained",
            "bootstrap_unit": "complete_song_block",
        },
        "raw_rows": raw_materialized,
        "treatment_interaction_rows": treatment_materialized,
        "raw_by_condition": raw_summary,
        "treatment_interaction_vs_sham": treatment_summary,
    }


def _nonnegative_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _lifecycle_row(
    schedule_row: Mapping[str, Any],
    attempt_dir: Path,
    verdict: Mapping[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    """Extract one RT run's structured lifecycle without zero-filling gaps."""
    validity = verdict.get("validity")
    if not isinstance(validity, Mapping):
        raise ValueError("verdict has no structured validity object")
    content = validity.get("content")
    operational = validity.get("operational")
    requests = validity.get("requests")
    drain = validity.get("drain")
    if not isinstance(content, Mapping):
        raise ValueError("validity.content is missing")
    if not isinstance(operational, Mapping):
        raise ValueError("validity.operational is missing")
    if not isinstance(requests, list) or not all(
        isinstance(request, Mapping) for request in requests
    ):
        raise ValueError("validity.requests must be a list of objects")
    if not isinstance(drain, Mapping):
        raise ValueError("validity.drain is missing")

    lifecycle_path = _single(attempt_dir, "request_lifecycle.jsonl")
    lifecycle_events = read_jsonl(lifecycle_path)
    event_counts = Counter(str(event.get("event")) for event in lifecycle_events)
    http_errors = sum(
        event.get("event") == "failed" and event.get("http_error") is True
        for event in lifecycle_events
    )
    request_counts = {
        field: sum(request.get(field) is True for request in requests)
        for field in (
            "expected", "enqueued", "started", "succeeded", "failed",
            "processed", "stale_dropped", "empty_success",
        )
    }
    operational_counts = {
        endpoint: _nonnegative_int(operational, endpoint)
        for endpoint in OPERATIONAL_COUNT_ENDPOINTS
    }
    max_lateness = _nonnegative_int(operational, "max_lateness_ticks")
    content_lists = {}
    for name in (
        "expected_request_ids", "succeeded_request_ids", "processed_request_ids",
        "failed_request_ids", "stale_dropped_request_ids",
        "metadata_invalid_request_ids", "pending_at_stop_request_ids",
        "missing_generation_start_ticks", "unexpected_generation_start_ticks",
        "duplicate_generation_start_ticks", "rejected_generation_start_ticks",
    ):
        value = content.get(name)
        if not isinstance(value, list):
            raise ValueError(f"validity.content.{name} must be a list")
        content_lists[name] = value
    request_count = _nonnegative_int(content, "request_count")
    coverage = _numeric(content.get("analysis_request_coverage"))
    if coverage is None or not 0.0 <= coverage <= 1.0:
        raise ValueError("analysis_request_coverage must be in [0, 1]")

    consistency_errors: list[str] = []
    expected_by_record = request_counts["expected"]
    if request_count != expected_by_record:
        consistency_errors.append("request_count disagrees with request records")
    for flag, list_name in (
        ("expected", "expected_request_ids"),
        ("succeeded", "succeeded_request_ids"),
        ("processed", "processed_request_ids"),
        ("failed", "failed_request_ids"),
        ("stale_dropped", "stale_dropped_request_ids"),
    ):
        if request_counts[flag] != len(content_lists[list_name]):
            consistency_errors.append(f"{flag} count disagrees with content summary")
        if event_counts[flag] != request_counts[flag]:
            consistency_errors.append(f"{flag} lifecycle events disagree with request records")
    if event_counts["enqueued"] != request_counts["enqueued"]:
        consistency_errors.append("enqueued lifecycle events disagree with request records")
    if event_counts["started"] != request_counts["started"]:
        consistency_errors.append("started lifecycle events disagree with request records")
    if operational_counts["stale_request_drops"] != request_counts["stale_dropped"]:
        consistency_errors.append("stale operational count disagrees with request records")

    flattened_counts = {
        "expected_requests": request_counts["expected"],
        "enqueued_requests": request_counts["enqueued"],
        "started_requests": request_counts["started"],
        "succeeded_requests": request_counts["succeeded"],
        "failed_requests": request_counts["failed"],
        "processed_requests": request_counts["processed"],
        "stale_dropped_requests": request_counts["stale_dropped"],
        "empty_success_requests": request_counts["empty_success"],
        "http_errors": int(http_errors),
        "pending_at_stop_requests": len(content_lists["pending_at_stop_request_ids"]),
        "metadata_invalid_requests": len(content_lists["metadata_invalid_request_ids"]),
        "missing_generation_ticks": len(content_lists["missing_generation_start_ticks"]),
        "unexpected_generation_ticks": len(
            content_lists["unexpected_generation_start_ticks"]
        ),
        "duplicate_generation_ticks": len(
            content_lists["duplicate_generation_start_ticks"]
        ),
        "rejected_generation_ticks": len(content_lists["rejected_generation_start_ticks"]),
        "drain_timeouts": int(drain.get("timed_out") is True),
        **operational_counts,
    }
    return {
        "campaign_config_sha256": config_sha,
        "run_id": schedule_row["run_id"],
        "song": schedule_row["song"],
        "condition": schedule_row["condition"],
        "perturb_seed": schedule_row.get("perturb_seed"),
        "sample_seed": schedule_row["sample_seed"],
        "attempt_id": verdict.get("attempt_id"),
        "schema_valid": not consistency_errors,
        "schema_errors": consistency_errors,
        "content_valid": content.get("valid") is True,
        "operational_valid": operational.get("valid") is True,
        "analysis_request_coverage": coverage,
        "counts": flattened_counts,
        "max_lateness_ticks": max_lateness,
        "event_counts": dict(sorted(event_counts.items())),
        "paths": {
            "attempt_dir": str(attempt_dir),
            "request_lifecycle": str(lifecycle_path),
        },
        "hashes": {"request_lifecycle": file_sha256(lifecycle_path)},
    }


def _collect_lifecycle_rows(
    schedule: Iterable[Mapping[str, Any]],
    output_root: Path,
    config_sha: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in schedule:
        if row.get("pipeline") != "rt":
            continue
        try:
            attempt_dir, verdict = _attempt_dir(output_root, str(row["run_id"]))
            result.append(_lifecycle_row(row, attempt_dir, verdict, config_sha))
        except Exception as exc:
            result.append(
                {
                    "campaign_config_sha256": config_sha,
                    "run_id": row["run_id"],
                    "song": row["song"],
                    "condition": row["condition"],
                    "perturb_seed": row.get("perturb_seed"),
                    "sample_seed": row["sample_seed"],
                    "schema_valid": False,
                    "schema_errors": [f"{type(exc).__name__}: {exc}"],
                    "content_valid": False,
                    "operational_valid": False,
                    "analysis_request_coverage": None,
                    "counts": None,
                    "max_lateness_ticks": None,
                }
            )
    return result


def _aggregate_lifecycle_group(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    valid = [row for row in materialized if row.get("schema_valid") is True]
    invalid_ids = [str(row["run_id"]) for row in materialized if row not in valid]
    count_keys = sorted(
        {
            key
            for row in valid
            for key in (row.get("counts") or {})
        }
    )
    observed_totals = {
        key: sum(int(row["counts"][key]) for row in valid)
        for key in count_keys
    }
    coverages = [float(row["analysis_request_coverage"]) for row in valid]
    max_lateness = [int(row["max_lateness_ticks"]) for row in valid]
    complete = len(valid) == len(materialized)
    return {
        "expected_run_count": len(materialized),
        "schema_valid_run_count": len(valid),
        "complete": complete,
        "invalid_run_ids": invalid_ids,
        # A partial total is never presented as the campaign total.
        "all_run_totals": observed_totals if complete else None,
        "observed_schema_valid_run_totals": observed_totals,
        "content_invalid_run_count": sum(
            row.get("content_valid") is not True for row in materialized
        ),
        "operational_invalid_run_count": sum(
            row.get("operational_valid") is not True for row in materialized
        ),
        "max_lateness_ticks": max(max_lateness) if max_lateness else None,
        "analysis_request_coverage_min": min(coverages) if coverages else None,
        "analysis_request_coverage_mean": (
            statistics.fmean(coverages) if coverages else None
        ),
    }


def _lifecycle_summary(
    rows: Iterable[Mapping[str, Any]], config_sha: str
) -> dict[str, Any]:
    materialized = list(rows)
    return {
        "campaign_config_sha256": config_sha,
        "semantics": {
            "count_aggregation": "sum across verdict-indexed latest RT attempts",
            "max_lateness_ticks": "maximum across runs",
            "missing_schema_policy": "all_run_totals_NA_not_zero_filled",
            "operational_count_endpoints": list(OPERATIONAL_COUNT_ENDPOINTS),
        },
        "rows": materialized,
        "aggregate": {
            "overall": _aggregate_lifecycle_group(materialized),
            "by_condition": {
                condition: _aggregate_lifecycle_group(
                    row for row in materialized if row["condition"] == condition
                )
                for condition in CONDITIONS
            },
        },
    }


def _factorial_effects(pairs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in pairs:
        if row["condition"] in {"pitch", "onset", "both"} and row["perturb_seed"] is not None:
            key = (row["song"], row["pipeline"], int(row["sample_seed"]), int(row["perturb_seed"]))
            indexed[key][row["condition"]] = row
    results = []
    for key, arms in sorted(indexed.items()):
        values = [arms.get(arm, {}).get("intended_fidelity_effect") for arm in ("both", "pitch", "onset")]
        value = values[0] - values[1] - values[2] if all(v is not None for v in values) else None
        results.append(
            {
                "song": key[0], "pipeline": key[1], "sample_seed": key[2],
                "perturb_seed": key[3], "factorial_interaction": value,
                "valid": value is not None,
            }
        )
    return results


def _targeted_controls(
    metrics: list[dict[str, Any]], rolls: Mapping[tuple[str, str], dict[str, Roll]],
    output_dir: Path, config_sha: str, expected_songs: set[str],
) -> dict[str, Any]:
    # Selection is deterministic and independent of treatment values: first
    # lexical sham theoretical/offline row per song, first sample seed.
    sources: dict[str, dict[str, Any]] = {}
    for metric in sorted(metrics, key=lambda row: (row["song"], row["pipeline"], row["sample_seed"])):
        if metric["condition"] == "sham" and metric["pipeline"] in {"offline", "rt_theoretical"}:
            sources.setdefault(metric["song"], metric)
    reports = []
    for song, source in sorted(sources.items()):
        source_rolls = rolls[(source["run_id"], source["pipeline"])]
        original = source_rolls["acc"]
        original_d = dissonance_metrics(source_rolls["clean"], original)
        original_coverage = coverage_metrics(original)
        song_dir = output_dir / "controls" / song
        controls: dict[str, Any] = {}
        transformed_rolls: dict[str, Roll] = {}
        for kind in ("identity", "harmonic_m2", "harmonic_tt", "rhythm_shift", "coverage_dropout", "coverage_empty"):
            transformed = transform_roll(original, kind)
            transformed_rolls[kind] = transformed
            path = song_dir / f"{kind}.mid"
            write_roll_midi(transformed, path, bpm=120)
            controls[kind] = {
                "path": str(path), "sha256": file_sha256(path),
                "quality": dissonance_metrics(source_rolls["clean"], transformed),
                "coverage": coverage_metrics(transformed),
                "sensitivity": sensitivity_metrics(original, transformed),
            }
        identity_pass = transformed_rolls["identity"] == original
        harmonic_pass = all(
            controls[kind]["quality"]["D_micro"] is not None
            and original_d["D_micro"] is not None
            and controls[kind]["quality"]["D_micro"] - original_d["D_micro"] >= 0.02
            for kind in ("harmonic_m2", "harmonic_tt")
        )
        rhythm_distance = controls["rhythm_shift"]["sensitivity"]["onset_distance"]["mean_ticks"]
        rhythm_pass = rhythm_distance is not None and rhythm_distance >= 1.0
        dropout_pass = (
            controls["coverage_dropout"]["coverage"]["active_pitch_ticks_per_beat"]
            <= original_coverage["active_pitch_ticks_per_beat"] * 0.75
        )
        empty_pass = (
            controls["coverage_empty"]["quality"]["D_micro"] is None
            and controls["coverage_empty"]["quality"]["coverage_failure"]
        )
        reports.append(
            {
                "campaign_config_sha256": config_sha, "song": song,
                "source_run_id": source["run_id"], "controls": controls,
                "acceptance": {
                    "identity": identity_pass, "harmonic": harmonic_pass,
                    "rhythm": rhythm_pass, "coverage_dropout": dropout_pass,
                    "coverage_empty": empty_pass,
                },
                "endpoint_validity": {
                    "harmonic": identity_pass and harmonic_pass,
                    "rhythm": identity_pass and rhythm_pass,
                    "coverage": identity_pass and dropout_pass and empty_pass,
                },
            }
        )
    return {
        "campaign_config_sha256": config_sha,
        "thresholds": {"harmonic_D_increase": 0.02, "rhythm_ticks": 1.0, "dropout_ratio": 0.75},
        "expected_songs": sorted(expected_songs),
        "missing_songs": sorted(expected_songs - set(sources)),
        "songs": reports,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_plot(path: Path, summaries: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(summaries)
    values = [summaries[label].get("estimate") for label in labels]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.bar(labels, [value if value is not None else 0 for value in values])
    axis.set_ylabel("Δ D_intended (higher = worse)")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyze_campaign(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    schedule_path = Path(args.schedule).resolve()
    allow_unpinned_schedule = bool(getattr(args, "allow_unpinned_schedule", False))
    (
        config,
        config_sha,
        _input_manifest,
        input_manifest_path,
        schedule,
        schedule_sha,
    ) = _validated_campaign_inputs(
        config_path,
        schedule_path,
        expected_config_sha256=getattr(args, "config_sha256", None),
        expected_schedule_sha256=getattr(args, "schedule_sha256", None),
        allow_unpinned_schedule=allow_unpinned_schedule,
    )
    expected_songs = {str(row["song"]) for row in schedule}
    output_root = Path(args.output_root).resolve()
    campaign_binding = _validated_output_binding(
        output_root,
        config_path=config_path,
        schedule_path=schedule_path,
        config=config,
        config_sha=config_sha,
        schedule_sha=schedule_sha,
        allow_missing_development_binding=allow_unpinned_schedule,
    )
    report_root = Path(args.report_dir).resolve()
    manifest_dir = input_manifest_path.parent
    metrics: list[dict[str, Any]] = []
    rolls: dict[tuple[str, str], dict[str, Roll]] = {}
    qc: list[dict[str, Any]] = []
    for row in schedule:
        try:
            for kind, output_path, track_filter, verdict in _output_paths(row, output_root):
                metric, roll_set = _metric_row(
                    row, kind, output_path, track_filter, verdict, manifest_dir, config_sha
                )
                metrics.append(metric)
                rolls[(metric["run_id"], kind)] = roll_set
                qc.append({
                    "campaign_config_sha256": config_sha,
                    "run_id": row["run_id"], "pipeline": kind, "status": "loaded",
                })
        except Exception as exc:
            qc.append({
                "campaign_config_sha256": config_sha,
                "run_id": row["run_id"], "pipeline": row["pipeline"],
                "status": "invalid", "reason": str(exc),
            })
    pairs = _paired_rows(metrics, rolls, config_sha)
    factorial = _factorial_effects(pairs)
    raw_operational_rows = _operational_rows(
        metrics, config_sha, expected_schedule=schedule
    )
    operational_treatment_rows = _paired_operational_rows(
        raw_operational_rows, config_sha
    )
    lifecycle_rows = _collect_lifecycle_rows(schedule, output_root, config_sha)
    qc.extend(
        {
            "campaign_config_sha256": config_sha,
            "run_id": row["run_id"],
            "pipeline": "rt_lifecycle",
            "status": "loaded" if row.get("schema_valid") is True else "invalid",
            **(
                {}
                if row.get("schema_valid") is True
                else {"reason": "; ".join(row.get("schema_errors", []))}
            ),
        }
        for row in lifecycle_rows
    )
    bootstrap: dict[str, Any] = {}
    seed = int(config["seeds"]["bootstrap"])
    actual_treatment_summary: dict[str, Any] = {
        "campaign_config_sha256": config_sha,
        "semantics": {
            "estimand": "D_actual(condition)-D_actual(sham)",
            "name": "joint_treatment_effect_not_adaptation",
            "pairing": "song_and_sample_seed",
            "bootstrap_unit": "complete_song_block",
        },
        "results": {},
    }
    for pipeline in ("offline", "rt_theoretical", "rt_combined"):
        actual_treatment_summary["results"][pipeline] = {}
        for condition in ("both", "pitch", "onset", "high"):
            name = f"{pipeline}:{condition}_vs_sham"
            bootstrap[name] = bootstrap_song_mean(
                _song_effects(
                    pairs, pipeline=pipeline, condition=condition, songs=expected_songs
                ), seed=seed
            )
            actual_per_song = _song_effects(
                pairs,
                pipeline=pipeline,
                condition=condition,
                endpoint="D_actual_cond_minus_sham",
                songs=expected_songs,
            )
            actual_endpoint = _endpoint_summary(actual_per_song, seed=seed)
            actual_treatment_summary["results"][pipeline][condition] = actual_endpoint
            bootstrap[
                f"D_actual_joint_treatment:{pipeline}:{condition}_vs_sham"
            ] = actual_endpoint["bootstrap"]
    # Generation interaction uses equal-weight song effects, never pooled ticks.
    off = _song_effects(pairs, pipeline="offline", condition="both", songs=expected_songs)
    rt = _song_effects(pairs, pipeline="rt_theoretical", condition="both", songs=expected_songs)
    songs = sorted(set(off) | set(rt))
    generation_interaction = {
        song: (rt.get(song) - off.get(song))
        if rt.get(song) is not None and off.get(song) is not None else None
        for song in songs
    }
    bootstrap["H_generation_interaction:both"] = bootstrap_song_mean(
        generation_interaction, seed=seed
    )
    operational_summary = _operational_endpoint_summary(
        raw_operational_rows,
        operational_treatment_rows,
        expected_songs=expected_songs,
        seed=seed,
        config_sha=config_sha,
    )
    for condition in CONDITIONS:
        bootstrap[f"H_operational_raw:{condition}"] = operational_summary[
            "raw_by_condition"
        ][condition]["D_intended"]["bootstrap"]
        if condition != "sham":
            bootstrap[
                f"H_operational_treatment_interaction:{condition}"
            ] = operational_summary["treatment_interaction_vs_sham"][condition][
                "D_intended"
            ]["bootstrap"]
    # Backwards-compatible key: this has always meant the both-vs-sham
    # treatment interaction, not the raw within-run operational delta.
    bootstrap["H_operational:both"] = bootstrap[
        "H_operational_treatment_interaction:both"
    ]
    coverage_summary = _coverage_summary(
        metrics,
        pairs,
        expected_songs=expected_songs,
        seed=seed,
        config_sha=config_sha,
    )
    lifecycle_summary = _lifecycle_summary(lifecycle_rows, config_sha)
    controls = _targeted_controls(metrics, rolls, report_root, config_sha, expected_songs)
    write_jsonl(report_root / "run_metrics.jsonl", metrics)
    write_jsonl(report_root / "paired_contrasts.jsonl", pairs)
    write_canonical_json(report_root / "factorial_interactions.json", {
        "campaign_config_sha256": config_sha, "rows": factorial,
    })
    write_canonical_json(
        report_root / "D_actual_joint_treatment.json", actual_treatment_summary
    )
    write_canonical_json(
        report_root / "operational_endpoints.json", operational_summary
    )
    write_canonical_json(report_root / "lifecycle_summary.json", lifecycle_summary)
    write_canonical_json(report_root / "coverage_summary.json", coverage_summary)
    write_canonical_json(report_root / "bootstrap.json", {
        "campaign_config_sha256": config_sha,
        "unit": "complete_song_block", "descriptive": True, "results": bootstrap,
    })
    write_canonical_json(report_root / "control_report.json", controls)
    write_canonical_json(report_root / "run_qc.json", {
        "campaign_config_sha256": config_sha, "rows": qc,
    })
    table_rows = [
        {"campaign_config_sha256": config_sha, "contrast": name, **value}
        for name, value in bootstrap.items()
    ]
    _write_csv(
        report_root / "per_song_summary.csv", table_rows,
        [
            "campaign_config_sha256", "contrast", "estimate", "interval",
            "raw_song_effects", "valid_song_count", "leave_one_song_out_range",
        ],
    )
    _write_plot(
        report_root / "figures" / "primary_effects.png",
        {
            name: summary
            for name, summary in bootstrap.items()
            if not name.startswith("D_actual_joint_treatment:")
            and not name.startswith("H_operational_raw:")
            and not name.startswith("H_operational_treatment_interaction:")
        },
    )
    index = {
        "campaign_config_sha256": config_sha,
        "input_manifest_sha256": str(config["input_manifest"]["sha256"]),
        "run_schedule_sha256": schedule_sha,
        "campaign_binding_sha256": campaign_binding["campaign_binding_sha256"],
        "metric_rows": len(metrics), "paired_rows": len(pairs),
        "qc_invalid": sum(row["status"] != "loaded" for row in qc),
        "lifecycle_rows": len(lifecycle_rows),
        "lifecycle_schema_invalid": sum(
            row.get("schema_valid") is not True for row in lifecycle_rows
        ),
        "artifacts": {},
    }
    for path in sorted(report_root.rglob("*")):
        if path.is_file() and path.name != "analysis_index.json":
            index["artifacts"][str(path.relative_to(report_root))] = file_sha256(path)
    write_canonical_json(report_root / "analysis_index.json", index)


def generate_controls(args: argparse.Namespace) -> None:
    source = Path(args.accompaniment).resolve()
    roll = load_midi_roll(source, end_tick=args.analysis_end_tick)
    output = Path(args.output_dir).resolve()
    records = []
    for kind in ("identity", "harmonic_m2", "harmonic_tt", "rhythm_shift", "coverage_dropout", "coverage_empty"):
        path = output / f"{kind}.mid"
        write_roll_midi(transform_roll(roll, kind), path, bpm=120)
        records.append({"kind": kind, "path": str(path), "sha256": file_sha256(path)})
    write_canonical_json(output / "manifest.json", {"source": str(source), "controls": records})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    campaign = sub.add_parser("campaign")
    campaign.add_argument("--config", required=True)
    campaign.add_argument(
        "--config-sha256",
        help="expected frozen config SHA-256 (strongly recommended for formal analysis)",
    )
    campaign.add_argument("--schedule", required=True)
    schedule_pin = campaign.add_mutually_exclusive_group(required=True)
    schedule_pin.add_argument(
        "--schedule-sha256",
        help="required schedule pin for formal analysis",
    )
    schedule_pin.add_argument(
        "--allow-unpinned-schedule",
        action="store_true",
        help="development only; still requires deterministic schedule reconstruction",
    )
    campaign.add_argument("--output-root", required=True)
    campaign.add_argument("--report-dir", required=True)
    campaign.set_defaults(func=analyze_campaign)
    controls = sub.add_parser("controls")
    controls.add_argument("--accompaniment", required=True)
    controls.add_argument("--analysis-end-tick", type=int, required=True)
    controls.add_argument("--output-dir", required=True)
    controls.set_defaults(func=generate_controls)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
