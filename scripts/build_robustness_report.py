#!/usr/bin/env python3
"""Build the constrained pilot report and cross-linked reproducibility index."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from streammuse.experiments.melody_robustness import (
    build_run_schedule,
    canonical_sha256,
    file_sha256,
    read_jsonl,
    validate_campaign_config,
    validate_frozen_qualification,
    validate_staged_input_manifest,
    verify_attempt_verdict,
    write_canonical_json,
)
from streammuse.experiments.triangle_listening import (
    PRIMARY_CONDITIONS as TRIANGLE_PRIMARY_CONDITIONS,
    TRIANGLE_CLIP_SECONDS,
    TRIANGLE_GAIN_POLICY,
    TRIANGLE_PRACTICE_COUNT,
    TRIANGLE_PRESENTATION_COUNT,
    TRIANGLE_RENDER_BPM,
    TRIANGLE_RENDER_SAMPLE_RATE,
    TRIANGLE_RETRY_AUTHORIZATION_SCHEMA_VERSION,
    TRIANGLE_RETRY_LINEAGE_SCHEMA_VERSION,
    TRIANGLE_SELECTION_SCHEMA_VERSION,
    TRIANGLE_SYNTH_GAIN,
    TRIANGLE_TRIAL_COUNT,
    progress_summary as triangle_progress_summary,
    validate_response_ledger as validate_triangle_response_ledger,
    validate_snapshot as validate_triangle_snapshot,
    validate_triangle_selection_manifest,
    validate_unblinded_summary as validate_triangle_unblinded_summary,
)


TRIANGLE_RENDER_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_render.v2"
)
TRIANGLE_AUDIT_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_audit.v2"
)
TRIANGLE_PRIVATE_KEY_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_private_key.v2"
)
TRIANGLE_PROGRESS_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_progress.v2"
)
LISTENING_ATTEMPT_PATTERN = re.compile(r"listening-attempt-([0-9]{3})")
TRIANGLE_RETRY_LINEAGE_FIELDS = {
    "schema_version",
    "authorization_reason",
    "current_attempt_id",
    "current_attempt_number",
    "previous_attempt_id",
    "previous_attempt_number",
    "base_blind_order_seed",
    "effective_blind_order_seed",
    "effective_blind_order_seed_sha256",
    "base_selection_path",
    "base_selection_sha256",
    "previous_package_path",
    "previous_selection_path",
    "previous_selection_sha256",
    "previous_package_audit_path",
    "previous_package_audit_sha256",
    "failed_snapshot_path",
    "failed_snapshot_id",
    "failed_sealed_responses_sha256",
    "failed_ledger_head_hash",
    "failed_response_ledger_path",
    "failed_response_ledger_sha256",
    "failed_sitting_ledger_path",
    "failed_sitting_ledger_sha256",
    "failed_sitting_ledger_head_hash",
    "failed_unblind_state_path",
    "failed_unblind_state_sha256",
    "failed_unblinded_scores_path",
    "failed_unblinded_scores_sha256",
    "failed_summary_path",
    "failed_summary_sha256",
    "previous_qc_status",
    "previous_retry_required",
    "retry_authorization_path",
    "retry_authorization_sha256",
}
TRIANGLE_RETRY_AUTHORIZATION_FIELDS = {
    "schema_version",
    "created_at",
    "authorization_reason",
    "previous_attempt_id",
    "previous_attempt_number",
    "next_attempt_id",
    "next_attempt_number",
    "base_blind_order_seed",
    "effective_blind_order_seed",
    "effective_blind_order_seed_sha256",
    "failed_package_path",
    "base_selection_path",
    "base_selection_sha256",
    "failed_selection_path",
    "failed_selection_sha256",
    "failed_package_audit_path",
    "failed_package_audit_sha256",
    "failed_render_manifest_path",
    "failed_render_manifest_sha256",
    "failed_private_key_path",
    "failed_private_key_sha256",
    "failed_response_ledger_path",
    "failed_response_ledger_sha256",
    "failed_sitting_ledger_path",
    "failed_sitting_ledger_sha256",
    "failed_sitting_ledger_head_hash",
    "failed_snapshot_path",
    "failed_snapshot_id",
    "failed_sealed_responses_sha256",
    "failed_ledger_head_hash",
    "failed_answered_count",
    "failed_unblind_state_path",
    "failed_unblind_state_sha256",
    "failed_unblinded_scores_path",
    "failed_unblinded_scores_sha256",
    "failed_summary_path",
    "failed_summary_sha256",
    "previous_qc_status",
    "previous_retry_required",
}


def _listening_attempt_number(value: Any) -> int:
    if not isinstance(value, str):
        raise RuntimeError("listening attempt ID must be a string")
    match = LISTENING_ATTEMPT_PATTERN.fullmatch(value)
    if match is None or int(match.group(1)) < 1:
        raise RuntimeError("listening attempt ID must match listening-attempt-NNN")
    return int(match.group(1))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _validated_campaign_inputs(
    config_path: Path,
    schedule_path: Path,
    *,
    expected_config_sha256: str,
    expected_schedule_sha256: str,
) -> tuple[
    dict[str, Any], str, Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Validate every immutable campaign input before any report is written."""
    config_path = config_path.resolve()
    schedule_path = schedule_path.resolve()
    config_sha = file_sha256(config_path)
    if config_sha != expected_config_sha256:
        raise RuntimeError("campaign config hash mismatch")
    config = _read(config_path)
    validate_campaign_config(config)
    validate_frozen_qualification(config, verify_files=True)
    checkpoint_path = Path(config["checkpoint"]["path"]).resolve()
    if file_sha256(checkpoint_path) != str(config["checkpoint"]["sha256"]):
        raise RuntimeError("checkpoint hash mismatch with frozen campaign config")

    input_manifest_path = Path(config["input_manifest"]["path"]).resolve()
    actual_manifest_sha = file_sha256(input_manifest_path)
    if actual_manifest_sha != str(config["input_manifest"]["sha256"]):
        raise RuntimeError(
            "input manifest hash mismatch: frozen campaign config does not match file"
        )
    input_manifest = _read(input_manifest_path)
    inputs = validate_staged_input_manifest(
        input_manifest, manifest_path=input_manifest_path, verify_files=True
    )

    if file_sha256(schedule_path) != expected_schedule_sha256:
        raise RuntimeError("run schedule hash mismatch")
    schedule = read_jsonl(schedule_path)
    if schedule != build_run_schedule(input_manifest, config):
        raise RuntimeError(
            "run schedule is not the deterministic schedule rebuilt from the frozen "
            "campaign config and input manifest"
        )
    return config, config_sha, input_manifest_path, input_manifest, inputs, schedule


def _require_campaign_hash(value: dict[str, Any], label: str, config_sha: str) -> None:
    actual = value.get("campaign_config_sha256")
    if actual != config_sha:
        raise RuntimeError(
            f"{label} campaign config hash mismatch: expected {config_sha}, got {actual!r}"
        )


def _validated_analysis_artifacts(
    analysis_root: Path,
    config_sha: str,
    *,
    input_manifest_sha: str,
    schedule_sha: str,
    campaign_binding_sha: str,
    qualification_result_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reject analysis files copied from, or modified outside, this campaign."""
    analysis_index = _read(analysis_root / "analysis_index.json")
    bootstrap = _read(analysis_root / "bootstrap.json")
    controls = _read(analysis_root / "control_report.json")
    _require_campaign_hash(analysis_index, "analysis index", config_sha)
    _require_campaign_hash(bootstrap, "bootstrap", config_sha)
    _require_campaign_hash(controls, "control report", config_sha)
    control_rows = controls.get("songs")
    if not isinstance(control_rows, list):
        raise RuntimeError("control report songs must be a list")
    for index, row in enumerate(control_rows):
        if not isinstance(row, dict) or row.get("campaign_config_sha256") != config_sha:
            raise RuntimeError(f"control report song row {index} campaign config hash mismatch")
    if analysis_index.get("input_manifest_sha256") != input_manifest_sha:
        raise RuntimeError("analysis index input manifest hash mismatch")
    if analysis_index.get("run_schedule_sha256") != schedule_sha:
        raise RuntimeError("analysis index run schedule hash mismatch")
    if analysis_index.get("campaign_binding_sha256") != campaign_binding_sha:
        raise RuntimeError("analysis index campaign binding hash mismatch")
    if analysis_index.get("qualification_result_sha256") != qualification_result_sha:
        raise RuntimeError("analysis index qualification result hash mismatch")

    artifacts = analysis_index.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("analysis index artifacts must be a path-to-SHA-256 object")
    root = analysis_root.resolve()
    for relative, expected_sha in artifacts.items():
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"analysis index references missing/escaping artifact: {relative}")
        if file_sha256(path) != expected_sha:
            raise RuntimeError(f"analysis artifact hash mismatch: {relative}")
    for required in (
        "bootstrap.json",
        "control_report.json",
        "run_metrics.jsonl",
        "paired_contrasts.jsonl",
    ):
        if required not in artifacts:
            raise RuntimeError(f"analysis index does not bind required artifact: {required}")

    for name in ("run_metrics.jsonl", "paired_contrasts.jsonl"):
        rows = read_jsonl(analysis_root / name)
        for index, row in enumerate(rows):
            if row.get("campaign_config_sha256") != config_sha:
                raise RuntimeError(
                    f"{name} row {index} campaign config hash mismatch"
                )
    return bootstrap, controls, analysis_index


def _validate_listening_known_different_controls(
    controls: Mapping[str, Any],
    *,
    base_selection: Mapping[str, Any],
    base_selection_sha256: str,
    config_sha: str,
) -> None:
    section = controls.get("listening_known_different")
    if not isinstance(section, Mapping):
        raise RuntimeError("control report lacks listening_known_different")
    expected = {
        "selection_sha256": base_selection_sha256,
        "expected_count": 6,
        "actual_count": 6,
        "all_recipe_bound": True,
        "all_source_selectors_bound": True,
        "all_not_identical": True,
    }
    for field, value in expected.items():
        if section.get(field) != value:
            raise RuntimeError(
                f"listening known-different control {field} must equal {value!r}"
            )
    selected = {
        str(trial.get("semantic_id")): trial
        for trial in base_selection.get("trials", [])
        if isinstance(trial, Mapping)
        and trial.get("block") == "known_different_control"
    }
    records = section.get("controls")
    if len(selected) != 6 or not isinstance(records, list) or len(records) != 6:
        raise RuntimeError("listening known-different controls must contain six exact rows")
    expected_row_fields = {
        "campaign_config_sha256",
        "semantic_id",
        "question_id",
        "selection_source_a_sha256",
        "selection_source_b_sha256",
        "selection_recipe_sha256",
        "formal_run_id",
        "formal_source_path",
        "formal_source_sha256",
        "formal_comparator_excerpt_path",
        "formal_comparator_excerpt_sha256",
        "formal_comparator_note_events_sha256",
        "synthetic_excerpt_path",
        "synthetic_excerpt_sha256",
        "synthetic_velocity",
        "not_identical",
    }
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping) or set(record) != expected_row_fields:
            raise RuntimeError(
                f"listening known-different control row {index} fields drifted"
            )
        semantic_id = str(record.get("semantic_id", ""))
        trial = selected.get(semantic_id)
        if trial is None or semantic_id in seen:
            raise RuntimeError("listening known-different semantic IDs drifted")
        seen.add(semantic_id)
        sources = trial.get("sources")
        if not isinstance(sources, Mapping):
            raise RuntimeError("known-different frozen trial lacks source selectors")
        formal = sources.get("a")
        synthetic = sources.get("b")
        recipe = synthetic.get("recipe") if isinstance(synthetic, Mapping) else None
        expected_bindings = {
            "campaign_config_sha256": config_sha,
            "question_id": trial.get("question_id"),
            "selection_source_a_sha256": canonical_sha256(dict(formal))
            if isinstance(formal, Mapping)
            else None,
            "selection_source_b_sha256": canonical_sha256(dict(synthetic))
            if isinstance(synthetic, Mapping)
            else None,
            "selection_recipe_sha256": canonical_sha256(dict(recipe))
            if isinstance(recipe, Mapping)
            else None,
            "synthetic_velocity": 96,
            "not_identical": True,
        }
        for field, value in expected_bindings.items():
            if record.get(field) != value:
                raise RuntimeError(
                    f"listening known-different row {semantic_id} {field} drifted"
                )
        for field in (
            "formal_run_id",
            "formal_comparator_note_events_sha256",
        ):
            value = record.get(field)
            if not isinstance(value, str) or not value:
                raise RuntimeError(
                    f"listening known-different row {semantic_id} lacks {field}"
                )
        if len(str(record["formal_comparator_note_events_sha256"])) != 64:
            raise RuntimeError(
                f"listening known-different row {semantic_id} note-event hash malformed"
            )
        for label, path_field, hash_field in (
            ("formal source", "formal_source_path", "formal_source_sha256"),
            (
                "formal comparator",
                "formal_comparator_excerpt_path",
                "formal_comparator_excerpt_sha256",
            ),
            (
                "synthetic excerpt",
                "synthetic_excerpt_path",
                "synthetic_excerpt_sha256",
            ),
        ):
            _hash_bound_file(
                record.get(path_field),
                record.get(hash_field),
                label=f"known-different {semantic_id} {label}",
            )
    if seen != set(selected):
        raise RuntimeError("listening known-different controls omit frozen semantic IDs")


def _hash_bound_file(path_value: Any, sha_value: Any, *, label: str) -> Path:
    if not isinstance(path_value, str) or not isinstance(sha_value, str):
        raise RuntimeError(f"{label} lacks an immutable path/hash binding")
    raw = Path(path_value)
    path = raw.resolve()
    if not raw.is_absolute() or raw.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing, relative, or a symlink")
    if file_sha256(path) != sha_value:
        raise RuntimeError(f"{label} hash mismatch")
    return path


def _validate_triangle_attempt_lineage(
    selection: Mapping[str, Any],
    *,
    selection_path: Path,
    current_package: Path,
) -> list[dict[str, Any]]:
    """Validate and disclose every sealed QC-failed predecessor attempt."""

    selection_path = selection_path.resolve()
    current_package = current_package.resolve()
    if not selection_path.is_file() or _read(selection_path) != dict(selection):
        raise RuntimeError("current triangle selection path/content drifted")
    cursor = dict(selection)
    cursor_path = selection_path
    cursor_package = current_package
    reverse_history: list[dict[str, Any]] = [
        {
            "listening_attempt_id": cursor.get("listening_attempt_id"),
            "listening_attempt_number": cursor.get("listening_attempt_number"),
            "role": "current_attempt",
            "package_path": str(cursor_package),
            "selection_path": str(cursor_path),
            "selection_sha256": file_sha256(cursor_path),
        }
    ]
    seen_attempts: set[str] = set()
    seen_packages: set[Path] = {cursor_package}
    base_path_from_chain: Path | None = None
    base_sha_from_chain: str | None = None

    while True:
        attempt_id = cursor.get("listening_attempt_id")
        attempt_number = _listening_attempt_number(attempt_id)
        if cursor.get("listening_attempt_number") != attempt_number:
            raise RuntimeError("triangle selection attempt ID/number mismatch")
        if attempt_id in seen_attempts:
            raise RuntimeError("triangle retry lineage contains an attempt cycle")
        seen_attempts.add(str(attempt_id))
        lineage = cursor.get("retry_lineage")
        lineage_sha = cursor.get("retry_lineage_sha256")
        if attempt_number == 1:
            if lineage is not None or lineage_sha is not None:
                raise RuntimeError("base listening attempt must not contain retry lineage")
            if cursor.get("retry_reblind_after_formal_without_semantic_change") not in {
                None,
                False,
            }:
                raise RuntimeError("base listening attempt is incorrectly marked reblinded")
            if base_path_from_chain is not None and (
                cursor_path != base_path_from_chain
                or file_sha256(cursor_path) != base_sha_from_chain
            ):
                raise RuntimeError("retry chain does not terminate at its C5 base selection")
            break
        if not isinstance(lineage, Mapping) or set(lineage) != TRIANGLE_RETRY_LINEAGE_FIELDS:
            raise RuntimeError("retry selection lineage schema/fields drifted")
        if (
            lineage.get("schema_version") != TRIANGLE_RETRY_LINEAGE_SCHEMA_VERSION
            or lineage_sha != canonical_sha256(dict(lineage))
            or lineage.get("authorization_reason") != "qc_failure"
            or lineage.get("current_attempt_id") != attempt_id
            or lineage.get("current_attempt_number") != attempt_number
            or lineage.get("previous_attempt_number") != attempt_number - 1
            or _listening_attempt_number(lineage.get("previous_attempt_id"))
            != attempt_number - 1
            or lineage.get("previous_qc_status") != "fail"
            or lineage.get("previous_retry_required") is not True
        ):
            raise RuntimeError("retry selection lineage attempt/QC contract drifted")
        if (
            cursor.get("base_blind_order_seed") != lineage.get("base_blind_order_seed")
            or cursor.get("effective_blind_order_seed")
            != lineage.get("effective_blind_order_seed")
            or cursor.get("effective_blind_order_seed_sha256")
            != lineage.get("effective_blind_order_seed_sha256")
        ):
            raise RuntimeError("retry selection seed derivation differs from lineage")

        base_path = _hash_bound_file(
            lineage.get("base_selection_path"),
            lineage.get("base_selection_sha256"),
            label="retry base selection",
        )
        base_sha = str(lineage["base_selection_sha256"])
        if base_path_from_chain is None:
            base_path_from_chain, base_sha_from_chain = base_path, base_sha
        elif base_path != base_path_from_chain or base_sha != base_sha_from_chain:
            raise RuntimeError("retry attempts disagree on the C5 base selection")

        previous_package = Path(str(lineage.get("previous_package_path", ""))).resolve()
        if (
            not Path(str(lineage.get("previous_package_path", ""))).is_absolute()
            or not previous_package.is_dir()
            or previous_package in seen_packages
            or previous_package == cursor_package
        ):
            raise RuntimeError("retry predecessor package is missing/reused")
        seen_packages.add(previous_package)
        previous_selection_path = _hash_bound_file(
            lineage.get("previous_selection_path"),
            lineage.get("previous_selection_sha256"),
            label="retry predecessor selection",
        )
        previous_selection = _read(previous_selection_path)
        if (
            previous_selection.get("listening_attempt_id")
            != lineage.get("previous_attempt_id")
            or previous_selection.get("listening_attempt_number")
            != lineage.get("previous_attempt_number")
        ):
            raise RuntimeError("retry predecessor selection attempt drifted")

        audit_path = _hash_bound_file(
            lineage.get("previous_package_audit_path"),
            lineage.get("previous_package_audit_sha256"),
            label="retry predecessor package audit",
        )
        ledger_path = _hash_bound_file(
            lineage.get("failed_response_ledger_path"),
            lineage.get("failed_response_ledger_sha256"),
            label="retry predecessor response ledger",
        )
        sitting_ledger_path = _hash_bound_file(
            lineage.get("failed_sitting_ledger_path"),
            lineage.get("failed_sitting_ledger_sha256"),
            label="retry predecessor sitting ledger",
        )
        unblind_state_path = _hash_bound_file(
            lineage.get("failed_unblind_state_path"),
            lineage.get("failed_unblind_state_sha256"),
            label="retry predecessor unblind state",
        )
        unblinded_path = _hash_bound_file(
            lineage.get("failed_unblinded_scores_path"),
            lineage.get("failed_unblinded_scores_sha256"),
            label="retry predecessor unblinded scores",
        )
        summary_path = _hash_bound_file(
            lineage.get("failed_summary_path"),
            lineage.get("failed_summary_sha256"),
            label="retry predecessor summary",
        )
        snapshot_path = Path(str(lineage.get("failed_snapshot_path", ""))).resolve()
        if (
            snapshot_path.name != lineage.get("failed_snapshot_id")
            or snapshot_path.parent != previous_package / "snapshots"
            or not snapshot_path.is_dir()
            or audit_path != previous_package / "package_audit.json"
            or ledger_path != previous_package / "blind" / "response_ledger.jsonl"
            or sitting_ledger_path
            != previous_package / "blind" / "sitting_ledger.jsonl"
            or unblind_state_path != previous_package / "unblind_state.json"
            or unblinded_path != snapshot_path / "partial_unblinded_scores.json"
            or summary_path != snapshot_path / "partial_discrimination_summary.json"
            or previous_selection_path
            != Path(str(_read(previous_package / "render_manifest.json").get("selection_path", ""))).resolve()
        ):
            raise RuntimeError("retry predecessor artifact layout/path binding drifted")
        sealed = validate_triangle_snapshot(previous_package, snapshot_path)
        _unblinded, summary = validate_triangle_unblinded_summary(
            previous_package, snapshot_path
        )
        if (
            sealed.get("listening_attempt_id") != lineage.get("previous_attempt_id")
            or sealed.get("answered_count") != TRIANGLE_TRIAL_COUNT
            or sealed.get("collection_status") != "full"
            or sealed.get("ledger_head_hash") != lineage.get("failed_ledger_head_hash")
            or sealed.get("sitting_ledger_head_hash")
            != lineage.get("failed_sitting_ledger_head_hash")
            or file_sha256(snapshot_path / "sealed_responses.json")
            != lineage.get("failed_sealed_responses_sha256")
            or summary.get("listening_attempt_id") != lineage.get("previous_attempt_id")
            or summary.get("qc_status") != "fail"
            or summary.get("retry_required") is not True
            or summary.get("attempt_disposition")
            != "sealed_qc_failure_retry_required"
        ):
            raise RuntimeError("retry predecessor is not a sealed full QC failure")
        predecessor_audit = _read(audit_path)
        if (
            predecessor_audit.get("listening_attempt_id")
            != lineage.get("previous_attempt_id")
            or predecessor_audit.get("valid") is not True
            or predecessor_audit.get("accepted_final") is not True
        ):
            raise RuntimeError("retry predecessor package audit is not accepted/valid")

        authorization_path = _hash_bound_file(
            lineage.get("retry_authorization_path"),
            lineage.get("retry_authorization_sha256"),
            label="retry authorization",
        )
        authorization_sidecar = authorization_path.with_name(
            authorization_path.name + ".sha256"
        )
        if not authorization_sidecar.is_file() or authorization_sidecar.read_text(
            encoding="ascii"
        ).strip().split() != [
            file_sha256(authorization_path),
            authorization_path.name,
        ]:
            raise RuntimeError("retry authorization checksum sidecar drifted")
        authorization = _read(authorization_path)
        if (
            set(authorization) != TRIANGLE_RETRY_AUTHORIZATION_FIELDS
            or authorization.get("schema_version")
            != TRIANGLE_RETRY_AUTHORIZATION_SCHEMA_VERSION
            or authorization.get("authorization_reason") != "qc_failure"
            or authorization.get("previous_attempt_id")
            != lineage.get("previous_attempt_id")
            or authorization.get("previous_attempt_number")
            != lineage.get("previous_attempt_number")
            or authorization.get("next_attempt_id") != attempt_id
            or authorization.get("next_attempt_number") != attempt_number
            or authorization.get("failed_answered_count") != TRIANGLE_TRIAL_COUNT
            or authorization.get("previous_qc_status") != "fail"
            or authorization.get("previous_retry_required") is not True
        ):
            raise RuntimeError("retry authorization schema/attempt/QC contract drifted")
        authorization_to_lineage = {
            "base_blind_order_seed": "base_blind_order_seed",
            "effective_blind_order_seed": "effective_blind_order_seed",
            "effective_blind_order_seed_sha256": "effective_blind_order_seed_sha256",
            "base_selection_path": "base_selection_path",
            "base_selection_sha256": "base_selection_sha256",
            "failed_package_path": "previous_package_path",
            "failed_selection_path": "previous_selection_path",
            "failed_selection_sha256": "previous_selection_sha256",
            "failed_package_audit_path": "previous_package_audit_path",
            "failed_package_audit_sha256": "previous_package_audit_sha256",
            "failed_response_ledger_path": "failed_response_ledger_path",
            "failed_response_ledger_sha256": "failed_response_ledger_sha256",
            "failed_sitting_ledger_path": "failed_sitting_ledger_path",
            "failed_sitting_ledger_sha256": "failed_sitting_ledger_sha256",
            "failed_sitting_ledger_head_hash": "failed_sitting_ledger_head_hash",
            "failed_snapshot_path": "failed_snapshot_path",
            "failed_snapshot_id": "failed_snapshot_id",
            "failed_sealed_responses_sha256": "failed_sealed_responses_sha256",
            "failed_ledger_head_hash": "failed_ledger_head_hash",
            "failed_unblind_state_path": "failed_unblind_state_path",
            "failed_unblind_state_sha256": "failed_unblind_state_sha256",
            "failed_unblinded_scores_path": "failed_unblinded_scores_path",
            "failed_unblinded_scores_sha256": "failed_unblinded_scores_sha256",
            "failed_summary_path": "failed_summary_path",
            "failed_summary_sha256": "failed_summary_sha256",
        }
        if any(
            authorization.get(auth_field) != lineage.get(lineage_field)
            for auth_field, lineage_field in authorization_to_lineage.items()
        ):
            raise RuntimeError("retry authorization differs from retry lineage")
        for field, relative in (
            ("failed_render_manifest_path", "render_manifest.json"),
            ("failed_private_key_path", "private/private_key.json"),
        ):
            path = _hash_bound_file(
                authorization.get(field),
                authorization.get(field.replace("_path", "_sha256")),
                label=f"retry authorization {field}",
            )
            if path != previous_package / relative:
                raise RuntimeError(f"retry authorization {field} layout drifted")

        reverse_history.append(
            {
                "listening_attempt_id": lineage["previous_attempt_id"],
                "listening_attempt_number": lineage["previous_attempt_number"],
                "role": "sealed_qc_failure",
                "package_path": str(previous_package),
                "selection_path": str(previous_selection_path),
                "selection_sha256": file_sha256(previous_selection_path),
                "snapshot_path": str(snapshot_path),
                "snapshot_sha256": file_sha256(
                    snapshot_path / "sealed_responses.json"
                ),
                "qc_status": "fail",
                "attempt_disposition": "sealed_qc_failure_retry_required",
                "retry_required": True,
                "retry_authorization_path": str(authorization_path),
                "retry_authorization_sha256": file_sha256(authorization_path),
            }
        )
        cursor = previous_selection
        cursor_path = previous_selection_path
        cursor_package = previous_package

    history = list(reversed(reverse_history))
    if [row["listening_attempt_number"] for row in history] != list(
        range(1, len(history) + 1)
    ):
        raise RuntimeError("retry lineage does not contain every consecutive attempt")
    return history


def _validate_audit_schedule(
    audit: dict[str, Any], schedule: list[dict[str, Any]]
) -> None:
    expected = {str(row["run_id"]): str(row["pipeline"]) for row in schedule}
    actual_rows = audit.get("runs")
    if not isinstance(actual_rows, list):
        raise RuntimeError("campaign audit is missing its run identities")
    actual_ids = [str(row.get("run_id")) for row in actual_rows]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected):
        raise RuntimeError("campaign audit run identities do not match the frozen schedule")
    for row in actual_rows:
        pipeline = row.get("pipeline")
        if pipeline is not None and str(pipeline) != expected[str(row["run_id"])]:
            raise RuntimeError("campaign audit pipeline does not match the frozen schedule")


def _formal_listening_sources(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    trials = selection.get("trials")
    if not isinstance(trials, list):
        raise RuntimeError("listening selection trials must be a list")
    for trial in trials:
        if not isinstance(trial, Mapping) or not isinstance(trial.get("sources"), Mapping):
            raise RuntimeError("listening selection trial lacks source selectors")
        for source in trial["sources"].values():
            if isinstance(source, Mapping) and source.get("kind") == "formal":
                frozen = dict(source)
                unique[canonical_sha256(frozen)] = frozen
    if not unique:
        raise RuntimeError("listening selection contains no formal generated sources")
    return unique


def _reverify_campaign_audit(
    campaign_root: Path,
    *,
    audit: Mapping[str, Any],
    schedule: list[dict[str, Any]],
    campaign_binding: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    """Re-open all formal attempts and the listening-source readiness contract."""

    expected_ids = {str(row["run_id"]) for row in schedule}
    runs_root = campaign_root / "runs"
    actual_ids = (
        {path.name for path in runs_root.iterdir() if path.is_dir()}
        if runs_root.is_dir()
        else set()
    )
    if actual_ids != expected_ids or audit.get("extra_run_ids") != []:
        raise RuntimeError("formal campaign contains missing or extra run directories")
    audit_rows = audit.get("runs")
    if not isinstance(audit_rows, list):
        raise RuntimeError("campaign audit runs must be a list")
    audit_by_id = {str(row.get("run_id")): row for row in audit_rows}
    retried = 0
    operational_invalid = 0
    source_empty = 0
    verified: dict[str, tuple[Path, dict[str, Any], set[Path]]] = {}
    source_empty_by_run: dict[str, bool] = {}
    for schedule_row in schedule:
        run_id = str(schedule_row["run_id"])
        attempt, verdict, indexed = verify_attempt_verdict(
            runs_root / run_id,
            schedule_row,
            campaign_binding,
            require_content_valid=True,
        )
        verified[run_id] = (attempt, verdict, indexed)
        attempts = len(list((runs_root / run_id).glob("attempt-*")))
        retried += int(attempts > 1)
        operational = bool(verdict["operational_valid"])
        operational_invalid += int(not operational)
        gate = verdict.get("validity", {}).get("driver_artifact_gate", {})
        empty = bool(gate.get("source_empty", False))
        source_empty_by_run[run_id] = empty
        source_empty += int(empty)
        audited = audit_by_id.get(run_id)
        if not isinstance(audited, Mapping):
            raise RuntimeError(f"campaign audit omits run {run_id}")
        expected_row = {
            "pipeline": schedule_row["pipeline"],
            "status": "valid",
            "operational_valid": operational,
            "attempts": attempts,
            "source_empty": empty,
        }
        for field, expected in expected_row.items():
            if audited.get(field) != expected:
                raise RuntimeError(f"campaign audit run {run_id} {field} drifted")
    expected_summary = {
        "expected": len(schedule),
        "present": len(schedule),
        "content_valid": len(schedule),
        "missing": 0,
        "invalid": 0,
        "retried": retried,
        "operational_invalid": operational_invalid,
        "source_empty": source_empty,
        "extra_run_ids": [],
    }
    for field, expected in expected_summary.items():
        if audit.get(field) != expected:
            raise RuntimeError(
                f"campaign audit {field} mismatch after exact run reverification"
            )

    expected_sources = _formal_listening_sources(selection)
    readiness = audit.get("listening_source_readiness")
    if not isinstance(readiness, Mapping):
        raise RuntimeError("campaign audit lacks listening-source readiness")
    if (
        readiness.get("schema_version")
        != "streammuse.melody_robustness.listening_source_readiness.v1"
        or readiness.get("selection_schema_version") != selection.get("schema_version")
        or readiness.get("expected_unique_sources") != len(expected_sources)
        or readiness.get("ready_sources") != len(expected_sources)
        or readiness.get("not_ready_sources") != 0
        or readiness.get("ready") is not True
    ):
        raise RuntimeError("listening-source readiness summary is not fully ready")
    readiness_rows = readiness.get("sources")
    if not isinstance(readiness_rows, list) or len(readiness_rows) != len(
        expected_sources
    ):
        raise RuntimeError("listening-source readiness row count drifted")
    seen: set[str] = set()
    for ready_row in readiness_rows:
        if not isinstance(ready_row, Mapping) or not isinstance(
            ready_row.get("selector"), Mapping
        ):
            raise RuntimeError("listening-source readiness row is invalid")
        selector = dict(ready_row["selector"])
        selector_key = canonical_sha256(selector)
        if (
            selector_key not in expected_sources
            or selector_key in seen
            or ready_row.get("ready") is not True
        ):
            raise RuntimeError("listening-source readiness selector set drifted")
        seen.add(selector_key)
        matches = [
            row
            for row in schedule
            if row.get("pipeline") == selector.get("formal_pipeline")
            and row.get("song") == selector.get("song")
            and row.get("condition") == selector.get("condition")
            and row.get("perturb_seed") == selector.get("perturb_seed")
            and row.get("sample_seed") == selector.get("sample_seed")
        ]
        if len(matches) != 1 or ready_row.get("run_id") != matches[0]["run_id"]:
            raise RuntimeError("listening-source readiness selector/run mapping drifted")
        attempt, verdict, indexed = verified[str(matches[0]["run_id"])]
        if (
            ready_row.get("attempt_id") != attempt.name
            or ready_row.get("operational_valid")
            is not bool(verdict["operational_valid"])
        ):
            raise RuntimeError("listening-source readiness attempt provenance drifted")
        artifact = ready_row.get("artifact")
        if not isinstance(artifact, Mapping):
            raise RuntimeError("listening-source readiness artifact record is invalid")
        gate_path = attempt / "rt_artifact_gate.json"
        if gate_path not in indexed:
            raise RuntimeError("listening-source readiness lacks an indexed RT artifact gate")
        artifact_gate = _read(gate_path)
        artifact_label = (
            "theoretical_model_midi"
            if selector.get("source_artifact") == "theoretical_model"
            else "combined_midi"
        )
        required = artifact_gate.get("required_artifacts", {}).get(artifact_label)
        if not isinstance(required, Mapping) or dict(required) != dict(artifact):
            raise RuntimeError(
                "listening-source readiness artifact differs from the RT artifact gate"
            )
        path = (attempt / str(artifact.get("path", ""))).resolve()
        if (
            not path.is_relative_to(attempt)
            or path not in indexed
            or not path.is_file()
            or path.stat().st_size != artifact.get("size")
            or file_sha256(path) != artifact.get("sha256")
        ):
            raise RuntimeError("listening-source readiness artifact drifted")
        if ready_row.get("source_empty") is not source_empty_by_run[
            str(matches[0]["run_id"])
        ]:
            raise RuntimeError("listening-source readiness empty-source provenance drifted")
    if seen != set(expected_sources):
        raise RuntimeError("listening-source readiness does not cover the frozen selection")


def _validated_campaign_binding(
    campaign_root: Path,
    *,
    config_path: Path,
    schedule_path: Path,
    config: dict[str, Any],
    config_sha: str,
    schedule_sha: str,
) -> tuple[dict[str, Any], str]:
    path = campaign_root / "campaign_binding.json"
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
        "qualification_result_sha256": str(config["qualification_result"]["sha256"]),
    }
    if not path.is_file() or _read(path) != expected:
        raise RuntimeError("campaign output root binding does not match config/schedule")
    return expected, file_sha256(path)


def _legacy_listening_completion(
    listening_root: Path,
    *,
    config_sha: str,
    schedule_sha: str,
    campaign_binding_sha: str,
    selection_sha: str,
    qualification_result_sha: str,
) -> tuple[dict[str, Any], bool, bool]:
    """Validate the historical 24-clip quality-rating workflow."""
    audit_path = listening_root / "package_audit.json"
    audit = _read(audit_path)
    expected_links = {
        "campaign_config_sha256": config_sha,
        "run_schedule_sha256": schedule_sha,
        "campaign_binding_sha256": campaign_binding_sha,
        "selection_sha256": selection_sha,
        "qualification_result_sha256": qualification_result_sha,
    }
    for field, expected in expected_links.items():
        if audit.get(field) != expected:
            raise RuntimeError(f"listening package audit {field} mismatch")
    key_path = listening_root / "private_key.json"
    key_sha = file_sha256(key_path)
    if audit.get("private_key_sha256") != key_sha:
        raise RuntimeError("listening package audit private key hash mismatch")
    if audit.get("render_manifest_sha256") != file_sha256(
        listening_root / "render_manifest.json"
    ):
        raise RuntimeError("listening package audit render manifest hash mismatch")
    key = _read(key_path)

    sealed_path = listening_root / "sealed_scores.json"
    sealed_valid = False
    if sealed_path.is_file():
        sealed = _read(sealed_path)
        score_path = Path(str(sealed.get("scores_path", ""))).resolve()
        expected_score_path = (listening_root / "blind" / "scores.csv").resolve()
        if score_path != expected_score_path or not score_path.is_relative_to(listening_root):
            raise RuntimeError("sealed listening score path mismatch")
        sealed_links = {
            "campaign_config_sha256": config_sha,
            "selection_sha256": selection_sha,
            "run_schedule_sha256": schedule_sha,
            "campaign_binding_sha256": campaign_binding_sha,
            "qualification_result_sha256": qualification_result_sha,
            "private_key_sha256": key_sha,
            "package_audit_sha256": file_sha256(audit_path),
            "scores_sha256": file_sha256(score_path),
        }
        for field, expected in sealed_links.items():
            if sealed.get(field) != expected:
                raise RuntimeError(f"sealed listening scores {field} mismatch")
        score_rows = list(csv.DictReader(score_path.open("r", encoding="utf-8")))
        expected_ids = {str(row["sample_id"]) for row in key.get("clips", [])}
        score_ids = [str(row.get("sample_id")) for row in score_rows]
        if len(score_ids) != 24 or len(set(score_ids)) != 24 or set(score_ids) != expected_ids:
            raise RuntimeError("sealed listening score rows do not match the private key")
        try:
            scores = [int(row["overall_quality_1_to_5"]) for row in score_rows]
        except Exception as exc:
            raise RuntimeError("sealed listening scores contain missing/invalid ratings") from exc
        if any(score not in {1, 2, 3, 4, 5} for score in scores):
            raise RuntimeError("sealed listening scores contain ratings outside 1..5")
        sealed_valid = sealed.get("sealed_before_unblinding") is True

    unblinded_path = listening_root / "unblinded_scores.json"
    unblinded_valid = False
    if unblinded_path.is_file():
        if not sealed_valid:
            raise RuntimeError("unblinded scores exist without valid sealed scores")
        unblinded = _read(unblinded_path)
        unblinded_links = {
            "campaign_config_sha256": config_sha,
            "selection_sha256": selection_sha,
            "run_schedule_sha256": schedule_sha,
            "campaign_binding_sha256": campaign_binding_sha,
            "qualification_result_sha256": qualification_result_sha,
            "private_key_sha256": key_sha,
            "sealed_scores_sha256": file_sha256(sealed_path),
        }
        for field, expected in unblinded_links.items():
            if unblinded.get(field) != expected:
                raise RuntimeError(f"unblinded listening scores {field} mismatch")
        expected_ids = {str(row["sample_id"]) for row in key.get("clips", [])}
        rows = unblinded.get("rows")
        actual_ids = (
            [str(row.get("sample_id")) for row in rows]
            if isinstance(rows, list) else []
        )
        if len(actual_ids) != 24 or len(set(actual_ids)) != 24 or set(actual_ids) != expected_ids:
            raise RuntimeError("unblinded listening rows do not match the private key")
        try:
            unblinded_scores = [int(row["overall_quality_1_to_5"]) for row in rows]
        except Exception as exc:
            raise RuntimeError("unblinded listening rows contain invalid ratings") from exc
        if any(score not in {1, 2, 3, 4, 5} for score in unblinded_scores):
            raise RuntimeError("unblinded listening ratings fall outside 1..5")
        unblinded_valid = True
    return audit, sealed_valid, unblinded_valid


def _require_exact_links(
    value: Mapping[str, Any],
    expected_links: Mapping[str, str],
    *,
    label: str,
) -> None:
    for field, expected in expected_links.items():
        if value.get(field) != expected:
            raise RuntimeError(
                f"{label} {field} mismatch: expected {expected}, "
                f"got {value.get(field)!r}"
            )


def _recompute_triangle_package_audit(listening_root: Path) -> dict[str, Any]:
    """Load the sibling renderer auditor and recompute every package invariant."""

    module_name = "_streammuse_report_triangle_package_auditor"
    module = sys.modules.get(module_name)
    if module is None:
        script_path = Path(__file__).resolve().with_name(
            "prepare_robustness_listening.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load the triangle package auditor")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    auditor = getattr(module, "audit_triangle_package_dir", None)
    if not callable(auditor):
        raise RuntimeError("triangle package auditor is unavailable")
    result = auditor(listening_root, require_wav=True)
    if not isinstance(result, dict):
        raise RuntimeError("triangle package auditor returned a non-object")
    return result


def _verify_relative_artifact(
    root: Path,
    relative: Any,
    expected_sha: Any,
    *,
    label: str,
    required: bool,
) -> None:
    if relative is None and expected_sha is None and not required:
        return
    if not isinstance(relative, str) or not relative or not isinstance(expected_sha, str):
        raise RuntimeError(f"{label} lacks a hash-pinned path")
    raw = Path(relative)
    path = (root / raw).resolve()
    if raw.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError(f"{label} is missing or escapes the listening package")
    if file_sha256(path) != expected_sha:
        raise RuntimeError(f"{label} hash mismatch")


def _validate_triangle_render_artifacts(
    listening_root: Path,
    render: Mapping[str, Any],
    *,
    require_wav: bool,
) -> None:
    groups = (
        ("trial", render.get("trials"), TRIANGLE_TRIAL_COUNT),
        ("practice", render.get("practice_trials"), TRIANGLE_PRACTICE_COUNT),
    )
    presentation_count = 0
    for group_name, rows, expected_count in groups:
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise RuntimeError(
                f"triangle render must contain exactly {expected_count} {group_name} rows"
            )
        for row_index, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise RuntimeError(f"triangle {group_name} render row is not an object")
            presentations = row.get("presentations")
            if not isinstance(presentations, list) or len(presentations) != 3:
                raise RuntimeError(
                    f"triangle {group_name} row {row_index} must contain three presentations"
                )
            for position, presentation in enumerate(presentations, start=1):
                if not isinstance(presentation, Mapping) or presentation.get("position") != position:
                    raise RuntimeError(
                        f"triangle {group_name} row {row_index} presentation order drifted"
                    )
                _verify_relative_artifact(
                    listening_root,
                    presentation.get("midi"),
                    presentation.get("midi_sha256"),
                    label=f"triangle {group_name} {row_index}/{position} MIDI",
                    required=True,
                )
                _verify_relative_artifact(
                    listening_root,
                    presentation.get("wav"),
                    presentation.get("wav_sha256"),
                    label=f"triangle {group_name} {row_index}/{position} WAV",
                    required=require_wav,
                )
                presentation_count += 1
    if presentation_count != TRIANGLE_PRESENTATION_COUNT + 3 * TRIANGLE_PRACTICE_COUNT:
        raise RuntimeError("triangle render presentation count drifted")


def _triangle_identity_counts(
    key: Mapping[str, Any], render: Mapping[str, Any]
) -> dict[str, int]:
    trials = key.get("trials")
    if not isinstance(trials, list):
        raise RuntimeError("triangle private key trials must be a list")
    counts = {
        "trial_count": len(trials),
        "raw_token_payload_comparable": 0,
        "raw_token_payload_identical": 0,
        "output_event_payload_comparable": 0,
        "output_event_payload_identical": 0,
        "theoretical_midi_comparable": 0,
        "theoretical_midi_identical": 0,
        "canonical_midi_comparable": 0,
        "canonical_midi_identical": 0,
        "canonical_wav_comparable": 0,
        "canonical_wav_identical": 0,
        "rendered_midi_comparable": 0,
        "rendered_midi_identical": 0,
        "rendered_wav_comparable": 0,
        "rendered_wav_identical": 0,
        "objective_identity": 0,
        "coverage_driven": 0,
        "source_empty_either": 0,
        "operational_invalid_either": 0,
    }
    rendered_rows = render.get("trials")
    if not isinstance(rendered_rows, list) or len(rendered_rows) != len(trials):
        raise RuntimeError("triangle render/private trial counts differ")
    render_by_id = {
        str(row.get("question_id")): row
        for row in rendered_rows
        if isinstance(row, Mapping)
    }
    for row in trials:
        if not isinstance(row, Mapping):
            raise RuntimeError("triangle private key trial is not an object")
        source_a = row.get("source_a")
        source_b = row.get("source_b")
        if not isinstance(source_a, Mapping) or not isinstance(source_b, Mapping):
            raise RuntimeError("triangle private key trial lacks source provenance")
        for source in (source_a, source_b):
            if source.get("kind") == "formal" and (
                not isinstance(source.get("raw_token_payload_sha256"), str)
                or not isinstance(source.get("output_event_payload_sha256"), str)
            ):
                raise RuntimeError(
                    "formal listening source lacks raw-token/output-event provenance"
                )
        for prefix, field in (
            ("raw_token_payload", "raw_token_payload_sha256"),
            ("output_event_payload", "output_event_payload_sha256"),
            ("theoretical_midi", "source_sha256"),
            ("canonical_midi", "excerpt_midi_sha256"),
            ("canonical_wav", "canonical_wav_sha256"),
        ):
            left, right = source_a.get(field), source_b.get(field)
            if isinstance(left, str) and isinstance(right, str):
                counts[f"{prefix}_comparable"] += 1
                counts[f"{prefix}_identical"] += int(left == right)
        rendered = render_by_id.get(str(row.get("question_id")))
        presentations = rendered.get("presentations") if isinstance(rendered, Mapping) else None
        pattern = str(row.get("presentation_pattern", ""))
        if not isinstance(presentations, list) or len(presentations) != 3 or len(pattern) != 3:
            raise RuntimeError("triangle rendered identity row lacks three mapped presentations")
        by_label: dict[str, list[Mapping[str, Any]]] = {"A": [], "B": []}
        for label, presentation in zip(pattern, presentations, strict=True):
            if label not in by_label or not isinstance(presentation, Mapping):
                raise RuntimeError("triangle presentation pattern/render mapping is invalid")
            by_label[label].append(presentation)
        for label_rows in by_label.values():
            for field in ("midi_sha256", "wav_sha256"):
                hashes = [presentation.get(field) for presentation in label_rows]
                if len(hashes) > 1 and len(set(hashes)) != 1:
                    raise RuntimeError("triangle duplicate presentations are not literal copies")
        for prefix, field in (
            ("rendered_midi", "midi_sha256"),
            ("rendered_wav", "wav_sha256"),
        ):
            if by_label["B"]:
                left = by_label["A"][0].get(field)
                right = by_label["B"][0].get(field)
            else:
                hashes = [presentation.get(field) for presentation in by_label["A"]]
                left = hashes[0] if hashes else None
                right = hashes[-1] if hashes else None
            if isinstance(left, str) and isinstance(right, str):
                counts[f"{prefix}_comparable"] += 1
                counts[f"{prefix}_identical"] += int(left == right)
        counts["objective_identity"] += int(bool(row.get("objective_identity")))
        counts["coverage_driven"] += int(bool(row.get("coverage_driven")))
        counts["source_empty_either"] += int(
            bool(source_a.get("source_empty")) or bool(source_b.get("source_empty"))
        )
        counts["operational_invalid_either"] += int(
            source_a.get("operational_valid") is False
            or source_b.get("operational_valid") is False
        )
    return counts


def _validate_generated_acc_export(
    listening_root: Path,
    private_key: Mapping[str, Any],
    *,
    require_snapshot_binding: bool = True,
) -> dict[str, Any]:
    export_root = listening_root / "generated_acc_after_unblind"
    json_path = export_root / "generated_acc_index.json"
    csv_path = export_root / "generated_acc_index.csv"
    if not json_path.is_file() or not csv_path.is_file():
        return {"valid": False, "row_count": 0, "reason": "missing"}
    index = _read(json_path)
    snapshot_links: dict[str, Any] = {}
    if require_snapshot_binding:
        expected_attempt_id = private_key.get("listening_attempt_id")
        _listening_attempt_number(expected_attempt_id)
        if index.get("schema_version") != (
            "streammuse.melody_robustness.generated_acc_export_index.v2"
        ):
            raise RuntimeError("generated acc index schema drifted")
        if index.get("listening_attempt_id") != expected_attempt_id:
            raise RuntimeError("generated acc index listening attempt drifted")
        key_path = listening_root / "private" / "private_key.json"
        unblind_state_path = listening_root / "unblind_state.json"
        if (
            index.get("selection_sha256") != private_key.get("selection_sha256")
            or index.get("private_key_sha256") != file_sha256(key_path)
            or not unblind_state_path.is_file()
            or index.get("unblind_state_sha256") != file_sha256(unblind_state_path)
        ):
            raise RuntimeError("generated acc index upstream bindings drifted")
        authorizing_id = index.get("export_authorizing_snapshot_id")
        if not isinstance(authorizing_id, str):
            raise RuntimeError("generated acc index lacks an authorizing snapshot")
        authorizing = listening_root / "snapshots" / authorizing_id
        sealed = validate_triangle_snapshot(listening_root, authorizing)
        _unblinded, summary = validate_triangle_unblinded_summary(
            listening_root, authorizing
        )
        snapshot_links = {
            "sealed_responses_sha256": file_sha256(
                authorizing / "sealed_responses.json"
            ),
            "unblinded_scores_sha256": file_sha256(
                authorizing / "partial_unblinded_scores.json"
            ),
            "discrimination_summary_sha256": file_sha256(
                authorizing / "partial_discrimination_summary.json"
            ),
            "answered_count": sealed["answered_count"],
            "collection_status": summary["collection_status"],
        }
        for field, expected in snapshot_links.items():
            if index.get(field) != expected:
                raise RuntimeError(
                    f"generated acc index authorizing snapshot {field} drifted"
                )
        if Path(str(index.get("csv_path", ""))).resolve() != csv_path.resolve():
            raise RuntimeError("generated acc index CSV path drifted")
        if index.get("csv_sha256") != file_sha256(csv_path):
            raise RuntimeError("generated acc index CSV hash drifted")
    sources_by_identity: dict[str, Mapping[str, Any]] = {}
    for trial in private_key.get("trials", []):
        if not isinstance(trial, Mapping):
            raise RuntimeError("triangle private key contains an invalid trial row")
        for side in ("source_a", "source_b"):
            source = trial.get(side)
            if isinstance(source, Mapping) and source.get("kind") == "formal":
                identity = ":".join(
                    str(source.get(field))
                    for field in ("song", "condition", "perturb_seed", "sample_seed")
                )
                previous = sources_by_identity.setdefault(identity, source)
                if dict(previous) != dict(source):
                    raise RuntimeError(
                        "one semantic generated-acc identity has conflicting provenance"
                    )
    if not sources_by_identity:
        raise RuntimeError("triangle private key contains no formal generated sources")

    expected_rows: list[dict[str, Any]] = []
    expected_midi_paths: set[Path] = set()
    expected_wav_paths: set[Path] = set()
    for index_number, (identity, source) in enumerate(
        sorted(sources_by_identity.items()), start=1
    ):
        song = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source.get("song")))
        perturb_seed = source.get("perturb_seed")
        perturb_label = "none" if perturb_seed is None else str(perturb_seed)
        stem = (
            f"{song}__{source.get('condition')}__p-{perturb_label}"
            f"__s-{source.get('sample_seed')}"
        )
        original_midi = Path(str(source.get("source_path", ""))).resolve()
        midi_out = (export_root / "midi" / f"{stem}.mid").resolve()
        canonical_wav = Path(str(source.get("canonical_wav_path", ""))).resolve()
        wav_out = (export_root / "wav_8s" / f"{stem}.wav").resolve()
        source_sha = source.get("source_sha256")
        canonical_wav_sha = source.get("canonical_wav_sha256")
        for path, expected_sha, label in (
            (original_midi, source_sha, "formal theoretical MIDI"),
            (midi_out, source_sha, "semantic MIDI export"),
            (canonical_wav, canonical_wav_sha, "canonical WAV"),
            (wav_out, canonical_wav_sha, "semantic WAV export"),
        ):
            if (
                not isinstance(expected_sha, str)
                or path.is_symlink()
                or not path.is_file()
                or file_sha256(path) != expected_sha
            ):
                raise RuntimeError(
                    f"generated acc row {index_number} {label} differs from private source"
                )
        if not midi_out.is_relative_to(export_root) or not wav_out.is_relative_to(
            export_root
        ):
            raise RuntimeError("generated acc semantic export escapes its export root")
        if midi_out in expected_midi_paths or wav_out in expected_wav_paths:
            raise RuntimeError("generated acc semantic stems are not unique")
        expected_midi_paths.add(midi_out)
        expected_wav_paths.add(wav_out)
        expected_rows.append({
            "song": source.get("song"),
            "condition": source.get("condition"),
            "perturb_seed": perturb_seed,
            "sample_seed": source.get("sample_seed"),
            "run_id": source.get("run_id"),
            "attempt_id": source.get("attempt_id"),
            "operational_valid": source.get("operational_valid"),
            "source_empty": source.get("source_empty"),
            "raw_token_payload_sha256": source.get("raw_token_payload_sha256"),
            "output_event_payload_sha256": source.get(
                "output_event_payload_sha256"
            ),
            "formal_theoretical_midi": source.get("source_path"),
            "formal_theoretical_midi_sha256": source_sha,
            "exported_midi": str(midi_out),
            "exported_midi_sha256": source_sha,
            "exported_excerpt_wav": str(wav_out),
            "exported_excerpt_wav_sha256": canonical_wav_sha,
            "post_unblinding_qualitative_followup_only": True,
        })

    actual_midi_paths = {
        path.resolve()
        for path in (export_root / "midi").rglob("*")
        if path.is_file()
    }
    actual_wav_paths = {
        path.resolve()
        for path in (export_root / "wav_8s").rglob("*")
        if path.is_file()
    }
    if actual_midi_paths != expected_midi_paths or actual_wav_paths != expected_wav_paths:
        raise RuntimeError("generated acc export file set differs from semantic derivation")

    fields = list(expected_rows[0])
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(expected_rows)
    expected_csv = csv_buffer.getvalue().encode("utf-8")
    if csv_path.read_bytes() != expected_csv:
        raise RuntimeError("generated acc CSV differs from exact private-key derivation")

    if require_snapshot_binding:
        expected_index = {
            "schema_version": (
                "streammuse.melody_robustness.generated_acc_export_index.v2"
            ),
            "listening_attempt_id": private_key["listening_attempt_id"],
            "selection_sha256": private_key["selection_sha256"],
            "private_key_sha256": file_sha256(
                listening_root / "private" / "private_key.json"
            ),
            "unblind_state_sha256": file_sha256(
                listening_root / "unblind_state.json"
            ),
            "export_authorizing_snapshot_id": index[
                "export_authorizing_snapshot_id"
            ],
            **snapshot_links,
            "row_count": len(expected_rows),
            "csv_path": str(csv_path),
            "csv_sha256": file_sha256(csv_path),
            "rows": expected_rows,
        }
    else:
        expected_index = {"rows": expected_rows}
    if index != expected_index:
        raise RuntimeError("generated acc JSON differs from exact private-key derivation")
    return {
        "valid": True,
        "row_count": len(expected_rows),
        "json_path": str(json_path),
        "json_sha256": file_sha256(json_path),
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
    }


def _validate_triangle_decision_language(
    summary: Mapping[str, Any], *, listening_attempt_id: str
) -> None:
    conditions = summary.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(
        TRIANGLE_PRIMARY_CONDITIONS
    ):
        raise RuntimeError("triangle summary must contain pitch/onset/both exactly")
    qc_status = summary.get("qc_status")
    expected_disposition = {
        "pending": "in_progress_qc_pending",
        "pass": "eligible_for_preregistered_decisions",
        "fail": "sealed_qc_failure_retry_required",
    }.get(qc_status)
    if (
        summary.get("listening_attempt_id") != listening_attempt_id
        or summary.get("attempt_disposition") != expected_disposition
        or summary.get("retry_required") is not (qc_status == "fail")
    ):
        raise RuntimeError("triangle listening-attempt disposition drifted")

    def require_split_views(value: Mapping[str, Any], label: str) -> None:
        views = value.get("views")
        expected_names = {
            "pre_unblind",
            "post_unblind_exploratory",
            "combined_descriptive",
        }
        if not isinstance(views, Mapping) or set(views) != expected_names:
            raise RuntimeError(f"{label} lacks exact pre/post/combined views")
        pre = views["pre_unblind"]
        post = views["post_unblind_exploratory"]
        combined = views["combined_descriptive"]
        if not all(isinstance(item, Mapping) for item in (pre, post, combined)):
            raise RuntimeError(f"{label} split view is not an object")
        if (
            pre.get("answered") != value.get("pre_unblind_answered")
            or post.get("answered") != value.get("post_unblind_answered")
            or combined.get("answered") != value.get("answered")
            or int(pre.get("answered", -1)) + int(post.get("answered", -1))
            != value.get("answered")
        ):
            raise RuntimeError(f"{label} pre/post/combined denominators drifted")

    for condition in TRIANGLE_PRIMARY_CONDITIONS:
        result = conditions[condition]
        if not isinstance(result, Mapping):
            raise RuntimeError(f"triangle {condition} result is not an object")
        require_split_views(result, f"triangle {condition}")
        per_song = result.get("per_song")
        if not isinstance(per_song, Mapping):
            raise RuntimeError(f"triangle {condition} per-song result is invalid")
        for song, song_result in per_song.items():
            if not isinstance(song_result, Mapping):
                raise RuntimeError(f"triangle {condition}/{song} result is invalid")
            require_split_views(song_result, f"triangle {condition}/{song}")
        eligible = (
            result.get("answered") == 20
            and result.get("pre_unblind_answered") == 20
            and result.get("post_unblind_answered") == 0
            and qc_status == "pass"
        )
        if result.get("decision_eligible") is not eligible:
            raise RuntimeError(f"triangle {condition} decision eligibility drifted")
        decision = result.get("decision")
        if eligible:
            if decision not in {
                "confirmed discriminable in this fixed listener/package",
                "not confirmed in this listening run",
            }:
                raise RuntimeError(f"triangle {condition} full decision wording is invalid")
        elif decision != "partial — preregistered decision pending":
            raise RuntimeError(
                f"triangle {condition} partial data uses a forbidden full decision"
            )


def _validate_triangle_blind_boundary(
    listening_root: Path,
    rows: list[dict[str, Any]],
    *,
    listening_attempt_id: str,
) -> dict[str, Any] | None:
    state_path = listening_root / "unblind_state.json"
    if not state_path.is_file():
        if any(row.get("blinding_phase") != "pre_semantic_unblind" for row in rows):
            raise RuntimeError("post-unblind response exists without semantic unblind state")
        return None
    state = _read(state_path)
    if state.get("schema_version") != (
        "streammuse.melody_robustness.listening_triangle_unblind_state.v2"
    ):
        raise RuntimeError("semantic unblind state schema drifted")
    if set(state) != {
        "schema_version",
        "created_at",
        "listening_attempt_id",
        "first_snapshot_id",
        "first_snapshot_sha256",
        "first_snapshot_ledger_head_hash",
        "first_snapshot_answered_count",
        "selection_sha256",
        "private_key_sha256",
    } or not isinstance(state.get("created_at"), str):
        raise RuntimeError("semantic unblind state fields are invalid")
    if state.get("listening_attempt_id") != listening_attempt_id:
        raise RuntimeError("semantic unblind state listening attempt drifted")
    count = state.get("first_snapshot_answered_count")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(rows):
        raise RuntimeError("first semantic-unblind response boundary is invalid")
    expected_phases = ["pre_semantic_unblind"] * count + [
        "post_partial_unblind_exploratory"
    ] * (len(rows) - count)
    if [row.get("blinding_phase") for row in rows] != expected_phases:
        raise RuntimeError("response blinding phases differ from first semantic-unblind boundary")
    snapshot_id = state.get("first_snapshot_id")
    if not isinstance(snapshot_id, str):
        raise RuntimeError("first semantic-unblind snapshot ID is invalid")
    first = validate_triangle_snapshot(
        listening_root, listening_root / "snapshots" / snapshot_id
    )
    if (
        first.get("answered_count") != count
        or first.get("ledger_head_hash")
        != state.get("first_snapshot_ledger_head_hash")
        or file_sha256(
            listening_root
            / "snapshots"
            / snapshot_id
            / "sealed_responses.json"
        )
        != state.get("first_snapshot_sha256")
        or first.get("selection_sha256") != state.get("selection_sha256")
        or file_sha256(listening_root / "private" / "private_key.json")
        != state.get("private_key_sha256")
    ):
        raise RuntimeError("first semantic-unblind state differs from its immutable snapshot")
    return state


def _triangle_listening_state(
    listening_root: Path,
    *,
    snapshot_path: Path | None,
    config_sha: str,
    schedule_sha: str,
    campaign_binding_sha: str,
    selection_sha: str,
    qualification_result_sha: str,
) -> dict[str, Any]:
    """Validate the flexible 95-trial package and one selected result snapshot."""

    audit_path = listening_root / "package_audit.json"
    render_path = listening_root / "render_manifest.json"
    key_path = listening_root / "private" / "private_key.json"
    public_path = listening_root / "blind" / "public_manifest.json"
    player_path = listening_root / "blind" / "player.html"
    audit = _read(audit_path)
    render = _read(render_path)
    key = _read(key_path)
    public = _read(public_path)
    rendered_selection_path = Path(str(render.get("selection_path", ""))).resolve()
    if (
        not rendered_selection_path.is_file()
        or file_sha256(rendered_selection_path) != selection_sha
    ):
        raise RuntimeError("triangle render selection path/hash drifted")
    rendered_selection = _read(rendered_selection_path)
    listening_attempt_id = render.get("listening_attempt_id")
    _listening_attempt_number(listening_attempt_id)
    if audit.get("schema_version") != TRIANGLE_AUDIT_SCHEMA_VERSION:
        raise RuntimeError("listening package audit is not triangle v2")
    if render.get("schema_version") != TRIANGLE_RENDER_SCHEMA_VERSION:
        raise RuntimeError("listening render manifest is not triangle v2")
    if key.get("schema_version") != TRIANGLE_PRIVATE_KEY_SCHEMA_VERSION:
        raise RuntimeError("listening private key is not triangle v2")
    if public.get("schema_version") != (
        "streammuse.melody_robustness.listening_triangle_public.v2"
    ):
        raise RuntimeError("listening public manifest is not triangle v2")
    links = {
        "campaign_config_sha256": config_sha,
        "run_schedule_sha256": schedule_sha,
        "campaign_binding_sha256": campaign_binding_sha,
        "selection_sha256": selection_sha,
        "qualification_result_sha256": qualification_result_sha,
    }
    _require_exact_links(audit, links, label="triangle package audit")
    _require_exact_links(render, links, label="triangle render manifest")
    _require_exact_links(key, links, label="triangle private key")
    for label, value in (
        ("triangle package audit", audit),
        ("triangle render manifest", render),
        ("triangle private key", key),
        ("triangle public manifest", public),
    ):
        if value.get("listening_attempt_id") != listening_attempt_id:
            raise RuntimeError(f"{label} listening attempt ID drifted")
    if rendered_selection.get("listening_attempt_id") != listening_attempt_id:
        raise RuntimeError("triangle selection listening attempt ID drifted")
    for label, value in (
        ("triangle package audit", audit),
        ("triangle render manifest", render),
        ("triangle private key", key),
    ):
        if (
            value.get("retry_lineage") != rendered_selection.get("retry_lineage")
            or value.get("retry_lineage_sha256")
            != rendered_selection.get("retry_lineage_sha256")
        ):
            raise RuntimeError(f"{label} retry lineage differs from selection")
    attempt_history = _validate_triangle_attempt_lineage(
        rendered_selection,
        selection_path=rendered_selection_path,
        current_package=listening_root,
    )
    if audit.get("valid") is not True or audit.get("errors") != []:
        raise RuntimeError("triangle listening package audit is not valid")
    if audit.get("trial_count") != TRIANGLE_TRIAL_COUNT:
        raise RuntimeError("triangle package audit trial count drifted")
    if audit.get("presentation_count") != TRIANGLE_PRESENTATION_COUNT:
        raise RuntimeError("triangle package audit presentation count drifted")
    if audit.get("practice_count") != TRIANGLE_PRACTICE_COUNT:
        raise RuntimeError("triangle package audit practice count drifted")
    if audit.get("private_key_sha256") != file_sha256(key_path):
        raise RuntimeError("triangle package audit private key hash mismatch")
    if audit.get("render_manifest_sha256") != file_sha256(render_path):
        raise RuntimeError("triangle package audit render manifest hash mismatch")
    if render.get("private_key_sha256") != file_sha256(key_path):
        raise RuntimeError("triangle render private key hash mismatch")
    if render.get("public_manifest_sha256") != file_sha256(public_path):
        raise RuntimeError("triangle public manifest hash mismatch")
    if not player_path.is_file() or render.get("player_sha256") != file_sha256(player_path):
        raise RuntimeError("triangle blind player is missing or hash-mismatched")
    expected_render = {
        "render_bpm": TRIANGLE_RENDER_BPM,
        "clip_seconds": TRIANGLE_CLIP_SECONDS,
        "sample_rate": TRIANGLE_RENDER_SAMPLE_RATE,
        "bit_depth": 16,
        "gain": TRIANGLE_SYNTH_GAIN,
        "gain_policy": TRIANGLE_GAIN_POLICY,
        "midi_only_development": False,
    }
    for field, expected in expected_render.items():
        if render.get(field) != expected:
            raise RuntimeError(f"triangle render {field} is not frozen to {expected!r}")
    if public.get("semantic_fields_present") is not False:
        raise RuntimeError("triangle public manifest exposes semantic fields")
    if key.get("unblind_only_from_immutable_snapshot") is not True:
        raise RuntimeError("triangle private key permits unsealed unblinding")
    if audit.get("accepted_final") is not True or audit.get("blinding_audited") is not True:
        raise RuntimeError("triangle package is not an accepted final blind WAV package")
    _validate_triangle_render_artifacts(listening_root, render, require_wav=True)

    rows, ledger_head = validate_triangle_response_ledger(listening_root)
    progress = triangle_progress_summary(listening_root)
    stored_progress = _read(listening_root / "blind" / "progress_state.json")
    if stored_progress != progress:
        raise RuntimeError("triangle atomic progress state differs from the response ledger")
    if progress.get("schema_version") != TRIANGLE_PROGRESS_SCHEMA_VERSION:
        raise RuntimeError("triangle progress schema drifted")
    if progress.get("listening_attempt_id") != listening_attempt_id:
        raise RuntimeError("triangle progress listening attempt drifted")
    if progress.get("ledger_head_hash") != ledger_head:
        raise RuntimeError("triangle progress ledger head mismatch")
    sittings = progress.get("sittings")
    sitting_event_count = progress.get("sitting_event_count")
    if (
        not isinstance(sittings, Mapping)
        or isinstance(sitting_event_count, bool)
        or not isinstance(sitting_event_count, int)
        or sitting_event_count < len(sittings)
    ):
        raise RuntimeError("triangle structured sitting provenance drifted")
    for sitting_id, sitting in sittings.items():
        if (
            not isinstance(sitting_id, str)
            or not isinstance(sitting, Mapping)
            or set(sitting)
            != {
                "device",
                "environment",
                "started_at",
                "ended_at",
                "anomalies",
                "answered_count",
                "active",
            }
            or not isinstance(sitting.get("device"), str)
            or not sitting.get("device")
            or not isinstance(sitting.get("environment"), str)
            or not sitting.get("environment")
            or not isinstance(sitting.get("anomalies"), list)
        ):
            raise RuntimeError("triangle sitting device/environment/anomaly record drifted")
    sitting_provenance = {
        "sitting_count": len(sittings),
        "sitting_event_count": sitting_event_count,
        "sitting_ledger_head_hash": progress.get("sitting_ledger_head_hash"),
        "sittings": dict(sittings),
    }
    unblind_state = _validate_triangle_blind_boundary(
        listening_root,
        rows,
        listening_attempt_id=listening_attempt_id,
    )
    identity_counts = _triangle_identity_counts(key, render)

    if not rows:
        if snapshot_path is not None:
            raise RuntimeError("a listening snapshot cannot be selected before any response")
        snapshots_root = listening_root / "snapshots"
        if snapshots_root.is_dir() and any(snapshots_root.iterdir()):
            raise RuntimeError("triangle snapshots exist without a response ledger")
        if unblind_state is not None:
            raise RuntimeError("semantic unblind exists before listening started")
        return {
            "contract": "triangle_v2",
            "listening_attempt_id": listening_attempt_id,
            "listening_attempt_history": attempt_history,
            "sitting_provenance": sitting_provenance,
            "package_audit": audit,
            "package_valid": True,
            "collection_status": "not_started",
            "answered_count": 0,
            "pending_count": TRIANGLE_TRIAL_COUNT,
            "ledger_answered_count": 0,
            "ledger_head_hash": None,
            "snapshot_status": "not_applicable",
            "snapshot": None,
            "snapshot_path": None,
            "snapshot_sha256": None,
            "semantic_result_status": "not_available",
            "qc_status": "not_started",
            "attempt_disposition": "not_started",
            "retry_required": False,
            "blinding_status": "fully_blind",
            "first_semantic_unblind": None,
            "summary": None,
            "identity_counts": identity_counts,
            "generated_acc_export": {
                "valid": True,
                "not_applicable": True,
                "row_count": 0,
            },
        }

    if snapshot_path is None:
        raise RuntimeError(
            "--listening-snapshot is required once triangle responses exist"
        )
    snapshot_root = snapshot_path.resolve()
    sealed = validate_triangle_snapshot(listening_root, snapshot_root)
    answered = int(sealed["answered_count"])
    if sealed.get("responses") != rows[:answered]:
        raise RuntimeError("selected snapshot is not an immutable prefix of the response ledger")
    if unblind_state is None:
        raise RuntimeError("selected semantic result lacks a semantic-unblind state")
    first_unblind_count = int(unblind_state["first_snapshot_answered_count"])
    expected_state_at_snapshot = (
        None if answered <= first_unblind_count else unblind_state
    )
    if sealed.get("unblind_state_at_snapshot") != expected_state_at_snapshot:
        raise RuntimeError(
            "selected snapshot records an invalid semantic-unblind state at creation"
        )
    unblinded, summary = validate_triangle_unblinded_summary(
        listening_root, snapshot_root
    )
    if unblinded.get("first_semantic_unblind") != unblind_state:
        raise RuntimeError("selected unblinded result uses another semantic-unblind boundary")
    if summary.get("answered_count") != answered:
        raise RuntimeError("triangle summary answered count differs from selected snapshot")
    expected_collection = "full" if answered == TRIANGLE_TRIAL_COUNT else "partial"
    if summary.get("collection_status") != expected_collection:
        raise RuntimeError("triangle summary collection status differs from selected snapshot")
    expected_blinding_status = (
        "fully_blind_for_answered_rows"
        if answered <= first_unblind_count
        else "partially_unblinded_during_collection"
    )
    if summary.get("blinding_status") != expected_blinding_status:
        raise RuntimeError(
            "triangle summary blinding status differs from the immutable first-unblind event"
        )
    _validate_triangle_decision_language(
        summary, listening_attempt_id=listening_attempt_id
    )
    if answered == TRIANGLE_TRIAL_COUNT:
        full_unblinded = listening_root / "full" / "unblinded_scores.json"
        full_summary = listening_root / "full" / "discrimination_summary.json"
        if (
            not full_unblinded.is_file()
            or not full_summary.is_file()
            or file_sha256(full_unblinded)
            != file_sha256(snapshot_root / "partial_unblinded_scores.json")
            or file_sha256(full_summary)
            != file_sha256(snapshot_root / "partial_discrimination_summary.json")
        ):
            raise RuntimeError("full triangle result copies are missing or differ from snapshot")
    generated_export = _validate_generated_acc_export(listening_root, key)
    if generated_export.get("valid") is not True:
        raise RuntimeError(
            "semantic unblinding requires a complete hash-bound generated-acc MIDI/WAV export"
        )
    return {
        "contract": "triangle_v2",
        "listening_attempt_id": listening_attempt_id,
        "listening_attempt_history": attempt_history,
        "sitting_provenance": sitting_provenance,
        "package_audit": audit,
        "package_valid": True,
        "collection_status": expected_collection,
        "answered_count": answered,
        "pending_count": TRIANGLE_TRIAL_COUNT - answered,
        "ledger_answered_count": len(rows),
        "ledger_head_hash": ledger_head,
        "snapshot_status": "valid",
        "snapshot": sealed,
        "snapshot_path": str(snapshot_root),
        "snapshot_sha256": file_sha256(snapshot_root / "sealed_responses.json"),
        "semantic_result_status": "valid",
        "qc_status": summary["qc_status"],
        "attempt_disposition": summary["attempt_disposition"],
        "retry_required": summary["retry_required"],
        "blinding_status": expected_blinding_status,
        "first_semantic_unblind": unblind_state,
        "summary": summary,
        "identity_counts": identity_counts,
        "generated_acc_export": generated_export,
    }


def _listening_state(
    listening_root: Path,
    *,
    snapshot_path: Path | None,
    config_sha: str,
    schedule_sha: str,
    campaign_binding_sha: str,
    selection_sha: str,
    qualification_result_sha: str,
) -> dict[str, Any]:
    render = _read(listening_root / "render_manifest.json")
    if render.get("schema_version") == TRIANGLE_RENDER_SCHEMA_VERSION:
        return _triangle_listening_state(
            listening_root,
            snapshot_path=snapshot_path,
            config_sha=config_sha,
            schedule_sha=schedule_sha,
            campaign_binding_sha=campaign_binding_sha,
            selection_sha=selection_sha,
            qualification_result_sha=qualification_result_sha,
        )
    audit, sealed, unblinded = _legacy_listening_completion(
        listening_root,
        config_sha=config_sha,
        schedule_sha=schedule_sha,
        campaign_binding_sha=campaign_binding_sha,
        selection_sha=selection_sha,
        qualification_result_sha=qualification_result_sha,
    )
    return {
        "contract": "legacy_quality_v1",
        "listening_attempt_id": None,
        "package_audit": audit,
        "package_valid": bool(audit.get("accepted_final")),
        "collection_status": "full" if sealed else "not_started",
        "answered_count": 24 if sealed else 0,
        "pending_count": 0 if sealed else 24,
        "ledger_answered_count": 24 if sealed else 0,
        "snapshot_status": "legacy_sealed" if sealed else "not_applicable",
        "semantic_result_status": "legacy_unblinded" if unblinded else "not_available",
        "qc_status": "not_applicable",
        "attempt_disposition": "legacy_contract",
        "retry_required": False,
        "blinding_status": "unblinded" if unblinded else "fully_blind",
        "summary": None,
        "identity_counts": {},
        "generated_acc_export": {"valid": False, "reason": "legacy_contract"},
    }


def _listening_completion(
    listening_root: Path,
    *,
    config_sha: str,
    schedule_sha: str,
    campaign_binding_sha: str,
    selection_sha: str,
    qualification_result_sha: str,
    snapshot_path: Path | None = None,
) -> tuple[dict[str, Any], bool, bool]:
    """Compatibility wrapper retained for callers of the historical helper."""

    state = _listening_state(
        listening_root,
        snapshot_path=snapshot_path,
        config_sha=config_sha,
        schedule_sha=schedule_sha,
        campaign_binding_sha=campaign_binding_sha,
        selection_sha=selection_sha,
        qualification_result_sha=qualification_result_sha,
    )
    return (
        state["package_audit"],
        state["snapshot_status"] in {"valid", "legacy_sealed"},
        state["semantic_result_status"] in {"valid", "legacy_unblinded"},
    )


def _artifact_tree(root: Path) -> dict[str, Any]:
    files = [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*")) if path.is_file()
    ]
    return {"root": str(root), "file_count": len(files), "tree_sha256": canonical_sha256(files), "files": files}


def _copy_generated_acc_export_to_report(
    listening_root: Path,
    output: Path,
    generated_export: Mapping[str, Any],
) -> dict[str, Any]:
    """Make an exact immutable report-local copy of the validated semantic export."""

    destination = output / "generated_acc_after_unblind"
    if generated_export.get("not_applicable") is True:
        if destination.exists():
            raise RuntimeError(
                "not-started report output contains a stale semantic generated-acc copy"
            )
        return {"valid": True, "not_applicable": True, "file_count": 0}
    if generated_export.get("valid") is not True:
        raise RuntimeError("cannot copy an invalid generated-acc export into the report")
    source_root = listening_root / "generated_acc_after_unblind"
    index = _read(source_root / "generated_acc_index.json")
    rows = index.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("validated generated-acc export has no rows to copy")
    sources: dict[Path, Path] = {
        source_root / "generated_acc_index.json": destination
        / "generated_acc_index.json",
        source_root / "generated_acc_index.csv": destination
        / "generated_acc_index.csv",
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("generated-acc copy encountered a non-object row")
        for field, subdir in (
            ("exported_midi", "midi"),
            ("exported_excerpt_wav", "wav_8s"),
        ):
            source = Path(str(row.get(field, ""))).resolve()
            expected_parent = (source_root / subdir).resolve()
            if source.parent != expected_parent or not source.is_file():
                raise RuntimeError("generated-acc copy source layout drifted")
            target = destination / subdir / source.name
            if source in sources or target in sources.values():
                raise RuntimeError("generated-acc copy contains duplicate semantic paths")
            sources[source] = target
    expected_relative = {
        target.relative_to(destination) for target in sources.values()
    }
    if destination.exists():
        actual_relative = {
            path.relative_to(destination)
            for path in destination.rglob("*")
            if path.is_file()
        }
        if actual_relative - expected_relative:
            raise RuntimeError("report-local generated-acc copy contains extra files")
    for source, target in sources.items():
        if source.is_symlink() or not source.is_file():
            raise RuntimeError("generated-acc copy source is missing or a symlink")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or not target.is_file() or file_sha256(
                target
            ) != file_sha256(source):
                raise RuntimeError("existing report-local generated-acc copy drifted")
        else:
            shutil.copyfile(source, target)
        if file_sha256(target) != file_sha256(source):
            raise RuntimeError("report-local generated-acc copy failed hash verification")
    actual_relative = {
        path.relative_to(destination)
        for path in destination.rglob("*")
        if path.is_file()
    }
    if actual_relative != expected_relative:
        raise RuntimeError("report-local generated-acc copy is incomplete")
    tree = _artifact_tree(destination)
    return {
        "valid": True,
        "not_applicable": False,
        "root": str(destination),
        "source_root": str(source_root),
        "file_count": tree["file_count"],
        "tree_sha256": tree["tree_sha256"],
    }


def _effect(result: dict[str, Any], name: str) -> str:
    value = result.get(name, {})
    estimate = value.get("estimate")
    interval = value.get("interval")
    songs = value.get("raw_song_effects")
    if estimate is None:
        return f"{name}: NA（有效 song block={value.get('valid_song_count', 0)}；NA pattern 已保留）。"
    return (
        f"{name}: equal-song estimate={estimate:.4f}, descriptive 95% interval="
        f"[{interval[0]:.4f}, {interval[1]:.4f}], raw song effects={songs}."
    )


def build(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    schedule_path = Path(args.schedule).resolve()
    campaign_root = Path(args.campaign_root).resolve()
    analysis_root = Path(args.analysis_dir).resolve()
    listening_root = Path(args.listening_package).resolve()
    output = Path(args.output_dir).resolve()
    (
        config,
        config_sha,
        input_manifest_path,
        input_manifest,
        inputs,
        schedule,
    ) = _validated_campaign_inputs(
        config_path,
        schedule_path,
        expected_config_sha256=args.config_sha256,
        expected_schedule_sha256=args.schedule_sha256,
    )
    campaign_binding, campaign_binding_sha = _validated_campaign_binding(
        campaign_root,
        config_path=config_path,
        schedule_path=schedule_path,
        config=config,
        config_sha=config_sha,
        schedule_sha=args.schedule_sha256,
    )
    audit_path = Path(args.campaign_audit).resolve()
    audit = _read(audit_path)
    _validate_audit_schedule(audit, schedule)
    for field, expected in {
        "campaign_config_sha256": config_sha,
        "run_schedule_sha256": args.schedule_sha256,
        "campaign_binding_sha256": campaign_binding_sha,
    }.items():
        if audit.get(field) != expected:
            raise RuntimeError(f"campaign audit {field} mismatch")
    bootstrap, controls, analysis_index = _validated_analysis_artifacts(
        analysis_root,
        config_sha,
        input_manifest_sha=str(config["input_manifest"]["sha256"]),
        schedule_sha=args.schedule_sha256,
        campaign_binding_sha=campaign_binding_sha,
        qualification_result_sha=config["qualification_result"]["sha256"],
    )
    base_selection_path = Path(
        config["listening"]["selection_manifest_path"]
    ).resolve()
    base_selection_sha = str(config["listening"]["selection_manifest_sha256"])
    if (
        not base_selection_path.is_file()
        or file_sha256(base_selection_path) != base_selection_sha
    ):
        raise RuntimeError("C5 base listening selection hash mismatch")
    base_selection = _read(base_selection_path)
    render_manifest = _read(listening_root / "render_manifest.json")
    if render_manifest.get("schema_version") == TRIANGLE_RENDER_SCHEMA_VERSION:
        selection_path = Path(str(render_manifest.get("selection_path", ""))).resolve()
        selection_sha = render_manifest.get("selection_sha256")
        if (
            not isinstance(selection_sha, str)
            or not selection_path.is_file()
            or file_sha256(selection_path) != selection_sha
        ):
            raise RuntimeError("current triangle listening selection path/hash mismatch")
    else:
        selection_path = base_selection_path
        selection_sha = base_selection_sha
    selection = _read(selection_path)
    if selection.get("schema_version") == TRIANGLE_SELECTION_SCHEMA_VERSION:
        validate_triangle_selection_manifest(
            selection,
            input_manifest,
            manifest_path=input_manifest_path,
            verify_files=True,
        )
        attempt_number = _listening_attempt_number(
            selection.get("listening_attempt_id")
        )
        if selection.get("listening_attempt_number") != attempt_number:
            raise RuntimeError("triangle selection attempt ID/number mismatch")
        if attempt_number == 1:
            if selection_path != base_selection_path or selection_sha != base_selection_sha:
                raise RuntimeError("attempt-001 differs from the C5 base selection")
        else:
            lineage = selection.get("retry_lineage")
            if (
                not isinstance(lineage, Mapping)
                or Path(str(lineage.get("base_selection_path", ""))).resolve()
                != base_selection_path
                or lineage.get("base_selection_sha256") != base_selection_sha
            ):
                raise RuntimeError("retry selection does not bind the C5 base selection")
        _validate_listening_known_different_controls(
            controls,
            base_selection=base_selection,
            base_selection_sha256=base_selection_sha,
            config_sha=config_sha,
        )
        _reverify_campaign_audit(
            campaign_root,
            audit=audit,
            schedule=schedule,
            campaign_binding={
                **campaign_binding,
                "campaign_binding_sha256": campaign_binding_sha,
            },
            selection=selection,
        )
    if render_manifest.get("selection_sha256") != selection_sha:
        raise RuntimeError("listening render manifest belongs to another selection/campaign")
    if render_manifest.get("schema_version") == TRIANGLE_RENDER_SCHEMA_VERSION:
        selection_attempt_id = selection.get("listening_attempt_id")
        _listening_attempt_number(selection_attempt_id)
        if render_manifest.get("listening_attempt_id") != selection_attempt_id:
            raise RuntimeError("triangle selection/render listening attempt mismatch")
        rendered_selection_path = Path(
            str(render_manifest.get("selection_path", ""))
        ).resolve()
        if rendered_selection_path != selection_path:
            raise RuntimeError("triangle render manifest selection path mismatch")
        stored_triangle_audit = _read(listening_root / "package_audit.json")
        recomputed_triangle_audit = _recompute_triangle_package_audit(listening_root)
        if recomputed_triangle_audit != stored_triangle_audit:
            raise RuntimeError(
                "stored triangle package audit differs from an exact current recomputation"
            )
    raw_snapshot = getattr(args, "listening_snapshot", None)
    listening_snapshot = Path(raw_snapshot).resolve() if raw_snapshot else None
    listening = _listening_state(
        listening_root,
        snapshot_path=listening_snapshot,
        config_sha=config_sha,
        schedule_sha=args.schedule_sha256,
        campaign_binding_sha=campaign_binding_sha,
        selection_sha=selection_sha,
        qualification_result_sha=config["qualification_result"]["sha256"],
    )
    output.mkdir(parents=True, exist_ok=True)
    report_generated_acc_copy = _copy_generated_acc_export_to_report(
        listening_root,
        output,
        listening["generated_acc_export"],
    )
    listening_audit = listening["package_audit"]
    results = bootstrap.get("results", {})
    control_endpoint_valid = {
        endpoint: not controls.get("missing_songs") and len(controls.get("songs", [])) == 5 and all(
            bool(song["endpoint_validity"][endpoint]) for song in controls.get("songs", [])
        )
        for endpoint in ("harmonic", "rhythm", "coverage")
    }
    checks = {
        "qualification_passed_and_bound": True,
        "input_staging_40": len(inputs) == 40,
        "formal_expected_160": audit.get("expected") == 160,
        "formal_content_valid_160": audit.get("content_valid") == 160,
        "formal_missing_zero": audit.get("missing") == 0,
        "formal_invalid_zero": audit.get("invalid") == 0,
        "formal_extra_run_ids_zero": audit.get("extra_run_ids") == [],
        "listening_source_readiness": audit.get("listening_source_readiness", {}).get(
            "ready"
        )
        is True,
        "analysis_qc_zero_invalid": analysis_index.get("qc_invalid") == 0,
        "targeted_controls": all(control_endpoint_valid.values()),
        "listening_known_different_controls": bool(
            listening["contract"] != "triangle_v2"
            or controls.get("listening_known_different", {}).get("selection_sha256")
            == base_selection_sha
        ),
        "triangle_listening_contract_v2": listening["contract"] == "triangle_v2",
        "blind_triangle_package": bool(listening["package_valid"]),
        "append_only_listening_ledger": listening["contract"] == "triangle_v2",
        "selected_snapshot_or_not_started": listening["snapshot_status"]
        in {"valid", "not_applicable"},
        "semantic_summary_or_not_started": listening["semantic_result_status"]
        in {"valid", "not_available"},
        "generated_acc_export_or_not_started": bool(
            listening["generated_acc_export"].get("valid")
        ),
        "report_local_generated_acc_copy_or_not_started": bool(
            report_generated_acc_copy.get("valid")
        ),
        "provenance_cross_links": bool(
            analysis_index.get("campaign_config_sha256") == config_sha
            and analysis_index.get("input_manifest_sha256")
            == config["input_manifest"]["sha256"]
            and analysis_index.get("run_schedule_sha256") == args.schedule_sha256
            and audit.get("campaign_binding_sha256") == campaign_binding_sha
            and listening_audit.get("selection_sha256") == selection_sha
            and analysis_index.get("qualification_result_sha256")
            == config["qualification_result"]["sha256"]
            and audit.get("qualification_result_sha256")
            == config["qualification_result"]["sha256"]
            and listening_audit.get("qualification_result_sha256")
            == config["qualification_result"]["sha256"]
        ),
    }
    complete = all(checks.values())
    collection_status = str(listening["collection_status"])
    qc_status = str(listening["qc_status"])
    blinding_status = str(listening["blinding_status"])
    attempt_disposition = str(listening["attempt_disposition"])
    retry_required = bool(listening["retry_required"])
    trees = {
        "campaign": _artifact_tree(campaign_root),
        "analysis": _artifact_tree(analysis_root),
        "listening": _artifact_tree(listening_root),
    }
    if report_generated_acc_copy.get("not_applicable") is not True:
        trees["report_generated_acc"] = _artifact_tree(
            output / "generated_acc_after_unblind"
        )
    runtime_info_files = sorted(campaign_root.rglob("command.json"))
    attempt_verdicts = sorted(campaign_root.rglob("attempt-*/verdict.json"))
    listening_key_path = (
        listening_root / "private" / "private_key.json"
        if listening["contract"] == "triangle_v2"
        else listening_root / "private_key.json"
    )
    index = {
        "schema_version": "streammuse.melody_robustness.reproducibility_index.v2",
        "campaign_config_sha256": config_sha,
        "status": "complete" if complete else "incomplete",
        "definition_of_done": checks,
        "code_identity": config["code_identity"],
        "campaign_config": {"path": str(config_path), "sha256": config_sha},
        "checkpoint": config["checkpoint"],
        "input_manifest": {"path": str(input_manifest_path), "sha256": file_sha256(input_manifest_path)},
        "run_manifest": {"path": str(schedule_path), "sha256": file_sha256(schedule_path)},
        "campaign_audit": {"path": str(audit_path), "sha256": file_sha256(audit_path)},
        "campaign_binding": {
            "path": str(campaign_root / "campaign_binding.json"),
            "sha256": campaign_binding_sha,
        },
        "qualification_candidate": config["qualification_candidate"],
        "qualification_result": config["qualification_result"],
        "attempt_verdict_count": len(attempt_verdicts),
        "runtime_command_record_count": len(runtime_info_files),
        "analysis_index": {"path": str(analysis_root / "analysis_index.json"), "sha256": file_sha256(analysis_root / "analysis_index.json")},
        "control_report": {"path": str(analysis_root / "control_report.json"), "sha256": file_sha256(analysis_root / "control_report.json")},
        "listening_base_selection": {
            "path": str(base_selection_path),
            "sha256": base_selection_sha,
        },
        "listening_selection": {
            "path": str(selection_path),
            "sha256": selection_sha,
        },
        "listening_key": {
            "path": str(listening_key_path),
            "sha256": file_sha256(listening_key_path),
        },
        "listening_result": {
            "contract": listening["contract"],
            "listening_attempt_id": listening.get("listening_attempt_id"),
            "listening_attempt_history": listening.get(
                "listening_attempt_history", []
            ),
            "collection_status": collection_status,
            "answered_count": listening["answered_count"],
            "pending_count": listening["pending_count"],
            "ledger_answered_count": listening["ledger_answered_count"],
            "ledger_head_hash": listening.get("ledger_head_hash"),
            "snapshot_status": listening["snapshot_status"],
            "snapshot_path": listening.get("snapshot_path"),
            "snapshot_sha256": listening.get("snapshot_sha256"),
            "semantic_result_status": listening["semantic_result_status"],
            "qc_status": qc_status,
            "attempt_disposition": attempt_disposition,
            "retry_required": retry_required,
            "blinding_status": blinding_status,
            "first_semantic_unblind": listening.get("first_semantic_unblind"),
            "sitting_provenance": listening.get("sitting_provenance"),
            "identity_counts": listening["identity_counts"],
            "generated_acc_export": listening["generated_acc_export"],
            "report_local_generated_acc_copy": report_generated_acc_copy,
            "summary": listening["summary"],
        },
        "artifact_trees": trees,
    }
    index_path = output / "reproducibility_index.json"
    write_canonical_json(index_path, index)
    primary = _effect(results, "rt_theoretical:both_vs_sham")
    offline = _effect(results, "offline:both_vs_sham")
    interaction = _effect(results, "H_generation_interaction:both")
    operational = _effect(results, "H_operational:both")
    incomplete = [key for key, passed in checks.items() if not passed]
    triangle_summary = listening["summary"]
    if isinstance(triangle_summary, Mapping):
        condition_lines = []
        for condition in TRIANGLE_PRIMARY_CONDITIONS:
            value = triangle_summary["conditions"][condition]
            condition_lines.append(
                "- "
                f"{condition}: correct={value['correct']}/{value['answered']}, "
                f"no_difference={value['no_difference']}, "
                f"mean_confidence={value['mean_confidence']}, "
                f"answered-row operational_invalid_either="
                f"{value['operational_invalid_either']}/{value['answered']}, "
                f"identity(raw/theoretical/canonical/rendered)="
                f"{value['raw_token_identity']}/"
                f"{value['theoretical_midi_identity']}/"
                f"{value['canonical_midi_identity']}/"
                f"{value['rendered_wav_identity']} of {value['answered']}, "
                f"pre_unblind={value['pre_unblind_answered']}, "
                f"post_unblind_exploratory={value['post_unblind_answered']}, "
                f"decision=`{value['decision']}`; "
                f"pre/post/combined views=`{value['views']}`; "
                f"per-song pre/post/combined=`{value['per_song']}`"
            )
        condition_result_text = "\n".join(condition_lines)
        reference_result_text = (
            f"high-dose=`{triangle_summary['high_exploratory']}`；"
            f"sham-sampling=`{triangle_summary['sham_sampling_baseline']}`。"
        )
        qc_result_text = str(triangle_summary["quality_control"])
        view_result_text = str(triangle_summary["views"])
    else:
        condition_result_text = (
            "- 尚无 human response；pitch/onset/both 均不得给出 perceptual conclusion。"
        )
        reference_result_text = "high-dose/sham-sampling 尚无 human denominator。"
        qc_result_text = "not started"
        view_result_text = "not available"
    attempt_history_text = str(listening.get("listening_attempt_history", []))
    report = f"""# Melody 扰动鲁棒性 exploratory pilot report

Technical campaign 状态：**{'COMPLETE' if complete else 'INCOMPLETE'}**。听测 attempt：**{listening.get('listening_attempt_id')}**；采集状态：**{collection_status}**（answered={listening['answered_count']}, pending={listening['pending_count']}）；QC：**{qc_status}**；attempt disposition：**{attempt_disposition}**；retry required：**{retry_required}**；blinding：**{blinding_status}**。本报告只描述当前 5 首 pilot 歌；不做正式显著性、等价性或人群外推。配置 hash：`{config_sha}`。

## H_input：固定 clean reference 下的输入扰动效应

Primary contrast 是 medium both vs sham；primary quality 是 D_intended，并与 coverage guardrail 联合解释。Sensitivity 只表示输出变化，不能替代质量。

- {offline}
- {primary}

D_actual 与 adaptation/direct-melody 交叉参照在 `paired_contrasts.jsonl` 中分列。D_actual(cond)−D_actual(sham) 是 joint treatment effect，不直接称为 adaptation。若 D 降低但 coverage 同时塌缩，不称为改善；完全空伴奏为 NA/coverage failure。

## H_generation_interaction：offline vs RT theoretical

- {interaction}

这里比较的是完整 generation pipeline（增量逐拍、自回馈、context 窗口等）的差异，不归因到单一组件，也不包含播放调度。

## H_operational：RT theoretical vs RT combined

- {operational}

Late/drop/clamp/forced note-off 和首 attempt failure 保留在 campaign audit 与 validity artifacts 中；重跑不覆盖首次失败。

## Targeted assay controls

Endpoint validity：`{control_endpoint_valid}`。阈值是预定义方向与最小变化，不使用“显著更差”作为验收标准。任一 endpoint control 失败时，该 endpoint 的质量结论无效。

## H_realworld

本实验没有真人输入，不能回答真实人类失误是否解释 live gap。4-ticks-per-beat 的合成错误不是人类微时序误差。

## 听测

这是 95 题、每题三段、每段 8 秒的 acc-only triangle discrimination；120 BPM、固定 gain、matched-pair common gain、true-peak protection only。答题可随时停止并恢复；当前 report 只使用所选 immutable snapshot，未回答题不视为错误。

Listening attempt history（所有 sealed QC-failed predecessor 均保留）：`{attempt_history_text}`

Structured sitting provenance（设备、环境、时间、异常、实际答题数）：`{listening.get('sitting_provenance')}`

{condition_result_text}

参照结果：{reference_result_text}

QC coverage/result：`{qc_result_text}`

Pre-unblind / post-partial-unblind exploratory views：`{view_result_text}`

Raw generated source / canonical excerpt MIDI / rendered WAV identity counts：`{listening['identity_counts']}`。Generated-acc semantic export：`{listening['generated_acc_export']}`。Report-local durable copy：`{report_generated_acc_copy}`。

collection_status 只表示答题数量，QC 与 blinding 单列。任一 condition 未完成其 20 题、20 题并非全部在第一次 semantic unblind 前完成、或 full QC 未通过时，只允许 `partial — preregistered decision pending`；post-partial-unblind 新答 rows 仅作 exploratory。即使完整结果 confirmed，也只表示这个 fixed listener/package 中可辨别，不表示更差、更好、和声原因或总体人群效应。

## Limitations

5 首歌、fixed 8-second excerpts、2 个 sample seeds、单听者、trial/source dependence、描述性 song-block bootstrap、合成 model-tick 扰动、潜在空输出/NA。Discrimination 不回答质量、偏好或和声正确性；未做显著性、等价性或总体外推。解盲后的自由 A/B 复听只能单列为 qualitative follow-up，不能回写 primary blind responses。

## 完成判据与未完成项

`{checks}`

未完成/无效项：`{incomplete if incomplete else 'none'}`。

完整 code SHA、checkpoint/config/input/run/attempt/runtime/metrics/control/listening key 交叉引用见 `reproducibility_index.json`（hash `{file_sha256(index_path)}`）。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    if not complete and not args.allow_incomplete:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--schedule-sha256", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--campaign-audit", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--listening-package", required=True)
    parser.add_argument(
        "--listening-snapshot",
        help=(
            "immutable triangle snapshot to report; required once one or more "
            "responses exist, omitted for an objective-only not_started report"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
