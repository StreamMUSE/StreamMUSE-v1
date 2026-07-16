#!/usr/bin/env python3
"""Build the constrained pilot report and cross-linked reproducibility index."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from streammuse.experiments.melody_robustness import (
    build_run_schedule,
    canonical_sha256,
    file_sha256,
    read_jsonl,
    validate_campaign_config,
    validate_staged_input_manifest,
    write_canonical_json,
)


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
    }
    if not path.is_file() or _read(path) != expected:
        raise RuntimeError("campaign output root binding does not match config/schedule")
    return expected, file_sha256(path)


def _listening_completion(
    listening_root: Path,
    *,
    config_sha: str,
    schedule_sha: str,
    campaign_binding_sha: str,
    selection_sha: str,
) -> tuple[dict[str, Any], bool, bool]:
    """Validate listening provenance; absence of human scores remains incomplete."""
    audit_path = listening_root / "package_audit.json"
    audit = _read(audit_path)
    expected_links = {
        "campaign_config_sha256": config_sha,
        "run_schedule_sha256": schedule_sha,
        "campaign_binding_sha256": campaign_binding_sha,
        "selection_sha256": selection_sha,
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
        _input_manifest,
        inputs,
        schedule,
    ) = _validated_campaign_inputs(
        config_path,
        schedule_path,
        expected_config_sha256=args.config_sha256,
        expected_schedule_sha256=args.schedule_sha256,
    )
    _binding, campaign_binding_sha = _validated_campaign_binding(
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
    )
    selection_path = Path(config["listening"]["selection_manifest_path"]).resolve()
    selection_sha = str(config["listening"]["selection_manifest_sha256"])
    if file_sha256(selection_path) != selection_sha:
        raise RuntimeError("listening selection hash mismatch")
    private_key = _read(listening_root / "private_key.json")
    render_manifest = _read(listening_root / "render_manifest.json")
    if private_key.get("selection_sha256") != selection_sha:
        raise RuntimeError("listening private key belongs to another selection/campaign")
    if render_manifest.get("selection_sha256") != selection_sha:
        raise RuntimeError("listening render manifest belongs to another selection/campaign")
    listening_audit, scores_sealed, scores_unblinded = _listening_completion(
        listening_root,
        config_sha=config_sha,
        schedule_sha=args.schedule_sha256,
        campaign_binding_sha=campaign_binding_sha,
        selection_sha=selection_sha,
    )
    results = bootstrap.get("results", {})
    control_endpoint_valid = {
        endpoint: not controls.get("missing_songs") and len(controls.get("songs", [])) == 5 and all(
            bool(song["endpoint_validity"][endpoint]) for song in controls.get("songs", [])
        )
        for endpoint in ("harmonic", "rhythm", "coverage")
    }
    checks = {
        "input_staging_40": len(inputs) == 40,
        "formal_expected_160": audit.get("expected") == 160,
        "formal_content_valid_160": audit.get("content_valid") == 160,
        "formal_missing_zero": audit.get("missing") == 0,
        "analysis_qc_zero_invalid": analysis_index.get("qc_invalid") == 0,
        "targeted_controls": all(control_endpoint_valid.values()),
        "blind_package": bool(listening_audit.get("accepted_final")),
        "listening_scores_sealed": scores_sealed,
        "listening_scores_unblinded": scores_unblinded,
        "provenance_cross_links": bool(
            analysis_index.get("campaign_config_sha256") == config_sha
            and analysis_index.get("input_manifest_sha256")
            == config["input_manifest"]["sha256"]
            and analysis_index.get("run_schedule_sha256") == args.schedule_sha256
            and audit.get("campaign_binding_sha256") == campaign_binding_sha
            and listening_audit.get("selection_sha256") == selection_sha
        ),
    }
    complete = all(checks.values())
    scores_status = (
        "sealed_and_unblinded" if scores_sealed and scores_unblinded
        else "sealed_not_unblinded" if scores_sealed
        else "not_yet_completed"
    )
    output.mkdir(parents=True, exist_ok=True)
    trees = {
        "campaign": _artifact_tree(campaign_root),
        "analysis": _artifact_tree(analysis_root),
        "listening": _artifact_tree(listening_root),
    }
    runtime_info_files = sorted(campaign_root.rglob("command.json"))
    attempt_verdicts = sorted(campaign_root.rglob("attempt-*/verdict.json"))
    index = {
        "schema_version": "streammuse.melody_robustness.reproducibility_index.v1",
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
        "attempt_verdict_count": len(attempt_verdicts),
        "runtime_command_record_count": len(runtime_info_files),
        "analysis_index": {"path": str(analysis_root / "analysis_index.json"), "sha256": file_sha256(analysis_root / "analysis_index.json")},
        "control_report": {"path": str(analysis_root / "control_report.json"), "sha256": file_sha256(analysis_root / "control_report.json")},
        "listening_selection": {
            "path": config["listening"]["selection_manifest_path"],
            "sha256": config["listening"]["selection_manifest_sha256"],
        },
        "listening_key": {"path": str(listening_root / "private_key.json"), "sha256": file_sha256(listening_root / "private_key.json")},
        "listening_scores_status": scores_status,
        "artifact_trees": trees,
    }
    index_path = output / "reproducibility_index.json"
    write_canonical_json(index_path, index)
    primary = _effect(results, "rt_theoretical:both_vs_sham")
    offline = _effect(results, "offline:both_vs_sham")
    interaction = _effect(results, "H_generation_interaction:both")
    operational = _effect(results, "H_operational:both")
    incomplete = [key for key, passed in checks.items() if not passed]
    report = f"""# Melody 扰动鲁棒性 exploratory pilot report

状态：**{'COMPLETE' if complete else 'INCOMPLETE'}**。本报告只描述当前 5 首 pilot 歌；不做正式显著性、等价性或人群外推。配置 hash：`{config_sha}`。

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

24 clips、120 BPM、固定 gain、true-peak protection only；ecological 评分回答 end-to-end joint quality，acc-solo 只回答 standalone coherence。单听者结果是 exploratory qualitative judgement。当前评分状态：`{scores_status}`；解盲后的观察必须另列 post-unblinding follow-up。

## Limitations

5 首歌、2 个 sample seeds、单听者、描述性 song-block bootstrap、合成 model-tick 扰动、潜在空输出/NA。未做显著性、等价性或总体外推。

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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
