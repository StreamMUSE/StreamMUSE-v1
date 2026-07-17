from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from streammuse.experiments.melody_robustness import (
    build_run_schedule,
    canonical_sha256,
    file_sha256,
    write_canonical_json,
    write_jsonl,
)
from streammuse.experiments.triangle_listening import (
    TRIANGLE_PRACTICE_COUNT,
    TRIANGLE_PRESENTATION_COUNT,
    TRIANGLE_TRIAL_COUNT,
    append_response,
    append_sitting_event,
    create_snapshot,
    progress_summary,
    summarize_unblinded,
    unblind_snapshot,
)


def _campaign_files(fixture, tmp_path):
    manifest = json.loads(fixture.input_manifest.read_text(encoding="utf-8"))
    config = json.loads(
        fixture.campaign_config.read_text(encoding="utf-8")
    )
    config["listening"]["selection_manifest_path"] = str(
        fixture.listening_manifest
    )
    config["listening"]["selection_manifest_sha256"] = file_sha256(
        fixture.listening_manifest
    )
    config_path = tmp_path / "campaign.json"
    write_canonical_json(config_path, config)
    schedule_path = tmp_path / "schedule.jsonl"
    write_jsonl(schedule_path, build_run_schedule(manifest, config))
    return config_path, schedule_path


def _triangle_trial_key(index: int) -> dict:
    question_id = f"Q{index + 1:03d}"
    if index < 60:
        condition = ("pitch", "onset", "both")[index % 3]
        song = f"song-{(index // 3) % 5 + 1}"
        block = "medium_primary"
        semantic_id = f"M:{index + 1:03d}"
        repeat_of = None
    elif index < 70:
        condition = "high"
        song = f"song-{(index - 60) % 5 + 1}"
        block = "high_exploratory"
        semantic_id = f"H:{index + 1:03d}"
        repeat_of = None
    elif index < 75:
        condition = "sham_sampling"
        song = f"song-{index - 69}"
        block = "sham_sampling_baseline"
        semantic_id = f"S:{index + 1:03d}"
        repeat_of = None
    elif index < 81:
        condition = "identity"
        song = f"song-{(index - 75) % 5 + 1}"
        block = "identity_catch"
        semantic_id = f"I:{index + 1:03d}"
        repeat_of = None
    elif index < 87:
        condition = "known_different"
        song = f"song-{(index - 81) % 5 + 1}"
        block = "known_different_control"
        semantic_id = f"K:{index + 1:03d}"
        repeat_of = None
    else:
        original = index - 87
        condition = ("pitch", "onset", "both")[original % 3]
        song = f"song-{(original // 3) % 5 + 1}"
        block = "exact_repeat"
        repeat_of = f"M:{original + 1:03d}"
        semantic_id = f"R:{index + 1:03d}:{repeat_of}"
    identity = block == "identity_catch"
    left_hash = f"{2 * index + 1:064x}"
    right_hash = left_hash if identity else f"{2 * index + 2:064x}"

    def source(digest: str) -> dict:
        return {
            "kind": "formal",
            "song": song,
            "condition": condition,
            "raw_token_payload_sha256": digest,
            "output_event_payload_sha256": digest,
            "source_sha256": digest,
            "excerpt_midi_sha256": digest,
            "canonical_wav_sha256": digest,
            "source_empty": False,
            "operational_valid": True,
        }

    return {
        "question_id": question_id,
        "semantic_id": semantic_id,
        "block": block,
        "condition": condition,
        "repeat_of": repeat_of,
        "repeat_distance": 20 if repeat_of else None,
        "presentation_pattern": "AAA" if identity else "AAB",
        "correct_choice": "no_difference" if identity else "3",
        "odd_position": None if identity else 3,
        "source_a": source(left_hash),
        "source_b": source(right_hash),
        "pair_id": f"C{index + 1:03d}",
        "objective_identity": identity,
        "coverage_driven": False,
    }


def _triangle_report_package(report, tmp_path):
    package = tmp_path / "triangle-listening"
    blind = package / "blind"
    private = package / "private"
    blind.mkdir(parents=True)
    private.mkdir()
    shared_midi = blind / "shared.mid"
    shared_wav = blind / "shared.wav"
    shared_midi.write_bytes(b"test-midi")
    shared_wav.write_bytes(b"test-wav")
    player = blind / "player.html"
    player.write_text("<html>blind triangle</html>", encoding="utf-8")
    public = blind / "public_manifest.json"
    write_canonical_json(
        public,
        {
            "schema_version": "streammuse.melody_robustness.listening_triangle_public.v2",
            "listening_attempt_id": "listening-attempt-001",
            "semantic_fields_present": False,
            "trials": [{"question_id": f"Q{i:03d}"} for i in range(1, 96)],
        },
    )
    selection_path = tmp_path / "triangle-selection.json"
    key_rows = [_triangle_trial_key(index) for index in range(TRIANGLE_TRIAL_COUNT)]
    write_canonical_json(
        selection_path,
        {
            "schema_version": report.TRIANGLE_SELECTION_SCHEMA_VERSION,
            "listening_attempt_id": "listening-attempt-001",
            "listening_attempt_number": 1,
            "trials": [
                {
                    "question_id": row["question_id"],
                    "block": row["block"],
                    "condition": row["condition"],
                }
                for row in key_rows
            ],
        },
    )
    selection_sha = file_sha256(selection_path)
    links = {
        "campaign_config_sha256": "a" * 64,
        "run_schedule_sha256": "b" * 64,
        "campaign_binding_sha256": "c" * 64,
        "selection_sha256": selection_sha,
        "qualification_result_sha256": "d" * 64,
    }
    key_path = private / "private_key.json"
    write_canonical_json(
        key_path,
        {
            "schema_version": report.TRIANGLE_PRIVATE_KEY_SCHEMA_VERSION,
            "listening_attempt_id": "listening-attempt-001",
            **links,
            "trials": key_rows,
            "practice_trials": [{"practice_id": f"P{i:03d}"} for i in range(1, 4)],
            "unblind_only_from_immutable_snapshot": True,
        },
    )
    midi_relative = str(shared_midi.relative_to(package))
    wav_relative = str(shared_wav.relative_to(package))

    def presentations():
        return [
            {
                "position": position,
                "midi": midi_relative,
                "midi_sha256": file_sha256(shared_midi),
                "wav": wav_relative,
                "wav_sha256": file_sha256(shared_wav),
            }
            for position in (1, 2, 3)
        ]

    render_path = package / "render_manifest.json"
    write_canonical_json(
        render_path,
        {
            "schema_version": report.TRIANGLE_RENDER_SCHEMA_VERSION,
            "listening_attempt_id": "listening-attempt-001",
            **links,
            "selection_path": str(selection_path),
            "private_key_sha256": file_sha256(key_path),
            "public_manifest_sha256": file_sha256(public),
            "player_sha256": file_sha256(player),
            "render_bpm": 120,
            "clip_seconds": 8,
            "sample_rate": 44100,
            "bit_depth": 16,
            "gain": 0.5,
            "gain_policy": "common_pair_gain_with_true_peak_protection_only",
            "midi_only_development": False,
            "trials": [
                {"question_id": f"Q{i:03d}", "presentations": presentations()}
                for i in range(1, 96)
            ],
            "practice_trials": [
                {"practice_id": f"P{i:03d}", "presentations": presentations()}
                for i in range(1, 4)
            ],
        },
    )
    write_canonical_json(
        package / "package_audit.json",
        {
            "schema_version": report.TRIANGLE_AUDIT_SCHEMA_VERSION,
            "listening_attempt_id": "listening-attempt-001",
            **links,
            "valid": True,
            "accepted_final": True,
            "blinding_audited": True,
            "errors": [],
            "trial_count": TRIANGLE_TRIAL_COUNT,
            "presentation_count": TRIANGLE_PRESENTATION_COUNT,
            "practice_count": TRIANGLE_PRACTICE_COUNT,
            "private_key_sha256": file_sha256(key_path),
            "render_manifest_sha256": file_sha256(render_path),
        },
    )
    write_canonical_json(blind / "progress_state.json", progress_summary(package))
    return package, links, key_rows


def _answer_triangle_prefix(package, key_rows, count):
    if count:
        append_sitting_event(
            package,
            event="start",
            sitting_id="test-sitting",
            device="pytest-device",
            environment="pytest-room",
            recorded_at="2026-07-16T23:59:59Z",
        )
    for index, key_row in enumerate(key_rows[:count]):
        append_response(
            package,
            odd_choice=key_row["correct_choice"],
            confidence_1_to_5=4,
            sitting_id="test-sitting",
            difference_tags=(),
            note="",
            play_counts=(1, 1, 1),
            response_time_ms=1000 + index,
            submitted_at=f"2026-07-17T00:{index // 60:02d}:{index % 60:02d}Z",
        )
    if count == 0:
        return None
    snapshot, _sealed = create_snapshot(package)
    unblind_snapshot(package, snapshot)
    summarize_unblinded(package, snapshot)
    return snapshot


def _retry_attempt_lineage_fixture(report, tmp_path):
    failed_package, _links, key_rows = _triangle_report_package(report, tmp_path)
    append_sitting_event(
        failed_package,
        event="start",
        sitting_id="failed-attempt-sitting",
        device="pytest-device",
        environment="pytest-room",
        recorded_at="2026-07-16T23:59:59Z",
    )
    for index, key_row in enumerate(key_rows):
        choice = key_row["correct_choice"]
        if key_row["block"] == "identity_catch":
            choice = "1"
        elif key_row["block"] == "known_different_control":
            choice = "no_difference"
        append_response(
            failed_package,
            odd_choice=choice,
            confidence_1_to_5=4,
            sitting_id="failed-attempt-sitting",
            play_counts=(1, 1, 1),
            response_time_ms=1000 + index,
            submitted_at=f"2026-07-17T00:{index // 60:02d}:{index % 60:02d}Z",
        )
    snapshot, sealed = create_snapshot(failed_package)
    unblind_snapshot(failed_package, snapshot)
    _summary_path, summary = summarize_unblinded(failed_package, snapshot)
    assert summary["qc_status"] == "fail"
    assert summary["retry_required"] is True

    previous_selection_path = Path(
        json.loads(
            (failed_package / "render_manifest.json").read_text(encoding="utf-8")
        )["selection_path"]
    ).resolve()
    audit_path = failed_package / "package_audit.json"
    render_path = failed_package / "render_manifest.json"
    private_path = failed_package / "private" / "private_key.json"
    ledger_path = failed_package / "blind" / "response_ledger.jsonl"
    sitting_ledger_path = failed_package / "blind" / "sitting_ledger.jsonl"
    state_path = failed_package / "unblind_state.json"
    unblinded_path = snapshot / "partial_unblinded_scores.json"
    summary_path = snapshot / "partial_discrimination_summary.json"
    authorization = {
        "schema_version": report.TRIANGLE_RETRY_AUTHORIZATION_SCHEMA_VERSION,
        "created_at": "2026-07-18T00:00:00Z",
        "authorization_reason": "qc_failure",
        "previous_attempt_id": "listening-attempt-001",
        "previous_attempt_number": 1,
        "next_attempt_id": "listening-attempt-002",
        "next_attempt_number": 2,
        "base_blind_order_seed": 7,
        "effective_blind_order_seed": 42,
        "effective_blind_order_seed_sha256": "e" * 64,
        "failed_package_path": str(failed_package),
        "base_selection_path": str(previous_selection_path),
        "base_selection_sha256": file_sha256(previous_selection_path),
        "failed_selection_path": str(previous_selection_path),
        "failed_selection_sha256": file_sha256(previous_selection_path),
        "failed_package_audit_path": str(audit_path),
        "failed_package_audit_sha256": file_sha256(audit_path),
        "failed_render_manifest_path": str(render_path),
        "failed_render_manifest_sha256": file_sha256(render_path),
        "failed_private_key_path": str(private_path),
        "failed_private_key_sha256": file_sha256(private_path),
        "failed_response_ledger_path": str(ledger_path),
        "failed_response_ledger_sha256": file_sha256(ledger_path),
        "failed_sitting_ledger_path": str(sitting_ledger_path),
        "failed_sitting_ledger_sha256": file_sha256(sitting_ledger_path),
        "failed_sitting_ledger_head_hash": sealed["sitting_ledger_head_hash"],
        "failed_snapshot_path": str(snapshot),
        "failed_snapshot_id": snapshot.name,
        "failed_sealed_responses_sha256": file_sha256(
            snapshot / "sealed_responses.json"
        ),
        "failed_ledger_head_hash": sealed["ledger_head_hash"],
        "failed_answered_count": 95,
        "failed_unblind_state_path": str(state_path),
        "failed_unblind_state_sha256": file_sha256(state_path),
        "failed_unblinded_scores_path": str(unblinded_path),
        "failed_unblinded_scores_sha256": file_sha256(unblinded_path),
        "failed_summary_path": str(summary_path),
        "failed_summary_sha256": file_sha256(summary_path),
        "previous_qc_status": "fail",
        "previous_retry_required": True,
    }
    authorization_path = (
        failed_package
        / "retry_authorizations"
        / "authorize-listening-attempt-002.json"
    )
    write_canonical_json(authorization_path, authorization)
    lineage = {
        "schema_version": report.TRIANGLE_RETRY_LINEAGE_SCHEMA_VERSION,
        "authorization_reason": "qc_failure",
        "current_attempt_id": "listening-attempt-002",
        "current_attempt_number": 2,
        "previous_attempt_id": "listening-attempt-001",
        "previous_attempt_number": 1,
        "base_blind_order_seed": 7,
        "effective_blind_order_seed": 42,
        "effective_blind_order_seed_sha256": "e" * 64,
        "base_selection_path": str(previous_selection_path),
        "base_selection_sha256": file_sha256(previous_selection_path),
        "previous_package_path": str(failed_package),
        "previous_selection_path": str(previous_selection_path),
        "previous_selection_sha256": file_sha256(previous_selection_path),
        "previous_package_audit_path": str(audit_path),
        "previous_package_audit_sha256": file_sha256(audit_path),
        "failed_snapshot_path": str(snapshot),
        "failed_snapshot_id": snapshot.name,
        "failed_sealed_responses_sha256": file_sha256(
            snapshot / "sealed_responses.json"
        ),
        "failed_ledger_head_hash": sealed["ledger_head_hash"],
        "failed_response_ledger_path": str(ledger_path),
        "failed_response_ledger_sha256": file_sha256(ledger_path),
        "failed_sitting_ledger_path": str(sitting_ledger_path),
        "failed_sitting_ledger_sha256": file_sha256(sitting_ledger_path),
        "failed_sitting_ledger_head_hash": sealed["sitting_ledger_head_hash"],
        "failed_unblind_state_path": str(state_path),
        "failed_unblind_state_sha256": file_sha256(state_path),
        "failed_unblinded_scores_path": str(unblinded_path),
        "failed_unblinded_scores_sha256": file_sha256(unblinded_path),
        "failed_summary_path": str(summary_path),
        "failed_summary_sha256": file_sha256(summary_path),
        "previous_qc_status": "fail",
        "previous_retry_required": True,
        "retry_authorization_path": str(authorization_path),
        "retry_authorization_sha256": file_sha256(authorization_path),
    }
    current_package = tmp_path / "triangle-listening-retry"
    current_package.mkdir()
    current_selection_path = tmp_path / "triangle-selection-retry.json"
    current_selection = {
        "schema_version": report.TRIANGLE_SELECTION_SCHEMA_VERSION,
        "listening_attempt_id": "listening-attempt-002",
        "listening_attempt_number": 2,
        "retry_reblind_after_formal_without_semantic_change": True,
        "base_blind_order_seed": 7,
        "effective_blind_order_seed": 42,
        "effective_blind_order_seed_sha256": "e" * 64,
        "retry_lineage": lineage,
        "retry_lineage_sha256": canonical_sha256(lineage),
        "trials": [],
    }
    write_canonical_json(current_selection_path, current_selection)
    return {
        "failed_package": failed_package,
        "current_package": current_package,
        "current_selection_path": current_selection_path,
        "current_selection": current_selection,
        "authorization_path": authorization_path,
        "authorization": authorization,
    }


def _campaign_audit_reverify_fixture(report, tmp_path, monkeypatch):
    campaign = tmp_path / "formal"
    attempt = campaign / "runs" / "run-1" / "attempt-001"
    attempt.mkdir(parents=True)
    artifact = attempt / "theoretical_model.mid"
    artifact.write_bytes(b"verified-generated-acc")
    artifact_record = {
        "path": artifact.name,
        "size": artifact.stat().st_size,
        "sha256": file_sha256(artifact),
    }
    gate_path = attempt / "rt_artifact_gate.json"
    write_canonical_json(
        gate_path,
        {
            "source_empty": False,
            "required_artifacts": {"theoretical_model_midi": artifact_record},
        },
    )
    selector = {
        "kind": "formal",
        "formal_pipeline": "rt",
        "source_artifact": "theoretical_model",
        "presentation": "acc_solo",
        "song": "song-1",
        "condition": "pitch",
        "perturb_seed": 11,
        "sample_seed": 22,
    }
    readiness_selector = {
        field: value for field, value in selector.items() if field != "kind"
    }
    schedule = [
        {
            "run_id": "run-1",
            "pipeline": "rt",
            "song": "song-1",
            "condition": "pitch",
            "perturb_seed": 11,
            "sample_seed": 22,
        }
    ]
    verdict = {
        "attempt_id": "attempt-001",
        "content_valid": True,
        "operational_valid": True,
        "validity": {"driver_artifact_gate": {"source_empty": False}},
    }
    monkeypatch.setattr(
        report,
        "verify_attempt_verdict",
        lambda *_args, **_kwargs: (
            attempt,
            verdict,
            {artifact.resolve(), gate_path.resolve()},
        ),
    )
    readiness = {
        "schema_version": "streammuse.melody_robustness.listening_source_readiness.v1",
        "selection_schema_version": report.TRIANGLE_SELECTION_SCHEMA_VERSION,
        "expected_unique_sources": 1,
        "ready_sources": 1,
        "not_ready_sources": 0,
        "ready": True,
        "sources": [
            {
                "selector": readiness_selector,
                "ready": True,
                "run_id": "run-1",
                "attempt_id": "attempt-001",
                "operational_valid": True,
                "source_empty": False,
                "artifact": artifact_record,
            }
        ],
    }
    audit = {
        "expected": 1,
        "present": 1,
        "content_valid": 1,
        "missing": 0,
        "invalid": 0,
        "retried": 0,
        "operational_invalid": 0,
        "source_empty": 0,
        "extra_run_ids": [],
        "listening_source_readiness": readiness,
        "runs": [
            {
                "run_id": "run-1",
                "pipeline": "rt",
                "status": "valid",
                "operational_valid": True,
                "source_empty": False,
                "attempts": 1,
            }
        ],
    }
    selection = {
        "schema_version": report.TRIANGLE_SELECTION_SCHEMA_VERSION,
        "trials": [{"sources": {"a": selector, "b": selector}}],
    }
    return campaign, audit, schedule, selection


def _effect_rows(*, songs: list[str], rows_per_song: int = 4):
    return [
        {
            "song": song,
            "pipeline": "rt_theoretical",
            "condition": "both",
            "intended_fidelity_effect": float(song_index) / 10 + replicate / 100,
        }
        for song_index, song in enumerate(songs, start=1)
        for replicate in range(rows_per_song)
    ]


def test_song_effects_preserve_missing_and_one_sided_na_blocks(load_script):
    analyzer = load_script("analyze_perturbation_robustness")
    songs = [f"song-{index}" for index in range(1, 6)]
    rows = _effect_rows(songs=songs[:4])
    # A single unavailable pair invalidates the whole equal-weight song block;
    # it must not be silently averaged over the remaining three values.
    rows[-1]["intended_fidelity_effect"] = None

    result = analyzer._song_effects(
        rows,
        pipeline="rt_theoretical",
        condition="both",
        songs=songs,
    )

    assert set(result) == set(songs)
    assert result["song-1"] is not None
    assert result["song-4"] is None
    assert result["song-5"] is None


def test_factorial_helper_emits_explicit_na_when_any_arm_is_missing(load_script):
    analyzer = load_script("analyze_perturbation_robustness")
    base = {
        "song": "song-1",
        "pipeline": "offline",
        "sample_seed": 11,
        "perturb_seed": 22,
    }
    complete = [
        {**base, "condition": "both", "intended_fidelity_effect": 0.7},
        {**base, "condition": "pitch", "intended_fidelity_effect": 0.2},
        {**base, "condition": "onset", "intended_fidelity_effect": 0.3},
    ]
    incomplete_base = {**base, "song": "song-2"}
    incomplete = [
        {**incomplete_base, "condition": "both", "intended_fidelity_effect": 0.7},
        {**incomplete_base, "condition": "pitch", "intended_fidelity_effect": None},
    ]

    rows = analyzer._factorial_effects(complete + incomplete)
    indexed = {row["song"]: row for row in rows}

    assert indexed["song-1"]["factorial_interaction"] == pytest.approx(0.2)
    assert indexed["song-1"]["valid"]
    assert indexed["song-2"]["factorial_interaction"] is None
    assert not indexed["song-2"]["valid"]


def _metric_row(
    *,
    run_id: str,
    pipeline: str,
    condition: str,
    sample_seed: int,
    intended: float,
    actual: float,
    perturb_seed: int | None = None,
    coverage: float = 1.0,
):
    return {
        "run_id": run_id,
        "pipeline": pipeline,
        "song": "song-1",
        "condition": condition,
        "perturb_seed": perturb_seed,
        "sample_seed": sample_seed,
        "content_valid": True,
        "operational_valid": True,
        "quality": {
            "D_intended": {"D_micro": intended},
            "D_actual": {"D_micro": actual},
        },
        "coverage": {
            "onsets_per_beat": coverage,
            "active_pitch_ticks_per_beat": coverage * 2,
            "empty_beat_ratio": 1.0 - min(coverage, 1.0),
        },
    }


def test_paired_rows_expose_D_actual_joint_treatment_separately_from_adaptation(
    load_script,
):
    analyzer = load_script("analyze_perturbation_robustness")
    sham = _metric_row(
        run_id="sham", pipeline="offline", condition="sham",
        sample_seed=11, intended=0.1, actual=0.2,
    )
    treated = _metric_row(
        run_id="both", pipeline="offline", condition="both",
        sample_seed=11, perturb_seed=22, intended=0.4, actual=0.55,
    )
    roll = analyzer.Roll(
        end_tick=4,
        sustain=frozenset({(0, 60), (1, 60)}),
        onsets=frozenset({(0, 60)}),
    )
    rolls = {
        ("sham", "offline"): {"clean": roll, "actual": roll, "acc": roll},
        ("both", "offline"): {"clean": roll, "actual": roll, "acc": roll},
    }

    pair = analyzer._paired_rows([sham, treated], rolls, "a" * 64)[0]

    assert pair["D_actual_cond"] == pytest.approx(0.55)
    assert pair["D_actual_sham"] == pytest.approx(0.2)
    assert pair["D_actual_cond_minus_sham"] == pytest.approx(0.35)
    assert pair["D_actual_joint_treatment_na_reason"] is None
    assert "adaptation_effect" in pair


def test_operational_endpoints_keep_raw_condition_delta_and_sham_interaction(
    load_script,
):
    analyzer = load_script("analyze_perturbation_robustness")
    metrics = []
    specifications = [
        ("sham-1", "sham", None, 1, 0.05),
        ("sham-2", "sham", None, 2, 0.05),
        ("both-11", "both", 10, 1, 0.25),
        ("both-12", "both", 10, 2, 0.25),
        ("both-21", "both", 20, 1, 0.25),
        ("both-22", "both", 20, 2, 0.25),
    ]
    for run_id, condition, perturb_seed, sample_seed, delta in specifications:
        metrics.extend(
            [
                _metric_row(
                    run_id=run_id, pipeline="rt_theoretical",
                    condition=condition, perturb_seed=perturb_seed,
                    sample_seed=sample_seed, intended=0.1, actual=0.2,
                    coverage=1.0,
                ),
                _metric_row(
                    run_id=run_id, pipeline="rt_combined",
                    condition=condition, perturb_seed=perturb_seed,
                    sample_seed=sample_seed, intended=0.1 + delta,
                    actual=0.2 + delta, coverage=1.0 + delta,
                ),
            ]
        )

    raw = analyzer._operational_rows(metrics, "a" * 64)
    treatment = analyzer._paired_operational_rows(raw, "a" * 64)
    summary = analyzer._operational_endpoint_summary(
        raw,
        treatment,
        expected_songs={"song-1"},
        seed=123,
        config_sha="a" * 64,
    )

    raw_both = summary["raw_by_condition"]["both"]["D_intended"]
    interaction = summary["treatment_interaction_vs_sham"]["both"]["D_intended"]
    assert raw_both["per_song"]["song-1"] == pytest.approx(0.25)
    assert raw_both["bootstrap"]["estimate"] == pytest.approx(0.25)
    assert interaction["per_song"]["song-1"] == pytest.approx(0.20)
    assert interaction["bootstrap"]["estimate"] == pytest.approx(0.20)
    assert "not sham-adjusted" in summary["semantics"]["raw_operational_delta"]


def test_coverage_summary_reports_absolute_and_paired_song_block_bootstraps(
    load_script,
):
    analyzer = load_script("analyze_perturbation_robustness")
    metrics = [
        _metric_row(
            run_id=f"sham-{sample_seed}", pipeline="rt_theoretical",
            condition="sham", sample_seed=sample_seed, intended=0.1,
            actual=0.1, coverage=1.0,
        )
        for sample_seed in (1, 2)
    ] + [
        _metric_row(
            run_id=f"both-{perturb_seed}-{sample_seed}",
            pipeline="rt_theoretical", condition="both",
            perturb_seed=perturb_seed, sample_seed=sample_seed,
            intended=0.1, actual=0.1, coverage=1.5,
        )
        for perturb_seed in (10, 20)
        for sample_seed in (1, 2)
    ]
    pairs = [
        {
            "song": "song-1", "pipeline": "rt_theoretical",
            "condition": "both", "perturb_seed": perturb_seed,
            "sample_seed": sample_seed,
            "coverage_delta_onsets_per_beat": 0.5,
            "coverage_delta_active_pitch_ticks_per_beat": 1.0,
            "coverage_delta_empty_beat_ratio": 0.0,
        }
        for perturb_seed in (10, 20)
        for sample_seed in (1, 2)
    ]

    summary = analyzer._coverage_summary(
        metrics, pairs, expected_songs={"song-1"}, seed=123,
        config_sha="a" * 64,
    )

    absolute = summary["absolute"]["rt_theoretical"]["both"]["onsets_per_beat"]
    paired = summary["paired_delta_vs_sham"]["rt_theoretical"]["both"][
        "onsets_per_beat"
    ]
    assert absolute["per_song"] == {"song-1": 1.5}
    assert absolute["bootstrap"]["estimate"] == pytest.approx(1.5)
    assert paired["per_song"] == {"song-1": 0.5}
    assert paired["bootstrap"]["estimate"] == pytest.approx(0.5)


def test_lifecycle_aggregation_counts_structured_failures_and_never_zero_fills_missing(
    load_script, tmp_path
):
    analyzer = load_script("analyze_perturbation_robustness")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    lifecycle = [
        {"event": "expected"}, {"event": "enqueued"},
        {"event": "started"}, {"event": "succeeded"},
        {"event": "processed"},
        {"event": "expected"}, {"event": "enqueued"},
        {"event": "stale_dropped"},
    ]
    write_jsonl(attempt / "request_lifecycle.jsonl", lifecycle)
    requests = [
        {
            "request_id": "r1", "expected": True, "enqueued": True,
            "started": True, "succeeded": True, "failed": False,
            "processed": True, "stale_dropped": False, "empty_success": False,
        },
        {
            "request_id": "r2", "expected": True, "enqueued": True,
            "started": False, "succeeded": False, "failed": False,
            "processed": False, "stale_dropped": True, "empty_success": False,
        },
    ]
    content = {
        "valid": False, "request_count": 2, "analysis_request_coverage": 0.5,
        "expected_request_ids": ["r1", "r2"],
        "succeeded_request_ids": ["r1"], "processed_request_ids": ["r1"],
        "failed_request_ids": [], "stale_dropped_request_ids": ["r2"],
        "metadata_invalid_request_ids": [], "pending_at_stop_request_ids": [],
        "missing_generation_start_ticks": [4],
        "unexpected_generation_start_ticks": [],
        "duplicate_generation_start_ticks": [],
        "rejected_generation_start_ticks": [],
    }
    operational = {
        "valid": False, "stale_request_drops": 1, "late_events": 2,
        "clamped_onsets": 3, "dropped_model_events": 4,
        "orphan_note_offs": 5, "forced_note_offs": 6,
        "max_lateness_ticks": 7,
    }
    verdict = {
        "attempt_id": "attempt-1",
        "validity": {
            "content": content, "operational": operational,
            "requests": requests, "drain": {"timed_out": False},
        },
    }
    schedule_row = {
        "run_id": "run-1", "song": "song-1", "condition": "both",
        "perturb_seed": 10, "sample_seed": 1,
    }

    row = analyzer._lifecycle_row(
        schedule_row, attempt, verdict, "a" * 64
    )
    assert row["schema_valid"]
    assert row["counts"]["stale_request_drops"] == 1
    assert row["counts"]["late_events"] == 2
    assert row["counts"]["clamped_onsets"] == 3
    assert row["counts"]["dropped_model_events"] == 4
    assert row["counts"]["forced_note_offs"] == 6
    assert row["max_lateness_ticks"] == 7

    missing = {
        **schedule_row, "run_id": "run-2", "schema_valid": False,
        "content_valid": False, "operational_valid": False,
        "counts": None, "max_lateness_ticks": None,
        "analysis_request_coverage": None,
    }
    aggregate = analyzer._aggregate_lifecycle_group([row, missing])
    assert not aggregate["complete"]
    assert aggregate["all_run_totals"] is None
    assert aggregate["observed_schema_valid_run_totals"]["late_events"] == 2
    assert aggregate["invalid_run_ids"] == ["run-2"]


def test_report_effect_formatter_labels_na_and_descriptive_values(load_script):
    report = load_script("build_robustness_report")

    na = report._effect(
        {
            "primary": {
                "estimate": None,
                "interval": None,
                "valid_song_count": 3,
                "raw_song_effects": {"song-1": 0.1, "song-2": None},
            }
        },
        "primary",
    )
    valid = report._effect(
        {
            "primary": {
                "estimate": 0.125,
                "interval": [0.025, 0.225],
                "valid_song_count": 5,
                "raw_song_effects": {"song-1": 0.1},
            }
        },
        "primary",
    )

    assert "NA" in na
    assert "有效 song block=3" in na
    assert "0.1250" in valid
    assert "[0.0250, 0.2250]" in valid


def test_analyzer_binds_config_manifest_and_exact_deterministic_schedule(
    load_script, robustness_fixture, tmp_path
):
    analyzer = load_script("analyze_perturbation_robustness")
    config_path, schedule_path = _campaign_files(robustness_fixture, tmp_path)

    loaded = analyzer._validated_campaign_inputs(
        config_path,
        schedule_path,
        expected_config_sha256=file_sha256(config_path),
        expected_schedule_sha256=file_sha256(schedule_path),
        allow_unpinned_schedule=False,
    )
    assert len(loaded[4]) == 160

    with pytest.raises(RuntimeError, match="run schedule SHA-256 is required"):
        analyzer._validated_campaign_inputs(
            config_path,
            schedule_path,
            expected_config_sha256=file_sha256(config_path),
            expected_schedule_sha256=None,
            allow_unpinned_schedule=False,
        )
    development = analyzer._validated_campaign_inputs(
        config_path,
        schedule_path,
        expected_config_sha256=file_sha256(config_path),
        expected_schedule_sha256=None,
        allow_unpinned_schedule=True,
    )
    assert development[5] == file_sha256(schedule_path)

    with pytest.raises(RuntimeError, match="run schedule hash mismatch"):
        analyzer._validated_campaign_inputs(
            config_path,
            schedule_path,
            expected_config_sha256=file_sha256(config_path),
            expected_schedule_sha256="0" * 64,
            allow_unpinned_schedule=False,
        )


def test_analyzer_rejects_rehashed_but_noncanonical_schedule(
    load_script, robustness_fixture, tmp_path
):
    analyzer = load_script("analyze_perturbation_robustness")
    config_path, schedule_path = _campaign_files(robustness_fixture, tmp_path)
    rows = analyzer.read_jsonl(schedule_path)
    write_jsonl(schedule_path, list(reversed(rows)))

    with pytest.raises(RuntimeError, match="not the deterministic schedule"):
        analyzer._validated_campaign_inputs(
            config_path,
            schedule_path,
            expected_config_sha256=file_sha256(config_path),
            expected_schedule_sha256=file_sha256(schedule_path),
            allow_unpinned_schedule=False,
        )


def test_analyzer_rejects_input_manifest_drift_even_when_schedule_is_unchanged(
    load_script, robustness_fixture, tmp_path
):
    analyzer = load_script("analyze_perturbation_robustness")
    config_path, schedule_path = _campaign_files(robustness_fixture, tmp_path)
    with robustness_fixture.input_manifest.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(RuntimeError, match="input manifest hash mismatch"):
        analyzer._validated_campaign_inputs(
            config_path,
            schedule_path,
            expected_config_sha256=file_sha256(config_path),
            expected_schedule_sha256=file_sha256(schedule_path),
            allow_unpinned_schedule=False,
        )


def test_analyzer_rejects_qualification_result_drift(
    load_script, robustness_fixture, tmp_path
):
    analyzer = load_script("analyze_perturbation_robustness")
    config_path, schedule_path = _campaign_files(robustness_fixture, tmp_path)
    with robustness_fixture.qualification_result.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="qualification_result hash mismatch"):
        analyzer._validated_campaign_inputs(
            config_path,
            schedule_path,
            expected_config_sha256=file_sha256(config_path),
            expected_schedule_sha256=file_sha256(schedule_path),
            allow_unpinned_schedule=False,
        )


def test_report_requires_hash_pinned_deterministic_campaign_inputs(
    load_script, robustness_fixture, tmp_path
):
    report = load_script("build_robustness_report")
    config_path, schedule_path = _campaign_files(robustness_fixture, tmp_path)

    loaded = report._validated_campaign_inputs(
        config_path,
        schedule_path,
        expected_config_sha256=file_sha256(config_path),
        expected_schedule_sha256=file_sha256(schedule_path),
    )
    assert len(loaded[-1]) == 160

    with pytest.raises(RuntimeError, match="campaign config hash mismatch"):
        report._validated_campaign_inputs(
            config_path,
            schedule_path,
            expected_config_sha256="0" * 64,
            expected_schedule_sha256=file_sha256(schedule_path),
        )


def test_report_rejects_cross_campaign_analysis_even_if_index_hash_is_updated(
    load_script, tmp_path
):
    report = load_script("build_robustness_report")
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    config_sha = "a" * 64
    other_sha = "b" * 64
    write_jsonl(
        analysis / "run_metrics.jsonl",
        [{"campaign_config_sha256": config_sha, "run_id": "run-1"}],
    )
    write_jsonl(
        analysis / "paired_contrasts.jsonl",
        [{"campaign_config_sha256": config_sha, "run_id": "run-1"}],
    )
    write_canonical_json(
        analysis / "bootstrap.json",
        {"campaign_config_sha256": other_sha, "results": {}},
    )
    write_canonical_json(
        analysis / "control_report.json",
        {"campaign_config_sha256": config_sha, "songs": []},
    )
    artifacts = {
        name: file_sha256(analysis / name)
        for name in (
            "run_metrics.jsonl",
            "paired_contrasts.jsonl",
            "bootstrap.json",
            "control_report.json",
        )
    }
    write_canonical_json(
        analysis / "analysis_index.json",
        {
            "campaign_config_sha256": config_sha,
            "input_manifest_sha256": "c" * 64,
            "run_schedule_sha256": "d" * 64,
            "campaign_binding_sha256": "e" * 64,
            "qualification_result_sha256": "f" * 64,
            "artifacts": artifacts,
        },
    )

    with pytest.raises(RuntimeError, match="bootstrap campaign config hash mismatch"):
        report._validated_analysis_artifacts(
            analysis,
            config_sha,
            input_manifest_sha="c" * 64,
            schedule_sha="d" * 64,
            campaign_binding_sha="e" * 64,
            qualification_result_sha="f" * 64,
        )


def test_report_keeps_valid_package_incomplete_until_scores_are_sealed_and_unblinded(
    load_script, tmp_path
):
    report = load_script("build_robustness_report")
    config_sha = "a" * 64
    schedule_sha = "b" * 64
    binding_sha = "d" * 64
    selection_sha = "c" * 64
    qualification_sha = "f" * 64
    package = tmp_path / "listening"
    package.mkdir()
    write_canonical_json(
        package / "private_key.json",
        {
            "clips": [{"sample_id": f"S{index:03d}"} for index in range(1, 25)]
        },
    )
    write_canonical_json(package / "render_manifest.json", {"clips": []})
    write_canonical_json(
        package / "package_audit.json",
        {
            "valid": True,
            "accepted_final": True,
            "campaign_config_sha256": config_sha,
            "run_schedule_sha256": schedule_sha,
            "campaign_binding_sha256": binding_sha,
            "selection_sha256": selection_sha,
            "qualification_result_sha256": qualification_sha,
            "private_key_sha256": file_sha256(package / "private_key.json"),
            "render_manifest_sha256": file_sha256(package / "render_manifest.json"),
        },
    )

    audit, sealed, unblinded = report._listening_completion(
        package,
        config_sha=config_sha,
        schedule_sha=schedule_sha,
        campaign_binding_sha=binding_sha,
        selection_sha=selection_sha,
        qualification_result_sha=qualification_sha,
    )

    assert audit["accepted_final"]
    assert not sealed
    assert not unblinded


@pytest.mark.parametrize(
    ("answered", "expected_collection", "expected_qc"),
    [
        (0, "not_started", "not_started"),
        (1, "partial", "pending"),
        (94, "partial", "pending"),
        (95, "full", "pass"),
    ],
)
def test_report_accepts_flexible_triangle_collection_horizons(
    load_script, tmp_path, monkeypatch, answered, expected_collection, expected_qc
):
    report = load_script("build_robustness_report")
    package, links, key_rows = _triangle_report_package(report, tmp_path)
    snapshot = _answer_triangle_prefix(package, key_rows, answered)
    if answered:
        monkeypatch.setattr(
            report,
            "_validate_generated_acc_export",
            lambda _package, _key: {"valid": True, "row_count": 1},
        )

    state = report._triangle_listening_state(
        package,
        snapshot_path=snapshot,
        config_sha=links["campaign_config_sha256"],
        schedule_sha=links["run_schedule_sha256"],
        campaign_binding_sha=links["campaign_binding_sha256"],
        selection_sha=links["selection_sha256"],
        qualification_result_sha=links["qualification_result_sha256"],
    )

    assert state["collection_status"] == expected_collection
    assert state["answered_count"] == answered
    assert state["pending_count"] == TRIANGLE_TRIAL_COUNT - answered
    assert state["qc_status"] == expected_qc
    assert state["identity_counts"]["raw_token_payload_comparable"] == 95
    assert state["identity_counts"]["output_event_payload_comparable"] == 95
    assert state["identity_counts"]["canonical_midi_comparable"] == 95
    assert state["identity_counts"]["rendered_wav_comparable"] == 95
    assert state["snapshot_status"] == (
        "not_applicable" if answered == 0 else "valid"
    )
    if answered == 0:
        assert state["summary"] is None
        assert state["semantic_result_status"] == "not_available"
    else:
        assert state["semantic_result_status"] == "valid"
        assert state["summary"]["answered_count"] == answered
    if answered == TRIANGLE_TRIAL_COUNT:
        assert all(
            result["decision"]
            == "confirmed discriminable in this fixed listener/package"
            for result in state["summary"]["conditions"].values()
        )


def test_report_marks_rows_after_partial_unblind_exploratory(
    load_script, tmp_path, monkeypatch
):
    report = load_script("build_robustness_report")
    package, links, key_rows = _triangle_report_package(report, tmp_path)
    first = _answer_triangle_prefix(package, key_rows, 1)
    assert first is not None
    append_sitting_event(
        package,
        event="start",
        sitting_id="later-sitting",
        device="pytest-device",
        environment="pytest-room",
        recorded_at="2026-07-17T23:59:59Z",
    )
    append_response(
        package,
        odd_choice=key_rows[1]["correct_choice"],
        confidence_1_to_5=3,
        sitting_id="later-sitting",
        play_counts=(1, 1, 1),
        response_time_ms=900,
        submitted_at="2026-07-18T00:00:00Z",
    )
    second, _sealed = create_snapshot(package)
    unblind_snapshot(package, second)
    summarize_unblinded(package, second)
    monkeypatch.setattr(
        report,
        "_validate_generated_acc_export",
        lambda _package, _key: {"valid": True, "row_count": 1},
    )

    state = report._triangle_listening_state(
        package,
        snapshot_path=second,
        config_sha=links["campaign_config_sha256"],
        schedule_sha=links["run_schedule_sha256"],
        campaign_binding_sha=links["campaign_binding_sha256"],
        selection_sha=links["selection_sha256"],
        qualification_result_sha=links["qualification_result_sha256"],
    )

    assert state["blinding_status"] == "partially_unblinded_during_collection"
    assert state["summary"]["views"]["pre_unblind"]["answered"] == 1
    assert (
        state["summary"]["views"]["post_unblind_exploratory"]["answered"] == 1
    )
    pitch = state["summary"]["conditions"]["pitch"]
    onset = state["summary"]["conditions"]["onset"]
    assert pitch["views"]["pre_unblind"]["answered"] == 1
    assert pitch["views"]["post_unblind_exploratory"]["answered"] == 0
    assert onset["views"]["pre_unblind"]["answered"] == 0
    assert onset["views"]["post_unblind_exploratory"]["answered"] == 1
    pitch_song = next(iter(pitch["per_song"].values()))
    onset_song = next(iter(onset["per_song"].values()))
    assert pitch_song["views"]["pre_unblind"]["answered"] == 1
    assert onset_song["views"]["post_unblind_exploratory"]["answered"] == 1
    assert all(
        result["decision"] == "partial — preregistered decision pending"
        for result in state["summary"]["conditions"].values()
    )


def test_report_rejects_tampered_partial_triangle_summary(load_script, tmp_path):
    report = load_script("build_robustness_report")
    package, links, key_rows = _triangle_report_package(report, tmp_path)
    snapshot = _answer_triangle_prefix(package, key_rows, 1)
    summary_path = snapshot / "partial_discrimination_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["conditions"]["pitch"]["decision"] = (
        "confirmed discriminable in this fixed listener/package"
    )
    write_canonical_json(summary_path, summary)

    with pytest.raises(ValueError, match="exact score derivation"):
        report._triangle_listening_state(
            package,
            snapshot_path=snapshot,
            config_sha=links["campaign_config_sha256"],
            schedule_sha=links["run_schedule_sha256"],
            campaign_binding_sha=links["campaign_binding_sha256"],
            selection_sha=links["selection_sha256"],
            qualification_result_sha=links["qualification_result_sha256"],
        )


def test_report_requires_generated_acc_midi_and_wav_after_unblind(
    load_script, tmp_path
):
    report = load_script("build_robustness_report")
    package, links, key_rows = _triangle_report_package(report, tmp_path)
    snapshot = _answer_triangle_prefix(package, key_rows, 1)

    with pytest.raises(RuntimeError, match="requires a complete hash-bound"):
        report._triangle_listening_state(
            package,
            snapshot_path=snapshot,
            config_sha=links["campaign_config_sha256"],
            schedule_sha=links["run_schedule_sha256"],
            campaign_binding_sha=links["campaign_binding_sha256"],
            selection_sha=links["selection_sha256"],
            qualification_result_sha=links["qualification_result_sha256"],
        )


def test_report_binds_generated_acc_export_to_private_key(load_script, tmp_path):
    report = load_script("build_robustness_report")
    formal = tmp_path / "formal.mid"
    formal.write_bytes(b"formal-generated-acc")
    export_root = tmp_path / "package" / "generated_acc_after_unblind"
    midi_root = export_root / "midi"
    wav_root = export_root / "wav_8s"
    midi_root.mkdir(parents=True)
    wav_root.mkdir()
    stem = "song-1__pitch__p-11__s-22"
    exported_midi = midi_root / f"{stem}.mid"
    exported_wav = wav_root / f"{stem}.wav"
    canonical_wav = tmp_path / "canonical.wav"
    exported_midi.write_bytes(formal.read_bytes())
    canonical_wav.write_bytes(b"rendered-wav")
    exported_wav.write_bytes(canonical_wav.read_bytes())
    source = {
        "kind": "formal",
        "song": "song-1",
        "condition": "pitch",
        "perturb_seed": 11,
        "sample_seed": 22,
        "run_id": "run-1",
        "attempt_id": "attempt-001",
        "operational_valid": True,
        "source_empty": False,
        "source_path": str(formal),
        "source_sha256": file_sha256(formal),
        "canonical_wav_path": str(canonical_wav),
        "canonical_wav_sha256": file_sha256(canonical_wav),
        "raw_token_payload_sha256": "a" * 64,
        "output_event_payload_sha256": "b" * 64,
    }
    row = {
        "song": source["song"],
        "condition": source["condition"],
        "perturb_seed": source["perturb_seed"],
        "sample_seed": source["sample_seed"],
        "run_id": source["run_id"],
        "attempt_id": source["attempt_id"],
        "operational_valid": True,
        "source_empty": False,
        "raw_token_payload_sha256": source["raw_token_payload_sha256"],
        "output_event_payload_sha256": source["output_event_payload_sha256"],
        "formal_theoretical_midi": str(formal),
        "formal_theoretical_midi_sha256": file_sha256(formal),
        "exported_midi": str(exported_midi),
        "exported_midi_sha256": file_sha256(exported_midi),
        "exported_excerpt_wav": str(exported_wav),
        "exported_excerpt_wav_sha256": file_sha256(exported_wav),
        "post_unblinding_qualitative_followup_only": True,
    }
    write_canonical_json(export_root / "generated_acc_index.json", {"rows": [row]})
    with (export_root / "generated_acc_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    key = {"trials": [{"source_a": source, "source_b": source}]}

    result = report._validate_generated_acc_export(
        tmp_path / "package", key, require_snapshot_binding=False
    )
    assert result["valid"]
    assert result["row_count"] == 1
    report_output = tmp_path / "report-output"
    copied = report._copy_generated_acc_export_to_report(
        tmp_path / "package", report_output, result
    )
    assert copied["valid"]
    assert copied["file_count"] == 4
    assert (
        report_output
        / "generated_acc_after_unblind"
        / "midi"
        / exported_midi.name
    ).is_file()
    assert (
        report_output
        / "generated_acc_after_unblind"
        / "wav_8s"
        / exported_wav.name
    ).is_file()

    tampered = {**row, "run_id": "another-run"}
    write_canonical_json(
        export_root / "generated_acc_index.json", {"rows": [tampered]}
    )
    with (export_root / "generated_acc_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tampered))
        writer.writeheader()
        writer.writerow(tampered)
    with pytest.raises(RuntimeError, match="exact private-key derivation"):
        report._validate_generated_acc_export(
            tmp_path / "package", key, require_snapshot_binding=False
        )


def test_report_preserves_and_validates_failed_retry_attempt(load_script, tmp_path):
    report = load_script("build_robustness_report")
    fixture = _retry_attempt_lineage_fixture(report, tmp_path)

    history = report._validate_triangle_attempt_lineage(
        fixture["current_selection"],
        selection_path=fixture["current_selection_path"],
        current_package=fixture["current_package"],
    )

    assert [row["listening_attempt_id"] for row in history] == [
        "listening-attempt-001",
        "listening-attempt-002",
    ]
    assert history[0]["role"] == "sealed_qc_failure"
    assert history[0]["qc_status"] == "fail"
    assert history[0]["retry_required"] is True
    assert history[1]["role"] == "current_attempt"


def test_report_rejects_swapped_generated_exports_even_after_rehash(
    load_script, tmp_path
):
    report = load_script("build_robustness_report")
    package = tmp_path / "package"
    export_root = package / "generated_acc_after_unblind"
    (export_root / "midi").mkdir(parents=True)
    (export_root / "wav_8s").mkdir()

    sources = []
    rows = []
    for index, (song, condition, perturb_seed, sample_seed) in enumerate(
        [
            ("song-a", "pitch", 11, 21),
            ("song-b", "onset", 12, 22),
        ],
        start=1,
    ):
        original = tmp_path / f"formal-{index}.mid"
        canonical_wav = tmp_path / f"canonical-{index}.wav"
        original.write_bytes(f"midi-{index}".encode())
        canonical_wav.write_bytes(f"wav-{index}".encode())
        source = {
            "kind": "formal",
            "song": song,
            "condition": condition,
            "perturb_seed": perturb_seed,
            "sample_seed": sample_seed,
            "run_id": f"run-{index}",
            "attempt_id": "attempt-001",
            "operational_valid": True,
            "source_empty": False,
            "source_path": str(original),
            "source_sha256": file_sha256(original),
            "canonical_wav_path": str(canonical_wav),
            "canonical_wav_sha256": file_sha256(canonical_wav),
            "raw_token_payload_sha256": f"{index:064x}",
            "output_event_payload_sha256": f"{index + 10:064x}",
        }
        stem = f"{song}__{condition}__p-{perturb_seed}__s-{sample_seed}"
        exported_midi = export_root / "midi" / f"{stem}.mid"
        exported_wav = export_root / "wav_8s" / f"{stem}.wav"
        exported_midi.write_bytes(original.read_bytes())
        exported_wav.write_bytes(canonical_wav.read_bytes())
        sources.append(source)
        rows.append(
            {
                "song": song,
                "condition": condition,
                "perturb_seed": perturb_seed,
                "sample_seed": sample_seed,
                "run_id": source["run_id"],
                "attempt_id": source["attempt_id"],
                "operational_valid": True,
                "source_empty": False,
                "raw_token_payload_sha256": source[
                    "raw_token_payload_sha256"
                ],
                "output_event_payload_sha256": source[
                    "output_event_payload_sha256"
                ],
                "formal_theoretical_midi": str(original),
                "formal_theoretical_midi_sha256": file_sha256(original),
                "exported_midi": str(exported_midi),
                "exported_midi_sha256": file_sha256(exported_midi),
                "exported_excerpt_wav": str(exported_wav),
                "exported_excerpt_wav_sha256": file_sha256(exported_wav),
                "post_unblinding_qualitative_followup_only": True,
            }
        )

    def write_export_index(values):
        write_canonical_json(
            export_root / "generated_acc_index.json", {"rows": values}
        )
        with (export_root / "generated_acc_index.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    write_export_index(rows)
    key = {"trials": [{"source_a": sources[0], "source_b": sources[1]}]}
    assert report._validate_generated_acc_export(
        package, key, require_snapshot_binding=False
    )["valid"]

    first_path = Path(rows[0]["exported_midi"])
    second_path = Path(rows[1]["exported_midi"])
    first_bytes, second_bytes = first_path.read_bytes(), second_path.read_bytes()
    first_path.write_bytes(second_bytes)
    second_path.write_bytes(first_bytes)
    swapped_rows = json.loads(json.dumps(rows))
    swapped_rows[0]["exported_midi_sha256"] = file_sha256(first_path)
    swapped_rows[1]["exported_midi_sha256"] = file_sha256(second_path)
    write_export_index(swapped_rows)

    with pytest.raises(RuntimeError, match="semantic MIDI export"):
        report._validate_generated_acc_export(
            package, key, require_snapshot_binding=False
        )


def test_report_rejects_retry_lineage_or_authorization_tamper(load_script, tmp_path):
    report = load_script("build_robustness_report")
    fixture = _retry_attempt_lineage_fixture(report, tmp_path)
    original = fixture["current_selection"]

    tampered = json.loads(json.dumps(original))
    tampered["retry_lineage"]["previous_retry_required"] = False
    tampered["retry_lineage_sha256"] = canonical_sha256(
        tampered["retry_lineage"]
    )
    write_canonical_json(fixture["current_selection_path"], tampered)
    with pytest.raises(RuntimeError, match="attempt/QC contract"):
        report._validate_triangle_attempt_lineage(
            tampered,
            selection_path=fixture["current_selection_path"],
            current_package=fixture["current_package"],
        )

    authorization = {**fixture["authorization"], "previous_qc_status": "pass"}
    write_canonical_json(fixture["authorization_path"], authorization)
    tampered = json.loads(json.dumps(original))
    tampered["retry_lineage"]["retry_authorization_sha256"] = file_sha256(
        fixture["authorization_path"]
    )
    tampered["retry_lineage_sha256"] = canonical_sha256(
        tampered["retry_lineage"]
    )
    write_canonical_json(fixture["current_selection_path"], tampered)
    with pytest.raises(RuntimeError, match="authorization schema/attempt/QC"):
        report._validate_triangle_attempt_lineage(
            tampered,
            selection_path=fixture["current_selection_path"],
            current_package=fixture["current_package"],
        )


@pytest.mark.parametrize(
    "attempt_id",
    [None, "listening-attempt-000", "listening-attempt-01", "attempt-002"],
)
def test_report_rejects_malformed_dynamic_listening_attempt_id(
    load_script, attempt_id
):
    report = load_script("build_robustness_report")
    with pytest.raises(RuntimeError, match="listening-attempt-NNN|string"):
        report._listening_attempt_number(attempt_id)


def test_report_accepts_dynamic_listening_attempt_ids(load_script):
    report = load_script("build_robustness_report")
    assert report._listening_attempt_number("listening-attempt-001") == 1
    assert report._listening_attempt_number("listening-attempt-002") == 2


def _known_different_report_fixture(tmp_path, *, selection_sha, config_sha):
    formal_source = tmp_path / "known-formal.mid"
    comparator = tmp_path / "known-comparator.mid"
    synthetic_excerpt = tmp_path / "known-synthetic.mid"
    formal_source.write_bytes(b"formal-source")
    comparator.write_bytes(b"formal-comparator")
    synthetic_excerpt.write_bytes(b"synthetic-control")
    trials = []
    records = []
    for index in range(6):
        semantic_id = f"K:{index + 1:03d}"
        formal = {
            "kind": "formal",
            "formal_pipeline": "rt",
            "source_artifact": "theoretical_model",
            "song": f"song-{index % 5 + 1}",
            "condition": "sham",
            "perturb_seed": None,
            "sample_seed": 17,
        }
        recipe = {"name": "fixed_four_bar_scale_v1", "velocity": 96}
        synthetic = {"kind": "synthetic_control", "recipe": recipe}
        trial = {
            "semantic_id": semantic_id,
            "question_id": f"Q{index + 1:03d}",
            "block": "known_different_control",
            "sources": {"a": formal, "b": synthetic},
        }
        trials.append(trial)
        records.append(
            {
                "campaign_config_sha256": config_sha,
                "semantic_id": semantic_id,
                "question_id": trial["question_id"],
                "selection_source_a_sha256": canonical_sha256(formal),
                "selection_source_b_sha256": canonical_sha256(synthetic),
                "selection_recipe_sha256": canonical_sha256(recipe),
                "formal_run_id": f"run-{index + 1}",
                "formal_source_path": str(formal_source),
                "formal_source_sha256": file_sha256(formal_source),
                "formal_comparator_excerpt_path": str(comparator),
                "formal_comparator_excerpt_sha256": file_sha256(comparator),
                "formal_comparator_note_events_sha256": "d" * 64,
                "synthetic_excerpt_path": str(synthetic_excerpt),
                "synthetic_excerpt_sha256": file_sha256(synthetic_excerpt),
                "synthetic_velocity": 96,
                "not_identical": True,
            }
        )
    return (
        {
            "listening_known_different": {
                "selection_sha256": selection_sha,
                "expected_count": 6,
                "actual_count": 6,
                "all_recipe_bound": True,
                "all_source_selectors_bound": True,
                "all_not_identical": True,
                "controls": records,
            }
        },
        {"trials": trials},
    )


def test_report_accepts_exact_known_different_control_binding(load_script, tmp_path):
    report = load_script("build_robustness_report")
    selection_sha = "a" * 64
    config_sha = "c" * 64
    controls, selection = _known_different_report_fixture(
        tmp_path, selection_sha=selection_sha, config_sha=config_sha
    )
    report._validate_listening_known_different_controls(
        controls,
        base_selection=selection,
        base_selection_sha256=selection_sha,
        config_sha=config_sha,
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("selection_sha256", "b" * 64),
        ("expected_count", 5),
        ("actual_count", 5),
        ("all_recipe_bound", False),
        ("all_source_selectors_bound", False),
        ("all_not_identical", False),
    ],
)
def test_report_rejects_known_different_control_gap(
    load_script, tmp_path, field, bad_value
):
    report = load_script("build_robustness_report")
    selection_sha = "a" * 64
    config_sha = "c" * 64
    controls, selection = _known_different_report_fixture(
        tmp_path, selection_sha=selection_sha, config_sha=config_sha
    )
    section = controls["listening_known_different"]
    section[field] = bad_value
    with pytest.raises(RuntimeError, match="known-different control"):
        report._validate_listening_known_different_controls(
            controls,
            base_selection=selection,
            base_selection_sha256=selection_sha,
            config_sha=config_sha,
        )


def test_report_rejects_known_different_row_or_midi_tamper(load_script, tmp_path):
    report = load_script("build_robustness_report")
    selection_sha = "a" * 64
    config_sha = "c" * 64
    controls, selection = _known_different_report_fixture(
        tmp_path, selection_sha=selection_sha, config_sha=config_sha
    )
    controls["listening_known_different"]["controls"][0][
        "selection_recipe_sha256"
    ] = "f" * 64
    with pytest.raises(RuntimeError, match="selection_recipe_sha256"):
        report._validate_listening_known_different_controls(
            controls,
            base_selection=selection,
            base_selection_sha256=selection_sha,
            config_sha=config_sha,
        )

    controls, selection = _known_different_report_fixture(
        tmp_path, selection_sha=selection_sha, config_sha=config_sha
    )
    Path(
        controls["listening_known_different"]["controls"][0][
            "synthetic_excerpt_path"
        ]
    ).write_bytes(b"tampered-midi")
    with pytest.raises(RuntimeError, match="synthetic excerpt hash mismatch"):
        report._validate_listening_known_different_controls(
            controls,
            base_selection=selection,
            base_selection_sha256=selection_sha,
            config_sha=config_sha,
        )


def test_report_reverifies_campaign_attempts_and_listening_readiness(
    load_script, tmp_path, monkeypatch
):
    report = load_script("build_robustness_report")
    campaign, audit, schedule, selection = _campaign_audit_reverify_fixture(
        report, tmp_path, monkeypatch
    )

    report._reverify_campaign_audit(
        campaign,
        audit=audit,
        schedule=schedule,
        campaign_binding={},
        selection=selection,
    )


@pytest.mark.parametrize("mutation", ["invalid", "extra", "not_ready"])
def test_report_rejects_campaign_audit_gap_even_when_other_counts_look_complete(
    load_script, tmp_path, monkeypatch, mutation
):
    report = load_script("build_robustness_report")
    campaign, audit, schedule, selection = _campaign_audit_reverify_fixture(
        report, tmp_path, monkeypatch
    )
    if mutation == "invalid":
        audit["invalid"] = 1
    elif mutation == "extra":
        audit["extra_run_ids"] = ["unplanned-run"]
    else:
        audit["listening_source_readiness"]["ready"] = False

    with pytest.raises(RuntimeError, match="formal campaign|reverification|readiness"):
        report._reverify_campaign_audit(
            campaign,
            audit=audit,
            schedule=schedule,
            campaign_binding={},
            selection=selection,
        )


def test_builder_emits_complete_objective_only_report_at_zero_responses(
    load_script, tmp_path, monkeypatch
):
    report = load_script("build_robustness_report")
    config_path = tmp_path / "campaign.json"
    schedule_path = tmp_path / "schedule.jsonl"
    input_path = tmp_path / "inputs.json"
    selection_path = tmp_path / "triangle-selection.json"
    audit_path = tmp_path / "campaign-audit.json"
    campaign_root = tmp_path / "campaign"
    analysis_root = tmp_path / "analysis"
    listening_root = tmp_path / "listening"
    output = tmp_path / "report"
    campaign_root.mkdir()
    analysis_root.mkdir()
    (listening_root / "private").mkdir(parents=True)
    for path in (config_path, schedule_path, input_path):
        path.write_text("{}\n", encoding="utf-8")
    write_canonical_json(
        selection_path,
            {
                "schema_version": report.TRIANGLE_SELECTION_SCHEMA_VERSION,
                "listening_attempt_id": "listening-attempt-001",
                "listening_attempt_number": 1,
                "trials": [],
        },
    )
    config_sha = "a" * 64
    binding_sha = "b" * 64
    qualification_sha = "c" * 64
    schedule_sha = file_sha256(schedule_path)
    input_sha = file_sha256(input_path)
    selection_sha = file_sha256(selection_path)
    config = {
        "code_identity": "d" * 40,
        "checkpoint": {"path": str(tmp_path / "model"), "sha256": "e" * 64},
        "input_manifest": {"path": str(input_path), "sha256": input_sha},
        "qualification_candidate": {"sha256": "f" * 64},
        "qualification_result": {"sha256": qualification_sha},
        "listening": {
            "selection_manifest_path": str(selection_path),
            "selection_manifest_sha256": selection_sha,
        },
    }
    campaign_audit = {
        "campaign_config_sha256": config_sha,
        "run_schedule_sha256": schedule_sha,
        "campaign_binding_sha256": binding_sha,
        "qualification_result_sha256": qualification_sha,
        "expected": 160,
        "content_valid": 160,
        "missing": 0,
        "invalid": 0,
        "extra_run_ids": [],
        "listening_source_readiness": {"ready": True},
    }
    write_canonical_json(audit_path, campaign_audit)
    write_canonical_json(campaign_root / "campaign_binding.json", {"bound": True})
    listening_audit = {
        "accepted_final": True,
        "selection_sha256": selection_sha,
        "qualification_result_sha256": qualification_sha,
    }
    write_canonical_json(listening_root / "package_audit.json", listening_audit)
    write_canonical_json(
        listening_root / "render_manifest.json",
            {
                "schema_version": report.TRIANGLE_RENDER_SCHEMA_VERSION,
                "listening_attempt_id": "listening-attempt-001",
                "selection_path": str(selection_path),
            "selection_sha256": selection_sha,
        },
    )
    write_canonical_json(
        listening_root / "private" / "private_key.json", {"trials": []}
    )
    analysis_index_path = analysis_root / "analysis_index.json"
    control_path = analysis_root / "control_report.json"
    write_canonical_json(analysis_index_path, {"objective": True})
    write_canonical_json(control_path, {"objective": True})
    analysis_index = {
        "campaign_config_sha256": config_sha,
        "input_manifest_sha256": input_sha,
        "run_schedule_sha256": schedule_sha,
        "campaign_binding_sha256": binding_sha,
        "qualification_result_sha256": qualification_sha,
        "qc_invalid": 0,
    }
    controls = {
        "missing_songs": [],
        "listening_known_different": {
            "selection_sha256": selection_sha,
            "expected_count": 6,
            "actual_count": 6,
            "all_recipe_bound": True,
            "all_source_selectors_bound": True,
            "all_not_identical": True,
        },
        "songs": [
            {
                "endpoint_validity": {
                    "harmonic": True,
                    "rhythm": True,
                    "coverage": True,
                }
            }
            for _ in range(5)
        ],
    }
    listening_state = {
        "contract": "triangle_v2",
        "listening_attempt_id": "listening-attempt-001",
        "package_audit": listening_audit,
        "package_valid": True,
        "collection_status": "not_started",
        "answered_count": 0,
        "pending_count": 95,
        "ledger_answered_count": 0,
        "ledger_head_hash": None,
        "snapshot_status": "not_applicable",
        "snapshot_path": None,
        "snapshot_sha256": None,
        "semantic_result_status": "not_available",
        "qc_status": "not_started",
        "attempt_disposition": "not_started",
        "retry_required": False,
        "blinding_status": "fully_blind",
        "first_semantic_unblind": None,
        "summary": None,
        "identity_counts": {"trial_count": 95},
        "generated_acc_export": {
            "valid": True,
            "not_applicable": True,
            "row_count": 0,
        },
    }
    monkeypatch.setattr(
        report,
        "_validated_campaign_inputs",
        lambda *_args, **_kwargs: (
            config,
            config_sha,
            input_path,
            {},
            [{} for _ in range(40)],
            [],
        ),
    )
    monkeypatch.setattr(
        report,
        "_validated_campaign_binding",
        lambda *_args, **_kwargs: ({}, binding_sha),
    )
    monkeypatch.setattr(report, "_validate_audit_schedule", lambda *_args: None)
    monkeypatch.setattr(
        report,
        "_validated_analysis_artifacts",
        lambda *_args, **_kwargs: ({"results": {}}, controls, analysis_index),
    )
    monkeypatch.setattr(
        report, "validate_triangle_selection_manifest", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        report,
        "_validate_listening_known_different_controls",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(report, "_reverify_campaign_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        report,
        "_recompute_triangle_package_audit",
        lambda *_args, **_kwargs: listening_audit,
    )
    monkeypatch.setattr(
        report, "_listening_state", lambda *_args, **_kwargs: listening_state
    )

    report.build(
        argparse.Namespace(
            config=str(config_path),
            config_sha256=config_sha,
            schedule=str(schedule_path),
            schedule_sha256=schedule_sha,
            campaign_root=str(campaign_root),
            campaign_audit=str(audit_path),
            analysis_dir=str(analysis_root),
            listening_package=str(listening_root),
            listening_snapshot=None,
            output_dir=str(output),
            allow_incomplete=False,
        )
    )

    index = json.loads(
        (output / "reproducibility_index.json").read_text(encoding="utf-8")
    )
    assert index["status"] == "complete"
    assert index["listening_result"]["collection_status"] == "not_started"
    report_text = (output / "report.md").read_text(encoding="utf-8")
    assert "采集状态：**not_started**" in report_text
    assert "尚无 human response" in report_text
    assert not (output / "generated_acc_after_unblind").exists()
