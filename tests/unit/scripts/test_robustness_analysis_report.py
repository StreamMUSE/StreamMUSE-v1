from __future__ import annotations

import json

import pytest

from streammuse.experiments.melody_robustness import (
    build_run_schedule,
    default_campaign_config,
    file_sha256,
    write_canonical_json,
    write_jsonl,
)


def _campaign_files(fixture, tmp_path):
    manifest = json.loads(fixture.input_manifest.read_text(encoding="utf-8"))
    config = default_campaign_config(
        code_identity="a" * 40,
        checkpoint_path=str(fixture.checkpoint),
        checkpoint_sha256=file_sha256(fixture.checkpoint),
        input_manifest_path=str(fixture.input_manifest),
        input_manifest_sha256=file_sha256(fixture.input_manifest),
        playback_tempo=60,
        tail_beats=24,
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
        )


def test_report_keeps_valid_package_incomplete_until_scores_are_sealed_and_unblinded(
    load_script, tmp_path
):
    report = load_script("build_robustness_report")
    config_sha = "a" * 64
    schedule_sha = "b" * 64
    binding_sha = "d" * 64
    selection_sha = "c" * 64
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
    )

    assert audit["accepted_final"]
    assert not sealed
    assert not unblinded
