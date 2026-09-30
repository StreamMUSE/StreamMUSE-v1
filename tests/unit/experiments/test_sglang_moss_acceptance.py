from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluate_sglang_moss_ab import main as evaluate_main
from scripts.freeze_sglang_moss_experiment import main as freeze_main
from scripts.prepare_sglang_moss_ab_root import main as prepare_root_main
from streammuse.experiments.sglang_moss_acceptance import (
    AcceptanceDataError,
    ARTIFACT_ROOT_MANIFEST_FILENAME,
    BLIND_SCORE_SCHEMA_VERSION,
    ExperimentDesignError,
    PROBE_SCHEMA_VERSION,
    ROW_SCHEMA_VERSION,
    build_artifact_root_manifest,
    evaluate_acceptance,
    freeze_experiment_manifest,
    validate_experiment_manifest,
    write_immutable_json,
)
from streammuse.infrastructure.rap.producer_manifest import canonical_json_bytes


def corpus() -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample-{index:03d}",
            "request_id": f"{index + 1:064x}",
            "request_payload_sha256": f"{index + 1001:064x}",
            "text": f"frozen lyric number {index}",
            "flow_id": f"flow-{index % 7}",
            "tempo_bpm": 96.0 + index % 5,
            "token_count": 64 + index % 4,
            "seed": 20_260_904 + index,
            "expected_frame_count": 120_000 + index,
        }
        for index in range(100)
    ]


def pins() -> dict[str, object]:
    return {
        "streammuse": {"commit": "abc1234", "patch_sha256": "1" * 64},
        "gpu": {
            "physical_id": "GPU-11111111-2222-3333-4444-555555555555",
            "runtime_image_digest": f"sha256:{'2' * 64}",
        },
        "model": {"id": "OpenMOSS-Team/MOSS-TTS-v1.5", "revision": "moss-rev-1"},
        "generation": {
            "settings_sha256": "3" * 64,
            "seed_policy_version": "streammuse.moss-seed.v1",
        },
        "producers": {
            "serving-only-parity": None,
            "production-candidate": {
                "baseline": _producer(
                    "4", "inprocess", "5", "unavailable", "baseline-1"
                ),
                "candidate": _producer(
                    "8", "sglang-omni", "9", "a" * 64, "candidate-1"
                ),
            },
        },
    }


def _producer(
    fingerprint: str,
    backend: str,
    config: str,
    text_hash: str,
    identity: str,
) -> dict[str, str]:
    return {
        "producer_fingerprint": fingerprint * 64,
        "sidecar_backend": backend,
        "model_revision": "moss-rev-1",
        "config_sha256": config * 64,
        "reference_audio_sha256": "6" * 64,
        "reference_text_sha256": text_hash,
        "runtime_environment_sha256": "7" * 64,
        "launch_manifest_sha256": ("b" if backend == "inprocess" else "c") * 64,
        "health_identity": identity,
        "health_version": "runtime-1.0",
    }


def manifest() -> dict[str, object]:
    return freeze_experiment_manifest(
        experiment_id="sglang-moss-20260904-v1",
        implementation_id="sglang-moss-http-v1",
        created_at_utc="2026-09-04T10:00:00Z",
        corpus=corpus(),
        pins=pins(),
        schedule_seed=20260904,
        bootstrap_seed=20260905,
        bootstrap_resamples=2_000,
        cohort_a_enabled=False,
        cohort_a_evidence="pinned-build-does-not-support-omitted-ref-text",
    )


def artifact_roots(frozen: dict[str, object]) -> list[dict[str, object]]:
    roots = [
        build_artifact_root_manifest(
            frozen,
            cohort="production-candidate",
            phase="qualification",
            block_id="qualification",
            backend=backend,
        )
        for backend in ("baseline", "candidate")
    ]
    for cohort, config in frozen["cohorts"].items():
        if not config["enabled"]:
            continue
        for block in frozen["schedule"]["blocks"]:
            for backend in ("baseline", "candidate"):
                roots.append(
                    build_artifact_root_manifest(
                        frozen,
                        cohort=cohort,
                        phase="final",
                        block_id=block["block_id"],
                        backend=backend,
                    )
                )
    return roots


def evidence(
    frozen: dict[str, object], *, candidate_moss_ms: float = 80.0
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    samples = {item["sample_id"]: item for item in frozen["corpus"]["samples"]}
    block_by_sample = {
        sample_id: block
        for block in frozen["schedule"]["blocks"]
        for sample_id in block["sample_ids"]
    }
    rows: list[dict[str, object]] = []
    for sample_id in frozen["corpus"]["qualification_sample_ids"]:
        for order, backend in enumerate(("baseline", "candidate")):
            rows.append(
                _row(
                    frozen,
                    samples[sample_id],
                    backend=backend,
                    phase="qualification",
                    block_id="qualification",
                    backend_order_index=order,
                    candidate_moss_ms=candidate_moss_ms,
                )
            )
    for cohort_name, cohort_config in frozen["cohorts"].items():
        if cohort_config["enabled"]:
            for sample_id, sample in samples.items():
                block = block_by_sample[sample_id]
                for backend in ("baseline", "candidate"):
                    rows.append(
                        _row(
                            frozen,
                            sample,
                            backend=backend,
                            phase="final",
                            block_id=block["block_id"],
                            backend_order_index=block["backend_order"].index(backend),
                            candidate_moss_ms=candidate_moss_ms,
                            cohort=cohort_name,
                        )
                    )
    probes = [
        {
            "schema_version": PROBE_SCHEMA_VERSION,
            "experiment_id": frozen["experiment_id"],
            "experiment_manifest_sha256": frozen["manifest_sha256"],
            "backend": backend,
            "probe_kind": kind,
            "producer_fingerprint": frozen["pins"]["producers"][
                "production-candidate"
            ][backend]["producer_fingerprint"],
            "queue_recovered": True,
            "upstream_request_released": True,
            "next_short_request_ms": 90.0,
            "normal_warm_p95_ms": 100.0,
        }
        for backend in ("baseline", "candidate")
        for kind in ("timeout", "disconnect", "cancellation")
    ]
    scores = [
        {
            "schema_version": BLIND_SCORE_SCHEMA_VERSION,
            "experiment_id": frozen["experiment_id"],
            "experiment_manifest_sha256": frozen["manifest_sha256"],
            "sample_id": sample_id,
            "reviewer_id": f"reviewer-{reviewer}",
            "presentation_id": f"{sample_id}-{reviewer}-{backend}",
            "backend": backend,
            "scores": {"intelligibility": 4, "voice_identity": 4, "rhythm": 4},
            "critical_artifact": False,
        }
        for sample_id in frozen["corpus"]["blind_sample_ids"]
        for reviewer in (1, 2)
        for backend in ("baseline", "candidate")
    ]
    return rows, probes, scores


def _row(
    frozen: dict[str, object],
    sample: dict[str, object],
    *,
    backend: str,
    phase: str,
    block_id: str,
    backend_order_index: int,
    candidate_moss_ms: float,
    cohort: str = "production-candidate",
) -> dict[str, object]:
    producer = frozen["pins"]["producers"][cohort][backend]
    moss_ms = 100.0 if backend == "baseline" else candidate_moss_ms
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "experiment_id": frozen["experiment_id"],
        "experiment_manifest_sha256": frozen["manifest_sha256"],
        "cohort": cohort,
        "phase": phase,
        "block_id": block_id,
        "sample_id": sample["sample_id"],
        "backend": backend,
        "backend_order_index": backend_order_index,
        "request_id": sample["request_id"],
        "block_valid": True,
        "invalidation_reasons": [],
        "competing_processes": [],
        "success": True,
        "failure_kind": None,
        "events": {
            "crash": False,
            "oom": False,
            "hung": False,
            "timeout": False,
            "deadline_miss": False,
            "schema_violation": False,
            "wrong_producer_cache_hit": False,
        },
        "service_call_count": 1 if backend == "candidate" else 0,
        "timings_ms": {
            "queue": 1.0,
            "moss": moss_ms,
            "mms": 10.0,
            "r3": 10.0,
            "package": 2.0,
            "complete_server": moss_ms + 30.0,
        },
        "artifact": {
            "cache_hit": False,
            "sidecar_valid": True,
            "package_valid": True,
            "public_schema_valid": True,
            "exact_frame_count": True,
            "expected_frame_count": sample["expected_frame_count"],
            "final_frame_count": sample["expected_frame_count"],
        },
        "quality": {
            "asr_reference_words": 10,
            "asr_substitutions": 0,
            "asr_deletions": 0,
            "asr_insertions": 0,
            "mms_coverage": 0.98,
            "mms_confidence": 0.95,
            "mms_below_hard_floor": False,
            "local_stretch_ratios": [1.0],
            "fallback_count": 0,
            "speaker_similarity": None,
        },
        "provenance": {
            "producer_fingerprint": producer["producer_fingerprint"],
            "health_producer_fingerprint": producer["producer_fingerprint"],
            "sidecar_backend": producer["sidecar_backend"],
            "health_backend": producer["sidecar_backend"],
            "model_revision": producer["model_revision"],
            "config_sha256": producer["config_sha256"],
            "reference_audio_sha256": producer["reference_audio_sha256"],
            "reference_text_sha256": producer["reference_text_sha256"],
            "health_identity": producer["health_identity"],
            "health_version": producer["health_version"],
            "artifact_root_manifest_sha256": build_artifact_root_manifest(
                frozen,
                cohort=cohort,
                phase=phase,
                block_id=block_id,
                backend=backend,
            )["root_manifest_sha256"],
        },
    }


def _rehash_manifest(value: dict[str, object]) -> None:
    unsigned = {key: item for key, item in value.items() if key != "manifest_sha256"}
    value["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def test_freeze_is_deterministic_counterbalanced_and_self_validating() -> None:
    first = manifest()
    second = manifest()

    assert first == second
    assert [block["backend_order"] for block in first["schedule"]["blocks"]].count(
        ["baseline", "candidate"]
    ) == 2
    assert len(first["corpus"]["qualification_sample_ids"]) == 30
    assert len(first["corpus"]["blind_sample_ids"]) == 20
    assert validate_experiment_manifest(first) == first


def test_manifest_rejects_rehashed_schedule_or_subset_tampering() -> None:
    frozen = manifest()
    frozen["corpus"]["qualification_sample_ids"].reverse()
    _rehash_manifest(frozen)

    with pytest.raises(ExperimentDesignError, match="subset"):
        validate_experiment_manifest(frozen)


def test_freeze_rejects_small_corpus_and_sensitive_pins() -> None:
    with pytest.raises(ExperimentDesignError, match="at least 100"):
        freeze_experiment_manifest(
            experiment_id="exp",
            implementation_id="impl",
            created_at_utc="2026-09-04T10:00:00Z",
            corpus=corpus()[:99],
            pins=pins(),
            schedule_seed=1,
            bootstrap_seed=2,
        )
    unsafe = pins()
    unsafe["streammuse"]["access_token"] = "secret"
    with pytest.raises(ExperimentDesignError):
        freeze_experiment_manifest(
            experiment_id="exp",
            implementation_id="impl",
            created_at_utc="2026-09-04T10:00:00Z",
            corpus=corpus(),
            pins=unsafe,
            schedule_seed=1,
            bootstrap_seed=2,
        )


def test_all_gates_pass_to_h200_promotion_candidate() -> None:
    frozen = manifest()
    rows, probes, scores = evidence(frozen)

    report = evaluate_acceptance(
        frozen,
        rows,
        artifact_roots=artifact_roots(frozen),
        recovery_probes=probes,
        blind_scores=scores,
    )

    assert report["decision"] == "promotion-candidate"
    assert report["mac_e2e_status"] == "pending_phase_8"
    assert all(
        gate["status"] == "pass"
        for gate in report["gates"]
        if gate["severity"] in {"hard", "optimization"}
    )
    assert report["latency"]["paired_moss_candidate_minus_baseline_ms"]["ci95"] == [
        -20.0,
        -20.0,
    ]


def test_enabled_cohort_a_uses_its_own_reference_less_producer_pins() -> None:
    pinned = pins()
    pinned["producers"]["serving-only-parity"] = {
        "baseline": _producer("d", "inprocess", "5", "unavailable", "baseline-1"),
        "candidate": _producer("e", "sglang-omni", "9", "unavailable", "candidate-1"),
    }
    frozen = freeze_experiment_manifest(
        experiment_id="sglang-moss-cohort-a-v1",
        implementation_id="sglang-moss-http-v1",
        created_at_utc="2026-09-04T10:00:00Z",
        corpus=corpus(),
        pins=pinned,
        schedule_seed=20260904,
        bootstrap_seed=20260905,
        bootstrap_resamples=2_000,
        cohort_a_enabled=True,
        cohort_a_evidence="omitted-ref-text-smoke-sha256-deadbeef",
    )
    rows, probes, scores = evidence(frozen)

    report = evaluate_acceptance(
        frozen,
        rows,
        artifact_roots=artifact_roots(frozen),
        recovery_probes=probes,
        blind_scores=scores,
    )

    assert report["decision"] == "promotion-candidate"
    assert report["cohort_a_status"] == "evaluated"
    assert report["sample_counts"]["final"]["serving-only-parity"] == {
        "baseline": 100,
        "candidate": 100,
    }


def test_quality_pass_but_primary_latency_fail_is_experimental() -> None:
    frozen = manifest()
    rows, probes, scores = evidence(frozen, candidate_moss_ms=100.0)

    report = evaluate_acceptance(
        frozen,
        rows,
        artifact_roots=artifact_roots(frozen),
        recovery_probes=probes,
        blind_scores=scores,
    )

    assert report["decision"] == "experimental"
    primary = next(gate for gate in report["gates"] if gate["name"] == "primary_moss_latency")
    assert primary["status"] == "fail"


def test_hard_failure_with_missing_metrics_produces_blocked_report() -> None:
    frozen = manifest()
    rows, probes, scores = evidence(frozen)
    failed = next(row for row in rows if row["phase"] == "final" and row["backend"] == "candidate")
    failed["success"] = False
    failed["failure_kind"] = "oom"
    failed["events"]["oom"] = True
    failed["timings_ms"] = {key: None for key in failed["timings_ms"]}
    failed["artifact"].update(
        sidecar_valid=False,
        package_valid=False,
        public_schema_valid=False,
        exact_frame_count=False,
        final_frame_count=None,
    )
    failed["quality"] = {
        "asr_reference_words": None,
        "asr_substitutions": None,
        "asr_deletions": None,
        "asr_insertions": None,
        "mms_coverage": None,
        "mms_confidence": None,
        "mms_below_hard_floor": None,
        "local_stretch_ratios": [],
        "fallback_count": None,
        "speaker_similarity": None,
    }

    report = evaluate_acceptance(
        frozen,
        rows,
        artifact_roots=artifact_roots(frozen),
        recovery_probes=probes,
        blind_scores=scores,
    )

    assert report["decision"] == "blocked"
    assert next(gate for gate in report["gates"] if gate["name"] == "final_reliability")[
        "status"
    ] == "fail"
    assert report["latency"]["status"] == "unavailable"


def test_duplicate_row_is_rejected_instead_of_silently_dropped() -> None:
    frozen = manifest()
    rows, probes, scores = evidence(frozen)
    rows.append(copy.deepcopy(rows[-1]))

    with pytest.raises(AcceptanceDataError, match="duplicate"):
        evaluate_acceptance(
            frozen,
            rows,
            artifact_roots=artifact_roots(frozen),
            recovery_probes=probes,
            blind_scores=scores,
        )


def test_artifact_root_coverage_and_row_identity_fail_closed() -> None:
    frozen = manifest()
    rows, probes, scores = evidence(frozen)
    roots = artifact_roots(frozen)

    with pytest.raises(AcceptanceDataError, match="coverage"):
        evaluate_acceptance(
            frozen,
            rows,
            artifact_roots=roots[:-1],
            recovery_probes=probes,
            blind_scores=scores,
        )

    rows[0]["provenance"]["artifact_root_manifest_sha256"] = "f" * 64
    report = evaluate_acceptance(
        frozen,
        rows,
        artifact_roots=roots,
        recovery_probes=probes,
        blind_scores=scores,
    )
    assert report["decision"] == "blocked"
    provenance = next(
        gate for gate in report["gates"] if gate["name"] == "provenance_identity"
    )
    assert provenance["status"] == "fail"


def test_prepare_artifact_root_cli_is_idempotent_only_while_unused(
    tmp_path: Path,
) -> None:
    frozen = manifest()
    manifest_path = tmp_path / "experiment.json"
    manifest_path.write_text(json.dumps(frozen), encoding="utf-8")
    root = tmp_path / "qualification-candidate"
    args = [
        "--manifest",
        str(manifest_path),
        "--root",
        str(root),
        "--cohort",
        "production-candidate",
        "--phase",
        "qualification",
        "--block-id",
        "qualification",
        "--backend",
        "candidate",
    ]

    assert prepare_root_main(args) == 0
    assert prepare_root_main(args) == 0
    marker = json.loads(
        (root / ARTIFACT_ROOT_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert marker == build_artifact_root_manifest(
        frozen,
        cohort="production-candidate",
        phase="qualification",
        block_id="qualification",
        backend="candidate",
    )

    mismatched = [*args[:-1], "baseline"]
    assert prepare_root_main(mismatched) == 2
    (root / "unexpected-complete-artifact").mkdir()
    assert prepare_root_main(args) == 2


def test_immutable_writer_allows_identical_replay_only(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_immutable_json(path, {"a": 1})
    write_immutable_json(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}

    with pytest.raises(FileExistsError):
        write_immutable_json(path, {"a": 2})


def test_freeze_and_evaluate_cli_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus_path = tmp_path / "corpus.json"
    pins_path = tmp_path / "pins.json"
    manifest_path = tmp_path / "experiment.json"
    corpus_path.write_text(json.dumps(corpus()), encoding="utf-8")
    pins_path.write_text(json.dumps(pins()), encoding="utf-8")
    freeze_args = [
        "--experiment-id",
        "sglang-moss-20260904-v1",
        "--implementation-id",
        "sglang-moss-http-v1",
        "--created-at-utc",
        "2026-09-04T10:00:00Z",
        "--corpus",
        str(corpus_path),
        "--pins",
        str(pins_path),
        "--schedule-seed",
        "20260904",
        "--bootstrap-seed",
        "20260905",
        "--bootstrap-resamples",
        "2000",
        "--cohort-a-evidence",
        "pinned-build-does-not-support-omitted-ref-text",
        "--output",
        str(manifest_path),
    ]
    assert freeze_main(freeze_args) == 0
    assert freeze_main(freeze_args) == 0
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows, probes, scores = evidence(frozen)
    rows_path = tmp_path / "rows.jsonl"
    roots_path = tmp_path / "artifact-roots.jsonl"
    probes_path = tmp_path / "probes.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    report_path = tmp_path / "report.json"
    for path, values in (
        (rows_path, rows),
        (roots_path, artifact_roots(frozen)),
        (probes_path, probes),
        (scores_path, scores),
    ):
        path.write_text(
            "".join(json.dumps(item) + "\n" for item in values),
            encoding="utf-8",
        )
    result = evaluate_main(
        [
            "--manifest",
            str(manifest_path),
            "--rows",
            str(rows_path),
            "--artifact-roots",
            str(roots_path),
            "--recovery-probes",
            str(probes_path),
            "--blind-scores",
            str(scores_path),
            "--output",
            str(report_path),
        ]
    )
    assert result == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["decision"] == (
        "promotion-candidate"
    )
    assert "decision=promotion-candidate" in capsys.readouterr().out
