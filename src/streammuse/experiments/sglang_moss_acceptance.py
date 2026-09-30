"""Frozen design and fail-closed acceptance gates for SGLang MOSS A/B runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from streammuse.infrastructure.rap.producer_manifest import canonical_json_bytes


EXPERIMENT_SCHEMA_VERSION = "streammuse.sglang_moss_experiment.v1"
ARTIFACT_ROOT_SCHEMA_VERSION = "streammuse.sglang_moss_ab_artifact_root.v1"
ARTIFACT_ROOT_MANIFEST_FILENAME = "_streammuse_ab_artifact_root.v1.json"
ROW_SCHEMA_VERSION = "streammuse.sglang_moss_ab_row.v1"
PROBE_SCHEMA_VERSION = "streammuse.sglang_moss_recovery_probe.v1"
BLIND_SCORE_SCHEMA_VERSION = "streammuse.sglang_moss_blind_score.v1"
MAC_EVIDENCE_SCHEMA_VERSION = "streammuse.sglang_moss_mac_e2e.v1"
REPORT_SCHEMA_VERSION = "streammuse.sglang_moss_acceptance_report.v1"

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_BACKENDS = ("baseline", "candidate")
_COHORT_A = "serving-only-parity"
_COHORT_B = "production-candidate"
_PHASES = ("qualification", "final")
_PROBE_KINDS = ("timeout", "disconnect", "cancellation")
_SCORE_DIMENSIONS = ("intelligibility", "voice_identity", "rhythm")

DEFAULT_THRESHOLDS: dict[str, object] = {
    "qualification_samples_per_backend": 30,
    "final_samples_per_backend": 100,
    "counterbalanced_block_count": 4,
    "blind_samples": 20,
    "blind_reviewers_per_sample": 2,
    "asr_wer_delta_max": 0.01,
    "asr_deletion_rate_delta_max": 0.005,
    "mms_median_coverage_delta_min": -0.01,
    "mms_median_confidence_delta_min": -0.02,
    "r3_local_stretch_p95_ratio_max": 1.10,
    "r3_fallback_extra_fraction": 0.01,
    "speaker_similarity_median_delta_min": -0.02,
    "blind_score_median_delta_min": -0.5,
    "moss_latency_p95_ratio_max": 1.10,
    "complete_server_latency_p95_ratio_max": 1.05,
    "primary_moss_paired_median_delta_max_exclusive_ms": 0.0,
    "primary_moss_paired_bootstrap_ci_upper_max_exclusive_ms": 0.0,
}


class ExperimentDesignError(ValueError):
    """Raised when an experiment cannot be frozen reproducibly."""


class AcceptanceDataError(ValueError):
    """Raised when supplied evidence is malformed or not the frozen design."""


def freeze_experiment_manifest(
    *,
    experiment_id: str,
    implementation_id: str,
    created_at_utc: str,
    corpus: Sequence[Mapping[str, object]],
    pins: Mapping[str, object],
    schedule_seed: int,
    bootstrap_seed: int,
    bootstrap_resamples: int = 10_000,
    cohort_a_enabled: bool = False,
    cohort_a_evidence: str = "not_supported",
    speaker_similarity_hard_gate: bool = False,
) -> dict[str, object]:
    """Return a self-hashed manifest whose design cannot drift after a run."""
    _require_id(experiment_id, "experiment id")
    _require_id(implementation_id, "implementation id")
    if not _UTC_TIMESTAMP.fullmatch(created_at_utc):
        raise ExperimentDesignError("created_at_utc must be second-resolution UTC ending in Z")
    _require_nonnegative_int(schedule_seed, "schedule seed")
    _require_nonnegative_int(bootstrap_seed, "bootstrap seed")
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 2_000:
        raise ExperimentDesignError("bootstrap resamples must be an integer >= 2000")
    if type(cohort_a_enabled) is not bool:
        raise ExperimentDesignError("cohort A enabled flag must be boolean")
    if type(speaker_similarity_hard_gate) is not bool:
        raise ExperimentDesignError("speaker-similarity gate flag must be boolean")
    if not isinstance(cohort_a_evidence, str) or not cohort_a_evidence.strip():
        raise ExperimentDesignError("cohort A capability evidence must be non-empty")

    frozen_corpus = _validate_corpus(corpus)
    frozen_pins = _validate_pins(pins)
    shuffled_ids = [sample["sample_id"] for sample in frozen_corpus]
    random.Random(schedule_seed).shuffle(shuffled_ids)
    blocks = _build_blocks(shuffled_ids, schedule_seed=schedule_seed)

    qualification_ids = list(shuffled_ids[:30])
    blind_ids = list(shuffled_ids[-20:])
    manifest: dict[str, object] = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "implementation_id": implementation_id,
        "created_at_utc": created_at_utc,
        "pins": frozen_pins,
        "corpus": {
            "schema_version": "streammuse.sglang_moss_corpus.v1",
            "sha256": _sha256(canonical_json_bytes(frozen_corpus)),
            "sample_count": len(frozen_corpus),
            "samples": frozen_corpus,
            "qualification_sample_ids": qualification_ids,
            "blind_sample_ids": blind_ids,
        },
        "cohorts": {
            _COHORT_A: {
                "enabled": cohort_a_enabled,
                "baseline_uses_reference_text": False,
                "candidate_uses_reference_text": False,
                "capability_evidence": cohort_a_evidence.strip(),
                "interpretation": "serving boundary only",
            },
            _COHORT_B: {
                "enabled": True,
                "baseline_uses_reference_text": False,
                "candidate_uses_reference_text": True,
                "interpretation": "production path with a recorded conditioning difference",
            },
        },
        "schedule": {
            "seed": schedule_seed,
            "blocks": blocks,
            "artifact_root_template": (
                "{experiment_id}/{cohort}/{phase}/{block_id}/{backend}"
            ),
            "cold_start_included_in_warm_distribution": False,
            "warmup_included_in_warm_distribution": False,
            "invalidate_block_on_competing_gpu_process": True,
        },
        "statistics": {
            "paired_unit": "sample_id",
            "estimator": "median(candidate_ms-baseline_ms)",
            "bootstrap_method": "paired empirical bootstrap with replacement",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "confidence_level": 0.95,
            "percentile_method": "linear interpolation",
        },
        "thresholds": json.loads(canonical_json_bytes(DEFAULT_THRESHOLDS)),
        "speaker_similarity_hard_gate": speaker_similarity_hard_gate,
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    validate_experiment_manifest(manifest)
    return manifest


def validate_experiment_manifest(value: Mapping[str, object]) -> dict[str, object]:
    manifest = _canonical_mapping(value, "experiment manifest")
    expected_keys = {
        "schema_version",
        "experiment_id",
        "implementation_id",
        "created_at_utc",
        "pins",
        "corpus",
        "cohorts",
        "schedule",
        "statistics",
        "thresholds",
        "speaker_similarity_hard_gate",
        "manifest_sha256",
    }
    _exact_keys(manifest, expected_keys, "experiment manifest")
    if manifest["schema_version"] != EXPERIMENT_SCHEMA_VERSION:
        raise ExperimentDesignError("unsupported experiment manifest schema")
    _require_id(manifest["experiment_id"], "experiment id")
    _require_id(manifest["implementation_id"], "implementation id")
    if not isinstance(manifest["created_at_utc"], str) or not _UTC_TIMESTAMP.fullmatch(
        manifest["created_at_utc"]
    ):
        raise ExperimentDesignError("created_at_utc must be second-resolution UTC ending in Z")
    if not isinstance(manifest["manifest_sha256"], str) or not _SHA256.fullmatch(
        manifest["manifest_sha256"]
    ):
        raise ExperimentDesignError("manifest SHA-256 is invalid")
    if manifest["manifest_sha256"] != _manifest_sha256(manifest):
        raise ExperimentDesignError("experiment manifest hash does not match")
    if manifest["thresholds"] != DEFAULT_THRESHOLDS:
        raise ExperimentDesignError("experiment thresholds differ from the frozen v1 gates")

    corpus = _require_mapping(manifest["corpus"], "corpus")
    _exact_keys(
        corpus,
        {
            "schema_version",
            "sha256",
            "sample_count",
            "samples",
            "qualification_sample_ids",
            "blind_sample_ids",
        },
        "corpus",
    )
    samples_value = corpus["samples"]
    if not isinstance(samples_value, list):
        raise ExperimentDesignError("corpus samples must be an array")
    samples = _validate_corpus(samples_value)
    if corpus["sample_count"] != len(samples):
        raise ExperimentDesignError("corpus sample count does not match")
    if corpus["sha256"] != _sha256(canonical_json_bytes(samples)):
        raise ExperimentDesignError("corpus SHA-256 does not match")

    sample_ids = {sample["sample_id"] for sample in samples}
    qualification_ids = _id_list(
        corpus["qualification_sample_ids"], "qualification sample ids", 30
    )
    blind_ids = _id_list(corpus["blind_sample_ids"], "blind sample ids", 20)
    if not set(qualification_ids).issubset(sample_ids) or not set(blind_ids).issubset(
        sample_ids
    ):
        raise ExperimentDesignError("manifest subsets contain unknown corpus samples")

    _validate_pins(_require_mapping(manifest["pins"], "pins"))
    cohorts = _require_mapping(manifest["cohorts"], "cohorts")
    if set(cohorts) != {_COHORT_A, _COHORT_B}:
        raise ExperimentDesignError("experiment cohorts do not match v1")
    for cohort_name, expected_candidate_text in ((_COHORT_A, False), (_COHORT_B, True)):
        cohort = _require_mapping(cohorts[cohort_name], cohort_name)
        required = {
            "enabled",
            "baseline_uses_reference_text",
            "candidate_uses_reference_text",
            "interpretation",
        }
        if cohort_name == _COHORT_A:
            required.add("capability_evidence")
        _exact_keys(cohort, required, cohort_name)
        if type(cohort["enabled"]) is not bool:
            raise ExperimentDesignError(f"{cohort_name} enabled must be boolean")
        if cohort["baseline_uses_reference_text"] is not False:
            raise ExperimentDesignError("baseline reference-text policy changed")
        if cohort["candidate_uses_reference_text"] is not expected_candidate_text:
            raise ExperimentDesignError("candidate reference-text policy changed")
    if cohorts[_COHORT_B]["enabled"] is not True:
        raise ExperimentDesignError("production-candidate cohort is mandatory")
    producer_pins = manifest["pins"]["producers"]
    if producer_pins[_COHORT_B] is None:
        raise ExperimentDesignError("production-candidate producer pins are mandatory")
    if cohorts[_COHORT_A]["enabled"] and producer_pins[_COHORT_A] is None:
        raise ExperimentDesignError("enabled cohort A requires dedicated producer pins")

    schedule = _require_mapping(manifest["schedule"], "schedule")
    _validate_schedule(schedule, sample_ids)
    shuffled_ids = [sample["sample_id"] for sample in samples]
    random.Random(schedule["seed"]).shuffle(shuffled_ids)
    if schedule["blocks"] != _build_blocks(
        shuffled_ids, schedule_seed=schedule["seed"]
    ):
        raise ExperimentDesignError("block schedule does not match its frozen seed")
    if qualification_ids != shuffled_ids[:30] or blind_ids != shuffled_ids[-20:]:
        raise ExperimentDesignError("qualification or blind subset does not match schedule seed")
    statistics_config = _require_mapping(manifest["statistics"], "statistics")
    _exact_keys(
        statistics_config,
        {
            "paired_unit",
            "estimator",
            "bootstrap_method",
            "bootstrap_seed",
            "bootstrap_resamples",
            "confidence_level",
            "percentile_method",
        },
        "statistics",
    )
    if statistics_config.get("paired_unit") != "sample_id":
        raise ExperimentDesignError("paired statistics unit must remain sample_id")
    if statistics_config.get("bootstrap_method") != "paired empirical bootstrap with replacement":
        raise ExperimentDesignError("bootstrap method changed")
    _require_nonnegative_int(statistics_config.get("bootstrap_seed"), "bootstrap seed")
    if type(statistics_config.get("bootstrap_resamples")) is not int or statistics_config[
        "bootstrap_resamples"
    ] < 2_000:
        raise ExperimentDesignError("bootstrap resamples are invalid")
    if manifest["speaker_similarity_hard_gate"] not in {True, False}:
        raise ExperimentDesignError("speaker-similarity gate flag is invalid")
    return manifest


def build_artifact_root_manifest(
    manifest_value: Mapping[str, object],
    *,
    cohort: str,
    phase: str,
    block_id: str,
    backend: str,
) -> dict[str, object]:
    """Bind one empty artifact root to a frozen experiment coordinate."""
    manifest = validate_experiment_manifest(manifest_value)
    return _artifact_root_manifest(
        manifest,
        cohort=cohort,
        phase=phase,
        block_id=block_id,
        backend=backend,
    )


def prepare_artifact_root(
    root: str | Path,
    manifest_value: Mapping[str, object],
    *,
    cohort: str,
    phase: str,
    block_id: str,
    backend: str,
) -> dict[str, object]:
    """Initialize an unused A/B root and publish its immutable identity."""
    root_path = Path(root)
    expected = build_artifact_root_manifest(
        manifest_value,
        cohort=cohort,
        phase=phase,
        block_id=block_id,
        backend=backend,
    )
    if root_path.is_symlink():
        raise AcceptanceDataError("artifact root cannot be a symbolic link")
    if root_path.exists() and not root_path.is_dir():
        raise AcceptanceDataError("artifact root must be a directory")
    root_path.mkdir(parents=True, exist_ok=True)
    if root_path.is_symlink():
        raise AcceptanceDataError("artifact root cannot be a symbolic link")

    marker = root_path / ARTIFACT_ROOT_MANIFEST_FILENAME
    entries = list(root_path.iterdir())
    if entries:
        if len(entries) != 1 or entries[0].name != ARTIFACT_ROOT_MANIFEST_FILENAME:
            raise AcceptanceDataError(
                "artifact root is not unused; choose a fresh empty root"
            )
        if marker.is_symlink() or not marker.is_file():
            raise AcceptanceDataError("artifact root manifest is not a regular file")
        existing = load_bounded_json(marker, byte_limit=64 * 1024)
        if existing != expected:
            raise AcceptanceDataError(
                "artifact root is already bound to a different experiment coordinate"
            )
        return expected

    write_immutable_json(marker, expected)
    if {entry.name for entry in root_path.iterdir()} != {
        ARTIFACT_ROOT_MANIFEST_FILENAME
    }:
        raise AcceptanceDataError("artifact root changed while it was initialized")
    return expected


def evaluate_acceptance(
    manifest_value: Mapping[str, object],
    rows_value: Sequence[Mapping[str, object]],
    *,
    artifact_roots: Sequence[Mapping[str, object]],
    recovery_probes: Sequence[Mapping[str, object]],
    blind_scores: Sequence[Mapping[str, object]],
    mac_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate all raw evidence and calculate the frozen acceptance decision."""
    manifest = validate_experiment_manifest(manifest_value)
    manifest_hash = str(manifest["manifest_sha256"])
    rows = [_validate_row(item, manifest) for item in rows_value]
    rows.sort(key=_row_sort_key)
    _validate_row_coverage(manifest, rows)
    roots = [_validate_artifact_root_manifest(item, manifest) for item in artifact_roots]
    roots.sort(
        key=lambda item: (
            item["phase"],
            item["cohort"],
            item["block_id"],
            item["backend"],
        )
    )
    _validate_artifact_root_coverage(manifest, roots)
    probes = [_validate_probe(item, manifest) for item in recovery_probes]
    probes.sort(key=lambda item: (item["backend"], item["probe_kind"]))
    scores = [_validate_blind_score(item, manifest) for item in blind_scores]
    scores.sort(
        key=lambda item: (
            item["sample_id"],
            item["reviewer_id"],
            item["backend"],
        )
    )
    mac = _validate_mac_evidence(mac_evidence, manifest) if mac_evidence else None

    gates: list[dict[str, object]] = []
    qualification_rows = [item for item in rows if item["phase"] == "qualification"]
    final_rows = [item for item in rows if item["phase"] == "final"]
    production_rows = [item for item in final_rows if item["cohort"] == _COHORT_B]
    production_by_backend = _by_backend(production_rows)

    gates.append(_qualification_gate(qualification_rows))
    gates.append(_valid_blocks_gate(final_rows))
    gates.append(_reliability_gate(manifest, final_rows))
    gates.append(_artifact_root_gate(roots))
    gates.append(_artifact_gate(production_rows))
    gates.append(_cache_gate(production_rows))
    gates.append(_provenance_gate(manifest, rows))
    gates.append(_recovery_gate(manifest, probes, production_by_backend))
    gates.extend(_asr_gates(production_by_backend))
    gates.extend(_alignment_gates(production_by_backend))
    gates.extend(_warp_gates(production_by_backend))
    gates.append(_speaker_gate(manifest, production_by_backend))
    gates.append(_blind_gate(manifest, scores))
    latency_summary, latency_gates = _latency_gates(manifest, production_by_backend)
    gates.extend(latency_gates)
    gates.append(_mac_gate(manifest, mac))

    hard_failed = any(
        gate["severity"] == "hard" and gate["status"] != "pass" for gate in gates
    )
    optimization_failed = any(
        gate["severity"] == "optimization" and gate["status"] != "pass"
        for gate in gates
    )
    decision = (
        "blocked"
        if hard_failed
        else "experimental"
        if optimization_failed
        else "promotion-candidate"
    )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "implementation_id": manifest["implementation_id"],
        "experiment_manifest_sha256": manifest_hash,
        "input_sha256": {
            "rows": _sha256(canonical_json_bytes(rows)),
            "artifact_roots": _sha256(canonical_json_bytes(roots)),
            "recovery_probes": _sha256(canonical_json_bytes(probes)),
            "blind_scores": _sha256(canonical_json_bytes(scores)),
            "mac_evidence": (
                _sha256(canonical_json_bytes(mac)) if mac is not None else "unavailable"
            ),
        },
        "cohort_a_status": (
            "evaluated"
            if manifest["cohorts"][_COHORT_A]["enabled"]
            else "not_supported"
        ),
        "conditioning_disclosure": (
            "production-candidate baseline omits reference text while candidate uses it; "
            "quality differences cannot be attributed solely to the serving framework"
        ),
        "sample_counts": _sample_counts(rows),
        "latency": latency_summary,
        "gates": gates,
        "decision": decision,
        "mac_e2e_status": "evaluated" if mac is not None else "pending_phase_8",
    }
    report["report_sha256"] = _report_sha256(report)
    return report


def write_immutable_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """Publish canonical evidence once; an existing different file is an error."""
    destination = Path(path)
    data = canonical_json_bytes(payload) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == data:
            return
        raise FileExistsError(f"immutable evidence already exists with different bytes: {destination}")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.partial"
    try:
        with temporary.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != data:
                raise
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_bounded_json(path: str | Path, *, byte_limit: int = 32 * 1024 * 1024) -> object:
    data = _bounded_read(Path(path), byte_limit)
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcceptanceDataError(f"invalid JSON evidence: {Path(path).name}") from exc


def load_bounded_jsonl(
    path: str | Path, *, byte_limit: int = 64 * 1024 * 1024
) -> list[Mapping[str, object]]:
    data = _bounded_read(Path(path), byte_limit)
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AcceptanceDataError(f"invalid JSONL at line {line_number}") from exc
        if not isinstance(value, Mapping):
            raise AcceptanceDataError(f"JSONL line {line_number} must be an object")
        rows.append(value)
    return rows


def _validate_corpus(corpus: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if not isinstance(corpus, (list, tuple)) or len(corpus) < 100:
        raise ExperimentDesignError("final corpus must contain at least 100 samples")
    expected = {
        "sample_id",
        "request_id",
        "request_payload_sha256",
        "text",
        "flow_id",
        "tempo_bpm",
        "token_count",
        "seed",
        "expected_frame_count",
    }
    result: list[dict[str, object]] = []
    sample_ids: set[str] = set()
    request_ids: set[str] = set()
    for index, raw in enumerate(corpus):
        sample = _canonical_mapping(raw, f"corpus sample {index}")
        _exact_keys(sample, expected, f"corpus sample {index}")
        _require_id(sample["sample_id"], "sample id")
        _require_sha256(sample["request_id"], "request id")
        _require_sha256(sample["request_payload_sha256"], "request payload")
        for key in ("text", "flow_id"):
            value = sample[key]
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
                or len(value) > 4096
            ):
                raise ExperimentDesignError(f"corpus {key} is invalid")
        _require_positive_number(sample["tempo_bpm"], "tempo")
        _require_positive_int(sample["token_count"], "token count")
        _require_nonnegative_int(sample["seed"], "sample seed")
        _require_positive_int(sample["expected_frame_count"], "expected frame count")
        sample_id = str(sample["sample_id"])
        request_id = str(sample["request_id"])
        if sample_id in sample_ids or request_id in request_ids:
            raise ExperimentDesignError("corpus sample and request ids must be unique")
        sample_ids.add(sample_id)
        request_ids.add(request_id)
        result.append(sample)
    result.sort(key=lambda item: str(item["sample_id"]))
    return result


def _validate_pins(pins_value: Mapping[str, object]) -> dict[str, object]:
    pins = _canonical_mapping(pins_value, "pins")
    _exact_keys(
        pins,
        {"streammuse", "gpu", "model", "generation", "producers"},
        "pins",
    )
    streammuse = _require_mapping(pins["streammuse"], "StreamMUSE pin")
    _exact_keys(streammuse, {"commit", "patch_sha256"}, "StreamMUSE pin")
    _require_pin(streammuse["commit"], "StreamMUSE commit")
    _require_sha256(streammuse["patch_sha256"], "StreamMUSE patch")
    gpu = _require_mapping(pins["gpu"], "GPU pin")
    _exact_keys(gpu, {"physical_id", "runtime_image_digest"}, "GPU pin")
    _require_pin(gpu["physical_id"], "physical GPU")
    digest = gpu["runtime_image_digest"]
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ExperimentDesignError("runtime image digest must be an immutable sha256 digest")
    model = _require_mapping(pins["model"], "model pin")
    _exact_keys(model, {"id", "revision"}, "model pin")
    _require_pin(model["id"], "model id")
    _require_pin(model["revision"], "model revision")
    generation = _require_mapping(pins["generation"], "generation pin")
    _exact_keys(
        generation,
        {"settings_sha256", "seed_policy_version"},
        "generation pin",
    )
    _require_sha256(generation["settings_sha256"], "generation settings")
    _require_pin(generation["seed_policy_version"], "seed policy version")
    producers = _require_mapping(pins["producers"], "producer pins")
    if set(producers) != {_COHORT_A, _COHORT_B}:
        raise ExperimentDesignError("producer pins must be separated by cohort")
    if producers[_COHORT_B] is None:
        raise ExperimentDesignError("production-candidate producer pins are mandatory")
    for cohort_name in (_COHORT_A, _COHORT_B):
        cohort_producers = producers[cohort_name]
        if cohort_producers is None:
            continue
        cohort_producers = _require_mapping(
            cohort_producers, f"{cohort_name} producer pins"
        )
        if set(cohort_producers) != set(_BACKENDS):
            raise ExperimentDesignError(
                f"{cohort_name} producer pins must contain baseline and candidate"
            )
        for backend in _BACKENDS:
            _validate_producer_pin(
                _require_mapping(
                    cohort_producers[backend],
                    f"{cohort_name} {backend} producer",
                ),
                cohort_name=cohort_name,
                backend=backend,
            )
    _reject_sensitive_values(pins, path="pins")
    return pins


def _validate_producer_pin(
    producer: Mapping[str, object], *, cohort_name: str, backend: str
) -> None:
    _exact_keys(
        producer,
        {
            "producer_fingerprint",
            "sidecar_backend",
            "model_revision",
            "config_sha256",
            "reference_audio_sha256",
            "reference_text_sha256",
            "runtime_environment_sha256",
            "launch_manifest_sha256",
            "health_identity",
            "health_version",
        },
        f"{cohort_name} {backend} producer",
    )
    for key in (
        "producer_fingerprint",
        "config_sha256",
        "reference_audio_sha256",
        "runtime_environment_sha256",
        "launch_manifest_sha256",
    ):
        _require_sha256(producer[key], f"{cohort_name} {backend} {key}")
    reference_text = producer["reference_text_sha256"]
    if reference_text != "unavailable":
        _require_sha256(
            reference_text, f"{cohort_name} {backend} reference text"
        )
    expected_backend = "inprocess" if backend == "baseline" else "sglang-omni"
    if producer["sidecar_backend"] != expected_backend:
        raise ExperimentDesignError(
            f"{cohort_name} {backend} sidecar backend is invalid"
        )
    for key in ("model_revision", "health_identity", "health_version"):
        _require_pin(producer[key], f"{cohort_name} {backend} {key}")
    if (
        cohort_name == _COHORT_A
        and producer["reference_text_sha256"] != "unavailable"
    ):
        raise ExperimentDesignError("cohort A producer pins must omit reference text")


def _build_blocks(sample_ids: Sequence[str], *, schedule_seed: int) -> list[dict[str, object]]:
    block_ids = [f"block-{index + 1}" for index in range(4)]
    orders: list[list[str]] = [
        ["baseline", "candidate"],
        ["baseline", "candidate"],
        ["candidate", "baseline"],
        ["candidate", "baseline"],
    ]
    random.Random(schedule_seed ^ 0x5A17).shuffle(orders)
    partitions = [[] for _ in range(4)]
    for index, sample_id in enumerate(sample_ids):
        partitions[index % 4].append(sample_id)
    return [
        {
            "block_id": block_id,
            "sample_ids": partitions[index],
            "backend_order": orders[index],
        }
        for index, block_id in enumerate(block_ids)
    ]


def _validate_schedule(schedule: Mapping[str, object], sample_ids: set[str]) -> None:
    _exact_keys(
        schedule,
        {
            "seed",
            "blocks",
            "artifact_root_template",
            "cold_start_included_in_warm_distribution",
            "warmup_included_in_warm_distribution",
            "invalidate_block_on_competing_gpu_process",
        },
        "schedule",
    )
    _require_nonnegative_int(schedule["seed"], "schedule seed")
    if (
        schedule["cold_start_included_in_warm_distribution"] is not False
        or schedule["warmup_included_in_warm_distribution"] is not False
        or schedule["invalidate_block_on_competing_gpu_process"] is not True
    ):
        raise ExperimentDesignError("warm-distribution or invalidation policy changed")
    blocks = schedule["blocks"]
    if not isinstance(blocks, list) or len(blocks) != 4:
        raise ExperimentDesignError("schedule must contain exactly four blocks")
    observed: list[str] = []
    order_counts = {("baseline", "candidate"): 0, ("candidate", "baseline"): 0}
    for index, raw in enumerate(blocks):
        block = _require_mapping(raw, f"block {index}")
        _exact_keys(block, {"block_id", "sample_ids", "backend_order"}, "block")
        if block["block_id"] != f"block-{index + 1}":
            raise ExperimentDesignError("block ids must be stable and ordered")
        ids = _id_list(block["sample_ids"], "block sample ids", minimum=25)
        observed.extend(ids)
        order = block["backend_order"]
        if not isinstance(order, list) or tuple(order) not in order_counts:
            raise ExperimentDesignError("block backend order is invalid")
        order_counts[tuple(order)] += 1
    if len(observed) != len(set(observed)) or set(observed) != sample_ids:
        raise ExperimentDesignError("block schedule must cover the corpus exactly once")
    if set(order_counts.values()) != {2}:
        raise ExperimentDesignError("backend order must be counter-balanced 2/2")


def _artifact_root_manifest(
    manifest: Mapping[str, object],
    *,
    cohort: str,
    phase: str,
    block_id: str,
    backend: str,
) -> dict[str, object]:
    _validate_artifact_root_coordinate(
        manifest,
        cohort=cohort,
        phase=phase,
        block_id=block_id,
        backend=backend,
        error=ExperimentDesignError,
    )
    producer = manifest["pins"]["producers"][cohort][backend]
    value: dict[str, object] = {
        "schema_version": ARTIFACT_ROOT_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "experiment_manifest_sha256": manifest["manifest_sha256"],
        "cohort": cohort,
        "phase": phase,
        "block_id": block_id,
        "backend": backend,
        "producer_fingerprint": producer["producer_fingerprint"],
    }
    value["root_manifest_sha256"] = _sha256(canonical_json_bytes(value))
    return value


def _validate_artifact_root_coordinate(
    manifest: Mapping[str, object],
    *,
    cohort: object,
    phase: object,
    block_id: object,
    backend: object,
    error: type[ValueError],
) -> None:
    if cohort not in {_COHORT_A, _COHORT_B}:
        raise error("artifact root cohort is invalid")
    if phase not in _PHASES:
        raise error("artifact root phase is invalid")
    if backend not in _BACKENDS:
        raise error("artifact root backend is invalid")
    _require_id(block_id, "artifact root block id", error=error)
    if not manifest["cohorts"][cohort]["enabled"]:
        raise error("artifact root targets a disabled cohort")
    if phase == "qualification":
        if cohort != _COHORT_B or block_id != "qualification":
            raise error("qualification root must target the production cohort")
        return
    block_ids = {item["block_id"] for item in manifest["schedule"]["blocks"]}
    if block_id not in block_ids:
        raise error("final artifact root block differs from the frozen schedule")


def _validate_artifact_root_manifest(
    raw: Mapping[str, object], manifest: Mapping[str, object]
) -> dict[str, object]:
    root = _canonical_mapping(raw, "artifact root manifest")
    _exact_keys(
        root,
        {
            "schema_version",
            "experiment_id",
            "experiment_manifest_sha256",
            "cohort",
            "phase",
            "block_id",
            "backend",
            "producer_fingerprint",
            "root_manifest_sha256",
        },
        "artifact root manifest",
        AcceptanceDataError,
    )
    if root["schema_version"] != ARTIFACT_ROOT_SCHEMA_VERSION:
        raise AcceptanceDataError("unsupported artifact root manifest schema")
    _require_manifest_identity(root, manifest)
    _validate_artifact_root_coordinate(
        manifest,
        cohort=root["cohort"],
        phase=root["phase"],
        block_id=root["block_id"],
        backend=root["backend"],
        error=AcceptanceDataError,
    )
    _require_sha256(
        root["producer_fingerprint"],
        "artifact root producer",
        error=AcceptanceDataError,
    )
    _require_sha256(
        root["root_manifest_sha256"],
        "artifact root manifest",
        error=AcceptanceDataError,
    )
    expected = _artifact_root_manifest(
        manifest,
        cohort=root["cohort"],
        phase=root["phase"],
        block_id=root["block_id"],
        backend=root["backend"],
    )
    if root != expected:
        raise AcceptanceDataError(
            "artifact root manifest differs from its frozen coordinate"
        )
    return root


def _validate_artifact_root_coverage(
    manifest: Mapping[str, object], roots: Sequence[Mapping[str, object]]
) -> None:
    expected = {
        (_COHORT_B, "qualification", "qualification", backend)
        for backend in _BACKENDS
    }
    for cohort, config in manifest["cohorts"].items():
        if not config["enabled"]:
            continue
        for block in manifest["schedule"]["blocks"]:
            for backend in _BACKENDS:
                expected.add((cohort, "final", block["block_id"], backend))
    observed = {
        (root["cohort"], root["phase"], root["block_id"], root["backend"])
        for root in roots
    }
    if len(observed) != len(roots):
        raise AcceptanceDataError("duplicate artifact root manifest")
    if observed != expected:
        raise AcceptanceDataError(
            "artifact root manifest coverage differs from the frozen design"
        )


def _validate_row(raw: Mapping[str, object], manifest: Mapping[str, object]) -> dict[str, object]:
    row = _canonical_mapping(raw, "A/B row")
    _exact_keys(
        row,
        {
            "schema_version",
            "experiment_id",
            "experiment_manifest_sha256",
            "cohort",
            "phase",
            "block_id",
            "sample_id",
            "backend",
            "backend_order_index",
            "request_id",
            "block_valid",
            "invalidation_reasons",
            "competing_processes",
            "success",
            "failure_kind",
            "events",
            "service_call_count",
            "timings_ms",
            "artifact",
            "quality",
            "provenance",
        },
        "A/B row",
    )
    if row["schema_version"] != ROW_SCHEMA_VERSION:
        raise AcceptanceDataError("unsupported A/B row schema")
    _require_manifest_identity(row, manifest)
    if row["cohort"] not in {_COHORT_A, _COHORT_B} or row["phase"] not in _PHASES:
        raise AcceptanceDataError("A/B row cohort or phase is invalid")
    if row["backend"] not in _BACKENDS:
        raise AcceptanceDataError("A/B row backend is invalid")
    if type(row["backend_order_index"]) is not int or row["backend_order_index"] not in {0, 1}:
        raise AcceptanceDataError("backend order index is invalid")
    _require_id(row["sample_id"], "sample id", error=AcceptanceDataError)
    _require_sha256(row["request_id"], "request id", error=AcceptanceDataError)
    if type(row["block_valid"]) is not bool or type(row["success"]) is not bool:
        raise AcceptanceDataError("row validity and success must be boolean")
    for key in ("invalidation_reasons", "competing_processes"):
        values = row[key]
        if not isinstance(values, list) or not all(
            isinstance(item, str) and item and len(item) <= 256 for item in values
        ):
            raise AcceptanceDataError(f"{key} must be a bounded string array")
    if bool(row["competing_processes"]) and row["block_valid"]:
        raise AcceptanceDataError("a block with competing GPU processes cannot be valid")
    if not row["block_valid"] and not row["invalidation_reasons"]:
        raise AcceptanceDataError("invalid blocks require an invalidation reason")
    if row["failure_kind"] is not None and (
        not isinstance(row["failure_kind"], str) or not row["failure_kind"]
    ):
        raise AcceptanceDataError("failure kind must be null or a non-empty string")
    if row["success"] == (row["failure_kind"] is not None):
        raise AcceptanceDataError("success and failure kind disagree")
    _require_nonnegative_int(row["service_call_count"], "service call count", AcceptanceDataError)
    _validate_events(row["events"])
    _validate_timings(row["timings_ms"], success=bool(row["success"]))
    _validate_artifact(row["artifact"], success=bool(row["success"]))
    _validate_quality(row["quality"], success=bool(row["success"]))
    _validate_provenance(row["provenance"])
    return row


def _validate_events(value: object) -> None:
    events = _require_mapping(value, "events", AcceptanceDataError)
    expected = {
        "crash",
        "oom",
        "hung",
        "timeout",
        "deadline_miss",
        "schema_violation",
        "wrong_producer_cache_hit",
    }
    _exact_keys(events, expected, "events", AcceptanceDataError)
    if any(type(item) is not bool for item in events.values()):
        raise AcceptanceDataError("event flags must be boolean")


def _validate_timings(value: object, *, success: bool) -> None:
    timings = _require_mapping(value, "timings", AcceptanceDataError)
    keys = {"queue", "moss", "mms", "r3", "package", "complete_server"}
    _exact_keys(timings, keys, "timings", AcceptanceDataError)
    for item in timings.values():
        if item is None and not success:
            continue
        _require_nonnegative_number(item, "timing", AcceptanceDataError)
    if success and (timings["moss"] <= 0 or timings["complete_server"] <= 0):
        raise AcceptanceDataError("successful rows require positive MOSS and server timing")


def _validate_artifact(value: object, *, success: bool) -> None:
    artifact = _require_mapping(value, "artifact", AcceptanceDataError)
    _exact_keys(
        artifact,
        {
            "cache_hit",
            "sidecar_valid",
            "package_valid",
            "public_schema_valid",
            "exact_frame_count",
            "expected_frame_count",
            "final_frame_count",
        },
        "artifact",
        AcceptanceDataError,
    )
    for key in (
        "cache_hit",
        "sidecar_valid",
        "package_valid",
        "public_schema_valid",
        "exact_frame_count",
    ):
        if type(artifact[key]) is not bool:
            raise AcceptanceDataError(f"artifact {key} must be boolean")
    for key in ("expected_frame_count", "final_frame_count"):
        if artifact[key] is None and not success:
            continue
        _require_positive_int(artifact[key], key, AcceptanceDataError)


def _validate_quality(value: object, *, success: bool) -> None:
    quality = _require_mapping(value, "quality", AcceptanceDataError)
    _exact_keys(
        quality,
        {
            "asr_reference_words",
            "asr_substitutions",
            "asr_deletions",
            "asr_insertions",
            "mms_coverage",
            "mms_confidence",
            "mms_below_hard_floor",
            "local_stretch_ratios",
            "fallback_count",
            "speaker_similarity",
        },
        "quality",
        AcceptanceDataError,
    )
    for key in (
        "asr_reference_words",
        "asr_substitutions",
        "asr_deletions",
        "asr_insertions",
        "fallback_count",
    ):
        if quality[key] is None and not success:
            continue
        _require_nonnegative_int(quality[key], key, AcceptanceDataError)
    if success and quality["asr_reference_words"] <= 0:
        raise AcceptanceDataError("successful rows require positive ASR reference words")
    for key in ("mms_coverage", "mms_confidence"):
        if quality[key] is None and not success:
            continue
        _require_unit_number(quality[key], key)
    if quality["mms_below_hard_floor"] is not None or success:
        if type(quality["mms_below_hard_floor"]) is not bool:
            raise AcceptanceDataError("MMS hard-floor flag must be boolean")
    ratios = quality["local_stretch_ratios"]
    if not isinstance(ratios, list) or (success and not ratios):
        raise AcceptanceDataError("successful rows require local stretch ratios")
    for ratio in ratios:
        _require_positive_number(ratio, "local stretch ratio", AcceptanceDataError)
    similarity = quality["speaker_similarity"]
    if similarity is not None:
        _require_unit_number(similarity, "speaker similarity")


def _validate_provenance(value: object) -> None:
    provenance = _require_mapping(value, "provenance", AcceptanceDataError)
    _exact_keys(
        provenance,
        {
            "producer_fingerprint",
            "health_producer_fingerprint",
            "sidecar_backend",
            "health_backend",
            "model_revision",
            "config_sha256",
            "reference_audio_sha256",
            "reference_text_sha256",
            "health_identity",
            "health_version",
            "artifact_root_manifest_sha256",
        },
        "provenance",
        AcceptanceDataError,
    )
    for key in (
        "producer_fingerprint",
        "health_producer_fingerprint",
        "config_sha256",
        "reference_audio_sha256",
        "artifact_root_manifest_sha256",
    ):
        _require_sha256(provenance[key], key, error=AcceptanceDataError)
    if provenance["reference_text_sha256"] != "unavailable":
        _require_sha256(
            provenance["reference_text_sha256"],
            "reference text",
            error=AcceptanceDataError,
        )
    for key in (
        "sidecar_backend",
        "health_backend",
        "model_revision",
        "health_identity",
        "health_version",
    ):
        _require_pin(provenance[key], key, error=AcceptanceDataError)


def _validate_row_coverage(manifest: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> None:
    corpus = manifest["corpus"]
    samples = {item["sample_id"]: item for item in corpus["samples"]}
    blocks = {
        sample_id: (block["block_id"], block["backend_order"])
        for block in manifest["schedule"]["blocks"]
        for sample_id in block["sample_ids"]
    }
    qualification = set(corpus["qualification_sample_ids"])
    enabled_cohorts = {
        name for name, config in manifest["cohorts"].items() if config["enabled"]
    }
    expected: set[tuple[str, str, str, str]] = set()
    for sample_id in qualification:
        for backend in _BACKENDS:
            expected.add(("qualification", _COHORT_B, sample_id, backend))
    for cohort in enabled_cohorts:
        for sample_id in samples:
            for backend in _BACKENDS:
                expected.add(("final", cohort, sample_id, backend))

    observed: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (row["phase"], row["cohort"], row["sample_id"], row["backend"])
        if key in observed:
            raise AcceptanceDataError(f"duplicate A/B row: {key}")
        observed.add(key)
        if row["sample_id"] not in samples:
            raise AcceptanceDataError("A/B row references an unknown sample")
        sample = samples[row["sample_id"]]
        if row["request_id"] != sample["request_id"]:
            raise AcceptanceDataError("A/B row request id differs from frozen corpus")
        artifact = row["artifact"]
        if artifact["expected_frame_count"] != sample["expected_frame_count"]:
            raise AcceptanceDataError("A/B row expected frames differ from frozen corpus")
        if row["phase"] == "qualification":
            if row["cohort"] != _COHORT_B or row["sample_id"] not in qualification:
                raise AcceptanceDataError("qualification row is outside the frozen subset")
            if row["block_id"] != "qualification":
                raise AcceptanceDataError("qualification rows must use block_id=qualification")
        else:
            block_id, order = blocks[row["sample_id"]]
            if row["cohort"] not in enabled_cohorts:
                raise AcceptanceDataError("row supplied for a disabled cohort")
            if row["block_id"] != block_id:
                raise AcceptanceDataError("A/B row block differs from frozen schedule")
            if row["backend_order_index"] != order.index(row["backend"]):
                raise AcceptanceDataError("A/B row backend order differs from frozen schedule")
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise AcceptanceDataError(
            f"A/B row coverage differs from frozen design: missing={len(missing)} extra={len(extra)}"
        )


def _validate_probe(raw: Mapping[str, object], manifest: Mapping[str, object]) -> dict[str, object]:
    probe = _canonical_mapping(raw, "recovery probe")
    _exact_keys(
        probe,
        {
            "schema_version",
            "experiment_id",
            "experiment_manifest_sha256",
            "backend",
            "probe_kind",
            "producer_fingerprint",
            "queue_recovered",
            "upstream_request_released",
            "next_short_request_ms",
            "normal_warm_p95_ms",
        },
        "recovery probe",
        AcceptanceDataError,
    )
    if probe["schema_version"] != PROBE_SCHEMA_VERSION:
        raise AcceptanceDataError("unsupported recovery-probe schema")
    _require_manifest_identity(probe, manifest)
    if probe["backend"] not in _BACKENDS or probe["probe_kind"] not in _PROBE_KINDS:
        raise AcceptanceDataError("recovery probe backend or kind is invalid")
    _require_sha256(probe["producer_fingerprint"], "probe producer", error=AcceptanceDataError)
    for key in ("queue_recovered", "upstream_request_released"):
        if type(probe[key]) is not bool:
            raise AcceptanceDataError(f"probe {key} must be boolean")
    for key in ("next_short_request_ms", "normal_warm_p95_ms"):
        _require_positive_number(probe[key], key, AcceptanceDataError)
    return probe


def _validate_blind_score(raw: Mapping[str, object], manifest: Mapping[str, object]) -> dict[str, object]:
    score = _canonical_mapping(raw, "blind score")
    _exact_keys(
        score,
        {
            "schema_version",
            "experiment_id",
            "experiment_manifest_sha256",
            "sample_id",
            "reviewer_id",
            "presentation_id",
            "backend",
            "scores",
            "critical_artifact",
        },
        "blind score",
        AcceptanceDataError,
    )
    if score["schema_version"] != BLIND_SCORE_SCHEMA_VERSION:
        raise AcceptanceDataError("unsupported blind-score schema")
    _require_manifest_identity(score, manifest)
    for key in ("sample_id", "reviewer_id", "presentation_id"):
        _require_id(score[key], key, error=AcceptanceDataError)
    if score["backend"] not in _BACKENDS or type(score["critical_artifact"]) is not bool:
        raise AcceptanceDataError("blind score backend or artifact flag is invalid")
    values = _require_mapping(score["scores"], "blind score values", AcceptanceDataError)
    _exact_keys(values, set(_SCORE_DIMENSIONS), "blind score values", AcceptanceDataError)
    for value in values.values():
        if type(value) is not int or not 1 <= value <= 5:
            raise AcceptanceDataError("blind scores must be integer values from 1 to 5")
    return score


def _validate_mac_evidence(
    raw: Mapping[str, object], manifest: Mapping[str, object]
) -> dict[str, object]:
    evidence = _canonical_mapping(raw, "Mac E2E evidence")
    _exact_keys(
        evidence,
        {
            "schema_version",
            "experiment_id",
            "experiment_manifest_sha256",
            "baseline",
            "candidate",
        },
        "Mac E2E evidence",
        AcceptanceDataError,
    )
    if evidence["schema_version"] != MAC_EVIDENCE_SCHEMA_VERSION:
        raise AcceptanceDataError("unsupported Mac E2E schema")
    _require_manifest_identity(evidence, manifest)
    for backend in _BACKENDS:
        item = _require_mapping(evidence[backend], f"Mac {backend}", AcceptanceDataError)
        _exact_keys(
            item,
            {
                "producer_fingerprint",
                "deadline_misses",
                "playback_underruns",
                "post_cancel_gpu_busy_requests",
            },
            f"Mac {backend}",
            AcceptanceDataError,
        )
        _require_sha256(item["producer_fingerprint"], "Mac producer", error=AcceptanceDataError)
        for key in ("deadline_misses", "playback_underruns", "post_cancel_gpu_busy_requests"):
            _require_nonnegative_int(item[key], key, AcceptanceDataError)
    return evidence


def _qualification_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    observed = {}
    passed = True
    for backend in _BACKENDS:
        selected = [row for row in rows if row["backend"] == backend]
        failures = sum(
            not _row_clean(row)
            or not all(
                row["artifact"][key]
                for key in (
                    "sidecar_valid",
                    "package_valid",
                    "public_schema_valid",
                    "exact_frame_count",
                )
            )
            for row in selected
        )
        service_calls = sum(row["service_call_count"] for row in selected)
        expected_calls = len(selected) if backend == "candidate" else 0
        observed[backend] = {
            "rows": len(selected),
            "failures": failures,
            "service_calls": service_calls,
        }
        passed &= (
            len(selected) == 30
            and failures == 0
            and service_calls == expected_calls
        )
    return _gate(
        "qualification_30_of_30",
        passed,
        "hard",
        observed,
        "30 clean, uncached successful rows per backend",
    )


def _valid_blocks_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    invalid = sorted(
        {str(row["block_id"]) for row in rows if not row["block_valid"]}
    )
    competing = sum(bool(row["competing_processes"]) for row in rows)
    return _gate(
        "counterbalanced_blocks_valid",
        not invalid and competing == 0,
        "hard",
        {"invalid_blocks": invalid, "rows_with_competing_processes": competing},
        "four frozen blocks, no invalidation or competing GPU process",
    )


def _reliability_gate(
    manifest: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    observed: dict[str, object] = {}
    passed = True
    for cohort, config in manifest["cohorts"].items():
        if not config["enabled"]:
            continue
        observed[cohort] = {}
        for backend in _BACKENDS:
            selected = [
                row for row in rows if row["cohort"] == cohort and row["backend"] == backend
            ]
            failures = sum(not _row_clean(row) for row in selected)
            observed[cohort][backend] = {"rows": len(selected), "failures": failures}
            passed &= len(selected) >= 100 and failures == 0
    return _gate(
        "final_reliability",
        passed,
        "hard",
        observed,
        ">=100/100 clean final rows per enabled cohort and backend",
    )


def _artifact_root_gate(
    roots: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return _gate(
        "artifact_root_isolation",
        True,
        "hard",
        {"root_manifests": len(roots)},
        "one immutable fresh-root identity per qualification/final coordinate",
    )


def _artifact_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    failed = [
        row["sample_id"]
        for row in rows
        if not all(
            row["artifact"][key]
            for key in (
                "sidecar_valid",
                "package_valid",
                "public_schema_valid",
                "exact_frame_count",
            )
        )
        or row["artifact"]["final_frame_count"] != row["artifact"]["expected_frame_count"]
    ]
    return _gate(
        "artifact_and_public_schema",
        not failed,
        "hard",
        {"failed_rows": len(failed)},
        "100% sidecar/package/schema/exact-frame validation",
    )


def _cache_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    candidate = [row for row in rows if row["backend"] == "candidate"]
    cache_hits = sum(row["artifact"]["cache_hit"] for row in rows)
    wrong_hits = sum(row["events"]["wrong_producer_cache_hit"] for row in rows)
    service_calls = sum(row["service_call_count"] for row in candidate)
    passed = cache_hits == 0 and wrong_hits == 0 and service_calls == len(candidate)
    return _gate(
        "uncached_candidate_calls",
        passed,
        "hard",
        {
            "cache_hits": cache_hits,
            "wrong_producer_cache_hits": wrong_hits,
            "candidate_service_calls": service_calls,
            "candidate_rows": len(candidate),
        },
        "zero cache hits and exactly one SGLang call per candidate row",
    )


def _provenance_gate(
    manifest: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    mismatches = 0
    for row in rows:
        expected = manifest["pins"]["producers"][row["cohort"]][row["backend"]]
        expected_root = _artifact_root_manifest(
            manifest,
            cohort=row["cohort"],
            phase=row["phase"],
            block_id=row["block_id"],
            backend=row["backend"],
        )
        actual = row["provenance"]
        expected_values = {
            "producer_fingerprint": expected["producer_fingerprint"],
            "health_producer_fingerprint": expected["producer_fingerprint"],
            "sidecar_backend": expected["sidecar_backend"],
            "health_backend": expected["sidecar_backend"],
            "model_revision": expected["model_revision"],
            "config_sha256": expected["config_sha256"],
            "reference_audio_sha256": expected["reference_audio_sha256"],
            "reference_text_sha256": expected["reference_text_sha256"],
            "health_identity": expected["health_identity"],
            "health_version": expected["health_version"],
            "artifact_root_manifest_sha256": expected_root[
                "root_manifest_sha256"
            ],
        }
        if any(actual[key] != value for key, value in expected_values.items()):
            mismatches += 1
    return _gate(
        "provenance_identity",
        mismatches == 0,
        "hard",
        {"mismatched_rows": mismatches, "rows": len(rows)},
        "health, sidecar, model, config, reference and producer pins all match",
    )


def _recovery_gate(
    manifest: Mapping[str, object],
    probes: Sequence[Mapping[str, object]],
    by_backend: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    keys: set[tuple[str, str]] = set()
    failed = 0
    for probe in probes:
        key = (str(probe["backend"]), str(probe["probe_kind"]))
        if key in keys:
            raise AcceptanceDataError(f"duplicate recovery probe: {key}")
        keys.add(key)
        expected_fingerprint = manifest["pins"]["producers"][_COHORT_B][
            probe["backend"]
        ]["producer_fingerprint"]
        if (
            probe["producer_fingerprint"] != expected_fingerprint
            or not probe["queue_recovered"]
            or not probe["upstream_request_released"]
            or probe["next_short_request_ms"] > probe["normal_warm_p95_ms"]
        ):
            failed += 1
    expected = {(backend, kind) for backend in _BACKENDS for kind in _PROBE_KINDS}
    return _gate(
        "timeout_disconnect_cancellation_recovery",
        keys == expected and failed == 0,
        "hard",
        {"probe_count": len(probes), "failed_probes": failed, "missing_probes": len(expected - keys)},
        "both backends recover after timeout/disconnect/cancellation within warm p95",
    )


def _asr_gates(by_backend: Mapping[str, Sequence[Mapping[str, object]]]) -> list[dict[str, object]]:
    required = (
        "asr_reference_words",
        "asr_substitutions",
        "asr_deletions",
        "asr_insertions",
    )
    incomplete = {
        backend: sum(
            not row["success"]
            or any(row["quality"][key] is None for key in required)
            for row in rows
        )
        for backend, rows in by_backend.items()
    }
    if any(incomplete.values()):
        observed = {"incomplete_rows": incomplete}
        return [
            _gate(
                "asr_wer",
                False,
                "hard",
                observed,
                "candidate corpus WER <= baseline + 0.01",
            ),
            _gate(
                "asr_deletion_rate",
                False,
                "hard",
                observed,
                "candidate deletion rate <= baseline + 0.005",
            ),
        ]
    metrics: dict[str, dict[str, float | int]] = {}
    for backend in _BACKENDS:
        quality = [row["quality"] for row in by_backend[backend]]
        references = sum(item["asr_reference_words"] for item in quality)
        substitutions = sum(item["asr_substitutions"] for item in quality)
        deletions = sum(item["asr_deletions"] for item in quality)
        insertions = sum(item["asr_insertions"] for item in quality)
        metrics[backend] = {
            "reference_words": references,
            "wer": (substitutions + deletions + insertions) / references,
            "deletion_rate": deletions / references,
            "substitution_rate": substitutions / references,
        }
    wer_delta = metrics["candidate"]["wer"] - metrics["baseline"]["wer"]
    deletion_delta = metrics["candidate"]["deletion_rate"] - metrics["baseline"]["deletion_rate"]
    return [
        _gate(
            "asr_wer",
            wer_delta <= 0.01,
            "hard",
            {"by_backend": metrics, "candidate_minus_baseline": wer_delta},
            "candidate corpus WER <= baseline + 0.01",
        ),
        _gate(
            "asr_deletion_rate",
            deletion_delta <= 0.005,
            "hard",
            {"candidate_minus_baseline": deletion_delta},
            "candidate deletion rate <= baseline + 0.005",
        ),
    ]


def _alignment_gates(
    by_backend: Mapping[str, Sequence[Mapping[str, object]]]
) -> list[dict[str, object]]:
    incomplete = {
        backend: sum(
            not row["success"]
            or row["quality"]["mms_coverage"] is None
            or row["quality"]["mms_confidence"] is None
            or row["quality"]["mms_below_hard_floor"] is None
            for row in rows
        )
        for backend, rows in by_backend.items()
    }
    if any(incomplete.values()):
        observed = {"incomplete_rows": incomplete}
        return [
            _gate("mms_median_coverage", False, "hard", observed, "candidate median coverage >= baseline - 0.01"),
            _gate("mms_median_confidence", False, "hard", observed, "candidate median confidence >= baseline - 0.02"),
            _gate("mms_no_new_hard_floor_failures", False, "hard", observed, "candidate introduces no new hard-floor failure"),
        ]
    coverage = {
        backend: statistics.median(row["quality"]["mms_coverage"] for row in rows)
        for backend, rows in by_backend.items()
    }
    confidence = {
        backend: statistics.median(row["quality"]["mms_confidence"] for row in rows)
        for backend, rows in by_backend.items()
    }
    below = {
        backend: {
            row["sample_id"]
            for row in rows
            if row["quality"]["mms_below_hard_floor"]
        }
        for backend, rows in by_backend.items()
    }
    new_below = sorted(below["candidate"] - below["baseline"])
    return [
        _gate(
            "mms_median_coverage",
            coverage["candidate"] - coverage["baseline"] >= -0.01,
            "hard",
            {"by_backend": coverage, "candidate_minus_baseline": coverage["candidate"] - coverage["baseline"]},
            "candidate median coverage >= baseline - 0.01",
        ),
        _gate(
            "mms_median_confidence",
            confidence["candidate"] - confidence["baseline"] >= -0.02,
            "hard",
            {"by_backend": confidence, "candidate_minus_baseline": confidence["candidate"] - confidence["baseline"]},
            "candidate median confidence >= baseline - 0.02",
        ),
        _gate(
            "mms_no_new_hard_floor_failures",
            not new_below,
            "hard",
            {"new_candidate_failures": new_below},
            "candidate introduces no new sample below the existing hard floor",
        ),
    ]


def _warp_gates(by_backend: Mapping[str, Sequence[Mapping[str, object]]]) -> list[dict[str, object]]:
    incomplete = {
        backend: sum(
            not row["success"]
            or not row["quality"]["local_stretch_ratios"]
            or row["quality"]["fallback_count"] is None
            for row in rows
        )
        for backend, rows in by_backend.items()
    }
    if any(incomplete.values()):
        observed = {"incomplete_rows": incomplete}
        return [
            _gate("r3_local_stretch_p95", False, "hard", observed, "candidate local-stretch p95 <= 1.10x baseline"),
            _gate("r3_fallback_count", False, "hard", observed, "candidate fallback count <= baseline + max(1, 1% * N)"),
        ]
    ratios = {
        backend: [ratio for row in rows for ratio in row["quality"]["local_stretch_ratios"]]
        for backend, rows in by_backend.items()
    }
    p95 = {backend: _percentile(values, 0.95) for backend, values in ratios.items()}
    fallback = {
        backend: sum(row["quality"]["fallback_count"] for row in rows)
        for backend, rows in by_backend.items()
    }
    allowance = max(1.0, 0.01 * len(by_backend["candidate"]))
    return [
        _gate(
            "r3_local_stretch_p95",
            p95["candidate"] <= p95["baseline"] * 1.10,
            "hard",
            {"by_backend": p95, "candidate_to_baseline_ratio": p95["candidate"] / p95["baseline"]},
            "candidate local-stretch p95 <= 1.10x baseline",
        ),
        _gate(
            "r3_fallback_count",
            fallback["candidate"] <= fallback["baseline"] + allowance,
            "hard",
            {"by_backend": fallback, "allowed_candidate_max": fallback["baseline"] + allowance},
            "candidate fallback count <= baseline + max(1, 1% * N)",
        ),
    ]


def _speaker_gate(
    manifest: Mapping[str, object], by_backend: Mapping[str, Sequence[Mapping[str, object]]]
) -> dict[str, object]:
    values = {
        backend: [
            row["quality"]["speaker_similarity"]
            for row in rows
            if row["quality"]["speaker_similarity"] is not None
        ]
        for backend, rows in by_backend.items()
    }
    hard = bool(manifest["speaker_similarity_hard_gate"])
    complete = all(len(values[backend]) == len(by_backend[backend]) for backend in _BACKENDS)
    medians = {
        backend: statistics.median(items) if items else None for backend, items in values.items()
    }
    passed = (
        complete
        and medians["candidate"] is not None
        and medians["baseline"] is not None
        and medians["candidate"] >= medians["baseline"] - 0.02
    )
    if not hard:
        return {
            "name": "speaker_similarity",
            "status": "pass" if complete and passed else "report-only",
            "severity": "report-only",
            "observed": {"complete": complete, "medians": medians},
            "threshold": "hard only when frozen in the experiment manifest",
        }
    return _gate(
        "speaker_similarity",
        passed,
        "hard",
        {"complete": complete, "medians": medians},
        "candidate median >= baseline - 0.02",
    )


def _blind_gate(
    manifest: Mapping[str, object], scores: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    blind_ids = set(manifest["corpus"]["blind_sample_ids"])
    grouped: dict[tuple[str, str, str], Mapping[str, object]] = {}
    presentation_ids: set[str] = set()
    for score in scores:
        if score["sample_id"] not in blind_ids:
            raise AcceptanceDataError("blind score references a non-frozen sample")
        key = (score["sample_id"], score["reviewer_id"], score["backend"])
        if key in grouped or score["presentation_id"] in presentation_ids:
            raise AcceptanceDataError("duplicate blind score or presentation id")
        grouped[key] = score
        presentation_ids.add(score["presentation_id"])
    complete = True
    reviewers_by_sample: dict[str, set[str]] = {}
    for sample_id in blind_ids:
        baseline_reviewers = {
            reviewer
            for candidate_sample, reviewer, backend in grouped
            if candidate_sample == sample_id and backend == "baseline"
        }
        candidate_reviewers = {
            reviewer
            for candidate_sample, reviewer, backend in grouped
            if candidate_sample == sample_id and backend == "candidate"
        }
        paired_reviewers = baseline_reviewers & candidate_reviewers
        reviewers_by_sample[sample_id] = paired_reviewers
        complete &= (
            len(paired_reviewers) >= 2
            and paired_reviewers == baseline_reviewers
            and paired_reviewers == candidate_reviewers
        )
    medians: dict[str, dict[str, float | None]] = {backend: {} for backend in _BACKENDS}
    deltas: dict[str, float | None] = {}
    for dimension in _SCORE_DIMENSIONS:
        for backend in _BACKENDS:
            values = [
                score["scores"][dimension]
                for score in scores
                if score["backend"] == backend
            ]
            medians[backend][dimension] = statistics.median(values) if values else None
        if medians["baseline"][dimension] is None or medians["candidate"][dimension] is None:
            deltas[dimension] = None
        else:
            deltas[dimension] = medians["candidate"][dimension] - medians["baseline"][dimension]
    new_critical = []
    for sample_id, reviewers in reviewers_by_sample.items():
        if len(reviewers) < 2:
            continue
        baseline_consensus = all(
            grouped[(sample_id, reviewer, "baseline")]["critical_artifact"]
            for reviewer in reviewers
        )
        candidate_consensus = all(
            grouped[(sample_id, reviewer, "candidate")]["critical_artifact"]
            for reviewer in reviewers
        )
        if candidate_consensus and not baseline_consensus:
            new_critical.append(sample_id)
    passed = (
        complete
        and all(delta is not None and delta >= -0.5 for delta in deltas.values())
        and not new_critical
    )
    return _gate(
        "blind_listening",
        passed,
        "hard",
        {
            "complete": complete,
            "score_medians": medians,
            "candidate_minus_baseline": deltas,
            "new_consensus_critical_artifacts": sorted(new_critical),
        },
        "20 frozen pairs, >=2 reviewers, median deltas >= -0.5, no new critical artifact",
    )


def _latency_gates(
    manifest: Mapping[str, object], by_backend: Mapping[str, Sequence[Mapping[str, object]]]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary: dict[str, object] = {}
    incomplete = {
        backend: sum(
            not row["success"]
            or any(row["timings_ms"][metric] is None for metric in ("moss", "complete_server"))
            for row in rows
        )
        for backend, rows in by_backend.items()
    }
    if any(incomplete.values()):
        observed = {"incomplete_rows": incomplete}
        return (
            {"status": "unavailable", **observed},
            [
                _gate("primary_moss_latency", False, "optimization", observed, "paired median delta < 0 ms and bootstrap 95% CI upper < 0 ms"),
                _gate("moss_tail_latency", False, "optimization", observed, "candidate warm MOSS p95 <= 1.10x baseline"),
                _gate("complete_server_tail_latency", False, "optimization", observed, "candidate complete H200 p95 <= 1.05x baseline"),
            ],
        )
    for metric in ("moss", "complete_server", "queue", "mms", "r3", "package"):
        summary[metric] = {
            backend: _distribution([row["timings_ms"][metric] for row in rows])
            for backend, rows in by_backend.items()
        }
    baseline = {row["sample_id"]: row for row in by_backend["baseline"]}
    candidate = {row["sample_id"]: row for row in by_backend["candidate"]}
    deltas = [
        candidate[sample_id]["timings_ms"]["moss"]
        - baseline[sample_id]["timings_ms"]["moss"]
        for sample_id in sorted(baseline)
    ]
    statistics_config = manifest["statistics"]
    bootstrap = _paired_median_bootstrap(
        deltas,
        seed=statistics_config["bootstrap_seed"],
        resamples=statistics_config["bootstrap_resamples"],
    )
    summary["paired_moss_candidate_minus_baseline_ms"] = bootstrap
    moss_p95 = summary["moss"]
    complete_p95 = summary["complete_server"]
    gates = [
        _gate(
            "primary_moss_latency",
            bootstrap["estimate"] < 0 and bootstrap["ci95"][1] < 0,
            "optimization",
            bootstrap,
            "paired median delta < 0 ms and bootstrap 95% CI upper < 0 ms",
        ),
        _gate(
            "moss_tail_latency",
            moss_p95["candidate"]["p95"] <= moss_p95["baseline"]["p95"] * 1.10,
            "optimization",
            {"by_backend": {backend: values["p95"] for backend, values in moss_p95.items()}},
            "candidate warm MOSS p95 <= 1.10x baseline",
        ),
        _gate(
            "complete_server_tail_latency",
            complete_p95["candidate"]["p95"]
            <= complete_p95["baseline"]["p95"] * 1.05,
            "optimization",
            {"by_backend": {backend: values["p95"] for backend, values in complete_p95.items()}},
            "candidate complete H200 p95 <= 1.05x baseline",
        ),
    ]
    return summary, gates


def _mac_gate(
    manifest: Mapping[str, object], evidence: Mapping[str, object] | None
) -> dict[str, object]:
    if evidence is None:
        return {
            "name": "mac_realtime_e2e",
            "status": "pending_phase_8",
            "severity": "phase-8",
            "observed": None,
            "threshold": "candidate misses/underruns <= baseline and zero post-cancel GPU work",
        }
    baseline = evidence["baseline"]
    candidate = evidence["candidate"]
    fingerprints_match = all(
        evidence[backend]["producer_fingerprint"]
        == manifest["pins"]["producers"][_COHORT_B][backend][
            "producer_fingerprint"
        ]
        for backend in _BACKENDS
    )
    passed = (
        fingerprints_match
        and candidate["deadline_misses"] <= baseline["deadline_misses"]
        and candidate["playback_underruns"] <= baseline["playback_underruns"]
        and candidate["post_cancel_gpu_busy_requests"] == 0
    )
    return _gate(
        "mac_realtime_e2e",
        passed,
        "hard",
        {"baseline": baseline, "candidate": candidate},
        "candidate misses/underruns <= baseline and zero post-cancel GPU work",
    )


def _row_clean(row: Mapping[str, object]) -> bool:
    return (
        row["success"]
        and row["block_valid"]
        and not row["artifact"]["cache_hit"]
        and not any(row["events"].values())
    )


def _by_backend(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    return {backend: [row for row in rows if row["backend"] == backend] for backend in _BACKENDS}


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise AcceptanceDataError("cannot summarize an empty distribution")
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _paired_median_bootstrap(
    deltas: Sequence[float], *, seed: int, resamples: int
) -> dict[str, object]:
    if not deltas:
        raise AcceptanceDataError("paired latency distribution is empty")
    rng = random.Random(seed)
    count = len(deltas)
    estimates = [
        statistics.median(deltas[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    return {
        "pair_count": count,
        "estimate": statistics.median(deltas),
        "ci95": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
        "resamples": resamples,
        "seed": seed,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise AcceptanceDataError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _gate(
    name: str,
    passed: bool,
    severity: str,
    observed: object,
    threshold: str,
) -> dict[str, object]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "severity": severity,
        "observed": observed,
        "threshold": threshold,
    }


def _sample_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for phase in _PHASES:
        result[phase] = {}
        for cohort in (_COHORT_A, _COHORT_B):
            selected = [row for row in rows if row["phase"] == phase and row["cohort"] == cohort]
            if selected:
                result[phase][cohort] = {
                    backend: sum(row["backend"] == backend for row in selected)
                    for backend in _BACKENDS
                }
    return result


def _manifest_sha256(manifest: Mapping[str, object]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return _sha256(canonical_json_bytes(payload))


def _report_sha256(report: Mapping[str, object]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    return _sha256(canonical_json_bytes(payload))


def _require_manifest_identity(
    evidence: Mapping[str, object], manifest: Mapping[str, object]
) -> None:
    if (
        evidence["experiment_id"] != manifest["experiment_id"]
        or evidence["experiment_manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise AcceptanceDataError("evidence does not belong to the frozen experiment")


def _row_sort_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return (
        str(row["phase"]),
        str(row["cohort"]),
        str(row["block_id"]),
        str(row["sample_id"]),
        str(row["backend"]),
    )


def _canonical_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ExperimentDesignError(f"{name} must be a JSON object")
    try:
        canonical = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError) as exc:
        raise ExperimentDesignError(f"{name} must contain finite JSON values") from exc
    if not isinstance(canonical, dict):
        raise ExperimentDesignError(f"{name} must be a JSON object")
    return canonical


def _require_mapping(
    value: object,
    name: str,
    error: type[ValueError] = ExperimentDesignError,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{name} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    name: str,
    error: type[ValueError] = ExperimentDesignError,
) -> None:
    if set(value) != expected:
        raise error(f"{name} keys must be exactly {sorted(expected)}")


def _require_id(
    value: object,
    name: str,
    error: type[ValueError] = ExperimentDesignError,
) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise error(f"{name} is invalid")


def _require_pin(
    value: object,
    name: str,
    error: type[ValueError] = ExperimentDesignError,
) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or value.lower() in {"main", "master", "latest", "unknown", "unavailable"}
        or any(ord(character) < 32 for character in value)
    ):
        raise error(f"{name} must be a bounded immutable identity")


def _require_sha256(
    value: object,
    name: str,
    error: type[ValueError] = ExperimentDesignError,
) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise error(f"{name} SHA-256 is invalid")


def _require_nonnegative_int(
    value: object,
    name: str,
    error: type[ValueError] = ExperimentDesignError,
) -> None:
    if type(value) is not int or value < 0:
        raise error(f"{name} must be a non-negative integer")


def _require_positive_int(
    value: object,
    name: str,
    error: type[ValueError] = ExperimentDesignError,
) -> None:
    if type(value) is not int or value <= 0:
        raise error(f"{name} must be a positive integer")


def _require_positive_number(
    value: object,
    name: str,
    error: type[ValueError] = ExperimentDesignError,
) -> None:
    if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
        raise error(f"{name} must be finite and positive")


def _require_nonnegative_number(
    value: object,
    name: str,
    error: type[ValueError] = ExperimentDesignError,
) -> None:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise error(f"{name} must be finite and non-negative")


def _require_unit_number(value: object, name: str) -> None:
    if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
        raise AcceptanceDataError(f"{name} must be finite and in [0, 1]")


def _id_list(value: object, name: str, minimum: int) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ExperimentDesignError(f"{name} must contain at least {minimum} ids")
    if len(value) != len(set(value)):
        raise ExperimentDesignError(f"{name} must not contain duplicates")
    for item in value:
        _require_id(item, name)
    return value


def _reject_sensitive_values(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in ("token", "password", "credential", "secret")):
                raise ExperimentDesignError(f"sensitive field is forbidden in {path}")
            _reject_sensitive_values(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_sensitive_values(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and (
        value.startswith("/home/") or value.startswith("/data/home/")
    ):
        raise ExperimentDesignError(f"personal absolute path is forbidden in {path}")


def _bounded_read(path: Path, limit: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AcceptanceDataError(f"unable to stat evidence file: {path.name}") from exc
    if size <= 0 or size > limit:
        raise AcceptanceDataError(f"evidence file size is invalid: {path.name}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AcceptanceDataError(f"unable to read evidence file: {path.name}") from exc
    if len(data) != size:
        raise AcceptanceDataError(f"evidence file changed while reading: {path.name}")
    return data


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
