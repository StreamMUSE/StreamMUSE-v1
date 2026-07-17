"""Frozen semantics for the July 2026 melody-robustness pilot.

This module deliberately contains no model code.  It is the shared contract
used by staging, execution, analysis, and listening-package tools, preventing
those phases from silently interpreting the plan differently.
"""

from __future__ import annotations

import hashlib
import json
import random
import copy
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CAMPAIGN_SCHEMA_VERSION = "streammuse.melody_robustness.campaign.v1"
RUN_SCHEMA_VERSION = "streammuse.melody_robustness.run.v1"
QUALIFICATION_SCHEMA_VERSION = "streammuse.melody_robustness.qualification.v1"
QUALIFICATION_DENSE_SONG = "2"
QUALIFICATION_TAIL_SONGS = ("2", "4")
QUALIFICATION_TEMPO_CANDIDATES = (60, 30)
QUALIFICATION_TAIL_CANDIDATES = (8, 16, 24)
QUALIFICATION_DECISION_ORDER = (
    "determinism",
    "static_input_gate",
    "tempo",
    "tail",
)
QUALIFICATION_TAIL_RULE = (
    "8=16=24->8; 16=24->16; otherwise stop_and_investigate"
)
ATTESTATION_BUNDLE_SCHEMA_VERSION = (
    "streammuse.melody_robustness.attestation_bundle.v1"
)
CODE_IDENTITY_SCHEMA_VERSION = (
    "streammuse.melody_robustness.code_identity.v1"
)
CHECKPOINT_IDENTITY_SCHEMA_VERSION = (
    "streammuse.melody_robustness.checkpoint_identity.v1"
)
ENVIRONMENT_IDENTITY_SCHEMA_VERSION = (
    "streammuse.melody_robustness.environment_identity.v1"
)
QUALIFICATION_SPEC_SCHEMA_VERSION = (
    "streammuse.melody_robustness.qualification_spec.v1"
)
LISTENING_SCHEMA_VERSION = "streammuse.melody_robustness.listening_selection.v1"
LISTENING_CLIP_COUNT = 24
LISTENING_CLIP_BEATS = 50
LISTENING_CLIP_SECONDS = 25
LISTENING_RENDER_BPM = 120
LISTENING_GAIN_POLICY = "fixed_pair_gain_with_true_peak_protection_only"
TRIANGLE_LISTENING_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_selection.v2"
)
TRIANGLE_LISTENING_TRIAL_COUNT = 95
TRIANGLE_LISTENING_PRACTICE_COUNT = 3
TRIANGLE_LISTENING_PRESENTATION_COUNT = 285
TRIANGLE_LISTENING_CLIP_BEATS = 16
TRIANGLE_LISTENING_CLIP_SECONDS = 8
TRIANGLE_LISTENING_RENDER_BPM = 120
TRIANGLE_LISTENING_SAMPLE_RATE = 44_100
TRIANGLE_LISTENING_SYNTH_GAIN = 0.5
TRIANGLE_LISTENING_GAIN_POLICY = (
    "common_pair_gain_with_true_peak_protection_only"
)

SEEDS: dict[str, Any] = {
    "perturb": [2026071001, 2026071002],
    "sample": [2026071101, 2026071102],
    "high_perturb": 2026071001,
    "run_order": 2026071201,
    "bootstrap": 2026071301,
    "blind_order": 2026071401,
}

CONDITION_TABLE: dict[str, dict[str, Any]] = {
    "sham": {"pitch_probability": 0.0, "onset_probability": 0.0, "pseed_count": 1},
    "pitch": {"pitch_probability": 0.05, "onset_probability": 0.0, "pseed_count": 2},
    "onset": {"pitch_probability": 0.0, "onset_probability": 0.15, "pseed_count": 2},
    "both": {"pitch_probability": 0.05, "onset_probability": 0.15, "pseed_count": 2},
    "high": {"pitch_probability": 0.20, "onset_probability": 0.40, "pseed_count": 1},
}


def qualification_config_contract() -> dict[str, Any]:
    """Return the exact qualification rules embedded in candidate and C5."""

    return {
        "dense_song": QUALIFICATION_DENSE_SONG,
        "tail_songs": list(QUALIFICATION_TAIL_SONGS),
        "sample_seed": int(SEEDS["sample"][0]),
        "perturb_seed": int(SEEDS["perturb"][0]),
        "tempo_candidates": list(QUALIFICATION_TEMPO_CANDIDATES),
        "tail_candidates": list(QUALIFICATION_TAIL_CANDIDATES),
        "decision_order": list(QUALIFICATION_DECISION_ORDER),
    }


def qualification_spec_contract() -> dict[str, Any]:
    """Return the independently hash-pinned fixed-20 qualification spec."""

    return {
        "schema_version": QUALIFICATION_SPEC_SCHEMA_VERSION,
        "qualification": qualification_config_contract(),
        "expected_run_count": 20,
        "attempt_policy": {
            "required_attempt_id": "attempt-001",
            "retry_allowed": False,
            "allow_dirty": False,
        },
        "execution_semantics": (
            "execute_all_20_rows_then_derive_decision_from_immutable_artifacts"
        ),
        "tail_rule": QUALIFICATION_TAIL_RULE,
    }


def triangle_listening_config_contract() -> dict[str, Any]:
    """Return the exact acc-only triangle-listening contract frozen into C5.

    The complete deterministic trial mapping lives in the separately hashed
    selection manifest.  This compact copy makes the experiment's question,
    source, counts, rendering, persistence, QC, and decision semantics visible
    to every formal consumer before any model output is produced.
    """

    return {
        "schema_version": TRIANGLE_LISTENING_SCHEMA_VERSION,
        "selection_must_precede_formal_outputs": True,
        "selection_manifest_path": None,
        "selection_manifest_sha256": None,
        "renderer_identity": None,
        "primary_question": "generated_acc_triangle_discrimination",
        "formal_pipeline": "rt",
        "source_artifact": "theoretical_model",
        "presentation": "acc_solo",
        "trial_count": TRIANGLE_LISTENING_TRIAL_COUNT,
        "practice_count": TRIANGLE_LISTENING_PRACTICE_COUNT,
        "presentation_count": TRIANGLE_LISTENING_PRESENTATION_COUNT,
        "block_counts": {
            "medium_primary": 60,
            "high_exploratory": 10,
            "sham_sampling_baseline": 5,
            "identity_catch": 6,
            "known_different_control": 6,
            "exact_repeat": 8,
        },
        "render_bpm": TRIANGLE_LISTENING_RENDER_BPM,
        "clip_seconds": TRIANGLE_LISTENING_CLIP_SECONDS,
        "clip_beats": TRIANGLE_LISTENING_CLIP_BEATS,
        "default_excerpt_model_beats": [16, 32],
        "sample_rate": TRIANGLE_LISTENING_SAMPLE_RATE,
        "bit_depth": 16,
        "synth_gain": TRIANGLE_LISTENING_SYNTH_GAIN,
        "gain_policy": TRIANGLE_LISTENING_GAIN_POLICY,
        "blind_order_seed": SEEDS["blind_order"],
        "prefix_chunk_size": 19,
        "prefix_chunk_count": 5,
        "presentation_patterns": ["AAB", "ABA", "BAA", "BBA", "BAB", "ABB"],
        "flexible_sittings": True,
        "per_trial_persistence": True,
        "partial_snapshot_allowed": True,
        "sitting_policy": {
            "append_only_hash_chain": True,
            "start_required_before_response": True,
            "start_fields": ["sitting_id", "device", "environment", "note"],
            "end_fields": ["sitting_id", "anomalies", "note"],
            "snapshot_seals_sitting_prefix": True,
            "sitting_boundaries_frozen": False,
        },
        "listening_attempt_id": "listening-attempt-001",
        "listening_attempt_policy": {
            "initial_attempt_id": "listening-attempt-001",
            "qc_failure": (
                "seal_and_preserve_failed_attempt_then_build_a_new_complete_"
                "domain_separated_blind_attempt"
            ),
            "retry_attempt_id_format": "listening-attempt-NNN",
            "retry_seed_derivation": (
                "sha256('streammuse-triangle-retry-v2' || base_blind_seed || "
                "attempt_number)"
            ),
            "failed_attempt_must_remain_reported": True,
            "answers_never_carry_between_attempts": True,
        },
        "response_schema": {
            "odd_choice": ["1", "2", "3", "no_difference"],
            "confidence": [1, 2, 3, 4, 5],
            "difference_tags": [
                "pitch_harmony",
                "rhythm_timing",
                "density",
                "texture_register",
                "silence_coverage",
                "other",
            ],
            "optional_note": True,
            "append_only_hash_chain": True,
            "atomic_progress": True,
        },
        "source_policy": {
            "attempt": "latest immutable content-valid formal attempt; do not substitute",
            "operational_invalid": "retain_and_report",
            "expected_empty": "retain_as_literal_silence_and_report",
            "nonempty_rendered_silent": "package_failure",
        },
        "qc_rules": {
            "identity_correct": 5,
            "identity_total": 6,
            "known_different_correct": 5,
            "known_different_total": 6,
            "repeat_consistent": 6,
            "repeat_total": 8,
        },
        "decision_rules": {
            "condition_total": 20,
            "condition_correct_threshold": 12,
            "song_total": 4,
            "song_correct_threshold": 2,
            "songs_meeting_threshold": 4,
            "requires_full_qc_pass": True,
            "requires_all_condition_rows_pre_first_semantic_unblind": True,
        },
        "blinding_policy": {
            "semantic_preview_before_first_unblind": False,
            "post_partial_unblind": "retain_but_mark_exploratory",
            "sitting_boundaries_frozen": False,
        },
    }


def legacy_listening_config_contract() -> dict[str, Any]:
    """Return the historical v1 contract for read/audit compatibility."""

    return {
        "selection_must_precede_formal_outputs": True,
        "selection_manifest_sha256": None,
        "render_bpm": LISTENING_RENDER_BPM,
        "clip_seconds": LISTENING_CLIP_SECONDS,
        "clip_count": LISTENING_CLIP_COUNT,
        "gain_policy": LISTENING_GAIN_POLICY,
        "blind_order_seed": SEEDS["blind_order"],
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_json(path: str | Path, value: Any) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(value))
    digest = file_sha256(destination)
    destination.with_name(destination.name + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="ascii"
    )
    return digest


def default_campaign_config(
    *,
    code_identity: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    input_manifest_path: str,
    input_manifest_sha256: str,
    attestation: Mapping[str, Any],
    playback_tempo: int = 60,
    tail_beats: int = 24,
) -> dict[str, Any]:
    """Return the immutable qualification-candidate semantic template.

    A caller must add the qualification design before validation.  Only the
    qualification finalizer may change this candidate into ``qualified_frozen``.
    """
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "status": "qualification_candidate",
        "code_identity": str(code_identity),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": str(checkpoint_sha256),
        },
        "input_manifest": {
            "path": str(input_manifest_path),
            "sha256": str(input_manifest_sha256),
            "expected_inputs": 40,
        },
        "attestation": copy.deepcopy(dict(attestation)),
        "conditions": copy.deepcopy(CONDITION_TABLE),
        "seeds": copy.deepcopy(SEEDS),
        "run_counts": {
            "songs": 5,
            "variants_per_song": 8,
            "inputs": 40,
            "sample_seeds": 2,
            "runs_per_pipeline": 80,
            "model_runs": 160,
            "known_bad_counts_as_model_run": False,
        },
        "sampling": {
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "selection_semantics": "keyed_bernoulli_threshold",
            "selection_denominator": "matched_non_drum_pitch_21_108_model_visible_notes",
            "dose_fields": [
                "target_rate",
                "selected_rate",
                "proposed_rate",
                "effective_rate",
                "giveup_rate",
                "model_visible_rate",
            ],
        },
        "perturbation": {
            "parser_writer": "pretty_midi_canonical_note_table",
            "non_model_visible_policy": {
                "dangling_note_on": "drop_in_all_arms",
                "spurious_or_zero_duration_off_pair": "drop_in_all_arms",
                "matched_drum_span": "preserve",
                "matched_out_of_range_span": "preserve",
                "cc_pedal_pitch_bend": "preserve",
                "tempo_key_time_signature_track_name_channel": "preserve",
                "rationale": (
                    "dropped unmatched events are absent from MidiConverter model roll and must "
                    "not become visible by accidentally pairing after onset edits"
                ),
            },
            "interval_semantics": "half_open_[start,end)",
            "traversal_order": ["start_tick", "track_index", "track_note_ordinal"],
            "selection": "keyed_bernoulli_threshold",
            "stable_prf": "blake2b_128_canonical_domain_separated_person=SMusePerturbV1",
            "pitch_candidate_set": [-2, -1, 1, 2],
            "onset_candidate_set": [-1, 1],
            "candidate_order": "per_note_per_decision_keyed_PRF_order",
            "collision_attempts": 3,
            "collision_scope": "reject_only_new_same_pitch_overlap",
            "native_overlap": (
                "allowed_if_edit_remains_MIDI_serializable; reject candidate when unequal "
                "edited ends would make note-off ownership ambiguous"
            ),
            "giveup": "retain_original_note_and_record_reason",
            "cross_arm_invariant": "latent_proposals_only",
            "effective_mismatch": (
                "allowed; record note_ids, counts, and collision_interaction per song/pseed"
            ),
            "boundary_onset_noop": "start_zero_minus_one_clamps_to_zero_and_is_logged",
            "dose_nesting": "shared_selection_score_medium_subset_of_high",
        },
        "runtime": {
            "model_name": "lekai",
            "inference_mode": "sliding_window",
            "device": "cuda",
            "dtype": "float16",
            "time_signature_index": 4,
            "max_generation_length_frames": None,
            "max_prompt_ticks": None,
            "use_cache": True,
            "generation_interval_ticks": 4,
            "generation_length_frames": 4,
            "model_condition_bpm": 120,
            "playback_tempo": int(playback_tempo),
            "prompt_context_beats": 128,
            "history_retention_ticks": 512,
            "tail_beats": int(tail_beats),
            "count_in_beats": 0,
            "ticks_per_beat": 4,
            "beats_per_bar": 4,
            "analysis_end_is_exclusive": True,
            "request_cutoff_formula": (
                "floor_to_generation_interval_strictly_before(analysis_end_tick)"
            ),
            "run_stop_formula": "max(last_input_note_off_tick,request_cutoff_tick)+tail_beats*4",
            "deterministic_boundary_generation": True,
            "artifact_tier": "debug",
            "inference_log_detail": "full",
        },
        "validity": {
            "content": {
                "require_expected_equals_processed": True,
                "allow_http_failures": False,
                "allow_pending_at_stop": False,
                "require_input_digest_match": True,
            },
            "operational": {
                "allow_stale_drop": False,
                "allow_late": False,
                "allow_clamp": False,
                "allow_forced_note_off": False,
                "operational_failure_excludes_theoretical": False,
            },
            "retry": {
                "content_failure_max_retries": 1,
                "preserve_first_attempt": True,
                "retry_scope": "matched_song_pipeline_sample_seed_block",
            },
        },
        "estimands": {
            "hypotheses": {
                "H_input": "input perturbation changes fixed-clean-reference quality",
                "H_generation_interaction": "offline_vs_rt_theoretical",
                "H_operational": "rt_theoretical_vs_rt_combined",
                "H_realworld": "not_answered",
            },
            "primary_contrast": "both_vs_sham",
            "primary_quality": "D_intended_with_coverage_guardrail",
            "d_actual_interpretation": "joint_treatment_effect_not_adaptation",
            "na_policy": {
                "fully_empty_accompaniment": "NA_coverage_failure",
                "partially_empty": "conditional_D_plus_explicit_coverage",
                "one_sided_pair_NA": "paired_contrast_NA_and_pattern_reported",
                "song_aggregation": "no_silent_complete_case_deletion",
                "bootstrap": "retain_NA_pattern_and_report_valid_block_count",
            },
            "bootstrap": {
                "unit": "complete_song_block",
                "iterations": 10000,
                "interval": [0.025, 0.975],
                "seed": SEEDS["bootstrap"],
                "interpretation": "descriptive",
            },
        },
        "listening": triangle_listening_config_contract(),
        "freeze_order": [
            "commit_clean_code",
            "freeze_code_and_checkpoint",
            "determinism_qualification",
            "tempo_and_tail_qualification",
            "freeze_campaign_config_C5",
            "formal_model_runs",
            "objective_analysis",
        ],
    }


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _require_lower_hex(value: Any, lengths: set[int], label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or set(value) - set("0123456789abcdef")
    ):
        rendered = "/".join(str(length) for length in sorted(lengths))
        raise ValueError(f"{label} must be lowercase hexadecimal ({rendered} characters)")
    return value


def _validate_file_record(
    value: Any,
    label: str,
    *,
    verify_file: bool,
) -> tuple[Path, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a path/hash object")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path is required")
    digest = _require_lower_hex(value.get("sha256"), {64}, f"{label}.sha256")
    path = Path(raw_path).resolve()
    if verify_file:
        if not path.is_file():
            raise FileNotFoundError(f"{label} missing: {path}")
        actual = file_sha256(path)
        if actual != digest:
            raise ValueError(
                f"{label} hash mismatch: expected {digest}, got {actual}"
            )
    return path, digest


def _validate_qualification_rules(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("qualification")
    if not isinstance(value, Mapping):
        raise ValueError("qualification rules are required")
    _assert_equal(
        set(value),
        {
            "dense_song",
            "tail_songs",
            "sample_seed",
            "perturb_seed",
            "tempo_candidates",
            "tail_candidates",
            "decision_order",
        },
        "qualification fields",
    )
    _assert_equal(
        value.get("dense_song"),
        QUALIFICATION_DENSE_SONG,
        "qualification.dense_song",
    )
    _assert_equal(
        value.get("tail_songs"),
        list(QUALIFICATION_TAIL_SONGS),
        "qualification.tail_songs",
    )
    _assert_equal(value.get("sample_seed"), SEEDS["sample"][0], "qualification.sample_seed")
    _assert_equal(value.get("perturb_seed"), SEEDS["perturb"][0], "qualification.perturb_seed")
    _assert_equal(
        value.get("tempo_candidates"),
        list(QUALIFICATION_TEMPO_CANDIDATES),
        "qualification.tempo_candidates",
    )
    _assert_equal(
        value.get("tail_candidates"),
        list(QUALIFICATION_TAIL_CANDIDATES),
        "qualification.tail_candidates",
    )
    _assert_equal(
        value.get("decision_order"),
        list(QUALIFICATION_DECISION_ORDER),
        "qualification.decision_order",
    )
    return value


def validate_campaign_attestation(
    attestation: Any,
    *,
    code_identity: str,
    checkpoint: Mapping[str, Any],
    qualification: Mapping[str, Any],
    verify_files: bool,
) -> dict[str, dict[str, Any]]:
    """Validate the four-file pre-qualification attestation and its semantics."""

    if not isinstance(attestation, Mapping):
        raise ValueError("campaign attestation bundle is required")
    expected_keys = {
        "schema_version",
        "code_identity",
        "checkpoint_identity",
        "environment",
        "qualification_spec",
    }
    _assert_equal(set(attestation), expected_keys, "attestation fields")
    _assert_equal(
        attestation.get("schema_version"),
        ATTESTATION_BUNDLE_SCHEMA_VERSION,
        "attestation schema_version",
    )
    paths: dict[str, Path] = {}
    for name in (
        "code_identity",
        "checkpoint_identity",
        "environment",
        "qualification_spec",
    ):
        path, _digest = _validate_file_record(
            attestation.get(name),
            f"attestation.{name}",
            verify_file=verify_files,
        )
        paths[name] = path
    if not verify_files:
        return {}

    payloads = {
        name: _read_json_object(path, f"attestation {name}")
        for name, path in paths.items()
    }
    code = payloads["code_identity"]
    _assert_equal(
        set(code),
        {
            "schema_version",
            "repository_root",
            "git_commit",
            "git_clean",
            "git_status_porcelain",
        },
        "code identity fields",
    )
    _assert_equal(
        code.get("schema_version"),
        CODE_IDENTITY_SCHEMA_VERSION,
        "code identity schema_version",
    )
    repository_root = code.get("repository_root")
    if (
        not isinstance(repository_root, str)
        or not repository_root
        or not Path(repository_root).is_absolute()
    ):
        raise ValueError("code identity repository_root must be absolute")
    _assert_equal(code.get("git_commit"), code_identity, "attested git commit")
    _assert_equal(code.get("git_clean"), True, "attested git clean")
    _assert_equal(
        code.get("git_status_porcelain"), "", "attested git status porcelain"
    )

    checkpoint_payload = payloads["checkpoint_identity"]
    _assert_equal(
        set(checkpoint_payload),
        {"schema_version", "path", "sha256", "size_bytes"},
        "checkpoint identity fields",
    )
    _assert_equal(
        checkpoint_payload.get("schema_version"),
        CHECKPOINT_IDENTITY_SCHEMA_VERSION,
        "checkpoint identity schema_version",
    )
    expected_checkpoint_path = str(Path(str(checkpoint["path"])).resolve())
    _assert_equal(
        str(Path(str(checkpoint_payload.get("path"))).resolve()),
        expected_checkpoint_path,
        "attested checkpoint path",
    )
    _assert_equal(
        checkpoint_payload.get("sha256"),
        checkpoint.get("sha256"),
        "attested checkpoint sha256",
    )
    checkpoint_path = Path(expected_checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"attested checkpoint is missing: {checkpoint_path}")
    _assert_equal(
        file_sha256(checkpoint_path),
        checkpoint.get("sha256"),
        "live attested checkpoint sha256",
    )
    if (
        isinstance(checkpoint_payload.get("size_bytes"), bool)
        or not isinstance(checkpoint_payload.get("size_bytes"), int)
        or int(checkpoint_payload["size_bytes"]) <= 0
    ):
        raise ValueError("attested checkpoint size_bytes must be positive")
    _assert_equal(
        checkpoint_payload["size_bytes"],
        checkpoint_path.stat().st_size,
        "attested checkpoint size",
    )

    spec = payloads["qualification_spec"]
    _assert_equal(spec, qualification_spec_contract(), "qualification attestation spec")
    _assert_equal(
        spec.get("qualification"),
        dict(qualification),
        "attested qualification/config rules",
    )

    environment = payloads["environment"]
    _assert_equal(
        set(environment),
        {
            "schema_version",
            "code_identity",
            "dependency_files",
            "python",
            "torch",
            "cuda_visible_devices",
            "gpus",
            "nvidia_smi",
        },
        "environment identity fields",
    )
    _assert_equal(
        environment.get("schema_version"),
        ENVIRONMENT_IDENTITY_SCHEMA_VERSION,
        "environment identity schema_version",
    )
    _assert_equal(
        environment.get("code_identity"), code_identity, "environment code identity"
    )
    dependencies = environment.get("dependency_files")
    if not isinstance(dependencies, Mapping) or set(dependencies) != {
        "uv_lock",
        "pyproject",
    }:
        raise ValueError("environment dependency_files must pin uv_lock and pyproject")
    for name, record in dependencies.items():
        _validate_file_record(
            record,
            f"environment dependency {name}",
            verify_file=True,
        )
    python_identity = environment.get("python")
    if not isinstance(python_identity, Mapping) or set(python_identity) != {
        "implementation",
        "version",
        "version_info",
        "executable",
    }:
        raise ValueError("environment Python identity is malformed")
    if (
        not isinstance(python_identity.get("implementation"), str)
        or not python_identity.get("implementation")
        or not isinstance(python_identity.get("version"), str)
        or not python_identity.get("version")
        or not isinstance(python_identity.get("version_info"), list)
        or len(python_identity["version_info"]) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in python_identity["version_info"]
        )
        or not isinstance(python_identity.get("executable"), str)
        or not python_identity.get("executable")
    ):
        raise ValueError("environment Python identity values are malformed")
    torch_identity = environment.get("torch")
    if not isinstance(torch_identity, Mapping) or set(torch_identity) != {
        "version",
        "cuda_version",
        "cudnn_version",
        "cuda_available",
    }:
        raise ValueError("environment Torch/CUDA identity is malformed")
    if (
        not isinstance(torch_identity.get("version"), str)
        or not torch_identity.get("version")
        or torch_identity.get("cuda_available") is not True
        or not isinstance(torch_identity.get("cuda_version"), str)
        or not torch_identity.get("cuda_version")
        or (
            torch_identity.get("cudnn_version") is not None
            and (
                isinstance(torch_identity.get("cudnn_version"), bool)
                or not isinstance(torch_identity.get("cudnn_version"), int)
            )
        )
    ):
        raise ValueError("environment Torch/CUDA identity values are malformed")
    visible = environment.get("cuda_visible_devices")
    if visible is not None and not isinstance(visible, str):
        raise ValueError("environment CUDA_VISIBLE_DEVICES must be string or null")
    gpus = environment.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise ValueError("environment requires at least one Torch-visible GPU")
    for gpu in gpus:
        if not isinstance(gpu, Mapping) or set(gpu) != {
            "visible_index",
            "name",
            "uuid",
            "total_memory_bytes",
            "compute_capability",
        }:
            raise ValueError("environment Torch GPU identity is malformed")
        if (
            isinstance(gpu.get("visible_index"), bool)
            or not isinstance(gpu.get("visible_index"), int)
            or not isinstance(gpu.get("name"), str)
            or not gpu.get("name")
            or gpu.get("uuid") is not None
            and not isinstance(gpu.get("uuid"), str)
            or isinstance(gpu.get("total_memory_bytes"), bool)
            or not isinstance(gpu.get("total_memory_bytes"), int)
            or int(gpu["total_memory_bytes"]) <= 0
            or not isinstance(gpu.get("compute_capability"), list)
            or len(gpu["compute_capability"]) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in gpu["compute_capability"]
            )
        ):
            raise ValueError("environment Torch GPU identity values are malformed")
    nvidia = environment.get("nvidia_smi")
    if not isinstance(nvidia, Mapping) or set(nvidia) != {
        "driver_version",
        "gpus",
    }:
        raise ValueError("environment nvidia-smi identity is malformed")
    if not isinstance(nvidia.get("driver_version"), str) or not nvidia.get(
        "driver_version"
    ):
        raise ValueError("environment NVIDIA driver version is missing")
    physical_gpus = nvidia.get("gpus")
    if not isinstance(physical_gpus, list) or not physical_gpus:
        raise ValueError("environment requires nvidia-smi GPU identities")
    for gpu in physical_gpus:
        if not isinstance(gpu, Mapping) or set(gpu) != {
            "physical_index",
            "uuid",
            "name",
            "memory_total_mib",
        }:
            raise ValueError("environment nvidia-smi GPU identity is malformed")
        if (
            isinstance(gpu.get("physical_index"), bool)
            or not isinstance(gpu.get("physical_index"), int)
            or not isinstance(gpu.get("uuid"), str)
            or not gpu.get("uuid")
            or not isinstance(gpu.get("name"), str)
            or not gpu.get("name")
            or isinstance(gpu.get("memory_total_mib"), bool)
            or not isinstance(gpu.get("memory_total_mib"), int)
            or int(gpu["memory_total_mib"]) <= 0
        ):
            raise ValueError("environment nvidia-smi GPU identity values are malformed")
    return payloads


def validate_campaign_config(
    config: Mapping[str, Any],
    *,
    require_frozen: bool = True,
    verify_attestations: bool = False,
) -> None:
    _assert_equal(config.get("schema_version"), CAMPAIGN_SCHEMA_VERSION, "schema_version")
    if require_frozen:
        _assert_equal(config.get("status"), "qualified_frozen", "status")
    elif config.get("status") not in {"qualification_candidate", "qualified_frozen"}:
        raise ValueError("status must be qualification_candidate or qualified_frozen")
    runtime = config.get("runtime", {})
    if runtime.get("playback_tempo") not in {30, 60}:
        raise ValueError("runtime.playback_tempo must be the qualified campaign-wide 30 or 60")
    if runtime.get("tail_beats") not in {8, 16, 24}:
        raise ValueError("runtime.tail_beats must be the qualified 8, 16, or 24")
    checkpoint = config.get("checkpoint", {})
    input_manifest = config.get("input_manifest", {})
    _validate_file_record(checkpoint, "checkpoint", verify_file=False)
    _validate_file_record(input_manifest, "input_manifest", verify_file=False)
    code_identity = _require_lower_hex(
        config.get("code_identity"), {40, 64}, "code_identity"
    )
    qualification = _validate_qualification_rules(config)
    validate_campaign_attestation(
        config.get("attestation"),
        code_identity=code_identity,
        checkpoint=checkpoint,
        qualification=qualification,
        verify_files=verify_attestations,
    )

    # Rebuild the authoritative template with only genuinely dynamic values
    # substituted, then exact-compare every frozen semantic subtree. This
    # prevents a hand-edited config with a freshly computed hash from changing
    # sampling, perturbation, runtime, validity, estimand, or freeze semantics.
    reference = default_campaign_config(
        code_identity=str(config["code_identity"]),
        checkpoint_path=str(checkpoint["path"]),
        checkpoint_sha256=str(checkpoint["sha256"]),
        input_manifest_path=str(input_manifest["path"]),
        input_manifest_sha256=str(input_manifest["sha256"]),
        attestation=config["attestation"],
        playback_tempo=int(runtime["playback_tempo"]),
        tail_beats=int(runtime["tail_beats"]),
    )
    for key in (
        "conditions",
        "seeds",
        "run_counts",
        "sampling",
        "perturbation",
        "runtime",
        "validity",
        "estimands",
        "attestation",
        "freeze_order",
    ):
        _assert_equal(config.get(key), reference[key], key)
    _assert_equal(
        input_manifest.get("expected_inputs"),
        reference["input_manifest"]["expected_inputs"],
        "input_manifest.expected_inputs",
    )
    actual_listening = dict(config.get("listening", {}))
    listening_schema = actual_listening.get("schema_version")
    for dynamic_key in (
        "selection_manifest_path",
        "selection_manifest_sha256",
        "renderer_identity",
    ):
        actual_listening.pop(dynamic_key, None)
    if listening_schema == TRIANGLE_LISTENING_SCHEMA_VERSION:
        expected_listening = triangle_listening_config_contract()
    elif listening_schema is None:
        # Historical C5 files did not carry an explicit listening schema.
        expected_listening = legacy_listening_config_contract()
    else:
        raise ValueError(f"unsupported listening schema: {listening_schema!r}")
    for dynamic_key in (
        "selection_manifest_path",
        "selection_manifest_sha256",
        "renderer_identity",
    ):
        expected_listening.pop(dynamic_key, None)
    _assert_equal(actual_listening, expected_listening, "listening")
    if require_frozen:
        selection_path = config.get("listening", {}).get("selection_manifest_path")
        if not isinstance(selection_path, str) or not selection_path:
            raise ValueError("frozen config requires listening selection path")
        selection_sha = config.get("listening", {}).get("selection_manifest_sha256")
        _require_lower_hex(selection_sha, {64}, "listening.selection_manifest_sha256")
        if listening_schema == TRIANGLE_LISTENING_SCHEMA_VERSION:
            _validate_file_record(
                config.get("listening", {}).get("renderer_identity"),
                "listening.renderer_identity",
                verify_file=False,
            )
        _validate_file_record(
            config.get("qualification_candidate"),
            "qualification_candidate",
            verify_file=False,
        )
        _validate_file_record(
            config.get("qualification_result"),
            "qualification_result",
            verify_file=False,
        )
    else:
        if config.get("status") == "qualification_candidate":
            if config.get("listening", {}).get("selection_manifest_sha256") is not None:
                raise ValueError("qualification candidate cannot freeze listening selection")
            if config.get("listening", {}).get("selection_manifest_path") is not None:
                raise ValueError("qualification candidate cannot freeze listening selection path")
            if config.get("listening", {}).get("renderer_identity") is not None:
                raise ValueError("qualification candidate cannot freeze renderer identity")
            if "qualification_candidate" in config or "qualification_result" in config:
                raise ValueError("qualification candidate cannot contain final evidence records")


def _manifest_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("entries", manifest.get("inputs"))
    if not isinstance(raw, list):
        raise ValueError("input manifest must contain an entries or inputs list")
    return [dict(entry) for entry in raw]


def _entry_value(entry: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in entry:
            return entry[key]
    return None


def validate_input_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = _manifest_entries(manifest)
    if len(entries) != 40:
        raise ValueError(f"input manifest must contain exactly 40 entries, got {len(entries)}")
    stems = [_entry_value(entry, "stem", "input_id", "output_stem") for entry in entries]
    if any(not stem for stem in stems) or len(set(stems)) != 40:
        raise ValueError("input manifest stems must be present and unique")
    song_counts: dict[str, int] = {}
    condition_counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        raw_song = _entry_value(entry, "song", "source_stem")
        if raw_song is None or not str(raw_song).strip():
            raise ValueError("every input manifest entry requires a non-empty song/source_stem")
        song = str(raw_song)
        condition = str(entry.get("condition"))
        if condition not in CONDITION_TABLE:
            raise ValueError(f"unknown condition in input manifest: {condition}")
        song_counts[song] = song_counts.get(song, 0) + 1
        condition_counts[(song, condition)] = condition_counts.get((song, condition), 0) + 1
        expected_spec = CONDITION_TABLE[condition]
        for field in ("pitch_probability", "onset_probability"):
            if float(entry.get(field, -1.0)) != float(expected_spec[field]):
                raise ValueError(
                    f"{song}/{condition}: {field} mismatch: {entry.get(field)!r}"
                )
        seed = entry.get("perturb_seed")
        if condition == "sham" and seed is not None:
            raise ValueError(f"{song}/sham must use perturb_seed=null")
        if condition == "high" and seed != SEEDS["high_perturb"]:
            raise ValueError(f"{song}/high must use frozen high perturb seed")
        if condition in {"pitch", "onset", "both"} and seed not in SEEDS["perturb"]:
            raise ValueError(f"{song}/{condition} uses non-frozen perturb seed {seed!r}")
    if len(song_counts) != 5 or set(song_counts.values()) != {8}:
        raise ValueError(f"expected five songs with eight variants each, got {song_counts}")
    for song in song_counts:
        for condition, spec in CONDITION_TABLE.items():
            expected = int(spec["pseed_count"])
            actual = condition_counts.get((song, condition), 0)
            if actual != expected:
                raise ValueError(f"{song}/{condition}: expected {expected}, got {actual}")
        for condition in ("pitch", "onset", "both"):
            actual_seeds = {
                entry.get("perturb_seed")
                for entry in entries
                if str(_entry_value(entry, "song", "source_stem")) == song
                and entry.get("condition") == condition
            }
            if actual_seeds != set(SEEDS["perturb"]):
                raise ValueError(
                    f"{song}/{condition}: expected perturb seeds {SEEDS['perturb']}, "
                    f"got {sorted(actual_seeds)}"
                )
    return entries


def validate_staged_input_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    verify_files: bool = True,
) -> list[dict[str, Any]]:
    """Validate the complete 40-input staging/artifact contract.

    ``validate_input_manifest`` intentionally validates the factorial identity
    only. Formal execution/analysis must use this stronger validator so clean
    references, sidecars, NPZs, accompaniment copies, and horizons cannot drift
    after the manifest is frozen.
    """
    entries = validate_input_manifest(manifest)
    _assert_equal(
        manifest.get("schema_version"),
        "streammuse.melody_perturbation.v1",
        "input manifest schema_version",
    )
    _assert_equal(manifest.get("input_count"), 40, "input manifest input_count")
    _assert_equal(
        manifest.get("expected_input_count"), 40, "input manifest expected_input_count"
    )
    _assert_equal(manifest.get("model_ticks_per_beat"), 4, "model_ticks_per_beat")
    _assert_equal(manifest.get("perturb_seeds"), SEEDS["perturb"], "perturb_seeds")
    _assert_equal(manifest.get("high_pseed"), SEEDS["high_perturb"], "high_pseed")
    expected_stems = sorted(
        str(_entry_value(entry, "stem", "input_id", "output_stem"))
        for entry in entries
    )
    _assert_equal(manifest.get("exact_stems"), expected_stems, "exact_stems")

    base = Path(manifest_path).resolve().parent
    sha_chars = set("0123456789abcdef")
    required_artifacts = (
        "output_midi",
        "source_midi",
        "npz",
        "acc_copy",
        "source_acc",
        "sidecar",
    )
    source_hash_by_song: dict[str, str] = {}
    acc_hash_by_song: dict[str, str] = {}
    for entry in entries:
        stem = str(_entry_value(entry, "stem", "input_id", "output_stem"))
        song = str(_entry_value(entry, "song", "source_stem"))
        analysis_end = entry.get("analysis_end_tick")
        last_note_off = entry.get("last_input_note_off_tick")
        validation_horizon = entry.get("validation_horizon_ticks")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (
            analysis_end, last_note_off, validation_horizon
        )):
            raise ValueError(f"{stem}: horizons must be integer ticks")
        if not (0 < int(analysis_end) <= int(validation_horizon)):
            raise ValueError(f"{stem}: invalid analysis/validation horizon")
        if not (0 <= int(last_note_off) <= int(validation_horizon)):
            raise ValueError(f"{stem}: invalid last_input_note_off_tick")
        if int(validation_horizon) % 16 != 0:
            raise ValueError(f"{stem}: validation horizon must be measure aligned")

        records: dict[str, Mapping[str, Any]] = {}
        for name in required_artifacts:
            value = entry.get(name)
            if not isinstance(value, Mapping):
                paths = entry.get("paths")
                value = paths.get(name) if isinstance(paths, Mapping) else None
            if not isinstance(value, Mapping):
                raise ValueError(f"{stem}: missing artifact record {name}")
            raw_path = value.get("path")
            digest = value.get("sha256")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"{stem}: {name}.path is required")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or set(digest) - sha_chars
            ):
                raise ValueError(f"{stem}: {name}.sha256 must be lowercase SHA-256")
            path = Path(raw_path)
            path = (path if path.is_absolute() else base / path).resolve()
            if verify_files:
                if not path.is_file():
                    raise FileNotFoundError(f"{stem}: missing {name}: {path}")
                actual = file_sha256(path)
                if actual != digest:
                    raise ValueError(
                        f"{stem}: {name} hash mismatch: expected {digest}, got {actual}"
                    )
            records[name] = value
        if records["source_acc"]["sha256"] != records["acc_copy"]["sha256"]:
            raise ValueError(f"{stem}: accompaniment copy differs from clean source")
        source_sha = str(records["source_midi"]["sha256"])
        acc_sha = str(records["source_acc"]["sha256"])
        if song in source_hash_by_song and source_hash_by_song[song] != source_sha:
            raise ValueError(f"{song}: clean melody hash differs between variants")
        if song in acc_hash_by_song and acc_hash_by_song[song] != acc_sha:
            raise ValueError(f"{song}: clean accompaniment hash differs between variants")
        source_hash_by_song[song] = source_sha
        acc_hash_by_song[song] = acc_sha
        if entry.get("condition") in {"pitch", "onset", "both"}:
            pairing = entry.get("factorial_pairing")
            if not isinstance(pairing, Mapping) or pairing.get("latent_pairing_verified") is not True:
                raise ValueError(f"{stem}: latent factorial pairing is not verified")
    return entries


def build_listening_selection_manifest(
    input_manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    perturb_seed: int,
    sample_seed: int,
    blind_order_seed: int,
    excerpt_starts: Mapping[str, int],
    verify_files: bool = True,
) -> dict[str, Any]:
    """Build the one authoritative pre-formal 24-clip selection payload."""
    resolved_manifest = Path(manifest_path).resolve()
    entries = validate_staged_input_manifest(
        input_manifest,
        manifest_path=resolved_manifest,
        verify_files=verify_files,
    )
    songs = sorted(
        {str(_entry_value(entry, "song", "source_stem")) for entry in entries}
    )
    if int(perturb_seed) not in {int(seed) for seed in SEEDS["perturb"]}:
        raise ValueError(f"perturb seed must be one of the frozen seeds: {SEEDS['perturb']}")
    if int(sample_seed) not in {int(seed) for seed in SEEDS["sample"]}:
        raise ValueError(f"sample seed must be one of the frozen seeds: {SEEDS['sample']}")
    if int(blind_order_seed) != int(SEEDS["blind_order"]):
        raise ValueError("blind-order seed must match the frozen contract")
    if set(map(str, excerpt_starts)) != set(songs):
        raise ValueError("excerpt starts must name each frozen song exactly once")

    horizon_sets: dict[str, set[int]] = {song: set() for song in songs}
    for entry in entries:
        song = str(_entry_value(entry, "song", "source_stem"))
        horizon_sets[song].add(int(entry["analysis_end_tick"]))
    if any(len(values) != 1 for values in horizon_sets.values()):
        raise ValueError("analysis horizon must be identical across variants of each song")
    horizons = {song: next(iter(horizon_sets[song])) for song in songs}
    starts: dict[str, int] = {}
    for song in songs:
        value = excerpt_starts[song]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{song}: excerpt start must be an integer")
        if value < 0:
            raise ValueError(f"{song}: excerpt start must be non-negative")
        if (value + LISTENING_CLIP_BEATS) * 4 > horizons[song]:
            raise ValueError(f"{song}: excerpt exceeds analysis_end_tick")
        starts[song] = int(value)

    semantic: list[dict[str, Any]] = []
    for presentation, pipeline in (
        ("ecological", "rt_combined"),
        ("acc_solo", "rt_theoretical"),
    ):
        for song in songs:
            for condition in ("sham", "both"):
                semantic.append(
                    {
                        "block": presentation,
                        "song": song,
                        "condition": condition,
                        "pipeline": pipeline,
                        "perturb_seed": (
                            None if condition == "sham" else int(perturb_seed)
                        ),
                        "sample_seed": int(sample_seed),
                        "excerpt_start_model_beat": starts[song],
                        "excerpt_end_model_beat": starts[song]
                        + LISTENING_CLIP_BEATS,
                        "analysis_end_tick": horizons[song],
                    }
                )
    first_song = songs[0]
    semantic.extend(
        [
            {
                "block": "anchor",
                "anchor_kind": "known_bad_harmonic_m2",
                "song": first_song,
                "condition": "sham",
                "pipeline": "control",
                "perturb_seed": None,
                "sample_seed": int(sample_seed),
                "excerpt_start_model_beat": starts[first_song],
                "excerpt_end_model_beat": starts[first_song]
                + LISTENING_CLIP_BEATS,
                "analysis_end_tick": horizons[first_song],
            },
            {
                "block": "anchor",
                "anchor_kind": "high_dose",
                "song": first_song,
                "condition": "high",
                "pipeline": "rt_combined",
                "perturb_seed": int(SEEDS["high_perturb"]),
                "sample_seed": int(sample_seed),
                "excerpt_start_model_beat": starts[first_song],
                "excerpt_end_model_beat": starts[first_song]
                + LISTENING_CLIP_BEATS,
                "analysis_end_tick": horizons[first_song],
            },
        ]
    )
    rng = random.Random(int(blind_order_seed))
    repeat_sources = rng.sample(range(20), 2)
    for source_index in repeat_sources:
        repeated = dict(semantic[source_index])
        repeated["block"] = "repeat"
        repeated["duplicate_semantic_index"] = source_index
        semantic.append(repeated)
    order = list(range(LISTENING_CLIP_COUNT))
    rng.shuffle(order)
    clips = [
        {
            "sample_id": f"S{presentation_index:03d}",
            "semantic_index": semantic_index,
            **semantic[semantic_index],
        }
        for presentation_index, semantic_index in enumerate(order, start=1)
    ]
    return {
        "schema_version": LISTENING_SCHEMA_VERSION,
        "frozen_before_formal": True,
        "input_manifest_path": str(resolved_manifest),
        "input_manifest_sha256": file_sha256(resolved_manifest),
        "analysis_horizons_ticks": horizons,
        "clip_count": LISTENING_CLIP_COUNT,
        "clip_seconds": LISTENING_CLIP_SECONDS,
        "render_bpm": LISTENING_RENDER_BPM,
        "question_semantics": {
            "ecological": "end_to_end_joint_quality_of_actual_melody_plus_accompaniment",
            "acc_solo": "standalone_accompaniment_coherence_not_harmonic_adaptation",
        },
        "gain_policy": LISTENING_GAIN_POLICY,
        "sample_seed": int(sample_seed),
        "perturb_seed": int(perturb_seed),
        "blind_order_seed": int(blind_order_seed),
        "repeat_source_semantic_indices": repeat_sources,
        "clips": clips,
    }


def validate_listening_selection_manifest(
    selection: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Exact-validate selection content, not merely its clip count and hash."""
    clips = selection.get("clips")
    if not isinstance(clips, list) or len(clips) != LISTENING_CLIP_COUNT:
        raise ValueError("listening selection must contain exactly 24 clip objects")
    starts_by_song: dict[str, set[int]] = {}
    for clip in clips:
        if not isinstance(clip, Mapping):
            raise ValueError("listening selection clip is not an object")
        song = clip.get("song")
        start = clip.get("excerpt_start_model_beat")
        if not isinstance(song, str) or not song:
            raise ValueError("listening selection clip lacks song")
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError(f"{song}: listening excerpt start must be an integer")
        starts_by_song.setdefault(song, set()).add(start)
    if any(len(values) != 1 for values in starts_by_song.values()):
        raise ValueError("listening selection must use one frozen excerpt per song")
    try:
        expected = build_listening_selection_manifest(
            input_manifest,
            manifest_path=manifest_path,
            perturb_seed=int(selection["perturb_seed"]),
            sample_seed=int(selection["sample_seed"]),
            blind_order_seed=int(selection["blind_order_seed"]),
            excerpt_starts={song: next(iter(values)) for song, values in starts_by_song.items()},
            verify_files=verify_files,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid listening selection contract: {exc}") from exc
    _assert_equal(dict(selection), expected, "listening selection manifest")
    return expected


def build_run_schedule(
    input_manifest: Mapping[str, Any],
    campaign_config: Mapping[str, Any],
    *,
    require_frozen: bool = True,
) -> list[dict[str, Any]]:
    """Build the immutable 160-row formal schedule with deterministic ordering."""
    validate_campaign_config(campaign_config, require_frozen=require_frozen)
    entries = validate_input_manifest(input_manifest)
    rows_by_pipeline: dict[str, list[dict[str, Any]]] = {"offline": [], "rt": []}
    sample_seeds = [int(seed) for seed in campaign_config["seeds"]["sample"]]
    for pipeline in ("offline", "rt"):
        for entry in entries:
            stem = str(_entry_value(entry, "stem", "input_id", "output_stem"))
            song = str(_entry_value(entry, "song", "source_stem"))
            condition = str(entry["condition"])
            perturb_seed = entry.get("perturb_seed")
            for sample_seed in sample_seeds:
                identity = {
                    "pipeline": pipeline,
                    "song": song,
                    "condition": condition,
                    "perturb_seed": perturb_seed,
                    "sample_seed": sample_seed,
                    "input_stem": stem,
                }
                run_id = "mr-" + canonical_sha256(identity)[:16]
                rows_by_pipeline[pipeline].append(
                    {
                        "schema_version": RUN_SCHEMA_VERSION,
                        "run_id": run_id,
                        **identity,
                        "input": dict(entry),
                        "expected_attempt_id": "attempt-001",
                    }
                )
    rows = rows_by_pipeline["offline"] + rows_by_pipeline["rt"]
    if len(rows) != 160 or len({row["run_id"] for row in rows}) != 160:
        raise AssertionError("formal schedule must contain 160 unique model runs")
    order_seed = int(campaign_config["seeds"]["run_order"])
    # Keep pipeline blocks separate so the dedicated GPU server never competes
    # with an offline model process.  Conditions remain randomized/interleaved
    # within each 80-run block.
    random.Random(order_seed).shuffle(rows_by_pipeline["offline"])
    random.Random(order_seed + 1).shuffle(rows_by_pipeline["rt"])
    rows = rows_by_pipeline["offline"] + rows_by_pipeline["rt"]
    for index, row in enumerate(rows):
        row["schedule_index"] = index
    return rows


def build_qualification_schedule(
    input_manifest: Mapping[str, Any],
    campaign_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Rebuild the exact 20-row pre-formal qualification design."""
    validate_campaign_config(campaign_config, require_frozen=False)
    _assert_equal(
        campaign_config.get("status"),
        "qualification_candidate",
        "qualification config status",
    )
    entries = validate_input_manifest(input_manifest)
    qualification = _validate_qualification_rules(campaign_config)
    songs = {
        str(_entry_value(entry, "song", "source_stem")) for entry in entries
    }
    dense_song = str(qualification["dense_song"])
    tail_songs = [str(song) for song in qualification["tail_songs"]]
    if dense_song not in songs or not set(tail_songs).issubset(songs):
        raise ValueError("qualification song selectors are not in the input manifest")
    formal = build_run_schedule(
        input_manifest,
        campaign_config,
        require_frozen=False,
    )
    sample_seed = int(qualification["sample_seed"])
    perturb_seed = int(qualification["perturb_seed"])

    def select(
        *, pipeline: str, song: str, condition: str, selected_pseed: int | None
    ) -> dict[str, Any]:
        matches = [
            row
            for row in formal
            if row["pipeline"] == pipeline
            and row["song"] == song
            and row["condition"] == condition
            and int(row["sample_seed"]) == sample_seed
            and row.get("perturb_seed") == selected_pseed
        ]
        if len(matches) != 1:
            raise ValueError(f"qualification selector matched {len(matches)} rows")
        return copy.deepcopy(matches[0])

    def qualify(
        base: Mapping[str, Any],
        kind: str,
        replicate: str,
        **overrides: Any,
    ) -> dict[str, Any]:
        row = copy.deepcopy(dict(base))
        identity = {
            "kind": kind,
            "replicate": replicate,
            "source_run_id": row["run_id"],
            **overrides,
        }
        row["formal_source_run_id"] = row["run_id"]
        row["run_id"] = "qual-" + canonical_sha256(identity)[:16]
        row["qualification_kind"] = kind
        row["qualification_replicate"] = replicate
        row["runtime_overrides"] = overrides
        return row

    offline = select(
        pipeline="offline",
        song=dense_song,
        condition="both",
        selected_pseed=perturb_seed,
    )
    realtime = select(
        pipeline="rt",
        song=dense_song,
        condition="both",
        selected_pseed=perturb_seed,
    )
    rows = [
        qualify(offline, "determinism_offline", replicate)
        for replicate in ("A", "B")
    ]
    rows.extend(
        qualify(
            realtime,
            "determinism_rt",
            replicate,
            playback_tempo=60,
            tail_beats=24,
        )
        for replicate in ("A", "B")
    )
    for tempo in qualification["tempo_candidates"]:
        for condition, selected_pseed in (("sham", None), ("both", perturb_seed)):
            base = select(
                pipeline="rt",
                song=dense_song,
                condition=condition,
                selected_pseed=selected_pseed,
            )
            rows.append(
                qualify(
                    base,
                    f"tempo_{tempo}",
                    condition,
                    playback_tempo=int(tempo),
                    tail_beats=24,
                )
            )
    for tempo in qualification["tempo_candidates"]:
        for song in tail_songs:
            base = select(
                pipeline="rt",
                song=song,
                condition="both",
                selected_pseed=perturb_seed,
            )
            for tail in qualification["tail_candidates"]:
                rows.append(
                    qualify(
                        base,
                        f"tail_{tempo}_{song}",
                        str(tail),
                        playback_tempo=int(tempo),
                        tail_beats=int(tail),
                    )
                )
    for index, row in enumerate(rows):
        row["schedule_index"] = index
    if len(rows) != 20 or len({row["run_id"] for row in rows}) != 20:
        raise AssertionError("qualification schedule must contain 20 unique rows")
    return rows


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def qualification_static_gate_errors(
    static_summary: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
) -> list[str]:
    """Derive the static-input gate errors from the pinned conversion summary."""
    errors: list[str] = []
    expected = {
        "schema_version": "streammuse.midi_to_npz_summary.v1",
        "status": "ok",
        "expected": 40,
        "converted": 40,
        "skipped": 0,
        "exact_stem_set": True,
        "ticks_per_beat": 4,
        "updated_manifest_sha256": candidate_config["input_manifest"]["sha256"],
    }
    for field, wanted in expected.items():
        if static_summary.get(field) != wanted:
            errors.append(
                f"static summary {field}: expected={wanted!r}, "
                f"got={static_summary.get(field)!r}"
            )
    if static_summary.get("errors") != []:
        errors.append("static summary contains conversion errors")

    manifest_path = Path(candidate_config["input_manifest"]["path"]).resolve()
    manifest = _read_json_object(manifest_path, "qualification input manifest")
    entries = {
        str(_entry_value(entry, "stem", "input_id", "output_stem")): entry
        for entry in _manifest_entries(manifest)
    }
    results = static_summary.get("results")
    if not isinstance(results, list) or len(results) != 40:
        errors.append("static summary must contain 40 per-input results")
        return errors
    result_stems = [
        str(row.get("stem")) for row in results if isinstance(row, Mapping)
    ]
    if sorted(result_stems) != sorted(entries):
        errors.append("static summary result stem set differs from frozen input manifest")
    for row in results:
        if not isinstance(row, Mapping):
            errors.append("static summary result is not an object")
            continue
        stem = str(row.get("stem"))
        entry = entries.get(stem)
        roll = row.get("roll_gate")
        if row.get("status") != "converted" or not isinstance(roll, Mapping):
            errors.append(f"{stem}: missing converted roll gate")
            continue
        differing = roll.get("differing_cells")
        if isinstance(differing, bool) or not isinstance(differing, int) or differing != 0:
            errors.append(f"{stem}: MIDI/NPZ roll differs")
        if entry is not None:
            horizon = roll.get("horizon_ticks")
            if (
                isinstance(horizon, bool)
                or not isinstance(horizon, int)
                or horizon != int(entry["validation_horizon_ticks"])
            ):
                errors.append(f"{stem}: validation horizon mismatch")
            npz = entry.get("npz", entry.get("paths", {}).get("npz", {}))
            if row.get("npz_sha256") != npz.get("sha256"):
                errors.append(f"{stem}: NPZ hash mismatch")
    return errors


def _qualification_single_artifact(root: Path, pattern: str) -> Path:
    matches = [path for path in root.rglob(pattern) if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one qualification artifact {pattern!r} beneath "
            f"{root}, got {matches}"
        )
    return matches[0]


def _qualification_offline_tokens(attempt: Path) -> list[int]:
    payload = _read_json_object(
        _qualification_single_artifact(attempt, "*_tokens.json"),
        "qualification offline tokens",
    )
    raw = payload.get("sampled_tokens")
    if not isinstance(raw, list):
        return []
    try:
        return [int(value) for value in raw]
    except (TypeError, ValueError):
        return []


def _qualification_rt_trace_signature(attempt: Path) -> list[dict[str, Any]]:
    validity = _read_json_object(
        _qualification_single_artifact(attempt, "validity.json"),
        "qualification RT validity",
    )
    content = validity.get("content", {})
    if not isinstance(content, Mapping):
        return []
    coverage = content.get("analysis_request_coverage")
    if (
        content.get("request_tick_contract_valid") is not True
        or isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or float(coverage) != 1.0
    ):
        return []
    requests = validity.get("requests")
    if not isinstance(requests, list) or not requests:
        return []
    if any(not isinstance(row, Mapping) for row in requests):
        return []
    try:
        ordered = sorted(requests, key=lambda row: int(row["generation_start_tick"]))
    except (KeyError, TypeError, ValueError):
        return []
    fields = (
        "generation_start_tick",
        "raw_token_digest",
        "input_increment_digest",
        "input_cumulative_digest",
        "part0_roll_digest",
        "part0_token_digest",
        "context_start_tick",
    )
    signature = [{field: row.get(field) for field in fields} for row in ordered]
    for row in signature:
        for field in fields:
            value = row[field]
            if field in {"generation_start_tick", "context_start_tick"}:
                if isinstance(value, bool) or not isinstance(value, int):
                    return []
            elif not isinstance(value, str) or not value.strip():
                return []
    return signature


def _qualification_roll_digest(path: Path, *, end_tick: int) -> str:
    # Local import keeps the shared semantic constants usable without eagerly
    # importing the MIDI analysis dependency in config-only callers.
    from streammuse.experiments.robustness_metrics import load_midi_roll

    roll = load_midi_roll(path, end_tick=end_tick)
    return canonical_sha256(
        {
            "end_tick": roll.end_tick,
            "sustain": sorted(roll.sustain),
            "onsets": sorted(roll.onsets),
        }
    )


def build_qualification_artifact_evidence(
    candidate_config: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    static_summary: Mapping[str, Any],
    attempts: Mapping[str, tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Normalize verified qualification artifacts into decision-only evidence.

    ``attempts`` must come from ``verify_attempt_verdict``.  This function does
    not trust any decision fields from a qualification result; it reads raw
    tokens, RT request traces and theoretical MIDI from the immutable attempt
    trees.
    """
    validate_campaign_config(candidate_config, require_frozen=False)
    if len(schedule) != 20 or len({str(row.get("run_id")) for row in schedule}) != 20:
        raise ValueError("qualification artifact evidence requires exact 20-row schedule")
    expected_ids = [str(row["run_id"]) for row in schedule]
    _assert_equal(set(attempts), set(expected_ids), "qualification attempt IDs")

    grouped: dict[str, list[tuple[Mapping[str, Any], Path, Mapping[str, Any]]]] = {}
    for row in schedule:
        run_id = str(row["run_id"])
        attempt, verdict = attempts[run_id]
        grouped.setdefault(str(row["qualification_kind"]), []).append(
            (row, Path(attempt), verdict)
        )

    offline: list[dict[str, Any]] = []
    for _row, attempt, verdict in grouped.get("determinism_offline", []):
        offline.append(
            {
                "content_valid": verdict.get("content_valid") is True,
                "sampled_tokens": _qualification_offline_tokens(attempt),
                "generated_midi_sha256": file_sha256(
                    _qualification_single_artifact(attempt, "*_generated.mid")
                ),
            }
        )

    realtime: list[dict[str, Any]] = []
    for row, attempt, verdict in grouped.get("determinism_rt", []):
        end_tick = int(row["input"]["analysis_end_tick"])
        realtime.append(
            {
                "content_valid": verdict.get("content_valid") is True,
                "trace_signature": _qualification_rt_trace_signature(attempt),
                "theoretical_roll_sha256": _qualification_roll_digest(
                    _qualification_single_artifact(attempt, "theoretical_model.mid"),
                    end_tick=end_tick,
                ),
            }
        )

    tempo: dict[str, list[dict[str, bool]]] = {}
    for tempo_value in QUALIFICATION_TEMPO_CANDIDATES:
        tempo[str(tempo_value)] = [
            {
                "content_valid": verdict.get("content_valid") is True,
                "operational_valid": verdict.get("operational_valid") is True,
            }
            for _row, _attempt, verdict in grouped.get(
                f"tempo_{tempo_value}", []
            )
        ]

    tail: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for tempo_value in QUALIFICATION_TEMPO_CANDIDATES:
        by_song: dict[str, dict[str, dict[str, Any]]] = {}
        for song in QUALIFICATION_TAIL_SONGS:
            by_tail: dict[str, dict[str, Any]] = {}
            for row, attempt, verdict in grouped.get(
                f"tail_{tempo_value}_{song}", []
            ):
                tail_value = str(int(row["runtime_overrides"]["tail_beats"]))
                end_tick = int(row["input"]["analysis_end_tick"])
                by_tail[tail_value] = {
                    "content_valid": verdict.get("content_valid") is True,
                    "trace_signature": _qualification_rt_trace_signature(attempt),
                    "theoretical_roll_sha256": _qualification_roll_digest(
                        _qualification_single_artifact(
                            attempt, "theoretical_model.mid"
                        ),
                        end_tick=end_tick,
                    ),
                }
            by_song[song] = by_tail
        tail[str(tempo_value)] = by_song

    return {
        "static_errors": qualification_static_gate_errors(
            static_summary, candidate_config
        ),
        "offline_runs": offline,
        "rt_runs": realtime,
        "tempo_runs": tempo,
        "tail_runs": tail,
    }


def derive_qualification_decision(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Purely derive all qualification decisions from normalized artifacts."""
    offline = evidence.get("offline_runs")
    realtime = evidence.get("rt_runs")
    tempo_runs = evidence.get("tempo_runs")
    tail_runs = evidence.get("tail_runs")
    static_errors = evidence.get("static_errors")
    if not isinstance(offline, list) or len(offline) != 2:
        raise ValueError("qualification requires exactly two offline determinism runs")
    if not isinstance(realtime, list) or len(realtime) != 2:
        raise ValueError("qualification requires exactly two RT determinism runs")
    if not isinstance(static_errors, list) or any(
        not isinstance(error, str) for error in static_errors
    ):
        raise ValueError("qualification static errors must be a list of strings")
    if not isinstance(tempo_runs, Mapping):
        raise ValueError("qualification tempo evidence must be an object")
    if not isinstance(tail_runs, Mapping):
        raise ValueError("qualification tail evidence must be an object")
    _assert_equal(
        set(tempo_runs),
        {str(value) for value in QUALIFICATION_TEMPO_CANDIDATES},
        "qualification tempo evidence keys",
    )
    _assert_equal(
        set(tail_runs),
        {str(value) for value in QUALIFICATION_TEMPO_CANDIDATES},
        "qualification tail tempo evidence keys",
    )

    offline_deterministic = bool(
        all(run.get("content_valid") is True for run in offline)
        and all(isinstance(run.get("sampled_tokens"), list) and run["sampled_tokens"] for run in offline)
        and offline[0]["sampled_tokens"] == offline[1]["sampled_tokens"]
        and offline[0].get("generated_midi_sha256")
        == offline[1].get("generated_midi_sha256")
    )
    rt_deterministic = bool(
        all(run.get("content_valid") is True for run in realtime)
        and all(isinstance(run.get("trace_signature"), list) and run["trace_signature"] for run in realtime)
        and realtime[0]["trace_signature"] == realtime[1]["trace_signature"]
        and realtime[0].get("theoretical_roll_sha256")
        == realtime[1].get("theoretical_roll_sha256")
    )

    tempo_checks: dict[str, bool] = {}
    for tempo_value in QUALIFICATION_TEMPO_CANDIDATES:
        key = str(tempo_value)
        runs = tempo_runs[key]
        if not isinstance(runs, list) or len(runs) != 2:
            raise ValueError(f"qualification tempo {tempo_value} requires exactly two runs")
        tempo_checks[key] = all(
            isinstance(run, Mapping)
            and run.get("content_valid") is True
            and run.get("operational_valid") is True
            for run in runs
        )
    selected_tempo = next(
        (
            tempo_value
            for tempo_value in QUALIFICATION_TEMPO_CANDIDATES
            if tempo_checks[str(tempo_value)]
        ),
        None,
    )

    tail_checks: dict[str, dict[str, Any]] = {}
    selected_by_song: dict[str, int | None] = {}
    if selected_tempo is None:
        for song in QUALIFICATION_TAIL_SONGS:
            selected_by_song[song] = None
            tail_checks[song] = {
                "content_valid": False,
                "trace_and_coverage_valid": False,
                "8_eq_16": False,
                "16_eq_24": False,
                "decision": None,
                "reason": "no_selected_tempo_tail_runs",
            }
    else:
        selected_tail_runs = tail_runs[str(selected_tempo)]
        if not isinstance(selected_tail_runs, Mapping):
            raise ValueError("qualification selected-tempo tail evidence must be an object")
        _assert_equal(
            set(selected_tail_runs),
            set(QUALIFICATION_TAIL_SONGS),
            "qualification tail evidence songs",
        )
        for song in QUALIFICATION_TAIL_SONGS:
            by_tail = selected_tail_runs[song]
            if not isinstance(by_tail, Mapping):
                raise ValueError(f"qualification tail evidence for song {song} must be an object")
            _assert_equal(
                set(by_tail),
                {str(value) for value in QUALIFICATION_TAIL_CANDIDATES},
                f"qualification tail evidence candidates for song {song}",
            )
            records = [by_tail[str(value)] for value in QUALIFICATION_TAIL_CANDIDATES]
            if any(not isinstance(record, Mapping) for record in records):
                raise ValueError(f"qualification tail evidence for song {song} is malformed")
            content_valid = all(
                record.get("content_valid") is True for record in records
            )
            trace_valid = all(
                isinstance(record.get("trace_signature"), list)
                and bool(record["trace_signature"])
                for record in records
            )
            tail_8, tail_16, tail_24 = records
            eq_8_16 = bool(
                content_valid
                and trace_valid
                and tail_8["trace_signature"] == tail_16["trace_signature"]
                and tail_8.get("theoretical_roll_sha256")
                == tail_16.get("theoretical_roll_sha256")
            )
            eq_16_24 = bool(
                content_valid
                and trace_valid
                and tail_16["trace_signature"] == tail_24["trace_signature"]
                and tail_16.get("theoretical_roll_sha256")
                == tail_24.get("theoretical_roll_sha256")
            )
            if eq_8_16 and eq_16_24:
                decision = 8
                reason = "8_16_24_converged"
            elif eq_16_24:
                decision = 16
                reason = "16_24_converged"
            else:
                decision = None
                reason = (
                    "tail_not_converged"
                    if content_valid and trace_valid
                    else "invalid_tail_evidence"
                )
            selected_by_song[song] = decision
            tail_checks[song] = {
                "content_valid": content_valid,
                "trace_and_coverage_valid": trace_valid,
                "8_eq_16": eq_8_16,
                "16_eq_24": eq_16_24,
                "decision": decision,
                "reason": reason,
            }

    selected_tail = (
        max(int(value) for value in selected_by_song.values() if value is not None)
        if selected_by_song and all(value is not None for value in selected_by_song.values())
        else None
    )
    static_valid = not static_errors
    failure_reasons: list[str] = []
    if not offline_deterministic:
        failure_reasons.append("offline_determinism_failed")
    if not rt_deterministic:
        failure_reasons.append("rt_determinism_failed")
    if not static_valid:
        failure_reasons.append("static_input_gate_failed")
    if selected_tempo is None:
        failure_reasons.append("no_operational_tempo")
    for song in QUALIFICATION_TAIL_SONGS:
        if tail_checks[song]["decision"] is None:
            failure_reasons.append(f"tail_{song}:{tail_checks[song]['reason']}")
    passed = not failure_reasons
    return {
        "offline_deterministic": offline_deterministic,
        "rt_deterministic": rt_deterministic,
        "static_input_gate": {"valid": static_valid, "errors": list(static_errors)},
        "tempo": {"checks": tempo_checks, "selected": selected_tempo},
        "tail": {"checks": tail_checks, "selected": selected_tail},
        "rule": QUALIFICATION_TAIL_RULE,
        "failure_reasons": failure_reasons,
        "passed": passed,
    }


def validate_qualification_result(
    result: Mapping[str, Any],
    *,
    candidate_config: Mapping[str, Any] | None = None,
    verify_files: bool = True,
    require_passed: bool = True,
) -> None:
    """Validate decision logic and the complete qualification evidence graph."""
    _assert_equal(
        result.get("schema_version"),
        QUALIFICATION_SCHEMA_VERSION,
        "qualification result schema_version",
    )
    candidate_path, candidate_sha = _validate_file_record(
        result.get("candidate_config"),
        "qualification candidate config",
        verify_file=verify_files,
    )
    _assert_equal(
        result.get("candidate_config_sha256"),
        candidate_sha,
        "qualification candidate_config_sha256",
    )
    schedule_path, schedule_sha = _validate_file_record(
        result.get("qualification_schedule"),
        "qualification schedule",
        verify_file=verify_files,
    )
    binding_path, binding_sha = _validate_file_record(
        result.get("qualification_campaign_binding"),
        "qualification campaign binding",
        verify_file=verify_files,
    )
    development_only = result.get("development_only")
    if not isinstance(development_only, bool):
        raise ValueError("qualification development_only must be boolean")
    if require_passed and development_only:
        raise ValueError("development-only qualification cannot satisfy a frozen campaign")

    candidate_from_file: dict[str, Any] | None = None
    if verify_files:
        candidate_from_file = _read_json_object(
            candidate_path, "qualification candidate"
        )
        if candidate_config is not None:
            _assert_equal(
                dict(candidate_config),
                candidate_from_file,
                "qualification candidate content",
            )
        candidate_config = candidate_from_file

    static = result.get("static_input_gate")
    if not isinstance(static, Mapping):
        raise ValueError("qualification static_input_gate must be an object")
    static_path, _ = _validate_file_record(
        {"path": static.get("summary_path"), "sha256": static.get("sha256")},
        "qualification static input summary",
        verify_file=verify_files,
    )
    if not isinstance(static.get("valid"), bool) or not isinstance(static.get("errors"), list):
        raise ValueError("qualification static gate requires boolean valid and errors list")
    if static.get("valid") is True and static.get("errors") != []:
        raise ValueError("valid qualification static gate cannot contain errors")

    for field in ("offline_deterministic", "rt_deterministic", "passed"):
        if not isinstance(result.get(field), bool):
            raise ValueError(f"qualification {field} must be boolean")
    tempo = result.get("tempo")
    if not isinstance(tempo, Mapping) or not isinstance(tempo.get("checks"), Mapping):
        raise ValueError("qualification tempo checks are required")
    tempo_checks = dict(tempo["checks"])
    _assert_equal(set(tempo_checks), {"30", "60"}, "qualification tempo check keys")
    if any(not isinstance(value, bool) for value in tempo_checks.values()):
        raise ValueError("qualification tempo checks must be boolean")
    selected_tempo = next(
        (value for value in (60, 30) if tempo_checks[str(value)]),
        None,
    )
    _assert_equal(tempo.get("selected"), selected_tempo, "qualification selected tempo")

    tail = result.get("tail")
    if not isinstance(tail, Mapping) or not isinstance(tail.get("checks"), Mapping):
        raise ValueError("qualification tail checks are required")
    tail_checks = dict(tail["checks"])
    if not tail_checks:
        raise ValueError("qualification tail checks cannot be empty")
    if candidate_config is not None:
        validate_campaign_config(
            candidate_config,
            require_frozen=False,
            verify_attestations=verify_files,
        )
        _assert_equal(
            candidate_config.get("status"),
            "qualification_candidate",
            "qualification candidate status",
        )
        _assert_equal(
            result.get("candidate_config"),
            {"path": str(candidate_path), "sha256": candidate_sha},
            "qualification candidate record",
        )
        _assert_equal(
            set(tail_checks),
            set(candidate_config["qualification"]["tail_songs"]),
            "qualification tail songs",
        )
    decisions: list[int | None] = []
    for song, raw_check in tail_checks.items():
        if not isinstance(raw_check, Mapping):
            raise ValueError(f"qualification tail check for {song} must be an object")
        for field in ("content_valid", "trace_and_coverage_valid", "8_eq_16", "16_eq_24"):
            if not isinstance(raw_check.get(field), bool):
                raise ValueError(f"qualification tail {song}.{field} must be boolean")
        valid = bool(
            raw_check["content_valid"] and raw_check["trace_and_coverage_valid"]
        )
        expected_decision = (
            8
            if valid and raw_check["8_eq_16"] and raw_check["16_eq_24"]
            else 16
            if valid and raw_check["16_eq_24"]
            else None
        )
        _assert_equal(
            raw_check.get("decision"),
            expected_decision,
            f"qualification tail decision for {song}",
        )
        decisions.append(expected_decision)
    selected_tail = (
        max(int(value) for value in decisions if value is not None)
        if decisions and all(value is not None for value in decisions)
        else None
    )
    _assert_equal(tail.get("selected"), selected_tail, "qualification selected tail")
    _assert_equal(
        result.get("rule"),
        QUALIFICATION_TAIL_RULE,
        "qualification tail rule",
    )
    failure_reasons = result.get("failure_reasons")
    if not isinstance(failure_reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in failure_reasons
    ):
        raise ValueError("qualification failure_reasons must be a list of strings")
    expected_passed = bool(
        result["offline_deterministic"]
        and result["rt_deterministic"]
        and static["valid"]
        and selected_tempo is not None
        and selected_tail is not None
    )
    _assert_equal(result.get("passed"), expected_passed, "qualification passed")
    if require_passed and not expected_passed:
        raise ValueError("qualification result did not pass all gates")

    evidence = result.get("run_evidence")
    if not isinstance(evidence, list) or len(evidence) != 20:
        raise ValueError("qualification result requires exactly 20 run evidence records")
    evidence_ids: list[str] = []
    for index, record in enumerate(evidence):
        if not isinstance(record, Mapping):
            raise ValueError("qualification run evidence must be objects")
        run_id = record.get("run_id")
        attempt_id = record.get("attempt_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("qualification run evidence requires run_id")
        if attempt_id != "attempt-001":
            raise ValueError(
                "qualification run evidence must use exact attempt-001 without retry"
            )
        verdict_path, _ = _validate_file_record(
            record.get("verdict"),
            f"qualification run evidence {index} verdict",
            verify_file=verify_files,
        )
        if verify_files and verdict_path.parent.name != attempt_id:
            raise ValueError("qualification evidence verdict path/attempt_id mismatch")
        evidence_ids.append(run_id)
    if len(set(evidence_ids)) != 20:
        raise ValueError("qualification run evidence IDs must be unique")

    if candidate_config is not None and verify_files:
        input_path, _ = _validate_file_record(
            candidate_config["input_manifest"],
            "qualification input manifest",
            verify_file=True,
        )
        input_manifest = _read_json_object(input_path, "qualification input manifest")
        expected_schedule = build_qualification_schedule(input_manifest, candidate_config)
        actual_schedule = read_jsonl(schedule_path)
        _assert_equal(actual_schedule, expected_schedule, "qualification schedule")
        _assert_equal(
            evidence_ids,
            [str(row["run_id"]) for row in expected_schedule],
            "qualification run evidence order",
        )
        binding = _read_json_object(binding_path, "qualification campaign binding")
        expected_binding = {
            "schema_version": "streammuse.melody_robustness.campaign_binding.v1",
            "qualification": True,
            "campaign_config_path": str(candidate_path),
            "campaign_config_sha256": candidate_sha,
            "run_schedule_path": str(schedule_path),
            "run_schedule_sha256": schedule_sha,
            "input_manifest_path": str(input_path),
            "input_manifest_sha256": str(candidate_config["input_manifest"]["sha256"]),
            "checkpoint_path": str(Path(candidate_config["checkpoint"]["path"]).resolve()),
            "checkpoint_sha256": str(candidate_config["checkpoint"]["sha256"]),
            "code_identity": str(candidate_config["code_identity"]),
        }
        _assert_equal(binding, expected_binding, "qualification campaign binding")
        _assert_equal(file_sha256(binding_path), binding_sha, "qualification binding sha256")
        bound = {**binding, "campaign_binding_sha256": binding_sha}
        runs_root = binding_path.parent / "runs"
        if not runs_root.is_dir():
            raise FileNotFoundError(
                f"qualification runs directory missing: {runs_root}"
            )
        actual_run_directories = {
            path.name for path in runs_root.iterdir() if path.is_dir()
        }
        expected_run_directories = {
            str(row["run_id"]) for row in expected_schedule
        }
        _assert_equal(
            actual_run_directories,
            expected_run_directories,
            "qualification run directory set",
        )
        verified_attempts: dict[
            str, tuple[Path, Mapping[str, Any]]
        ] = {}
        for row, record in zip(expected_schedule, evidence, strict=True):
            verdict_path, verdict_sha = _validate_file_record(
                record["verdict"],
                f"qualification evidence {row['run_id']} verdict",
                verify_file=True,
            )
            expected_verdict_path = (
                runs_root
                / str(row["run_id"])
                / "attempt-001"
                / "verdict.json"
            ).resolve()
            _assert_equal(
                verdict_path,
                expected_verdict_path,
                "qualification immutable verdict path",
            )
            attempt, verdict, _ = verify_attempt_verdict(
                verdict_path.parent.parent,
                row,
                bound,
                expected_attempt_id="attempt-001",
                forbid_other_attempts=True,
            )
            _assert_equal(attempt.name, record["attempt_id"], "qualification attempt_id")
            _assert_equal(
                file_sha256(attempt / "verdict.json"),
                verdict_sha,
                "qualification immutable verdict sha256",
            )
            _assert_equal(verdict["run_id"], record["run_id"], "qualification evidence run_id")
            verified_attempts[str(row["run_id"])] = (attempt, verdict)
        if static_path != Path(str(static["summary_path"])).resolve():
            raise AssertionError("unreachable static summary path normalization mismatch")
        static_summary = _read_json_object(
            static_path, "qualification static input summary"
        )
        artifact_evidence = build_qualification_artifact_evidence(
            candidate_config,
            expected_schedule,
            static_summary,
            verified_attempts,
        )
        derived = derive_qualification_decision(artifact_evidence)
        for field in (
            "offline_deterministic",
            "rt_deterministic",
            "tempo",
            "tail",
            "rule",
            "failure_reasons",
            "passed",
        ):
            _assert_equal(
                result.get(field),
                derived[field],
                f"artifact-derived qualification {field}",
            )
        _assert_equal(
            {
                "valid": static.get("valid"),
                "errors": static.get("errors"),
            },
            derived["static_input_gate"],
            "artifact-derived qualification static_input_gate",
        )


def validate_frozen_qualification(
    config: Mapping[str, Any],
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Verify that a C5 config is the exact qualified finalization of its candidate."""
    validate_campaign_config(
        config,
        require_frozen=True,
        verify_attestations=verify_files,
    )
    candidate_path, candidate_sha = _validate_file_record(
        config.get("qualification_candidate"),
        "qualification_candidate",
        verify_file=verify_files,
    )
    result_path, result_sha = _validate_file_record(
        config.get("qualification_result"),
        "qualification_result",
        verify_file=verify_files,
    )
    renderer_identity = config.get("listening", {}).get("renderer_identity")
    if config.get("listening", {}).get("schema_version") == TRIANGLE_LISTENING_SCHEMA_VERSION:
        renderer_path, renderer_sha = _validate_file_record(
            renderer_identity,
            "listening.renderer_identity",
            verify_file=verify_files,
        )
        if verify_files:
            from streammuse.experiments.triangle_listening import (
                validate_triangle_renderer_identity,
            )

            renderer_payload = _read_json_object(
                renderer_path, "triangle renderer identity"
            )
            validate_triangle_renderer_identity(renderer_payload, verify_files=True)
        renderer_identity = {"path": str(renderer_path), "sha256": renderer_sha}
    if not verify_files:
        return {}
    candidate = _read_json_object(candidate_path, "qualification candidate")
    result = _read_json_object(result_path, "qualification result")
    validate_qualification_result(
        result,
        candidate_config=candidate,
        verify_files=True,
        require_passed=True,
    )
    _assert_equal(
        result.get("candidate_config"),
        config.get("qualification_candidate"),
        "frozen qualification candidate record",
    )
    _assert_equal(
        result.get("candidate_config_sha256"),
        candidate_sha,
        "frozen candidate sha256",
    )
    expected = copy.deepcopy(candidate)
    expected["status"] = "qualified_frozen"
    expected["runtime"]["playback_tempo"] = int(result["tempo"]["selected"])
    expected["runtime"]["tail_beats"] = int(result["tail"]["selected"])
    expected["qualification_candidate"] = {
        "path": str(candidate_path),
        "sha256": candidate_sha,
    }
    expected["qualification_result"] = {
        "path": str(result_path),
        "sha256": result_sha,
    }
    expected["listening"]["selection_manifest_path"] = config["listening"][
        "selection_manifest_path"
    ]
    expected["listening"]["selection_manifest_sha256"] = config["listening"][
        "selection_manifest_sha256"
    ]
    if config["listening"].get("schema_version") == TRIANGLE_LISTENING_SCHEMA_VERSION:
        expected["listening"]["renderer_identity"] = renderer_identity
    _assert_equal(dict(config), expected, "frozen campaign finalization")
    return result


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


VERDICT_BINDING_FIELDS = (
    "campaign_config_sha256",
    "run_schedule_sha256",
    "input_manifest_sha256",
    "checkpoint_sha256",
    "code_identity",
    "campaign_binding_sha256",
    "qualification_result_sha256",
)

OFFLINE_REQUIRED_ARTIFACT_SCHEMA = (
    "streammuse.melody_robustness.offline_required_artifacts.v1"
)
RT_REQUIRED_ARTIFACT_SCHEMA = (
    "streammuse.melody_robustness.rt_required_artifacts.v1"
)
OFFLINE_REQUIRED_ATTEMPT_ARTIFACTS = {
    "command": "command.json",
    "process_log": "process.log",
    "run_config": "_run_config.json",
    "token_trace": "_tokens.json",
    "generated_midi": "_generated.mid",
    "ground_truth_midi": "_gt.mid",
}
RT_REQUIRED_ATTEMPT_ARTIFACTS = {
    "command": "command.json",
    "process_log": "process.log",
    "rt_input_gate": "rt_input_gate.json",
    "post_run_runtime_info": "post_run_runtime_info.json",
    "session_config": "session_config.json",
    "theoretical_model_midi": "theoretical_model.mid",
    "theoretical_model_summary": "theoretical_model_summary.json",
    "combined_midi": "combined.mid",
    "model_schedule_trace": "model_schedule_trace.jsonl",
    "request_lifecycle": "request_lifecycle.jsonl",
    "validity": "validity.json",
    "full_inference_trace": "inferences.json",
    "actual_event_trace": "events.jsonl",
}


def _verify_required_attempt_contract(
    *,
    attempt: Path,
    verdict: Mapping[str, Any],
    indexed_records: Mapping[str, Mapping[str, Any]],
    campaign_binding: Mapping[str, Any],
) -> None:
    """Apply the same formal artifact strength to every downstream consumer."""
    if "qualification_result_sha256" not in campaign_binding:
        return
    pipeline = verdict.get("pipeline")
    if pipeline == "rt":
        expected_schema = RT_REQUIRED_ARTIFACT_SCHEMA
        expected_artifacts = RT_REQUIRED_ATTEMPT_ARTIFACTS
        expected_gate = "rt_artifact_gate.json"
    elif pipeline == "offline":
        expected_schema = OFFLINE_REQUIRED_ARTIFACT_SCHEMA
        expected_artifacts = OFFLINE_REQUIRED_ATTEMPT_ARTIFACTS
        expected_gate = "offline_post_run_gate.json"
    else:
        raise ValueError(f"unsupported formal verdict pipeline: {pipeline!r}")
    expected_labels = set(expected_artifacts)
    contract = verdict.get("required_artifact_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("formal verdict lacks required_artifact_contract")
    _assert_equal(contract.get("schema_version"), expected_schema, "artifact contract schema")
    _assert_equal(contract.get("enforced"), True, "artifact contract enforced")
    _assert_equal(contract.get("gate_path"), expected_gate, "artifact contract gate path")
    _assert_equal(
        contract.get("required_labels"),
        sorted(expected_labels),
        "artifact contract required labels",
    )
    gate_path = attempt / expected_gate
    if expected_gate not in indexed_records or not gate_path.is_file():
        raise FileNotFoundError(f"required artifact gate missing or unindexed: {gate_path}")
    gate = _read_json_object(gate_path, "required artifact gate")
    _assert_equal(gate.get("schema_version"), expected_schema, "artifact gate schema")
    _assert_equal(gate.get("enforced", True), True, "artifact gate enforced")
    _assert_equal(
        gate.get("content_valid"),
        verdict.get("content_valid"),
        "artifact gate/verdict content_valid",
    )
    # A failed formal attempt is immutable retry evidence, not a successful
    # source artifact.  Its binding, verdict, complete file index, hashes and
    # gate identity are still verified above; only successful attempts claim
    # the complete pipeline-specific artifact/semantic contract below.
    if verdict.get("content_valid") is not True:
        return
    required = gate.get("required_artifacts")
    if not isinstance(required, Mapping):
        raise ValueError("artifact gate required_artifacts must be an object")
    _assert_equal(set(required), expected_labels, "artifact gate required labels")
    for label, expected_name in expected_artifacts.items():
        record = required.get(label)
        if not isinstance(record, Mapping):
            raise ValueError(f"artifact gate record {label} must be an object")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"artifact gate record {label} lacks path")
        basename = Path(relative).name
        if expected_name.startswith("_"):
            if not basename.endswith(expected_name):
                raise ValueError(f"artifact gate record {label} has wrong filename")
        elif basename != expected_name:
            raise ValueError(f"artifact gate record {label} has wrong filename")
        indexed = indexed_records.get(relative)
        if indexed is None or dict(indexed) != dict(record):
            raise ValueError(f"artifact gate record {label} differs from artifact index")
    if pipeline == "rt":
        semantic = gate.get("semantic_evidence")
        if not isinstance(semantic, Mapping):
            raise ValueError("RT artifact gate lacks semantic_evidence")
        for field in (
            "raw_token_trace_complete",
            "token_decode_reconciliation_complete",
            "output_event_digest_complete",
            "lifecycle_reconciliation_complete",
            "trace_inference_equal",
            "theoretical_semantic_equal",
            "combined_schedule_consistent",
            "combined_semantic_equal",
        ):
            _assert_equal(semantic.get(field), True, f"RT semantic evidence {field}")


def verify_attempt_verdict(
    run_dir: str | Path,
    row: Mapping[str, Any],
    campaign_binding: Mapping[str, Any],
    *,
    require_content_valid: bool | None = None,
    expected_attempt_id: str | None = None,
    forbid_other_attempts: bool = False,
    require_latest_pointer: bool = True,
) -> tuple[Path, dict[str, Any], set[Path]]:
    """Verify the immutable verdict, binding, exact artifact set, sizes and hashes."""
    root = Path(run_dir).resolve()
    pointer = root / "latest_verdict.json"
    if require_latest_pointer:
        if not pointer.is_file():
            raise FileNotFoundError(f"latest verdict missing: {pointer}")
        verdict = _read_json_object(pointer, "latest verdict")
    else:
        if expected_attempt_id is None:
            raise ValueError("historical verification requires expected_attempt_id")
        if re.fullmatch(r"attempt-[0-9]{3}", expected_attempt_id) is None:
            raise ValueError("expected historical attempt_id is invalid")
        immutable_path = root / expected_attempt_id / "verdict.json"
        if not immutable_path.is_file():
            raise FileNotFoundError(f"immutable verdict missing: {immutable_path}")
        verdict = _read_json_object(immutable_path, "immutable verdict")
    _assert_equal(
        verdict.get("schema_version"),
        "streammuse.melody_robustness.verdict.v1",
        "verdict schema_version",
    )
    _assert_equal(verdict.get("run_id"), row.get("run_id"), "verdict run_id")
    _assert_equal(verdict.get("pipeline"), row.get("pipeline"), "verdict pipeline")
    if not isinstance(verdict.get("content_valid"), bool) or not isinstance(
        verdict.get("operational_valid"), bool
    ):
        raise ValueError("verdict validity fields must be boolean")
    if require_content_valid is not None:
        _assert_equal(
            verdict["content_valid"],
            require_content_valid,
            "verdict content_valid",
        )
    attempt_id = verdict.get("attempt_id")
    if not isinstance(attempt_id, str) or re.fullmatch(r"attempt-[0-9]{3}", attempt_id) is None:
        raise ValueError("verdict attempt_id is invalid")
    if expected_attempt_id is not None:
        _assert_equal(attempt_id, expected_attempt_id, "verdict attempt_id")
    if forbid_other_attempts:
        attempt_directories = {
            path.name
            for path in root.iterdir()
            if path.is_dir() and re.fullmatch(r"attempt-[0-9]{3}", path.name)
        }
        expected_attempts = {
            expected_attempt_id if expected_attempt_id is not None else attempt_id
        }
        _assert_equal(
            attempt_directories,
            expected_attempts,
            "immutable attempt directory set",
        )
    attempt = (root / attempt_id).resolve()
    if not attempt.is_dir() or not attempt.is_relative_to(root):
        raise ValueError("verdict attempt directory escapes or is missing")
    immutable = attempt / "verdict.json"
    if not immutable.is_file() or _read_json_object(immutable, "immutable verdict") != verdict:
        raise ValueError("latest verdict does not match immutable attempt verdict")
    for field in VERDICT_BINDING_FIELDS:
        if field in campaign_binding:
            _assert_equal(
                verdict.get(field),
                campaign_binding[field],
                f"verdict {field}",
            )
    index = verdict.get("artifact_index")
    if not isinstance(index, list) or not index:
        raise ValueError("verdict artifact_index must be a non-empty list")
    indexed: set[Path] = set()
    indexed_records: dict[str, Mapping[str, Any]] = {}
    relative_names: set[str] = set()
    for record in index:
        if not isinstance(record, Mapping):
            raise ValueError("verdict artifact index records must be objects")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative or relative in relative_names:
            raise ValueError("verdict artifact index path is missing or duplicated")
        relative_names.add(relative)
        indexed_records[relative] = record
        path = (attempt / relative).resolve()
        if not path.is_relative_to(attempt):
            raise ValueError("verdict artifact path escapes the attempt directory")
        size = record.get("size")
        digest = _require_lower_hex(
            record.get("sha256"), {64}, f"verdict artifact {relative} sha256"
        )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"verdict artifact {relative} size is invalid")
        if not path.is_file():
            raise FileNotFoundError(f"indexed artifact missing: {path}")
        if path.stat().st_size != size or file_sha256(path) != digest:
            raise ValueError(f"indexed artifact size/hash mismatch: {path}")
        indexed.add(path)
    actual = {
        path.resolve()
        for path in attempt.rglob("*")
        if path.is_file()
        and path.name not in {"verdict.json", "verdict.json.sha256"}
    }
    if actual != indexed:
        raise ValueError(
            "attempt artifact set differs from verdict index: "
            f"missing={sorted(map(str, indexed - actual))}, "
            f"extra={sorted(map(str, actual - indexed))}"
        )
    _verify_required_attempt_contract(
        attempt=attempt,
        verdict=verdict,
        indexed_records=indexed_records,
        campaign_binding=campaign_binding,
    )
    return attempt, verdict, indexed


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)))
    digest = file_sha256(destination)
    destination.with_name(destination.name + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="ascii"
    )
    return digest


def require_exact_artifacts(directory: str | Path, expected: Sequence[str]) -> None:
    root = Path(directory)
    actual = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
    wanted = sorted(str(item) for item in expected)
    if actual != wanted:
        raise ValueError(f"artifact set mismatch: expected={wanted}, actual={actual}")
