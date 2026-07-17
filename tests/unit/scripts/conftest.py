from __future__ import annotations

import importlib.util
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

import pytest

from streammuse.experiments.melody_robustness import (
    ATTESTATION_BUNDLE_SCHEMA_VERSION,
    CHECKPOINT_IDENTITY_SCHEMA_VERSION,
    CODE_IDENTITY_SCHEMA_VERSION,
    CONDITION_TABLE,
    ENVIRONMENT_IDENTITY_SCHEMA_VERSION,
    SEEDS,
    build_qualification_schedule,
    default_campaign_config,
    file_sha256,
    qualification_spec_contract,
    write_canonical_json,
)
from streammuse.experiments.robustness_metrics import Roll, write_roll_midi
from streammuse.experiments.triangle_listening import build_triangle_selection_manifest
from streammuse.experiments.triangle_listening import (
    TRIANGLE_GAIN_POLICY,
    TRIANGLE_RENDERER_SCHEMA_VERSION,
    TRIANGLE_RENDER_BPM,
    TRIANGLE_RENDER_SAMPLE_RATE,
    TRIANGLE_SYNTH_GAIN,
    TRIANGLE_TRUE_PEAK_IMPLEMENTATION,
    TRIANGLE_TRUE_PEAK_LIMIT_DBTP,
)


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RobustnessFixture:
    root: Path
    checkpoint: Path
    attestation_dir: Path
    input_manifest: Path
    listening_manifest: Path
    renderer_identity: Path
    campaign_config: Path
    qualification_candidate: Path
    qualification_result: Path
    qualification_schedule: Path
    qualification_output_root: Path
    melody_midi: Path
    accompaniment_midi: Path
    condition_npz: Path


@pytest.fixture
def load_script() -> Callable[[str], ModuleType]:
    """Load a repository script without requiring ``scripts`` to be a package."""

    loaded: list[str] = []

    def load(name: str) -> ModuleType:
        module_name = f"_streammuse_test_script_{name}_{len(loaded)}"
        script_path = ROOT / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        loaded.append(module_name)
        spec.loader.exec_module(module)
        return module

    yield load

    for module_name in loaded:
        sys.modules.pop(module_name, None)


def _write_manifest(path: Path, artifacts: dict[str, Path]) -> None:
    entries = []
    for song_index in range(1, 6):
        song = str(song_index)
        for condition, spec in CONDITION_TABLE.items():
            if condition == "sham":
                perturb_seeds: list[int | None] = [None]
            elif condition == "high":
                perturb_seeds = [int(SEEDS["high_perturb"])]
            else:
                perturb_seeds = [int(seed) for seed in SEEDS["perturb"]]
            for perturb_seed in perturb_seeds:
                suffix = "none" if perturb_seed is None else str(perturb_seed)
                entries.append(
                    {
                        "stem": f"{song}__{condition}__ps-{suffix}",
                        "song": song,
                        "source_stem": song,
                        "condition": condition,
                        "perturb_seed": perturb_seed,
                        "pitch_probability": spec["pitch_probability"],
                        "onset_probability": spec["onset_probability"],
                        # The listening package freezes a 50-beat (200 model-tick)
                        # excerpt, so the shared campaign fixture must expose a
                        # real analysis window large enough to contain it.
                        "analysis_end_tick": 240,
                        "last_input_note_off_tick": 16,
                        "validation_horizon_ticks": 240,
                        **(
                            {
                                "factorial_pairing": {
                                    "latent_pairing_verified": True,
                                    "collision_interaction": False,
                                }
                            }
                            if condition in {"pitch", "onset", "both"}
                            else {}
                        ),
                        "paths": {
                            key: {
                                "path": str(value.relative_to(path.parent)),
                                "sha256": file_sha256(value),
                            }
                            for key, value in artifacts.items()
                        },
                    }
                )
    assert len(entries) == 40
    payload = {
        "schema_version": "streammuse.melody_perturbation.v1",
        "input_count": 40,
        "expected_input_count": 40,
        "model_ticks_per_beat": 4,
        "entries": entries,
        "exact_stems": sorted(entry["stem"] for entry in entries),
        "perturb_seeds": list(SEEDS["perturb"]),
        "high_pseed": SEEDS["high_perturb"],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


@pytest.fixture
def robustness_fixture(tmp_path: Path) -> RobustnessFixture:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    checkpoint = artifact_dir / "model.safetensors"
    checkpoint.write_bytes(b"deterministic-test-checkpoint")
    melody = artifact_dir / "melody.mid"
    accompaniment = artifact_dir / "accompaniment.mid"
    npz = artifact_dir / "condition.npz"
    sidecar = artifact_dir / "condition.perturbation.json"
    # Driver dry-runs only hash these fixtures; MIDI/NPZ parsing is tested elsewhere.
    melody.write_bytes(b"MThd-test-melody")
    accompaniment.write_bytes(b"MThd-test-accompaniment")
    npz.write_bytes(b"PK-test-npz")
    sidecar.write_text('{"schema_version":"test-sidecar"}\n', encoding="utf-8")
    input_manifest = tmp_path / "input_manifest.json"
    _write_manifest(
        input_manifest,
        {
            "output_midi": melody,
            "source_midi": melody,
            "npz": npz,
            "acc_copy": accompaniment,
            "source_acc": accompaniment,
            "sidecar": sidecar,
        },
    )
    listening_manifest = tmp_path / "listening_selection.json"
    write_canonical_json(
        listening_manifest,
        build_triangle_selection_manifest(
            json.loads(input_manifest.read_text(encoding="utf-8")),
            manifest_path=input_manifest,
            blind_order_seed=int(SEEDS["blind_order"]),
            excerpt_starts={str(index): 0 for index in range(1, 6)},
        ),
    )
    renderer_binary = artifact_dir / "fluidsynth-fixture"
    renderer_binary.write_bytes(b"fixture-fluidsynth-2.3.4")
    renderer_library = artifact_dir / "libfluidsynth-fixture.so"
    renderer_library.write_bytes(b"fixture-libfluidsynth")
    soundfont = artifact_dir / "fixture.sf2"
    soundfont.write_bytes(b"fixture-soundfont")
    renderer_identity = tmp_path / "renderer_identity.json"
    write_canonical_json(
        renderer_identity,
        {
            "schema_version": TRIANGLE_RENDERER_SCHEMA_VERSION,
            "fluidsynth": {
                "binary_path": str(renderer_binary.resolve()),
                "binary_sha256": file_sha256(renderer_binary),
                "ld_library_path": str(artifact_dir.resolve()),
                "library_files": [
                    {
                        "path": str(renderer_library.resolve()),
                        "size": renderer_library.stat().st_size,
                        "sha256": file_sha256(renderer_library),
                    }
                ],
                "version": "FluidSynth runtime version 2.3.4",
            },
            "soundfont": {
                "path": str(soundfont.resolve()),
                "size": soundfont.stat().st_size,
                "sha256": file_sha256(soundfont),
            },
            "midi_program": 0,
            "midi_bank": 0,
            "render_bpm": TRIANGLE_RENDER_BPM,
            "sample_rate": TRIANGLE_RENDER_SAMPLE_RATE,
            "bit_depth": 16,
            "synth_gain": TRIANGLE_SYNTH_GAIN,
            "gain_policy": TRIANGLE_GAIN_POLICY,
            "true_peak_limit_dbtp": TRIANGLE_TRUE_PEAK_LIMIT_DBTP,
            "true_peak_implementation": TRIANGLE_TRUE_PEAK_IMPLEMENTATION,
            "command_template": [
                "{fluidsynth_binary}",
                "-ni",
                "{soundfont}",
                "{input_midi}",
                "-F",
                "{output_wav}",
                "-r",
                str(TRIANGLE_RENDER_SAMPLE_RATE),
                "-g",
                str(TRIANGLE_SYNTH_GAIN),
            ],
        },
    )
    attestation_dir = tmp_path / "attestation"
    dependency_dir = attestation_dir / "dependencies"
    dependency_dir.mkdir(parents=True)
    uv_lock = dependency_dir / "uv.lock"
    uv_lock.write_text("version = 1\n", encoding="utf-8")
    pyproject = dependency_dir / "pyproject.toml"
    pyproject.write_text("[project]\nname='fixture'\n", encoding="utf-8")
    write_canonical_json(
        attestation_dir / "code_identity.json",
        {
            "schema_version": CODE_IDENTITY_SCHEMA_VERSION,
            "repository_root": str(tmp_path.resolve()),
            "git_commit": "a" * 40,
            "git_clean": True,
            "git_status_porcelain": "",
        },
    )
    write_canonical_json(
        attestation_dir / "checkpoint_identity.json",
        {
            "schema_version": CHECKPOINT_IDENTITY_SCHEMA_VERSION,
            "path": str(checkpoint.resolve()),
            "sha256": file_sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
    )
    write_canonical_json(
        attestation_dir / "environment.json",
        {
            "schema_version": ENVIRONMENT_IDENTITY_SCHEMA_VERSION,
            "code_identity": "a" * 40,
            "dependency_files": {
                "uv_lock": {
                    "path": str(uv_lock.resolve()),
                    "sha256": file_sha256(uv_lock),
                },
                "pyproject": {
                    "path": str(pyproject.resolve()),
                    "sha256": file_sha256(pyproject),
                },
            },
            "python": {
                "implementation": "CPython",
                "version": "3.10.0 fixture",
                "version_info": [3, 10, 0],
                "executable": "/fixture/python",
            },
            "torch": {
                "version": "2.7.1+cu128",
                "cuda_version": "12.8",
                "cudnn_version": 90701,
                "cuda_available": True,
            },
            "cuda_visible_devices": "0",
            "gpus": [
                {
                    "visible_index": 0,
                    "name": "Fixture GPU",
                    "uuid": "GPU-fixture",
                    "total_memory_bytes": 1024,
                    "compute_capability": [9, 0],
                }
            ],
            "nvidia_smi": {
                "driver_version": "fixture-driver",
                "gpus": [
                    {
                        "physical_index": 0,
                        "uuid": "GPU-fixture",
                        "name": "Fixture GPU",
                        "memory_total_mib": 1,
                    }
                ],
            },
        },
    )
    write_canonical_json(
        attestation_dir / "qualification_spec.json",
        qualification_spec_contract(),
    )
    attestation = {
        "schema_version": ATTESTATION_BUNDLE_SCHEMA_VERSION,
        **{
            name: {
                "path": str((attestation_dir / filename).resolve()),
                "sha256": file_sha256(attestation_dir / filename),
            }
            for name, filename in {
                "code_identity": "code_identity.json",
                "checkpoint_identity": "checkpoint_identity.json",
                "environment": "environment.json",
                "qualification_spec": "qualification_spec.json",
            }.items()
        },
    }
    candidate = default_campaign_config(
        code_identity="a" * 40,
        checkpoint_path=str(checkpoint.resolve()),
        checkpoint_sha256=file_sha256(checkpoint),
        input_manifest_path=str(input_manifest.resolve()),
        input_manifest_sha256=file_sha256(input_manifest),
        attestation=attestation,
        playback_tempo=60,
        tail_beats=24,
    )
    candidate["qualification"] = {
        "dense_song": "2",
        "tail_songs": ["2", "4"],
        "sample_seed": int(SEEDS["sample"][0]),
        "perturb_seed": int(SEEDS["perturb"][0]),
        "tempo_candidates": [60, 30],
        "tail_candidates": [8, 16, 24],
        "decision_order": ["determinism", "static_input_gate", "tempo", "tail"],
    }
    qualification_candidate = tmp_path / "qualification_config.json"
    write_canonical_json(qualification_candidate, candidate)
    qualification_schedule = tmp_path / "qualification_manifest.jsonl"
    schedule_rows = build_qualification_schedule(
        json.loads(input_manifest.read_text(encoding="utf-8")),
        candidate,
    )
    from streammuse.experiments.melody_robustness import write_jsonl

    write_jsonl(qualification_schedule, schedule_rows)
    qualification_output_root = tmp_path / "fixture-qualification-runs"
    binding_path = qualification_output_root / "campaign_binding.json"
    binding = {
        "schema_version": "streammuse.melody_robustness.campaign_binding.v1",
        "qualification": True,
        "campaign_config_path": str(qualification_candidate.resolve()),
        "campaign_config_sha256": file_sha256(qualification_candidate),
        "run_schedule_path": str(qualification_schedule.resolve()),
        "run_schedule_sha256": file_sha256(qualification_schedule),
        "input_manifest_path": str(input_manifest.resolve()),
        "input_manifest_sha256": file_sha256(input_manifest),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "code_identity": "a" * 40,
    }
    write_canonical_json(binding_path, binding)
    bound_fields = {
        key: binding[key]
        for key in (
            "campaign_config_sha256",
            "run_schedule_sha256",
            "input_manifest_sha256",
            "checkpoint_sha256",
            "code_identity",
        )
    }
    bound_fields["campaign_binding_sha256"] = file_sha256(binding_path)
    run_evidence = []
    for row in schedule_rows:
        attempt = (
            qualification_output_root
            / "runs"
            / row["run_id"]
            / "attempt-001"
        )
        attempt.mkdir(parents=True)
        kind = row["qualification_kind"]
        if kind == "determinism_offline":
            write_canonical_json(
                attempt / f"{row['run_id']}_tokens.json",
                {"sampled_tokens": [11, 22, 33, 44]},
            )
            write_roll_midi(
                Roll(
                    end_tick=16,
                    sustain=frozenset((tick, 60) for tick in range(16)),
                    onsets=frozenset({(0, 60)}),
                ),
                attempt / f"{row['run_id']}_generated.mid",
                bpm=120,
            )
        if row["pipeline"] == "rt":
            write_canonical_json(
                attempt / "validity.json",
                {
                    "content": {
                        "request_tick_contract_valid": True,
                        "analysis_request_coverage": 1.0,
                    },
                    "requests": [
                        {
                            "generation_start_tick": tick,
                            "raw_token_digest": f"raw-{tick}",
                            "input_increment_digest": f"increment-{tick}",
                            "input_cumulative_digest": f"cumulative-{tick}",
                            "part0_roll_digest": f"roll-{tick}",
                            "part0_token_digest": f"tokens-{tick}",
                            "context_start_tick": 0,
                        }
                        for tick in (4, 8)
                    ],
                },
            )
        if kind == "determinism_rt" or kind.startswith("tail_"):
            write_roll_midi(
                Roll(
                    end_tick=16,
                    sustain=frozenset((tick, 60) for tick in range(16)),
                    onsets=frozenset({(0, 60)}),
                ),
                attempt / "theoretical_model.mid",
                bpm=120,
            )
        artifacts = [
            artifact
            for artifact in sorted(attempt.rglob("*"))
            if artifact.is_file()
        ]
        verdict = {
            "schema_version": "streammuse.melody_robustness.verdict.v1",
            "run_id": row["run_id"],
            "attempt_id": "attempt-001",
            "pipeline": row["pipeline"],
            "content_valid": True,
            "operational_valid": True,
            "validity": {},
            "artifact_index": [
                {
                    "path": str(artifact.relative_to(attempt)),
                    "size": artifact.stat().st_size,
                    "sha256": file_sha256(artifact),
                }
                for artifact in artifacts
            ],
            **bound_fields,
        }
        immutable = attempt / "verdict.json"
        write_canonical_json(immutable, verdict)
        write_canonical_json(attempt.parent / "latest_verdict.json", verdict)
        run_evidence.append(
            {
                "run_id": row["run_id"],
                "attempt_id": "attempt-001",
                "verdict": {
                    "path": str(immutable.resolve()),
                    "sha256": file_sha256(immutable),
                },
            }
        )
    static_summary = tmp_path / "conversion_summary.json"
    manifest_payload = json.loads(input_manifest.read_text(encoding="utf-8"))
    write_canonical_json(
        static_summary,
        {
            "schema_version": "streammuse.midi_to_npz_summary.v1",
            "status": "ok",
            "expected": 40,
            "converted": 40,
            "skipped": 0,
            "exact_stem_set": True,
            "ticks_per_beat": 4,
            "updated_manifest_sha256": file_sha256(input_manifest),
            "errors": [],
            "results": [
                {
                    "stem": entry["stem"],
                    "status": "converted",
                    "npz_sha256": entry["paths"]["npz"]["sha256"],
                    "roll_gate": {
                        "differing_cells": 0,
                        "horizon_ticks": entry["validation_horizon_ticks"],
                    },
                }
                for entry in manifest_payload["entries"]
            ],
        },
    )
    qualification_result = tmp_path / "qualification_result.json"
    result = {
        "schema_version": "streammuse.melody_robustness.qualification.v1",
        "development_only": False,
        "passed": True,
        "candidate_config": {
            "path": str(qualification_candidate.resolve()),
            "sha256": file_sha256(qualification_candidate),
        },
        "candidate_config_sha256": file_sha256(qualification_candidate),
        "qualification_schedule": {
            "path": str(qualification_schedule.resolve()),
            "sha256": file_sha256(qualification_schedule),
        },
        "qualification_campaign_binding": {
            "path": str(binding_path.resolve()),
            "sha256": file_sha256(binding_path),
        },
        "run_evidence": run_evidence,
        "offline_deterministic": True,
        "rt_deterministic": True,
        "static_input_gate": {
            "valid": True,
            "errors": [],
            "summary_path": str(static_summary.resolve()),
            "sha256": file_sha256(static_summary),
        },
        "tempo": {"checks": {"30": True, "60": True}, "selected": 60},
        "tail": {
            "checks": {
                song: {
                    "content_valid": True,
                    "trace_and_coverage_valid": True,
                    "8_eq_16": True,
                    "16_eq_24": True,
                    "decision": 8,
                    "reason": "8_16_24_converged",
                }
                for song in candidate["qualification"]["tail_songs"]
            },
            "selected": 8,
        },
        "rule": "8=16=24->8; 16=24->16; otherwise stop_and_investigate",
        "failure_reasons": [],
    }
    write_canonical_json(qualification_result, result)
    frozen = copy.deepcopy(candidate)
    frozen["status"] = "qualified_frozen"
    frozen["runtime"]["playback_tempo"] = 60
    frozen["runtime"]["tail_beats"] = 8
    frozen["qualification_candidate"] = {
        "path": str(qualification_candidate.resolve()),
        "sha256": file_sha256(qualification_candidate),
    }
    frozen["qualification_result"] = {
        "path": str(qualification_result.resolve()),
        "sha256": file_sha256(qualification_result),
    }
    frozen["listening"]["selection_manifest_path"] = str(
        listening_manifest.resolve()
    )
    frozen["listening"]["selection_manifest_sha256"] = file_sha256(
        listening_manifest
    )
    frozen["listening"]["renderer_identity"] = {
        "path": str(renderer_identity.resolve()),
        "sha256": file_sha256(renderer_identity),
    }
    campaign_config = tmp_path / "campaign_config.json"
    write_canonical_json(campaign_config, frozen)
    return RobustnessFixture(
        root=tmp_path,
        checkpoint=checkpoint,
        attestation_dir=attestation_dir,
        input_manifest=input_manifest,
        listening_manifest=listening_manifest,
        renderer_identity=renderer_identity,
        campaign_config=campaign_config,
        qualification_candidate=qualification_candidate,
        qualification_result=qualification_result,
        qualification_schedule=qualification_schedule,
        qualification_output_root=qualification_output_root,
        melody_midi=melody,
        accompaniment_midi=accompaniment,
        condition_npz=npz,
    )
