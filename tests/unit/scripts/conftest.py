from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

import pytest

from streammuse.experiments.melody_robustness import (
    CONDITION_TABLE,
    SEEDS,
    build_listening_selection_manifest,
    file_sha256,
    write_canonical_json,
)


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RobustnessFixture:
    root: Path
    checkpoint: Path
    input_manifest: Path
    listening_manifest: Path
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
        song = f"song-{song_index}"
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
        build_listening_selection_manifest(
            json.loads(input_manifest.read_text(encoding="utf-8")),
            manifest_path=input_manifest,
            perturb_seed=int(SEEDS["perturb"][0]),
            sample_seed=int(SEEDS["sample"][0]),
            blind_order_seed=int(SEEDS["blind_order"]),
            excerpt_starts={f"song-{index}": 0 for index in range(1, 6)},
        ),
    )
    return RobustnessFixture(
        root=tmp_path,
        checkpoint=checkpoint,
        input_manifest=input_manifest,
        listening_manifest=listening_manifest,
        melody_midi=melody,
        accompaniment_midi=accompaniment,
        condition_npz=npz,
    )
