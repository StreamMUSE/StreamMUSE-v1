from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_script() -> ModuleType:
    module_name = "_streammuse_test_run_aligned_moss_warp_experiment"
    script_path = ROOT / "scripts" / "run_aligned_moss_warp_experiment.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_select_chunks_returns_distinct_clean_median_and_worst_cases() -> None:
    script = _load_script()
    candidates = tuple(
        script.ChunkCandidate(
            song_id="song",
            chunk_index=index,
            distortion_score=float(index + 1),
            baseline_wav_path=Path(f"chunk-{index:03d}.wav"),
            diagnostics_path=Path(f"chunk-{index:03d}.json"),
        )
        for index in range(10)
    )

    selected = script.select_chunks(candidates, clean_count=2, median_count=2, worst_count=2)

    assert [(item.category, item.candidate.chunk_index) for item in selected] == [
        ("clean", 0),
        ("clean", 1),
        ("median", 4),
        ("median", 5),
        ("worst", 8),
        ("worst", 9),
    ]


def test_boundary_jump_metrics_measure_target_map_discontinuities() -> None:
    script = _load_script()
    samples = np.zeros(12, dtype=np.float32)
    samples[4] = 0.5
    samples[5] = -0.5
    samples[8] = 0.25
    samples[9] = -0.25

    metrics = script.boundary_jump_metrics(samples, (4, 8))

    assert metrics["boundary_count"] == 2
    assert metrics["mean_absolute_jump"] == pytest.approx(0.75)
    assert metrics["max_absolute_jump"] == pytest.approx(1.0)
