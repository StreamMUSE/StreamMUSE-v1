from __future__ import annotations

import copy

import mido
import pytest

from streammuse.experiments.robustness_metrics import Roll, write_roll_midi
from streammuse.experiments.triangle_listening import (
    KNOWN_DIFFERENT_RECIPE,
    build_triangle_control_roll,
)
from streammuse.experiments.triangle_midi import write_triangle_note_event_midi


def _synthetic() -> dict:
    return {
        "kind": "synthetic_control",
        "source_artifact": "known_different_scale_v1",
        "presentation": "acc_solo",
        "song": "1",
        "condition": "known_different_control",
        "perturb_seed": None,
        "sample_seed": 17,
        "recipe": {
            **KNOWN_DIFFERENT_RECIPE,
            "pitches": list(KNOWN_DIFFERENT_RECIPE["pitches"]),
        },
    }


def test_known_different_recipe_velocity_is_not_replaced(tmp_path):
    source = _synthetic()
    roll, velocity = build_triangle_control_roll(source)
    assert velocity == 96
    path = tmp_path / "known.mid"
    write_roll_midi(roll, path, velocity=velocity)
    velocities = {
        message.velocity
        for track in mido.MidiFile(path).tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    }
    assert velocities == {96}

    tampered = copy.deepcopy(source)
    tampered["recipe"]["velocity"] = 80
    with pytest.raises(ValueError, match="frozen recipe"):
        build_triangle_control_roll(tampered)


def test_analyzer_controls_bind_exact_selection_sources_and_recipe(load_script, tmp_path):
    analyzer = load_script("analyze_perturbation_robustness")
    formal = {
        "kind": "formal",
        "formal_pipeline": "rt",
        "source_artifact": "theoretical_model",
        "presentation": "acc_solo",
        "song": "1",
        "condition": "sham",
        "perturb_seed": None,
        "sample_seed": 17,
    }
    synthetic = _synthetic()
    trials = [
        {
            "semantic_id": f"K:{index}",
            "question_id": f"Q{index:03d}",
            "block": "known_different_control",
            "sources": {"a": formal, "b": synthetic},
            "excerpt": {"start_model_tick": 64, "end_model_tick": 128},
        }
        for index in range(1, 7)
    ]
    formal_roll = Roll(
        end_tick=128,
        sustain=frozenset((tick, 36) for tick in range(64, 72)),
        onsets=frozenset({(64, 36)}),
    )
    formal_path = tmp_path / "theoretical_model.mid"
    write_triangle_note_event_midi(
        [
            {
                "start_model_tick": 64,
                "end_model_tick": 72,
                "pitch": 36,
                "velocity": 73,
            }
        ],
        formal_path,
    )
    metrics = [
        {
            "run_id": "run-sham",
            "pipeline": "rt_theoretical",
            "song": "1",
            "condition": "sham",
            "perturb_seed": None,
            "sample_seed": 17,
            "paths": {"accompaniment": str(formal_path)},
            "hashes": {"accompaniment": "a" * 64},
        }
    ]
    result = analyzer._selection_known_different_controls(
        {"trials": trials},
        metrics,
        {("run-sham", "rt_theoretical"): {"acc": formal_roll}},
        tmp_path,
        config_sha="b" * 64,
        selection_sha="c" * 64,
    )
    assert result["expected_count"] == result["actual_count"] == 6
    assert result["all_recipe_bound"] is True
    assert result["all_source_selectors_bound"] is True
    assert result["all_not_identical"] is True
    assert {row["synthetic_velocity"] for row in result["controls"]} == {96}
