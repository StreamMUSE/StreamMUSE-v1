from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "run_rule_s_if_else_prompt_selection.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_rule_s_if_else_prompt_selection_under_test",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_rule_s_if_else_runner_defaults_to_n10_and_deployment_sampling():
    args = runner.parse_args(
        [
            "--npz-file",
            "input.npz",
            "--prompt-checkpoint",
            "prompt.safetensors",
        ]
    )

    assert args.candidate_count == 10
    assert args.temperature == 1.1
    assert args.top_p == 0.95
    assert args.top_k == 50
    assert args.repetition_penalty == 1.0


def test_rule_s_if_else_ranking_rows_expose_complete_stage_one_status():
    candidate = {
        "selection_status": "UNSELECTED",
        "candidate_number": 1,
        "complete_prompt": False,
        "stage_1_note_count_pass": True,
        "stage_1_pass": False,
        "stage_2_note_count_cap_pass": True,
        "stage_2_in_key_pass": False,
        "stage_2_tonal_fallback_top5": False,
        "stage_2_tonal_top_up": False,
        "stage_2_pool_member": False,
        "pitch_change_rank": None,
        "stage_3_pitch_change_rank_1_to_4": False,
        "entropy_rank": None,
        "entropy_top3": False,
        "entropy_rank_points": 0,
        "note_count_rank": None,
        "note_count_top3": False,
        "note_count_rank_points": 0,
        "combined_rank_score": None,
        "final_rank": None,
        "prompt_ppl": None,
        "acc_note_count": 8,
        "out_of_key_pitch_classes": [],
        "out_of_key_note_count": 0,
        "in_key_note_ratio": 1.0,
        "acc_pitch_class_note_entropy": 0.0,
        "acc_pitch_change_score": 0,
        "midi_file": "unselected/candidate_01_UNSELECTED.mid",
    }

    rows = runner.ranking_rows([candidate])

    assert rows[0]["complete_prompt"] is False
    assert rows[0]["stage_1_note_count_pass"] is True
    assert rows[0]["stage_1_pass"] is False
