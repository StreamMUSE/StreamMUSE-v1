from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "run_lekai_prompt_continuation_offline.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_lekai_prompt_continuation_offline_under_test",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
offline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(offline)

_RUNNER_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "run_rule_s_offline_streammuse_experiment.py"
)
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_rule_s_offline_streammuse_experiment_under_test",
    _RUNNER_PATH,
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(runner)


def _parse_prompt_mode(mode: str, *extra: str):
    return offline.parse_args(
        [
            "--midi-file",
            "input.mid",
            "--output-dir",
            "output",
            "--prompt-selection-mode",
            mode,
            *extra,
        ]
    )


def test_offline_runner_uses_rule_s_if_else_defaults_when_omitted():
    args = _parse_prompt_mode("rule_s_if_else")

    assert args.prompt_selection_mode == "rule_s_if_else"
    assert args.prompt_batch_candidates == 10
    assert args.prompt_temperature == 1.1
    assert args.prompt_top_p == 0.95
    assert args.prompt_top_k == 50
    assert args.prompt_repetition_penalty == 1.0


@pytest.mark.parametrize("mode", ["single", "batch_first", "rule_s", "rule_s_v3"])
def test_offline_runner_preserves_existing_mode_defaults(mode):
    args = _parse_prompt_mode(mode)

    assert args.prompt_batch_candidates == 5
    assert args.prompt_temperature == 1.1
    assert args.prompt_top_p == 0.95
    assert args.prompt_top_k == 0
    assert args.prompt_repetition_penalty == 1.0


def test_offline_runner_explicit_values_override_rule_s_if_else_defaults():
    args = _parse_prompt_mode(
        "rule_s_if_else",
        "--prompt-batch-candidates",
        "7",
        "--prompt-temperature",
        "0.7",
        "--prompt-top-p",
        "0.8",
        "--prompt-top-k",
        "11",
        "--prompt-repetition-penalty",
        "1.2",
    )

    assert args.prompt_batch_candidates == 7
    assert args.prompt_temperature == 0.7
    assert args.prompt_top_p == 0.8
    assert args.prompt_top_k == 11
    assert args.prompt_repetition_penalty == 1.2


def test_trim_leading_rest_applies_max_tick_after_shift(tmp_path, monkeypatch):
    notes = [
        {"pitch": 60, "tick": 76, "duration": 2},
        {"pitch": 62, "tick": 127, "duration": 2},
        {"pitch": 64, "tick": 203, "duration": 2},
        {"pitch": 65, "tick": 204, "duration": 2},
    ]

    def _parse(*args, **kwargs):
        assert kwargs["max_tick"] is None
        return notes, 4, 206

    monkeypatch.setattr(offline.MidiFileInput, "_midi_to_notes", _parse)

    events, info = offline.load_midi_events(
        tmp_path / "melody.mid",
        ticks_per_beat=4,
        max_tick=128,
        trim_leading_rest=True,
    )

    note_ons = [event for event in events if event["type"] == "note_on"]
    assert [(event["pitch"], event["tick"]) for event in note_ons] == [
        (60, 0),
        (62, 51),
        (64, 127),
    ]
    assert all(int(event["tick"]) < 128 for event in events)
    assert not any(
        event["type"] == "note_off"
        and event["pitch"] == 64
        for event in events
    )
    assert info["first_note_tick_original"] == 76
    assert info["offset_ticks"] == 76
    assert info["original_max_tick"] == 206
    assert info["actual_max_tick"] == 128


def test_parse_execution_paths_accepts_explicit_subsets():
    assert runner.parse_execution_paths("offline") == ["offline"]
    assert runner.parse_execution_paths("streammuse") == ["streammuse"]
    assert runner.parse_execution_paths("offline,streammuse") == [
        "offline",
        "streammuse",
    ]


@pytest.mark.parametrize("raw", ["", "offline,offline", "offline,unknown"])
def test_parse_execution_paths_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        runner.parse_execution_paths(raw)


def test_full_length_max_tick_is_trimmed_and_rounded_to_beat(monkeypatch):
    notes = [{"pitch": 60, "tick": 76, "duration": 2}]

    def _parse(*args, **kwargs):
        return notes, 4, 1094

    monkeypatch.setattr(runner.MidiFileInput, "_midi_to_notes", _parse)

    assert runner.midi_max_tick(
        Path("melody.mid"),
        ticks_per_beat=4,
        tail_beats=0,
        max_eval_beats=0,
        trim_leading_rest=True,
    ) == 1020
    assert runner.midi_max_tick(
        Path("melody.mid"),
        ticks_per_beat=4,
        tail_beats=0,
        max_eval_beats=32,
        trim_leading_rest=True,
    ) == 128
