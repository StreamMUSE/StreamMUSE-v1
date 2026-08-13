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
