from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_fixed_events_midi_file import compare_runs


def sample_run():
    event = {"tick": 1, "event_type": "note_on", "pitch": 65, "velocity": 80, "channel": 0}
    call = {"request": {"melody_events": [event]},
            "trace": {"part0_roll_sha256": "abc", "part0_tokens": [1], "prompt_tokens": [2],
                      "raw_tokens": [3], "structural_tokens": []},
            "raw_output": [{"type": "note_on", "pitch": 48, "tick": 4}]}
    return {"calls": {4: call}, "samples": [3], "input_events": [event],
            "playback": [(48, 4, 5, 80)]}


def test_missing_call_cannot_be_an_exact_match():
    a = sample_run()
    b = deepcopy(a)
    b["calls"][8] = deepcopy(b["calls"][4])
    result = compare_runs(a, b)
    assert result["only_right_generation_ticks"] == [8]
    assert not result["model_inputs_exact"]
    assert not result["raw_output_exact"]
    assert not result["strict_exact"]


def test_velocity_difference_is_not_hidden_by_timing_diagnostic():
    a = sample_run()
    b = deepcopy(a)
    b["input_events"][0]["velocity"] = 64
    result = compare_runs(a, b)
    assert not result["delivered_events_exact"]
    assert result["delivered_timing_pitch_channel_order_exact"]


def test_equal_raw_output_does_not_imply_equal_model_input():
    a = sample_run()
    b = deepcopy(a)
    b["calls"][4]["trace"]["prompt_tokens"] = [99]
    result = compare_runs(a, b)
    assert result["raw_output_exact"]
    assert not result["model_inputs_exact"]
    assert result["first_difference_examples"]["prompt_tokens"]["generation_start_tick"] == 4
