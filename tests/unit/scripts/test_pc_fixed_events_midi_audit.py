import copy
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from pc_fixed_events_midi_audit import FixedPromptInput, compare


def test_fixed_adapter_preserves_current_tick_order_and_both_buffers():
    calls = []
    service = SimpleNamespace(_input_window_lock=threading.Lock(),
        _input_window_events=[], _event_q=queue.Queue(),
        _drain_user_events=lambda: calls.append("drained"))
    rows = [dict(tick=0, pitch=60, event_type="note_on", velocity=75, channel=0),
            dict(tick=2, pitch=60, event_type="note_off", velocity=0, channel=0),
            dict(tick=2, pitch=60, event_type="note_on", velocity=83, channel=0)]
    adapter = FixedPromptInput(service, rows)
    adapter.drain()
    assert len(service._input_window_events) == service._event_q.qsize() == 1
    adapter.drain()
    assert len(service._input_window_events) == 1
    adapter.drain()
    assert len(service._input_window_events) == service._event_q.qsize() == 3
    assert [e.velocity for e in service._input_window_events] == [75, 0, 83]
    assert [e.event_type.value for e in service._input_window_events] == ["note_on", "note_off", "note_on"]
    assert adapter.delivered == [{**r, "injected_at_logical_tick": r["tick"]} for r in rows]
    assert len(calls) == 3


def evidence():
    return {"calls": {32: {"trace": {"prompt_tokens": [1, 2],
        "raw_tokens": [3], "structural_tokens": []}, "raw_output": [{"tick": 32}]}},
        "trace_complete": True,
        "prompt": {"prompt_tokens": [8], "generated_tokens": [8, 9], "new_tokens": [9]},
        "prompt_history": [{"tick": 0}], "raw_history": [{"tick": 32, "type": "note_on"}],
        "playback": [(60, 32, 35, 64)]}


def test_equal_raw_is_not_a_playback_claim():
    a = evidence()
    b = copy.deepcopy(a)
    b["playback"] = []
    result = compare(a, b)
    assert result["primary_raw_exact"] and not result["playback_exact"]


def test_equal_output_does_not_imply_equal_input():
    a = evidence()
    b = copy.deepcopy(a)
    b["calls"][32]["trace"]["prompt_tokens"] = [1, 4]
    result = compare(a, b)
    assert result["primary_raw_exact"] and not result["model_inputs_exact"]
    assert result["mismatched_generation_ticks"]["prompt_tokens"] == [32]


@pytest.mark.parametrize("field", ["raw_tokens", "structural_tokens"])
def test_raw_token_differences_fail_primary(field):
    a = evidence()
    b = copy.deepcopy(a)
    b["calls"][32]["trace"][field] = [99]
    assert not compare(a, b)["primary_raw_exact"]


@pytest.mark.parametrize("kind", ["missing_call", "incomplete_trace", "raw_history", "prompt_history", "raw_output"])
def test_missing_or_different_raw_evidence_cannot_pass(kind):
    a = evidence()
    b = copy.deepcopy(a)
    if kind == "missing_call":
        b["calls"] = {}
    elif kind == "incomplete_trace":
        b["trace_complete"] = False
    elif kind == "raw_output":
        b["calls"][32]["raw_output"] = []
    else:
        b[kind] = []
    assert not compare(a, b)["primary_raw_exact"]
