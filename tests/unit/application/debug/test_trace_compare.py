from __future__ import annotations

from streammuse.application.debug.compare import compare_trace_events
from streammuse.domain.debug.trace import DebugTraceEvent


def test_compare_reports_first_mismatch_stage() -> None:
    left = [
        DebugTraceEvent(
            run_id="l",
            runner_kind="offline_direct",
            scenario="lekai-prompt-continuation",
            stage="input_midi_loaded",
            output_hash="same",
        ),
        DebugTraceEvent(
            run_id="l",
            runner_kind="offline_direct",
            scenario="lekai-prompt-continuation",
            stage="prompt_tokenization",
            output_hash="left",
        ),
    ]
    right = [
        DebugTraceEvent(
            run_id="r",
            runner_kind="realtime_sim",
            scenario="lekai-prompt-continuation",
            stage="input_midi_loaded",
            output_hash="same",
        ),
        DebugTraceEvent(
            run_id="r",
            runner_kind="realtime_sim",
            scenario="lekai-prompt-continuation",
            stage="prompt_tokenization",
            output_hash="right",
        ),
    ]

    comparison = compare_trace_events(left, right)

    assert comparison["first_mismatch_stage"] == "prompt_tokenization"
    assert comparison["matching_stage_count"] == 1


def test_compare_includes_structured_diff_summary() -> None:
    left = [
        DebugTraceEvent(
            run_id="l",
            runner_kind="offline_direct",
            scenario="lekai-prompt-continuation",
            stage="prompt_decode_events",
            output_hash="left",
            summary={"event_count": 4, "first_mismatch": {"position": 2}},
        ),
    ]
    right = [
        DebugTraceEvent(
            run_id="r",
            runner_kind="realtime_sim",
            scenario="lekai-prompt-continuation",
            stage="prompt_decode_events",
            output_hash="right",
            summary={"event_count": 5, "first_mismatch": {"position": 2}},
        ),
    ]

    comparison = compare_trace_events(left, right)

    diff = comparison["stages"][0]["diff_summary"]
    assert diff["status"] == "different"
    assert diff["changed_fields"] == [{"field": "event_count", "left": 4, "right": 5}]
    assert diff["message"] == "event_count changed: offline=4, realtime=5"
