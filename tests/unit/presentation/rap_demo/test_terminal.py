"""Tests for the terminal-only rap monitor."""

from streammuse.domain.rap import RapEvent, RapEventType
from streammuse.presentation.rap_demo.terminal import TerminalRapSink


def _event(event_type: RapEventType, payload: dict, *, bar: int = 1, tick: int = 16) -> RapEvent:
    return RapEvent(
        session_id="session",
        sequence=1,
        event_type=event_type,
        utc_time="2026-08-07T00:00:00+00:00",
        monotonic_ns=1,
        bar=bar,
        tick=tick,
        request_id="request-1",
        payload=payload,
    )


def test_terminal_full_prints_candidate_validity_rejections_components_and_selection(capsys) -> None:
    sink = TerminalRapSink(detail="full")
    sink(
        _event(
            RapEventType.CANDIDATE_EVALUATED,
            {
                "candidate_id": "candidate-1",
                "text": "space",
                "valid": False,
                "rejection_reasons": ["syllable_count:1!=9"],
                "components": [
                    {"name": "stress_alignment", "value": 0.8, "weight": 0.3, "contribution": 0.24}
                ],
                "total_score": None,
                "selected": False,
            },
        )
    )
    output = capsys.readouterr().out

    assert "valid=False" in output
    assert "rejection_reasons" in output
    assert "stress_alignment" in output
    assert "selected=False" in output


def test_terminal_prints_planning_errors_freeze_tick_and_syllable_progress(capsys) -> None:
    sink = TerminalRapSink(detail="full")
    sink(_event(RapEventType.BAR_RESERVED, {"source": "prevalidated_fallback", "text": "beat"}))
    sink(_event(RapEventType.BAR_PLANNING_STARTED, {"topic": "space", "template_id": "one"}))
    sink(_event(RapEventType.GENERATION_FAILED, {"error_type": "generation_error", "error_message": "down"}))
    sink(_event(RapEventType.BAR_FROZEN, {"source": "prevalidated_fallback", "fallback": True, "fallback_reason": "generation_error"}))
    sink(_event(RapEventType.TICK, {"beat": 0, "tick_in_beat": 0}))
    sink(_event(RapEventType.SYLLABLE_EMITTED, {"label": "beat", "jitter_ms": 0.2}))
    output = capsys.readouterr().out

    for text in ("RESERVE", "PLAN", "ERROR", "FREEZE", "TICK", "SYLLABLE", "generation_error"):
        assert text in output
