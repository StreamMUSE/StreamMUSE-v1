"""Tests for the terminal-specific immutable rap view state."""

from __future__ import annotations

from streammuse.domain.rap import RapEvent, RapEventType
from streammuse.presentation.rap_demo.terminal_state import TerminalRapStateProjector


def _event(sequence: int, event_type: RapEventType, payload: dict, *, bar: int | None = 1, tick: int | None = 16) -> RapEvent:
    return RapEvent("session", sequence, event_type, "2026-08-08T00:00:00+00:00", sequence, bar, tick, "request-1", payload)


def test_projector_tracks_structured_request_candidates_and_bounded_history() -> None:
    flow = {"template_id": "baseline_syncopated_9", "slots": [{"slot_index": 0, "tick_in_bar": 0}]}
    scheduled = [{"slot_index": 0, "tick_in_bar": 0, "label": "Galaxies", "word": "Galaxies", "stress": 1, "stressed": True, "target_stress": 1.0}]
    events = (
        _event(1, RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0}),
        _event(2, RapEventType.BAR_RESERVED, {"topic": "space", "source": "prevalidated_fallback", "text": "Fallback", "flow": flow}),
        _event(3, RapEventType.BAR_PLANNING_STARTED, {"topic": "space", "flow": flow, "context_lines": [], "seed": 20260808}),
        _event(4, RapEventType.CANDIDATE_BATCH_RECEIVED, {"prompt": [{"role": "system", "content": "exact"}], "source": "local_chat"}),
        _event(5, RapEventType.CANDIDATE_EVALUATED, {"candidate_id": "bad", "text": "bad", "rejection_reasons": ["syllable_count:10!=9"], "selected": False}),
        _event(6, RapEventType.BAR_REPLACED, {"source": "local_chat", "text": "Galaxies dance in a cosmic fight", "scheduled_syllables": scheduled, "flow": flow}),
        _event(7, RapEventType.BAR_FROZEN, {"source": "local_chat", "text": "Galaxies dance in a cosmic fight", "scheduled_syllables": scheduled, "flow": flow}),
        _event(8, RapEventType.TICK, {"beat": 0, "tick_in_beat": 0}),
        _event(9, RapEventType.SYLLABLE_EMITTED, {"label": "Galaxies"}),
    )
    projector = TerminalRapStateProjector(history_limit=5)
    for event in events:
        projector.apply(event)

    state = projector.state
    assert state.current_bar == 1
    assert state.bars[1].text == "Galaxies dance in a cosmic fight"
    assert state.bars[1].scheduled_syllables[0]["tick_in_bar"] == 0
    assert state.latest_request.flow["template_id"] == "baseline_syncopated_9"
    assert state.latest_batch.prompt[0]["role"] == "system"
    assert state.candidates[0].rejection_reasons == ("syllable_count:10!=9",)
    assert len(state.recent_events) == 5


def test_projector_copies_nested_event_payloads() -> None:
    payload = {"flow": {"template_id": "original", "slots": []}, "context_lines": []}
    projector = TerminalRapStateProjector()
    projector.apply(_event(1, RapEventType.BAR_PLANNING_STARTED, payload))
    payload["flow"]["template_id"] = "mutated"
    payload["context_lines"].append("mutated")

    state = projector.state
    assert state.latest_request.flow["template_id"] == "original"
    assert state.latest_request.context_lines == ()
