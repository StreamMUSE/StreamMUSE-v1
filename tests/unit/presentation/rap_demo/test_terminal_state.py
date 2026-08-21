"""Tests for the terminal-specific immutable rap view state."""

from __future__ import annotations

from streammuse.domain.rap import RapEvent, RapEventType
from streammuse.presentation.rap_demo.terminal_state import TerminalRapStateProjector


def _event(
    sequence: int,
    event_type: RapEventType,
    payload: dict,
    *,
    bar: int | None = 1,
    tick: int | None = 16,
    request_id: str | None = "request-1",
) -> RapEvent:
    return RapEvent("session", sequence, event_type, "2026-08-08T00:00:00+00:00", sequence, bar, tick, request_id, payload)


def test_projector_tracks_structured_request_candidates_and_bounded_history() -> None:
    flow = {"template_id": "baseline_syncopated_9", "slots": [{"slot_index": 0, "tick_in_bar": 0}]}
    scheduled = [{"slot_index": 0, "tick_in_bar": 0, "label": "Galaxies", "word": "Galaxies", "stress": 1, "stressed": True, "target_stress": 1.0}]
    events = (
        _event(1, RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0}),
        _event(2, RapEventType.BAR_RESERVED, {"topic": "space", "source": "prevalidated_fallback", "text": "Fallback", "flow": flow}),
        _event(3, RapEventType.BAR_PLANNING_STARTED, {"topic": "space", "flow": flow, "context_lines": [], "seed": 20260808}),
        _event(4, RapEventType.CANDIDATE_BATCH_RECEIVED, {"prompt": [{"role": "system", "content": "exact"}], "source": "local_chat"}),
        _event(5, RapEventType.CANDIDATE_EVALUATED, {"candidate_id": "bad", "text": "bad", "rejection_reasons": ["syllable_count:10!=9"], "selected": False}),
        _event(6, RapEventType.BAR_REPLACED, {"source": "local_chat", "text": "Galaxies dance in a cosmic fight", "scheduled_syllables": scheduled, "flow": flow, "total_score": 0.84}),
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
    assert state.bars[1].total_score == 0.84
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


def test_projector_ignores_stale_request_results_but_preserves_them_in_history() -> None:
    projector = TerminalRapStateProjector()
    projector.apply(_event(1, RapEventType.BAR_RESERVED, {"text": "fallback", "fallback": True}, request_id=None))
    projector.apply(_event(2, RapEventType.BAR_PLANNING_STARTED, {"flow": {"template_id": "r1"}}, request_id="r1"))
    projector.apply(_event(3, RapEventType.BAR_PLANNING_STARTED, {"flow": {"template_id": "r2"}}, request_id="r2"))
    projector.apply(_event(4, RapEventType.CANDIDATE_BATCH_RECEIVED, {"prompt": [{"role": "system"}]}, request_id="r1"))
    projector.apply(_event(5, RapEventType.CANDIDATE_EVALUATED, {"candidate_id": "stale"}, request_id="r1"))
    projector.apply(_event(6, RapEventType.BAR_REPLACED, {"text": "stale", "fallback": False}, request_id="r1"))

    state = projector.state
    assert state.latest_request.request_id == "r2"
    assert state.latest_batch is None
    assert state.candidates == ()
    assert state.bars[1].text == "fallback"
    assert [event.request_id for event in state.recent_events[-3:]] == ["r1", "r1", "r1"]


def test_projector_never_replaces_or_reserves_a_frozen_bar_and_clears_fallback() -> None:
    projector = TerminalRapStateProjector()
    projector.apply(_event(1, RapEventType.BAR_RESERVED, {"text": "fallback", "fallback": True, "fallback_reason": "generation_pending"}, request_id=None))
    projector.apply(_event(2, RapEventType.BAR_PLANNING_STARTED, {}, request_id="r1"))
    projector.apply(_event(3, RapEventType.BAR_REPLACED, {"text": "selected", "fallback": False, "fallback_reason": None}, request_id="r1"))
    selected = projector.state.bars[1]
    assert selected.fallback is False
    assert selected.fallback_reason is None
    projector.apply(_event(4, RapEventType.BAR_FROZEN, {"text": "selected", "fallback": False}, request_id="r1"))
    projector.apply(_event(5, RapEventType.BAR_RESERVED, {"text": "late reserve", "fallback": True}, request_id=None))
    projector.apply(_event(6, RapEventType.BAR_REPLACED, {"text": "late replacement", "fallback": False}, request_id="r1"))

    frozen = projector.state.bars[1]
    assert frozen.frozen is True
    assert frozen.text == "selected"
    assert frozen.fallback is False


def test_projector_ignores_malformed_structured_values_without_losing_raw_trace() -> None:
    projector = TerminalRapStateProjector()
    projector.apply(
        _event(
            1,
            RapEventType.BAR_RESERVED,
            {"flow": ["not-a-mapping"], "scheduled_syllables": None},
            request_id=None,
        )
    )
    projector.apply(_event(2, RapEventType.BAR_PLANNING_STARTED, {"flow": ["not-a-mapping"]}, request_id="r1"))
    projector.apply(
        _event(
            3,
            RapEventType.CANDIDATE_BATCH_RECEIVED,
            {"prompt": [{"role": "system"}, "not-a-mapping"]},
            request_id="r1",
        )
    )

    state = projector.state
    assert state.bars[1].flow is None
    assert state.bars[1].scheduled_syllables == ()
    assert state.latest_request.flow is None
    assert state.latest_batch.prompt == ({"role": "system"},)
    assert state.recent_events[-1].payload["prompt"][1] == "not-a-mapping"


def test_audio_render_pipeline_does_not_replace_terminal_playback_lifecycle_state() -> None:
    projector = TerminalRapStateProjector()
    for event in (
        _event(1, RapEventType.SESSION_STARTED, {"playback_state": "running"}, bar=None),
        _event(2, RapEventType.BAR_PLAYBACK_STARTED, {"absolute_frame": 192_000}, bar=1),
        _event(3, RapEventType.AUDIO_RENDER_COMPLETED, {"render_latency_ms": 12.0}, bar=4),
        _event(4, RapEventType.BAR_AUDIO_READY, {}, bar=4),
        _event(5, RapEventType.BAR_AUDIO_COMMITTED, {"deadline_slack_ms": 40.0}, bar=3),
    ):
        projector.apply(event)

    assert projector.state.audio["state"] == "running"
    assert projector.state.audio["current_bar"] == 1
    assert projector.state.audio["render_state"] == "committed"
    assert projector.state.audio["render_bar"] == 3


def test_projector_exposes_only_the_latest_bounded_remote_chunk_diagnostics() -> None:
    payload = {
        "state": "committed",
        "renderer_decision": "moss_aligned_remote",
        "chunk_index": 2,
        "bars": [4, 5],
        "selected_lines": ["First remote line", "Second remote line"],
        "flows": [
            {
                "template_id": "flow-a",
                "selected_syllable_schedule": "t0:first/stress1",
                "slots": [{"tick_in_bar": 0, "target_stress": 1.0}],
            },
            {"template_id": "flow-b", "slots": [{"tick_in_bar": 2, "target_stress": 0.5}]},
        ],
        "candidate_counts": {"requested": 32, "parseable": 30, "valid": 8, "selectable": 4},
        "selected_scores": [
            {"bar": 4, "total": 0.91, "component_scores": {"stress_alignment": 0.88}},
            {"bar": 5, "total": 0.87, "component_scores": {"continuity": 0.82}},
        ],
        "prompt_summary": "system: clean rap | user: both exact schedules",
        "context_lines": ["Prior committed line"],
        "stage_timings_ms": {"generation": 1_000.0, "mac": 6.0, "total": 3_276.0},
        "deadline_slack_ms": 1_700.0,
        "alignment": {
            "method": "mms_forced_alignment",
            "confidence": 0.94,
            "fallback_counts": {"transcript_proportional": 1},
        },
        "stretch_warnings": ["stretch_ratio_high:1.22"],
        "hashes": {"vocal_sha256": "vocal-hash"},
        "artifact_refs": {"manifest": "/h200/request-2/manifest.json"},
        "raw_wav": "RIFF-forbidden",
    }
    projector = TerminalRapStateProjector()

    projector.apply(
        _event(
            1,
            RapEventType.CHUNK_COMMITTED,
            payload,
            bar=4,
            request_id="request-2",
        )
    )

    remote = projector.state.remote_chunk
    assert remote is not None
    assert remote["event_type"] == "chunk_committed"
    assert remote["request_id"] == "request-2"
    assert remote["selected_lines"] == ("First remote line", "Second remote line")
    assert remote["flows"][0]["slot_stress_schedule"] == "t0@1.00"
    assert remote["flows"][0]["selected_syllable_schedule"] == "t0:first/stress1"
    assert remote["candidate_counts"]["selectable"] == 4
    assert remote["stage_timings_ms"]["mac"] == 6.0
    assert "RIFF-forbidden" not in repr(remote)
    assert "RIFF-forbidden" not in repr(projector.state.recent_events[-1].payload)
