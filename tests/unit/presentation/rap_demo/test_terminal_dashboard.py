"""Tests for the pure Rich realtime rap dashboard."""

from __future__ import annotations

import pytest
from rich.console import Console

from streammuse.domain.rap import RapEvent, RapEventType
from streammuse.presentation.rap_demo.terminal_dashboard import build_dashboard
from streammuse.presentation.rap_demo.terminal_state import TerminalRapStateProjector, TerminalRapViewState


FLOW = {
    "template_id": "baseline_syncopated_9",
    "name": "Syncopated nine-slot baseline",
    "ticks_per_beat": 4,
    "beats_per_bar": 4,
    "provenance": {"kind": "hand_authored_mcflow_inspired", "source": "StreamMUSE baseline"},
    "slots": [
        {
            "slot_index": index,
            "tick_in_bar": tick,
            "duration_ticks": duration,
            "target_stress": stress,
            "boundary_strength": 3 if index == 8 else 0,
            "rhyme_group": "A" if index == 8 else None,
        }
        for index, (tick, duration, stress) in enumerate(
            zip(
                (0, 2, 3, 5, 7, 8, 10, 13, 15),
                (2, 1, 2, 2, 1, 2, 3, 2, 1),
                (1.0, 0.2, 0.7, 0.2, 0.6, 1.0, 0.2, 0.7, 0.9),
            )
        )
    ],
}
SCHEDULED = [
    {
        "slot_index": index,
        "tick_in_bar": tick,
        "target_stress": stress,
        "label": label,
        "word": word,
        "stress": 1 if stress >= 0.5 else 0,
        "stressed": stress >= 0.5,
    }
    for index, (tick, stress, label, word) in enumerate(
        zip(
            (0, 2, 3, 5, 7, 8, 10, 13, 15),
            (1.0, 0.2, 0.7, 0.2, 0.6, 1.0, 0.2, 0.7, 0.9),
            ("Gal", "ax", "ies", "dance", "in", "a", "cos", "mic", "fight"),
            ("Galaxies", "Galaxies", "Galaxies", "dance", "in", "a", "cosmic", "cosmic", "fight"),
        )
    )
]


def _event(
    sequence: int,
    event_type: RapEventType,
    payload: dict,
    *,
    bar: int | None = 1,
    tick: int | None = 16,
    request_id: str | None = "request-1",
) -> RapEvent:
    return RapEvent(
        "session",
        sequence,
        event_type,
        "2026-08-08T00:00:00+00:00",
        sequence,
        bar,
        tick,
        request_id,
        payload,
    )


@pytest.fixture
def projected_state() -> TerminalRapViewState:
    projector = TerminalRapStateProjector()
    events = (
        _event(
            1,
            RapEventType.SESSION_STARTED,
            {"tempo_bpm": 92.0, "max_bars": 3, "ticks_per_beat": 4, "beats_per_bar": 4},
            bar=None,
            tick=None,
            request_id=None,
        ),
        _event(
            2,
            RapEventType.BAR_RESERVED,
            {
                "topic": "space",
                "source": "prevalidated_fallback",
                "text": "Space dreams rise while bright stars cross dark night",
                "flow": FLOW,
                "fallback": True,
                "fallback_reason": "generation_pending",
            },
            request_id=None,
        ),
        _event(
            3,
            RapEventType.BAR_PLANNING_STARTED,
            {
                "topic": "space",
                "required_syllables": 9,
                "candidate_count": 3,
                "context_lines": ["Stars keep time in the night"],
                "seed": 20260808,
                "flow": FLOW,
            },
        ),
        _event(
            4,
            RapEventType.CANDIDATE_BATCH_RECEIVED,
            {
                "source": "local_chat",
                "candidate_count": 3,
                "prompt": [
                    {"role": "system", "content": "Exact system prompt body."},
                    {
                        "role": "user",
                        "content": "Exact user prompt body that must remain completely visible to the researcher.",
                    },
                ],
                "raw_response": "candidate one\ncandidate two",
                "latency_ms": 86.4,
                "deadline_slack_ms": 410.0,
                "late": False,
            },
        ),
        _event(
            5,
            RapEventType.CANDIDATE_EVALUATED,
            {
                "candidate_id": "candidate-rejected",
                "text": "Quasarword breaks this line",
                "syllables": [{"label": str(index)} for index in range(7)],
                "valid": False,
                "selected": False,
                "rejection_reasons": ["syllable_count:7!=9", "oov_words"],
                "oov_words": ["quasarword"],
                "total_score": None,
                "components": [],
            },
        ),
        _event(
            6,
            RapEventType.CANDIDATE_EVALUATED,
            {
                "candidate_id": "candidate-valid-low",
                "text": "Lower valid comparison line",
                "syllables": [{"label": item["label"]} for item in SCHEDULED],
                "valid": True,
                "selected": False,
                "rejection_reasons": [],
                "oov_words": [],
                "total_score": 0.61,
                "components": [{"name": "novelty", "value": 0.3, "weight": 0.05, "contribution": 0.015}],
            },
        ),
        _event(
            7,
            RapEventType.CANDIDATE_EVALUATED,
            {
                "candidate_id": "candidate-valid-high",
                "text": "Higher valid comparison line",
                "syllables": [{"label": item["label"]} for item in SCHEDULED],
                "valid": True,
                "selected": False,
                "rejection_reasons": [],
                "oov_words": [],
                "total_score": 0.72,
                "components": [{"name": "novelty", "value": 0.8, "weight": 0.05, "contribution": 0.04}],
            },
        ),
        _event(
            8,
            RapEventType.CANDIDATE_EVALUATED,
            {
                "candidate_id": "candidate-selected",
                "text": "Galaxies dance in a cosmic fight",
                "syllables": [{"label": item["label"]} for item in SCHEDULED],
                "valid": True,
                "selected": True,
                "rejection_reasons": [],
                "oov_words": [],
                "total_score": 0.84,
                "components": [
                    {"name": "stress_alignment", "value": 0.9, "weight": 0.3, "contribution": 0.27},
                    {"name": "topic_coverage", "value": 0.8, "weight": 0.2, "contribution": 0.16},
                ],
            },
        ),
        _event(
            9,
            RapEventType.BAR_REPLACED,
            {
                "source": "local_chat",
                "candidate_id": "candidate-selected",
                "text": "Galaxies dance in a cosmic fight",
                "total_score": 0.84,
                "flow": FLOW,
                "scheduled_syllables": SCHEDULED,
                "fallback": False,
                "fallback_reason": None,
            },
        ),
        _event(
            10,
            RapEventType.BAR_FROZEN,
            {
                "source": "local_chat",
                "text": "Galaxies dance in a cosmic fight",
                "flow": FLOW,
                "scheduled_syllables": SCHEDULED,
                "fallback": False,
            },
        ),
        _event(11, RapEventType.TICK, {"beat": 0, "tick_in_beat": 2}, tick=18),
        _event(12, RapEventType.SYLLABLE_EMITTED, {"label": "ax", "stressed": False, "jitter_ms": 0.2}, tick=18),
    )
    for event in events:
        projector.apply(event)
    return projector.state


def _text(state: TerminalRapViewState, *, detail: str = "full", width: int = 160) -> str:
    console = Console(record=True, color_system=None, width=width)
    console.print(build_dashboard(state, detail=detail, width=width))
    return console.export_text()


def test_wide_dashboard_contains_performance_flow_context_and_ranking(
    projected_state: TerminalRapViewState,
) -> None:
    output = _text(projected_state)

    for label in (
        "LIVE DELIVERY",
        "QUEUE",
        "CLOCK + HEALTH",
        "LLM REQUEST",
        "EXACT CONTEXT",
        "CANDIDATE GATE + RANKING",
        "SELECTED SCORE",
    ):
        assert label in output
    assert "0  1  2  3" in output
    assert "S . w M" in output
    assert "Syllable ticks: [0, 2, 3, 5, 7, 8, 10, 13, 15]" in output
    assert "Duration ticks: [2, 1, 2, 2, 1, 2, 3, 2, 1]" in output
    assert "Boundary strengths: [0, 0, 0, 0, 0, 0, 0, 0, 3]" in output
    assert "Rhyme groups: [None, None" in output
    assert "None, 'A']" in output


def test_dashboard_marks_candidates_scores_current_tick_and_scheduled_labels(
    projected_state: TerminalRapViewState,
) -> None:
    output = _text(projected_state)

    for expected in (
        "SELECTED",
        "REJECT",
        "OOV: quasarword",
        "syllable_count:7!=9",
        "stress_alignment",
        "0.27",
        "Current tick: 18",
        "Gal ax ies dance in a cos mic fight",
    ):
        assert expected in output
    assert output.index("candidate-selected") < output.index("candidate-valid-high")
    assert output.index("candidate-valid-high") < output.index("candidate-valid-low")
    assert "novelty=0.800 (contrib=0.040)" in output


def test_full_dashboard_preserves_exact_prompt_and_raw_response(
    projected_state: TerminalRapViewState,
) -> None:
    output = _text(projected_state, width=90)
    normalized = " ".join(output.split())

    assert "Exact system prompt body." in output
    assert "Exact user prompt body that must remain completely visible to the researcher." in normalized
    assert "candidate one" in output
    assert "candidate two" in output
    assert "…" not in output


@pytest.mark.parametrize(
    ("detail", "has_candidates", "has_prompt", "has_trace"),
    (
        ("summary", False, False, False),
        ("candidates", True, False, False),
        ("full", True, True, True),
    ),
)
def test_dashboard_respects_detail_boundaries(
    projected_state: TerminalRapViewState,
    detail: str,
    has_candidates: bool,
    has_prompt: bool,
    has_trace: bool,
) -> None:
    output = _text(projected_state, detail=detail)

    assert ("candidate-rejected" in output) is has_candidates
    assert ("Exact system prompt body." in output) is has_prompt
    assert ("EVENT TRACE" in output) is has_trace
