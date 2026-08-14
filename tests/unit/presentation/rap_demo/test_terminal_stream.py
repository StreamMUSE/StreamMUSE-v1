"""Tests for the append-only realtime rap research stream."""

from __future__ import annotations

import pytest

from streammuse.domain.rap import RapEvent, RapEventType
from streammuse.presentation.rap_demo.terminal_state import TerminalRapStateProjector
from streammuse.presentation.rap_demo.terminal_stream import StructuredStreamRenderer


FLOW = {
    "template_id": "baseline_syncopated_9",
    "name": "Baseline syncopated nine",
    "ticks_per_beat": 4,
    "beats_per_bar": 4,
    "slots": [
        {
            "slot_index": index,
            "tick_in_bar": tick,
            "duration_ticks": 1,
            "target_stress": stress,
            "boundary_strength": 3 if index == 8 else 0,
            "rhyme_group": "A" if index == 8 else None,
        }
        for index, (tick, stress) in enumerate(
            zip((0, 2, 3, 5, 7, 8, 10, 13, 15), (1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0))
        )
    ],
}
PROMPT = (
    {"role": "system", "content": "Write concise, clean rap lyric candidates."},
    {
        "role": "user",
        "content": "Use every supplied flow slot.\nSyllable ticks: [0, 2, 3, 5, 7, 8, 10, 13, 15]",
    },
)


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


def _render(renderer: StructuredStreamRenderer, projector: TerminalRapStateProjector, event: RapEvent) -> None:
    renderer.render(projector.apply(event), event)


def test_stream_groups_events_by_bar_and_phase_without_ansi() -> None:
    lines: list[str] = []
    renderer = StructuredStreamRenderer(detail="full", write=lines.append)
    projector = TerminalRapStateProjector()
    events = (
        _event(
            1,
            RapEventType.BAR_RESERVED,
            {"source": "prevalidated_fallback", "topic": "space", "text": "Fallback line", "flow": FLOW},
            request_id=None,
        ),
        _event(
            2,
            RapEventType.BAR_PLANNING_STARTED,
            {
                "topic": "space",
                "required_syllables": 9,
                "candidate_count": 3,
                "context_lines": ["Stars keep time"],
                "seed": 42,
                "flow": FLOW,
            },
        ),
        _event(
            3,
            RapEventType.CANDIDATE_BATCH_RECEIVED,
            {
                "source": "local_chat",
                "candidate_count": 3,
                "latency_ms": 86.4,
                "late": False,
                "prompt_tokens": 21,
                "completion_tokens": 13,
                "warning": "short_batch",
                "prompt": list(PROMPT),
                "raw_response": "Galaxies dance in a cosmic fight",
            },
        ),
        _event(
            4,
            RapEventType.CANDIDATE_EVALUATED,
            {
                "candidate_id": "candidate-1",
                "text": "Galaxies dance in a cosmic fight",
                "syllables": [{"label": label} for label in ("Gal", "ax", "ies", "dance", "in", "a", "cos", "mic", "fight")],
                "valid": True,
                "selected": True,
                "rejection_reasons": [],
                "oov_words": [],
                "total_score": 0.84,
                "components": [
                    {"name": "stress_alignment", "value": 0.9, "weight": 0.3, "contribution": 0.27}
                ],
            },
        ),
        _event(
            5,
            RapEventType.BAR_REPLACED,
            {"source": "local_chat", "text": "Galaxies dance in a cosmic fight", "total_score": 0.84},
        ),
        _event(6, RapEventType.BAR_FROZEN, {"source": "local_chat", "text": "Galaxies dance in a cosmic fight"}),
        _event(7, RapEventType.TICK, {"beat": 0, "tick_in_beat": 0}),
        _event(8, RapEventType.SYLLABLE_EMITTED, {"label": "Gal", "stressed": True, "jitter_ms": 0.2}),
        _event(9, RapEventType.FALLBACK_ACTIVATED, {"fallback_reason": "no_valid_candidate"}),
        _event(
            10,
            RapEventType.GENERATION_FAILED,
            {"error_type": "generation_error", "error_message": "server unavailable", "late": False},
        ),
    )

    for event in events:
        _render(renderer, projector, event)

    output = "\n".join(lines)
    for phase in ("PLAN", "MODEL", "GATE", "SELECT", "PLAY", "FALLBACK", "ERROR"):
        assert f"[BAR 02][{phase}]" in output
    assert "9/9" in output
    assert "stress_alignment" in output
    assert "Write concise, clean rap lyric candidates." in output
    assert "[PROMPT][user] Use every supplied flow slot." in output
    assert "[PROMPT+][user] Syllable ticks" in output
    assert "durations=[1, 1, 1" in output
    assert "boundaries=[0, 0, 0, 0, 0, 0, 0, 0, 3]" in output
    assert "rhymes=[None, None, None, None, None, None, None, None, 'A']" in output
    assert "tokens=21/13" in output
    assert "warning=short_batch" in output
    assert "\x1b[" not in output


def test_stream_escapes_control_characters_and_prefixes_every_prompt_line() -> None:
    lines: list[str] = []
    renderer = StructuredStreamRenderer(detail="full", write=lines.append)
    projector = TerminalRapStateProjector()
    _render(renderer, projector, _event(1, RapEventType.BAR_PLANNING_STARTED, {"flow": FLOW}))
    _render(
        renderer,
        projector,
        _event(
            2,
            RapEventType.CANDIDATE_BATCH_RECEIVED,
            {"prompt": [{"role": "user", "content": "line one\n\x1b[31mline two\tend"}]},
        ),
    )

    prompt_lines = [line for line in lines if "[PROMPT" in line]
    assert len(prompt_lines) == 2
    assert "[PROMPT][user] line one" in prompt_lines[0]
    assert "[PROMPT+][user] \\x1b[31mline two\\tend" in prompt_lines[1]
    assert all("\x1b" not in line for line in lines)


@pytest.mark.parametrize(
    ("detail", "has_candidate", "has_components", "has_prompt"),
    (
        ("summary", False, False, False),
        ("candidates", True, False, False),
        ("full", True, True, True),
    ),
)
def test_stream_respects_detail_boundaries(
    detail: str,
    has_candidate: bool,
    has_components: bool,
    has_prompt: bool,
) -> None:
    lines: list[str] = []
    renderer = StructuredStreamRenderer(detail=detail, write=lines.append)
    projector = TerminalRapStateProjector()
    planning = _event(
        1,
        RapEventType.BAR_PLANNING_STARTED,
        {"topic": "space", "required_syllables": 9, "candidate_count": 1, "flow": FLOW},
    )
    batch = _event(
        2,
        RapEventType.CANDIDATE_BATCH_RECEIVED,
        {"source": "local_chat", "candidate_count": 1, "prompt": list(PROMPT), "raw_response": "raw model body"},
    )
    candidate = _event(
        3,
        RapEventType.CANDIDATE_EVALUATED,
        {
            "candidate_id": "candidate-1",
            "text": "candidate detail marker",
            "syllables": [{"label": str(index)} for index in range(9)],
            "valid": False,
            "selected": False,
            "rejection_reasons": ["oov_words"],
            "oov_words": ["quasarword"],
            "components": [{"name": "stress_alignment", "value": 0.2, "weight": 0.3, "contribution": 0.06}],
        },
    )

    for event in (planning, batch, candidate):
        _render(renderer, projector, event)

    output = "\n".join(lines)
    assert ("candidate detail marker" in output) is has_candidate
    assert ("stress_alignment" in output) is has_components
    assert ("Write concise, clean rap lyric candidates." in output) is has_prompt
    assert ("raw model body" in output) is has_prompt


def test_stream_prints_session_and_presentation_error_without_host_terminal_assumptions() -> None:
    lines: list[str] = []
    renderer = StructuredStreamRenderer(write=lines.append)
    projector = TerminalRapStateProjector()

    _render(
        renderer,
        projector,
        _event(1, RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0, "max_bars": 3}, bar=None, tick=None),
    )
    _render(
        renderer,
        projector,
        _event(2, RapEventType.PRESENTATION_ERROR, {"sink": "web", "error_type": "RuntimeError"}),
    )
    _render(renderer, projector, _event(3, RapEventType.SESSION_STOPPED, {}, bar=None, tick=None))

    assert lines[0].startswith("[SESSION][START]")
    assert "[BAR 02][ERROR]" in lines[1]
    assert lines[-1].startswith("[SESSION][STOP]")
    assert renderer.is_tty is False
    assert renderer.terminal_width > 0


def test_stream_renders_dense_audio_warning_and_playback_evidence() -> None:
    lines: list[str] = []
    renderer = StructuredStreamRenderer(detail="full", write=lines.append)
    projector = TerminalRapStateProjector()
    events = (
        _event(1, RapEventType.BAR_AUDIO_READY, {"source": "local_chat", "frame_count": 192000, "render_latency_ms": 48, "warnings": ["pronunciation_fallback"]}, bar=2),
        _event(2, RapEventType.PRONUNCIATION_FALLBACK, {"word": "StreamMUSE", "source": "espeak_g2p", "action": "best_effort_rendered", "target_sample": 12345, "renderer_phonemes": ["str", "i:"]}, bar=2),
        _event(3, RapEventType.TIMING_PRESSURE, {"slot_index": 7, "word": "trans", "available_ms": 163, "rendered_ms": 241, "compression_ratio": 1.45, "overlap_ms": 12}, bar=2),
        _event(4, RapEventType.BAR_PLAYBACK_STARTED, {"queue_depth": 3, "buffered_seconds": 12, "absolute_frame": 384000}, bar=2),
    )

    for event in events:
        _render(renderer, projector, event)

    output = "\n".join(lines)
    assert "[BAR 03][AUDIO] ready source=local_chat frames=192000 render_ms=48" in output
    assert "[BAR 03][WARN] pronunciation word='StreamMUSE' source=espeak_g2p action=best_effort_rendered" in output
    assert "target_sample=12345 renderer_phonemes=['str', 'i:']" in output
    assert "[BAR 03][WARN] timing slot=7 word='trans' available_ms=163 rendered_ms=241 compression=1.45 overlap_ms=12" in output
    assert "[BAR 03][PLAY] started queue=3 buffered_s=12 absolute_frame=384000" in output
