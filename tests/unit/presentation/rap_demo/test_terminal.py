"""Tests for terminal renderer selection, compatibility, and failure fallback."""

from __future__ import annotations

import pytest
from rich.console import Console

from streammuse.domain.rap import RapEvent, RapEventType
from streammuse.presentation.rap_demo.terminal import TerminalRapSink
from streammuse.presentation.rap_demo.terminal_dashboard import RichLiveRenderer, build_dashboard
from streammuse.presentation.rap_demo.terminal_state import TerminalRapStateProjector


def _event(
    event_type: RapEventType,
    payload: dict,
    *,
    sequence: int = 1,
    bar: int | None = 1,
    tick: int | None = 16,
) -> RapEvent:
    return RapEvent(
        session_id="session",
        sequence=sequence,
        event_type=event_type,
        utc_time="2026-08-08T00:00:00+00:00",
        monotonic_ns=sequence,
        bar=bar,
        tick=tick,
        request_id="request-1",
        payload=payload,
    )


class RecordingDashboard:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.states = []
        self.closed = False

    def render(self, state) -> None:
        if self.fail:
            raise RuntimeError("render failed")
        self.states.append(state)

    def close(self) -> None:
        self.closed = True


def test_auto_uses_stream_for_non_tty() -> None:
    lines: list[str] = []
    sink = TerminalRapSink(layout="auto", write=lines.append, is_tty=False)

    sink(_event(RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0, "max_bars": 3}, bar=None, tick=None))

    assert any("[SESSION][START]" in line for line in lines)


def test_injected_writer_preserves_stream_behavior_without_explicit_tty() -> None:
    lines: list[str] = []
    sink = TerminalRapSink("full", write=lines.append)

    sink(
        _event(
            RapEventType.CANDIDATE_EVALUATED,
            {
                "candidate_id": "candidate-1",
                "text": "space",
                "valid": False,
                "selected": False,
                "rejection_reasons": ["syllable_count:1!=9"],
                "oov_words": [],
                "components": [
                    {"name": "stress_alignment", "value": 0.8, "weight": 0.3, "contribution": 0.24}
                ],
            },
        )
    )

    output = "\n".join(lines)
    assert "REJECT" in output
    assert "rejection_reasons" in output
    assert "stress_alignment" in output


def test_auto_uses_split_for_wide_tty() -> None:
    dashboard = RecordingDashboard()
    sink = TerminalRapSink(
        layout="auto",
        is_tty=True,
        terminal_width=160,
        dashboard_factory=lambda **_: dashboard,
    )

    sink(_event(RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0, "max_bars": 3}, bar=None, tick=None))

    assert len(dashboard.states) == 1


def test_auto_uses_stream_for_narrow_tty() -> None:
    lines: list[str] = []
    dashboard = RecordingDashboard()
    sink = TerminalRapSink(
        layout="auto",
        write=lines.append,
        is_tty=True,
        terminal_width=90,
        dashboard_factory=lambda **_: dashboard,
    )

    sink(_event(RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0}, bar=None, tick=None))

    assert lines and not dashboard.states


def test_explicit_split_stacks_at_narrow_width() -> None:
    projector = TerminalRapStateProjector()
    state = projector.apply(
        _event(RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0}, bar=None, tick=None)
    )
    console = Console(record=True, color_system=None, width=90)
    console.print(build_dashboard(state, detail="full", width=90))
    output = console.export_text()

    assert output.index("LIVE DELIVERY") < output.index("LLM REQUEST")


def test_dashboard_failure_switches_once_to_stream() -> None:
    lines: list[str] = []
    dashboard = RecordingDashboard(fail=True)
    sink = TerminalRapSink(
        layout="split",
        write=lines.append,
        is_tty=True,
        terminal_width=160,
        dashboard_factory=lambda **_: dashboard,
    )

    sink(_event(RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0, "max_bars": 3}, bar=None, tick=None))
    sink(_event(RapEventType.TICK, {"beat": 0, "tick_in_beat": 0}, sequence=2, tick=0))

    assert sum("[PRESENTATION][WARN]" in line for line in lines) == 1
    assert any("[TICK]" in line for line in lines)
    assert dashboard.closed is True


def test_session_stop_closes_live_renderer() -> None:
    dashboard = RecordingDashboard()
    sink = TerminalRapSink(
        layout="split",
        is_tty=True,
        terminal_width=160,
        dashboard_factory=lambda **_: dashboard,
    )

    sink(_event(RapEventType.SESSION_STOPPED, {}, bar=None, tick=None))

    assert dashboard.closed is True


def test_rich_live_renderer_renders_and_closes_idempotently() -> None:
    projector = TerminalRapStateProjector()
    state = projector.apply(
        _event(RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0}, bar=None, tick=None)
    )
    console = Console(record=True, color_system=None, width=90)
    renderer = RichLiveRenderer(detail="summary", console=console, width=90)

    renderer.render(state)
    renderer.close()
    renderer.close()

    assert "LIVE DELIVERY" in console.export_text()


def test_rich_live_renderer_reads_console_width_on_each_refresh(monkeypatch) -> None:
    projector = TerminalRapStateProjector()
    state = projector.apply(_event(RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0}, bar=None, tick=None))
    console = Console(record=True, color_system=None, width=160)
    widths: list[int] = []

    def dashboard(_state, detail: str, width: int):
        widths.append(width)
        return f"{detail}:{width}"

    monkeypatch.setattr("streammuse.presentation.rap_demo.terminal_dashboard.build_dashboard", dashboard)
    renderer = RichLiveRenderer(detail="summary", console=console)
    renderer.render(state)
    console.width = 90
    renderer.render(state)
    renderer.close()

    assert widths == [160, 90]


@pytest.mark.parametrize("detail", ("minimal", "debug", ""))
def test_sink_rejects_unknown_detail(detail: str) -> None:
    with pytest.raises(ValueError, match="terminal detail"):
        TerminalRapSink(detail=detail)


@pytest.mark.parametrize("layout", ("wide", "live", ""))
def test_sink_rejects_unknown_layout(layout: str) -> None:
    with pytest.raises(ValueError, match="terminal layout"):
        TerminalRapSink(layout=layout)
