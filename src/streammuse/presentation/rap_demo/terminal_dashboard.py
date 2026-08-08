"""Pure Rich dashboard construction for the realtime rap research monitor."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

from streammuse.presentation.rap_demo.terminal_state import (
    TerminalRapBarView,
    TerminalRapCandidateView,
    TerminalRapViewState,
)


class RichLiveRenderer:
    """Own a Rich Live display while delegating state interpretation to the builder."""

    def __init__(self, *, detail: str = "full", console: Console | None = None, width: int | None = None) -> None:
        if detail not in {"summary", "candidates", "full"}:
            raise ValueError("terminal detail must be summary, candidates, or full")
        self._detail = detail
        self._console = console or Console()
        self._width = width
        self._live: Live | None = None
        self._closed = False

    def render(self, state: TerminalRapViewState) -> None:
        if self._closed:
            return
        width = self._width if self._width is not None else self._console.width
        dashboard = build_dashboard(state, detail=self._detail, width=width)
        if self._live is None:
            self._live = Live(
                dashboard,
                console=self._console,
                auto_refresh=False,
                refresh_per_second=10,
                transient=False,
            )
            self._live.start(refresh=True)
            return
        self._live.update(dashboard, refresh=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._live is not None:
            self._live.stop()


def build_dashboard(state: TerminalRapViewState, detail: str, width: int) -> RenderableType:
    """Build a deterministic wide or stacked dashboard from projected state."""
    if detail not in {"summary", "candidates", "full"}:
        raise ValueError("terminal detail must be summary, candidates, or full")
    left = Group(*_performance_sections(state, detail))
    right = Group(*_research_sections(state, detail))
    if width < 120:
        return Group(left, right)

    columns = Table.grid(expand=True, padding=(0, 2))
    columns.add_column(ratio=1)
    columns.add_column(ratio=1)
    columns.add_row(left, right)
    return columns


def _performance_sections(state: TerminalRapViewState, detail: str) -> tuple[RenderableType, ...]:
    bar = _active_bar(state)
    flow = bar.flow if bar is not None and bar.flow is not None else (
        state.latest_request.flow if state.latest_request is not None else None
    )
    return (
        _live_delivery(state, bar),
        _flow_strip(state, bar, flow, detail),
        _queue(state),
        _clock_and_health(state),
        _recent_commits(state),
    )


def _research_sections(state: TerminalRapViewState, detail: str) -> tuple[RenderableType, ...]:
    sections: list[RenderableType] = [_llm_request(state)]
    if detail == "full":
        sections.append(_exact_context(state))
    sections.append(_model_response(state, detail))
    if detail in {"candidates", "full"}:
        sections.append(_candidate_ranking(state, detail))
    if detail == "full":
        sections.extend((_selected_score(state), _event_trace(state)))
    return tuple(sections)


def _live_delivery(state: TerminalRapViewState, bar: TerminalRapBarView | None) -> RenderableType:
    rows: list[tuple[str, RenderableType]] = []
    if bar is None:
        rows.append(("State", Text("Waiting for first reserved bar", style="yellow")))
    else:
        status = "FROZEN" if bar.frozen else "FALLBACK ARMED" if bar.fallback else "READY"
        status_style = "green" if bar.frozen and not bar.fallback else "yellow"
        rows.extend(
            (
                ("Bar", Text(f"{bar.bar + 1:02d}  {_safe(bar.topic or '-')}  {status}", style=status_style)),
                ("Lyric", Text(_safe(bar.text or "-"), style="bold green" if not bar.fallback else "yellow")),
                ("Source", Text(_safe(bar.source or "-"), style="dim")),
                ("Score", Text(_score(bar.total_score), style="green" if bar.total_score is not None else "dim")),
                ("Fallback", Text(_fallback_status(bar), style="yellow" if bar.fallback else "dim")),
            )
        )
    return _section("LIVE DELIVERY", rows)


def _flow_strip(
    state: TerminalRapViewState,
    bar: TerminalRapBarView | None,
    flow: Mapping[str, Any] | None,
    detail: str,
) -> RenderableType:
    rows: list[tuple[str, RenderableType]] = []
    if flow is None:
        rows.append(("Flow", Text("No structured flow snapshot", style="yellow")))
        return _section("FLOW", rows)

    slots = _mapping_items(flow.get("slots"))
    ticks_per_beat = _positive_int(flow.get("ticks_per_beat"), 4)
    beats_per_bar = _positive_int(flow.get("beats_per_bar"), 4)
    tick_count = ticks_per_beat * beats_per_bar
    grid = ["."] * tick_count
    for slot in slots:
        tick = slot.get("tick_in_bar")
        stress = slot.get("target_stress")
        if isinstance(tick, int) and not isinstance(tick, bool) and 0 <= tick < tick_count:
            grid[tick] = _stress_marker(stress)
    beat_ticks = [
        " ".join(f"{tick:>2}" for tick in range(start, start + ticks_per_beat))
        for start in range(0, tick_count, ticks_per_beat)
    ]
    beat_pattern = [
        " ".join(grid[start : start + ticks_per_beat])
        for start in range(0, tick_count, ticks_per_beat)
    ]
    slot_ticks = [slot.get("tick_in_bar") for slot in slots]
    stresses = [slot.get("target_stress") for slot in slots]
    durations = [slot.get("duration_ticks") for slot in slots]
    boundaries = [slot.get("boundary_strength") for slot in slots]
    rhymes = [slot.get("rhyme_group") for slot in slots]
    scheduled = bar.scheduled_syllables if bar is not None else ()
    labels = [_safe(item.get("label")) for item in scheduled if item.get("label") is not None]
    local_tick = state.current_tick % tick_count if isinstance(state.current_tick, int) else None

    rows.extend(
        (
            ("Template", Text(_safe(flow.get("template_id", "-")), style="cyan")),
            ("Ticks", Text(" | ".join(beat_ticks), style="cyan")),
            ("Pattern", Text(" | ".join(beat_pattern), style="cyan")),
            ("Now", Text(f"Current tick: {state.current_tick} (bar tick {local_tick})", style="bold cyan")),
            ("Delivery", Text(" ".join(labels) if labels else "-")),
        )
    )
    if detail == "full":
        rows.extend(
            (
                ("Exact", Text(f"Syllable ticks: {slot_ticks}")),
                ("Stress", Text(f"Target stress: {stresses}")),
                ("Duration", Text(f"Duration ticks: {durations}")),
                ("Boundary", Text(f"Boundary strengths: {boundaries}")),
                ("Rhyme", Text(f"Rhyme groups: {rhymes}")),
                ("Meter", Text(f"{beats_per_bar}/4, {ticks_per_beat} ticks/beat", style="dim")),
                ("Source", Text(_provenance(flow), style="dim")),
            )
        )
    return _section("FLOW", rows)


def _queue(state: TerminalRapViewState) -> RenderableType:
    current = state.current_bar
    future = [bar for index, bar in sorted(state.bars.items()) if current is None or index > current]
    rows: list[tuple[str, RenderableType]] = []
    if future:
        next_bar = future[0]
        rows.extend(
            (
                ("Next", Text(f"Bar {next_bar.bar + 1:02d}  {_safe(next_bar.topic or '-')}", style="cyan")),
                ("Lyric", Text(_safe(next_bar.text or "-"))),
                ("Safety", Text(_fallback_status(next_bar), style="yellow" if next_bar.fallback else "green")),
                ("Score", Text(_score(next_bar.total_score), style="green" if next_bar.total_score is not None else "dim")),
            )
        )
    else:
        request = state.latest_request
        pending = request is not None and (current is None or request.bar is None or request.bar >= current)
        rows.append(("Next", Text("Model request pending" if pending else "No reserved future bar", style="yellow" if pending else "dim")))
    return _section("QUEUE", rows)


def _clock_and_health(state: TerminalRapViewState) -> RenderableType:
    metadata = state.session_metadata
    batch = state.latest_batch.payload if state.latest_batch is not None else {}
    pending = state.latest_request is not None and state.latest_batch is None
    health = "STOPPED" if state.stopped else "ERROR" if state.last_error is not None else "LIVE"
    health_style = "red" if health == "ERROR" else "green" if health == "LIVE" else "dim"
    rows = (
        ("Clock", Text(f"{metadata.get('tempo_bpm', '-')} BPM  tick={state.current_tick}", style="cyan")),
        ("Model", Text("pending" if pending else "idle/complete", style="yellow" if pending else "green")),
        ("Latency", Text(f"{batch.get('latency_ms', '-')} ms  slack={batch.get('deadline_slack_ms', '-')} ms")),
        ("Health", Text(health, style=health_style)),
    )
    return _section("CLOCK + HEALTH", rows)


def _recent_commits(state: TerminalRapViewState) -> RenderableType:
    frozen = [bar for _, bar in sorted(state.bars.items()) if bar.frozen][-3:]
    rows = [
        (f"Bar {bar.bar + 1:02d}", Text(f"{_safe(bar.source or '-')} | {_safe(bar.text or '-')}", style="dim"))
        for bar in frozen
    ]
    if not rows:
        rows.append(("Bars", Text("No frozen bars", style="dim")))
    return _section("RECENT COMMITS", rows)


def _llm_request(state: TerminalRapViewState) -> RenderableType:
    request = state.latest_request
    if request is None:
        return _section("LLM REQUEST", (("State", Text("No request yet", style="dim")),))
    flow = request.flow or {}
    slots = _mapping_items(flow.get("slots"))
    rows: list[tuple[str, RenderableType]] = [
        ("Request", Text(_safe(request.request_id or "-"), style="cyan")),
        ("Target", Text(f"Bar {_display_bar(request.bar)}  topic={_safe(request.topic or '-')}")),
        ("Shape", Text(f"{request.required_syllables} syllables  {request.candidate_count} candidates  seed={request.seed}")),
        ("Flow", Text(_safe(flow.get("template_id", "-")), style="cyan")),
        ("Ticks", Text(f"Syllable ticks: {[slot.get('tick_in_bar') for slot in slots]}")),
        ("Stress", Text(f"Target stress: {[slot.get('target_stress') for slot in slots]}")),
        ("Duration", Text(f"Duration ticks: {[slot.get('duration_ticks') for slot in slots]}")),
        ("Boundary", Text(f"Boundary strengths: {[slot.get('boundary_strength') for slot in slots]}")),
        ("Rhyme", Text(f"Rhyme groups: {[slot.get('rhyme_group') for slot in slots]}")),
    ]
    return _section("LLM REQUEST", rows)


def _exact_context(state: TerminalRapViewState) -> RenderableType:
    rows: list[tuple[str, RenderableType]] = []
    request = state.latest_request
    if request is not None:
        history = "\n".join(_safe(item) for item in request.context_lines) if request.context_lines else "(none)"
        rows.append(("History", Text(history)))
    batch = state.latest_batch
    if batch is not None:
        for item in batch.prompt:
            rows.append(
                (
                    _safe(item.get("role", "message")).upper(),
                    Text(_safe(item.get("content", ""), preserve_newlines=True)),
                )
            )
    if not rows:
        rows.append(("Prompt", Text("Not received", style="dim")))
    return _section("EXACT CONTEXT", rows)


def _model_response(state: TerminalRapViewState, detail: str) -> RenderableType:
    batch = state.latest_batch
    if batch is None:
        return _section("MODEL RESPONSE", (("State", Text("Waiting", style="yellow")),))
    payload = batch.payload
    late = payload.get("late") is True
    error = payload.get("error_type")
    status = "ERROR" if error else "LATE" if late else "ON TIME"
    style = "red" if error else "yellow" if late else "green"
    rows: list[tuple[str, RenderableType]] = [
        ("Status", Text(status, style=style)),
        ("Source", Text(_safe(batch.source or "-"))),
        ("Timing", Text(f"latency={payload.get('latency_ms', '-')} ms  slack={payload.get('deadline_slack_ms', '-')} ms")),
        ("Count", Text(str(payload.get("candidate_count", "-")))),
        ("Tokens", Text(f"{payload.get('prompt_tokens', '-')} / {payload.get('completion_tokens', '-')}")),
        ("Warning", Text(_safe(payload.get("warning") or "none"), style="yellow" if payload.get("warning") else "dim")),
    ]
    if detail == "full":
        rows.append(("Raw", Text(_safe(batch.raw_response or "-", preserve_newlines=True))))
    return _section("MODEL RESPONSE", rows)


def _candidate_ranking(state: TerminalRapViewState, detail: str) -> RenderableType:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=9)
    table.add_column(width=22)
    table.add_column(width=9)
    table.add_column(ratio=1)
    request = state.latest_request
    required = request.required_syllables if request is not None else None
    if not state.candidates:
        table.add_row(Text("PENDING", style="yellow"), "-", "-", "No evaluated candidates")
    candidates = sorted(state.candidates, key=_candidate_sort_key)
    for candidate in candidates:
        marker, style = _candidate_marker(candidate)
        payload = candidate.payload
        syllables = payload.get("syllables")
        observed = len(syllables) if isinstance(syllables, tuple) else "?"
        score = payload.get("total_score")
        table.add_row(
            Text(marker, style=style),
            Text(_safe(candidate.candidate_id or "-"), style="dim"),
            Text(f"{observed}/{required}  {_score(score)}"),
            Text(_safe(candidate.text or "-")),
        )
        reasons = ", ".join(candidate.rejection_reasons)
        oov = payload.get("oov_words")
        oov_words = ", ".join(str(item) for item in oov) if isinstance(oov, tuple) else ""
        if reasons or oov_words:
            rejection_detail = " | ".join(item for item in (reasons, f"OOV: {oov_words}" if oov_words else "") if item)
            table.add_row("", "", "", Text(_safe(rejection_detail), style="red"))
        if detail == "full":
            components = _mapping_items(payload.get("components"))
            if components:
                comparison = " | ".join(
                    f"{item.get('name', 'component')}={_score(item.get('value'))}"
                    f" (contrib={_score(item.get('contribution'))})"
                    for item in components
                )
                table.add_row("", "", "", Text(_safe(comparison), style="dim"))
    return Group(_heading("CANDIDATE GATE + RANKING"), table, Text(""))


def _selected_score(state: TerminalRapViewState) -> RenderableType:
    selected = next((candidate for candidate in state.candidates if candidate.selected is True), None)
    rows: list[tuple[str, RenderableType]] = []
    if selected is None:
        rows.append(("Score", Text("No selected candidate", style="yellow")))
    else:
        rows.append(("Total", Text(_score(selected.payload.get("total_score")), style="green")))
        for component in _mapping_items(selected.payload.get("components")):
            rows.append(
                (
                    _safe(component.get("name", "component")),
                    Text(
                        f"value={component.get('value', '-')}  weight={component.get('weight', '-')}  "
                        f"contribution={component.get('contribution', '-')}"
                    ),
                )
            )
    return _section("SELECTED SCORE", rows)


def _event_trace(state: TerminalRapViewState) -> RenderableType:
    rows = [
        (
            f"#{event.sequence}",
            Text(
                f"{event.event_type.value}  bar={_display_bar(event.bar)} tick={event.tick} "
                f"request={_safe(event.request_id or '-')}",
                style="red" if "error" in event.event_type.value or "failed" in event.event_type.value else "dim",
            ),
        )
        for event in state.recent_events[-8:]
    ]
    if not rows:
        rows.append(("Events", Text("None", style="dim")))
    return _section("EVENT TRACE", rows)


def _section(title: str, rows: Iterable[tuple[str, RenderableType]]) -> RenderableType:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=18, style="dim")
    table.add_column(ratio=1)
    for label, value in rows:
        table.add_row(label, value)
    return Group(_heading(title), table, Text(""))


def _heading(title: str) -> Text:
    return Text(title, style="bold cyan")


def _active_bar(state: TerminalRapViewState) -> TerminalRapBarView | None:
    if state.current_bar is not None and state.current_bar in state.bars:
        return state.bars[state.current_bar]
    if state.latest_request is not None and state.latest_request.bar in state.bars:
        return state.bars[state.latest_request.bar]
    return next(reversed(state.bars.values()), None) if state.bars else None


def _mapping_items(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _positive_int(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else default


def _stress_marker(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "?"
    return "S" if value >= 0.85 else "M" if value >= 0.5 else "w"


def _provenance(flow: Mapping[str, Any]) -> str:
    provenance = flow.get("provenance")
    if not isinstance(provenance, Mapping):
        return "-"
    return f"{_safe(provenance.get('kind', '-'))} | {_safe(provenance.get('source', '-'))}"


def _fallback_status(bar: TerminalRapBarView) -> str:
    if not bar.fallback:
        return "disarmed"
    return f"armed: {_safe(bar.fallback_reason or 'unspecified')}"


def _safe(value: object, *, preserve_newlines: bool = False) -> str:
    escaped: list[str] = []
    for character in str(value):
        codepoint = ord(character)
        if character == "\n" and preserve_newlines:
            escaped.append(character)
        elif codepoint < 32 or codepoint == 127:
            escaped.append(f"\\x{codepoint:02x}")
        else:
            escaped.append(character)
    return "".join(escaped)


def _display_bar(bar: int | None) -> str:
    return "-" if bar is None else f"{bar + 1:02d}"


def _score(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"


def _candidate_marker(candidate: TerminalRapCandidateView) -> tuple[str, str]:
    if candidate.selected is True:
        return "SELECTED", "bold green"
    if candidate.valid is True:
        return "VALID", "green"
    return "REJECT", "red"


def _candidate_sort_key(candidate: TerminalRapCandidateView) -> tuple[int, float, str]:
    score = candidate.payload.get("total_score")
    numeric_score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else float("-inf")
    gate = 2 if candidate.selected is True else 1 if candidate.valid is True else 0
    return (-gate, -numeric_score, candidate.candidate_id or "")
