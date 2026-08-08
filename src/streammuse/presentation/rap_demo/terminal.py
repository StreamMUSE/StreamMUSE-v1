"""Terminal facade selecting live or append-only realtime rap monitoring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from rich.console import Console

from streammuse.domain.rap import RapEvent, RapEventType
from streammuse.presentation.rap_demo.terminal_dashboard import RichLiveRenderer
from streammuse.presentation.rap_demo.terminal_state import TerminalRapStateProjector, TerminalRapViewState
from streammuse.presentation.rap_demo.terminal_stream import StructuredStreamRenderer


class DashboardRenderer(Protocol):
    def render(self, state: TerminalRapViewState) -> None: ...

    def close(self) -> None: ...


class TerminalRapSink:
    """Project each event once and route it to the selected terminal renderer."""

    def __init__(
        self,
        detail: str = "full",
        *,
        layout: str = "auto",
        write: Callable[[str], None] | None = None,
        console: Console | None = None,
        is_tty: bool | None = None,
        terminal_width: int | None = None,
        dashboard_factory: Callable[..., DashboardRenderer] | None = None,
    ) -> None:
        if detail not in {"summary", "candidates", "full"}:
            raise ValueError("terminal detail must be summary, candidates, or full")
        if layout not in {"auto", "split", "stream"}:
            raise ValueError("terminal layout must be auto, split, or stream")
        self._projector = TerminalRapStateProjector()
        self._stream = StructuredStreamRenderer(detail=detail, write=write)
        tty = bool(is_tty) if is_tty is not None else self._stream.is_tty
        width = terminal_width or self._stream.terminal_width
        use_split = layout == "split" or (layout == "auto" and tty and width >= 120)
        factory = dashboard_factory or RichLiveRenderer
        self._dashboard = factory(detail=detail, console=console, width=width) if use_split else None
        self._dashboard_failed = False

    def __call__(self, event: RapEvent) -> None:
        state = self._projector.apply(event)
        if self._dashboard is None:
            self._stream.render(state, event)
            return
        try:
            self._dashboard.render(state)
            if event.event_type == RapEventType.SESSION_STOPPED:
                self._dashboard.close()
        except Exception as exc:
            try:
                self._dashboard.close()
            except Exception:
                pass
            self._dashboard = None
            if not self._dashboard_failed:
                self._stream.warning(f"dashboard disabled: {type(exc).__name__}")
                self._dashboard_failed = True
            self._stream.render(state, event)
