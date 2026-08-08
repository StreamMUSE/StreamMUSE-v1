"""Append-only, ANSI-free rendering for realtime rap research events."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Mapping
from typing import Any

from streammuse.domain.rap import RapEvent, RapEventType
from streammuse.presentation.rap_demo.terminal_state import TerminalRapViewState


class StructuredStreamRenderer:
    """Render canonical events as dense phase-labelled text records."""

    def __init__(self, detail: str = "full", *, write: Callable[[str], None] | None = None) -> None:
        if detail not in {"summary", "candidates", "full"}:
            raise ValueError("terminal detail must be summary, candidates, or full")
        self._detail = detail
        self._write = write or (lambda line: print(line, flush=True))
        self._is_tty = False if write is not None else bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._terminal_width = max(1, shutil.get_terminal_size((120, 24)).columns)

    @property
    def is_tty(self) -> bool:
        return self._is_tty

    @property
    def terminal_width(self) -> int:
        return self._terminal_width

    def warning(self, message: str) -> None:
        self._write(f"[PRESENTATION][WARN] {message}")

    def render(self, state: TerminalRapViewState, event: RapEvent) -> None:
        kind = event.event_type
        payload = event.payload
        prefix = self._prefix(event)

        if kind == RapEventType.SESSION_STARTED:
            self._write(
                f"[SESSION][START] tempo={_value(payload.get('tempo_bpm'))} "
                f"max_bars={_value(payload.get('max_bars'))} session={event.session_id}"
            )
        elif kind == RapEventType.SESSION_STOPPED:
            self._write(f"[SESSION][STOP] session={event.session_id}")
        elif kind == RapEventType.BAR_RESERVED:
            self._write(
                f"{prefix('PLAN')} reserve source={_value(payload.get('source'))} "
                f"topic={_quoted(payload.get('topic'))} fallback={_value(payload.get('fallback'))} "
                f"reason={_value(payload.get('fallback_reason'))} text={_quoted(payload.get('text'))}"
            )
            if self._detail == "full":
                self._write(f"{_continuation()} {_flow_summary(payload.get('flow'))}")
        elif kind == RapEventType.BAR_PLANNING_STARTED:
            request = state.latest_request
            self._write(
                f"{prefix('PLAN')} request={_value(event.request_id)} topic={_quoted(payload.get('topic'))} "
                f"syllables={_value(payload.get('required_syllables'))} candidates={_value(payload.get('candidate_count'))} "
                f"seed={_value(payload.get('seed'))}"
            )
            if self._detail == "full":
                flow = request.flow if request is not None else payload.get("flow")
                self._write(f"{_continuation()} {_flow_summary(flow)}")
                context = request.context_lines if request is not None else ()
                self._write(f"{_continuation()} context={list(context)!r}")
        elif kind == RapEventType.CANDIDATE_BATCH_RECEIVED:
            self._write(
                f"{prefix('MODEL')} source={_value(payload.get('source'))} count={_value(payload.get('candidate_count'))} "
                f"latency_ms={_number(payload.get('latency_ms'))} late={_value(payload.get('late'))} "
                f"error={_value(payload.get('error_type'))}"
            )
            if self._detail == "full":
                batch = state.latest_batch
                prompt = batch.prompt if batch is not None else ()
                for item in prompt:
                    self._write(
                        f"{_continuation()} [PROMPT][{_value(item.get('role'))}] {_value(item.get('content'))}"
                    )
                raw_response = batch.raw_response if batch is not None else payload.get("raw_response")
                self._write(f"{_continuation()} raw_response={raw_response!r}")
        elif kind == RapEventType.CANDIDATE_EVALUATED:
            if self._detail == "summary":
                return
            required = state.latest_request.required_syllables if state.latest_request is not None else None
            syllables = payload.get("syllables")
            observed = len(syllables) if isinstance(syllables, (list, tuple)) else "?"
            marker = "SELECTED" if payload.get("selected") is True else "PASS" if payload.get("valid") is True else "REJECT"
            self._write(
                f"{prefix('GATE')} {marker} id={_value(payload.get('candidate_id'))} "
                f"syllables={observed}/{_value(required)} score={_number(payload.get('total_score'))} "
                f"text={_quoted(payload.get('text'))}"
            )
            self._write(
                f"{_continuation()} rejection_reasons={_value(payload.get('rejection_reasons'))} "
                f"oov_words={_value(payload.get('oov_words'))}"
            )
            if self._detail == "full":
                for component in _mappings(payload.get("components")):
                    self._write(
                        f"{_continuation()} {_value(component.get('name'))}: value={_number(component.get('value'))} "
                        f"weight={_number(component.get('weight'))} contribution={_number(component.get('contribution'))}"
                    )
        elif kind == RapEventType.BAR_REPLACED:
            self._write(
                f"{prefix('SELECT')} source={_value(payload.get('source'))} candidate={_value(payload.get('candidate_id'))} "
                f"score={_number(payload.get('total_score'))} text={_quoted(payload.get('text'))}"
            )
        elif kind == RapEventType.BAR_FROZEN:
            self._write(
                f"{prefix('SELECT')} frozen source={_value(payload.get('source'))} fallback={_value(payload.get('fallback'))} "
                f"reason={_value(payload.get('fallback_reason'))} text={_quoted(payload.get('text'))}"
            )
        elif kind == RapEventType.FALLBACK_ACTIVATED:
            self._write(f"{prefix('FALLBACK')} reason={_value(payload.get('fallback_reason'))}")
        elif kind in (RapEventType.GENERATION_FAILED, RapEventType.PRESENTATION_ERROR):
            self._write(
                f"{prefix('ERROR')} type={_value(payload.get('error_type'))} "
                f"message={_quoted(payload.get('error_message'))} late={_value(payload.get('late'))}"
            )
        elif kind == RapEventType.TICK:
            if self._detail == "full" or payload.get("tick_in_beat") == 0:
                self._write(
                    f"{prefix('PLAY')}[TICK] beat={_ordinal(payload.get('beat'))} "
                    f"subdivision={_ordinal(payload.get('tick_in_beat'))} absolute={_value(event.tick)}"
                )
        elif kind == RapEventType.SYLLABLE_EMITTED:
            self._write(
                f"{prefix('PLAY')} syllable label={_quoted(payload.get('label'))} stressed={_value(payload.get('stressed'))} "
                f"jitter_ms={_number(payload.get('jitter_ms'), digits=3)}"
            )

    @staticmethod
    def _prefix(event: RapEvent) -> Callable[[str], str]:
        bar = "--" if event.bar is None else f"{event.bar + 1:02d}"
        return lambda phase: f"[BAR {bar}][{phase}]"


def _continuation() -> str:
    return "                 "


def _value(value: object) -> str:
    return "-" if value is None else str(value)


def _quoted(value: object) -> str:
    return repr(value) if value is not None else "-"


def _number(value: object, *, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return _value(value)


def _ordinal(value: object) -> str:
    return str(value + 1) if isinstance(value, int) and not isinstance(value, bool) else "-"


def _mappings(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _flow_summary(value: object) -> str:
    if not isinstance(value, Mapping):
        return "flow=-"
    slots = _mappings(value.get("slots"))
    ticks = [slot.get("tick_in_bar") for slot in slots]
    stresses = [slot.get("target_stress") for slot in slots]
    return f"flow={_value(value.get('template_id'))} ticks={ticks} stress={stresses}"
