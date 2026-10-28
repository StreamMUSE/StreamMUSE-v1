"""Immutable one-bar capture and logical response identities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, isfinite

from streammuse.domain.rap.flow import FlowProvenance, FlowSlot, FlowTemplate


class InteractionPhase(str, Enum):
    DRUMS_IDLE = "DRUMS_IDLE"
    ARMED = "ARMED"
    CAPTURING = "CAPTURING"
    WAIT_REPLAY = "WAIT_REPLAY"
    MODEL_PLAY = "MODEL_PLAY"


@dataclass(frozen=True)
class TapRecord:
    server_time_seconds: float
    received_time_seconds: float
    client_time_ms: float | None = None
    uncertainty_ms: float = 0.0
    event_id: str | None = None

    def __post_init__(self) -> None:
        values = (self.server_time_seconds, self.received_time_seconds, self.uncertainty_ms)
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) for value in values):
            raise ValueError("tap times must be finite numbers")
        if self.client_time_ms is not None and not isfinite(self.client_time_ms):
            raise ValueError("client_time_ms must be finite")
        if self.uncertainty_ms < 0:
            raise ValueError("uncertainty_ms must be nonnegative")


@dataclass(frozen=True)
class CaptureWindow:
    bar: int
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.bar, int) or isinstance(self.bar, bool) or self.bar < 0:
            raise ValueError("capture bar must be a nonnegative integer")
        if not isfinite(self.start_seconds) or not isfinite(self.end_seconds) or self.end_seconds <= self.start_seconds:
            raise ValueError("capture window must have a positive finite duration")


@dataclass(frozen=True)
class CapturedRhythm:
    window: CaptureWindow
    raw_taps: tuple[TapRecord, ...]
    quantized_ticks: tuple[int, ...]
    template: FlowTemplate | None
    collision_count: int
    rejected_count: int


def quantize_capture(window: CaptureWindow, taps: tuple[TapRecord, ...], *, template_id: str) -> CapturedRhythm:
    """Quantize calibrated event time, never HTTP receipt time, to sixteen ticks."""
    tick_seconds = (window.end_seconds - window.start_seconds) / 16
    accepted = tuple(tap for tap in taps if window.start_seconds <= tap.server_time_seconds < window.end_seconds)
    positions = tuple((tap.server_time_seconds - window.start_seconds) / tick_seconds for tap in accepted)
    ticks = tuple(min(15, max(0, ceil(position - 0.5 - 1e-10))) for position in positions)
    onsets = sorted(set(ticks))
    template = None
    if onsets:
        template = FlowTemplate(
            template_id=template_id,
            name="Captured keyboard rhythm",
            ticks_per_beat=4,
            beats_per_bar=4,
            slots=tuple(FlowSlot(tick, next_tick - tick, 0.5) for tick, next_tick in zip(onsets, onsets[1:] + [16])),
            provenance=FlowProvenance(
                kind="user_capture", source="calibrated_keyboard_taps",
                quantization_error_ticks=max(abs(position - tick) for position, tick in zip(positions, ticks)),
            ),
        )
    return CapturedRhythm(window, taps, ticks, template, len(ticks) - len(onsets), len(taps) - len(accepted))


@dataclass(frozen=True)
class TurnRequest:
    turn_id: str
    epoch: int
    template: FlowTemplate
    topic: str
    response_bars: int
    history: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.turn_id or not self.topic.strip():
            raise ValueError("turn id and topic must be nonempty")
        if not isinstance(self.response_bars, int) or isinstance(self.response_bars, bool) or not 1 <= self.response_bars <= 8:
            raise ValueError("response_bars must be between one and eight")
