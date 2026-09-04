"""Shared input timing policy for realtime human performance."""

from __future__ import annotations

import math
from dataclasses import dataclass

from streammuse.domain.musical import MusicalEvent
from streammuse.domain.timing import Tempo


REALTIME_INPUT_TYPES = frozenset({"midi_device", "keyboard", "queue"})


@dataclass(frozen=True, slots=True)
class InputQuantizationResult:
    """Pure diagnostic result for one realtime input timestamp."""

    elapsed_seconds: float
    raw_tick: float
    floor_tick: int
    tick_phase: float
    snap_forward_fraction: float
    snapped_forward: bool
    quantized_tick: int
    signed_error_ticks: float
    signed_error_ms: float


def clamp_snap_forward_fraction(value: float) -> float:
    """Clamp snap-forward tolerance to the supported [0, 1] range."""
    return min(1.0, max(0.0, float(value)))


def effective_input_snap_forward_fraction(input_type: str, configured_fraction: float) -> float:
    """Return the snap-forward fraction for an input mode.

    Human realtime inputs receive the configured tolerance. Deterministic inputs
    keep exact floor-based timing.
    """
    if input_type in REALTIME_INPUT_TYPES:
        return clamp_snap_forward_fraction(configured_fraction)
    return 0.0


def seconds_to_input_tick(
    elapsed_seconds: float,
    tempo: Tempo,
    *,
    snap_forward_fraction: float,
) -> int:
    """Convert elapsed realtime seconds to an input tick with optional snap-forward."""
    return diagnose_input_quantization(
        elapsed_seconds,
        tempo,
        snap_forward_fraction=snap_forward_fraction,
    ).quantized_tick


def diagnose_input_quantization(
    elapsed_seconds: float,
    tempo: Tempo,
    *,
    snap_forward_fraction: float,
) -> InputQuantizationResult:
    """Describe the existing floor/snap-forward input quantization decision."""
    seconds = max(0.0, float(elapsed_seconds))
    raw_tick = seconds / tempo.seconds_per_tick
    floor_tick = int(math.floor(raw_tick))
    fraction = clamp_snap_forward_fraction(snap_forward_fraction)
    tick_phase = raw_tick - floor_tick
    snapped_forward = fraction > 0.0 and tick_phase >= (1.0 - fraction)
    quantized_tick = floor_tick + int(snapped_forward)
    signed_error_ticks = float(quantized_tick) - raw_tick
    return InputQuantizationResult(
        elapsed_seconds=seconds,
        raw_tick=raw_tick,
        floor_tick=floor_tick,
        tick_phase=tick_phase,
        snap_forward_fraction=fraction,
        snapped_forward=snapped_forward,
        quantized_tick=quantized_tick,
        signed_error_ticks=signed_error_ticks,
        signed_error_ms=signed_error_ticks * tempo.seconds_per_tick * 1000.0,
    )


def stamp_user_input_event_at_tick(event: MusicalEvent, *, tick: int) -> MusicalEvent:
    """Copy an input adapter event onto an already-decided timeline tick."""
    return MusicalEvent(
        tick=int(tick),
        pitch=event.pitch,
        event_type=event.event_type,
        velocity=event.velocity,
        channel=event.channel,
        program=event.program,
        is_placeholder=event.is_placeholder,
        source="user",
    )


def build_input_quantization_trace_row(
    *,
    service: str,
    event_sequence: int,
    event: MusicalEvent,
    result: InputQuantizationResult,
    received_time_s: float,
    timeline_start_time_s: float,
    clock_domain: str,
    tempo: Tempo,
) -> dict[str, object]:
    """Build the shared sidecar schema without performing I/O."""
    return {
        "schema_version": 1,
        "record_type": "input_quantization",
        "service": service,
        "event_sequence": int(event_sequence),
        "event_type": event.event_type.value,
        "pitch": int(event.pitch),
        "velocity": int(event.velocity),
        "channel": int(event.channel),
        "adapter_tick": int(event.tick),
        "clock_domain": clock_domain,
        "application_received_time_s": float(received_time_s),
        "timeline_start_time_s": float(timeline_start_time_s),
        "bpm": float(tempo.bpm),
        "ticks_per_beat": int(tempo.ticks_per_beat),
        "seconds_per_tick": float(tempo.seconds_per_tick),
        "elapsed_seconds": result.elapsed_seconds,
        "raw_tick": result.raw_tick,
        "floor_tick": result.floor_tick,
        "tick_phase": result.tick_phase,
        "snap_forward_fraction": result.snap_forward_fraction,
        "snapped_forward": result.snapped_forward,
        "quantized_tick": result.quantized_tick,
        "signed_error_ticks": result.signed_error_ticks,
        "signed_error_ms": result.signed_error_ms,
    }


def stamp_user_input_event(
    event: MusicalEvent,
    *,
    elapsed_seconds: float,
    tempo: Tempo,
    snap_forward_fraction: float,
) -> MusicalEvent:
    """Copy an input adapter event into a user event stamped on the timeline."""
    return stamp_user_input_event_at_tick(
        event,
        tick=seconds_to_input_tick(
            elapsed_seconds,
            tempo,
            snap_forward_fraction=snap_forward_fraction,
        ),
    )
