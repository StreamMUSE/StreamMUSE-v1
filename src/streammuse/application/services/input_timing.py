"""Shared input timing policy for realtime human performance."""

from __future__ import annotations

import math

from streammuse.domain.musical import MusicalEvent
from streammuse.domain.timing import Tempo


REALTIME_INPUT_TYPES = frozenset({"midi_device", "keyboard", "queue"})


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
    seconds = max(0.0, float(elapsed_seconds))
    raw_tick = seconds / tempo.seconds_per_tick
    base_tick = int(math.floor(raw_tick))
    fraction = clamp_snap_forward_fraction(snap_forward_fraction)
    if fraction <= 0.0:
        return base_tick

    tick_phase = raw_tick - base_tick
    if tick_phase >= (1.0 - fraction):
        return base_tick + 1
    return base_tick


def stamp_user_input_event(
    event: MusicalEvent,
    *,
    elapsed_seconds: float,
    tempo: Tempo,
    snap_forward_fraction: float,
) -> MusicalEvent:
    """Copy an input adapter event into a user event stamped on the timeline."""
    return MusicalEvent(
        tick=seconds_to_input_tick(
            elapsed_seconds,
            tempo,
            snap_forward_fraction=snap_forward_fraction,
        ),
        pitch=event.pitch,
        event_type=event.event_type,
        velocity=event.velocity,
        channel=event.channel,
        program=event.program,
        is_placeholder=event.is_placeholder,
        source="user",
    )
