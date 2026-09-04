from __future__ import annotations

import math

import pytest

from streammuse.application.services.input_timing import (
    diagnose_input_quantization,
    effective_input_snap_forward_fraction,
    seconds_to_input_tick,
    stamp_user_input_event,
)
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import Tempo


def _legacy_seconds_to_input_tick(
    elapsed_seconds: float,
    tempo: Tempo,
    *,
    snap_forward_fraction: float,
) -> int:
    seconds = max(0.0, float(elapsed_seconds))
    raw_tick = seconds / tempo.seconds_per_tick
    base_tick = int(math.floor(raw_tick))
    fraction = min(1.0, max(0.0, float(snap_forward_fraction)))
    if fraction <= 0.0:
        return base_tick
    tick_phase = raw_tick - base_tick
    if tick_phase >= (1.0 - fraction):
        return base_tick + 1
    return base_tick


def test_seconds_to_input_tick_snaps_last_fraction_to_next_tick() -> None:
    tempo = Tempo(bpm=60.0, ticks_per_beat=4, beats_per_bar=4)

    assert seconds_to_input_tick(0.149, tempo, snap_forward_fraction=0.4) == 0
    assert seconds_to_input_tick(0.150, tempo, snap_forward_fraction=0.4) == 1


def test_seconds_to_input_tick_preserves_floor_when_tolerance_disabled() -> None:
    tempo = Tempo(bpm=60.0, ticks_per_beat=4, beats_per_bar=4)

    assert seconds_to_input_tick(0.249, tempo, snap_forward_fraction=0.0) == 0


def test_diagnostic_preserves_legacy_ticks_at_boundaries() -> None:
    tempo = Tempo(bpm=60.0, ticks_per_beat=4, beats_per_bar=4)
    cases = [
        (-0.1, 0.4),
        (0.0, 0.0),
        (0.149999, 0.4),
        (0.150, 0.4),
        (0.249999, 0.4),
        (0.250, 0.4),
        (0.375, -1.0),
        (0.500, 1.0),
        (0.875, 2.0),
    ]

    for elapsed_seconds, fraction in cases:
        expected = _legacy_seconds_to_input_tick(
            elapsed_seconds,
            tempo,
            snap_forward_fraction=fraction,
        )
        result = diagnose_input_quantization(
            elapsed_seconds,
            tempo,
            snap_forward_fraction=fraction,
        )
        assert result.quantized_tick == expected
        assert seconds_to_input_tick(
            elapsed_seconds,
            tempo,
            snap_forward_fraction=fraction,
        ) == expected


def test_diagnostic_reports_signed_tick_and_millisecond_error() -> None:
    tempo = Tempo(bpm=60.0, ticks_per_beat=4, beats_per_bar=4)

    result = diagnose_input_quantization(
        0.150,
        tempo,
        snap_forward_fraction=0.4,
    )

    assert result.raw_tick == pytest.approx(0.6)
    assert result.floor_tick == 0
    assert result.tick_phase == pytest.approx(0.6)
    assert result.snapped_forward is True
    assert result.quantized_tick == 1
    assert result.signed_error_ticks == pytest.approx(0.4)
    assert result.signed_error_ms == pytest.approx(100.0)


def test_stamp_user_input_event_copies_payload_and_assigns_user_source() -> None:
    tempo = Tempo(bpm=60.0, ticks_per_beat=4, beats_per_bar=4)
    event = MusicalEvent(
        tick=99,
        pitch=64,
        event_type=EventType.NOTE_ON,
        velocity=87,
        channel=2,
        program=5,
        is_placeholder=False,
        source="model",
    )

    stamped = stamp_user_input_event(
        event,
        elapsed_seconds=0.150,
        tempo=tempo,
        snap_forward_fraction=0.4,
    )

    assert stamped.tick == 1
    assert stamped.pitch == 64
    assert stamped.event_type == EventType.NOTE_ON
    assert stamped.velocity == 87
    assert stamped.channel == 2
    assert stamped.program == 5
    assert stamped.is_placeholder is False
    assert stamped.source == "user"


def test_effective_snap_forward_only_applies_to_realtime_inputs() -> None:
    assert effective_input_snap_forward_fraction("midi_device", 0.4) == 0.4
    assert effective_input_snap_forward_fraction("keyboard", 0.4) == 0.4
    assert effective_input_snap_forward_fraction("midi_file", 0.4) == 0.0
    assert effective_input_snap_forward_fraction("list", 0.4) == 0.0
