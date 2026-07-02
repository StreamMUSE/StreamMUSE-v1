from __future__ import annotations

from streammuse.application.services.input_timing import (
    effective_input_snap_forward_fraction,
    seconds_to_input_tick,
    stamp_user_input_event,
)
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import Tempo


def test_seconds_to_input_tick_snaps_last_fraction_to_next_tick() -> None:
    tempo = Tempo(bpm=60.0, ticks_per_beat=4, beats_per_bar=4)

    assert seconds_to_input_tick(0.149, tempo, snap_forward_fraction=0.4) == 0
    assert seconds_to_input_tick(0.150, tempo, snap_forward_fraction=0.4) == 1


def test_seconds_to_input_tick_preserves_floor_when_tolerance_disabled() -> None:
    tempo = Tempo(bpm=60.0, ticks_per_beat=4, beats_per_bar=4)

    assert seconds_to_input_tick(0.249, tempo, snap_forward_fraction=0.0) == 0


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
