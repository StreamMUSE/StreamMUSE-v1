"""MIDI device input adapter implementing InputSource."""

from __future__ import annotations

from typing import Iterator

import mido

from streammuse.domain.musical import MusicalEvent, EventType


class MidiDeviceInput:
    """
    InputSource that reads note_on/note_off from a MIDI device.

    Yields MusicalEvent with tick=0; the application layer should set
    event.tick from timestamp (tempo.seconds_to_tick(elapsed)) per TIMING_DESIGN.md.
    """

    def __init__(self, device_name: str | None = None) -> None:
        self._device_name = device_name
        self._port: mido.ports.BaseInput | None = None
        self._running = False

    def read_events(self) -> Iterator[MusicalEvent]:
        """Yield note_on/note_off events from the MIDI port. Blocks when idle."""
        if self._port is None:
            self._port = mido.open_input(self._device_name)
        self._running = True
        try:
            while self._running and self._port is not None:
                msg = self._port.poll()
                if msg is None:
                    continue
                if msg.type == "note_on" and msg.velocity > 0:
                    yield MusicalEvent(
                        tick=0,
                        pitch=msg.note,
                        event_type=EventType.NOTE_ON,
                        velocity=msg.velocity,
                    )
                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    yield MusicalEvent(
                        tick=0,
                        pitch=msg.note,
                        event_type=EventType.NOTE_OFF,
                        velocity=0,
                    )
        finally:
            self._running = False

    def close(self) -> None:
        """Stop the iterator and close the MIDI port."""
        self._running = False
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None
