"""Tests for MidiDeviceInput adapter (with mock port)."""

import pytest

from streammuse.domain.musical import MusicalEvent, EventType
from streammuse.infrastructure.input.midi_device import MidiDeviceInput


def test_midi_device_input_close_before_open():
    """close() is safe to call when port was never opened."""
    source = MidiDeviceInput(device_name=None)
    # Don't call read_events() so port is never opened
    source.close()
    # No exception
