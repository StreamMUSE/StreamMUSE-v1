"""Shared protocol conformance tests for input adapters."""

from __future__ import annotations

import pytest

from streammuse.infrastructure.input.keyboard import KeyboardInput
from streammuse.infrastructure.input.list_input import ListInput
from streammuse.infrastructure.input.midi_device import MidiDeviceInput


@pytest.mark.parametrize(
    "factory",
    [
        lambda: KeyboardInput(),
        lambda: ListInput([]),
        lambda: MidiDeviceInput(device_name=None),
    ],
)
def test_input_source_protocol_conformance(factory):
    source = factory()
    assert hasattr(source, "read_events")
    assert hasattr(source, "close")
    assert callable(source.read_events)
    assert callable(source.close)
