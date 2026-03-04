"""Input adapters implementing InputSource (MIDI, keyboard, file)."""

from streammuse.infrastructure.input.list_input import ListInput
from streammuse.infrastructure.input.keyboard import KeyboardInput
from streammuse.infrastructure.input.midi_file import MidiFileInput, MidiFileInputConfig
from streammuse.infrastructure.input.midi_device import MidiDeviceInput

__all__ = [
    "ListInput",
    "KeyboardInput",
    "MidiFileInput",
    "MidiFileInputConfig",
    "MidiDeviceInput",
]
