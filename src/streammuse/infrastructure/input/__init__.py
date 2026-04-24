"""Input adapters implementing InputSource (MIDI, keyboard, file, queue)."""

from streammuse.infrastructure.input.list_input import ListInput
from streammuse.infrastructure.input.keyboard import KeyboardInput
from streammuse.infrastructure.input.midi_file import MidiFileInput, MidiFileInputConfig
from streammuse.infrastructure.input.midi_device import MidiDeviceInput
from streammuse.infrastructure.input.queue_input import QueueInput

__all__ = [
    "ListInput",
    "KeyboardInput",
    "MidiFileInput",
    "MidiFileInputConfig",
    "MidiDeviceInput",
    "QueueInput",
]
