"""Output adapters implementing OutputSink (audio, MIDI file, console, WebSocket)."""

from streammuse.infrastructure.output.audio import AudioOutputConfig, AudioOutputSink
from streammuse.infrastructure.output.composite import CompositeOutputSink
from streammuse.infrastructure.output.console import ConsoleOutputConfig, ConsoleOutputSink
from streammuse.infrastructure.output.json_logger import JsonLoggerOutputSink
from streammuse.infrastructure.output.midi_file import MidiFileOutputConfig, MidiFileOutputSink
from streammuse.infrastructure.output.session_logger import SessionLoggerOutputSink
from streammuse.infrastructure.output.websocket import WebSocketOutputConfig, WebSocketOutputSink

__all__ = [
    "AudioOutputConfig",
    "AudioOutputSink",
    "MidiFileOutputConfig",
    "MidiFileOutputSink",
    "ConsoleOutputConfig",
    "ConsoleOutputSink",
    "WebSocketOutputConfig",
    "WebSocketOutputSink",
    "CompositeOutputSink",
    "JsonLoggerOutputSink",
    "SessionLoggerOutputSink",
]
