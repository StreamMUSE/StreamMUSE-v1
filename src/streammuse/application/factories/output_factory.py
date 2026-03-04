"""OutputSink factory."""

from __future__ import annotations

from streammuse.application.config import ApplicationConfig
from streammuse.domain.interfaces import OutputSink
from streammuse.infrastructure.output import (
    AudioOutputConfig,
    AudioOutputSink,
    CompositeOutputSink,
    ConsoleOutputConfig,
    ConsoleOutputSink,
    MidiFileOutputConfig,
    MidiFileOutputSink,
    WebSocketOutputSink,
)


class OutputSinkFactory:
    @staticmethod
    def create(app_config: ApplicationConfig) -> OutputSink:
        cfg = app_config.output
        tempo = app_config.tempo

        if cfg.type == "console":
            return ConsoleOutputSink(ConsoleOutputConfig())

        if cfg.type == "audio":
            return AudioOutputSink(AudioOutputConfig(port_name=cfg.midi_out_port))

        if cfg.type == "midi_file":
            if not cfg.midi_file_output_path:
                raise ValueError("midi_file_output_path is required for midi_file output")
            return MidiFileOutputSink(
                MidiFileOutputConfig(
                    bpm=float(tempo.bpm),
                    ticks_per_beat=int(tempo.ticks_per_beat),
                    output_path=cfg.midi_file_output_path,
                )
            )

        if cfg.type == "websocket":
            return WebSocketOutputSink()

        if cfg.type == "composite":
            # Default composite: console + websocket.
            return CompositeOutputSink([ConsoleOutputSink(ConsoleOutputConfig()), WebSocketOutputSink()])

        raise ValueError(f"Unknown output type: {cfg.type}")

