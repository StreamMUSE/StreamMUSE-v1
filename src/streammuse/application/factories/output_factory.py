"""OutputSink factory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from streammuse.application.config import ApplicationConfig
from streammuse.domain.interfaces import OutputSink
from streammuse.domain.logging import SessionManager
from streammuse.infrastructure.output import (
    AudioOutputConfig,
    AudioOutputSink,
    CompositeOutputSink,
    ConsoleOutputConfig,
    ConsoleOutputSink,
    JsonLoggerOutputSink,
    MidiFileOutputConfig,
    MidiFileOutputSink,
    SessionLoggerOutputSink,
    WebSocketOutputSink,
)


class OutputSinkFactory:
    @staticmethod
    def create(
        app_config: ApplicationConfig,
        session_manager: Optional[SessionManager] = None,
    ) -> OutputSink:
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

        if cfg.type == "json_log":
            if not session_manager:
                raise ValueError("session_manager is required for json_log output")
            return JsonLoggerOutputSink(
                session_manager.get_session_dir(),
                inference_log_detail=cfg.inference_log_detail,
            )

        if cfg.type == "session":
            if not session_manager:
                raise ValueError("session_manager is required for session output")
            return SessionLoggerOutputSink(
                session_dir=session_manager.get_session_dir(),
                include_midi=True,
                include_json=True,
                inference_log_detail=cfg.inference_log_detail,
            )

        if cfg.type == "composite":
            if session_manager:
                return CompositeOutputSink(
                    [
                        ConsoleOutputSink(ConsoleOutputConfig()),
                        SessionLoggerOutputSink(
                            session_manager.get_session_dir(),
                            inference_log_detail=cfg.inference_log_detail,
                        ),
                    ]
                )
            return CompositeOutputSink(
                [ConsoleOutputSink(ConsoleOutputConfig()), WebSocketOutputSink()]
            )

        raise ValueError(f"Unknown output type: {cfg.type}")


