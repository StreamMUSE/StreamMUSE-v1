"""OutputSink factory."""

from __future__ import annotations

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
    def _attach_auto_midi_if_needed(
        *,
        base_sink: OutputSink,
        app_config: ApplicationConfig,
        session_manager: Optional[SessionManager],
    ) -> OutputSink:
        if session_manager is None:
            return base_sink

        tempo = app_config.tempo
        auto_midi_sink = MidiFileOutputSink(
            MidiFileOutputConfig(
                bpm=float(tempo.bpm),
                ticks_per_beat=int(tempo.ticks_per_beat),
                output_path=str(session_manager.get_session_dir() / "combined.mid"),
            )
        )
        return CompositeOutputSink([base_sink, auto_midi_sink])

    @staticmethod
    def create(
        app_config: ApplicationConfig,
        session_manager: Optional[SessionManager] = None,
    ) -> OutputSink:
        cfg = app_config.output
        tempo = app_config.tempo

        if cfg.type == "console":
            return OutputSinkFactory._attach_auto_midi_if_needed(
                base_sink=ConsoleOutputSink(ConsoleOutputConfig()),
                app_config=app_config,
                session_manager=session_manager,
            )

        if cfg.type == "audio":
            return OutputSinkFactory._attach_auto_midi_if_needed(
                base_sink=AudioOutputSink(AudioOutputConfig(port_name=cfg.midi_out_port)),
                app_config=app_config,
                session_manager=session_manager,
            )

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
            return OutputSinkFactory._attach_auto_midi_if_needed(
                base_sink=WebSocketOutputSink(),
                app_config=app_config,
                session_manager=session_manager,
            )

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
                bpm=float(tempo.bpm),
                ticks_per_beat=int(tempo.ticks_per_beat),
            )

        if cfg.type == "composite":
            if session_manager:
                return CompositeOutputSink(
                    [
                        ConsoleOutputSink(ConsoleOutputConfig()),
                        SessionLoggerOutputSink(
                            session_manager.get_session_dir(),
                            inference_log_detail=cfg.inference_log_detail,
                            bpm=float(tempo.bpm),
                            ticks_per_beat=int(tempo.ticks_per_beat),
                        ),
                    ]
                )
            return CompositeOutputSink(
                [ConsoleOutputSink(ConsoleOutputConfig()), WebSocketOutputSink()]
            )

        raise ValueError(f"Unknown output type: {cfg.type}")

