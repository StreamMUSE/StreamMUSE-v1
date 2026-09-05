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
    MetronomeOutputConfig,
    MetronomeOutputSink,
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
                beats_per_bar=int(tempo.beats_per_bar),
                output_path=str(session_manager.get_session_dir() / "combined.mid"),
                close_active_notes_on_finalize=bool(
                    app_config.output.close_active_notes_on_finalize
                ),
                record_metronome=bool(app_config.output.metronome_enabled),
            )
        )
        return CompositeOutputSink([base_sink, auto_midi_sink])

    @staticmethod
    def _attach_metronome_if_needed(
        *,
        base_sink: OutputSink,
        app_config: ApplicationConfig,
    ) -> OutputSink:
        cfg = app_config.output
        if not cfg.metronome_enabled:
            return base_sink

        tempo = app_config.tempo
        metronome_sink = MetronomeOutputSink(
            MetronomeOutputConfig(
                port_name=cfg.metronome_port or cfg.midi_out_port,
                ticks_per_beat=int(tempo.ticks_per_beat),
                beats_per_bar=int(tempo.beats_per_bar),
                channel=int(cfg.metronome_channel),
            )
        )

        if isinstance(base_sink, CompositeOutputSink):
            return CompositeOutputSink([*base_sink.sinks, metronome_sink])
        return CompositeOutputSink([base_sink, metronome_sink])

    @staticmethod
    def create(
        app_config: ApplicationConfig,
        session_manager: Optional[SessionManager] = None,
    ) -> OutputSink:
        cfg = app_config.output
        tempo = app_config.tempo

        if cfg.type == "console":
            sink = OutputSinkFactory._attach_auto_midi_if_needed(
                base_sink=ConsoleOutputSink(ConsoleOutputConfig()),
                app_config=app_config,
                session_manager=session_manager,
            )
            return OutputSinkFactory._attach_metronome_if_needed(base_sink=sink, app_config=app_config)

        if cfg.type == "audio":
            sink = OutputSinkFactory._attach_auto_midi_if_needed(
                base_sink=AudioOutputSink(
                    AudioOutputConfig(
                        port_name=cfg.midi_out_port,
                        mute_melody_output=cfg.mute_melody_output,
                    )
                ),
                app_config=app_config,
                session_manager=session_manager,
            )
            return OutputSinkFactory._attach_metronome_if_needed(base_sink=sink, app_config=app_config)

        if cfg.type == "midi_file":
            if not cfg.midi_file_output_path:
                raise ValueError("midi_file_output_path is required for midi_file output")
            sink = MidiFileOutputSink(
                MidiFileOutputConfig(
                    bpm=float(tempo.bpm),
                    ticks_per_beat=int(tempo.ticks_per_beat),
                    beats_per_bar=int(tempo.beats_per_bar),
                    output_path=cfg.midi_file_output_path,
                    close_active_notes_on_finalize=bool(
                        cfg.close_active_notes_on_finalize
                    ),
                    record_metronome=bool(cfg.metronome_enabled),
                )
            )
            return OutputSinkFactory._attach_metronome_if_needed(base_sink=sink, app_config=app_config)

        if cfg.type == "websocket":
            sink = OutputSinkFactory._attach_auto_midi_if_needed(
                base_sink=WebSocketOutputSink(),
                app_config=app_config,
                session_manager=session_manager,
            )
            return OutputSinkFactory._attach_metronome_if_needed(base_sink=sink, app_config=app_config)

        if cfg.type == "json_log":
            if not session_manager:
                raise ValueError("session_manager is required for json_log output")
            sink = JsonLoggerOutputSink(
                session_manager.get_session_dir(),
                inference_log_detail=cfg.inference_log_detail,
            )
            return OutputSinkFactory._attach_metronome_if_needed(base_sink=sink, app_config=app_config)

        if cfg.type == "session":
            if not session_manager:
                raise ValueError("session_manager is required for session output")
            sink = SessionLoggerOutputSink(
                session_dir=session_manager.get_session_dir(),
                include_midi=True,
                include_json=(cfg.session_artifact_tier == "debug"),
                inference_log_detail=cfg.inference_log_detail,
                bpm=float(tempo.bpm),
                ticks_per_beat=int(tempo.ticks_per_beat),
                beats_per_bar=int(tempo.beats_per_bar),
                record_metronome=bool(cfg.metronome_enabled),
                close_active_notes_on_finalize=bool(
                    cfg.close_active_notes_on_finalize
                ),
                artifact_tier=cfg.session_artifact_tier,
            )
            return OutputSinkFactory._attach_metronome_if_needed(base_sink=sink, app_config=app_config)

        if cfg.type == "composite":
            if session_manager:
                sink = CompositeOutputSink(
                    [
                        ConsoleOutputSink(ConsoleOutputConfig()),
                        SessionLoggerOutputSink(
                            session_manager.get_session_dir(),
                            include_json=(cfg.session_artifact_tier == "debug"),
                            inference_log_detail=cfg.inference_log_detail,
                            bpm=float(tempo.bpm),
                            ticks_per_beat=int(tempo.ticks_per_beat),
                            beats_per_bar=int(tempo.beats_per_bar),
                            record_metronome=bool(cfg.metronome_enabled),
                            close_active_notes_on_finalize=bool(
                                cfg.close_active_notes_on_finalize
                            ),
                            artifact_tier=cfg.session_artifact_tier,
                        ),
                    ]
                )
                return OutputSinkFactory._attach_metronome_if_needed(base_sink=sink, app_config=app_config)
            sink = CompositeOutputSink([ConsoleOutputSink(ConsoleOutputConfig()), WebSocketOutputSink()])
            return OutputSinkFactory._attach_metronome_if_needed(base_sink=sink, app_config=app_config)

        raise ValueError(f"Unknown output type: {cfg.type}")
