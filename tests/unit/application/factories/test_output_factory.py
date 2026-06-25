from __future__ import annotations

import pytest

from streammuse.application.config import ApplicationConfig, InferenceConfig, OutputConfig, TempoConfig
from streammuse.application.factories.output_factory import OutputSinkFactory
from streammuse.domain.logging import SessionManager
from streammuse.infrastructure.output import (
    AudioOutputSink,
    CompositeOutputSink,
    ConsoleOutputSink,
    JsonLoggerOutputSink,
    MetronomeOutputSink,
    MidiFileOutputSink,
    SessionLoggerOutputSink,
)


def _make_config(output_type: str, bpm: float = 120.0, ticks_per_beat: int = 4) -> ApplicationConfig:
    return ApplicationConfig(
        tempo=TempoConfig(bpm=bpm, ticks_per_beat=ticks_per_beat, beats_per_bar=4),
        output=OutputConfig(type=output_type),  # type: ignore[arg-type]
    )


def test_output_factory_console_with_session_manager_attaches_auto_midi(tmp_path):
    config = _make_config("console", bpm=96.0, ticks_per_beat=8)
    session_manager = SessionManager(base_log_dir=str(tmp_path))
    session_dir = session_manager.create_session_directory()

    sink = OutputSinkFactory.create(config, session_manager=session_manager)
    assert isinstance(sink, CompositeOutputSink)
    assert len(sink.sinks) == 2
    assert isinstance(sink.sinks[0], ConsoleOutputSink)
    assert isinstance(sink.sinks[1], MidiFileOutputSink)
    assert sink.sinks[1]._config.bpm == 96.0
    assert sink.sinks[1]._config.ticks_per_beat == 8
    assert sink.sinks[1]._config.output_path == str(session_dir / "combined.mid")
    sink.close()


def test_output_factory_audio_with_session_manager_attaches_auto_midi(tmp_path):
    config = _make_config("audio", bpm=110.0, ticks_per_beat=4)
    session_manager = SessionManager(base_log_dir=str(tmp_path))
    session_dir = session_manager.create_session_directory()

    sink = OutputSinkFactory.create(config, session_manager=session_manager)
    assert isinstance(sink, CompositeOutputSink)
    assert isinstance(sink.sinks[0], AudioOutputSink)
    assert isinstance(sink.sinks[1], MidiFileOutputSink)
    assert sink.sinks[1]._config.output_path == str(session_dir / "combined.mid")
    sink.close()


def test_output_factory_json_log_does_not_attach_midi(tmp_path):
    config = _make_config("json_log")
    session_manager = SessionManager(base_log_dir=str(tmp_path))
    session_manager.create_session_directory()

    sink = OutputSinkFactory.create(config, session_manager=session_manager)
    assert isinstance(sink, JsonLoggerOutputSink)
    sink.close()


def test_output_factory_session_keeps_session_logger_without_double_attach(tmp_path):
    config = _make_config("session", bpm=110.0, ticks_per_beat=8)
    session_manager = SessionManager(base_log_dir=str(tmp_path))
    session_manager.create_session_directory()

    sink = OutputSinkFactory.create(config, session_manager=session_manager)
    assert isinstance(sink, SessionLoggerOutputSink)
    assert sink.midi_sink is not None
    assert sink.midi_sink._config.bpm == 110.0
    assert sink.midi_sink._config.ticks_per_beat == 8
    sink.close()


def test_output_factory_console_without_session_manager_returns_plain_console_sink():
    config = _make_config("console")
    sink = OutputSinkFactory.create(config, session_manager=None)
    assert isinstance(sink, ConsoleOutputSink)


def test_output_factory_attaches_metronome_and_enables_recording(tmp_path):
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=120.0, ticks_per_beat=8, beats_per_bar=3),
        output=OutputConfig(
            type="console",
            metronome_enabled=True,
            metronome_port="Metronome Port",
            metronome_channel=10,
        ),
    )
    session_manager = SessionManager(base_log_dir=str(tmp_path))
    session_manager.create_session_directory()

    sink = OutputSinkFactory.create(config, session_manager=session_manager)
    assert isinstance(sink, CompositeOutputSink)
    assert isinstance(sink.sinks[0], ConsoleOutputSink)
    assert isinstance(sink.sinks[1], MidiFileOutputSink)
    assert isinstance(sink.sinks[2], MetronomeOutputSink)
    assert sink.sinks[1]._config.record_metronome is True
    assert sink.sinks[1]._config.beats_per_bar == 3
    assert sink.sinks[2]._config.port_name == "Metronome Port"
    assert sink.sinks[2]._config.ticks_per_beat == 8
    assert sink.sinks[2]._config.beats_per_bar == 3
    assert sink.sinks[2]._config.channel == 10
    sink.close()


@pytest.mark.parametrize("output_type", ["json_log", "composite"])
def test_output_factory_propagates_inference_log_detail(tmp_path, output_type):
    session_manager = SessionManager(base_log_dir=str(tmp_path))
    session_manager.create_session_directory()

    cfg = ApplicationConfig(
        output=OutputConfig(type=output_type, inference_log_detail="full"),
        inference=InferenceConfig(type="http", server_generate_url="http://x/generate_accompaniment"),
    )

    sink = OutputSinkFactory.create(cfg, session_manager=session_manager)
    assert getattr(sink, "inference_log_detail", "summary") == "full"


def test_session_logger_uses_passed_bpm_and_ticks(tmp_path):
    sink = SessionLoggerOutputSink(
        session_dir=tmp_path,
        include_midi=True,
        include_json=False,
        bpm=110.0,
        ticks_per_beat=8,
    )

    assert sink.midi_sink is not None
    assert sink.midi_sink._config.bpm == 110.0
    assert sink.midi_sink._config.ticks_per_beat == 8
    sink.close()
