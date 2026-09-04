from __future__ import annotations

from unittest.mock import MagicMock, patch

from streammuse.application.config import (
    ApplicationConfig,
    InferenceConfig,
    OutputConfig,
    RapConfig,
    TempoConfig,
)
from streammuse.application.runtime import RuntimeSession, RuntimeSessionBuilder
from streammuse.infrastructure.output.composite import CompositeOutputSink
from streammuse.infrastructure.output.metronome import MetronomeOutputSink
from streammuse.infrastructure.output.midi_file import MidiFileOutputSink
from streammuse.infrastructure.output.session_logger import SessionLoggerOutputSink
from streammuse.infrastructure.output.websocket import WebSocketOutputSink


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
@patch("streammuse.application.runtime.builder.OutputSinkFactory")
def test_builder_creates_standard_runtime_session(
    output_factory,
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        output=OutputConfig(type="console"),
        inference=InferenceConfig(generation_interval_ticks=8),
    )
    output = MagicMock()
    input_source = MagicMock()
    engine = MagicMock()
    service = MagicMock(running=False)
    output_factory.create.return_value = output
    input_factory.create.return_value = input_source
    inference_factory.create.return_value = engine
    service_cls.return_value = service

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_cli()

    assert session.service is service
    assert session.output_sink is output
    service_cls.assert_called_once()
    assert service_cls.call_args.kwargs["generation_interval_ticks"] == 8


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
def test_builder_creates_web_standard_runtime_session(
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        output=OutputConfig(close_active_notes_on_finalize=False),
        inference=InferenceConfig(generation_interval_ticks=4),
    )
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service = MagicMock(running=False)
    service_cls.return_value = service

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_web()

    assert session.service is service
    assert isinstance(session.websocket_sink, WebSocketOutputSink)
    midi_sink = next(
        sink for sink in session.output_sink.sinks if isinstance(sink, MidiFileOutputSink)
    )
    assert midi_sink._config.close_active_notes_on_finalize is False
    assert session.session_config["close_active_notes_on_finalize"] is False
    assert service_cls.call_args.kwargs["generation_interval_ticks"] == 4


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
def test_builder_web_composite_honors_metronome_and_midi_recording_config(
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=90.0, ticks_per_beat=6, beats_per_bar=3),
        output=OutputConfig(
            metronome_enabled=True,
            metronome_port="click-port",
            metronome_channel=8,
        ),
    )
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service_cls.return_value = MagicMock(running=False)

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_web()

    midi_sink = next(
        sink for sink in session.output_sink.sinks if isinstance(sink, MidiFileOutputSink)
    )
    metronome_sink = next(
        sink for sink in session.output_sink.sinks if isinstance(sink, MetronomeOutputSink)
    )
    assert midi_sink._config.beats_per_bar == 3
    assert midi_sink._config.record_metronome is True
    assert metronome_sink._config.port_name == "click-port"
    assert metronome_sink._config.ticks_per_beat == 6
    assert metronome_sink._config.beats_per_bar == 3
    assert metronome_sink._config.channel == 8


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
def test_builder_creates_unique_web_session_directory_on_fast_restart(
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service_cls.return_value = MagicMock(running=False)
    builder = RuntimeSessionBuilder(config=ApplicationConfig(), log_dir=str(tmp_path))

    first = builder.build_web()
    second = builder.build_web()

    assert first.session_dir != second.session_dir
    assert first.session_dir.exists()
    assert second.session_dir.exists()


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
def test_builder_adds_trace_only_session_logger_to_web_when_enabled(
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    config = ApplicationConfig(input_quantization_trace_enabled=True)
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service_cls.return_value = MagicMock(running=False)

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_web()

    assert isinstance(session.output_sink, CompositeOutputSink)
    session_sinks = [
        sink
        for sink in session.output_sink.sinks
        if isinstance(sink, SessionLoggerOutputSink)
    ]
    midi_sinks = [
        sink
        for sink in session.output_sink.sinks
        if isinstance(sink, MidiFileOutputSink)
    ]
    assert len(session_sinks) == 1
    assert len(midi_sinks) == 1
    assert session_sinks[0].midi_sink is None
    assert session_sinks[0].json_sink is None
    assert service_cls.call_args.kwargs["input_quantization_trace_enabled"] is True


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
def test_builder_reuses_cli_session_logger_for_quantization_trace(
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        output=OutputConfig(type="session"),
        input_quantization_trace_enabled=True,
    )
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service_cls.return_value = MagicMock(running=False)

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_cli()

    assert isinstance(session.output_sink, SessionLoggerOutputSink)
    assert service_cls.call_args.kwargs["input_quantization_trace_enabled"] is True


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
@patch("streammuse.application.runtime.builder.OutputSinkFactory")
def test_builder_preserves_standard_rap_and_session_contract(
    output_factory,
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    controller = MagicMock()
    config = ApplicationConfig(
        output=OutputConfig(type="console", session_artifact_tier="debug"),
        inference=InferenceConfig(
            generation_interval_ticks=8,
            generation_length_frames=24,
        ),
        rap=RapConfig(topic="space travel"),
        count_in_beats=4,
        input_snap_forward_fraction=0.25,
    )
    output_factory.create.return_value = MagicMock()
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service_cls.return_value = MagicMock(running=False)

    session = RuntimeSessionBuilder(
        config=config,
        log_dir=str(tmp_path),
        tick_observer_factory=lambda _tempo: controller,
    ).build_cli()

    kwargs = service_cls.call_args.kwargs
    assert kwargs["generation_interval_ticks"] == 8
    assert kwargs["generation_length_frames"] == 24
    assert kwargs["count_in_beats"] == 4
    assert kwargs["input_snap_forward_fraction"] == 0.25
    assert kwargs["tick_observer"] is controller
    assert session.session_config["session_artifact_tier"] == "debug"
    assert session.session_config["rap_enabled"] is True
    assert session.session_config["continuation_mode"] == "standard"


def test_runtime_session_forwards_standard_run_horizons() -> None:
    service = MagicMock(running=False)
    output = MagicMock()
    session = RuntimeSession(
        config=ApplicationConfig(),
        session_manager=None,
        output_sink=output,
        service=service,
        session_config={"mode": "standard"},
        emit_output_config=False,
    )

    session.start(
        run_stop_tick=64,
        analysis_end_tick=48,
        last_input_note_off_tick=44,
        request_cutoff_tick=40,
        drain_timeout_seconds=3.5,
    )

    output.output_config.assert_not_called()
    service.start.assert_called_once_with(
        run_stop_tick=64,
        analysis_end_tick=48,
        last_input_note_off_tick=44,
        request_cutoff_tick=40,
        drain_timeout_seconds=3.5,
    )


def test_runtime_session_maps_prompt_run_stop_to_max_ticks() -> None:
    service = MagicMock(running=False)
    session = RuntimeSession(
        config=ApplicationConfig(continuation_mode="prompt_continuation"),
        session_manager=None,
        output_sink=MagicMock(),
        service=service,
        session_config={},
        emit_output_config=False,
    )

    session.start(run_stop_tick=64)

    service.start.assert_called_once_with(max_ticks=64)


def test_runtime_session_stop_calls_idempotent_service_stop_after_natural_end() -> None:
    service = MagicMock(running=False)
    session = RuntimeSession(
        config=ApplicationConfig(),
        session_manager=None,
        output_sink=MagicMock(),
        service=service,
        session_config={},
        emit_output_config=False,
    )

    session.stop()

    service.stop.assert_called_once_with()


@patch("streammuse.application.runtime.builder.InputSourceFactory")
def test_builder_creates_prompt_continuation_runtime_with_override(
    input_factory,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=90.6),
        continuation_mode="prompt_continuation",
        count_in_beats=4,
        input_snap_forward_fraction=0.25,
        inference=InferenceConfig(
            prompt_length_ticks=64,
            generation_interval_ticks=4,
        ),
    )
    prompt_client = MagicMock()
    output = MagicMock()
    service_cls = MagicMock()
    service = MagicMock(running=False)
    service_cls.return_value = service
    input_factory.create.return_value = MagicMock()
    builder = RuntimeSessionBuilder(
        config=config,
        log_dir=str(tmp_path),
        prompt_client_override=prompt_client,
        output_sink_override=output,
    )

    with patch.object(
        builder,
        "_prompt_continuation_service_cls",
        return_value=service_cls,
    ):
        session = builder.build_cli()

    assert session.service is service
    assert session.prompt_client is prompt_client
    assert session.inference_engine is None
    assert service_cls.call_args.kwargs["prompt_length_ticks"] == 64
    assert service_cls.call_args.kwargs["generation_interval_ticks"] == 4
    assert service_cls.call_args.kwargs["count_in_beats"] == 4
    assert service_cls.call_args.kwargs["input_snap_forward_fraction"] == 0.25
    assert service_cls.call_args.kwargs["model_condition_bpm"] == 91
    assert session.session_config["effective_model_bpm"] == 91


@patch("streammuse.application.runtime.builder.InputSourceFactory")
def test_prompt_runtime_uses_explicit_model_condition_bpm_override(
    input_factory,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=80.0),
        continuation_mode="prompt_continuation",
        inference=InferenceConfig(model_condition_bpm=137),
    )
    service_cls = MagicMock(return_value=MagicMock(running=False))
    input_factory.create.return_value = MagicMock()
    builder = RuntimeSessionBuilder(
        config=config,
        log_dir=str(tmp_path),
        prompt_client_override=MagicMock(),
        output_sink_override=MagicMock(),
    )

    with patch.object(
        builder,
        "_prompt_continuation_service_cls",
        return_value=service_cls,
    ):
        session = builder.build_cli()

    assert service_cls.call_args.kwargs["model_condition_bpm"] == 137
    assert session.session_config["tempo_bpm"] == 80.0
    assert session.session_config["model_condition_bpm"] == 137
    assert session.session_config["effective_model_bpm"] == 137


@patch("streammuse.application.runtime.builder.RealTimeMusicService")
@patch("streammuse.application.runtime.builder.InputSourceFactory")
@patch("streammuse.application.runtime.builder.InferenceEngineFactory")
@patch("streammuse.application.runtime.builder.OutputSinkFactory")
def test_builder_allows_eight_steps_per_beat_for_standard_runtime(
    output_factory,
    inference_factory,
    input_factory,
    service_cls,
    tmp_path,
) -> None:
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=120.0, ticks_per_beat=8, beats_per_bar=4),
        output=OutputConfig(type="console"),
    )
    output_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    input_factory.create.return_value = MagicMock()
    service_cls.return_value = MagicMock(running=False)

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_cli()

    assert session.service is service_cls.return_value
    assert service_cls.call_args.kwargs["tempo"].ticks_per_beat == 8


def test_runtime_session_cleanup_is_idempotent_and_closes_output_after_clear_error() -> None:
    engine = MagicMock()
    engine.clear_history.side_effect = RuntimeError("server unavailable")
    output = MagicMock()
    session = RuntimeSession(
        config=ApplicationConfig(),
        session_manager=None,
        output_sink=output,
        service=MagicMock(running=False),
        session_config={},
        inference_engine=engine,
        emit_output_config=False,
    )

    session.cleanup()
    session.cleanup()

    engine.clear_history.assert_called_once_with()
    output.close.assert_called_once_with()
