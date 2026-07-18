from __future__ import annotations

from unittest.mock import MagicMock, patch

from streammuse.application.config import (
    ApplicationConfig,
    InferenceConfig,
    OutputConfig,
    RapConfig,
)
from streammuse.application.runtime import RuntimeSession, RuntimeSessionBuilder
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
        inference=InferenceConfig(generation_interval_ticks=4),
    )
    input_factory.create.return_value = MagicMock()
    inference_factory.create.return_value = MagicMock()
    service = MagicMock(running=False)
    service_cls.return_value = service

    session = RuntimeSessionBuilder(config=config, log_dir=str(tmp_path)).build_web()

    assert session.service is service
    assert isinstance(session.websocket_sink, WebSocketOutputSink)
    assert service_cls.call_args.kwargs["generation_interval_ticks"] == 4


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
        continuation_mode="prompt_continuation",
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
