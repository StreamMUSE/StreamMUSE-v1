from __future__ import annotations

from unittest.mock import MagicMock, patch

from streammuse.application.config import ApplicationConfig, InferenceConfig, OutputConfig
from streammuse.application.runtime import RuntimeSessionBuilder
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
