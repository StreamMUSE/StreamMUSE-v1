"""Tests for the StreamMUSE web viewer.

The viewer is read-only — there are no /api/start, /api/stop, or /api/inject
endpoints. These tests cover the two surfaces that do exist:
  - GET / returns the static index.html
  - WS /ws receives broadcast envelopes from the WebSocketOutputSink

To avoid spinning up RealTimeMusicService threads in tests, we set up the
module-level state directly with a fake WebSocketOutputSink.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from streammuse.application.config import ApplicationConfig, InferenceConfig, InputConfig
from streammuse.infrastructure.output.websocket import WebSocketOutputSink
from streammuse.presentation.web import server as webserver


@pytest.fixture(autouse=True)
def reset_module_state():
    """Clear server module state between tests.

    Critical: also reset broadcaster task/event between tests because
    each TestClient run creates its own event loop, and a leftover task
    from a previous test would fail with "future belongs to a different
    loop" when lifespan's finally block tries to join it.
    """
    def _reset():
        webserver._ws_sink = None
        webserver._service = None
        webserver._composite_sink = None
        webserver._connections.clear()
        webserver._broadcaster_task = None
        webserver._broadcaster_stop = None

    _reset()
    yield
    _reset()


def test_index_serves_html():
    app = webserver.create_app()
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "<html" in resp.text.lower()
        assert "piano-canvas" in resp.text


def test_websocket_broadcasts_pending_messages_from_sink():
    """The broadcaster drains _ws_sink.get_pending_messages() and forwards
    each envelope to connected clients."""
    sink = WebSocketOutputSink()
    webserver._ws_sink = sink

    app = webserver.create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # Push a status envelope into the sink; the broadcaster polls at
            # ~20ms intervals so it should arrive shortly.
            sink.output_status("running", "test")
            envelope = ws.receive_json()
            assert envelope["type"] == "status"
            assert envelope["state"] == "running"
            assert envelope["message"] == "test"


def test_websocket_disconnect_does_not_crash_server():
    """Closing the websocket cleanly removes the connection from the pool."""
    sink = WebSocketOutputSink()
    webserver._ws_sink = sink

    app = webserver.create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            sink.output_status("running", "")
            ws.receive_json()
        # After the `with` block exits the WS is closed. The connection list
        # may not be empty immediately (the broadcaster reconciles lazily),
        # but the server must still serve subsequent requests.
        resp = client.get("/")
        assert resp.status_code == 200


def test_websocket_endpoint_works_when_no_sink_attached():
    """No sink → no broadcaster task → WS endpoint still accepts connects.

    This guards the case where a test or unusual boot path doesn't wire a
    sink; the page must not 500.
    """
    webserver._ws_sink = None
    app = webserver.create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # No broadcast traffic because there's no sink — that's fine.
            # The endpoint should remain open until we close it.
            assert ws is not None


@patch("streammuse.presentation.web.server.RealTimeMusicService")
@patch("streammuse.presentation.web.server.InputSourceFactory")
@patch("streammuse.presentation.web.server.InferenceEngineFactory")
def test_build_realtime_service_standard_mode_uses_standard_engine(
    mock_inference_factory,
    mock_input_factory,
    mock_service_cls,
):
    config = ApplicationConfig(inference=InferenceConfig(generation_interval_ticks=8))
    input_source = MagicMock()
    inference_engine = MagicMock()
    output_sink = MagicMock()
    service = MagicMock()
    mock_input_factory.create.return_value = input_source
    mock_inference_factory.create.return_value = inference_engine
    mock_service_cls.return_value = service

    result = webserver._build_realtime_service(config=config, output_sink=output_sink)

    assert result is service
    mock_inference_factory.create.assert_called_once_with(config)
    mock_service_cls.assert_called_once()
    kwargs = mock_service_cls.call_args.kwargs
    assert kwargs["input_source"] is input_source
    assert kwargs["inference_engine"] is inference_engine
    assert kwargs["output_sink"] is output_sink
    assert kwargs["generation_interval_ticks"] == 8
    assert kwargs["input_snap_forward_fraction"] == 0.4


@patch("streammuse.presentation.web.server.PromptContinuationRealtimeService")
@patch("streammuse.presentation.web.server.PromptContinuationHttpClient")
@patch("streammuse.presentation.web.server.InputSourceFactory")
@patch("streammuse.presentation.web.server.InferenceEngineFactory")
def test_build_realtime_service_prompt_continuation_uses_prompt_client(
    mock_inference_factory,
    mock_input_factory,
    mock_client_cls,
    mock_service_cls,
):
    config = ApplicationConfig(
        inference=InferenceConfig(
            server_generate_url="http://x/generate_accompaniment",
            timeout_s=12.5,
            inference_mode="batch",
            checkpoint_path="/tmp/checkpoint.pt",
            prompt_length_ticks=64,
            generation_interval_ticks=8,
        ),
        continuation_mode="prompt_continuation",
    )
    input_source = MagicMock()
    prompt_client = MagicMock()
    output_sink = MagicMock()
    service = MagicMock()
    mock_input_factory.create.return_value = input_source
    mock_client_cls.return_value = prompt_client
    mock_service_cls.return_value = service

    result = webserver._build_realtime_service(config=config, output_sink=output_sink)

    assert result is service
    mock_inference_factory.create.assert_not_called()
    mock_client_cls.assert_called_once()
    client_config = mock_client_cls.call_args.args[0]
    assert client_config.base_url == "http://x/generate_accompaniment"
    assert client_config.timeout_s == 12.5
    assert client_config.model_name == "lekai_prompt_continuation"
    assert client_config.inference_mode == "batch"
    assert client_config.checkpoint_path == "/tmp/checkpoint.pt"
    mock_service_cls.assert_called_once()
    kwargs = mock_service_cls.call_args.kwargs
    assert kwargs["input_source"] is input_source
    assert kwargs["prompt_client"] is prompt_client
    assert kwargs["output_sink"] is output_sink
    assert kwargs["prompt_length_ticks"] == 64
    assert kwargs["generation_interval_ticks"] == 8
    assert kwargs["input_snap_forward_fraction"] == 0.4


@patch("streammuse.presentation.web.server.RealTimeMusicService")
@patch("streammuse.presentation.web.server.InputSourceFactory")
@patch("streammuse.presentation.web.server.InferenceEngineFactory")
def test_build_realtime_service_disables_snap_forward_for_midi_file(
    mock_inference_factory,
    mock_input_factory,
    mock_service_cls,
):
    config = ApplicationConfig(
        input=InputConfig(type="midi_file", midi_file_path="/tmp/song.mid"),
        input_snap_forward_fraction=0.4,
    )
    mock_input_factory.create.return_value = MagicMock()
    mock_inference_factory.create.return_value = MagicMock()
    mock_service_cls.return_value = MagicMock()

    webserver._build_realtime_service(config=config, output_sink=MagicMock())

    assert mock_service_cls.call_args.kwargs["input_snap_forward_fraction"] == 0.0
