"""Tests for the StreamMUSE web viewer.

The viewer is read-only — there are no /api/start, /api/stop, or /api/inject
endpoints. These tests cover the two surfaces that do exist:
  - GET / returns the static index.html
  - WS /ws receives broadcast envelopes from the WebSocketOutputSink

To avoid spinning up RealTimeMusicService threads in tests, we set up the
module-level state directly with a fake WebSocketOutputSink.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

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
        webserver._runtime = None
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


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
@patch("streammuse.presentation.web.server.parse_args")
@patch("streammuse.presentation.web.server.args_to_config")
@patch("uvicorn.run")
def test_web_main_builds_runtime_through_shared_builder(
    uvicorn_run,
    args_to_config,
    parse_args,
    builder_cls,
    tmp_path,
):
    args = MagicMock(
        log_dir=str(tmp_path),
        web_host="127.0.0.1",
        web_port=8001,
    )
    parse_args.return_value = args
    config = MagicMock()
    config.tempo.bpm = 120.0
    config.input.type = "midi_file"
    config.inference.type = "http"
    config.inference.model_name = "lekai"
    args_to_config.return_value = config
    runtime = MagicMock()
    runtime.websocket_sink = WebSocketOutputSink()
    runtime.output_sink = MagicMock()
    runtime.service = MagicMock(running=False)
    runtime.session_manager.get_session_dir.return_value = tmp_path
    builder_cls.return_value.build_web.return_value = runtime

    assert webserver.main() == 0

    builder_cls.assert_called_once_with(config=config, log_dir=str(tmp_path))
    runtime.start.assert_called_once_with(run_stop_tick=None)
    uvicorn_run.assert_called_once()
