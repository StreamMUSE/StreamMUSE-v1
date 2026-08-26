"""Tests for the StreamMUSE web viewer and runtime lifecycle API."""

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
        webserver._config = None
        webserver._log_dir = "logs"
        webserver._last_session_dir = None
        webserver._lifecycle_state = "idle"

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
        assert 'id="btn-start"' in resp.text
        assert 'id="btn-stop"' in resp.text


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
    """The broadcaster remains safe while the server is idle."""
    webserver._ws_sink = None
    app = webserver.create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # No broadcast traffic because there's no sink — that's fine.
            # The endpoint should remain open until we close it.
            assert ws is not None


def _fake_runtime(tmp_path, name: str):
    runtime = MagicMock()
    runtime.running = True
    runtime.websocket_sink = WebSocketOutputSink()
    runtime.output_sink = MagicMock()
    runtime.service = MagicMock(running=True)
    runtime.session_dir = tmp_path / name
    return runtime


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_web_boots_idle_without_building_runtime(builder_cls):
    app = webserver.create_app()

    with TestClient(app) as client:
        status = client.get("/api/status").json()

    assert status == {"is_running": False, "state": "idle", "session_dir": None}
    builder_cls.assert_not_called()


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_start_is_idempotent_and_stop_cleans_up(builder_cls, tmp_path):
    runtime = _fake_runtime(tmp_path, "session-1")
    builder_cls.return_value.build_web.return_value = runtime
    config = MagicMock()
    webserver._configure_server(config=config, log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        first = client.post("/api/start")
        second = client.post("/api/start")
        running_status = client.get("/api/status").json()
        stopped = client.post("/api/stop")
        stopped_again = client.post("/api/stop")

    assert first.status_code == 200
    assert first.json()["message"] == "started"
    assert second.json()["message"] == "already running"
    assert running_status["is_running"] is True
    assert stopped.json()["message"] == "stopped"
    assert stopped.json()["is_running"] is False
    assert stopped_again.json()["message"] == "already stopped"
    builder_cls.assert_called_once_with(config=config, log_dir=str(tmp_path))
    runtime.start.assert_called_once_with(run_stop_tick=None)
    runtime.stop.assert_called_once_with()
    runtime.cleanup.assert_called_once_with()


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_stop_then_start_builds_fresh_runtime_and_sink(builder_cls, tmp_path):
    first_runtime = _fake_runtime(tmp_path, "session-1")
    second_runtime = _fake_runtime(tmp_path, "session-2")
    builder_cls.return_value.build_web.side_effect = [first_runtime, second_runtime]
    webserver._configure_server(config=MagicMock(), log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        client.post("/api/start")
        first_sink = webserver._ws_sink
        client.post("/api/stop")
        second = client.post("/api/start")

        assert second.json()["session_dir"].endswith("session-2")
        assert webserver._runtime is second_runtime
        assert webserver._ws_sink is second_runtime.websocket_sink
        assert webserver._ws_sink is not first_sink

    assert builder_cls.return_value.build_web.call_count == 2
    first_runtime.stop.assert_called_once_with()
    first_runtime.cleanup.assert_called_once_with()
    second_runtime.start.assert_called_once_with(run_stop_tick=None)


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_broadcaster_handles_sink_assignment_after_idle_boot(builder_cls, tmp_path):
    runtime = _fake_runtime(tmp_path, "session-1")
    builder_cls.return_value.build_web.return_value = runtime
    webserver._configure_server(config=MagicMock(), log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            client.post("/api/start")
            runtime.websocket_sink.output_status("running", "assigned")
            envelope = ws.receive_json()

    assert envelope["type"] == "status"
    assert envelope["state"] == "running"
    assert envelope["message"] == "assigned"


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
@patch("streammuse.presentation.web.server.parse_args")
@patch("streammuse.presentation.web.server.args_to_config")
@patch("uvicorn.run")
def test_web_main_configures_idle_server_without_building_runtime(
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
    assert webserver.main() == 0

    builder_cls.assert_not_called()
    assert webserver._config is config
    assert webserver._log_dir == str(tmp_path)
    assert webserver._runtime is None
    uvicorn_run.assert_called_once()
