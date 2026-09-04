"""Tests for the StreamMUSE web viewer and runtime lifecycle API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from streammuse.application.config import ApplicationConfig, TempoConfig
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
        assert 'id="session-bpm"' in resp.text
        assert 'type="number"' in resp.text
        assert 'min="30"' in resp.text
        assert 'max="300"' in resp.text
        assert 'step="1"' in resp.text
        assert 'inputmode="numeric"' in resp.text


def test_web_javascript_posts_bpm_and_locks_control_outside_idle():
    app = webserver.create_app()
    with TestClient(app) as client:
        resp = client.get("/js/main.js")

    assert resp.status_code == 200
    assert "body: JSON.stringify({bpm})" in resp.text
    assert "bpmInput.disabled = state !== 'idle' || running" in resp.text
    assert "Number.isInteger(bpm)" in resp.text
    assert (
        "const displayBpm = !running && bpmInitialized ? selectedBpm() : bpm"
        in resp.text
    )
    assert "bpmInput.addEventListener('input'" in resp.text


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
    runtime.config = ApplicationConfig()
    runtime.running = True
    runtime.websocket_sink = WebSocketOutputSink()
    runtime.output_sink = MagicMock()
    runtime.service = MagicMock(running=True)
    runtime.session_dir = tmp_path / name
    return runtime


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_web_boots_idle_without_building_runtime(builder_cls):
    webserver._configure_server(config=ApplicationConfig(), log_dir="logs")
    app = webserver.create_app()

    with TestClient(app) as client:
        status = client.get("/api/status").json()

    assert status == {
        "is_running": False,
        "state": "idle",
        "session_dir": None,
        "configured_bpm": 120,
        "active_bpm": None,
    }
    builder_cls.assert_not_called()


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_start_is_idempotent_and_stop_cleans_up(builder_cls, tmp_path):
    runtime = _fake_runtime(tmp_path, "session-1")
    config = ApplicationConfig(tempo=TempoConfig(bpm=96))
    runtime.config = config
    builder_cls.return_value.build_web.return_value = runtime
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
    assert first.json()["configured_bpm"] == 96
    assert first.json()["active_bpm"] == 96
    assert isinstance(first.json()["configured_bpm"], int)
    assert isinstance(first.json()["active_bpm"], int)
    assert second.json()["message"] == "already running"
    assert running_status["is_running"] is True
    assert running_status["configured_bpm"] == 96
    assert running_status["active_bpm"] == 96
    assert stopped.json()["message"] == "stopped"
    assert stopped.json()["is_running"] is False
    assert stopped.json()["configured_bpm"] == 96
    assert stopped.json()["active_bpm"] is None
    assert stopped_again.json()["message"] == "already stopped"
    builder_cls.assert_called_once()
    session_config = builder_cls.call_args.kwargs["config"]
    assert session_config == config
    assert session_config is not config
    assert session_config.tempo is not config.tempo
    assert isinstance(session_config.tempo.bpm, int)
    runtime.start.assert_called_once_with(run_stop_tick=None)
    runtime.stop.assert_called_once_with()
    runtime.cleanup.assert_called_once_with()


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_start_uses_requested_bpm_without_mutating_base_config(builder_cls, tmp_path):
    runtime = _fake_runtime(tmp_path, "session-1")
    base_config = ApplicationConfig(
        tempo=TempoConfig(bpm=120, ticks_per_beat=8, beats_per_bar=3)
    )
    runtime.config = ApplicationConfig(
        tempo=TempoConfig(bpm=80, ticks_per_beat=8, beats_per_bar=3)
    )
    builder_cls.return_value.build_web.return_value = runtime
    webserver._configure_server(config=base_config, log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        response = client.post("/api/start", json={"bpm": 80})

    assert response.status_code == 200
    assert response.json()["configured_bpm"] == 120
    assert response.json()["active_bpm"] == 80
    assert isinstance(response.json()["active_bpm"], int)
    session_config = builder_cls.call_args.kwargs["config"]
    assert session_config is not base_config
    assert session_config.tempo == TempoConfig(
        bpm=80,
        ticks_per_beat=8,
        beats_per_bar=3,
    )
    assert isinstance(session_config.tempo.bpm, int)
    assert base_config.tempo.bpm == 120


@pytest.mark.parametrize("bpm", [29, 301, 80.5])
@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_start_rejects_invalid_web_bpm(builder_cls, bpm, tmp_path):
    webserver._configure_server(config=ApplicationConfig(), log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        response = client.post("/api/start", json={"bpm": bpm})

    assert response.status_code == 422
    builder_cls.assert_not_called()


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_fractional_cli_bpm_is_rounded_for_default_web_session(builder_cls, tmp_path):
    runtime = _fake_runtime(tmp_path, "session-1")
    runtime.config = ApplicationConfig(tempo=TempoConfig(bpm=121))
    builder_cls.return_value.build_web.return_value = runtime
    base_config = ApplicationConfig(tempo=TempoConfig(bpm=120.6))
    webserver._configure_server(config=base_config, log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        idle_status = client.get("/api/status").json()
        response = client.post("/api/start")

    assert idle_status["configured_bpm"] == 121
    assert response.status_code == 200
    assert response.json()["configured_bpm"] == 121
    assert response.json()["active_bpm"] == 121
    session_config = builder_cls.call_args.kwargs["config"]
    assert session_config.tempo.bpm == 121
    assert isinstance(session_config.tempo.bpm, int)
    assert base_config.tempo.bpm == 120.6


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_stop_then_start_builds_fresh_runtime_and_sink(builder_cls, tmp_path):
    first_runtime = _fake_runtime(tmp_path, "session-1")
    second_runtime = _fake_runtime(tmp_path, "session-2")
    first_runtime.config = ApplicationConfig(tempo=TempoConfig(bpm=80))
    second_runtime.config = ApplicationConfig(tempo=TempoConfig(bpm=100))
    builder_cls.return_value.build_web.side_effect = [first_runtime, second_runtime]
    base_config = ApplicationConfig(tempo=TempoConfig(bpm=120))
    webserver._configure_server(config=base_config, log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        first = client.post("/api/start", json={"bpm": 80})
        first_sink = webserver._ws_sink
        client.post("/api/stop")
        second = client.post("/api/start", json={"bpm": 100})

        assert first.json()["active_bpm"] == 80
        assert second.json()["active_bpm"] == 100
        assert second.json()["session_dir"].endswith("session-2")
        assert webserver._runtime is second_runtime
        assert webserver._ws_sink is second_runtime.websocket_sink
        assert webserver._ws_sink is not first_sink

    assert builder_cls.return_value.build_web.call_count == 2
    session_configs = [call.kwargs["config"] for call in builder_cls.call_args_list]
    assert [config.tempo.bpm for config in session_configs] == [80, 100]
    assert all(isinstance(config.tempo.bpm, int) for config in session_configs)
    assert session_configs[0] is not session_configs[1]
    assert base_config.tempo.bpm == 120
    first_runtime.stop.assert_called_once_with()
    first_runtime.cleanup.assert_called_once_with()
    second_runtime.start.assert_called_once_with(run_stop_tick=None)


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_broadcaster_handles_sink_assignment_after_idle_boot(builder_cls, tmp_path):
    runtime = _fake_runtime(tmp_path, "session-1")
    builder_cls.return_value.build_web.return_value = runtime
    webserver._configure_server(config=ApplicationConfig(), log_dir=str(tmp_path))
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
