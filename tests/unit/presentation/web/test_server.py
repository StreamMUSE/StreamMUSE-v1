"""Tests for the StreamMUSE web viewer and runtime lifecycle API."""

from __future__ import annotations

import mido
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from streammuse.application.config import (
    ApplicationConfig,
    InferenceConfig,
    InputConfig,
    TempoConfig,
)
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
        assert 'id="session-generation-controls"' in resp.text
        assert 'id="prompt-selection-mode"' in resp.text
        assert 'value="rule_s_if_else"' in resp.text
        assert 'id="prompt-batch-candidates"' in resp.text
        assert 'id="generation-temperature"' in resp.text
        assert 'id="generation-top-p"' in resp.text
        assert 'id="generation-top-k"' in resp.text
        assert 'id="generation-repetition-penalty"' in resp.text
        assert '<option value="">Backend default</option>' in resp.text
        assert resp.text.count('placeholder="Default"') == 5
        assert resp.text.count('step="any"') == 3


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_idle_web_lifespan_does_not_access_configured_midi_device(
    builder_cls,
    monkeypatch,
    tmp_path,
):
    def fail_if_midi_is_accessed(*_args, **_kwargs):
        pytest.fail("idle Web lifespan must not access MIDI devices")

    for function_name in (
        "get_input_names",
        "get_output_names",
        "open_input",
        "open_output",
    ):
        monkeypatch.setattr(mido, function_name, fail_if_midi_is_accessed)

    webserver._configure_server(
        config=ApplicationConfig(
            input=InputConfig(
                type="midi_device",
                midi_device_name="missing-midi-device",
            )
        ),
        log_dir=str(tmp_path),
    )

    with TestClient(webserver.create_app()) as client:
        status = client.get("/api/status")

    assert status.status_code == 200
    assert status.json()["state"] == "idle"
    assert status.json()["is_running"] is False
    assert webserver._runtime is None
    builder_cls.assert_not_called()


def test_web_javascript_posts_session_config_and_locks_controls_outside_idle():
    app = webserver.create_app()
    with TestClient(app) as client:
        resp = client.get("/js/main.js")

    assert resp.status_code == 200
    assert "body: JSON.stringify({bpm, ...generationConfig})" in resp.text
    assert "bpmInput.disabled = state !== 'idle' || running" in resp.text
    assert "generationControls.disabled = state !== 'idle' || running" in resp.text
    assert "if (raw === '') continue" in resp.text
    assert "values.prompt_batch_candidates < 2" in resp.text
    assert "Number.isInteger(bpm)" in resp.text
    assert (
        "const displayBpm = !running && bpmInitialized ? selectedBpm() : bpm"
        in resp.text
    )
    assert "bpmInput.addEventListener('input'" in resp.text
    assert "PianoVisualizer.clearNotes();" in resp.text
    assert "PianoVisualizer.setCurrentTick(0);" in resp.text
    assert "Stats.reset();" in resp.text
    start_fetch = resp.text.index("const response = await fetch('/api/start'")
    assert resp.text.index("PianoVisualizer.clearNotes();") < start_fetch
    assert resp.text.index("PianoVisualizer.setCurrentTick(0);") < start_fetch
    assert resp.text.index("Stats.reset();") < start_fetch


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
        "configured_prompt_selection_mode": None,
        "active_prompt_selection_mode": None,
        "configured_prompt_batch_candidates": None,
        "active_prompt_batch_candidates": None,
        "configured_temperature": None,
        "active_temperature": None,
        "configured_top_p": None,
        "active_top_p": None,
        "configured_top_k": None,
        "active_top_k": None,
        "configured_repetition_penalty": None,
        "active_repetition_penalty": None,
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
        tempo=TempoConfig(bpm=80, ticks_per_beat=8, beats_per_bar=3),
        inference=InferenceConfig(
            prompt_selection_mode="rule_s_if_else",
            prompt_batch_candidates=10,
            temperature=1.1,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.0,
        ),
    )
    builder_cls.return_value.build_web.return_value = runtime
    webserver._configure_server(config=base_config, log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/start",
            json={
                "bpm": 80,
                "prompt_selection_mode": "rule_s_if_else",
                "prompt_batch_candidates": 10,
                "temperature": 1.1,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.0,
            },
        )

    assert response.status_code == 200
    assert response.json()["configured_bpm"] == 120
    assert response.json()["active_bpm"] == 80
    assert isinstance(response.json()["active_bpm"], int)
    assert response.json()["configured_prompt_selection_mode"] is None
    assert response.json()["active_prompt_selection_mode"] == "rule_s_if_else"
    assert response.json()["active_prompt_batch_candidates"] == 10
    assert response.json()["active_temperature"] == 1.1
    assert response.json()["active_top_p"] == 0.95
    assert response.json()["active_top_k"] == 50
    assert response.json()["active_repetition_penalty"] == 1.0
    session_config = builder_cls.call_args.kwargs["config"]
    assert session_config is not base_config
    assert session_config.tempo == TempoConfig(
        bpm=80,
        ticks_per_beat=8,
        beats_per_bar=3,
    )
    assert isinstance(session_config.tempo.bpm, int)
    assert session_config.inference.prompt_selection_mode == "rule_s_if_else"
    assert session_config.inference.prompt_batch_candidates == 10
    assert session_config.inference.temperature == 1.1
    assert session_config.inference.top_p == 0.95
    assert session_config.inference.top_k == 50
    assert session_config.inference.repetition_penalty == 1.0
    assert base_config.tempo.bpm == 120
    assert base_config.inference.prompt_selection_mode is None
    assert session_config.inference is not base_config.inference


@pytest.mark.parametrize("bpm", [29, 301, 80.5])
@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_start_rejects_invalid_web_bpm(builder_cls, bpm, tmp_path):
    webserver._configure_server(config=ApplicationConfig(), log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        response = client.post("/api/start", json={"bpm": bpm})

    assert response.status_code == 422
    builder_cls.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt_selection_mode": "unknown"},
        {"prompt_batch_candidates": 0},
        {"prompt_selection_mode": "rule_s", "prompt_batch_candidates": 1},
        {"temperature": -0.1},
        {"top_p": -0.1},
        {"top_p": 1.1},
        {"top_k": -1},
        {"repetition_penalty": 0},
    ],
)
@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_start_rejects_invalid_web_generation_config(builder_cls, payload, tmp_path):
    webserver._configure_server(config=ApplicationConfig(), log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        response = client.post("/api/start", json=payload)

    assert response.status_code == 422
    builder_cls.assert_not_called()


@pytest.mark.parametrize("top_p", [0.0, 1.0])
@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_start_accepts_zero_temperature_and_top_p_boundaries(
    builder_cls,
    top_p,
    tmp_path,
):
    runtime = _fake_runtime(tmp_path, f"boundary-{top_p}")
    runtime.config = ApplicationConfig(
        inference=InferenceConfig(temperature=0.0, top_p=top_p)
    )
    builder_cls.return_value.build_web.return_value = runtime
    webserver._configure_server(config=ApplicationConfig(), log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/start",
            json={"temperature": 0, "top_p": top_p},
        )

    assert response.status_code == 200
    assert response.json()["active_temperature"] == 0.0
    assert response.json()["active_top_p"] == top_p
    session_config = builder_cls.call_args.kwargs["config"]
    assert session_config.inference.temperature == 0.0
    assert session_config.inference.top_p == top_p


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_running_session_ignores_later_generation_overrides(builder_cls, tmp_path):
    runtime = _fake_runtime(tmp_path, "active-generation")
    runtime.config = ApplicationConfig(
        inference=InferenceConfig(
            prompt_selection_mode="rule_s",
            prompt_batch_candidates=5,
            temperature=1.1,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.0,
        )
    )
    builder_cls.return_value.build_web.return_value = runtime
    webserver._configure_server(config=ApplicationConfig(), log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        first = client.post(
            "/api/start",
            json={
                "prompt_selection_mode": "rule_s",
                "prompt_batch_candidates": 5,
                "temperature": 1.1,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.0,
            },
        )
        second = client.post(
            "/api/start",
            json={
                "prompt_selection_mode": "single",
                "prompt_batch_candidates": 1,
                "temperature": 0,
                "top_p": 0,
                "top_k": 0,
                "repetition_penalty": 2.0,
            },
        )

    assert first.json()["message"] == "started"
    assert second.json()["message"] == "already running"
    assert second.json()["active_prompt_selection_mode"] == "rule_s"
    assert second.json()["active_prompt_batch_candidates"] == 5
    assert second.json()["active_temperature"] == 1.1
    assert second.json()["active_top_p"] == 0.95
    assert second.json()["active_top_k"] == 50
    assert second.json()["active_repetition_penalty"] == 1.0
    builder_cls.assert_called_once()


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
    first_runtime.config = ApplicationConfig(
        tempo=TempoConfig(bpm=80),
        inference=InferenceConfig(
            prompt_selection_mode="rule_s",
            prompt_batch_candidates=5,
            temperature=1.1,
        ),
    )
    second_runtime.config = ApplicationConfig(tempo=TempoConfig(bpm=100))
    builder_cls.return_value.build_web.side_effect = [first_runtime, second_runtime]
    base_config = ApplicationConfig(tempo=TempoConfig(bpm=120))
    webserver._configure_server(config=base_config, log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        first = client.post(
            "/api/start",
            json={
                "bpm": 80,
                "prompt_selection_mode": "rule_s",
                "prompt_batch_candidates": 5,
                "temperature": 1.1,
            },
        )
        first_sink = webserver._ws_sink
        client.post("/api/stop")
        second = client.post("/api/start", json={"bpm": 100})

        assert first.json()["active_bpm"] == 80
        assert first.json()["active_prompt_selection_mode"] == "rule_s"
        assert second.json()["active_bpm"] == 100
        assert second.json()["active_prompt_selection_mode"] is None
        assert second.json()["active_prompt_batch_candidates"] is None
        assert second.json()["active_temperature"] is None
        assert second.json()["session_dir"].endswith("session-2")
        assert webserver._runtime is second_runtime
        assert webserver._ws_sink is second_runtime.websocket_sink
        assert webserver._ws_sink is not first_sink

    assert builder_cls.return_value.build_web.call_count == 2
    session_configs = [call.kwargs["config"] for call in builder_cls.call_args_list]
    assert [config.tempo.bpm for config in session_configs] == [80, 100]
    assert session_configs[0].inference.prompt_selection_mode == "rule_s"
    assert session_configs[0].inference.temperature == 1.1
    assert session_configs[1].inference.prompt_selection_mode is None
    assert session_configs[1].inference.temperature is None
    assert all(isinstance(config.tempo.bpm, int) for config in session_configs)
    assert session_configs[0] is not session_configs[1]
    assert base_config.tempo.bpm == 120
    first_runtime.stop.assert_called_once_with()
    first_runtime.cleanup.assert_called_once_with()
    second_runtime.start.assert_called_once_with(run_stop_tick=None)


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_worker_stop_failure_retains_runtime_until_retry_succeeds(
    builder_cls,
    tmp_path,
):
    first_runtime = _fake_runtime(tmp_path, "session-1")
    second_runtime = _fake_runtime(tmp_path, "session-2")
    first_runtime.stop.side_effect = RuntimeError("worker still running")
    builder_cls.return_value.build_web.side_effect = [
        first_runtime,
        second_runtime,
    ]
    webserver._configure_server(config=ApplicationConfig(), log_dir=str(tmp_path))

    with TestClient(webserver.create_app()) as client:
        assert client.post("/api/start").status_code == 200
        failed_stop = client.post("/api/stop")

        assert failed_stop.status_code == 500
        assert webserver._runtime is first_runtime
        assert webserver._ws_sink is first_runtime.websocket_sink
        first_runtime.cleanup.assert_not_called()
        assert builder_cls.return_value.build_web.call_count == 1

        # Simulate the bounded worker finishing after the failed Stop. The next
        # Start must retire that same runtime before constructing a fresh one.
        first_runtime.running = False
        first_runtime.stop.side_effect = None
        restarted = client.post("/api/start")

        assert restarted.status_code == 200
        assert restarted.json()["message"] == "started"
        assert webserver._runtime is second_runtime
        first_runtime.cleanup.assert_called_once_with()
        assert first_runtime.stop.call_count == 2
        assert builder_cls.return_value.build_web.call_count == 2


@patch("streammuse.presentation.web.server.RuntimeSessionBuilder")
def test_restart_never_broadcasts_pending_messages_from_old_sink(
    builder_cls,
    tmp_path,
):
    first_runtime = _fake_runtime(tmp_path, "session-1")
    second_runtime = _fake_runtime(tmp_path, "session-2")
    builder_cls.return_value.build_web.side_effect = [
        first_runtime,
        second_runtime,
    ]
    webserver._configure_server(config=ApplicationConfig(), log_dir=str(tmp_path))
    app = webserver.create_app()

    with TestClient(app) as client:
        client.post("/api/start")
        old_sink = first_runtime.websocket_sink
        old_sink.output_status("old-session", "must not leak")
        client.post("/api/stop")
        client.post("/api/start")

        with client.websocket_connect("/ws") as ws:
            second_runtime.websocket_sink.output_status("running", "fresh")
            envelope = ws.receive_json()

    assert envelope["state"] == "running"
    assert envelope["message"] == "fresh"


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
def test_web_main_boots_idle_without_accessing_midi_devices(
    uvicorn_run,
    args_to_config,
    parse_args,
    builder_cls,
    tmp_path,
    monkeypatch,
):
    def fail_if_midi_is_accessed(*_args, **_kwargs):
        pytest.fail("idle Web boot must not access MIDI devices")

    for function_name in (
        "get_input_names",
        "get_output_names",
        "open_input",
        "open_output",
    ):
        monkeypatch.setattr(mido, function_name, fail_if_midi_is_accessed)

    args = MagicMock(
        log_dir=str(tmp_path),
        web_host="127.0.0.1",
        web_port=8001,
    )
    parse_args.return_value = args
    config = MagicMock()
    config.tempo.bpm = 120.0
    config.input.type = "midi_device"
    config.inference.type = "http"
    config.inference.model_name = "lekai"
    args_to_config.return_value = config
    assert webserver.main() == 0

    builder_cls.assert_not_called()
    assert webserver._config is config
    assert webserver._log_dir == str(tmp_path)
    assert webserver._runtime is None
    uvicorn_run.assert_called_once()
