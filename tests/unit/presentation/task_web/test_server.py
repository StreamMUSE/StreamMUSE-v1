from __future__ import annotations

import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from streammuse.application.tasks import TaskWebConfig
from streammuse.infrastructure.task_view import QueueTaskEventSink
from streammuse.presentation.task_web import TaskWebServer, TaskWebStartupError


def make_server(port: int = 8002) -> TaskWebServer:
    sink = QueueTaskEventSink(session_id="session", task="zip_zap_zop")
    return TaskWebServer(
        config=TaskWebConfig(enabled=True, port=port),
        sink=sink,
        session_id="session",
    )


def test_index_has_csp_and_websocket_requires_snapshot_ack() -> None:
    server = make_server()
    with TestClient(server.app, base_url="http://127.0.0.1:8002") as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Content-Security-Policy" in response.headers
        with client.websocket_connect(
            f"/ws?token={server.token}",
            headers={"origin": "http://127.0.0.1:8002"},
        ) as websocket:
            snapshot = websocket.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["payload"]["status"] == "waiting_for_game"
            assert not server.viewer_ready.is_set()
            websocket.send_text("not-json")
            assert not server.viewer_ready.is_set()
            websocket.send_json({"type": "viewer_ready", "session_id": "wrong"})
            assert not server.viewer_ready.is_set()
            websocket.send_json({"type": "viewer_ready", "session_id": "session"})
            assert server.viewer_ready.wait(1.0)


def test_second_viewer_receives_latest_complete_snapshot() -> None:
    server = make_server()
    server.sink.update_status("initializing")
    with TestClient(server.app, base_url="http://127.0.0.1:8002") as client:
        with client.websocket_connect(
            f"/ws?token={server.token}",
            headers={"origin": "http://127.0.0.1:8002"},
        ) as websocket:
            snapshot = websocket.receive_json()
            assert snapshot["payload"]["status"] == "initializing"


def test_frontend_uses_text_content_and_has_no_inline_script() -> None:
    static_root = Path(__file__).parents[4] / "src/streammuse/presentation/task_web/static"
    script = (static_root / "js/main.js").read_text(encoding="utf-8")
    html = (static_root / "index.html").read_text(encoding="utf-8")
    assert "innerHTML" not in script
    assert ".textContent" in script
    assert "<script src=" in html
    assert "<script>" not in html


def test_wrong_origin_or_token_is_rejected() -> None:
    server = make_server()
    with TestClient(server.app, base_url="http://127.0.0.1:8002") as client:
        with pytest.raises(Exception):
            with client.websocket_connect(
                f"/ws?token={server.token}",
                headers={"origin": "http://attacker.example"},
            ):
                pass
        with pytest.raises(Exception):
            with client.websocket_connect(
                "/ws?token=wrong",
                headers={"origin": "http://127.0.0.1:8002"},
            ):
                pass


def test_port_conflict_is_reported_synchronously() -> None:
    occupied = socket.socket()
    occupied.bind(("127.0.0.1", 0))
    occupied.listen(1)
    port = occupied.getsockname()[1]
    server = make_server(port)
    try:
        with pytest.raises(TaskWebStartupError, match="could not bind"):
            server.start()
    finally:
        server.close()
        occupied.close()


def test_background_uvicorn_starts_without_installing_main_thread_signals() -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    server = make_server(port)
    try:
        server.start()
        assert server.server_ready.is_set()
    finally:
        server.close()
