"""Research additions must not start models or change the legacy dashboard."""

from queue import Queue
from types import SimpleNamespace

from fastapi.testclient import TestClient

from streammuse.presentation.rap_demo.server import create_app


def test_research_routes_are_isolated_and_idle(tmp_path):
    calls = []
    runtime = SimpleNamespace(
        autostart=False, control_state="stopped", session_dir=tmp_path,
        close=lambda: calls.append("close"), start=lambda: calls.append("start"),
    )
    projector = SimpleNamespace(snapshot=lambda: {"session_id": "test", "audio": {}})
    app = create_app(runtime=runtime, projector=projector, websocket_queue=Queue())
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        page = client.get("/research")
        assert page.status_code == 200
        assert "rap-research.js" in page.text
        assert client.get("/api/characters/status").json()["state"] == "idle"
        assert client.get("/api/microphone").json()["busy"] is False
        assert client.get("/api/story").status_code == 409
        assert client.get("/api/state").json()["session_id"] == "test"
        assert calls == []
    assert calls == ["close"]
