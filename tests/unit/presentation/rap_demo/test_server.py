"""Contract tests for the read-only realtime rap research monitor."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Event, Lock
from types import MappingProxyType
from typing import Any

from fastapi.testclient import TestClient

from streammuse.presentation.rap_demo.server import create_app


class _State(Enum):
    RUNNING = "running"


class FakeProjector:
    def __init__(self) -> None:
        self.value: object = MappingProxyType(
            {
                "session_id": "rap-test",
                "status": _State.RUNNING,
                "current": MappingProxyType(
                    {
                        "bar": 1,
                        "tick": 19,
                        "topic": "orbital gardens",
                        "template_id": "syncopated-nine",
                    }
                ),
                "candidates": (
                    MappingProxyType(
                        {
                            "candidate_id": "candidate-1",
                            "text": "roots find routes beyond the moon",
                            "components": (
                                MappingProxyType({"name": "stress_alignment", "value": 0.875}),
                            ),
                        }
                    ),
                ),
            }
        )

    def snapshot(self) -> object:
        return self.value


class FakeRuntime:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.session_metadata = MappingProxyType(
            {"tempo_bpm": 92.0, "generator": "local_chat", "model": "qwen-rap"}
        )
        self.started = Event()
        self._lock = Lock()
        self.start_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        with self._lock:
            self.start_calls += 1
        self.started.set()

    def close(self) -> None:
        with self._lock:
            self.close_calls += 1


def _app(tmp_path: Path, queue: Queue[Any] | None = None) -> tuple[Any, FakeRuntime, Queue[Any]]:
    runtime = FakeRuntime(tmp_path / "rap-test")
    websocket_queue: Queue[Any] = queue or Queue()
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=websocket_queue)
    return app, runtime, websocket_queue


def test_state_endpoint_returns_complete_json_safe_monitor_snapshot(tmp_path: Path) -> None:
    app, _, _ = _app(tmp_path)

    response = TestClient(app).get("/api/state")

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["current"]["bar"] == 1
    assert response.json()["candidates"][0]["components"][0]["name"] == "stress_alignment"


def test_session_endpoint_reports_runtime_identity_without_exposing_runtime_objects(tmp_path: Path) -> None:
    app, _, _ = _app(tmp_path)

    response = TestClient(app).get("/api/session")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "rap-test",
        "session_dir": str(tmp_path / "rap-test"),
        "metadata": {"tempo_bpm": 92.0, "generator": "local_chat", "model": "qwen-rap"},
    }


def test_static_monitor_routes_are_served_and_model_text_uses_text_content(tmp_path: Path) -> None:
    app, _, _ = _app(tmp_path)
    client = TestClient(app)

    index = client.get("/")
    css = client.get("/static/css/rap-demo.css")
    script = client.get("/static/js/rap-demo.js")

    assert index.status_code == css.status_code == script.status_code == 200
    assert "StreamMUSE Rap Research Monitor" in index.text
    assert "candidate-table" in index.text
    assert "textContent" in script.text
    assert "innerHTML" not in script.text


def test_websocket_sends_snapshot_before_ordered_live_events(tmp_path: Path) -> None:
    app, _, websocket_queue = _app(tmp_path)
    live_events = [
        {"session_id": "rap-test", "sequence": 21, "event_type": "tick", "tick": 20, "payload": {}},
        {"session_id": "rap-test", "sequence": 22, "event_type": "syllable_emitted", "tick": 20, "payload": {"label": "roots"}},
    ]

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            first = websocket.receive_json()
            for event in live_events:
                websocket_queue.put(json.dumps(event))
            second = websocket.receive_json()
            third = websocket.receive_json()

    assert first["type"] == "snapshot"
    assert first["payload"]["current"]["bar"] == 1
    assert [second["type"], third["type"]] == ["event", "event"]
    assert [second["payload"]["sequence"], third["payload"]["sequence"]] == [21, 22]


def test_disconnected_websocket_does_not_block_later_clients(tmp_path: Path) -> None:
    app, _, websocket_queue = _app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as first:
            assert first.receive_json()["type"] == "snapshot"
        with client.websocket_connect("/ws") as second:
            assert second.receive_json()["type"] == "snapshot"
            websocket_queue.put({"sequence": 23, "event_type": "tick", "payload": {}})
            assert second.receive_json()["payload"]["sequence"] == 23


def test_lifespan_starts_runtime_and_closes_it_exactly_once(tmp_path: Path) -> None:
    app, runtime, _ = _app(tmp_path)

    with TestClient(app) as client:
        assert runtime.started.wait(timeout=1.0)
        assert client.get("/api/state").status_code == 200

    assert runtime.start_calls == 1
    assert runtime.close_calls == 1
