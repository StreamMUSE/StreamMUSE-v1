"""Contract tests for the read-only realtime rap research monitor."""

from __future__ import annotations

import json
import asyncio
import time
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Event, Lock
from types import MappingProxyType
from typing import Any

from fastapi.testclient import TestClient
import pytest

from streammuse.presentation.rap_demo.server import _ConnectionPool, _MonitorLifecycle, create_app


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
                "latest_request": MappingProxyType(
                    {"request_id": "request-1", "context_lines": ("prior line",), "flow": {"slots": ({"tick_in_bar": 0},)}}
                ),
                "latest_batch": MappingProxyType(
                    {"request_id": "request-1", "prompt": ({"role": "user", "content": "exact flow prompt"},)}
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
    assert response.json()["latest_request"]["context_lines"] == ["prior line"]
    assert response.json()["latest_batch"]["prompt"][0]["content"] == "exact flow prompt"


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
    assert "prompt-tokens" in index.text
    assert "research_metrics" in script.text
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


def test_connect_replays_events_newer_than_snapshot_before_live_registration(tmp_path: Path) -> None:
    class Projector:
        def snapshot(self) -> dict[str, object]:
            return {
                "session_id": "rap-test",
                "last_sequence": 40,
                "recent_events": [{"sequence": 40, "event_type": "tick"}],
            }

    class Socket:
        def __init__(self) -> None:
            self.messages: list[object] = []

        async def accept(self) -> None:
            return None

        async def send_json(self, message: object) -> None:
            self.messages.append(message)

    websocket_queue: Queue[Any] = Queue()
    websocket_queue.put({"sequence": 41, "event_type": "syllable_emitted", "payload": {"label": "new"}})
    lifecycle = _MonitorLifecycle(FakeRuntime(tmp_path), Projector(), websocket_queue)
    socket = Socket()

    asyncio.run(lifecycle.connect(socket))  # type: ignore[arg-type]

    assert socket.messages == [
        {
            "type": "snapshot",
            "payload": {
                "session_id": "rap-test",
                "last_sequence": 40,
                "recent_events": [{"sequence": 40, "event_type": "tick"}],
            },
        },
        {
            "type": "event",
            "payload": {"sequence": 41, "event_type": "syllable_emitted", "payload": {"label": "new"}},
        },
    ]


def test_connect_forwards_drained_events_to_existing_clients(tmp_path: Path) -> None:
    class Projector:
        def snapshot(self) -> dict[str, object]:
            return {"session_id": "rap-test", "last_sequence": 41, "recent_events": [{"sequence": 41}]}

    class Socket:
        def __init__(self) -> None:
            self.messages: list[object] = []

        async def accept(self) -> None:
            return None

        async def send_json(self, message: object) -> None:
            self.messages.append(message)

    async def exercise() -> tuple[list[object], list[object]]:
        websocket_queue: Queue[Any] = Queue()
        lifecycle = _MonitorLifecycle(FakeRuntime(tmp_path), Projector(), websocket_queue)
        existing = Socket()
        joining = Socket()
        await lifecycle.connections.connect(existing, {"last_sequence": 40})  # type: ignore[arg-type]
        websocket_queue.put({"sequence": 41, "event_type": "tick", "payload": {}})
        await lifecycle.connect(joining)  # type: ignore[arg-type]
        return existing.messages, joining.messages

    existing_messages, joining_messages = asyncio.run(exercise())

    assert existing_messages[-1] == {
        "type": "event",
        "payload": {"sequence": 41, "event_type": "tick", "payload": {}},
    }
    assert joining_messages == [
        {
            "type": "snapshot",
            "payload": {"session_id": "rap-test", "last_sequence": 41, "recent_events": [{"sequence": 41}]},
        }
    ]


def test_disconnected_websocket_does_not_block_later_clients(tmp_path: Path) -> None:
    app, _, websocket_queue = _app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as first:
            assert first.receive_json()["type"] == "snapshot"
        with client.websocket_connect("/ws") as second:
            assert second.receive_json()["type"] == "snapshot"
            websocket_queue.put({"sequence": 23, "event_type": "tick", "payload": {}})
            assert second.receive_json()["payload"]["sequence"] == 23


def test_broadcaster_discards_live_queue_events_without_clients_because_snapshot_is_authoritative(tmp_path: Path) -> None:
    app, _, websocket_queue = _app(tmp_path)

    with TestClient(app):
        websocket_queue.put({"sequence": 24, "event_type": "tick", "payload": {}})
        deadline = time.monotonic() + 1.0
        while not websocket_queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)

        assert websocket_queue.empty()


def test_lifespan_starts_runtime_and_closes_it_exactly_once(tmp_path: Path) -> None:
    app, runtime, _ = _app(tmp_path)

    with TestClient(app) as client:
        assert runtime.started.wait(timeout=1.0)
        assert client.get("/api/state").status_code == 200

    assert runtime.start_calls == 1
    assert runtime.close_calls == 1


def test_slow_websocket_is_dropped_without_blocking_a_healthy_client() -> None:
    class Socket:
        def __init__(self) -> None:
            self.block = False
            self.messages: list[object] = []

        async def accept(self) -> None:
            return None

        async def send_json(self, message: object) -> None:
            if self.block:
                await asyncio.Event().wait()
            self.messages.append(message)

    async def exercise() -> tuple[list[object], bool]:
        pool = _ConnectionPool()
        slow = Socket()
        healthy = Socket()
        await pool.connect(slow, {})  # type: ignore[arg-type]
        await pool.connect(healthy, {})  # type: ignore[arg-type]
        slow.block = True
        await pool.send({"type": "event", "payload": {"sequence": 1}})
        return healthy.messages, slow in pool._connections

    messages, slow_connected = asyncio.run(exercise())
    assert messages[-1] == {"type": "event", "payload": {"sequence": 1}}
    assert slow_connected is False


def test_monitor_cleanup_stops_broadcaster_even_when_runtime_close_raises(tmp_path: Path) -> None:
    class Runtime(FakeRuntime):
        def close(self) -> None:
            super().close()
            raise RuntimeError("close failed")

    runtime = Runtime(tmp_path)
    lifecycle = _MonitorLifecycle(runtime, FakeProjector(), Queue())

    async def exercise() -> None:
        await lifecycle.start()
        with pytest.raises(RuntimeError, match="close failed"):
            await lifecycle.close()
        assert lifecycle._broadcaster_task is not None and lifecycle._broadcaster_task.done()
        assert lifecycle._runtime_thread is not None and not lifecycle._runtime_thread.is_alive()

    asyncio.run(exercise())
