"""Unit tests for the web UI server.

Uses the FastAPI TestClient. Does NOT start the real-time service — we only
exercise routes + websocket broadcasting with stub state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

from streammuse.infrastructure.output.websocket import (
    WebSocketOutputConfig,
    WebSocketOutputSink,
)
from streammuse.presentation.web import server as webserver


class StubRepository:
    def __init__(self, metadata: Dict[str, List[Dict[str, Any]]]) -> None:
        self._metadata = metadata
        self.last_load: Tuple[str, Dict[str, Any], int] | None = None

    def get_prompts_for_key(self, key: str) -> List[Dict[str, Any]]:
        return list(self._metadata.get(key, []))

    def select_prompt(self, key: str, strategy):
        prompts = self._metadata.get(key, [])
        if not prompts:
            return None
        if strategy == "first":
            return prompts[0]
        if isinstance(strategy, int) and 0 <= strategy < len(prompts):
            return prompts[strategy]
        return prompts[0]

    def load_prompt_events(self, *, key, prompt, max_ticks, load_melody=True, load_accompaniment=True):
        self.last_load = (key, prompt, max_ticks)
        return ([], [])  # empty events; we only check the call


class StubEngine:
    def __init__(self) -> None:
        self.clear_calls = 0
        self.inject_calls: List[Dict[str, Any]] = []

    def clear_history(self) -> Dict[str, Any]:
        self.clear_calls += 1
        return {}

    def inject_history(self, *, melody_events, accompaniment_events, injection_length_ticks) -> None:
        self.inject_calls.append(
            {
                "melody_len": len(melody_events),
                "acc_len": len(accompaniment_events),
                "ticks": injection_length_ticks,
            }
        )

    # Not exercised by these tests
    def generate_accompaniment(self, *a, **kw):
        raise NotImplementedError

    def set_injection_offset(self, offset_ticks: int) -> None:
        pass


class StubConfig:
    class _Tempo:
        ticks_per_beat = 4
        beats_per_bar = 4
    tempo = _Tempo()


@pytest.fixture(autouse=True)
def reset_state(tmp_path: Path):
    # Reset module-level state around every test
    webserver.state = webserver.AppState()
    yield
    webserver.state = webserver.AppState()


@pytest.fixture
def wired_state(tmp_path: Path):
    """State wired with stubs and a real session dir in tmp_path."""
    sm_cls = webserver.SessionManager
    sm = sm_cls(str(tmp_path / "logs"))
    sm.create_session_directory()

    metadata = {
        "C_major": [
            {
                "name": "pop909_001",
                "melody_path": "/absolute/missing_mel.mid",
                "accompaniment_path": "/absolute/missing_acc.mid",
                "duration_ticks": 1920,
            }
        ],
        "G_major": [],
    }
    repo = StubRepository(metadata)
    repo._metadata = metadata  # make it look like FileSystemPromptRepository
    engine = StubEngine()
    ws_sink = WebSocketOutputSink(WebSocketOutputConfig(include_timestamps=False))

    webserver.state.ws_sink = ws_sink
    webserver.state.inference_engine = engine
    webserver.state.repository = repo
    webserver.state.session_manager = sm
    webserver.state.config = StubConfig()

    return {"repo": repo, "engine": engine, "ws": ws_sink, "sm": sm}


def test_index_serves_html():
    app = webserver.create_app()
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "<html" in resp.text.lower()


def test_list_prompts_empty_when_no_repo():
    app = webserver.create_app()
    with TestClient(app) as client:
        resp = client.get("/api/prompts")
        assert resp.status_code == 200
        assert resp.json() == {"keys": {}}


def test_list_prompts_shape(wired_state):
    app = webserver.create_app()
    with TestClient(app) as client:
        resp = client.get("/api/prompts")
        assert resp.status_code == 200
        body = resp.json()
        assert "keys" in body
        assert "C_major" in body["keys"]
        assert body["keys"]["C_major"][0]["name"] == "pop909_001"
        assert body["keys"]["C_major"][0]["duration_ticks"] == 1920


def test_list_prompts_for_key(wired_state):
    app = webserver.create_app()
    with TestClient(app) as client:
        resp = client.get("/api/prompts/C_major")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "pop909_001"


def test_inject_calls_engine(wired_state, tmp_path: Path):
    app = webserver.create_app()
    engine = wired_state["engine"]
    repo = wired_state["repo"]
    sm = wired_state["sm"]

    with TestClient(app) as client:
        resp = client.post(
            "/api/inject",
            json={"key": "C_major", "strategy": "first"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["injected_melody"] == 0  # stub repo returns empty events
        assert body["injected_acc"] == 0
        assert body["prompt_name"] == "pop909_001"

    assert engine.clear_calls == 1
    assert len(engine.inject_calls) == 1
    assert repo.last_load is not None

    # Artifacts written
    sdir = sm.get_session_dir()
    assert (sdir / "melody_history.json").exists()
    assert (sdir / "accompaniment_history.json").exists()


def test_inject_missing_key_returns_message(wired_state):
    app = webserver.create_app()
    with TestClient(app) as client:
        resp = client.post("/api/inject", json={"key": "Nonexistent", "strategy": "random"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["injected_melody"] == 0
        assert "no prompts" in body["message"].lower()


def test_inject_returns_503_when_not_initialized():
    app = webserver.create_app()
    with TestClient(app) as client:
        resp = client.post("/api/inject", json={"key": "C_major", "strategy": "first"})
        assert resp.status_code == 503


def test_websocket_receives_queued_messages(wired_state):
    app = webserver.create_app()
    ws_sink = wired_state["ws"]

    with TestClient(app) as client:
        # Push a status before the WS connects so the broadcaster sees the queue prewarmed.
        ws_sink.output_status(state="running", message="hi")

        with client.websocket_connect("/ws") as ws:
            # Push another after connect
            ws_sink.output_tick(tick=7, bar=1, beat=3)

            # Drain up to 2 messages with a reasonable timeout.
            received: List[Dict[str, Any]] = []
            for _ in range(2):
                data = ws.receive_text()
                received.append(json.loads(data))

    types = {m["type"] for m in received}
    assert "status" in types or "tick" in types
    # at least one of the two we pushed should arrive
    assert any(m.get("state") == "running" or m.get("tick") == 7 for m in received)
