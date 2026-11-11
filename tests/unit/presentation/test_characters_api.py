from __future__ import annotations

import asyncio
import threading
import time

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from streammuse.infrastructure.inference.local_chat_client import LocalChatModelClient
from streammuse.presentation.rap_demo.characters_api import create_characters_router


def make_app(handler):
    app = FastAPI()
    app.include_router(create_characters_router(client_factory=lambda config: LocalChatModelClient(
        config, transport=httpx.MockTransport(handler),
    )))
    return app


def completed(client):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        data = client.get("/api/characters/status").json()
        if data["state"] not in ("running", "stopping"):
            return data
        time.sleep(0.01)
    raise AssertionError("run did not finish")


def test_router_start_poll_events_and_restart_have_run_scoped_ids():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    with TestClient(make_app(handler)) as client:
        assert client.get("/api/characters/status").json()["state"] == "idle"
        assert client.post("/api/characters/stop").json()["state"] == "idle"
        assert client.post("/api/characters/start", json={"mode": "dialogue", "turns": 2}).status_code == 202
        result = completed(client)
        assert result["state"] == "completed"
        assert len(result["accepted"]) == 2
        page = client.get("/api/characters/events?after=0").json()
        assert page["run_id"] == result["run_id"]
        assert [e["sequence"] for e in page["events"]] == list(range(1, page["next_cursor"] + 1))
        assert {e["kind"] for e in page["events"]} >= {"proposal", "accepted"}
        assert client.get(f"/api/characters/events?after={page['next_cursor']}").json()["events"] == []
        assert client.post("/api/characters/start", json={"mode": "dialogue", "turns": 1}).status_code == 202
        again = completed(client)
        assert again["run_id"] != result["run_id"]
        reset_page = client.get(f"/api/characters/events?after=999&run_id={result['run_id']}").json()
        assert reset_page["events"]


def test_running_conflicts_and_stop_drains_transport():
    started = threading.Event()
    cancelled = threading.Event()

    async def handler(request):
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with TestClient(make_app(handler)) as client:
        client.post("/api/characters/start", json={"mode": "dialogue"})
        assert started.wait(2)
        assert client.post("/api/characters/start", json={"mode": "counting"}).status_code == 409
        result = client.post("/api/characters/stop").json()
        assert result["state"] == "cancelled"
        assert result["accepted"] == []
        assert cancelled.is_set()
        assert client.post("/api/characters/stop").json()["state"] == "cancelled"


def test_router_shutdown_cancels_its_active_run():
    started = threading.Event()
    cancelled = threading.Event()

    async def handler(request):
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    with TestClient(make_app(handler)) as client:
        client.post("/api/characters/start", json={"mode": "dialogue"})
        assert started.wait(2)
    assert cancelled.wait(2)


def test_router_validates_modes_policies_turns_and_query():
    with TestClient(make_app(lambda request: httpx.Response(500))) as client:
        for body in ({}, {"mode": "audio"}, {"mode": "dialogue", "turns": 13},
                     {"mode": "counting", "policy": "bad"},
                     {"mode": "dialogue", "policy": "coordinated"}):
            assert client.post("/api/characters/start", json=body).status_code == 422
        assert client.get("/api/characters/events?after=-1").status_code == 422
        assert client.get("/api/characters/status").json()["state"] == "idle"


def test_router_exposes_unavailable_model_failure_without_fallback():
    def handler(request):
        raise httpx.ConnectError("unavailable", request=request)

    with TestClient(make_app(handler)) as client:
        response = client.post("/api/characters/start", json={"mode": "dialogue"})
        assert response.status_code == 202
        result = completed(client)
        assert result["state"] == "failed"
        assert result["error"]
        assert result["accepted"] == []
