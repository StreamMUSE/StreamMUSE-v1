from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from streammuse.presentation.rap_demo.story_api import create_story_router
from tests.unit.application.rap.test_bar_stream import _controller, _Stream, _Source


@pytest.fixture
def client():
    controller = _controller(_Stream(), _Source(), [])
    app = FastAPI()
    app.include_router(create_story_router(SimpleNamespace(controller=controller)))
    with TestClient(app) as client:
        yield client
    controller.close()


def test_snapshot_post_dedup_and_conflict(client):
    assert client.get("/api/story").json()["enabled"] is False
    body = {"text": "a courier loses the map", "event_id": "map"}
    first = client.post("/api/story/events", json=body)
    assert first.status_code == 200
    assert first.json()["status"] == "accepted"
    assert first.json()["event"]["source"] == "typed"
    duplicate = client.post("/api/story/events", json=body)
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["event"] == first.json()["event"]
    assert client.post("/api/story/events", json={**body, "text": "different"}).status_code == 409
    state = client.get("/api/story").json()
    assert state["version"] == 1
    assert state["enabled"] is True
    assert state["events"] == [first.json()["event"]]


@pytest.mark.parametrize("body", [
    {}, {"text": " "}, {"text": "x" * 513}, {"text": 12},
    {"text": "map", "event_id": ""}, {"text": "map", "event_id": 12},
    {"text": "map", "extra": True},
])
def test_bad_input(client, body):
    assert client.post("/api/story/events", json=body).status_code == 422
    assert client.get("/api/story").json()["events"] == []


def test_cumulative_context_budget_rejects_without_truncating(client):
    assert client.post("/api/story/events", json={"text": "x" * 500}).status_code == 200
    assert client.post("/api/story/events", json={"text": "y" * 20}).status_code == 422
    assert client.get("/api/story").json()["version"] == 1


def test_wrong_runtime_is_explicitly_unsupported():
    app = FastAPI()
    app.include_router(create_story_router(SimpleNamespace(controller=object())))
    with TestClient(app) as client:
        assert client.get("/api/story").status_code == 409


@pytest.mark.parametrize("source", ["typed", "microphone"])
def test_explicit_source_is_preserved_in_ack_snapshot_and_duplicate(client, source):
    body = {"text": "a stranger offers help", "event_id": "help", "source": source}
    first = client.post("/api/story/events", json=body)
    assert first.status_code == 200
    assert first.json()["event"]["source"] == source
    assert client.get("/api/story").json()["events"][0]["source"] == source
    duplicate = client.post("/api/story/events", json=body)
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["event"] == first.json()["event"]
    other_source = "typed" if source == "microphone" else "microphone"
    assert client.post("/api/story/events", json={**body, "source": other_source}).status_code == 409


@pytest.mark.parametrize("source", ["unknown", "Microphone", "", None, 1, True, []])
def test_invalid_source_is_rejected_without_adding_event(client, source):
    assert client.post("/api/story/events", json={"text": "map lost", "source": source}).status_code == 422
    assert client.get("/api/story").json()["events"] == []
