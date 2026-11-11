"""Delayed browser actions must not become directions in a reset session."""
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from streammuse.presentation.rap_demo.story_api import create_story_router
from tests.unit.application.rap.test_bar_stream import _controller, _Stream, _Source


def test_story_submission_is_bound_to_the_observed_session():
    controller = _controller(_Stream(), _Source(), [])
    app = FastAPI()
    app.include_router(create_story_router(SimpleNamespace(controller=controller)))
    try:
        with TestClient(app) as client:
            original = client.get("/api/story").json()["session_id"]
            controller.reset()
            body = {"text": "A stranger offers help.", "event_id": "delayed-mic",
                    "source": "microphone", "session_id": original}
            assert client.post("/api/story/events", json=body).status_code == 409
            snapshot = client.get("/api/story").json()
            assert snapshot["events"] == []
            assert snapshot["session_id"] != original
            accepted = client.post("/api/story/events", json={**body, "session_id": snapshot["session_id"]})
            assert accepted.status_code == 200
            assert accepted.json()["session_id"] == snapshot["session_id"]
    finally:
        controller.close()
