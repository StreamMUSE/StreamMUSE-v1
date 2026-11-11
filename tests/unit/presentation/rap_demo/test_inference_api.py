"""The inference page is an alternate controller view, not another session."""

from queue import Queue
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from streammuse.presentation.rap_demo.inference_api import create_inference_router
from streammuse.presentation.rap_demo.server import create_app


class Service:
    def __init__(self):
        self.calls = []

    def snapshot(self):
        return {"supported": True, "session_id": "shared", "state": "stopped"}

    def configure(self, payload):
        self.calls.append(("configure", payload))
        return self.snapshot()

    def start(self, owner_id=None):
        self.calls.append(("start", owner_id))
        return self.snapshot()

    def stop(self):
        self.calls.append(("stop", None))
        return self.snapshot()

    def reset(self):
        self.calls.append(("reset", None))
        return self.snapshot()

    def steer(self, payload):
        self.calls.append(("steer", payload))
        return self.snapshot()

    def tap(self, payload):
        self.calls.append(("tap", payload))
        return self.snapshot()

    def heartbeat(self, payload):
        self.calls.append(("heartbeat", payload))
        return self.snapshot()

    def export_trace(self):
        return {"session_id": "shared", "events": []}

    def export_metrics_csv(self):
        return "request_id,latency_ms\n"


def make_client(runtime):
    app = FastAPI()
    app.include_router(create_inference_router(runtime))
    return TestClient(app)


def test_mount_and_poll_do_not_start_or_create_a_session():
    service = Service()
    client = make_client(SimpleNamespace(inference=service))
    assert client.get("/api/inference/state").json()["session_id"] == "shared"
    assert service.calls == []


@pytest.mark.parametrize("route,method,payload", [
    ("config", "configure", {"mode": "cycle"}),
    ("steer", "steer", {"topic": "rain"}),
    ("tap", "tap", {"owner_id": "browser", "event_id": "tap1"}),
    ("heartbeat", "heartbeat", {"owner_id": "browser"}),
])
def test_controls_delegate_to_current_shared_service(route, method, payload):
    runtime = SimpleNamespace(inference=Service())
    client = make_client(runtime)
    replacement = Service()
    runtime.inference = replacement
    response = client.post(f"/api/inference/{route}", json=payload)
    assert response.status_code == 200
    assert replacement.calls == [(method, payload)]


def test_lifecycle_controls_keep_one_session():
    service = Service()
    client = make_client(SimpleNamespace(inference=service))
    assert client.post("/api/inference/start", json={"owner_id": "tab-a"}).status_code == 202
    assert client.post("/api/inference/stop", json={}).status_code == 202
    assert client.post("/api/inference/reset", json={}).status_code == 200
    assert service.calls == [("start", "tab-a"), ("stop", None), ("reset", None)]


def test_unavailable_runtime_is_explicit_and_never_constructed():
    client = make_client(SimpleNamespace())
    assert client.get("/api/inference/state").json()["supported"] is False
    assert client.post("/api/inference/start", json={}).status_code == 409


@pytest.mark.parametrize("body", [b"[]", b"null", b"not json", b'{"owner_id":NaN}'])
def test_invalid_control_body_is_rejected(body):
    service = Service()
    client = make_client(SimpleNamespace(inference=service))
    assert client.post("/api/inference/start", content=body).status_code == 422
    assert service.calls == []


def test_control_payload_is_bounded():
    client = make_client(SimpleNamespace(inference=Service()))
    assert client.post("/api/inference/config", content=b" " * 65537).status_code == 413


@pytest.mark.parametrize("route,payload", [
    ("start", {"owner_id": 12}), ("start", {"owner_id": ""}),
    ("start", {"unrecognized": True}), ("stop", {"mode": "cycle"}),
    ("reset", {"mode": "cycle"}),
])
def test_lifecycle_rejects_ambiguous_fields(route, payload):
    client = make_client(SimpleNamespace(inference=Service()))
    assert client.post(f"/api/inference/{route}", json=payload).status_code == 422


@pytest.mark.parametrize("failure,code", [(ValueError("bad config"), 422),
                                         (RuntimeError("stop first"), 409),
                                         (ConnectionError("offline"), 503)])
def test_expected_backend_failures_are_visible(failure, code):
    service = Service()
    def fail(payload):
        raise failure
    service.configure = fail
    response = make_client(SimpleNamespace(inference=service)).post("/api/inference/config", json={})
    assert response.status_code == code


def test_trace_and_csv_export_preserve_formats():
    client = make_client(SimpleNamespace(inference=Service()))
    assert client.get("/api/inference/export?format=json").json()["session_id"] == "shared"
    response = client.get("/api/inference/export?format=csv")
    assert response.text == "request_id,latency_ms\n"
    assert response.headers["content-type"].startswith("text/csv")
    assert client.get("/api/inference/export?format=unsupported").status_code == 422


def test_original_and_inference_pages_share_runtime_without_autostart(tmp_path):
    calls = []
    service = Service()
    runtime = SimpleNamespace(
        autostart=False, control_state="stopped", session_dir=tmp_path,
        inference=service, close=lambda: calls.append("close"),
        start=lambda: calls.append("start"),
    )
    projector = SimpleNamespace(snapshot=lambda: {"session_id": "shared", "audio": {}})
    app = create_app(runtime=runtime, projector=projector, websocket_queue=Queue())
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        page = client.get("/inference")
        assert page.status_code == 200
        assert "rap-inference.js" in page.text
        assert client.get("/api/state").json()["session_id"] == "shared"
        assert client.get("/api/inference/state").json()["session_id"] == "shared"
        assert calls == []
        assert service.calls == []
    assert calls == ["close"]


@pytest.mark.parametrize("route,payload", [
    ("plan", {"topic": "ignored legacy topic"}),
    ("interaction/arm", {}),
    ("interaction/tap", {"server_time_seconds": 1.0}),
])
def test_legacy_mode_specific_writes_reject_active_inference(tmp_path, route, payload):
    runtime = SimpleNamespace(
        autostart=False, control_state="running", session_dir=tmp_path,
        inference=SimpleNamespace(active=True), close=lambda: None,
    )
    projector = SimpleNamespace(snapshot=lambda: {"session_id": "shared", "audio": {}})
    app = create_app(runtime=runtime, projector=projector, websocket_queue=Queue())
    with TestClient(app) as client:
        result = client.post(f"/api/control/{route}", json=payload)
        assert result.status_code == 409
        assert "inference" in result.json()["detail"].lower()
