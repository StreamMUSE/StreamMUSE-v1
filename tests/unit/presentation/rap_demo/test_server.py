"""Contract tests for the read-only realtime rap research monitor."""

from __future__ import annotations

import json
import asyncio
import time
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from types import MappingProxyType, SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest
import httpx

from streammuse.application.rap.runtime import RapAudioDemoDependencies
from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher, RapStateProjector
from streammuse.domain.rap import PlaybackState
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES, TemplateCatalog
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
    autostart = True

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


class ControllableFakeRuntime(FakeRuntime):
    autostart = False

    def __init__(self, session_dir: Path) -> None:
        super().__init__(session_dir)
        self.control_state = "stopped"
        self.stop_calls = 0
        self.reset_calls = 0
        self.finished = Event()

    def start(self) -> None:
        with self._lock:
            self.start_calls += 1
            self.control_state = "running"
        self.started.set()
        self.finished.wait(timeout=1)

    def request_stop(self) -> None:
        with self._lock:
            self.stop_calls += 1
            self.control_state = "stop_requested"

    def finish_bar(self) -> None:
        with self._lock:
            self.control_state = "stopped"
        self.finished.set()

    def reset(self) -> None:
        with self._lock:
            self.reset_calls += 1
            self.control_state = "stopped"


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


def test_settings_endpoints_delegate_typed_runtime_contract(tmp_path: Path) -> None:
    runtime = ControllableFakeRuntime(tmp_path)
    updates = []
    runtime.settings_state = lambda: {"supported": True, "values": {"mode": "continuous"}}
    runtime.update_settings = lambda payload: updates.append(payload) or {"effective_from_bar": 3}
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())
    with TestClient(app) as client:
        assert client.get("/api/control/settings").json()["values"]["mode"] == "continuous"
        response = client.post("/api/control/settings", json={"tempo_bpm": 50})
        assert response.status_code == 200
        assert response.json()["effective_from_bar"] == 3
        assert updates == [{"tempo_bpm": 50}]
        assert client.post("/api/control/settings", content="not json").status_code == 422


def test_new_controls_reject_unsupported_runtime_and_report_capabilities(tmp_path: Path) -> None:
    app, _, _ = _app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/control/settings").json()["supported"] is False
        assert client.get("/api/control/interaction").json()["supported"] is False
        assert client.post("/api/control/settings", json={"mode": "alternating"}).status_code == 404
        assert client.post("/api/control/interaction/arm", json={}).status_code == 404


def test_interaction_routes_validate_clock_measurement(tmp_path: Path) -> None:
    runtime = ControllableFakeRuntime(tmp_path)
    taps = []
    runtime.interaction_state = lambda: {"supported": True, "phase": "CAPTURING"}
    runtime.arm_capture = lambda: {"phase": "ARMED"}
    runtime.record_tap = lambda **payload: taps.append(payload) or {"accepted": True}
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())
    with TestClient(app) as client:
        state = client.get("/api/control/interaction").json()
        assert state["monotonic_seconds"] > 0
        assert client.post("/api/control/interaction/arm", json={}).json()["phase"] == "ARMED"
        payload = {"server_time_seconds": time.monotonic(), "client_time_ms": 100.0, "uncertainty_ms": 2.0}
        assert client.post("/api/control/interaction/tap", json=payload).status_code == 200
        assert taps == [payload]
        for patch in ({"server_time_seconds": True}, {"server_time_seconds": "bad"}, {**payload, "uncertainty_ms": -1}):
            assert client.post("/api/control/interaction/tap", json=patch).status_code == 422
        for name in payload:
            assert client.post(
                "/api/control/interaction/tap", json={**payload, name: 10 ** 400},
            ).status_code == 422
        assert taps == [payload]


def test_settings_runtime_conflicts_are_explicit(tmp_path: Path) -> None:
    runtime = ControllableFakeRuntime(tmp_path)
    def reject(_payload):
        raise RuntimeError("mode changes require stopped playback")
    runtime.update_settings = reject
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())
    with TestClient(app) as client:
        response = client.post("/api/control/settings", json={"mode": "alternating"})
        assert response.status_code == 409
        assert "stopped" in response.json()["detail"]


def test_installed_settings_post_reports_committed_mix_despite_discovery_error(tmp_path):
    from streammuse.application.rap.audio_settings import install_audio_settings
    from streammuse.application.rap.settings import RapSettings

    runtime = ControllableFakeRuntime(tmp_path)
    runtime.control_state = PlaybackState.RUNNING
    applied = []
    runtime.controller = SimpleNamespace(update_mix=lambda config: applied.append(config) or {})
    status_calls = 0

    def status():
        nonlocal status_calls
        status_calls += 1
        if status_calls == 2:
            raise OSError("transient status failure")
        return {"supports_device_selection": True}

    sink = SimpleNamespace(
        output_device_status=status, set_output_device=lambda value: None,
        list_output_devices=lambda: [],
    )
    service = install_audio_settings(runtime, RapSettings(), sink=sink)
    runtime.update_settings = service.update
    runtime.settings_state = service.snapshot
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())

    with TestClient(app) as client:
        response = client.post("/api/control/settings", json={"vocal_gain": 0.4})
        assert response.status_code == 200
        assert response.json()["values"]["vocal_gain"] == 0.4
        assert response.json()["device_error"] == "transient status failure"
        assert applied[0].vocal_gain == 0.4
        assert client.get("/api/control/settings").json()["values"]["vocal_gain"] == 0.4


def test_reset_maps_pending_runtime_settings_update_to_conflict(tmp_path):
    entered, release = Event(), Event()
    errors = []

    def update(payload):
        entered.set()
        assert release.wait(2)
        return {}

    runtime = RapAudioDemoDependencies(
        tempo=Tempo(60, 4, 4), controller=SimpleNamespace(close=lambda: None), coordinator=None,
        playback=SimpleNamespace(state=PlaybackState.STOPPED, close=lambda: None),
        publisher=object(), dispatcher=SimpleNamespace(flush_and_close=lambda: None),
        session_dir=tmp_path, settings_service=SimpleNamespace(update=update),
    )

    def run_update():
        try:
            runtime.update_settings({"tempo_bpm": 90})
        except BaseException as error:
            errors.append(error)

    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())
    with TestClient(app, raise_server_exceptions=False) as client:
        worker = Thread(target=run_update)
        worker.start()
        try:
            assert entered.wait(1)
            response = client.post("/api/control/reset")
            assert response.status_code == 409
            assert "settings update" in response.json()["detail"]
            assert runtime.control_state == PlaybackState.STOPPED
        finally:
            release.set()
            worker.join(2)
        assert not worker.is_alive()
        assert errors == []


def test_grouped_local_settings_endpoint_remains_available_during_tempo_wait(tmp_path):
    from streammuse.application.rap.settings import RapSettings, RapSettingsService

    entered, release = Event(), Event()

    def apply(current, updated, changed):
        if "tempo_bpm" in changed:
            entered.set()
            assert release.wait(2)
            return {"effective_from_bar": 5}
        return {"mix_revision": 1}

    runtime = RapAudioDemoDependencies(
        tempo=Tempo(60, 4, 4), controller=SimpleNamespace(close=lambda: None), coordinator=None,
        playback=SimpleNamespace(state=PlaybackState.RUNNING, close=lambda: None),
        publisher=object(), dispatcher=SimpleNamespace(flush_and_close=lambda: None), session_dir=tmp_path,
    )
    runtime.settings_service = RapSettingsService(
        RapSettings(), state=lambda: "running", apply=apply,
        capabilities=lambda: {"tempo_bpm": True, "drum_gain": True},
    )
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            remote = asyncio.create_task(client.post("/api/control/settings/generation", json={"tempo_bpm": 80}))
            try:
                assert await asyncio.to_thread(entered.wait, 0.5)
                response = await asyncio.wait_for(client.post("/api/control/settings/local", json={"drum_gain": 0.2}), timeout=0.5)
                assert response.status_code == 200, response.text
                assert response.json()["values"]["drum_gain"] == 0.2
                assert not remote.done()
                wrong_group = await client.post("/api/control/settings/local", json={"tempo_bpm": 70})
                assert wrong_group.status_code == 422
            finally:
                release.set()
                response = await remote
            assert response.status_code == 200, response.text
            current = await client.get("/api/control/settings/local")
            assert current.json()["values"]["tempo_bpm"] == 80
            assert current.json()["values"]["drum_gain"] == 0.2

    asyncio.run(exercise())


def test_slow_plan_send_does_not_block_local_settings_http(tmp_path):
    entered, release = Event(), Event()
    runtime = ControllableFakeRuntime(tmp_path)

    def edit(**kwargs):
        entered.set()
        release.wait(1)
        return SimpleNamespace(edit_seq=1, from_bar=0)

    runtime.controller = SimpleNamespace(update_plan=edit, next_editable_bar=0)
    runtime.update_settings = lambda patch, group=None: {"group": group, "values": patch}
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            request = asyncio.create_task(client.post("/api/control/plan", json={"topic": "rain"}))
            try:
                assert await asyncio.to_thread(entered.wait, 0.5)
                local = await asyncio.wait_for(client.post("/api/control/settings/local", json={"drum_gain": 0.2}), timeout=0.25)
                assert local.status_code == 200
                assert not request.done(), "plan I/O blocked the API event loop"
            finally:
                release.set()
                response = await request
            assert response.status_code == 200, response.text

    asyncio.run(exercise())


@pytest.mark.parametrize("concurrent_control", [None, "start", "stop"])
def test_reset_teardown_and_waiting_controls_do_not_block_api_event_loop(tmp_path, concurrent_control):
    entered, release, reset_finished = Event(), Event(), Event()
    runtime = ControllableFakeRuntime(tmp_path)

    def reset():
        entered.set()
        release.wait(1)
        reset_finished.set()

    runtime.reset = reset
    runtime.start = lambda: None
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pending = [asyncio.create_task(client.post("/api/control/reset"))]
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                if concurrent_control is not None:
                    pending.append(asyncio.create_task(client.post(f"/api/control/{concurrent_control}")))
                    await asyncio.sleep(0)
                state = await asyncio.wait_for(client.get("/api/state"), timeout=0.25)
                assert state.status_code == 200
                assert not reset_finished.is_set(), "API loop stalled until blocking reset completed"
            finally:
                release.set()
                results = await asyncio.gather(*pending)
            assert results[0].status_code == 200

    asyncio.run(exercise())


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
    state_script = client.get("/static/js/rap-demo-state.js")

    assert index.status_code == css.status_code == script.status_code == state_script.status_code == 200
    assert "StreamMUSE Rap Lab" in index.text
    for element_id in (
        "follow-live",
        "current-syllable",
        "next-bar",
        "model-health",
        "last-error",
        "flow-track",
        "flow-exact",
        "flow-provenance",
        "context-lines",
        "prompt",
        "raw-response",
        "response-status",
        "selected-score",
        "candidate-table",
        "candidate-sort",
        "generator-error-rate",
        "generation-p50",
        "generation-p95",
        "history-rows",
        "event-console",
        "event-announcer",
        "start-runtime",
        "stop-runtime",
        "reset-runtime",
        "audio-state",
        "audio-warning-rows",
        "remote-chunk-panel",
        "remote-request-id",
        "remote-renderer",
        "remote-state",
        "remote-lines",
        "remote-flows",
        "remote-candidate-counts",
        "remote-scores",
        "remote-prompt-summary",
        "remote-context-lines",
        "remote-stage-timings",
        "remote-alignment",
        "remote-stretch-warnings",
        "remote-hashes",
        "remote-artifacts",
        "remote-failure-reason",
    ):
        assert f'id="{element_id}"' in index.text

    runtime_controls = index.text.split('<div class="runtime-controls"', maxsplit=1)[1].split("</div>", maxsplit=1)[0]
    assert runtime_controls.count("<button") == 3
    assert all(label in runtime_controls for label in ("Start", "Stop", "Reset"))

    for css_contract in (
        "--brand: #e91e63",
        ".flow-rail",
        ".candidate-record",
        ".mobile-sort",
        ".remote-chunk-grid",
        ":focus-visible",
        "@media (max-width: 760px)",
        "@media (prefers-reduced-motion: reduce)",
    ):
        assert css_contract in css.text

    for renderer_contract in (
        "renderQueueAndHealth",
        "renderFlowFacts",
        "renderResponse",
        "renderSelectedScore",
        "renderCurrentSyllable",
        "generationFailure",
        "createEventRow",
        "renderedEventSequences",
        "monitor.followLive",
        "component.weight",
        "component.contribution",
        "generator_error",
        "raw_response",
        "context_lines",
        "research_metrics",
        "sendControl",
        "renderAudio",
    ):
        assert renderer_contract in script.text
    assert "textContent" in script.text
    assert "innerHTML" not in script.text
    for state_contract in (
        "projectRemoteChunk",
        "sanitizeChunkPayload",
        "renderRemoteChunk",
        "remote-candidate-counts",
        "remote-stage-timings",
        "remote-alignment",
    ):
        assert state_contract in state_script.text
    assert "textContent" in state_script.text
    assert "innerHTML" not in state_script.text


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


def test_audio_runtime_waits_for_start_and_controls_are_restart_safe(tmp_path: Path) -> None:
    runtime = ControllableFakeRuntime(tmp_path / "rap-test")
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())

    with TestClient(app) as client:
        assert runtime.start_calls == 0
        assert client.post("/api/control/stop").status_code == 409
        assert client.post("/api/control/start").json() == {"state": "priming"}
        assert client.post("/api/control/start").status_code == 409
        assert runtime.started.wait(timeout=1)
        assert client.post("/api/control/stop").json() == {"state": "stop_requested"}
        assert runtime.stop_calls == 1
        runtime.finish_bar()
        assert client.post("/api/control/reset").json() == {"state": "stopped"}
        assert runtime.reset_calls == 1
        runtime.finished = Event()
        assert client.post("/api/control/start").status_code == 202
        deadline = time.monotonic() + 1.0
        while runtime.start_calls != 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runtime.start_calls == 2
        runtime.finish_bar()


def test_concurrent_http_start_and_stop_requests_return_one_transition_and_one_conflict(tmp_path: Path) -> None:
    runtime = ControllableFakeRuntime(tmp_path / "rap-test")
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())

    with TestClient(app) as client:
        start_results: list[int] = []
        starters = [Thread(target=lambda: start_results.append(client.post("/api/control/start").status_code)) for _ in range(2)]
        for starter in starters:
            starter.start()
        for starter in starters:
            starter.join(timeout=1)
        assert sorted(start_results) == [202, 409]
        assert runtime.started.wait(timeout=1)
        stop_results: list[int] = []
        stoppers = [Thread(target=lambda: stop_results.append(client.post("/api/control/stop").status_code)) for _ in range(2)]
        for stopper in stoppers:
            stopper.start()
        for stopper in stoppers:
            stopper.join(timeout=1)
        assert sorted(stop_results) == [202, 409]
        assert runtime.start_calls == 1
        assert runtime.stop_calls == 1
        runtime.finish_bar()


def test_control_endpoints_drive_the_concrete_restartable_audio_runtime(tmp_path: Path) -> None:
    class Controller:
        def __init__(self, playback: object) -> None:
            self.playback = playback
            self.starts = 0
            self.stops = 0
            self.resets = 0
            self.closed = 0

        def start(self) -> None:
            self.starts += 1
            self.playback.state = PlaybackState.PRIMING

        def resume_audio(self, _bar: int) -> None:
            self.playback.state = PlaybackState.PRIMING

        def resume_after_stop(self) -> None:
            return None

        def request_stop(self, *, successor_bar: int) -> None:
            assert successor_bar == 1
            self.stops += 1

        def reset(self) -> int:
            self.resets += 1
            return self.resets

        def close(self) -> None:
            self.closed += 1

    class Playback:
        def __init__(self) -> None:
            self.state = PlaybackState.STOPPED
            self.current_tick: int | None = None
            self.next_start_bar = 0
            self.stop_successor_bar = 1
            self.started = Event()
            self.starts = 0
            self.stops = 0
            self.resets = 0
            self.closed = 0

        def start(self) -> None:
            self.starts += 1
            self.state = PlaybackState.RUNNING
            self.started.set()

        def request_stop(self) -> int:
            self.stops += 1
            self.state = PlaybackState.STOPPED
            return 1

        def wait(self, timeout: float | None = None) -> None:
            return None

        def reset(self, *, coordinator_epoch: int) -> None:
            assert self.state == PlaybackState.STOPPED
            self.resets += 1
            assert coordinator_epoch == self.resets

        def close(self) -> None:
            self.closed += 1
            self.state = PlaybackState.CLOSED

    class Coordinator:
        def close(self) -> None:
            return None

    playback = Playback()
    controller = Controller(playback)
    monitor = RapStateProjector()
    publisher = RapEventPublisher("audio-control")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(monitor,))
    dispatcher.start()
    runtime = RapAudioDemoDependencies(
        tempo=Tempo(60.0, 4, 4),
        controller=controller,
        coordinator=Coordinator(),
        playback=playback,
        publisher=publisher,
        dispatcher=dispatcher,
        session_dir=tmp_path / "rap-test",
        session_metadata={"audio": {"audio_device": "Test output", "artifact_paths": {"wav": "test.wav"}}},
    )
    app = create_app(runtime=runtime, projector=monitor, websocket_queue=Queue())

    with TestClient(app) as client:
        assert client.post("/api/control/start").status_code == 202
        assert playback.started.wait(timeout=1)
        audio = client.get("/api/state").json()["audio"]
        assert audio["device"] == "Test output"
        assert audio["recording_path"] == "test.wav"
        assert client.post("/api/control/start").status_code == 409
        assert client.post("/api/control/stop").status_code == 202
        deadline = time.monotonic() + 1.0
        while playback.state != PlaybackState.STOPPED and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.post("/api/control/reset").status_code == 200
        playback.started = Event()
        assert client.post("/api/control/start").status_code == 202
        assert playback.started.wait(timeout=1)
        assert client.post("/api/control/stop").status_code == 202

    assert controller.starts == 2
    assert controller.stops == 2
    assert controller.resets == 1
    assert playback.starts == 2
    assert playback.resets == 1
    assert playback.closed == 1


def test_control_endpoints_reject_unsupported_text_runtime(tmp_path: Path) -> None:
    app, _, _ = _app(tmp_path)

    with TestClient(app) as client:
        assert client.post("/api/control/start").status_code == 404
        assert client.post("/api/control/stop").status_code == 404
        assert client.post("/api/control/reset").status_code == 404


def test_reset_serializes_a_concurrent_start_request(tmp_path: Path) -> None:
    class ResetBlockingRuntime(ControllableFakeRuntime):
        def __init__(self, session_dir: Path) -> None:
            super().__init__(session_dir)
            self.reset_entered = Event()
            self.release_reset = Event()

        def reset(self) -> None:
            self.reset_entered.set()
            assert self.release_reset.wait(timeout=1)
            super().reset()

    runtime = ResetBlockingRuntime(tmp_path / "rap-test")
    lifecycle = _MonitorLifecycle(runtime, FakeProjector(), Queue())
    reset_done = Event()
    start_result: list[dict[str, str]] = []
    reset_thread = Thread(target=lambda: (lifecycle.reset_control(), reset_done.set()))
    start_thread = Thread(target=lambda: start_result.append(lifecycle.start_control()))

    reset_thread.start()
    assert runtime.reset_entered.wait(timeout=1)
    start_thread.start()
    time.sleep(0.02)
    assert start_thread.is_alive()
    runtime.release_reset.set()
    reset_thread.join(timeout=1)
    start_thread.join(timeout=1)
    runtime.finish_bar()

    assert reset_done.is_set()
    assert start_result == [{"state": "priming"}]


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


class _PlanController:
    next_editable_bar = 7

    def __init__(self) -> None:
        self.edits = []

    def update_plan(self, *, from_bar, topic=None, template_id=None):
        if template_id == "missing":
            raise ValueError("unknown flow template: missing")
        self.edits.append((from_bar, topic, template_id))
        return SimpleNamespace(edit_seq=len(self.edits), from_bar=from_bar)


def test_plan_control_forwards_live_edits_to_a_v3_controller(tmp_path: Path) -> None:
    app, runtime, _ = _app(tmp_path)
    runtime.controller = _PlanController()
    with TestClient(app) as client:
        response = client.post("/api/control/plan", json={"topic": "deep sea"})
        assert response.status_code == 200
        assert response.json() == {"edit_seq": 1, "from_bar": 7, "effective_from_bar": 7}
        assert runtime.controller.edits == [(7, "deep sea", None)]
        assert client.post("/api/control/plan", json={}).status_code == 422
        assert client.post("/api/control/plan", json={"template_id": "missing"}).status_code == 422


def test_plan_control_is_unavailable_without_a_v3_controller(tmp_path: Path) -> None:
    app, _, _ = _app(tmp_path)
    with TestClient(app) as client:
        assert client.post("/api/control/plan", json={"topic": "x"}).status_code == 404


def test_plan_controls_are_discoverable_before_start_with_full_flow_previews(tmp_path: Path) -> None:
    runtime = ControllableFakeRuntime(tmp_path)
    runtime.controller = _PlanController()
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())

    with TestClient(app) as client:
        response = client.get("/api/control/plan")
        assert response.status_code == 200
        payload = response.json()
        assert payload["supported"] is True
        assert payload["state"] == "stopped"
        assert payload["next_editable_bar"] == 7
        assert payload.get("last_edit") is None
        assert "current_plan" not in payload
        assert runtime.start_calls == 0
        templates = {item["template_id"]: item for item in payload["templates"]}
        assert len(templates) == 6
        assert all(key.startswith("mcflow_") for key in templates)
        flow = next(iter(templates.values()))
        assert flow["ticks_per_beat"] == flow["beats_per_bar"] == 4
        assert all(len(item["slots"]) >= 8 for item in templates.values())
        assert sum(item["slots"][0]["tick_in_bar"] > 0 for item in templates.values()) >= 2
        assert all(item["provenance"]["kind"] == "mcflow_extracted_anonymous" for item in templates.values())
        assert len(flow["provenance"]["source_hash"]) == 64


def test_plan_controls_list_the_controller_catalog_not_unavailable_builtins(tmp_path: Path) -> None:
    app, runtime, _ = _app(tmp_path)
    runtime.controller = _PlanController()
    runtime.controller._templates = TemplateCatalog((BUILTIN_TEMPLATES.get("baseline_straight_9"),))

    with TestClient(app) as client:
        response = client.get("/api/control/plan")

    assert response.status_code == 200
    assert [item["template_id"] for item in response.json()["templates"]] == ["baseline_straight_9"]


@pytest.mark.parametrize("controller", [None, SimpleNamespace(), SimpleNamespace(update_plan=None)])
def test_plan_discovery_disables_unsupported_controllers(tmp_path: Path, controller: object) -> None:
    runtime = ControllableFakeRuntime(tmp_path)
    runtime.controller = controller
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())

    with TestClient(app) as client:
        response = client.get("/api/control/plan")
        assert response.status_code == 200
        assert response.json() == {
            "supported": False, "templates": [], "next_editable_bar": None, "state": "stopped",
        }
        assert client.post("/api/control/plan", json={"topic": "deep sea"}).status_code == 404


def test_plan_discovery_uses_the_atomic_controller_snapshot(tmp_path: Path) -> None:
    class Controller(_PlanController):
        @property
        def plan_control_state(self):
            return {
                "next_editable_bar": 10,
                "last_edit": {"edit_seq": 2, "from_bar": 7, "effective_from_bar": 8,
                              "topic": "deep sea", "template_id": None},
            }

    app, runtime, _ = _app(tmp_path)
    runtime.controller = Controller()
    runtime.control_state = PlaybackState.RUNNING

    with TestClient(app) as client:
        response = client.get("/api/control/plan")

    assert response.status_code == 200
    assert response.json()["state"] == "running"
    assert response.json()["next_editable_bar"] == 10
    assert response.json()["last_edit"] == {
        "edit_seq": 2, "from_bar": 7, "effective_from_bar": 8,
        "topic": "deep sea", "template_id": None,
    }


@pytest.mark.parametrize("body", [b"", b"{", b'{"topic":', b'\xff', b"null", b"[]", b'"topic"', b"3", b"true"])
def test_plan_control_rejects_malformed_or_nonobject_json(tmp_path: Path, body: bytes) -> None:
    app, runtime, _ = _app(tmp_path)
    runtime.controller = _PlanController()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/control/plan", content=body, headers={"Content-Type": "application/json"})

    assert response.status_code == 422
    assert response.json()["detail"]
    assert runtime.controller.edits == []


@pytest.mark.parametrize("field", ["topic", "template_id"])
@pytest.mark.parametrize("value", ["", " \t\n", 1, 1.5, True, False, [], {}])
def test_plan_control_rejects_invalid_topic_or_template(tmp_path: Path, field: str, value: object) -> None:
    app, runtime, _ = _app(tmp_path)
    runtime.controller = _PlanController()
    payload = {"topic": "deep sea", "template_id": "baseline_straight_9", field: value}

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/control/plan", json=payload)

    assert response.status_code == 422
    assert runtime.controller.edits == []


@pytest.mark.parametrize("value", [True, False, -1, 1.5, 2.0, [], {}, "", "later", "1.5", "-2"])
def test_plan_control_rejects_invalid_from_bar(tmp_path: Path, value: object) -> None:
    app, runtime, _ = _app(tmp_path)
    runtime.controller = _PlanController()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/control/plan", json={"topic": "deep sea", "from_bar": value})

    assert response.status_code == 422
    assert runtime.controller.edits == []


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_plan_control_rejects_nonfinite_from_bar(tmp_path: Path, value: str) -> None:
    app, runtime, _ = _app(tmp_path)
    runtime.controller = _PlanController()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/control/plan", content='{"topic":"deep sea","from_bar":' + value + '}',
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422
    assert runtime.controller.edits == []


@pytest.mark.parametrize("error_type", [ValueError, KeyError, TypeError])
def test_plan_control_converts_controller_validation_errors_to_422(tmp_path: Path, error_type: type) -> None:
    class Controller(_PlanController):
        def update_plan(self, **kwargs):
            raise error_type("unknown flow template: missing")

    app, runtime, _ = _app(tmp_path)
    runtime.controller = Controller()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/control/plan", json={"template_id": "missing"})

    assert response.status_code == 422
    assert "unknown flow template" in response.json()["detail"]


@pytest.mark.parametrize("from_bar, requested, effective", [(None, 7, 7), (0, 0, 7), (12, 12, 12), ("9", 9, 9)])
def test_plan_control_preserves_explicit_bar_and_partial_edit_compatibility(
    tmp_path: Path, from_bar: object, requested: int, effective: int,
) -> None:
    app, runtime, _ = _app(tmp_path)
    runtime.controller = _PlanController()

    with TestClient(app) as client:
        response = client.post("/api/control/plan", json={
            "topic": None, "template_id": "baseline_straight_9", "from_bar": from_bar,
        })

    assert response.status_code == 200
    assert response.json() == {"edit_seq": 1, "from_bar": requested, "effective_from_bar": effective}
    assert runtime.controller.edits == [(requested, None, "baseline_straight_9")]


def test_plan_control_returns_matching_effective_target_not_the_later_playhead(tmp_path: Path) -> None:
    class Controller(_PlanController):
        last_edit = None

        @property
        def plan_control_state(self):
            return {"next_editable_bar": self.next_editable_bar, "last_edit": self.last_edit}

        def update_plan(self, **kwargs):
            edit = super().update_plan(**kwargs)
            self.last_edit = {"edit_seq": edit.edit_seq, "from_bar": edit.from_bar,
                              "effective_from_bar": 9, "topic": kwargs["topic"], "template_id": None}
            self.next_editable_bar = 12
            return edit

    app, runtime, _ = _app(tmp_path)
    runtime.controller = Controller()

    with TestClient(app) as client:
        response = client.post("/api/control/plan", json={"topic": "deep sea"})
        discovery = client.get("/api/control/plan")

    assert response.status_code == 200
    assert response.json() == {"edit_seq": 1, "from_bar": 7, "effective_from_bar": 9}
    assert discovery.json()["next_editable_bar"] == 12
    assert discovery.json()["last_edit"]["effective_from_bar"] == 9


def test_plan_control_falls_back_to_the_updated_bar_without_a_control_snapshot(tmp_path: Path) -> None:
    class Controller(_PlanController):
        def update_plan(self, **kwargs):
            self.next_editable_bar = 9
            return super().update_plan(**kwargs)

    app, runtime, _ = _app(tmp_path)
    runtime.controller = Controller()

    with TestClient(app) as client:
        response = client.post("/api/control/plan", json={"topic": "deep sea"})

    assert response.status_code == 200
    assert response.json() == {"edit_seq": 1, "from_bar": 7, "effective_from_bar": 9}


def test_plan_control_does_not_use_an_unrelated_last_edit_target(tmp_path: Path) -> None:
    class Controller(_PlanController):
        @property
        def plan_control_state(self):
            return {"next_editable_bar": 8,
                    "last_edit": {"edit_seq": 99, "from_bar": 20, "effective_from_bar": 20}}

    app, runtime, _ = _app(tmp_path)
    runtime.controller = Controller()

    with TestClient(app) as client:
        response = client.post("/api/control/plan", json={"topic": "deep sea"})

    assert response.status_code == 200
    assert response.json() == {"edit_seq": 1, "from_bar": 8, "effective_from_bar": 8}
