"""Contract tests for the read-only realtime rap research monitor."""

from __future__ import annotations

import json
import asyncio
import time
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from types import MappingProxyType
from typing import Any

from fastapi.testclient import TestClient
import pytest

from streammuse.application.rap.runtime import RapAudioDemoDependencies
from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher, RapStateProjector
from streammuse.domain.rap import PlaybackState
from streammuse.domain.timing import Tempo
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
