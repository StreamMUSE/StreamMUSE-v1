"""FastAPI observer for the canonical realtime rap research event stream."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from contextlib import asynccontextmanager
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from queue import Empty
from threading import Lock, Thread
from typing import Any, AsyncIterator, Mapping

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


STATIC_DIR = Path(__file__).parent / "static"
_POLL_INTERVAL_S = 0.02
_MAX_EVENTS_PER_POLL = 256


class _ConnectionPool:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, snapshot: object) -> None:
        await websocket.accept()
        async with self._lock:
            await websocket.send_json({"type": "snapshot", "payload": snapshot})
            self._connections.append(websocket)

    async def remove(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)

    async def send(self, message: object) -> None:
        async with self._lock:
            disconnected: list[WebSocket] = []
            for websocket in tuple(self._connections):
                try:
                    await websocket.send_json(message)
                except Exception:
                    disconnected.append(websocket)
            for websocket in disconnected:
                self._connections.remove(websocket)

    async def has_connections(self) -> bool:
        async with self._lock:
            return bool(self._connections)


class _MonitorLifecycle:
    def __init__(self, runtime: object, projector: object, websocket_queue: object) -> None:
        self.runtime = runtime
        self.projector = projector
        self.websocket_queue = websocket_queue
        self.connections = _ConnectionPool()
        self._runtime_thread: Thread | None = None
        self._broadcaster_stop: asyncio.Event | None = None
        self._broadcaster_task: asyncio.Task[None] | None = None
        self._shutdown_lock = Lock()
        self._closed = False

    def snapshot(self) -> object:
        snapshot = getattr(self.projector, "snapshot", None)
        if callable(snapshot):
            return _json_safe(snapshot())
        if hasattr(self.projector, "state"):
            return _json_safe(getattr(self.projector, "state"))
        raise TypeError("projector must expose snapshot() or state")

    def session(self) -> dict[str, object]:
        snapshot = self.snapshot()
        state = snapshot if isinstance(snapshot, dict) else {}
        session_dir = getattr(self.runtime, "session_dir", None)
        metadata = getattr(self.runtime, "session_metadata", state.get("session_metadata", {}))
        return {
            "session_id": state.get("session_id"),
            "session_dir": str(session_dir) if session_dir is not None else None,
            "metadata": _json_safe(metadata),
        }

    async def start(self) -> None:
        self._broadcaster_stop = asyncio.Event()
        self._broadcaster_task = asyncio.create_task(self._broadcast_loop(), name="rap-websocket-broadcaster")
        self._runtime_thread = Thread(target=self._start_runtime, name="streammuse-rap-runtime", daemon=True)
        self._runtime_thread.start()

    async def close(self) -> None:
        with self._shutdown_lock:
            if self._closed:
                return
            self._closed = True
        close = getattr(self.runtime, "close", None)
        if callable(close):
            close()
        if self._runtime_thread is not None:
            await asyncio.to_thread(self._runtime_thread.join, 1.0)
        if self._broadcaster_stop is not None:
            self._broadcaster_stop.set()
        if self._broadcaster_task is not None:
            try:
                await asyncio.wait_for(self._broadcaster_task, timeout=1.0)
            except asyncio.TimeoutError:
                self._broadcaster_task.cancel()

    def _start_runtime(self) -> None:
        start = getattr(self.runtime, "start", None)
        if callable(start):
            start()
            return
        run = getattr(self.runtime, "run", None)
        if not callable(run):
            return
        parameters = inspect.signature(run).parameters
        if "max_bars" in parameters:
            run(max_bars=0)
        else:
            run()

    async def _broadcast_loop(self) -> None:
        assert self._broadcaster_stop is not None
        while not self._broadcaster_stop.is_set():
            if await self.connections.has_connections():
                for item in self._drain_queue():
                    await self.connections.send({"type": "event", "payload": _event_payload(item)})
            try:
                await asyncio.wait_for(self._broadcaster_stop.wait(), timeout=_POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                pass

    def _drain_queue(self) -> list[object]:
        items: list[object] = []
        get_nowait = getattr(self.websocket_queue, "get_nowait", None)
        if not callable(get_nowait):
            return items
        for _ in range(_MAX_EVENTS_PER_POLL):
            try:
                items.append(get_nowait())
            except Empty:
                break
        return items


def create_app(*, runtime: object, projector: object, websocket_queue: object) -> FastAPI:
    """Create an isolated read-only monitor around one rap runtime."""

    lifecycle = _MonitorLifecycle(runtime, projector, websocket_queue)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await lifecycle.start()
        try:
            yield
        finally:
            await lifecycle.close()

    app = FastAPI(title="StreamMUSE Rap Research Monitor", lifespan=lifespan)
    app.state.monitor = lifecycle
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="rap-demo-static")

    @app.get("/")
    async def index() -> FileResponse:
        path = STATIC_DIR / "index.html"
        if not path.exists():
            raise HTTPException(status_code=500, detail="rap monitor index is missing")
        return FileResponse(path)

    @app.get("/api/state")
    async def state() -> object:
        return lifecycle.snapshot()

    @app.get("/api/session")
    async def session() -> object:
        return lifecycle.session()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await lifecycle.connections.connect(websocket, lifecycle.snapshot())
        try:
            while True:
                await websocket.receive()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await lifecycle.connections.remove(websocket)

    return app


def _event_payload(value: object) -> object:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {"raw": value}
    return _json_safe(value)


def _json_safe(value: object, *, _seen: set[int] | None = None) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return _json_safe(value.value, _seen=_seen)
    if isinstance(value, (Path, datetime, date)):
        return str(value)

    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return "<cycle>"

    if is_dataclass(value) and not isinstance(value, type):
        seen.add(identity)
        try:
            return {field.name: _json_safe(getattr(value, field.name), _seen=seen) for field in fields(value)}
        finally:
            seen.remove(identity)
    if isinstance(value, Mapping):
        seen.add(identity)
        try:
            return {str(key): _json_safe(item, _seen=seen) for key, item in value.items()}
        finally:
            seen.remove(identity)
    if isinstance(value, (list, tuple, set, frozenset)):
        seen.add(identity)
        try:
            return [_json_safe(item, _seen=seen) for item in value]
        finally:
            seen.remove(identity)
    return repr(value)
