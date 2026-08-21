"""Background FastAPI server for the read-only interactive task viewer."""

from __future__ import annotations

import asyncio
import json
import secrets
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from streammuse.application.tasks import TaskWebConfig
from streammuse.infrastructure.task_view import QueueTaskEventSink


class TaskWebStartupError(RuntimeError):
    """The required task viewer could not start before gameplay."""


class TaskWebServer:
    def __init__(
        self,
        *,
        config: TaskWebConfig,
        sink: QueueTaskEventSink,
        session_id: str,
    ) -> None:
        if not config.enabled:
            raise ValueError("TaskWebServer requires an enabled TaskWebConfig")
        self.config = config
        self.sink = sink
        self.session_id = session_id
        self.token = secrets.token_urlsafe(24)
        self.viewer_ready = threading.Event()
        self.server_ready = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._socket: socket.socket | None = None
        self._startup_error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connections: set[WebSocket] = set()
        self._active_subscription_ids: set[str] = set()
        self._inflight = 0
        self._closed = False
        self._close_lock = threading.Lock()
        self.app = self._build_app()

    @property
    def url(self) -> str:
        display_host = "127.0.0.1" if self.config.host == "0.0.0.0" else self.config.host
        host = f"[{display_host}]" if ":" in display_host else display_host
        return f"http://{host}:{self.config.port}/?token={self.token}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._socket = self._bind_socket()
        uvicorn_config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="warning",
            access_log=False,
            lifespan="on",
        )
        self._server = uvicorn.Server(uvicorn_config)
        self._thread = threading.Thread(
            target=self._run,
            name="streammuse-task-web",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + self.config.server_start_timeout_s
        while time.monotonic() < deadline:
            if bool(self._server.started):
                self.server_ready.set()
                return
            if self._stopped.wait(0.01):
                break
        error = self._startup_error
        self.close()
        if error is not None:
            raise TaskWebStartupError(f"task Web server failed to start: {error}") from error
        raise TaskWebStartupError("task Web server did not become ready before timeout")

    def wait_for_viewer(self) -> None:
        while not self.viewer_ready.wait(0.1):
            if self._stopped.is_set():
                error = self._startup_error
                if error is not None:
                    raise TaskWebStartupError(
                        f"task Web server stopped before viewer ready: {error}"
                    ) from error
                raise TaskWebStartupError("task Web server stopped before viewer ready")

    def flush(self, timeout_s: float | None = None) -> bool:
        loop = self._loop
        if loop is None or loop.is_closed() or self._stopped.is_set():
            return not self.sink.has_pending()
        timeout = self.config.flush_timeout_s if timeout_s is None else float(timeout_s)
        future = asyncio.run_coroutine_threadsafe(self._flush_async(timeout), loop)
        try:
            return bool(future.result(timeout=max(0.05, timeout + 0.1)))
        except Exception:
            future.cancel()
            return False

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self.flush()
        loop = self._loop
        if loop is not None and not loop.is_closed() and not self._stopped.is_set():
            future = asyncio.run_coroutine_threadsafe(self._close_connections(), loop)
            try:
                future.result(timeout=self.config.shutdown_timeout_s)
            except Exception:
                future.cancel()
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=self.config.shutdown_timeout_s)
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass

    def _bind_socket(self) -> socket.socket:
        family = socket.AF_INET6 if ":" in self.config.host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.config.host, self.config.port))
            sock.listen(128)
            sock.setblocking(False)
            return sock
        except OSError as exc:
            sock.close()
            raise TaskWebStartupError(
                f"could not bind task Web UI to {self.config.host}:{self.config.port}: {exc}"
            ) from exc

    def _run(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            if self._server is None or self._socket is None:
                raise RuntimeError("task Web server was not initialized")
            self._loop.run_until_complete(self._server.serve(sockets=[self._socket]))
        except BaseException as exc:
            self._startup_error = exc
        finally:
            loop = self._loop
            if loop is not None and not loop.is_closed():
                loop.close()
            self._stopped.set()

    def _build_app(self) -> FastAPI:
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        static_root = Path(__file__).with_name("static")
        app.mount("/css", StaticFiles(directory=static_root / "css"), name="task-web-css")
        app.mount("/js", StaticFiles(directory=static_root / "js"), name="task-web-js")

        @app.middleware("http")
        async def security_headers(request, call_next):  # type: ignore[no-untyped-def]
            response = await call_next(request)
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self' ws: wss:; img-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'"
            )
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Cache-Control"] = "no-store"
            return response

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(static_root / "index.html")

        @app.get("/healthz")
        async def health() -> JSONResponse:
            return JSONResponse({"status": "ok", "session_id": self.session_id})

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            if not self._websocket_authorized(websocket):
                await websocket.close(code=1008)
                return
            subscription, initial = self.sink.subscribe()
            await websocket.accept()
            try:
                await websocket.send_json(initial)
                self._connections.add(websocket)
                self._active_subscription_ids.add(subscription.id)
                sender = asyncio.create_task(self._sender(websocket, subscription))
                receiver = asyncio.create_task(self._receiver(websocket, subscription.id))
                done, pending = await asyncio.wait(
                    {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*done, *pending, return_exceptions=True)
            except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
                pass
            finally:
                self._active_subscription_ids.discard(subscription.id)
                self._connections.discard(websocket)
                self.sink.unsubscribe(subscription)

        return app

    def _websocket_authorized(self, websocket: WebSocket) -> bool:
        if websocket.query_params.get("token") != self.token:
            return False
        origin = websocket.headers.get("origin")
        if not origin:
            return False
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"}:
            return False
        origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if origin_port != self.config.port:
            return False
        if self.config.host in {"0.0.0.0", "::"} and self.config.allow_remote:
            return bool(parsed.hostname)
        allowed_hosts = {self.config.host.lower()}
        if self.config.host in {"127.0.0.1", "::1", "localhost"}:
            allowed_hosts.update({"127.0.0.1", "::1", "localhost"})
        return bool(parsed.hostname and parsed.hostname.lower() in allowed_hosts)

    async def _sender(self, websocket: WebSocket, subscription) -> None:  # type: ignore[no-untyped-def]
        while True:
            items = self.sink.drain(subscription)
            if not items:
                await asyncio.sleep(0.01)
                continue
            for item in items:
                self._inflight += 1
                try:
                    await websocket.send_json(item)
                finally:
                    self._inflight -= 1

    async def _receiver(self, websocket: WebSocket, subscription_id: str) -> None:
        while True:
            try:
                message = await websocket.receive_text()
            except WebSocketDisconnect:
                return
            try:
                payload = json.loads(message)
            except (TypeError, ValueError):
                continue
            if (
                isinstance(payload, dict)
                and payload.get("type") == "viewer_ready"
                and payload.get("session_id") == self.session_id
                and subscription_id in self._active_subscription_ids
            ):
                self.viewer_ready.set()

    async def _flush_async(self, timeout_s: float) -> bool:
        deadline = asyncio.get_running_loop().time() + max(0.0, timeout_s)
        while self.sink.has_pending() or self._inflight:
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(0.005)
        return True

    async def _close_connections(self) -> None:
        connections = tuple(self._connections)
        if connections:
            await asyncio.gather(
                *(connection.close(code=1001) for connection in connections),
                return_exceptions=True,
            )
