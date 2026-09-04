"""StreamMUSE web viewer with explicit realtime session lifecycle controls."""

from __future__ import annotations

import asyncio
import signal
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from streammuse.application.config import ApplicationConfig
from streammuse.application.runtime import RuntimeSession, RuntimeSessionBuilder
from streammuse.infrastructure.output import CompositeOutputSink, WebSocketOutputSink
from streammuse.presentation.cli.config_parser import args_to_config, parse_args


STATIC_DIR = Path(__file__).parent / "static"
MIN_SESSION_BPM = 30
MAX_SESSION_BPM = 300


class StartSessionRequest(BaseModel):
    """Optional settings applied to one newly-created Web session."""

    bpm: int | None = Field(
        default=None,
        ge=MIN_SESSION_BPM,
        le=MAX_SESSION_BPM,
    )

# ---------------------------------------------------------------------------
# Module-level state. Set once at boot, read by request handlers.
# ---------------------------------------------------------------------------

_ws_sink: Optional[WebSocketOutputSink] = None
_runtime: Optional[RuntimeSession] = None
_service: Optional[object] = None
_composite_sink: Optional[CompositeOutputSink] = None
_connections: List[WebSocket] = []
_broadcaster_stop: Optional[asyncio.Event] = None
_broadcaster_task: Optional[asyncio.Task] = None
_config: Optional[ApplicationConfig] = None
_log_dir = "logs"
_last_session_dir: Optional[str] = None
_lifecycle_state = "idle"
_lifecycle_lock = threading.RLock()


# ---------------------------------------------------------------------------
# WebSocket broadcaster — drains _ws_sink's queue and pushes to all clients.
# ---------------------------------------------------------------------------

async def _broadcaster_loop(poll_interval_s: float = 0.02) -> None:
    """Broadcast from whichever session sink is currently attached."""
    assert _broadcaster_stop is not None
    while not _broadcaster_stop.is_set():
        sink = _ws_sink
        messages = sink.get_pending_messages() if sink is not None else []
        if messages:
            dead: List[WebSocket] = []
            for ws in list(_connections):
                try:
                    for msg in messages:
                        await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                if ws in _connections:
                    _connections.remove(ws)
        try:
            await asyncio.wait_for(_broadcaster_stop.wait(), timeout=poll_interval_s)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _broadcaster_stop, _broadcaster_task
    _broadcaster_stop = asyncio.Event()
    _broadcaster_task = asyncio.create_task(_broadcaster_loop())
    try:
        yield
    finally:
        if _broadcaster_stop is not None:
            _broadcaster_stop.set()
        if _broadcaster_task is not None:
            try:
                await asyncio.wait_for(_broadcaster_task, timeout=1.0)
            except asyncio.TimeoutError:
                _broadcaster_task.cancel()
        await asyncio.to_thread(_shutdown_service)


def _configure_server(*, config: ApplicationConfig, log_dir: str) -> None:
    global _config, _log_dir
    with _lifecycle_lock:
        _config = config
        _log_dir = str(log_dir)


def _runtime_status() -> dict[str, object]:
    with _lifecycle_lock:
        running = _runtime is not None and _runtime.running
        state = _lifecycle_state
        if state == "running" and not running:
            state = "idle"
        configured_bpm = (
            int(round(float(_config.tempo.bpm)))
            if _config is not None
            else None
        )
        active_bpm = (
            int(_runtime.config.tempo.bpm)
            if running and _runtime is not None
            else None
        )
        return {
            "is_running": bool(running),
            "state": state,
            "session_dir": _last_session_dir,
            "configured_bpm": configured_bpm,
            "active_bpm": active_bpm,
        }


def _derive_session_config(*, bpm: int | None) -> ApplicationConfig:
    """Create an immutable per-session config from the server's base config."""
    if _config is None:
        raise RuntimeError("server not initialized")
    session_bpm = (
        int(round(float(_config.tempo.bpm)))
        if bpm is None
        else int(bpm)
    )
    session_tempo = replace(_config.tempo, bpm=session_bpm)
    return replace(_config, tempo=session_tempo)


def _start_service(*, bpm: int | None = None) -> tuple[RuntimeSession, bool]:
    """Build and start one fresh runtime; return (runtime, newly_started)."""
    global _runtime, _service, _composite_sink, _ws_sink
    global _last_session_dir, _lifecycle_state

    with _lifecycle_lock:
        if _runtime is not None and _runtime.running:
            return _runtime, False
        if _config is None:
            raise RuntimeError("server not initialized")
        if _runtime is not None:
            _stop_service_locked()

        _lifecycle_state = "starting"
        runtime: RuntimeSession | None = None
        try:
            session_config = _derive_session_config(bpm=bpm)
            runtime = RuntimeSessionBuilder(
                config=session_config,
                log_dir=_log_dir,
            ).build_web()
            ws_sink = runtime.websocket_sink
            if not isinstance(ws_sink, WebSocketOutputSink):
                raise RuntimeError("web runtime did not provide a WebSocketOutputSink")

            _runtime = runtime
            _service = runtime.service
            _composite_sink = runtime.output_sink
            _ws_sink = ws_sink
            _last_session_dir = str(runtime.session_dir)
            runtime.start(run_stop_tick=None)
            _lifecycle_state = "running"
            return runtime, True
        except Exception:
            if runtime is not None:
                try:
                    runtime.stop()
                except Exception:
                    pass
                runtime.cleanup()
            _runtime = None
            _service = None
            _composite_sink = None
            _ws_sink = None
            _lifecycle_state = "idle"
            raise


def _stop_service_locked() -> bool:
    global _runtime, _service, _composite_sink, _ws_sink, _lifecycle_state

    runtime = _runtime
    if runtime is None:
        _lifecycle_state = "idle"
        _service = None
        _composite_sink = None
        _ws_sink = None
        return False

    _lifecycle_state = "stopping"
    stop_error: Exception | None = None
    try:
        runtime.stop()
    except Exception as exc:
        stop_error = exc
    finally:
        try:
            runtime.cleanup()
        finally:
            _runtime = None
            _service = None
            _composite_sink = None
            _ws_sink = None
            _lifecycle_state = "idle"
    if stop_error is not None:
        raise stop_error
    return True


def _stop_service() -> bool:
    with _lifecycle_lock:
        return _stop_service_locked()


def _shutdown_service() -> None:
    """Stop and clean up the current session. Safe to call repeatedly."""
    try:
        _stop_service()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(title="StreamMUSE Viewer", lifespan=lifespan)

    if STATIC_DIR.exists():
        css_dir = STATIC_DIR / "css"
        js_dir = STATIC_DIR / "js"
        if css_dir.exists():
            app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
        if js_dir.exists():
            app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")

    @app.get("/")
    async def index() -> FileResponse:
        path = STATIC_DIR / "index.html"
        if not path.exists():
            raise HTTPException(status_code=500, detail="static/index.html missing")
        return FileResponse(str(path))

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        return JSONResponse(_runtime_status())

    @app.post("/api/start")
    async def api_start(request: StartSessionRequest | None = None) -> JSONResponse:
        try:
            _, newly_started = await asyncio.to_thread(
                _start_service,
                bpm=request.bpm if request is not None else None,
            )
        except Exception as exc:
            return JSONResponse(
                {"success": False, "message": str(exc)},
                status_code=500,
            )
        status = _runtime_status()
        return JSONResponse(
            {
                "success": True,
                "message": "started" if newly_started else "already running",
                **status,
            }
        )

    @app.post("/api/stop")
    async def api_stop() -> JSONResponse:
        try:
            stopped = await asyncio.to_thread(_stop_service)
        except Exception as exc:
            return JSONResponse(
                {"success": False, "message": str(exc)},
                status_code=500,
            )
        return JSONResponse(
            {
                "success": True,
                "message": "stopped" if stopped else "already stopped",
                **_runtime_status(),
            }
        )

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        _connections.append(ws)
        try:
            while True:
                # The viewer is read-only — we accept inbound messages but
                # ignore them. Keeping the receive loop alive lets the server
                # detect disconnects promptly.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            if ws in _connections:
                _connections.remove(ws)

    return app


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point for `streammuse-web`."""
    import uvicorn

    args = parse_args()
    config = args_to_config(args)

    _configure_server(config=config, log_dir=args.log_dir)

    host = getattr(args, "web_host", "127.0.0.1")
    port = int(getattr(args, "web_port", 8001))

    app = create_app()

    # SIGINT during uvicorn.run() is handled by uvicorn itself; we still
    # install a handler for the rare case where Python sees the signal
    # first (e.g. while the service is starting).
    def _sigint(*_):
        _shutdown_service()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    print("Starting StreamMUSE Viewer...", flush=True)
    print(f"  http://{host}:{port}/", flush=True)
    print(f"  Tempo: {config.tempo.bpm} BPM", flush=True)
    print(f"  Input: {config.input.type}", flush=True)
    print(f"  Inference: {config.inference.type} (model: {config.inference.model_name})", flush=True)
    print(f"  Logging root: {args.log_dir}", flush=True)
    print("  Session: idle (use the Web Start control)", flush=True)

    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
