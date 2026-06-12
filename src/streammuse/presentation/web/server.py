"""StreamMUSE web viewer.

A single-process FastAPI server that:
  1. Boots a RealTimeMusicService with the same CLI surface as `streammuse-cli`.
  2. Wires a WebSocketOutputSink into the service's output composite alongside
     console + auto MIDI recording + optional audio (--midi-out-port).
  3. Serves a static piano-roll viewer at GET / and broadcasts the sink's
     queued JSON envelopes to all connected browsers via WS /ws.

There are no /api/start, /api/stop, or /api/inject endpoints. The viewer is
read-only: the service runs from process start to process stop. Everything
the UI shows comes from the WebSocket broadcast — same envelope shape as
the legacy reference UI.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from streammuse.application.config import ApplicationConfig
from streammuse.application.factories import (
    InferenceEngineFactory,
    InputSourceFactory,
)
from streammuse.application.services.prompt_continuation_realtime_service import (
    PromptContinuationRealtimeService,
)
from streammuse.application.services.input_timing import effective_input_snap_forward_fraction
from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.logging import SessionManager
from streammuse.domain.timing import PlaybackScheduler, Tempo
from streammuse.infrastructure.inference.prompt_continuation_http_client import (
    PromptContinuationHttpClient,
    PromptContinuationHttpClientConfig,
)
from streammuse.infrastructure.output import (
    AudioOutputConfig,
    AudioOutputSink,
    CompositeOutputSink,
    ConsoleOutputConfig,
    ConsoleOutputSink,
    MidiFileOutputConfig,
    MidiFileOutputSink,
    WebSocketOutputSink,
)
from streammuse.presentation.cli.config_parser import args_to_config, parse_args


STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Module-level state. Set once at boot, read by request handlers.
# ---------------------------------------------------------------------------

_ws_sink: Optional[WebSocketOutputSink] = None
_service: Optional[RealTimeMusicService | PromptContinuationRealtimeService] = None
_composite_sink: Optional[CompositeOutputSink] = None
_connections: List[WebSocket] = []
_broadcaster_stop: Optional[asyncio.Event] = None
_broadcaster_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# WebSocket broadcaster — drains _ws_sink's queue and pushes to all clients.
# ---------------------------------------------------------------------------

async def _broadcaster_loop(poll_interval_s: float = 0.02) -> None:
    """Drain WebSocketOutputSink.get_pending_messages() and send to clients."""
    assert _ws_sink is not None and _broadcaster_stop is not None
    while not _broadcaster_stop.is_set():
        messages = _ws_sink.get_pending_messages()
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
    if _ws_sink is not None:
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
        _shutdown_service()


def _shutdown_service() -> None:
    """Stop the realtime service and close the output sink. Safe to call twice."""
    global _service, _composite_sink
    if _service is not None and _service.running:
        try:
            _service.stop()
        except Exception:
            pass
    if _composite_sink is not None:
        try:
            _composite_sink.close()
        except Exception:
            pass
    _service = None
    _composite_sink = None


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

def _build_composite(
    *,
    session_manager: SessionManager,
    bpm: float,
    ticks_per_beat: int,
    midi_out_port: Optional[str],
) -> tuple[CompositeOutputSink, WebSocketOutputSink]:
    """Build the viewer's output composite directly.

    Returns the composite plus a reference to the embedded WebSocketOutputSink
    so the broadcaster can drain it without walking the composite tree.

    Order matters only for human-visible side effects (console log order).
    Functionally each sink is independent.
    """
    ws_sink = WebSocketOutputSink()
    auto_midi = MidiFileOutputSink(
        MidiFileOutputConfig(
            bpm=float(bpm),
            ticks_per_beat=int(ticks_per_beat),
            output_path=str(session_manager.get_session_dir() / "combined.mid"),
        )
    )
    sinks: List[Any] = [ws_sink, auto_midi, ConsoleOutputSink(ConsoleOutputConfig())]

    if midi_out_port:
        try:
            audio_sink = AudioOutputSink(AudioOutputConfig(port_name=midi_out_port))
            # Eagerly open the port so a bad name fails fast at boot.
            audio_sink._ensure_port()  # noqa: SLF001
            sinks.append(audio_sink)
            print(f"  Audio output: {midi_out_port}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(
                f"  Audio output: DISABLED — could not open '{midi_out_port}': {exc}",
                file=sys.stderr,
                flush=True,
            )

    return CompositeOutputSink(sinks), ws_sink


def _build_realtime_service(
    *,
    config: ApplicationConfig,
    output_sink: CompositeOutputSink,
) -> RealTimeMusicService | PromptContinuationRealtimeService:
    input_source = InputSourceFactory.create(config)
    tempo = Tempo(
        bpm=config.tempo.bpm,
        ticks_per_beat=config.tempo.ticks_per_beat,
        beats_per_bar=config.tempo.beats_per_bar,
    )
    scheduler = PlaybackScheduler()
    input_snap_forward_fraction = effective_input_snap_forward_fraction(
        config.input.type,
        config.input_snap_forward_fraction,
    )

    if config.continuation_mode == "prompt_continuation":
        prompt_client = PromptContinuationHttpClient(
            PromptContinuationHttpClientConfig(
                base_url=config.inference.server_generate_url,
                timeout_s=float(config.inference.timeout_s),
                model_name="lekai_prompt_continuation",
                inference_mode=config.inference.inference_mode,
                checkpoint_path=config.inference.checkpoint_path,
            )
        )
        return PromptContinuationRealtimeService(
            input_source=input_source,
            prompt_client=prompt_client,
            output_sink=output_sink,
            tempo=tempo,
            scheduler=scheduler,
            prompt_length_ticks=config.inference.prompt_length_ticks,
            generation_interval_ticks=config.inference.generation_interval_ticks,
            input_snap_forward_fraction=input_snap_forward_fraction,
        )

    inference_engine = InferenceEngineFactory.create(config)
    return RealTimeMusicService(
        input_source=input_source,
        inference_engine=inference_engine,
        output_sink=output_sink,
        tempo=tempo,
        scheduler=scheduler,
        generation_interval_ticks=config.inference.generation_interval_ticks,
        generation_length_frames=config.inference.generation_length_frames,
        input_snap_forward_fraction=input_snap_forward_fraction,
    )


def main() -> int:
    """Entry point for `streammuse-web`."""
    global _ws_sink, _service, _composite_sink

    import uvicorn

    args = parse_args()
    config = args_to_config(args)

    session_manager = SessionManager(args.log_dir)
    session_manager.create_session_directory()
    session_config = {
        "tempo_bpm": config.tempo.bpm,
        "ticks_per_beat": config.tempo.ticks_per_beat,
        "beats_per_bar": config.tempo.beats_per_bar,
        "input_type": config.input.type,
        "continuation_mode": config.continuation_mode,
        "inference_type": config.inference.type,
        "prompt_length_ticks": config.inference.prompt_length_ticks,
        "generation_interval_ticks": config.inference.generation_interval_ticks,
        "generation_length_frames": config.inference.generation_length_frames,
        "input_snap_forward_fraction": effective_input_snap_forward_fraction(
            config.input.type,
            config.input_snap_forward_fraction,
        ),
    }
    session_manager.save_config(session_config)

    composite, ws_sink = _build_composite(
        session_manager=session_manager,
        bpm=config.tempo.bpm,
        ticks_per_beat=config.tempo.ticks_per_beat,
        midi_out_port=config.output.midi_out_port,
    )

    service = _build_realtime_service(config=config, output_sink=composite)

    # Broadcast the session config so the UI's initial state matches the
    # service's actual configuration without waiting for the first tick.
    composite.output_config(session_config)

    _composite_sink = composite
    _ws_sink = ws_sink
    _service = service

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

    service.start(max_ticks=None)

    print("Starting StreamMUSE Viewer...", flush=True)
    print(f"  http://{host}:{port}/", flush=True)
    print(f"  Tempo: {config.tempo.bpm} BPM", flush=True)
    print(f"  Input: {config.input.type}", flush=True)
    print(f"  Continuation mode: {config.continuation_mode}", flush=True)
    if config.continuation_mode == "prompt_continuation":
        print(f"  Prompt length: {config.inference.prompt_length_ticks} ticks", flush=True)
    print(f"  Inference: {config.inference.type} (model: {config.inference.model_name})", flush=True)
    print(f"  Logging: {session_manager.get_session_dir()}", flush=True)

    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
