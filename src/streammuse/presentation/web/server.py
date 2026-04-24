"""FastAPI server for the StreamMUSE web UI.

Lifecycle:
  - At server boot, `build_long_lived()` constructs once-per-process state:
    session manager, repository, and a composite OutputSink that wraps the
    long-lived WebSocketOutputSink + auto MIDI recorder (+ console).
  - A QueueInput singleton lives for the whole process. Browser keyboard
    events pushed over /ws land in this queue; the running service only
    consumes them if its current input mode is "queue".
  - `start_service(config)` / `stop_service()` / `restart_service(config)`
    create, tear down, and recreate the RealTimeMusicService bound to the
    same long-lived composite sink.

Endpoints match the reference web UI contract (`AHY123/StreamMUSE@demo_lekai_fixed`
app/web_client.py) so the ported static files under static/ work without edits
(except the documented note_off wiring in piano.js / main.js).
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from streammuse.application.config import (
    ApplicationConfig,
    InferenceConfig,
    InputConfig,
    OutputConfig,
    TempoConfig,
)
from streammuse.application.factories import (
    InferenceEngineFactory,
    InputSourceFactory,
)
from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.interfaces import InferenceEngine
from streammuse.domain.logging import SessionManager
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import PlaybackScheduler, Tempo
from streammuse.infrastructure.input import QueueInput
from streammuse.infrastructure.output.composite import CompositeOutputSink
from streammuse.infrastructure.output.console import ConsoleOutputSink
from streammuse.infrastructure.output.midi_file import (
    MidiFileOutputConfig,
    MidiFileOutputSink,
)
from streammuse.infrastructure.output.websocket import (
    WebSocketOutputConfig,
    WebSocketOutputSink,
)
from streammuse.infrastructure.storage.prompt_repository import (
    FileSystemPromptRepository,
    PromptRepositoryConfig,
)
from streammuse.presentation.cli.config_parser import args_to_config, parse_args
from streammuse.presentation.web.schema import (
    InjectRequest,
    InjectResponse,
    PromptSummary,
    PromptsResponse,
)

STATIC_DIR = Path(__file__).parent / "static"

# UI input-mode labels → our InputType
_UI_INPUT_MODE_MAP = {
    "keyboard": "queue",     # browser keyboard → QueueInput
    "midi": "midi_device",
    "file": "midi_file",
}
_UI_INPUT_MODE_INV = {v: k for k, v in _UI_INPUT_MODE_MAP.items()}


@dataclass
class AppState:
    # Long-lived (boot → shutdown):
    ws_sink: Optional[WebSocketOutputSink] = None
    composite_sink: Optional[CompositeOutputSink] = None
    session_manager: Optional[SessionManager] = None
    repository: Optional[FileSystemPromptRepository] = None
    queue_input: Optional[QueueInput] = None
    prompts_dir: str = "prompts"
    log_dir: str = "logs"
    initial_config: Optional[ApplicationConfig] = None
    # Last UI-config dict received (verbatim, for status echo).
    ui_config: Dict[str, Any] = field(default_factory=dict)

    # Per-service (start → stop):
    service: Optional[RealTimeMusicService] = None
    inference_engine: Optional[InferenceEngine] = None
    input_source: Any = None
    current_config: Optional[ApplicationConfig] = None

    # WebSocket broadcast infra:
    connections: List[WebSocket] = field(default_factory=list)
    _broadcaster_task: Optional[asyncio.Task] = None
    _broadcaster_stop: Optional[asyncio.Event] = None

    # Lifecycle lock — serialize start/stop/restart:
    lock: threading.Lock = field(default_factory=threading.Lock)


state = AppState()


# ---------------------------------------------------------------------------
# WebSocket broadcaster
# ---------------------------------------------------------------------------

async def _broadcaster_loop(poll_interval_s: float = 0.02) -> None:
    sink = state.ws_sink
    stop = state._broadcaster_stop
    assert sink is not None and stop is not None

    while not stop.is_set():
        messages = sink.get_pending_messages()
        if messages:
            dead: List[WebSocket] = []
            for ws in list(state.connections):
                try:
                    for msg in messages:
                        await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                if ws in state.connections:
                    state.connections.remove(ws)
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_interval_s)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state._broadcaster_stop = asyncio.Event()
    if state.ws_sink is not None:
        state._broadcaster_task = asyncio.create_task(_broadcaster_loop())
    try:
        yield
    finally:
        if state._broadcaster_stop is not None:
            state._broadcaster_stop.set()
        if state._broadcaster_task is not None:
            try:
                await asyncio.wait_for(state._broadcaster_task, timeout=1.0)
            except asyncio.TimeoutError:
                state._broadcaster_task.cancel()
        if state.service is not None and state.service.running:
            state.service.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt_summary(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": str(p.get("name", "")),
        "melody_path": str(p.get("melody_path", "")),
        "accompaniment_path": p.get("accompaniment_path"),
        "duration_ticks": p.get("duration_ticks"),
        "source": p.get("source"),
        "song_number": p.get("song_number"),
        "melody_notes_count": p.get("melody_notes_count"),
        "accompaniment_notes_count": p.get("accompaniment_notes_count"),
    }


def _event_to_dict(e: MusicalEvent) -> Dict[str, Any]:
    return {
        "type": "note_on" if e.event_type == EventType.NOTE_ON else "note_off",
        "pitch": e.pitch,
        "tick": e.tick,
        "velocity": e.velocity,
        "channel": e.channel,
        "program": e.program,
    }


def _save_injection_artifacts(
    mel_events: List[MusicalEvent], acc_events: List[MusicalEvent]
) -> None:
    sm = state.session_manager
    if sm is None:
        return
    sdir = sm.get_session_dir()
    with open(sdir / "melody_history.json", "w") as f:
        json.dump([_event_to_dict(e) for e in mel_events], f, indent=2)
    with open(sdir / "accompaniment_history.json", "w") as f:
        json.dump([_event_to_dict(e) for e in acc_events], f, indent=2)


def _compose_ui_config(app_config: ApplicationConfig, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a UI-friendly config dict (matching reference field names).

    Fields the UI surfaces but that aren't part of ApplicationConfig
    (accompaniment_velocity, metronome, listening_*, manual_prompt_path) are
    preserved via the overrides dict so they round-trip through /api/status.
    Key-detection fields intentionally not tracked — feature cut.
    """
    ov = overrides or {}
    ui_input_mode = _UI_INPUT_MODE_INV.get(app_config.input.type, app_config.input.type)
    merged: Dict[str, Any] = {
        "server_url": app_config.inference.server_generate_url,
        "tempo": app_config.tempo.bpm,
        "ticks_per_beat": app_config.tempo.ticks_per_beat,
        "beats_per_bar": app_config.tempo.beats_per_bar,
        "generation_interval_ticks": app_config.inference.generation_interval_ticks,
        "generation_length_per_request": app_config.inference.generation_length_frames,
        "accompaniment_velocity": ov.get("accompaniment_velocity", 50),
        "input_mode": ui_input_mode,
        "midi_input_name": app_config.input.midi_device_name,
        "midi_output_name": app_config.output.midi_out_port,
        "midi_file_path": app_config.input.midi_file_path,
        "metronome": ov.get("metronome", True),
        "listening_duration_ticks": ov.get("listening_duration_ticks", 0),
        "manual_prompt_path": ov.get("manual_prompt_path"),
    }
    return merged


def _apply_ui_config(ui_cfg: Dict[str, Any]) -> ApplicationConfig:
    """Translate the UI's runtime config dict into our ApplicationConfig.

    Fields the UI exposes but we don't currently wire through (metronome,
    accompaniment_velocity, listening_*, key_detection_method) are recorded
    into `state.ui_config` so they survive the UI round trip via /api/status.
    """
    base = state.initial_config or ApplicationConfig()

    bpm = float(ui_cfg.get("tempo", base.tempo.bpm))
    tpb = int(ui_cfg.get("ticks_per_beat", base.tempo.ticks_per_beat))
    bpbar = int(ui_cfg.get("beats_per_bar", base.tempo.beats_per_bar))
    tempo = TempoConfig(bpm=bpm, ticks_per_beat=tpb, beats_per_bar=bpbar)

    ui_mode = ui_cfg.get("input_mode", _UI_INPUT_MODE_INV.get(base.input.type, "keyboard"))
    app_mode = _UI_INPUT_MODE_MAP.get(ui_mode, "queue")
    input_cfg = InputConfig(
        type=app_mode,  # type: ignore[arg-type]
        midi_device_name=ui_cfg.get("midi_input_name") or base.input.midi_device_name,
        midi_file_path=ui_cfg.get("midi_file_path") or base.input.midi_file_path,
        midi_file_delay_ticks=base.input.midi_file_delay_ticks,
    )

    output_cfg = OutputConfig(
        type=base.output.type,
        midi_out_port=ui_cfg.get("midi_output_name") or base.output.midi_out_port,
        midi_file_output_path=base.output.midi_file_output_path,
        inference_log_detail=base.output.inference_log_detail,
    )

    inf_cfg = InferenceConfig(
        type=base.inference.type,
        server_generate_url=ui_cfg.get("server_url", base.inference.server_generate_url),
        timeout_s=base.inference.timeout_s,
        model_name=base.inference.model_name,
        inference_mode=base.inference.inference_mode,
        checkpoint_path=base.inference.checkpoint_path,
        model_size=base.inference.model_size,
        model_max_seq_len_frames=base.inference.model_max_seq_len_frames,
        generation_length_frames=int(ui_cfg.get("generation_length_per_request", base.inference.generation_length_frames)),
        generation_interval_ticks=int(ui_cfg.get("generation_interval_ticks", base.inference.generation_interval_ticks)),
    )

    return ApplicationConfig(tempo=tempo, input=input_cfg, output=output_cfg, inference=inf_cfg)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def build_long_lived(
    *,
    log_dir: str,
    prompts_dir: str,
    initial_config: ApplicationConfig,
    include_console: bool = True,
) -> None:
    """Boot-time wiring. Creates the session dir, ws_sink, composite sink,
    prompt repository, and QueueInput singleton. Does NOT start the service."""
    sm = SessionManager(log_dir)
    sm.create_session_directory()

    ws_sink = WebSocketOutputSink(WebSocketOutputConfig(include_timestamps=True))
    auto_midi = MidiFileOutputSink(
        MidiFileOutputConfig(
            bpm=float(initial_config.tempo.bpm),
            ticks_per_beat=int(initial_config.tempo.ticks_per_beat),
            output_path=str(sm.get_session_dir() / "combined.mid"),
        )
    )
    child_sinks: List[Any] = [ws_sink, auto_midi]
    if include_console:
        child_sinks.append(ConsoleOutputSink())
    composite = CompositeOutputSink(sinks=child_sinks)

    repository = FileSystemPromptRepository(
        PromptRepositoryConfig(
            library_path=prompts_dir,
            ticks_per_beat=initial_config.tempo.ticks_per_beat,
        )
    )
    queue_input = QueueInput()

    state.ws_sink = ws_sink
    state.composite_sink = composite
    state.session_manager = sm
    state.repository = repository
    state.queue_input = queue_input
    state.prompts_dir = prompts_dir
    state.log_dir = log_dir
    state.initial_config = initial_config
    state.current_config = initial_config
    state.ui_config = _compose_ui_config(initial_config)


def _build_input_source(config: ApplicationConfig):
    """Wire the service's input source; reuse the QueueInput singleton."""
    return InputSourceFactory.create(config, queue_input=state.queue_input)


def _find_prompt_by_manual_path(manual_path: str) -> tuple[str, Dict[str, Any]] | None:
    """Locate (key, prompt) in the repository whose melody_path matches.

    The UI sends whatever `path` value /api/manual_prompts returned — which
    for us is the raw `melody_path`. Match by exact equality then by
    basename as a fallback (handles old absolute-path entries).
    """
    repo = state.repository
    if repo is None or not manual_path:
        return None
    metadata = repo._metadata  # noqa: SLF001
    basename = manual_path.rsplit("/", 1)[-1]
    for key, prompts in metadata.items():
        for p in prompts:
            mp = str(p.get("melody_path", ""))
            if mp == manual_path or mp.endswith("/" + basename) or basename == mp.rsplit("/", 1)[-1]:
                return key, p
    return None


def _build_listening_callback(ui_cfg: Dict[str, Any], config: ApplicationConfig) -> Optional[Callable[[], None]]:
    """If the UI selected a manual prompt to inject at listening-end, bake
    the injection call into a callback the service can invoke at the
    boundary tick. Returns None if no manual prompt is selected."""
    manual_path = ui_cfg.get("manual_prompt_path") or ""
    if not manual_path:
        return None
    engine_ref = [state.inference_engine]  # captured by closure; resolved at call time

    def _callback() -> None:
        engine = engine_ref[0] or state.inference_engine
        repo = state.repository
        if engine is None or repo is None:
            return
        found = _find_prompt_by_manual_path(manual_path)
        if found is None:
            if state.composite_sink is not None:
                state.composite_sink.output_status(
                    "error", f"manual prompt not found: {manual_path}"
                )
            return
        key, prompt = found
        max_ticks = (
            config.tempo.ticks_per_beat
            * config.tempo.beats_per_bar
            * 16
        )
        mel_events, acc_events = repo.load_prompt_events(
            key=key,
            prompt=prompt,
            max_ticks=max_ticks,
        )
        try:
            engine.inject_history(
                melody_events=mel_events,
                accompaniment_events=acc_events,
                injection_length_ticks=max_ticks,
            )
            _save_injection_artifacts(mel_events, acc_events)
            if state.composite_sink is not None:
                state.composite_sink.output_status(
                    "injected",
                    f"listening-end injected {prompt.get('name', key)} "
                    f"(mel={len(mel_events)}, acc={len(acc_events)})",
                )
        except Exception as exc:
            if state.composite_sink is not None:
                state.composite_sink.output_status(
                    "error", f"listening-end injection failed: {exc}"
                )

    return _callback


def start_service(config: ApplicationConfig, ui_cfg: Dict[str, Any] | None = None) -> None:
    with state.lock:
        if state.service is not None and state.service.running:
            return  # idempotent — already running

        # Flush any stale events in the queue singleton from a prior session.
        if state.queue_input is not None:
            while not state.queue_input._q.empty():  # noqa: SLF001
                try:
                    state.queue_input._q.get_nowait()  # noqa: SLF001
                except Exception:
                    break

        ui_cfg = ui_cfg or {}
        inference_engine = InferenceEngineFactory.create(config)
        input_source = _build_input_source(config)

        # Publish engine early so the listening callback closure can resolve it.
        state.inference_engine = inference_engine
        listening_duration = int(ui_cfg.get("listening_duration_ticks", 0) or 0)
        acc_velocity_raw = ui_cfg.get("accompaniment_velocity")
        acc_velocity = int(acc_velocity_raw) if acc_velocity_raw is not None else None
        listening_cb = (
            _build_listening_callback(ui_cfg, config)
            if listening_duration > 0
            else None
        )

        tempo = Tempo(
            bpm=config.tempo.bpm,
            ticks_per_beat=config.tempo.ticks_per_beat,
            beats_per_bar=config.tempo.beats_per_bar,
        )
        scheduler = PlaybackScheduler()
        service = RealTimeMusicService(
            input_source=input_source,
            inference_engine=inference_engine,
            output_sink=state.composite_sink,  # type: ignore[arg-type]
            tempo=tempo,
            scheduler=scheduler,
            generation_interval_ticks=config.inference.generation_interval_ticks,
            generation_length_frames=config.inference.generation_length_frames,
            listening_duration_ticks=listening_duration,
            accompaniment_velocity=acc_velocity,
            on_listening_end=listening_cb,
        )

        state.service = service
        state.input_source = input_source
        state.current_config = config
        state.ui_config = _compose_ui_config(config, ui_cfg)

        # Broadcast config snapshot to any connected UI clients.
        if state.composite_sink is not None:
            state.composite_sink.output_config(state.ui_config)

        service.start(max_ticks=None)


def stop_service() -> None:
    with state.lock:
        service = state.service
        if service is None:
            return
        try:
            service.stop()
        except Exception:
            pass
        state.service = None
        state.inference_engine = None
        state.input_source = None
        if state.composite_sink is not None:
            state.composite_sink.output_status("stopped", "")


def restart_service(config: ApplicationConfig, ui_cfg: Dict[str, Any] | None = None) -> None:
    stop_service()
    start_service(config, ui_cfg=ui_cfg)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(title="StreamMUSE Web", lifespan=lifespan)

    # Match reference static paths: /css/*, /js/* served at root.
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

    # ---- Status / lifecycle ------------------------------------------------

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        svc = state.service
        running = svc is not None and svc.running
        session_dir = str(state.session_manager.get_session_dir()) if state.session_manager else None
        return JSONResponse(
            {
                "is_running": running,
                "config": state.ui_config,
                "session_dir": session_dir,
            }
        )

    @app.post("/api/start")
    async def api_start(body: Dict[str, Any] | None = None) -> JSONResponse:
        if state.initial_config is None:
            return JSONResponse({"success": False, "message": "server not initialized"}, status_code=503)
        ui_cfg = body or {}
        try:
            cfg = _apply_ui_config(ui_cfg) if ui_cfg else (state.current_config or state.initial_config)
            start_service(cfg, ui_cfg=ui_cfg)
        except Exception as exc:  # pragma: no cover — user-facing
            return JSONResponse({"success": False, "message": str(exc)}, status_code=500)
        return JSONResponse({"success": True, "message": "started"})

    @app.post("/api/stop")
    async def api_stop() -> JSONResponse:
        stop_service()
        return JSONResponse({"success": True, "message": "stopped"})

    @app.post("/api/restart")
    async def api_restart(body: Dict[str, Any] | None = None) -> JSONResponse:
        if state.initial_config is None:
            return JSONResponse({"success": False, "message": "server not initialized"}, status_code=503)
        ui_cfg = body or {}
        try:
            cfg = _apply_ui_config(ui_cfg) if ui_cfg else (state.current_config or state.initial_config)
            restart_service(cfg, ui_cfg=ui_cfg)
        except Exception as exc:  # pragma: no cover
            return JSONResponse({"success": False, "message": str(exc)}, status_code=500)
        return JSONResponse({"success": True, "message": "restarted"})

    # ---- MIDI device discovery --------------------------------------------

    @app.get("/api/midi_devices")
    async def api_midi_devices() -> JSONResponse:
        try:
            import mido

            inputs = list(mido.get_input_names())
            outputs = list(mido.get_output_names())
        except Exception as exc:
            return JSONResponse(
                {"input_devices": [], "output_devices": [], "error": str(exc)}
            )
        return JSONResponse({"input_devices": inputs, "output_devices": outputs})

    # ---- Prompt library ----------------------------------------------------

    @app.get("/api/prompts", response_model=PromptsResponse)
    async def list_prompts() -> PromptsResponse:
        repo = state.repository
        if repo is None:
            return PromptsResponse(keys={})
        metadata = repo._metadata  # noqa: SLF001
        keys: Dict[str, List[PromptSummary]] = {}
        for key, prompts in metadata.items():
            keys[key] = [PromptSummary(**_prompt_summary(p)) for p in prompts]
        return PromptsResponse(keys=keys)

    @app.get("/api/prompts/{key}", response_model=List[PromptSummary])
    async def list_prompts_for_key(key: str) -> List[PromptSummary]:
        repo = state.repository
        if repo is None:
            return []
        return [PromptSummary(**_prompt_summary(p)) for p in repo.get_prompts_for_key(key)]

    @app.get("/api/manual_prompts")
    async def list_manual_prompts() -> JSONResponse:
        """Flatten prompt library into the reference UI's {path, name} shape."""
        repo = state.repository
        if repo is None:
            return JSONResponse({"prompts": []})
        flat: List[Dict[str, Any]] = []
        metadata = repo._metadata  # noqa: SLF001
        for key, prompts in metadata.items():
            for p in prompts:
                flat.append(
                    {
                        "key": key,
                        "name": f"{key}/{p.get('name', '')}",
                        "path": p.get("melody_path", ""),
                    }
                )
        return JSONResponse({"prompts": flat})

    @app.post("/api/inject", response_model=InjectResponse)
    async def inject(body: InjectRequest) -> InjectResponse:
        repo = state.repository
        engine = state.inference_engine
        config = state.current_config
        if repo is None or engine is None or config is None:
            raise HTTPException(status_code=503, detail="server not initialized")

        prompt = repo.select_prompt(body.key, body.strategy)
        if prompt is None:
            return InjectResponse(
                injected_melody=0,
                injected_acc=0,
                message=f"no prompts for key {body.key}",
            )

        max_ticks = body.length_ticks or (
            config.tempo.ticks_per_beat * config.tempo.beats_per_bar * 16
        )
        mel_events, acc_events = repo.load_prompt_events(
            key=body.key,
            prompt=prompt,
            max_ticks=max_ticks,
        )

        engine.clear_history()
        engine.inject_history(
            melody_events=mel_events,
            accompaniment_events=acc_events,
            injection_length_ticks=max_ticks,
        )
        _save_injection_artifacts(mel_events, acc_events)

        return InjectResponse(
            injected_melody=len(mel_events),
            injected_acc=len(acc_events),
            message=f"injected {prompt.get('name', body.key)}",
            prompt_name=str(prompt.get("name", "")) or None,
        )

    # ---- WebSocket ---------------------------------------------------------

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        state.connections.append(ws)
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "keyboard_input":
                    _handle_inbound_keyboard(msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            if ws in state.connections:
                state.connections.remove(ws)

    return app


def _handle_inbound_keyboard(msg: Dict[str, Any]) -> None:
    """Translate a browser keyboard event into a MusicalEvent and enqueue.

    Only effective when the running service's input mode is "queue". For
    other modes, the queue accumulates but nothing consumes it (harmless).
    """
    if state.queue_input is None:
        return
    cfg = state.current_config
    if cfg is None or cfg.input.type != "queue":
        return

    event = msg.get("event")
    try:
        pitch = int(msg.get("pitch"))
    except (TypeError, ValueError):
        return
    velocity = int(msg.get("velocity", 100))

    if event == "note_on":
        et = EventType.NOTE_ON
    elif event == "note_off":
        et = EventType.NOTE_OFF
        velocity = 0
    else:
        return

    # Tick is assigned by the service's input_worker based on arrival time;
    # supply a placeholder tick=0 here. (Verified by regression bug 1.1:
    # the service stamps tick via wall-clock, not by this field.)
    ev = MusicalEvent(
        tick=0,
        pitch=pitch,
        event_type=et,
        velocity=velocity,
        source="user",
    )
    state.queue_input.push(ev)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    import uvicorn

    args = parse_args()
    config = args_to_config(args)

    host = getattr(args, "web_host", "127.0.0.1")
    port = int(getattr(args, "web_port", 8001))
    prompts_dir = getattr(args, "prompts_dir", None) or "prompts"
    log_dir = getattr(args, "log_dir", None) or "logs"

    build_long_lived(
        log_dir=log_dir,
        prompts_dir=prompts_dir,
        initial_config=config,
    )

    app = create_app()

    # Auto-start the service with CLI-provided config so navigating to the
    # page immediately shows live events. The user can Stop/Restart via UI.
    try:
        start_service(config)
    except Exception as exc:
        print(f"Warning: auto-start failed: {exc}", file=sys.stderr)

    print("Starting StreamMUSE Web UI...")
    print(f"  http://{host}:{port}/")
    print(f"  Tempo: {config.tempo.bpm} BPM")
    print(f"  Input: {config.input.type}")
    print(f"  Inference: {config.inference.type} (model: {config.inference.model_name})")
    if state.session_manager:
        print(f"  Logging: {state.session_manager.get_session_dir()}")

    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
