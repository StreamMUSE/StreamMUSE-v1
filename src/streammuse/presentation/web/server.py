"""FastAPI server + entry point for the StreamMUSE web UI.

Reuses WebSocketOutputSink (queue-based) and FileSystemPromptRepository.
Composes sinks directly rather than going through OutputSinkFactory, because
the WebSocket sink requires the server to be alive to drain it.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from streammuse.application.config import ApplicationConfig
from streammuse.application.factories import (
    InferenceEngineFactory,
    InputSourceFactory,
)
from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.interfaces import InferenceEngine
from streammuse.domain.logging import SessionManager
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import PlaybackScheduler, Tempo
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


@dataclass
class AppState:
    ws_sink: Optional[WebSocketOutputSink] = None
    service: Optional[RealTimeMusicService] = None
    inference_engine: Optional[InferenceEngine] = None
    repository: Optional[FileSystemPromptRepository] = None
    session_manager: Optional[SessionManager] = None
    config: Optional[ApplicationConfig] = None
    connections: List[WebSocket] = field(default_factory=list)
    _broadcaster_task: Optional[asyncio.Task] = None
    _broadcaster_stop: Optional[asyncio.Event] = None


state = AppState()


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


def create_app() -> FastAPI:
    app = FastAPI(title="StreamMUSE Web", lifespan=lifespan)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        path = STATIC_DIR / "index.html"
        if not path.exists():
            raise HTTPException(status_code=500, detail="static/index.html missing")
        return FileResponse(str(path))

    @app.get("/api/prompts", response_model=PromptsResponse)
    async def list_prompts() -> PromptsResponse:
        repo = state.repository
        if repo is None:
            return PromptsResponse(keys={})
        metadata = repo._metadata  # noqa: SLF001 — intentional index inspection
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

    @app.post("/api/inject", response_model=InjectResponse)
    async def inject(body: InjectRequest) -> InjectResponse:
        repo = state.repository
        engine = state.inference_engine
        config = state.config
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

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        state.connections.append(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            if ws in state.connections:
                state.connections.remove(ws)

    return app


def build_state(
    config: ApplicationConfig,
    *,
    log_dir: str,
    prompts_dir: str,
    include_console: bool = True,
) -> None:
    """Wire up AppState: session, sinks, engine, input, service, repository.

    Does NOT start the service — caller does that (so tests can build state
    without spawning threads).
    """
    session_manager = SessionManager(log_dir)
    session_manager.create_session_directory()
    session_config: Dict[str, Any] = {
        "tempo_bpm": config.tempo.bpm,
        "ticks_per_beat": config.tempo.ticks_per_beat,
        "beats_per_bar": config.tempo.beats_per_bar,
        "input_type": config.input.type,
        "output_type": "web",
        "inference_type": config.inference.type,
        "generation_interval_ticks": config.inference.generation_interval_ticks,
        "generation_length_frames": config.inference.generation_length_frames,
    }
    session_manager.save_config(session_config)

    ws_sink = WebSocketOutputSink(WebSocketOutputConfig(include_timestamps=True))
    auto_midi = MidiFileOutputSink(
        MidiFileOutputConfig(
            bpm=float(config.tempo.bpm),
            ticks_per_beat=int(config.tempo.ticks_per_beat),
            output_path=str(session_manager.get_session_dir() / "combined.mid"),
        )
    )
    child_sinks: List[Any] = [ws_sink, auto_midi]
    if include_console:
        child_sinks.append(ConsoleOutputSink())
    composite = CompositeOutputSink(sinks=child_sinks)

    inference_engine = InferenceEngineFactory.create(config)
    input_source = InputSourceFactory.create(config)

    composite.output_config(session_config)

    tempo = Tempo(
        bpm=config.tempo.bpm,
        ticks_per_beat=config.tempo.ticks_per_beat,
        beats_per_bar=config.tempo.beats_per_bar,
    )
    scheduler = PlaybackScheduler()
    service = RealTimeMusicService(
        input_source=input_source,
        inference_engine=inference_engine,
        output_sink=composite,
        tempo=tempo,
        scheduler=scheduler,
        generation_interval_ticks=config.inference.generation_interval_ticks,
        generation_length_frames=config.inference.generation_length_frames,
    )

    repository = FileSystemPromptRepository(
        PromptRepositoryConfig(
            library_path=prompts_dir,
            ticks_per_beat=config.tempo.ticks_per_beat,
        )
    )

    state.ws_sink = ws_sink
    state.service = service
    state.inference_engine = inference_engine
    state.repository = repository
    state.session_manager = session_manager
    state.config = config


def main() -> int:
    import uvicorn

    args = parse_args()
    config = args_to_config(args)

    host = getattr(args, "web_host", "127.0.0.1")
    port = int(getattr(args, "web_port", 8001))
    prompts_dir = getattr(args, "prompts_dir", None) or "prompts"
    log_dir = getattr(args, "log_dir", None) or "logs"

    build_state(config, log_dir=log_dir, prompts_dir=prompts_dir)

    app = create_app()

    assert state.service is not None
    state.service.start(max_ticks=args.max_ticks)

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
