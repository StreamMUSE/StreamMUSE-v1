"""A protocol v3 server on a real socket, for tests of the real client transports.

uvicorn serves the stand-in server (the real v3 routes over a deterministic lyric
writer and bar renderer, no models). Every request's TCP connection is recorded,
so a test can tell a reused connection from a new one.
"""

from __future__ import annotations

from dataclasses import dataclass
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import uvicorn

from scripts.experiments.inference_opt.standin_server import DelayProxy, build_service, create_app
from streammuse.domain.rap.audio import AudioFormat, PcmAudio
from streammuse.infrastructure.rap.inference_backend import RealInferenceBackend
from streammuse.infrastructure.rap.lyric_stream_client import HttpBarAudioSource, LyricStreamConnection

__all__ = ["DelayProxy", "LiveClient", "LiveLyricServer", "Seen", "FORMAT"]

FORMAT = AudioFormat(sample_rate_hz=100, channels=1)


@dataclass(frozen=True)
class Seen:
    kind: str  # "http" or "websocket"
    path: str
    peer: tuple[str, int]  # the client's address: one per TCP connection
    at: float


class LiveLyricServer:
    def __init__(self, workspace, *, keep_alive_seconds: int = 75, write_seconds: float = 0.0,
                 render_seconds: float = 0.0) -> None:
        self.service, self.generator, self.source = build_service(
            workspace, write_seconds=write_seconds, render_seconds=render_seconds)
        application = create_app(self.service)
        self.seen: list[Seen] = []
        self._lock = threading.Lock()

        async def recording(scope, receive, send):
            if scope["type"] in ("http", "websocket"):
                with self._lock:
                    self.seen.append(Seen(scope["type"], scope["path"], tuple(scope["client"]), time.monotonic()))
            await application(scope, receive, send)

        self._socket = socket.socket()
        self._socket.bind(("127.0.0.1", 0))
        self.port = self._socket.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        # log_config/log_level None: leave the process's logging alone (other tests read uvicorn's loggers).
        self._server = uvicorn.Server(uvicorn.Config(
            recording, log_config=None, log_level=None, lifespan="off", timeout_keep_alive=keep_alive_seconds))
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [self._socket]}, name="live-lyric-server", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("live lyric server did not start")
            time.sleep(0.01)

    def requests(self, *, since: int = 0) -> list[Seen]:
        with self._lock:
            return list(self.seen[since:])

    def mark(self) -> int:
        with self._lock:
            return len(self.seen)

    def peers(self, *, since: int = 0) -> set[tuple[str, int]]:
        return {item.peer for item in self.requests(since=since)}

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(10)
        self.service.close()
        self._socket.close()


class LiveClient:
    """The real backend and real transports against a LiveLyricServer; preparation is local and trivial."""

    def __init__(self, url: str, **backend_options) -> None:
        self.changed = threading.Condition()
        self.spans, self.events, self.streams, self.sources = [], [], [], []

        def stream_factory():
            stream = LyricStreamConnection(url, connect_timeout_seconds=5.0)
            self.streams.append(stream)
            return stream

        def source_factory():
            source = HttpBarAudioSource(url, keepalive_expiry_seconds=60.0)
            self.sources.append(source)
            return source

        def preparer_factory(request):
            def prepare(record, manifest, source_wav):
                frame_at = lambda tick: round(tick * FORMAT.sample_rate_hz * 60 / request.bpm / 4)
                frames = frame_at((record.bar + 1) * 16) - frame_at(record.bar * 16)
                audio = PcmAudio(FORMAT, frames, np.full((frames, 1), 0.25, dtype=np.float32).tobytes())
                return SimpleNamespace(audio=audio, text=record.text, render_latency_ms=1.0)

            return SimpleNamespace(prepare=prepare)

        self.backend = RealInferenceBackend(
            stream_factory=stream_factory, source_factory=source_factory, preparer_factory=preparer_factory,
            templates=backend_options.pop("templates", {}), request_timeout_seconds=20, **backend_options)

    def on_span(self, span):
        with self.changed:
            self.spans.append(span)
            self.changed.notify_all()
        return True

    def on_event(self, event):
        with self.changed:
            self.events.append(event)
            self.changed.notify_all()

    def wait(self, predicate, timeout: float = 15.0) -> None:
        with self.changed:
            assert self.changed.wait_for(predicate, timeout=timeout), (self.events[-6:], len(self.spans))

    def submit(self, request) -> None:
        self.backend.submit(request, self.on_span, self.on_event)

    def finished(self, request_id: str, kind: str = "complete") -> None:
        self.wait(lambda: any(e["type"] in (kind, "error") and e["request_id"] == request_id for e in self.events))
        errors = [e for e in self.events if e["type"] == "error" and e["request_id"] == request_id]
        assert not errors or kind == "error", errors

    def prepare(self) -> None:
        assert self.backend.prepare() is True
        for future in list(self.backend._preparing.values()):
            future.result(timeout=15)

    def close(self) -> None:
        self.backend.close()
