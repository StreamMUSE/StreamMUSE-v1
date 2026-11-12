"""A protocol v3 server on a real socket, for tests of the real client transports.

uvicorn serves ``mount_lyric_stream_routes`` with a deterministic lyric writer
and bar renderer (no models). Every request's TCP connection is recorded, so a
test can tell a reused connection from a new one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import socket
import threading
import time
from types import SimpleNamespace

from fastapi import FastAPI
import numpy as np
import uvicorn

from streammuse.application.rap.bar_audio import MossBarAudioRenderer
from streammuse.application.rap.lyric_writer import LyricWriterPolicy, SequentialLyricWriter
from streammuse.domain.rap import ScoreWeights
from streammuse.domain.rap.audio import AudioFormat, PcmAudio
from streammuse.infrastructure.rap.bar_package import PcmBarPackageEncoder
from streammuse.infrastructure.rap.inference_backend import RealInferenceBackend
from streammuse.infrastructure.rap.lyric_stream_client import HttpBarAudioSource, LyricStreamConnection
from streammuse.presentation.rap_lyric_stream_server import LyricStreamService, mount_lyric_stream_routes

from tests.unit.application.rap.test_lyric_session import _Analyzer, _Generator
from tests.unit.presentation.test_rap_lyric_stream_server import _SourceRenderer

FORMAT = AudioFormat(sample_rate_hz=100, channels=1)


@dataclass(frozen=True)
class Seen:
    kind: str  # "http" or "websocket"
    path: str
    peer: tuple[str, int]  # the client's address: one per TCP connection
    at: float


class LiveLyricServer:
    def __init__(self, workspace, *, keep_alive_seconds: int = 75) -> None:
        self.source = _SourceRenderer()
        self.renderer = MossBarAudioRenderer(self.source, PcmBarPackageEncoder(), workspace_root=workspace)

        def writer_factory():
            writer = SequentialLyricWriter(
                _Generator(), _Analyzer(), ScoreWeights(),
                LyricWriterPolicy(candidates=2, rescue_candidates=0, minimum_valid=1, rhyme_scheme="none"),
            )
            return writer, lambda: None

        self.service = LyricStreamService(writer_factory=writer_factory, renderer=self.renderer)
        application = FastAPI()
        mount_lyric_stream_routes(application, self.service)
        self.seen: list[Seen] = []
        self._lock = threading.Lock()

        async def recording(scope, receive, send):
            if scope["type"] in ("http", "websocket"):
                with self._lock:
                    self.seen.append(Seen(scope["type"], scope["path"], tuple(scope["client"]), time.monotonic()))
            await application(scope, receive, send)

        self._socket = socket.socket()
        self._socket.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self._socket.getsockname()[1]}"
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


class DelayProxy:
    """A TCP forwarder that adds latency, standing in for the SSH tunnel.

    Every chunk is delivered ``one_way_seconds`` late in each direction, and a
    new connection waits one extra round trip before its first byte moves (the
    connection setup that a real remote link pays and loopback does not).
    """

    def __init__(self, upstream_port: int, *, one_way_seconds: float) -> None:
        self._upstream = ("127.0.0.1", upstream_port)
        self._delay = one_way_seconds
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(32)
        self.url = f"http://127.0.0.1:{self._listener.getsockname()[1]}"
        self.connections = 0
        self._closed = threading.Event()
        self._sockets: list[socket.socket] = []
        threading.Thread(target=self._accept, name="delay-proxy", daemon=True).start()

    def _accept(self) -> None:
        while not self._closed.is_set():
            try:
                downstream, _ = self._listener.accept()
            except OSError:
                return
            self.connections += 1
            try:
                upstream = socket.create_connection(self._upstream, timeout=5)
            except OSError:
                downstream.close()
                continue
            self._sockets += [downstream, upstream]
            setup = time.monotonic() + 2 * self._delay
            for source, target, not_before in ((downstream, upstream, setup), (upstream, downstream, 0.0)):
                self._pump(source, target, not_before)

    def _pump(self, source: socket.socket, target: socket.socket, not_before: float) -> None:
        pending: deque[tuple[float, bytes | None]] = deque()
        ready = threading.Condition()

        def read() -> None:
            while True:
                try:
                    data = source.recv(65536)
                except OSError:
                    data = b""
                with ready:
                    pending.append((max(time.monotonic(), not_before) + self._delay, data or None))
                    ready.notify()
                if not data:
                    return

        def write() -> None:
            while True:
                with ready:
                    while not pending:
                        ready.wait()
                    due, data = pending.popleft()
                time.sleep(max(0.0, due - time.monotonic()))
                try:
                    if data is None:
                        target.shutdown(socket.SHUT_WR)
                        return
                    target.sendall(data)
                except OSError:
                    return

        for job in (read, write):
            threading.Thread(target=job, name="delay-proxy-pump", daemon=True).start()

    def close(self) -> None:
        self._closed.set()
        for item in (self._listener, *self._sockets):
            try:
                item.close()
            except OSError:
                pass
