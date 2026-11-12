"""A stand-in protocol v3 render server with no models, for running the Mac side locally.

It serves the real v3 routes (``LyricStreamService`` and ``mount_lyric_stream_routes``)
with a deterministic lyric generator and a tone-burst "vocal", so the real Mac
client, inference page and acceptance scripts run end to end without H200. An
optional delaying TCP forwarder stands in for the SSH tunnel.

What it can show: connection handling, session and protocol behaviour, and how
latency changes with the number of round trips. What it cannot show: real model
latency, lyric quality, or how the voice sounds.

    uv run python scripts/experiments/inference_opt/standin_server.py \\
        --port 18620 --delay-port 18621 --one-way-ms 250 --write-ms 150 --render-ms 470

Point the Mac client at the delayed port (``--renderer http://127.0.0.1:18621``).
"""

from __future__ import annotations

import argparse
from collections import deque
import io
import itertools
import math
from pathlib import Path
import re
import socket
import struct
import tempfile
import threading
import time
import wave

from streammuse.application.rap.bar_audio import MossBarAudioRenderer
from streammuse.application.rap.chunk_orchestration import PhraseSourceResult
from streammuse.application.rap.lyric_writer import LyricWriterPolicy, SequentialLyricWriter
from streammuse.domain.rap import CandidateBatch, ProsodyAnalysis, ScoreWeights, Syllable, normalize_text
from streammuse.infrastructure.rap.bar_package import PcmBarPackageEncoder
from streammuse.presentation.rap_lyric_stream_server import LyricStreamService, mount_lyric_stream_routes

# One-syllable words; lines are built from these so the stand-in analyzer's count is true enough.
_WORDS = (
    "ash", "bell", "cart", "dune", "elm", "fern", "glow", "hush", "ink", "jade", "kite", "loom", "mist",
    "nest", "oak", "pine", "quill", "reef", "salt", "tide", "urn", "vine", "wren", "yarn", "zest", "brim",
    "cove", "drift", "ember", "flint", "grove", "husk", "isle", "knoll", "ledge", "marsh", "north", "pearl",
)
SAMPLE_RATE = 24_000


class StandInGenerator:
    """Lines of exactly one word per flow slot that never repeat an earlier opening or ending.

    The writer rejects a candidate whose last word, or whose first three words,
    match a recent line, so every candidate takes fresh words from a counter
    shared by all sessions. The first word names the topic, so a test (or a
    listener reading the page) can tell which plan a line was written for.
    """

    def __init__(self, *, write_seconds: float = 0.0) -> None:
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._write_seconds = write_seconds
        self.requests = []

    def generate(self, request):
        if self._write_seconds:
            time.sleep(self._write_seconds)
        topic = re.sub(r"[^a-z]", "", request.topic.lower()) or "topic"
        slots = len(request.flow_template.slots)
        lines = []
        with self._lock:
            self.requests.append(request)
            for _ in range(request.count):
                index = next(self._counter)
                words = [_WORDS[(index * 7 + position * 11 + position * position) % len(_WORDS)]
                         for position in range(slots)]
                words[0] = topic
                if slots > 1:
                    words[1] = _WORDS[index % len(_WORDS)]           # a new opening every candidate
                    words[-1] = _WORDS[(index + 19) % len(_WORDS)]   # and a new end word
                lines.append(" ".join(words))
        return CandidateBatch(request_id=request.request_id, candidates=tuple(lines), source="stand-in",
                              prompt=(), raw_response="", latency_ms=self._write_seconds * 1000)


class StandInAnalyzer:
    """One syllable per word."""

    def analyze(self, text: str) -> ProsodyAnalysis:
        normalized = normalize_text(text)
        return ProsodyAnalysis(
            text=text, normalized_text=normalized,
            syllables=tuple(Syllable(word, 0, 1, 1, ("AH1",), "stand-in") for word in normalized.split()),
            end_rhyme_tail=("AH1",), oov_words=(), heuristic_words=(), punctuation_boundary_after=(),
        )


class StandInSourceRenderer:
    """A short tone burst per syllable, with exact onsets; ``render_seconds`` stands in for MOSS and alignment."""

    def __init__(self, *, render_seconds: float = 0.0) -> None:
        self._render_seconds = render_seconds
        self._lock = threading.Lock()
        self.requests = []
        self.prefixes = []

    def render_source(self, request, workspace, *, prefix=None):
        with self._lock:
            self.requests.append(request)
            self.prefixes.append(prefix)
        if self._render_seconds:
            time.sleep(self._render_seconds)
        count = len(request.syllables)
        step = 0.22
        frames = int((0.1 + count * step + 0.2) * SAMPLE_RATE)
        onsets = [0.1 + index * step for index in range(count)]
        samples = [0] * frames
        burst = int(0.12 * SAMPLE_RATE)
        for index, onset in enumerate(onsets):
            first = int(onset * SAMPLE_RATE)
            frequency = 180.0 + 40.0 * (index % 5)
            for offset in range(burst):
                envelope = math.sin(math.pi * offset / burst)
                samples[first + offset] = int(9000 * envelope * math.sin(2 * math.pi * frequency * offset / SAMPLE_RATE))
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(struct.pack(f"<{frames}h", *samples))
        return PhraseSourceResult(
            source_wav=buffer.getvalue(),
            alignment_diagnostics={"fallback_counts": {}, "source_onsets": onsets, "onset_confidence": [1.0] * count},
            audio_diagnostics={"sample_rate_hz": SAMPLE_RATE, "frame_count": frames,
                               "duration_seconds": frames / SAMPLE_RATE, "peak": 0.27},
            model_tool_versions={"moss": "stand-in", "aligner": "stand-in"},
            warnings=(),
            stage_timings_ms={"moss": self._render_seconds * 1000.0, "aligner": 0.0},
            monitoring_summary={},
        )


def build_service(workspace: Path, *, write_seconds: float = 0.0, render_seconds: float = 0.0,
                  continuation: bool = True):
    """The real v3 session service over stand-in models. Returns (service, generator, source renderer)."""
    generator = StandInGenerator(write_seconds=write_seconds)
    source = StandInSourceRenderer(render_seconds=render_seconds)

    def writer_factory():
        writer = SequentialLyricWriter(
            generator, StandInAnalyzer(), ScoreWeights(),
            LyricWriterPolicy(candidates=4, rescue_candidates=4, minimum_valid=1, rhyme_scheme="none"),
        )
        return writer, lambda: None

    service = LyricStreamService(
        writer_factory=writer_factory,
        renderer=MossBarAudioRenderer(source, PcmBarPackageEncoder(), workspace_root=workspace,
                                      continuation=continuation),
    )
    return service, generator, source


def create_app(service: LyricStreamService):
    from fastapi import FastAPI

    app = FastAPI()
    mount_lyric_stream_routes(app, service)

    @app.get("/health")
    async def health():
        return {"ready": True, "stand_in": True}

    return app


class DelayProxy:
    """A TCP forwarder that adds latency, standing in for the SSH tunnel.

    Every chunk is delivered ``one_way_seconds`` late in each direction, and a
    new connection waits one extra round trip before its first byte moves (the
    connection setup that a real remote link pays and loopback does not).
    """

    def __init__(self, upstream_port: int, *, one_way_seconds: float, listen_port: int = 0) -> None:
        self._upstream = ("127.0.0.1", upstream_port)
        self._delay = one_way_seconds
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", listen_port))
        self._listener.listen(64)
        self.port = self._listener.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
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
                upstream.settimeout(None)
            except OSError:
                downstream.close()
                continue
            for item in (downstream, upstream):
                item.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=18620, help="the stand-in server itself (no added delay)")
    parser.add_argument("--delay-port", type=int, default=0,
                        help="also listen here and forward with --one-way-ms of delay each way (0: off)")
    parser.add_argument("--one-way-ms", type=float, default=250.0)
    parser.add_argument("--write-ms", type=float, default=150.0, help="time one lyric generation call takes")
    parser.add_argument("--render-ms", type=float, default=470.0, help="time one bar's synthesis takes")
    parser.add_argument("--http-keep-alive-seconds", type=int, default=75)
    parser.add_argument("--no-continuation", action="store_true")
    args = parser.parse_args(argv)

    import uvicorn

    with tempfile.TemporaryDirectory(prefix="streammuse-standin-") as workspace:
        service, _, _ = build_service(Path(workspace), write_seconds=args.write_ms / 1000,
                                      render_seconds=args.render_ms / 1000,
                                      continuation=not args.no_continuation)
        proxy = None
        if args.delay_port:
            proxy = DelayProxy(args.port, one_way_seconds=args.one_way_ms / 1000, listen_port=args.delay_port)
            print(f"delayed port: {proxy.url} ({args.one_way_ms:.0f} ms each way)", flush=True)
        try:
            uvicorn.run(create_app(service), host="127.0.0.1", port=args.port, log_level="warning",
                        timeout_keep_alive=args.http_keep_alive_seconds)
        finally:
            if proxy is not None:
                proxy.close()
            service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
