"""Client transports for protocol v3: the lyric stream WebSocket and bar audio fetches."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import threading
import time

import httpx

from streammuse.application.rap.bar_stream import BarAudioFetch
from streammuse.infrastructure.rap.bar_package import OpusCodec, decode_bar_package


def _ws_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :] + path
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :] + path
    raise ValueError("render URL must be http(s)")


class LyricStreamConnection:
    """One session's WebSocket: a receive thread hands each JSON message to ``on_message``.

    If the connection drops, it reconnects (bounded attempts) and resends a
    fresh ``start`` built by ``start_provider`` so the server reattaches.
    """

    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout_seconds: float = 10.0,
        reconnect_attempts: int = 5,
        start_provider: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        self._base_url = base_url
        self._connect_timeout = connect_timeout_seconds
        self._reconnect_attempts = reconnect_attempts
        self._start_provider = start_provider
        self._lock = threading.Lock()
        self._socket = None
        self._thread: threading.Thread | None = None
        self._closing = False
        self._session_id: str | None = None
        self._start: Mapping[str, object] | None = None
        self._on_message: Callable[[Mapping[str, object]], None] | None = None

    def open(
        self, session_id: str, start: Mapping[str, object], on_message: Callable[[Mapping[str, object]], None]
    ) -> None:
        with self._lock:
            self._closing = False
            self._session_id = session_id
            self._start = start
            self._on_message = on_message
        self._connect()
        thread = threading.Thread(target=self._receive_loop, name="rap-lyric-stream", daemon=True)
        self._thread = thread
        thread.start()

    def send(self, message: Mapping[str, object]) -> None:
        with self._lock:
            socket = self._socket
        if socket is None:
            raise ConnectionError("lyric stream is not connected")
        socket.send(json.dumps(message, separators=(",", ":")))

    def close(self) -> None:
        with self._lock:
            self._closing = True
            socket = self._socket
            self._socket = None
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(5.0)

    def _connect(self) -> None:
        from websockets.sync.client import connect

        socket = connect(
            _ws_url(self._base_url, f"/v1/rap/sessions/{self._session_id}"),
            open_timeout=self._connect_timeout,
            max_size=4 * 1024 * 1024,
        )
        start = self._start_provider() if (self._start_provider is not None and self._socket is not None) else self._start
        socket.send(json.dumps(dict(start or {}), separators=(",", ":")))
        with self._lock:
            self._socket = socket

    def _receive_loop(self) -> None:
        attempts = 0
        while True:
            with self._lock:
                socket, closing, handler = self._socket, self._closing, self._on_message
            if closing or socket is None:
                return
            try:
                raw = socket.recv()
                attempts = 0
            except Exception:
                with self._lock:
                    if self._closing:
                        return
                attempts += 1
                if attempts > self._reconnect_attempts:
                    if handler is not None:
                        handler({"type": "error", "message": "lyric stream disconnected", "bar": None})
                    return
                time.sleep(min(2.0, 0.2 * attempts))
                try:
                    self._connect()
                except Exception:
                    continue
                continue
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            if handler is not None:
                try:
                    handler(message)
                except Exception:  # a bad message must not stop the stream
                    pass


class HttpBarAudioSource:
    """GET bar audio by lyric id (long-poll), decoding PCM or Opus packages."""

    def __init__(
        self,
        base_url: str,
        *,
        opus_codec: OpusCodec | None = None,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._codec = opus_codec
        self._client = client or httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0))
        self._clock = clock

    def fetch(self, session_id: str, bar: int, lyric_id: str, *, wait_ms: int, accept_older: bool) -> BarAudioFetch:
        started = self._clock()
        response = self._client.get(
            f"{self._base_url}/v1/rap/sessions/{session_id}/bars/{bar}/audio",
            params={"lyric_id": lyric_id, "wait_ms": wait_ms, "accept_older": str(accept_older).lower()},
            timeout=httpx.Timeout(wait_ms / 1000.0 + 10.0, connect=5.0),
        )
        request_ms = (self._clock() - started) * 1000.0
        status = response.headers.get("x-streammuse-fetch-status", "")
        if response.status_code != 200:
            return BarAudioFetch(status or f"http_{response.status_code}", request_ms=request_ms)
        audio_lyric_id = response.headers.get("x-streammuse-lyric-id", lyric_id)
        decoded = decode_bar_package(response.content, expected_lyric_id=audio_lyric_id, codec=self._codec)
        return BarAudioFetch(
            status=status or "ready",
            lyric_id=audio_lyric_id,
            manifest=decoded.manifest,
            source_wav=decoded.source_wav,
            request_ms=request_ms,
        )

    def close(self) -> None:
        self._client.close()
