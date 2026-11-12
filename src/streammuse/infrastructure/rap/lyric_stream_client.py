"""Client transports for protocol v3: the lyric stream WebSocket and bar audio fetches."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import logging
import queue
import threading
import time

import httpx

from streammuse.application.rap.bar_stream import BarAudioFetch
from streammuse.domain.rap.lyric_stream import TEMPO_UPDATES_CAPABILITY, LYRIC_STREAM_SCHEMA_VERSION, TempoUpdate
from streammuse.infrastructure.rap.bar_package import OpusCodec, decode_bar_package

_LOGGER = logging.getLogger(__name__)


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

    ``connect`` may be called before ``open`` to do the WebSocket handshake
    ahead of time: the server accepts the socket and waits for ``start``, so
    ``open`` for the same session then only has to send it.
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
        self._capabilities: frozenset[str] = frozenset()
        self._tempo_requests: dict[int, tuple[threading.Event, dict[str, object]]] = {}
        self._dispatch_queue: queue.Queue | None = None
        self._dispatch_thread: threading.Thread | None = None
        self._preconnected = False

    @property
    def capabilities(self) -> frozenset[str]:
        with self._lock:
            return self._capabilities

    def get_capabilities(self) -> frozenset[str]:
        """Preflight optional operations without creating an unbounded old-server session."""
        response = httpx.get(f"{self._base_url.rstrip('/')}/v1/rap/capabilities", timeout=self._connect_timeout)
        if response.status_code == 404:
            return frozenset()
        response.raise_for_status()
        payload = response.json()
        if payload.get("schema_version") != LYRIC_STREAM_SCHEMA_VERSION:
            return frozenset()
        return frozenset(payload.get("capabilities", ()))

    def request_tempo(self, payload: Mapping[str, object], *, timeout: float = 10.0) -> dict[str, object]:
        update = TempoUpdate.from_payload(dict(payload))
        event, result = threading.Event(), {}
        with self._lock:
            if TEMPO_UPDATES_CAPABILITY not in self._capabilities:
                raise ValueError("server did not negotiate tempo update capability")
            if self._tempo_requests:
                raise RuntimeError("another tempo update is pending")
            self._tempo_requests[update.edit_seq] = (event, result)
        started = time.monotonic()
        deadline = started + timeout
        try:
            for attempt in range(2):
                try:
                    self.send({"type": "tempo_update", "update": update.to_payload()})
                except Exception as error:
                    # A partial send cannot establish whether the server applied it.
                    raise ConnectionError("tempo update delivery failed; Stop, wait stopped, Reset, then Start") from error
                wait = max(0.0, deadline - time.monotonic())
                if event.wait(min(wait, timeout / 2) if attempt == 0 else wait):
                    break
                if attempt == 0:
                    # The server deduplicates this exact edit identity and payload.
                    _LOGGER.warning("tempo_ack_retry session=%s edit_seq=%s elapsed_ms=%.1f",
                                    self._session_id, update.edit_seq, (time.monotonic() - started) * 1000)
            else:
                raise TimeoutError("tempo acknowledgement timed out; Stop, wait stopped, Reset, then Start")
            if result.get("type") == "error":
                raise ValueError(str(result.get("message")))
            return dict(result)
        finally:
            with self._lock:
                self._tempo_requests.pop(update.edit_seq, None)

    def _accept_control_message(self, message: Mapping[str, object]) -> bool:
        with self._lock:
            if message.get("type") == "started":
                self._capabilities = frozenset(message.get("capabilities", ()))
            if (message.get("type") == "tempo_updated"
                    or message.get("type") == "error" and message.get("operation") == "tempo_update"):
                pending = self._tempo_requests.get(message.get("edit_seq"))
                if pending is not None:
                    event, result = pending
                    result.update(message)
                    event.set()
                else:
                    _LOGGER.warning("tempo_ack_unmatched session=%s edit_seq=%s type=%s",
                                    self._session_id, message.get("edit_seq"), message.get("type"))
                return True
        return False

    def connect(self, session_id: str) -> None:
        """Handshake only: no ``start`` is sent and no session exists until ``open``."""
        with self._lock:
            stale = self._socket
            self._socket = None
            self._preconnected = False
            self._closing = False
            self._capabilities = frozenset()
            self._session_id = session_id
            self._start = None
        if stale is not None:
            self._close_socket(stale)
        socket = self._handshake()
        with self._lock:
            closing = self._closing or self._session_id != session_id
            if not closing:
                self._socket = socket
                self._preconnected = True
        if closing:
            self._close_socket(socket)

    def open(
        self, session_id: str, start: Mapping[str, object], on_message: Callable[[Mapping[str, object]], None]
    ) -> None:
        with self._lock:
            # Only a handshake made for this very session can carry its ``start``:
            # the session id is part of the socket's URL.
            ready = self._socket if self._preconnected and self._session_id == session_id else None
            unused = self._socket if self._preconnected and ready is None else None
            if self._preconnected:
                self._socket = None
            self._preconnected = False
            self._closing = False
            self._capabilities = frozenset()
            self._session_id = session_id
            self._start = start
            self._on_message = on_message
        if unused is not None:
            self._close_socket(unused)
        if ready is not None:
            try:
                ready.send(json.dumps(dict(start or {}), separators=(",", ":")))
            except Exception as error:
                # The early handshake went stale; connect again exactly as without it.
                _LOGGER.warning("lyric_stream_preconnect_stale session=%s error=%s",
                                session_id, type(error).__name__)
                self._close_socket(ready)
                ready = None
        if ready is not None:
            with self._lock:
                self._socket = ready
        else:
            self._connect()
        dispatch_queue = queue.Queue(maxsize=256)
        self._dispatch_queue = dispatch_queue
        dispatch_thread = threading.Thread(target=self._dispatch_loop, args=(dispatch_queue,),
                                           name="rap-lyric-callbacks", daemon=True)
        self._dispatch_thread = dispatch_thread
        dispatch_thread.start()
        thread = threading.Thread(target=self._receive_loop, name="rap-lyric-stream", daemon=True)
        self._thread = thread
        thread.start()

    def send(self, message: Mapping[str, object]) -> None:
        self._send(message)

    def send_for_session(self, session_id: str, message: Mapping[str, object]) -> None:
        """Capture only the intended session's socket; never redirect a stale edit."""
        self._send(message, expected_session_id=session_id)

    def _send(self, message: Mapping[str, object], *, expected_session_id: str | None = None) -> None:
        with self._lock:
            if expected_session_id is not None and expected_session_id != self._session_id:
                raise ConnectionError("lyric stream session changed before delivery")
            socket = self._socket
        if socket is None:
            raise ConnectionError("lyric stream is not connected")
        socket.send(json.dumps(message, separators=(",", ":")))

    def close(self) -> None:
        with self._lock:
            self._closing = True
            self._preconnected = False
            socket = self._socket
            self._socket = None
        if socket is not None:
            self._close_socket(socket)
        if self._thread is not None:
            self._thread.join(5.0)
        dispatch_queue, dispatch_thread = self._dispatch_queue, self._dispatch_thread
        if dispatch_queue is not None:
            while True:
                try:
                    dispatch_queue.get_nowait()
                except queue.Empty:
                    break
            dispatch_queue.put_nowait(None)
        if dispatch_thread is not None and dispatch_thread is not threading.current_thread():
            dispatch_thread.join(5.0)

    @staticmethod
    def _close_socket(socket) -> None:
        try:
            socket.close()
        except Exception:
            pass

    def _handshake(self):
        from websockets.sync.client import connect

        return connect(
            _ws_url(self._base_url, f"/v1/rap/sessions/{self._session_id}"),
            open_timeout=self._connect_timeout,
            max_size=4 * 1024 * 1024,
        )

    def _connect(self) -> None:
        socket = self._handshake()
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
            if self._accept_control_message(message):
                continue
            if handler is not None:
                # Control acknowledgements must remain readable when a callback
                # is waiting for the controller lock held by its tempo edit.
                try:
                    self._dispatch_queue.put_nowait((handler, message))
                except queue.Full:
                    with self._lock:
                        self._closing = True
                    socket.close()
                    return

    @staticmethod
    def _dispatch_loop(messages: queue.Queue) -> None:
        while True:
            item = messages.get()
            if item is None:
                return
            handler, message = item
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
        keepalive_expiry_seconds: float | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._codec = opus_codec
        if client is None:
            options: dict[str, object] = {"timeout": httpx.Timeout(15.0, connect=5.0)}
            if keepalive_expiry_seconds is not None:
                # httpx drops idle connections after 5 s by default. A caller whose
                # fetches come in bursts (one per turn) asks for longer; keep it below
                # the server's own keep-alive timeout so the client expires them first.
                options["limits"] = httpx.Limits(keepalive_expiry=keepalive_expiry_seconds)
            client = httpx.Client(**options)
        self._client = client
        self._clock = clock

    def warm(self, connections: int = 2, *, timeout_seconds: float = 5.0) -> int:
        """Open or refresh ``connections`` pooled connections; returns how many answered.

        Concurrent cheap GETs each take their own connection, so later fetches
        (bars are fetched in parallel) do not pay for connection setup.
        """
        answered = []

        def probe() -> None:
            try:
                self._client.get(f"{self._base_url}/v1/rap/capabilities", timeout=timeout_seconds)
                answered.append(True)
            except httpx.HTTPError:
                pass

        threads = [threading.Thread(target=probe, name="rap-bar-audio-warm", daemon=True)
                   for _ in range(max(0, connections))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout_seconds + 1.0)
        return len(answered)

    def fetch(self, session_id: str, bar: int, lyric_id: str, *, wait_ms: int, accept_older: bool) -> BarAudioFetch:
        started = self._clock()
        server_wait_deadline = started + wait_ms / 1000.0
        request_wait_ms = wait_ms
        request_timeout = wait_ms / 1000.0 + 10.0
        retry_deadline = started + request_timeout
        for attempt in range(2):
            try:
                response = self._client.get(
                    f"{self._base_url}/v1/rap/sessions/{session_id}/bars/{bar}/audio",
                    params={"lyric_id": lyric_id, "wait_ms": request_wait_ms,
                            "accept_older": str(accept_older).lower()},
                    timeout=httpx.Timeout(request_timeout, connect=min(5.0, request_timeout)),
                )
                break
            except (httpx.RemoteProtocolError, httpx.NetworkError) as error:
                remaining = retry_deadline - self._clock()
                if attempt or remaining <= 0:
                    raise
                # GET is idempotent; httpx discards the failed connection. Do not
                # spend a second full long-poll window after a stale keepalive.
                request_wait_ms = max(0, int((server_wait_deadline - self._clock()) * 1000))
                request_timeout = remaining
                _LOGGER.warning("bar_audio_retry session=%s bar=%s elapsed_ms=%.1f error=%s",
                                session_id, bar, (self._clock() - started) * 1000, type(error).__name__)
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
