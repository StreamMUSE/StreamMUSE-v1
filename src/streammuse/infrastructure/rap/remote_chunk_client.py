"""Cancellable, bounded HTTP transport for remote two-bar rap packages."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from threading import Event, RLock
from time import monotonic
from types import MappingProxyType
from typing import Callable, Mapping

import httpx

from streammuse.domain.rap.remote_chunk import REMOTE_CHUNK_SCHEMA_VERSION, RemoteRapChunkRequest
from streammuse.infrastructure.rap.chunk_package import (
    MAX_RAP_CHUNK_PACKAGE_BYTES,
    RAP_CHUNK_PACKAGE_MEDIA_TYPE,
    DecodedRapChunkPackage,
    decode_chunk_package,
)


_RENDER_PATH = "/v1/rap/chunks/render"
_HEALTH_PATH = "/health"
_RETRYABLE_STATUS_CODES = frozenset((502, 503, 504))


class RemoteChunkClientError(RuntimeError):
    """Base class for bounded remote-render transport failures."""


class RemoteChunkDeadlineExceeded(RemoteChunkClientError):
    """The client reached its useful musical deadline."""


class RemoteChunkCancelled(RemoteChunkClientError):
    """The current useful wait was cancelled locally."""


class RemoteChunkProtocolError(RemoteChunkClientError):
    """The server response does not meet the remote chunk contract."""


@dataclass(frozen=True)
class RemoteChunkTransferTiming:
    request_ms: float
    first_byte_ms: float
    download_ms: float
    response_bytes: int
    attempts: int


@dataclass(frozen=True)
class RemoteChunkResponse:
    package: DecodedRapChunkPackage
    timing: RemoteChunkTransferTiming


@dataclass(frozen=True)
class RemoteChunkHealth:
    schema_version: str
    ready: bool
    details: Mapping[str, object]


class RemoteChunkClient:
    """Keep one HTTP connection pool while allowing an active response to abort."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = monotonic,
        max_response_bytes: int = MAX_RAP_CHUNK_PACKAGE_BYTES,
    ) -> None:
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(max_response_bytes, int) or isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        self._client = httpx.Client(base_url=base_url.rstrip("/") + "/", transport=transport)
        self._clock = clock
        self._max_response_bytes = max_response_bytes
        self._lock = RLock()
        self._cancelled = Event()
        self._active_response: httpx.Response | None = None
        self._closed = False

    def health(self, *, timeout_seconds: float = 5.0) -> RemoteChunkHealth:
        self._ensure_open()
        try:
            response = self._client.get(_HEALTH_PATH, timeout=_timeout(timeout_seconds))
        except httpx.RequestError as error:
            raise RemoteChunkClientError("remote health request failed") from error
        try:
            if response.status_code != 200:
                raise RemoteChunkProtocolError(f"remote health request failed: HTTP {response.status_code}")
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type != "application/json":
                raise RemoteChunkProtocolError("remote health response must be JSON")
            payload = response.json()
        except (ValueError, TypeError) as error:
            raise RemoteChunkProtocolError("remote health response is invalid") from error
        finally:
            response.close()
        if not isinstance(payload, dict):
            raise RemoteChunkProtocolError("remote health response is invalid")
        if payload.get("schema_version") != REMOTE_CHUNK_SCHEMA_VERSION:
            raise RemoteChunkProtocolError("remote health schema version is incompatible")
        ready = payload.get("ready")
        if type(ready) is not bool:
            raise RemoteChunkProtocolError("remote health response is invalid")
        return RemoteChunkHealth(REMOTE_CHUNK_SCHEMA_VERSION, ready, MappingProxyType(dict(payload)))

    def prepare(
        self,
        request: RemoteRapChunkRequest,
        timeout_seconds: float,
        *,
        deadline_monotonic: float | None = None,
    ) -> RemoteChunkResponse:
        if not isinstance(request, RemoteRapChunkRequest):
            raise ValueError("request must be a RemoteRapChunkRequest")
        self._ensure_open()
        timeout_seconds = _timeout(timeout_seconds)
        deadline = deadline_monotonic if deadline_monotonic is not None else self._clock() + timeout_seconds
        if not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or not isfinite(deadline):
            raise ValueError("deadline_monotonic must be a finite real number")
        attempt = request.transport_attempt()
        attempts = 0
        while True:
            started = self._ensure_before_deadline(deadline)
            self._cancelled.clear()
            attempts += 1
            try:
                result = self._request_once(attempt.body, request.request_id, timeout_seconds, deadline, started, attempts)
            except _RetryableRemoteFailure:
                if self._clock() >= deadline:
                    raise RemoteChunkDeadlineExceeded("remote chunk deadline elapsed before retry")
                continue
            return result

    def abort(self) -> None:
        """Stop the active response read while retaining the shared HTTP client."""
        self._cancelled.set()
        with self._lock:
            if self._active_response is not None:
                self._active_response.close()

    def close(self) -> None:
        """Permanently close the client; repeated calls are harmless."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancelled.set()
            if self._active_response is not None:
                self._active_response.close()
            self._client.close()

    def _request_once(
        self,
        body: bytes,
        request_id: str,
        timeout_seconds: float,
        deadline: float,
        started: float,
        attempts: int,
    ) -> RemoteChunkResponse:
        headers = {
            "accept": RAP_CHUNK_PACKAGE_MEDIA_TYPE,
            "content-type": "application/json",
            "idempotency-key": request_id,
        }
        try:
            with self._client.stream("POST", _RENDER_PATH, content=body, headers=headers, timeout=timeout_seconds) as response:
                response_started = self._clock()
                with self._lock:
                    self._active_response = response
                try:
                    self._raise_if_cancelled()
                    if response.status_code in _RETRYABLE_STATUS_CODES:
                        raise _RetryableRemoteFailure()
                    if response.status_code != 200:
                        raise RemoteChunkProtocolError(f"remote render request failed: HTTP {response.status_code}")
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if content_type != RAP_CHUNK_PACKAGE_MEDIA_TYPE:
                        raise RemoteChunkProtocolError("remote render response has an unsupported media type")
                    declared_length = response.headers.get("content-length")
                    if declared_length is not None and (not declared_length.isdigit() or int(declared_length) > self._max_response_bytes):
                        raise RemoteChunkProtocolError("remote render response exceeds configured limit")
                    data, first_byte = self._read_bounded_response(response, deadline)
                    completed = self._clock()
                finally:
                    with self._lock:
                        if self._active_response is response:
                            self._active_response = None
        except RemoteChunkClientError:
            raise
        except httpx.RequestError as error:
            if self._cancelled.is_set():
                raise RemoteChunkCancelled("remote chunk request was cancelled") from error
            raise _RetryableRemoteFailure() from error

        if completed >= deadline:
            raise RemoteChunkDeadlineExceeded("remote chunk deadline elapsed before response preparation")
        try:
            package = decode_chunk_package(data, expected_request_id=request_id)
        except ValueError as error:
            raise RemoteChunkProtocolError("remote chunk package is invalid") from error
        return RemoteChunkResponse(
            package,
            RemoteChunkTransferTiming(
                request_ms=(response_started - started) * 1000.0,
                first_byte_ms=(first_byte - response_started) * 1000.0,
                download_ms=(completed - first_byte) * 1000.0,
                response_bytes=len(data),
                attempts=attempts,
            ),
        )

    def _read_bounded_response(self, response: httpx.Response, deadline: float) -> tuple[bytes, float]:
        chunks: list[bytes] = []
        total = 0
        first_byte: float | None = None
        for chunk in response.iter_bytes():
            self._raise_if_cancelled()
            now = self._clock()
            if now >= deadline:
                raise RemoteChunkDeadlineExceeded("remote chunk deadline elapsed during response download")
            if chunk and first_byte is None:
                first_byte = now
            total += len(chunk)
            if total > self._max_response_bytes:
                raise RemoteChunkProtocolError("remote render response exceeds configured limit")
            chunks.append(chunk)
        self._raise_if_cancelled()
        if first_byte is None:
            raise RemoteChunkProtocolError("remote render response is empty")
        return b"".join(chunks), first_byte

    def _ensure_before_deadline(self, deadline: float) -> float:
        now = self._clock()
        if now >= deadline:
            raise RemoteChunkDeadlineExceeded("remote chunk deadline elapsed before request")
        return now

    def _raise_if_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise RemoteChunkCancelled("remote chunk request was cancelled")

    def _ensure_open(self) -> None:
        with self._lock:
            if self._closed:
                raise RemoteChunkClientError("remote chunk client is closed")


class _RetryableRemoteFailure(Exception):
    pass


def _timeout(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    return float(value)
