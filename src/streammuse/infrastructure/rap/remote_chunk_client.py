"""Cancellable, bounded HTTP transport for remote two-bar rap packages."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from threading import Condition, RLock, Thread
from time import monotonic, monotonic as system_monotonic
from types import MappingProxyType
from typing import Callable, Mapping

import httpx

from streammuse.domain.rap.remote_chunk import REMOTE_CHUNK_SCHEMA_VERSION, RemoteRapChunkRequest
from streammuse.infrastructure.rap.chunk_package import (
    MAX_RAP_CHUNK_PACKAGE_BYTES,
    RAP_CHUNK_PACKAGE_MEDIA_TYPE,
    RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE,
    DecodedRapChunkPackage,
    decode_opus_chunk_package,
    decode_chunk_package,
)


_RENDER_PATH = "/v1/rap/chunks/render"
_HEALTH_PATH = "/health"
_RETRYABLE_STATUS_CODES = frozenset((502, 503, 504))
_MAX_HEALTH_RESPONSE_BYTES = 16 * 1024


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


class _PrepareOperation:
    def __init__(self, generation: int) -> None:
        self.generation = generation
        self.condition = Condition()
        self.cancelled = False
        self.active_response: httpx.Response | None = None

    def cancel(self) -> None:
        with self.condition:
            self.cancelled = True
            response = self.active_response
            self.condition.notify_all()
        if response is not None:
            response.close()

    def register_response(self, response: httpx.Response) -> bool:
        with self.condition:
            if self.cancelled:
                return False
            self.active_response = response
            return True

    def is_cancelled(self) -> bool:
        with self.condition:
            return self.cancelled

    def clear_response(self, response: httpx.Response) -> None:
        with self.condition:
            if self.active_response is response:
                self.active_response = None


@dataclass
class _AttemptCompletion:
    done: bool = False
    value: RemoteChunkResponse | None = None
    error: BaseException | None = None


class RemoteChunkClient:
    """Keep one HTTP connection pool while allowing an active response to abort."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = monotonic,
        max_response_bytes: int = MAX_RAP_CHUNK_PACKAGE_BYTES,
        max_health_response_bytes: int = _MAX_HEALTH_RESPONSE_BYTES,
        audio_transport: str = "pcm",
    ) -> None:
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(max_response_bytes, int) or isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        if (
            not isinstance(max_health_response_bytes, int)
            or isinstance(max_health_response_bytes, bool)
            or max_health_response_bytes <= 0
        ):
            raise ValueError("max_health_response_bytes must be a positive integer")
        if audio_transport not in {"pcm", "opus"}:
            raise ValueError("audio_transport must be pcm or opus")
        self._client = httpx.Client(base_url=base_url.rstrip("/") + "/", transport=transport)
        self._clock = clock
        self._max_response_bytes = max_response_bytes
        self._max_health_response_bytes = max_health_response_bytes
        self._audio_transport = audio_transport
        self._opus_codec = None
        self._lock = RLock()
        self._generation = 0
        self._active_operation: _PrepareOperation | None = None
        self._closed = False

    def health(self, *, timeout_seconds: float = 5.0) -> RemoteChunkHealth:
        self._ensure_open()
        try:
            with self._client.stream("GET", _HEALTH_PATH, timeout=_timeout(timeout_seconds)) as response:
                if response.status_code != 200:
                    raise RemoteChunkProtocolError(f"remote health request failed: HTTP {response.status_code}")
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if content_type != "application/json":
                    raise RemoteChunkProtocolError("remote health response must be JSON")
                self._validate_declared_length(
                    response,
                    self._max_health_response_bytes,
                    "remote health response exceeds configured limit",
                )
                data = self._read_limited_bytes(
                    response,
                    self._max_health_response_bytes,
                    "remote health response exceeds configured limit",
                )
        except RemoteChunkClientError:
            raise
        except httpx.RequestError as error:
            raise RemoteChunkProtocolError("remote health response is truncated or invalid") from error
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, TypeError) as error:
            raise RemoteChunkProtocolError("remote health response is invalid") from error
        if not isinstance(payload, dict):
            raise RemoteChunkProtocolError("remote health response is invalid")
        if payload.get("schema_version") != REMOTE_CHUNK_SCHEMA_VERSION:
            raise RemoteChunkProtocolError("remote health schema version is incompatible")
        ready = payload.get("ready")
        if type(ready) is not bool:
            raise RemoteChunkProtocolError("remote health response is invalid")
        return RemoteChunkHealth(REMOTE_CHUNK_SCHEMA_VERSION, ready, MappingProxyType(dict(payload)))

    @staticmethod
    def _validate_declared_length(response: httpx.Response, limit: int, message: str) -> None:
        declared = response.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            raise RemoteChunkProtocolError(message)

    @staticmethod
    def _read_limited_bytes(response: httpx.Response, limit: int, message: str) -> bytes:
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > limit:
                raise RemoteChunkProtocolError(message)
            chunks.append(chunk)
        return b"".join(chunks)

    def prepare(
        self,
        request: RemoteRapChunkRequest,
        timeout_seconds: float,
        *,
        deadline_monotonic: float | None = None,
    ) -> RemoteChunkResponse:
        if not isinstance(request, RemoteRapChunkRequest):
            raise ValueError("request must be a RemoteRapChunkRequest")
        timeout_seconds = _timeout(timeout_seconds)
        operation = self._begin_operation()
        try:
            deadline = deadline_monotonic if deadline_monotonic is not None else self._clock() + timeout_seconds
            if not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or not isfinite(deadline):
                raise ValueError("deadline_monotonic must be a finite real number")
            attempt = request.transport_attempt()
            attempts = 0
            while True:
                started = self._ensure_before_deadline(deadline, operation)
                attempt_timeout = min(timeout_seconds, deadline - started)
                attempts += 1
                try:
                    return self._run_attempt(
                        operation,
                        attempt.body,
                        request.request_id,
                        attempt_timeout,
                        deadline,
                        started,
                        attempts,
                    )
                except _RetryableRemoteFailure:
                    self._raise_if_cancelled(operation)
                    with self._lock:
                        if self._closed:
                            raise RemoteChunkCancelled("remote chunk request was cancelled")
                    if self._clock() >= deadline:
                        raise RemoteChunkDeadlineExceeded("remote chunk deadline elapsed before retry")
        finally:
            self._end_operation(operation)

    def abort(self) -> None:
        """Stop the active response read while retaining the shared HTTP client."""
        with self._lock:
            operation = self._active_operation
        if operation is not None:
            operation.cancel()

    def close(self) -> None:
        """Permanently close the client; repeated calls are harmless."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            operation = self._active_operation
        if operation is not None:
            operation.cancel()
        with self._lock:
            self._client.close()

    def _run_attempt(
        self,
        operation: _PrepareOperation,
        body: bytes,
        request_id: str,
        timeout_seconds: float,
        deadline: float,
        started: float,
        attempts: int,
    ) -> RemoteChunkResponse:
        completion = _AttemptCompletion()

        def request() -> None:
            try:
                completion.value = self._request_once(
                    operation,
                    body,
                    request_id,
                    timeout_seconds,
                    deadline,
                    started,
                    attempts,
                )
            except BaseException as error:
                completion.error = error
            finally:
                with operation.condition:
                    completion.done = True
                    operation.condition.notify_all()

        Thread(target=request, name=f"remote-rap-chunk-{operation.generation}-{attempts}", daemon=True).start()
        wait_deadline = system_monotonic() + timeout_seconds
        with operation.condition:
            while not completion.done and not operation.cancelled:
                wait_seconds = wait_deadline - system_monotonic()
                if wait_seconds <= 0:
                    break
                operation.condition.wait(wait_seconds)
            if operation.cancelled:
                raise RemoteChunkCancelled("remote chunk request was cancelled")
            if not completion.done:
                if self._clock() >= deadline:
                    raise RemoteChunkDeadlineExceeded("remote chunk deadline elapsed while waiting for response headers")
                raise _RetryableRemoteFailure()
        if completion.error is not None:
            raise completion.error
        if completion.value is None:
            raise RemoteChunkClientError("remote chunk request completed without a response")
        return completion.value

    def _request_once(
        self,
        operation: _PrepareOperation,
        body: bytes,
        request_id: str,
        timeout_seconds: float,
        deadline: float,
        started: float,
        attempts: int,
    ) -> RemoteChunkResponse:
        self._ensure_before_deadline(deadline, operation)
        headers = {
            "accept": self._accept_header(),
            "content-type": "application/json",
            "idempotency-key": request_id,
        }
        try:
            with self._client.stream("POST", _RENDER_PATH, content=body, headers=headers, timeout=timeout_seconds) as response:
                response_started = self._clock()
                if not operation.register_response(response):
                    raise RemoteChunkCancelled("remote chunk request was cancelled")
                try:
                    self._raise_if_cancelled(operation)
                    if response.status_code in _RETRYABLE_STATUS_CODES:
                        raise _RetryableRemoteFailure()
                    if response.status_code != 200:
                        raise RemoteChunkProtocolError(f"remote render request failed: HTTP {response.status_code}")
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if content_type not in self._accepted_media_types():
                        raise RemoteChunkProtocolError("remote render response has an unsupported media type")
                    declared_length = response.headers.get("content-length")
                    if declared_length is not None and (not declared_length.isdigit() or int(declared_length) > self._max_response_bytes):
                        raise RemoteChunkProtocolError("remote render response exceeds configured limit")
                    data, first_byte = self._read_bounded_response(response, deadline, operation)
                    completed = self._clock()
                finally:
                    operation.clear_response(response)
        except RemoteChunkClientError:
            raise
        except httpx.RequestError as error:
            if operation.is_cancelled():
                raise RemoteChunkCancelled("remote chunk request was cancelled") from error
            raise _RetryableRemoteFailure() from error

        if completed >= deadline:
            raise RemoteChunkDeadlineExceeded("remote chunk deadline elapsed before response preparation")
        try:
            package = self._decode_package(data, content_type, request_id)
        except ValueError as error:
            raise RemoteChunkProtocolError("remote chunk package is invalid") from error
        self._raise_if_cancelled(operation)
        if self._clock() >= deadline:
            raise RemoteChunkDeadlineExceeded("remote chunk deadline elapsed during response preparation")
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

    def _accept_header(self) -> str:
        if self._audio_transport == "opus":
            return f"{RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE}, {RAP_CHUNK_PACKAGE_MEDIA_TYPE};q=0.9"
        return RAP_CHUNK_PACKAGE_MEDIA_TYPE

    def _accepted_media_types(self) -> frozenset[str]:
        if self._audio_transport == "opus":
            return frozenset((RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE, RAP_CHUNK_PACKAGE_MEDIA_TYPE))
        return frozenset((RAP_CHUNK_PACKAGE_MEDIA_TYPE,))

    def _decode_package(
        self, data: bytes, content_type: str, request_id: str
    ) -> DecodedRapChunkPackage:
        if content_type == RAP_CHUNK_PACKAGE_MEDIA_TYPE:
            return decode_chunk_package(data, expected_request_id=request_id)
        if content_type == RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE and self._audio_transport == "opus":
            if self._opus_codec is None:
                from streammuse.infrastructure.rap.opus_codec import FFmpegOpusCodec

                self._opus_codec = FFmpegOpusCodec()
            return decode_opus_chunk_package(
                data, expected_request_id=request_id, codec=self._opus_codec
            )
        raise ValueError("remote render response has an unsupported media type")

    def _read_bounded_response(
        self,
        response: httpx.Response,
        deadline: float,
        operation: _PrepareOperation,
    ) -> tuple[bytes, float]:
        chunks: list[bytes] = []
        total = 0
        first_byte: float | None = None
        for chunk in response.iter_bytes():
            self._raise_if_cancelled(operation)
            now = self._clock()
            if now >= deadline:
                raise RemoteChunkDeadlineExceeded("remote chunk deadline elapsed during response download")
            if chunk and first_byte is None:
                first_byte = now
            total += len(chunk)
            if total > self._max_response_bytes:
                raise RemoteChunkProtocolError("remote render response exceeds configured limit")
            chunks.append(chunk)
        self._raise_if_cancelled(operation)
        if first_byte is None:
            raise RemoteChunkProtocolError("remote render response is empty")
        return b"".join(chunks), first_byte

    def _ensure_before_deadline(self, deadline: float, operation: _PrepareOperation) -> float:
        self._raise_if_cancelled(operation)
        now = self._clock()
        if now >= deadline:
            raise RemoteChunkDeadlineExceeded("remote chunk deadline elapsed before request")
        return now

    @staticmethod
    def _raise_if_cancelled(operation: _PrepareOperation) -> None:
        if operation.is_cancelled():
            raise RemoteChunkCancelled("remote chunk request was cancelled")

    def _begin_operation(self) -> _PrepareOperation:
        with self._lock:
            if self._closed:
                raise RemoteChunkClientError("remote chunk client is closed")
            if self._active_operation is not None:
                raise RemoteChunkClientError("remote chunk client already has an active request")
            self._generation += 1
            operation = _PrepareOperation(self._generation)
            self._active_operation = operation
            return operation

    def _end_operation(self, operation: _PrepareOperation) -> None:
        with self._lock:
            if self._active_operation is operation:
                self._active_operation = None

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
